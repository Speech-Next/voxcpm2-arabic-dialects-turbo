"""A distilled LocDiT's own LocDiT LoRAs for the adapters (<stem>.adapters/<adapter>/, issue #12): the loader replaces
an adapter's tensors of the same names, nothing else; DialectTTS and the HTTP server find them next to the checkpoint
and register each adapter with its override; without them, every call is as before."""

import asyncio
import json
import os

import pytest
import torch
from safetensors.torch import save_file

import nanovllm_voxcpm
from nanovllm_voxcpm.dialect_tts import DialectTTS, student_adapters
from nanovllm_voxcpm.models.voxcpm2.lora_loader import load_voxcpm2_lora_checkpoint
from test_wiring import FakeEngine, serve, server_args

LM = "base_lm.layers.0.self_attn"
DIT = "feat_decoder.estimator.decoder.layers.0.self_attn"
CONFIG = {"lora_config": {"enable_lm": True, "enable_dit": True, "r": 4, "alpha": 8,
                          "target_modules_lm": ["q_proj", "k_proj", "v_proj", "o_proj"],
                          "target_modules_dit": ["q_proj", "k_proj", "v_proj", "o_proj"]}}


def lora_dir(path, prefixes, seed, r=4, config=CONFIG, extra=None):
    g = torch.Generator().manual_seed(seed)
    t = {}
    for p in prefixes:
        for m in ("q_proj", "k_proj", "v_proj", "o_proj"):
            t[f"{p}.{m}.lora_A"] = torch.randn(r, 16, generator=g)
            t[f"{p}.{m}.lora_B"] = torch.randn(16, r, generator=g)
    t.update(extra or {})
    os.makedirs(path, exist_ok=True)
    save_file(t, os.path.join(path, "lora_weights.safetensors"))
    json.dump(config, open(os.path.join(path, "lora_config.json"), "w"))
    return str(path), t


def module_tensors(payload, name):
    m = payload.modules[name]
    b = m.lora_b if isinstance(m.lora_b, list) else [m.lora_b]
    return [m.lora_a, *b]


def test_loader_override(tmp_path):
    base, bt = lora_dir(tmp_path / "base", [LM, DIT], 0)
    over, ot = lora_dir(tmp_path / "over", [DIT], 1)
    plain = load_voxcpm2_lora_checkpoint(base)
    merged = load_voxcpm2_lora_checkpoint(base, overrides=[over])
    assert set(plain.modules) == set(merged.modules) == {f"{LM}.qkv_proj", f"{LM}.o_proj", f"{DIT}.qkv_proj",
                                                          f"{DIT}.o_proj"}
    for name in (f"{LM}.qkv_proj", f"{LM}.o_proj"):  # the LM's tensors stay the adapter's
        assert all(torch.equal(a, b) for a, b in zip(module_tensors(plain, name), module_tensors(merged, name)))
    # the LocDiT's: the override's, in the same packing (q, k, v stacked) and scaling
    assert torch.equal(merged.modules[f"{DIT}.qkv_proj"].lora_a,
                       torch.stack([ot[f"{DIT}.{m}.lora_A"] for m in ("q_proj", "k_proj", "v_proj")]))
    assert torch.equal(merged.modules[f"{DIT}.o_proj"].lora_b, ot[f"{DIT}.o_proj.lora_B"])
    assert torch.equal(plain.modules[f"{DIT}.o_proj"].lora_b, bt[f"{DIT}.o_proj.lora_B"])
    assert merged.modules[f"{DIT}.o_proj"].scaling == plain.modules[f"{DIT}.o_proj"].scaling == 2.0


def test_loader_override_must_match(tmp_path):
    base, _ = lora_dir(tmp_path / "base", [DIT], 0)
    extra, _ = lora_dir(tmp_path / "extra", [DIT, LM], 1)  # a tensor the adapter does not have
    with pytest.raises(ValueError, match="is not in"):
        load_voxcpm2_lora_checkpoint(base, overrides=[extra])
    cfg = json.loads(json.dumps(CONFIG))
    cfg["lora_config"]["alpha"] = 16
    alpha, _ = lora_dir(tmp_path / "alpha", [DIT], 1, config=cfg)
    with pytest.raises(ValueError, match="alpha"):
        load_voxcpm2_lora_checkpoint(base, overrides=[alpha])
    cfg = json.loads(json.dumps(CONFIG))
    cfg["lora_config"]["r"] = 8
    rank, _ = lora_dir(tmp_path / "rank", [DIT], 1, r=8, config=cfg)
    with pytest.raises(ValueError, match="rank"):
        load_voxcpm2_lora_checkpoint(base, overrides=[rank])


def student(tmp_path, adapters=("najdi",)):
    (tmp_path / "dit").mkdir()
    ck = tmp_path / "dit" / "student.pt"
    ck.write_bytes(b"checkpoint")
    for a in adapters:
        lora_dir(tmp_path / "dit" / "student.adapters" / a, [DIT], 7)
    return str(ck)


def test_student_adapters_next_to_a_local_checkpoint(tmp_path):
    ck = student(tmp_path, ("najdi", "qatari"))
    assert student_adapters(ck) == {n: str(tmp_path / "dit" / "student.adapters" / n) for n in ("najdi", "qatari")}
    os.remove(ck.replace(".pt", ".adapters/qatari/lora_weights.safetensors"))  # not an adapter directory any more
    assert set(student_adapters(ck)) == {"najdi"}
    other = tmp_path / "other.pt"
    other.write_bytes(b"checkpoint")
    assert student_adapters(str(other)) == {} and student_adapters(None) == {}


class RecordingEngine(FakeEngine):
    calls: list = []

    async def register_lora(self, name, path, *overrides):
        RecordingEngine.calls.append((name, path, *overrides))


def model_with_adapters(tmp_path, names=("najdi", "saudi")):
    for n in names:
        lora_dir(tmp_path / "model" / "adapters" / n, [LM, DIT], 3)
    return str(tmp_path / "model")


@pytest.mark.parametrize("use", [True, False])
def test_dialect_tts_registers_the_overrides(tmp_path, monkeypatch, use):
    monkeypatch.setattr(nanovllm_voxcpm, "VoxCPM", RecordingEngine, raising=False)
    monkeypatch.setattr(RecordingEngine, "calls", [])
    model, ck = model_with_adapters(tmp_path), student(tmp_path, ("najdi", "emirati"))
    tts = asyncio.run(DialectTTS.load(model, ck, w8a8=False, dit_adapters=use))
    adapters = os.path.join(model, "adapters")
    over = [[str(tmp_path / "dit" / "student.adapters" / "najdi")]] if use else []
    # najdi with the student's LocDiT LoRA; saudi has none (as before); emirati is not one of the model's adapters
    assert RecordingEngine.calls == [("najdi", os.path.join(adapters, "najdi"), *over),
                                     ("saudi", os.path.join(adapters, "saudi"))]
    assert tts.dit_adapters == (["najdi"] if use else [])


def test_server_registers_the_overrides(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")
    from nanovllm_voxcpm.entrypoints import openai_server

    monkeypatch.setattr(nanovllm_voxcpm, "VoxCPM", RecordingEngine, raising=False)
    monkeypatch.setattr(RecordingEngine, "calls", [])
    monkeypatch.delenv("NANOVLLM_VOXCPM2_TURBO", raising=False)
    model, ck = model_with_adapters(tmp_path), student(tmp_path)
    asyncio.run(serve(openai_server.build_app(server_args(model, dit=ck)), []))
    adapters = os.path.join(model, "adapters")
    assert RecordingEngine.calls == [("najdi", os.path.join(adapters, "najdi"),
                                      [str(tmp_path / "dit" / "student.adapters" / "najdi")]),
                                     ("saudi", os.path.join(adapters, "saudi"))]
    monkeypatch.setattr(RecordingEngine, "calls", [])
    monkeypatch.delenv("NANOVLLM_VOXCPM2_TURBO", raising=False)
    asyncio.run(serve(openai_server.build_app(server_args(model, dit=ck, dit_adapters=False)), []))
    assert [c[0] for c in RecordingEngine.calls] == ["najdi", "saudi"] and all(len(c) == 2 for c in RecordingEngine.calls)
