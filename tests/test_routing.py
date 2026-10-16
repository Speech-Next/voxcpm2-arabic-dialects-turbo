"""Per-dialect routing on fake engines (no GPU): the name -> (model, adapter or tag) table of the two default models,
dispatch of requests and voices to the right engine, and the loader's per-engine configuration."""

import asyncio
import json
import os

import numpy as np
import pytest

from nanovllm_voxcpm import dialect_routing, dialect_tts
from nanovllm_voxcpm.dialect_routing import (ADAPTERS, DEFAULT_ROUTES, TAGGED, Route, RoutedDialectTTS, route_table,
                                             split_kv_blocks)
from nanovllm_voxcpm.dialect_tts import DialectTags, DialectTTS

# the default models' names: adapters.json of VoxCPM-Kernel/VoxCPM2-Arabic-Dialects, dialects.json of
# VoxCPM-Kernel/VoxCPM2-Arabic-Teacher-v3
ADAPTER_NAMES = ["ab", "egyptian", "emirati", "hejazi", "hindi_english", "iraqi", "jordanian", "najdi", "palestinian",
                 "qatari", "saudi"]
V3_DIALECTS = ["msa", "egyptian", "palestinian", "lebanese", "syrian", "iraqi", "saudi", "najdi", "hejazi", "gulf",
               "bahraini", "yemeni", "sudanese", "libyan", "tunisian", "algerian", "moroccan"]
V3 = {"tags": {d: f"<reserved_{100 + i}>" for i, d in enumerate(V3_DIALECTS)},
      "aliases": {"emirati": "gulf", "qatari": "gulf", "khaliji": "gulf", "jordanian": "palestinian",
                  "ab": "palestinian", "levantine": "palestinian"}}


class FakeEngine:
    """The engine's async API, recording what it is sent and the engine environment it was started with."""

    started: list = []

    def __init__(self, **kw):
        self.kw = kw
        self.env = {k: os.environ.get(k) for k in dialect_routing._ENGINE_ENV}
        self.requests: list[dict] = []
        self.voices: dict[str, str | None] = {}
        self.fail_register = False
        self.stopped = False

    @classmethod
    def from_pretrained(cls, **kw):
        engine = cls(**kw)
        cls.started.append(engine)
        return engine

    async def wait_for_ready(self):
        pass

    async def register_lora(self, name, path):
        pass

    async def get_model_info(self):
        return {"sample_rate": 48000}

    async def encode_latents(self, audio, audio_format):
        return b"latents"

    async def register_voice(self, name, latents, text):
        if self.fail_register:
            raise RuntimeError("engine refused the voice")
        self.voices[name] = text

    async def unregister_voice(self, name):
        self.voices.pop(name, None)

    async def generate(self, target_text, **kw):
        self.requests.append({"text": target_text, **kw})
        yield np.zeros(4, dtype=np.int16)

    async def stop(self):
        self.stopped = True


def run(coro):
    return asyncio.run(coro)


def routed(routes=None) -> RoutedDialectTTS:
    adapters = DialectTTS(FakeEngine(), ADAPTER_NAMES, 48000)
    tagged = DialectTTS(FakeEngine(), [], 48000, V3)
    return RoutedDialectTTS(adapters, tagged, routes)


def test_default_route_table():
    table = route_table(ADAPTER_NAMES, DialectTags(V3), DEFAULT_ROUTES)
    assert table["bahraini"] == Route(ADAPTERS, "qatari")
    assert table["egyptian"] == Route(ADAPTERS, "egyptian")
    assert table["saudi"] == Route(ADAPTERS, "saudi")
    assert table["hindi_english"] == Route(ADAPTERS, "hindi_english")  # only the adapters model has it
    for name in ("gulf", "emirati", "qatari", "khaliji"):  # aliases follow the tagged model's dialects.json
        assert table[name] == Route(TAGGED, "gulf"), name
    for name in ("palestinian", "jordanian", "ab", "levantine"):
        assert table[name] == Route(TAGGED, "palestinian"), name
    for name in ("msa", "lebanese", "syrian", "iraqi", "najdi", "hejazi", "yemeni", "sudanese", "libyan", "tunisian",
                 "algerian", "moroccan"):
        assert table[name] == Route(TAGGED, name), name
    assert sorted(table) == sorted(set(ADAPTER_NAMES) | set(V3["tags"]) | set(V3["aliases"]))
    assert sum(r.model == ADAPTERS for r in table.values()) == 4


def test_route_overrides_and_errors():
    tags = DialectTags(V3)
    table = route_table(ADAPTER_NAMES, tags, {**DEFAULT_ROUTES, "qatari": "qatari"})  # a name wins over its dialect
    assert table["qatari"] == Route(ADAPTERS, "qatari") and table["gulf"] == Route(TAGGED, "gulf")
    table = route_table(ADAPTER_NAMES, tags, {"gulf": "emirati"})  # a dialect takes its aliases along
    assert table["khaliji"] == table["qatari"] == Route(ADAPTERS, "emirati")
    assert table["saudi"] == Route(TAGGED, "saudi")
    assert route_table(ADAPTER_NAMES, tags, {})["egyptian"] == Route(TAGGED, "egyptian")  # no routes: all tagged
    with pytest.raises(ValueError, match="bahraini"):
        route_table(ADAPTER_NAMES, tags, {"bahraini": "bahraini"})  # no such adapter


def test_split_kv_blocks():
    assert split_kv_blocks(600) == (300, 300)
    assert split_kv_blocks(601) == (301, 300)
    assert split_kv_blocks((200, 400)) == (200, 400)
    with pytest.raises(ValueError, match="automatic"):
        split_kv_blocks(0)


def test_dispatch_to_the_routed_engine():
    tts = routed()
    a, t = tts.engines[ADAPTERS].server, tts.engines[TAGGED].server
    run(tts.synthesize("هلا", "bahraini"))
    assert a.requests[-1]["lora_name"] == "qatari" and a.requests[-1]["text"] == "هلا"
    run(tts.synthesize("هلا", "najdi"))
    assert t.requests[-1]["text"] == V3["tags"]["najdi"] + "هلا" and t.requests[-1]["lora_name"] is None
    run(tts.synthesize("هلا", "qatari"))  # Gulf on teacher-v3
    assert t.requests[-1]["text"] == V3["tags"]["gulf"] + "هلا"
    run(tts.synthesize("मीटिंग 9 बजे", "hindi_english"))
    assert (a.requests[-1]["text"], a.requests[-1]["lora_name"]) == ("मीटिंग 9 बजे", "hindi_english")
    run(tts.synthesize("hello", None))  # no dialect: the base model, as DialectTTS
    assert a.requests[-1]["lora_name"] is None and a.requests[-1]["text"] == "hello"
    assert (len(a.requests), len(t.requests)) == (3, 2)
    run(tts.synthesize("الساعة 9:05", "msa", seed=7, cfg_value=1.5, max_patches=50))  # formal, arguments passed on
    assert t.requests[-1] | {"text": None} == {"text": None, "lora_name": None, "seed": 7, "cfg_value": 1.5,
                                               "max_generate_length": 50}
    assert t.requests[-1]["text"] == V3["tags"]["msa"] + "الساعة تسعة وخمس دقائق"
    with pytest.raises(ValueError, match="unknown dialect 'kuwaiti'"):
        run(tts.synthesize("هلا", "kuwaiti"))
    assert "bahraini" in tts.dialects and "hindi_english" in tts.dialects and tts.sample_rate == 48000


def test_voices_on_both_engines():
    tts = routed()
    a, t = tts.engines[ADAPTERS].server, tts.engines[TAGGED].server
    run(tts.add_voice("v", b"wav", text="الساعة 9:30"))
    assert a.voices["v"] == t.voices["v"] == "الساعة تسعة ونص" and tts.voices == {"v": True}
    run(tts.synthesize("هلا", "saudi", voice="v"))
    assert a.requests[-1]["voice"] == "v" and a.requests[-1]["clone_mode"] == "prompt_ref"
    run(tts.synthesize("هلا", "msa", voice="v"))  # prompt_ref on the tagged model: the dialect's copy of the voice
    assert t.requests[-1]["voice"] == "v@" + V3["tags"]["msa"]
    assert t.voices["v@" + V3["tags"]["msa"]] == V3["tags"]["msa"] + "الساعة تسعة والنصف"
    run(tts.synthesize("هلا", "najdi", voice="v", clone_mode="ref"))
    assert t.requests[-1]["voice"] == "v" and t.requests[-1]["text"] == V3["tags"]["najdi"] + "هلا"
    run(tts.remove_voice("v"))
    assert a.voices == {} and t.voices == {} and tts.voices == {}
    with pytest.raises(ValueError, match="not registered"):
        run(tts.synthesize("هلا", "najdi", voice="v"))


def test_voice_registration_is_all_or_nothing():
    tts = routed()
    a, t = tts.engines[ADAPTERS].server, tts.engines[TAGGED].server
    t.fail_register = True
    with pytest.raises(RuntimeError, match="refused"):
        run(tts.add_voice("v", b"wav"))
    assert a.voices == {} and "v" not in tts.engines[ADAPTERS].voices and tts.voices == {}


def test_rejects_models_of_the_wrong_kind():
    adapters = DialectTTS(FakeEngine(), ADAPTER_NAMES, 48000)
    tagged = DialectTTS(FakeEngine(), [], 48000, V3)
    with pytest.raises(ValueError, match="adapters model first"):
        RoutedDialectTTS(tagged, adapters)
    with pytest.raises(ValueError, match="sample rates"):
        RoutedDialectTTS(adapters, DialectTTS(FakeEngine(), [], 24000, V3))


@pytest.fixture
def models(tmp_path, monkeypatch):
    """An adapters model with SmoothQuant statistics and a tagged model on disk, the fake engine as VoxCPM."""
    import nanovllm_voxcpm

    a, t = tmp_path / "adapters-model", tmp_path / "tagged-model"
    for name in ADAPTER_NAMES:
        (a / "adapters" / name).mkdir(parents=True)
        (a / "adapters" / name / "lora_weights.safetensors").write_bytes(b"")
    (a / "smooth_stats.pt").write_bytes(b"")
    t.mkdir()
    (t / "dialects.json").write_text(json.dumps(V3), encoding="utf-8")
    for dit in ("adapters-dit.pt", "tagged-dit.pt"):
        (tmp_path / dit).write_bytes(b"")
    monkeypatch.setattr(nanovllm_voxcpm, "VoxCPM", FakeEngine, raising=False)
    monkeypatch.setattr(dialect_tts, "adapters_lora_config", lambda adapters: None)
    monkeypatch.setattr(dialect_routing, "DEFAULT_TAGGED_DIT", str(tmp_path / "tagged-dit.pt"))
    for k in dialect_routing._ENGINE_ENV:
        monkeypatch.delenv(k, raising=False)
    FakeEngine.started = []
    return tmp_path


def test_load_starts_each_engine_with_its_own_configuration(models):
    tts = run(RoutedDialectTTS.load(str(models / "adapters-model"), str(models / "tagged-model"),
                                    str(models / "adapters-dit.pt"), kv_blocks=(250, 350), max_concurrency=64))
    a, t = FakeEngine.started
    assert tts.engines[ADAPTERS].server is a and tts.engines[TAGGED].server is t
    turbo_a, turbo_t = json.loads(a.env["NANOVLLM_VOXCPM2_TURBO"]), json.loads(t.env["NANOVLLM_VOXCPM2_TURBO"])
    assert turbo_a["dit_checkpoint"].endswith("adapters-dit.pt")
    assert turbo_t["dit_checkpoint"].endswith("tagged-dit.pt")  # "auto" in fast mode: DEFAULT_TAGGED_DIT
    assert (a.env["NANOVLLM_SERVERPOOL_NUM_KVCACHE_BLOCKS"], t.env["NANOVLLM_SERVERPOOL_NUM_KVCACHE_BLOCKS"]) == \
        ("250", "350")
    assert a.env["VOXCPM_KERNELS_SMOOTH"].endswith("smooth_stats.pt")
    assert t.env["VOXCPM_KERNELS_SMOOTH"] is None  # not the adapters model's statistics
    assert a.kw["max_num_seqs"] == t.kw["max_num_seqs"] == 64
    assert all(os.environ.get(k) is None for k in dialect_routing._ENGINE_ENV)  # restored after both starts
    assert tts.table == route_table(ADAPTER_NAMES, DialectTags(V3), DEFAULT_ROUTES)
    run(tts.close())
    assert a.stopped and t.stopped


def test_load_quality_mode_and_failures(models):
    tts = run(RoutedDialectTTS.load(str(models / "adapters-model"), str(models / "tagged-model"), None,
                                    dit_cache="light"))
    for engine in FakeEngine.started:  # "auto" with dit=None: both on their own LocDiT
        turbo = json.loads(engine.env["NANOVLLM_VOXCPM2_TURBO"])
        assert "dit_checkpoint" not in turbo and turbo["dit_cache"]
        assert engine.env["NANOVLLM_SERVERPOOL_NUM_KVCACHE_BLOCKS"] == "300"
    run(tts.close())
    with pytest.raises(ValueError, match="dialects.json"):  # the models swapped
        run(RoutedDialectTTS.load(str(models / "tagged-model"), str(models / "adapters-model"), None))
    with pytest.raises(ValueError, match="lacks"):  # a bad route fails before any engine starts
        FakeEngine.started = []
        run(RoutedDialectTTS.load(str(models / "adapters-model"), str(models / "tagged-model"), None,
                                  routes={"bahraini": "kuwaiti"}))
    assert FakeEngine.started == []
