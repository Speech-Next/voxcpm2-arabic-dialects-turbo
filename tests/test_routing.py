"""Per-dialect routing on a fake engine (no GPU): the name -> (adapter or tag) table of the two default models,
dispatch of requests and voices, the one engine's LoRA configuration and adapters, the tagged-adapter loader."""

import asyncio
import json
import os

import numpy as np
import pytest

from nanovllm_voxcpm.dialect_routing import (ADAPTERS, DEFAULT_ROUTES, TAGGED, TAGGED_ADAPTER, TEACHER, Route,
                                             RoutedDialectTTS, describe, parse_routes, route_table, routed_lora_config,
                                             used_adapters)
from nanovllm_voxcpm.dialect_tts import DialectTags

# the default models' names: adapters.json of VoxCPM-Kernel/VoxCPM2-Arabic-Dialects, dialects.json of
# VoxCPM-Kernel/VoxCPM2-Arabic-Teacher-v3
ADAPTER_NAMES = ["ab", "egyptian", "emirati", "hejazi", "hindi_english", "iraqi", "jordanian", "najdi", "palestinian",
                 "qatari", "saudi"]
V3_DIALECTS = ["msa", "egyptian", "palestinian", "lebanese", "syrian", "iraqi", "saudi", "najdi", "hejazi", "gulf",
               "bahraini", "yemeni", "sudanese", "libyan", "tunisian", "algerian", "moroccan"]
V3 = {"tags": {d: f"<reserved_{100 + i}>" for i, d in enumerate(V3_DIALECTS)},
      "aliases": {"emirati": "gulf", "qatari": "gulf", "khaliji": "gulf", "jordanian": "palestinian",
                  "ab": "palestinian", "levantine": "palestinian"}}
ADAPTER_LORA = {"enable_lm": True, "enable_dit": True, "enable_proj": False, "r": 64, "alpha": 128,
                "target_modules_lm": ["q_proj", "v_proj", "k_proj", "o_proj"],
                "target_modules_dit": ["q_proj", "v_proj", "k_proj", "o_proj"]}
ALL7 = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
V3_STATE = {"lora": {"enable_lm": True, "enable_dit": False, "enable_proj": False, "r": 64, "alpha": 64,
                     "target_modules_lm": ALL7, "target_modules_dit": ALL7, "train_modules": ["base_lm"]},
            "dialects": V3_DIALECTS}


class FakeEngine:
    """The engine's async API, recording what it is sent."""

    started: list = []

    def __init__(self, **kw):
        self.kw = kw
        self.env = {k: os.environ.get(k) for k in ("NANOVLLM_VOXCPM2_TURBO", "VOXCPM_KERNELS_SMOOTH")}
        self.requests: list[dict] = []
        self.voices: dict[str, str | None] = {}
        self.loras: dict[str, str] = {}
        self.stopped = False

    @classmethod
    def from_pretrained(cls, **kw):
        engine = cls(**kw)
        cls.started.append(engine)
        return engine

    async def wait_for_ready(self):
        pass

    async def register_lora(self, name, path):
        self.loras[name] = path

    async def get_model_info(self):
        return {"sample_rate": 48000}

    async def encode_latents(self, audio, audio_format):
        return b"latents"

    async def register_voice(self, name, latents, text):
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
    return RoutedDialectTTS(FakeEngine(), ADAPTER_NAMES, V3, 48000, routes)


def test_default_route_table():
    table = route_table(ADAPTER_NAMES, DialectTags(V3), DEFAULT_ROUTES)
    assert table["bahraini"] == Route(ADAPTERS, "qatari")
    assert table["egyptian"] == Route(ADAPTERS, "egyptian")
    assert table["saudi"] == Route(ADAPTERS, "saudi")
    assert table["hejazi"] == Route(ADAPTERS, "hejazi")  # a name of both models: the route wins over the tag
    assert table["hindi_english"] == Route(ADAPTERS, "hindi_english")  # only the adapters model has it
    for name in ("gulf", "emirati", "qatari", "khaliji"):  # aliases follow the tagged model's dialects.json
        assert table[name] == Route(TAGGED, "gulf"), name
    for name in ("palestinian", "jordanian", "ab", "levantine"):
        assert table[name] == Route(TAGGED, "palestinian"), name
    for name in ("msa", "lebanese", "syrian", "iraqi", "najdi", "yemeni", "sudanese", "libyan", "tunisian", "algerian",
                 "moroccan"):
        assert table[name] == Route(TAGGED, name), name
    assert sorted(table) == sorted(set(ADAPTER_NAMES) | set(V3["tags"]) | set(V3["aliases"]))
    assert used_adapters(table) == ["egyptian", "hejazi", "hindi_english", "qatari", "saudi"]  # the engine loads these


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


def test_route_options_and_teacher_target():
    """--route DIALECT=ADAPTER / DIALECT=teacher (parse_routes) over the default table."""
    tags = DialectTags(V3)
    routes = parse_routes(["najdi=najdi", " egyptian = teacher "])
    assert routes == {**DEFAULT_ROUTES, "najdi": "najdi", "egyptian": TEACHER}
    table = route_table(ADAPTER_NAMES, tags, routes)
    assert table["najdi"] == Route(ADAPTERS, "najdi") and table["egyptian"] == Route(TAGGED, "egyptian")
    assert used_adapters(table) == ["hejazi", "hindi_english", "najdi", "qatari", "saudi"]
    assert describe(table) == "adapters: bahraini->qatari, hejazi, hindi_english, najdi, saudi; teacher: 19 names"
    teacher_only = parse_routes([f"{d}={TEACHER}" for d in DEFAULT_ROUTES])  # every dialect of teacher-v3 on it
    assert used_adapters(route_table(ADAPTER_NAMES, tags, teacher_only)) == ["hindi_english"]
    assert parse_routes(None) == DEFAULT_ROUTES and parse_routes([], base={}) == {}
    for bad in ("najdi", "=najdi", "najdi="):
        with pytest.raises(ValueError, match="DIALECT=ADAPTER"):
            parse_routes([bad])
    with pytest.raises(ValueError, match="does not speak"):
        route_table(ADAPTER_NAMES, tags, parse_routes(["hindi_english=teacher"]))
    with pytest.raises(ValueError, match="lacks"):
        route_table(ADAPTER_NAMES, tags, parse_routes(["najdi=nadji"]))


def test_ws_server_route_and_no_route(models, monkeypatch):
    """ws_server's --route changes the routed engine's table (and the audio cache's namespace); --no-route serves
    every adapter of the adapters model, no teacher-v3 (DialectTTS)."""
    from types import SimpleNamespace

    from nanovllm_voxcpm.entrypoints import warmup
    from nanovllm_voxcpm.entrypoints import ws_server

    async def no_warm_up(speak):
        pass

    monkeypatch.setattr(warmup, "warm_up", no_warm_up)

    def args(**kw):
        return SimpleNamespace(**{**dict(model=str(models / "adapters-model"), tagged_model=str(models / "tagged-model"),
                                         dit="none", head_tokens=None, max_concurrency=None, kv_blocks=None, voice=[],
                                         dialect="egyptian", default_voice=None, route=None, no_route=False,
                                         audio_cache_mb=1, audio_cache_dir=None, audio_cache_dir_mb=0), **kw})

    tts = run(ws_server.load_tts(args(route=["najdi=najdi", "egyptian=teacher"])))
    assert sorted(FakeEngine.started[-1].loras) == ["@tagged", "hejazi", "hindi_english", "najdi", "qatari", "saudi"]
    assert tts.route("egyptian") == Route(TAGGED, "egyptian") and tts.route("najdi") == Route(ADAPTERS, "najdi")
    tts = run(ws_server.load_tts(args(no_route=True)))
    assert sorted(FakeEngine.started[-1].loras) == sorted(ADAPTER_NAMES) and sorted(tts.dialects) == ADAPTER_NAMES
    caches = [ws_server.audio_cache(args(**kw), 48000) for kw in ({}, {"route": ["najdi=najdi"]}, {"no_route": True})]
    assert len({c.namespace for c in caches}) == 3  # another table, another audio: no file shared


def test_dispatch_adapters_tags_and_base():
    tts = routed()
    e = tts.server
    run(tts.synthesize("هلا", "bahraini"))
    assert e.requests[-1]["lora_name"] == "qatari" and e.requests[-1]["text"] == "هلا"
    run(tts.synthesize("هلا", "najdi"))  # teacher-v3: its adapter, the tag in front of the text
    assert e.requests[-1]["lora_name"] == TAGGED_ADAPTER and e.requests[-1]["text"] == V3["tags"]["najdi"] + "هلا"
    run(tts.synthesize("هلا", "qatari"))  # Gulf on teacher-v3
    assert e.requests[-1]["text"] == V3["tags"]["gulf"] + "هلا"
    run(tts.synthesize("मीटिंग 9 बजे", "hindi_english"))
    assert (e.requests[-1]["text"], e.requests[-1]["lora_name"]) == ("मीटिंग 9 बजे", "hindi_english")
    run(tts.synthesize("hello", None))  # no dialect: the base model, as DialectTTS
    assert e.requests[-1]["lora_name"] is None and e.requests[-1]["text"] == "hello"
    run(tts.synthesize("الساعة 9:05", "msa", seed=7, cfg_value=1.5, max_patches=50))  # formal, arguments passed on
    assert e.requests[-1] == {"text": V3["tags"]["msa"] + "الساعة تسعة وخمس دقائق", "lora_name": TAGGED_ADAPTER,
                              "seed": 7, "cfg_value": 1.5, "max_generate_length": 50}
    run(tts.synthesize("الساعة 9:05", "egyptian"))  # an adapter: colloquial, no tag
    assert e.requests[-1]["text"] == "الساعة تسعة وخمس دقايق"
    with pytest.raises(ValueError, match="unknown dialect 'kuwaiti'"):
        run(tts.synthesize("هلا", "kuwaiti"))
    with pytest.raises(ValueError, match="empty"):
        run(tts.synthesize("ـــ", "najdi"))
    assert "bahraini" in tts.dialects and "hindi_english" in tts.dialects and tts.sample_rate == 48000


def test_voices_and_tagged_copies():
    tts = routed()
    e = tts.server
    run(tts.add_voice("v", b"wav", text="الساعة 9:30"))
    assert e.voices == {"v": "الساعة تسعة ونص"} and tts.voices == {"v": True}
    run(tts.synthesize("هلا", "saudi", voice="v"))  # an adapter: the voice itself, prompt_ref
    assert (e.requests[-1]["voice"], e.requests[-1]["clone_mode"]) == ("v", "prompt_ref")
    run(tts.synthesize("هلا", "msa", voice="v"))  # teacher-v3 prompt_ref: the tag leads the transcript of a copy
    tag = V3["tags"]["msa"]
    assert e.requests[-1]["voice"] == "v@" + tag and e.voices["v@" + tag] == tag + "الساعة تسعة والنصف"
    assert e.requests[-1]["lora_name"] == TAGGED_ADAPTER and e.requests[-1]["text"] == "هلا"
    run(tts.synthesize("هلا", "najdi", voice="v", clone_mode="ref"))  # ref: the tag leads the text
    assert e.requests[-1]["voice"] == "v" and e.requests[-1]["text"] == V3["tags"]["najdi"] + "هلا"
    run(tts.remove_voice("v"))
    assert e.voices == {} and tts.voices == {}
    with pytest.raises(ValueError, match="not registered"):
        run(tts.synthesize("هلا", "najdi", voice="v"))


@pytest.fixture
def models(tmp_path, monkeypatch):
    """An adapters model and a tagged model's training state on disk, the fake engine as VoxCPM."""
    import nanovllm_voxcpm

    a, t = tmp_path / "adapters-model", tmp_path / "tagged-model"
    for name in ADAPTER_NAMES:
        (a / "adapters" / name).mkdir(parents=True)
        (a / "adapters" / name / "lora_weights.safetensors").write_bytes(b"")
        (a / "adapters" / name / "lora_config.json").write_text(json.dumps({"lora_config": ADAPTER_LORA}))
    (t / "lora").mkdir(parents=True)
    (t / "lora" / "trainable.safetensors").write_bytes(b"")
    (t / "lora" / "train_state.json").write_text(json.dumps(V3_STATE))
    (t / "dialects.json").write_text(json.dumps(V3), encoding="utf-8")
    (tmp_path / "dit.pt").write_bytes(b"")
    monkeypatch.setattr(nanovllm_voxcpm, "VoxCPM", FakeEngine, raising=False)
    for k in ("NANOVLLM_VOXCPM2_TURBO", "NANOVLLM_SERVERPOOL_NUM_KVCACHE_BLOCKS", "VOXCPM_KERNELS_SMOOTH"):
        monkeypatch.delenv(k, raising=False)
    FakeEngine.started = []
    return tmp_path


def test_routed_lora_config(models):
    adapters = {n: str(models / "adapters-model" / "adapters" / n) for n in ("egyptian", "qatari", "saudi")}
    cfg = routed_lora_config(adapters, str(models / "tagged-model"))
    assert cfg.max_loras == 4 and cfg.max_lora_rank == 64 and cfg.tag_rows == 17 and cfg.max_mlp_loras == 1
    assert cfg.target_modules_lm == ALL7
    assert cfg.target_modules_residual_lm == ["q_proj", "k_proj", "v_proj", "o_proj"]  # the tagged LoRA: base_lm
    assert cfg.enable_dit and cfg.target_modules_dit == ["q_proj", "k_proj", "v_proj", "o_proj"]  # adapters' only


def test_load_one_engine_with_the_used_adapters(models):
    tts = run(RoutedDialectTTS.load(str(models / "adapters-model"), str(models / "tagged-model"),
                                    str(models / "dit.pt"), max_concurrency=64, kv_blocks=300,
                                    smooth_stats=str(models / "stats.pt")))
    (e,) = FakeEngine.started
    assert tts.server is e
    assert sorted(e.loras) == ["@tagged", "egyptian", "hejazi", "hindi_english", "qatari", "saudi"]
    assert e.loras["@tagged"] == str(models / "tagged-model")
    cfg = e.kw["lora_config"]
    assert cfg.max_loras == 6 and cfg.max_mlp_loras == 1 and cfg.tag_rows == 17
    turbo = json.loads(e.env["NANOVLLM_VOXCPM2_TURBO"])
    assert e.kw["max_num_seqs"] == 64 and turbo["dit_checkpoint"].endswith("dit.pt") and turbo["w8a8"]
    assert e.env["VOXCPM_KERNELS_SMOOTH"].endswith("stats.pt")
    assert tts.route("najdi") == Route(TAGGED, "najdi") and tts.route("bahraini") == Route(ADAPTERS, "qatari")
    run(tts.close())
    assert e.stopped
    FakeEngine.started = []
    run(RoutedDialectTTS.load(str(models / "adapters-model"), str(models / "tagged-model"), None,
                              routes={**DEFAULT_ROUTES, "najdi": "najdi"}))
    assert sorted(FakeEngine.started[0].loras) == ["@tagged", "egyptian", "hejazi", "hindi_english", "najdi", "qatari",
                                                   "saudi"]
    with pytest.raises(ValueError, match="not a tagged model"):
        run(RoutedDialectTTS.load(str(models / "adapters-model"), str(models / "adapters-model"), None))
    with pytest.raises(ValueError, match="lacks"):
        run(RoutedDialectTTS.load(str(models / "adapters-model"), str(models / "tagged-model"), None,
                                  routes={"bahraini": "kuwaiti"}))


def test_tagged_adapter_loader(tmp_path):
    """A tagged model's lora/: the LoRA (attention and MLP, packed like the adapters'), the tag rows, exact_merge."""
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    from nanovllm_voxcpm.model.lora_loader import is_tagged_adapter, load_voxcpm2_lora_checkpoint

    (tmp_path / "lora").mkdir()
    g = torch.Generator().manual_seed(0)
    t = {"base_lm.embed_tokens.tag_embed": torch.randn(17, 8, generator=g)}
    for leaf, out_dim, in_dim in (("self_attn.q_proj", 8, 8), ("self_attn.k_proj", 4, 8), ("self_attn.v_proj", 4, 8),
                                  ("self_attn.o_proj", 8, 8), ("mlp.gate_proj", 16, 8), ("mlp.up_proj", 16, 8),
                                  ("mlp.down_proj", 8, 16)):
        t[f"base_lm.layers.0.{leaf}.lora_A"] = torch.randn(2, in_dim, generator=g)
        t[f"base_lm.layers.0.{leaf}.lora_B"] = torch.randn(out_dim, 2, generator=g)
    save_file(t, str(tmp_path / "lora" / "trainable.safetensors"))
    state = {**V3_STATE, "lora": {**V3_STATE["lora"], "r": 2, "alpha": 4}}
    (tmp_path / "lora" / "train_state.json").write_text(json.dumps(state))
    assert not is_tagged_adapter(str(tmp_path))  # no dialects.json yet
    (tmp_path / "dialects.json").write_text(json.dumps(V3))
    assert is_tagged_adapter(str(tmp_path))
    p = load_voxcpm2_lora_checkpoint(str(tmp_path))
    assert p.exact_merge and p.rank == 2 and torch.equal(p.tag_embed, t["base_lm.embed_tokens.tag_embed"])
    assert sorted(p.modules) == ["base_lm.layers.0.mlp.down_proj", "base_lm.layers.0.mlp.gate_up_proj",
                                 "base_lm.layers.0.self_attn.o_proj", "base_lm.layers.0.self_attn.qkv_proj"]
    gu = p.modules["base_lm.layers.0.mlp.gate_up_proj"]
    assert gu.scaling == 2.0 and torch.equal(gu.lora_a[1], t["base_lm.layers.0.mlp.up_proj.lora_A"])
    assert torch.equal(gu.lora_b[0], t["base_lm.layers.0.mlp.gate_proj.lora_B"])
    qkv = p.modules["base_lm.layers.0.self_attn.qkv_proj"]
    assert [b.shape[0] for b in qkv.lora_b] == [8, 4, 4]


def test_extra_embedding_rows():
    torch = pytest.importorskip("torch")
    from nanovllm_voxcpm.model.layers import VocabParallelEmbedding

    emb = VocabParallelEmbedding(10, 4, extra_rows=3)
    emb.weight_loader(emb.weight, torch.arange(40, dtype=torch.float32).view(10, 4))
    assert emb.weight.shape == (13, 4) and torch.equal(emb.weight[:10].flatten(), torch.arange(40.0))
    assert not emb.weight[10:].any()
    emb.set_rows(11, torch.ones(2, 4))
    out = emb(torch.tensor([2, 11, 12, 10]))
    assert torch.equal(out[0], torch.arange(8.0, 12.0)) and out[1:3].eq(1).all() and not out[3].any()
    with pytest.raises(ValueError, match="extra embedding rows"):
        emb.set_rows(9, torch.ones(1, 4))  # the vocabulary's own rows stay the checkpoint's


def test_engine_maps_a_tagged_adapters_tags_to_its_rows(tmp_path, monkeypatch):
    """VoxCPM2Engine._tagged_adapter / add_request on a fake tokenizer: the tags become the extra rows (after the
    vocabulary) in the tagged adapter's requests only; the engine reads tags with the adapter's tokenizer."""
    pytest.importorskip("torch")
    from types import SimpleNamespace

    from nanovllm_voxcpm.model import engine as E

    vocab = {"<reserved_100>": 189, "<reserved_101>": 190, "a": 5, "b": 6}

    class Tok:
        def __init__(self, special):
            self.special = special
            self.tokenizer = SimpleNamespace(get_vocab=lambda: dict(vocab))

        def __call__(self, text):
            out = []
            for tag, i in (("<reserved_100>", 189), ("<reserved_101>", 190)):
                if self.special and text.startswith(tag):
                    out.append(i)
                    text = text[len(tag):]
            return out + [vocab.get(c, 1) for c in text]

    (tmp_path / "lora").mkdir()
    (tmp_path / "lora" / "train_state.json").write_text(json.dumps({"lora": {}, "dialects": ["msa", "egyptian"]}))
    (tmp_path / "dialects.json").write_text(json.dumps({"tags": {"msa": "<reserved_100>",
                                                                 "egyptian": "<reserved_101>"}}))
    monkeypatch.setattr(E.LlamaTokenizerFast, "from_pretrained", staticmethod(lambda p: "tagged"))
    monkeypatch.setattr(E, "mask_multichar_chinese_tokens", lambda t: Tok(special=True))
    eng = E.VoxCPM2Engine.__new__(E.VoxCPM2Engine)
    eng.tokenizer, eng.vocab_size, eng.tag_rows = Tok(special=False), 100, 17
    payload = SimpleNamespace(tag_embed=np.zeros((2, 4)), tag_row_start=-1)
    remap, tok = eng._tagged_adapter(str(tmp_path), payload)
    assert remap == {189: 100, 190: 101} and payload.tag_row_start == 100 and tok.special  # the engine's misread tags
    eng._tag_rows_used = 16
    with pytest.raises(ValueError, match="tag_rows=17"):
        eng._tagged_adapter(str(tmp_path), payload)


def test_server_routes_by_default(models, monkeypatch):
    """openai_server: a model with adapters/ is served routed unless --no-route; a request's dialect picks its adapter,
    or teacher-v3 with its tag (a voice copy for prompt_ref); /v1/models lists the dialects."""
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")
    from test_wiring import serve, server_args

    from nanovllm_voxcpm.entrypoints import openai_server

    voices = models / "voices.json"
    (models / "v.wav").write_bytes(b"wav")
    voices.write_text(json.dumps([{"id": "v", "wav": str(models / "v.wav"), "text": "عمري 25 سنة"}]))
    asgi = openai_server.build_app(server_args(models / "adapters-model", no_route=False,
                                               tagged_model=str(models / "tagged-model"), voices=str(voices)))
    out = run(serve(asgi, [
        {"input": "هلا", "dialect": "najdi"},                      # teacher-v3, its tag
        {"input": "هلا", "dialect": "bahraini"},                   # the qatari adapter
        {"input": "هلا", "model": "hejazi"},                       # a dialect named as the model: its adapter
        {"input": "الساعة 9:05", "dialect": "msa", "voice": "v"},  # prompt_ref on teacher-v3: the voice's msa copy
        {"input": "هلا", "dialect": "kuwaiti"},
    ]))
    assert [status for status, _ in out] == [200, 200, 200, 200, 404]
    (e,) = FakeEngine.started
    assert sorted(e.loras) == ["@tagged", "egyptian", "hejazi", "hindi_english", "qatari", "saudi"]
    assert e.kw["lora_config"].max_mlp_loras == 1 and e.kw["lora_config"].tag_rows == 17
    reqs = [r for r in e.requests if r["text"] != "Warm up." and r.get("voice") != "_warmup"]  # the start's warm-ups
    assert reqs[0]["text"] == V3["tags"]["najdi"] + "هلا" and reqs[0]["lora_name"] == TAGGED_ADAPTER
    assert reqs[1]["text"] == "هلا" and reqs[1]["lora_name"] == "qatari"
    assert reqs[2]["lora_name"] == "hejazi"
    tag = V3["tags"]["msa"]
    assert reqs[3]["voice"] == "v@" + tag and reqs[3]["text"] == "الساعة تسعة وخمس دقائق"
    assert e.voices["v@" + tag] == tag + "عمري خمسة وعشرين سنة"
    FakeEngine.started = []
    run(serve(openai_server.build_app(server_args(models / "adapters-model", no_route=True)), []))
    assert sorted(FakeEngine.started[0].loras) == sorted(ADAPTER_NAMES)  # --no-route: every adapter, no teacher-v3
    FakeEngine.started = []  # --route: najdi on its adapter, egyptian on teacher-v3
    asgi = openai_server.build_app(server_args(models / "adapters-model", no_route=False, route=["najdi=najdi",
                                                                                                 "egyptian=teacher"],
                                               tagged_model=str(models / "tagged-model")))
    out = run(serve(asgi, [{"input": "هلا", "dialect": "najdi"}, {"input": "هلا", "dialect": "egyptian"}]))
    assert [status for status, _ in out] == [200, 200]
    (e,) = FakeEngine.started
    assert sorted(e.loras) == ["@tagged", "hejazi", "hindi_english", "najdi", "qatari", "saudi"]
    reqs = [r for r in e.requests if r["text"] != "Warm up." and r.get("voice") != "_warmup"]
    assert (reqs[0]["lora_name"], reqs[0]["text"]) == ("najdi", "هلا")
    assert (reqs[1]["lora_name"], reqs[1]["text"]) == (TAGGED_ADAPTER, V3["tags"]["egyptian"] + "هلا")
