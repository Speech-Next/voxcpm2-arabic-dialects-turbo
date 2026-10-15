"""Where the normalizer applies: DialectTTS and the server, on a fake engine (no GPU): Arabic dialects only, the style
of the resolved dialect (aliases, adapters, --default-lora), the empty check after normalization, reference
transcripts read like target text."""

import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest

import nanovllm_voxcpm
from nanovllm_voxcpm.dialect_tts import DialectTTS, read_as

TAGS = {"tags": {"msa": "<msa>", "gulf": "<gulf>"}, "aliases": {"fusha": "msa", "qatari": "gulf"}}


class FakeEngine:
    """The engine's async API, recording what it is sent."""

    def __init__(self):
        self.targets: list[str] = []
        self.voices: dict[str, str | None] = {}

    @classmethod
    def from_pretrained(cls, **kw):
        cls.last = cls()
        return cls.last

    async def wait_for_ready(self):
        pass

    async def register_lora(self, name, path):
        pass

    async def get_model_info(self):
        return {"sample_rate": 48000}

    async def encode_latents(self, audio, audio_format):
        return b"latents"

    async def register_voice(self, name, latents, text):
        self.voices[name] = text

    async def unregister_voice(self, name):
        self.voices.pop(name, None)

    async def generate(self, target_text, **kw):
        self.targets.append(target_text)
        yield np.zeros(4, dtype=np.int16)

    async def stop(self):
        pass


def test_read_as():
    assert read_as("मीटिंग 9 बजे", "hindi_english") == "मीटिंग 9 बजे"
    assert read_as("الساعة 9", "hindi_english") == "الساعة 9"
    assert read_as("meet at 9", "najdi") == "meet at 9"  # no Arabic letter: not Arabic text
    assert read_as("الساعة 9:05", "najdi") == "الساعة تسعة وخمس دقايق"
    assert read_as("الساعة 9:05", "msa") == "الساعة تسعة وخمس دقائق"
    assert read_as("الساعة 9:05") == "الساعة تسعة وخمس دقايق"


def run(coro):
    return asyncio.run(coro)


async def say(tts, text, dialect, **kw):
    await tts.synthesize(text, dialect, **kw)
    return tts.server.targets[-1]


def test_dialect_tts_adapters():
    tts = DialectTTS(FakeEngine(), ["hindi_english", "msa", "najdi"], 48000)
    assert run(say(tts, "मीटिंग 9 बजे", "hindi_english")) == "मीटिंग 9 बजे"
    assert run(say(tts, "الساعة 9:05", "najdi")) == "الساعة تسعة وخمس دقايق"
    assert run(say(tts, "الساعة 9:05", "msa")) == "الساعة تسعة وخمس دقائق"
    assert run(say(tts, "الساعة 9:05", "najdi", normalize=False)) == "الساعة 9:05"
    with pytest.raises(ValueError, match="empty"):
        run(say(tts, "ـــ", "najdi"))  # tatweel alone reads as nothing
    with pytest.raises(ValueError, match="empty"):
        run(say(tts, "  ", "najdi", normalize=False))


def test_dialect_tts_tagged_aliases_and_transcripts():
    tts = DialectTTS(FakeEngine(), [], 48000, TAGS)
    assert run(say(tts, "الساعة 9:05", "fusha")) == "<msa>الساعة تسعة وخمس دقائق"  # fusha -> msa: formal
    assert run(say(tts, "الساعة 9:05", "qatari")) == "<gulf>الساعة تسعة وخمس دقايق"
    run(tts.add_voice("v", b"wav", text="الساعة 9:30"))
    assert tts.server.voices["v"] == "الساعة تسعة ونص"  # the transcript as target text is read
    run(say(tts, "هلا", "fusha", voice="v"))  # prompt_ref: the dialect's copy of the voice, its transcript in style
    assert tts.server.voices["v@<msa>"] == "<msa>الساعة تسعة والنصف"
    run(say(tts, "هلا", "qatari", voice="v"))
    assert tts.server.voices["v@<gulf>"] == "<gulf>الساعة تسعة ونص"
    run(tts.add_voice("raw", b"wav", text="الساعة 9:30", normalize=False))
    assert tts.server.voices["raw"] == "الساعة 9:30"


def server_args(model, **kw):
    args = dict(model=str(model), dit=None, dit_adapters=True, dit_cache=None, head_tokens=8, kv_blocks=0,
                lora_dir=None, lora=[], default_lora=None, max_lora_rank=None, inference_timesteps=10, max_num_seqs=8,
                max_model_len=2048, max_num_batched_tokens=2048, gpu_memory_utilization=0.5, voice_dir=None,
                voices=None, served_model_name="voxcpm2", split_max_chars=200, max_tokens=600)
    return SimpleNamespace(**{**args, **kw})


async def serve(asgi, requests: list[dict]) -> list[tuple[int, bytes]]:
    """Start the app's lifespan, POST each request to /v1/audio/speech, shut down."""
    inbox, outbox = asyncio.Queue(), asyncio.Queue()
    await inbox.put({"type": "lifespan.startup"})
    life = asyncio.ensure_future(asgi({"type": "lifespan", "asgi": {"version": "3.0"}, "state": {}}, inbox.get,
                                      outbox.put))
    msg = await outbox.get()
    assert msg["type"] == "lifespan.startup.complete", msg
    out = []
    for req in requests:
        sent = []

        async def receive(body=json.dumps(req).encode()):
            return {"type": "http.request", "body": body, "more_body": False}

        async def send(m):
            sent.append(m)

        await asgi({"type": "http", "method": "POST", "path": "/v1/audio/speech", "headers": []}, receive, send)
        out.append((sent[0]["status"], b"".join(m.get("body", b"") for m in sent[1:])))
    await inbox.put({"type": "lifespan.shutdown"})
    await life
    return out


@pytest.fixture
def server(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")
    from nanovllm_voxcpm.entrypoints import openai_server

    monkeypatch.setattr(nanovllm_voxcpm, "VoxCPM", FakeEngine, raising=False)
    monkeypatch.delenv("NANOVLLM_VOXCPM2_TURBO", raising=False)
    return openai_server


def test_server_adapters(server, monkeypatch, tmp_path):
    monkeypatch.setattr(server, "find_adapters", lambda *a: {"hindi_english": "a", "msa": "b", "najdi": "c"})
    monkeypatch.setattr(server, "adapters_lora_config", lambda *a: None)
    asgi = server.build_app(server_args(tmp_path, default_lora="msa"))
    out = run(serve(asgi, [
        {"input": "الساعة 9:05"},                                   # --default-lora msa: formal
        {"input": "الساعة 9:05", "model": "najdi"},                 # the adapter named as the model
        {"input": "मीटिंग 9 बजे", "lora": "hindi_english"},
        {"input": "الساعة 9:05", "dialect": "najdi", "normalize": False},
        {"input": "ـــ", "dialect": "najdi"},                        # empty once the tatweel is gone
    ]))
    assert [status for status, _ in out] == [200, 200, 200, 200, 400]
    assert b"empty" in out[-1][1]
    assert FakeEngine.last.targets[-4:] == ["الساعة تسعة وخمس دقائق", "الساعة تسعة وخمس دقايق", "मीटिंग 9 बजे",
                                            "الساعة 9:05"]


def test_server_tagged_model_voice_transcript(server, tmp_path):
    (tmp_path / "dialects.json").write_text(json.dumps(TAGS), encoding="utf-8")
    (tmp_path / "v.wav").write_bytes(b"wav")
    voices = tmp_path / "voices.json"
    voices.write_text(json.dumps([{"id": "v", "wav": str(tmp_path / "v.wav"), "text": "عمري 25 سنة"}]),
                      encoding="utf-8")
    asgi = server.build_app(server_args(tmp_path, voices=str(voices)))
    out = run(serve(asgi, [{"input": "الساعة 9:05", "dialect": "fusha"},
                           {"input": "هلا", "dialect": "fusha", "voice": "v"}]))
    assert [status for status, _ in out] == [200, 200]
    engine = FakeEngine.last
    assert engine.voices["v"] == "عمري خمسة وعشرين سنة"
    assert engine.voices["v@<msa>"] == "<msa>عمري خمسة وعشرين سنة"
    assert "<msa>الساعة تسعة وخمس دقائق" in engine.targets
