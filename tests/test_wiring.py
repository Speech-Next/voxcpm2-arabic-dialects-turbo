"""Where the normalizer applies: DialectTTS and the server, on a fake engine (no GPU): Arabic dialects only, the style
of the resolved dialect (aliases, adapters, --default-lora), the empty check after normalization, reference
transcripts read like target text."""

import asyncio
import contextlib
import io
import json
import os
import shutil
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
    args = dict(model=str(model), dit=None, dit_cache=None, head_tokens=8, kv_blocks=0, lora_dir=None, lora=[],
                default_lora=None, max_lora_rank=None, inference_timesteps=10, max_num_seqs=8, max_model_len=2048,
                max_num_batched_tokens=2048, gpu_memory_utilization=0.5, voice_dir=None, voices=None,
                served_model_name="voxcpm2", split_max_chars=200, max_tokens=600, no_route=True, tagged_model=None,
                host="127.0.0.1", port=8092, api_key=None, max_streams=0, max_input_chars=4096, max_voice_mb=10.0,
                max_voice_seconds=30.0, hang_timeout=60.0, shutdown_timeout=10.0)
    return SimpleNamespace(**{**args, **kw})


@contextlib.asynccontextmanager
async def running(asgi):
    """The app between its lifespan's startup and shutdown."""
    inbox, outbox = asyncio.Queue(), asyncio.Queue()
    await inbox.put({"type": "lifespan.startup"})
    life = asyncio.ensure_future(asgi({"type": "lifespan", "asgi": {"version": "3.0"}, "state": {}}, inbox.get,
                                      outbox.put))
    msg = await outbox.get()
    assert msg["type"] == "lifespan.startup.complete", msg
    try:
        yield
    finally:
        await inbox.put({"type": "lifespan.shutdown"})
        await life


async def request(asgi, method: str, path: str, body: bytes = b"", headers=(), content_type=b"application/json",
                  on_send=None) -> tuple[int, dict, bytes]:
    """(status, headers, body) of one request; receive() as an ASGI server's: the body, then (only) the
    disconnect, once the response is done or ``on_send(message, gone)`` sets ``gone``."""
    sent, gone, first = [], asyncio.Event(), [True]

    async def receive():
        if first[0]:
            first[0] = False
            return {"type": "http.request", "body": body, "more_body": False}
        await gone.wait()
        return {"type": "http.disconnect"}

    async def send(m):
        sent.append(m)
        if on_send is not None:
            on_send(m, gone)

    scope = {"type": "http", "method": method, "path": path, "raw_path": path.encode(), "query_string": b"",
             "headers": [(b"content-type", content_type), *headers], "http_version": "1.1", "scheme": "http",
             "server": ("127.0.0.1", 8092), "client": ("127.0.0.1", 50000), "root_path": "", "asgi": {"version": "3.0"}}
    await asgi(scope, receive, send)
    gone.set()
    start = next(m for m in sent if m["type"] == "http.response.start")
    return start["status"], dict(start["headers"]), b"".join(m.get("body", b"") for m in sent[1:])


async def serve(asgi, requests: list[dict]) -> list[tuple[int, bytes]]:
    """Start the app's lifespan, POST each request to /v1/audio/speech, shut down."""
    out = []
    async with running(asgi):
        for req in requests:
            status, _, body = await request(asgi, "POST", "/v1/audio/speech", json.dumps(req).encode())
            out.append((status, body))
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


# --- production: API key, limits, busy, client disconnects, engine errors and death ---


class SlowEngine(FakeEngine):
    """50 chunks, 10 ms apart; counts the requests closed before their end."""

    closed_early = 0

    async def generate(self, target_text, **kw):
        self.targets.append(target_text)
        try:
            for _ in range(50):
                await asyncio.sleep(0.01)
                yield np.zeros(4, dtype=np.int16)
        except BaseException:
            type(self).closed_early += 1
            raise


class RejectingEngine(FakeEngine):
    async def generate(self, target_text, **kw):  # (as AsyncVoxCPM2Server: the engine's ValueError as itself)
        if target_text == "hello":  # (not the start-up warm-up's)
            raise ValueError("Request may exceed max_model_len")
        yield np.zeros(4, dtype=np.int16)


class DyingEngine(FakeEngine):
    alive = True

    async def wait_dead(self):
        await asyncio.sleep(0.05)
        self.alive = False


def speech_body(**kw) -> bytes:
    return json.dumps({"input": "hello", **kw}).encode()


def test_api_key(server, tmp_path):
    asgi = server.build_app(server_args(tmp_path, api_key="k"))

    async def main():
        async with running(asgi):
            return [(await request(asgi, "POST", "/v1/audio/speech", speech_body(), headers=h))[0] for h in
                    ((), [(b"authorization", b"Bearer wrong")], [(b"authorization", b"Bearer k")])] + [
                (await request(asgi, "GET", "/health"))[0], (await request(asgi, "GET", "/v1/models"))[0]]

    assert asyncio.run(main()) == [401, 401, 200, 200, 401]


def test_limits(server, tmp_path):
    asgi = server.build_app(server_args(tmp_path, max_input_chars=10))
    out = run(serve(asgi, [{"input": "x" * 11}, {"input": "hello", "cfg_value": float("nan")},
                           {"input": "hello", "cfg_value": 11}, {"input": "hello", "max_tokens": 0},
                           {"input": "hello", "cfg_value": 3.0, "max_tokens": 5}]))
    assert [status for status, _ in out] == [400, 422, 422, 422, 200]
    assert b"at most 10 characters" in out[0][1]


def test_engine_value_error_is_a_400(server, monkeypatch, tmp_path):
    monkeypatch.setattr(nanovllm_voxcpm, "VoxCPM", RejectingEngine)
    out = run(serve(server.build_app(server_args(tmp_path)), [{"input": "hello"}]))
    assert out[0][0] == 400 and b"max_model_len" in out[0][1]


def test_busy_beyond_max_streams(server, monkeypatch, tmp_path):
    monkeypatch.setattr(nanovllm_voxcpm, "VoxCPM", SlowEngine)
    asgi = server.build_app(server_args(tmp_path, max_streams=1))

    async def main():
        async with running(asgi):
            first = asyncio.ensure_future(request(asgi, "POST", "/v1/audio/speech", speech_body()))
            await asyncio.sleep(0.05)
            busy = await request(asgi, "POST", "/v1/audio/speech", speech_body())
            return (await first)[0], busy

    status, (busy_status, headers, body) = asyncio.run(main())
    assert status == 200 and busy_status == 503 and headers[b"retry-after"] == b"1" and b"busy" in body


def test_a_client_gone_ends_its_engine_request(server, monkeypatch, tmp_path):
    # uvicorn's send() returns quietly once the client is gone: only the disconnect stops the generation
    monkeypatch.setattr(nanovllm_voxcpm, "VoxCPM", SlowEngine)
    SlowEngine.closed_early = 0
    asgi = server.build_app(server_args(tmp_path))

    def leave_after_two_chunks(m, gone):
        if m.get("more_body") and m.get("body"):
            leave_after_two_chunks.n += 1
            if leave_after_two_chunks.n == 2:
                gone.set()

    leave_after_two_chunks.n = 0

    async def main():
        async with running(asgi):
            await request(asgi, "POST", "/v1/audio/speech", speech_body(), on_send=leave_after_two_chunks)
            await asyncio.sleep(0.05)

    asyncio.run(main())
    assert SlowEngine.closed_early == 1
    assert leave_after_two_chunks.n < 10


def test_engine_death_stops_the_server(server, monkeypatch, tmp_path):
    monkeypatch.setattr(nanovllm_voxcpm, "VoxCPM", DyingEngine)
    asgi = server.build_app(server_args(tmp_path))
    stopped = []
    asgi.state["on_engine_dead"] = lambda: stopped.append(True)

    async def main():
        async with running(asgi):
            ok = (await request(asgi, "GET", "/health"))[0]
            await asyncio.sleep(0.1)
            return ok, (await request(asgi, "GET", "/health"))[0], (await request(asgi, "POST", "/v1/audio/speech",
                                                                                   speech_body()))[0]

    assert asyncio.run(main()) == (200, 503, 503)
    assert stopped == [True] and asgi.state["dead"]


def multipart(fields: dict, files: dict) -> tuple[bytes, bytes]:
    out = []
    for k, v in fields.items():
        out.append(f'--b0undary\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    for k, (name, data) in files.items():
        out.append(f'--b0undary\r\nContent-Disposition: form-data; name="{k}"; filename="{name}"\r\n'
                   f"Content-Type: application/octet-stream\r\n\r\n".encode() + data + b"\r\n")
    return b"".join(out) + b"--b0undary--\r\n", b"multipart/form-data; boundary=b0undary"


def wav(seconds: float, sr: int = 24000) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    t = np.arange(int(seconds * sr)) / sr
    sf.write(buf, (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_voice_upload_limits(server, monkeypatch, tmp_path):
    pytest.importorskip("multipart")
    import soundfile as sf

    clips = []

    class Engine(FakeEngine):
        async def encode_latents(self, audio, audio_format):
            clips.append((audio, audio_format))
            return b"latents"

    monkeypatch.setattr(nanovllm_voxcpm, "VoxCPM", Engine)
    asgi = server.build_app(server_args(tmp_path, max_voice_mb=1.0))

    async def upload(name, data):
        body, ctype = multipart({"name": name}, {"audio_sample": ("clip.wav", data)})
        return await request(asgi, "POST", "/v1/audio/voices", body, content_type=ctype)

    async def main():
        async with running(asgi):
            return [await upload(n, d) for n, d in (("ok", wav(2.0)), ("noise", b"not audio" * 100),
                                                     ("long", wav(31.0, 8000)), ("big", wav(25.0)), ("ok", wav(2.0)))]

    out = asyncio.run(main())
    assert [status for status, _, _ in out] == [200, 400, 400, 413, 409]
    assert b"at most 30 s" in out[2][2]
    audio, fmt = clips[-1]  # (startup's warm-up voice first): decoded here, 16 kHz float WAV for the engine
    info = sf.info(io.BytesIO(audio))
    assert fmt == "wav" and info.samplerate == 16000 and info.subtype == "FLOAT" and abs(info.duration - 2.0) < 0.01


def test_a_body_that_is_not_json_is_a_422(server, tmp_path):
    asgi = server.build_app(server_args(tmp_path))

    async def main():
        async with running(asgi):
            return [(await request(asgi, "POST", "/v1/audio/speech", body))[:3:2] for body in (b"not json", b"{")]

    for status, body in asyncio.run(main()):
        assert status == 422 and b"json" in body.lower()


def test_nothing_to_speak(server, tmp_path):
    from nanovllm_voxcpm.dialect_tts import speakable

    assert not any(map(speakable, ("...", "؟!", "🙂 🙂", "ـــ", "—", "")))
    assert all(map(speakable, ("ok", "مرحبا", "123", "你好")))
    out = run(serve(server.build_app(server_args(tmp_path)), [{"input": "..."}, {"input": "🙂 !!"}, {"input": "؟!"},
                                                              {"input": "ok!"}]))
    assert [status for status, _ in out] == [400, 400, 400, 200]
    assert b"nothing to speak" in out[0][1]
    tts = DialectTTS(FakeEngine(), ["najdi"], 48000)
    with pytest.raises(ValueError, match="nothing to speak"):
        run(say(tts, "!!! ...", "najdi"))


def test_a_voice_in_use_is_not_deleted(server, monkeypatch, tmp_path):
    # its next sentence would fail, and the response stop after a 200
    monkeypatch.setattr(nanovllm_voxcpm, "VoxCPM", SlowEngine)
    (tmp_path / "v.wav").write_bytes(b"wav")
    voices = tmp_path / "voices.json"
    voices.write_text(json.dumps([{"id": "v", "wav": str(tmp_path / "v.wav")}]), encoding="utf-8")
    asgi = server.build_app(server_args(tmp_path, voices=str(voices)))

    async def main():
        async with running(asgi):
            speaking = asyncio.ensure_future(request(asgi, "POST", "/v1/audio/speech", speech_body(voice="v")))
            await asyncio.sleep(0.05)
            in_use = await request(asgi, "DELETE", "/v1/audio/voices/v")
            spoken = await speaking
            return in_use, spoken[0], (await request(asgi, "DELETE", "/v1/audio/voices/v"))[0]

    (status, _, body), spoken, deleted = asyncio.run(main())
    assert (status, spoken, deleted) == (409, 200, 200) and b"in use" in body


def test_the_engine_is_watched_for_hangs_once_warmed_up(server, monkeypatch, tmp_path):
    class Engine(FakeEngine):
        watched = None

        def watch_hangs(self, timeout_s):
            type(self).watched = (timeout_s, len(self.targets))  # (after the warm-up's requests)

    monkeypatch.setattr(nanovllm_voxcpm, "VoxCPM", Engine)
    run(serve(server.build_app(server_args(tmp_path, hang_timeout=7.0)), []))
    assert Engine.watched[0] == 7.0 and Engine.watched[1] > 0


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("fmt", ["pcm", "wav", "mp3", "opus", "flac"])
def test_response_formats(server, monkeypatch, tmp_path, fmt, stream):
    """#47: mp3, opus and flac besides pcm and wav, streamed as they are encoded (wav whole); each decodes to the
    second of audio the engine made."""
    import shutil
    import subprocess

    sf = pytest.importorskip("soundfile")

    async def generate(self, target_text, **kw):  # 1 s at 48 kHz in ten chunks
        t = np.arange(4800) / 48000
        for i in range(10):
            yield (np.sin(2 * np.pi * 220 * (t + i * 0.1)) * 8000).astype(np.int16)

    monkeypatch.setattr(FakeEngine, "generate", generate)
    asgi = server.build_app(server_args(tmp_path))
    sent = []

    async def main():
        async with running(asgi):
            body = json.dumps({"input": "ok", "response_format": fmt, "stream": stream}).encode()
            return await request(asgi, "POST", "/v1/audio/speech", body,
                                 on_send=lambda m, gone: sent.append(m.get("body")))

    status, headers, body = run(main())
    assert status == 200 and headers[b"content-type"] == server.AUDIO_FORMATS[fmt][0]
    assert (b"content-length" in headers) == (not stream or fmt == "wav")
    if stream and fmt != "wav":  # sent as it comes, not at the end
        assert sum(1 for b in sent if b) >= (10 if fmt == "pcm" else 3)
    if fmt == "pcm":
        assert len(body) == 2 * 48000
        return
    if fmt == "flac" and stream:  # a streamed flac has no length (0: unknown); a stream decoder reads it all
        assert body[:4] == b"fLaC"
        if shutil.which("ffmpeg") is None:
            return
        pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", "-", "-f", "s16le", "-"], input=body,
                             capture_output=True, check=True).stdout
        assert len(pcm) == 2 * 48000
        return
    audio, rate = sf.read(io.BytesIO(body))
    # (a streamed mp3 also holds the encoder's delay and padding: its Xing frame, which tells them, comes at close)
    assert rate == 48000 and abs(len(audio) / rate - 1.0) <= (0.06 if fmt == "mp3" and stream else 0.0)


def test_unsupported_format_and_speed(server, monkeypatch, tmp_path):
    """Without ffmpeg on the PATH: no aac, no speed but 1.0 (400s that say so); OpenAI's speed bounds (422)."""
    monkeypatch.setenv("PATH", str(tmp_path))  # (no ffmpeg)
    out = run(serve(server.build_app(server_args(tmp_path)), [
        {"input": "ok", "response_format": "aac"}, {"input": "ok", "speed": 1.5}, {"input": "ok", "speed": 5.0},
        {"input": "ok", "speed": 0.1}, {"input": "ok", "speed": 1.0}, {"input": "ok", "response_format": "vorbis"}]))
    assert [status for status, _ in out] == [400, 400, 422, 422, 200, 400]
    assert b"pcm, wav, mp3, opus, flac (not 'aac'); it needs ffmpeg" in out[0][1] and b"only 1.0" in out[1][1]


TONE_HZ = 220.0


def tone_engine(monkeypatch, chunks: int = 10, pause: float = 0.005):
    """The fake engine's audio: a 220 Hz tone at 48 kHz, ``chunks`` of 0.1 s, a pause between them (a stream)."""

    async def generate(self, target_text, **kw):
        t = np.arange(4800) / 48000
        for i in range(chunks if target_text.endswith("ok") else 1):  # (the startup warm-up's texts: one chunk)
            if i:
                await asyncio.sleep(pause)
            yield (np.sin(2 * np.pi * TONE_HZ * (t + i * 0.1)) * 8000).astype(np.int16)

    monkeypatch.setattr(FakeEngine, "generate", generate)


def decoded(body: bytes, fmt: str) -> np.ndarray:
    """A response's audio as 48 kHz mono int16 (ffmpeg decodes all but pcm)."""
    import subprocess

    if fmt == "pcm":
        return np.frombuffer(body, np.int16)
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", "-", "-f", "s16le", "-ac", "1", "-ar", "48000", "-"],
                         input=body, capture_output=True, check=True).stdout
    return np.frombuffer(out, np.int16)


def pitch(audio: np.ndarray) -> float:
    spectrum = np.abs(np.fft.rfft(audio.astype(np.float64) * np.hanning(len(audio))))
    return float(np.argmax(spectrum) * 48000 / len(audio))


needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="no ffmpeg on the PATH")


@needs_ffmpeg
@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("speed", [1.0, 0.5, 1.5, 2.5])
@pytest.mark.parametrize("fmt", ["pcm", "wav", "mp3", "opus", "flac", "aac"])
def test_speed_and_aac_through_ffmpeg(server, monkeypatch, tmp_path, fmt, speed, stream):
    """With ffmpeg: aac (ADTS), and a speed other than 1.0 by atempo for every format, streamed too: 1 s of a 220 Hz
    tone comes back 1/speed s long at 220 Hz (the pitch kept)."""
    tone_engine(monkeypatch)
    asgi = server.build_app(server_args(tmp_path))
    sent = []

    async def main():
        async with running(asgi):
            body = json.dumps({"input": "ok", "response_format": fmt, "speed": speed, "stream": stream}).encode()
            return await request(asgi, "POST", "/v1/audio/speech", body,
                                 on_send=lambda m, gone: sent.append(m.get("body")))

    status, headers, body = run(main())
    assert status == 200, body[:300]
    assert headers[b"content-type"] == (b"audio/aac" if fmt == "aac" else server.AUDIO_FORMATS[fmt][0])
    if stream and fmt != "wav":  # sent as it comes
        assert sum(1 for b in sent if b) >= 2
    audio = decoded(body, fmt)
    assert abs(len(audio) / 48000 - 1.0 / speed) < 0.08 + (0.06 if fmt == "mp3" else 0.0)  # (mp3: its padding)
    assert abs(pitch(audio) - TONE_HZ) < 4.0


@needs_ffmpeg
def test_speed_1_takes_no_ffmpeg(server, monkeypatch, tmp_path):
    """speed 1.0 (also given) is today's path, byte for byte: no ffmpeg process."""
    tone_engine(monkeypatch)

    def no_ffmpeg(*a, **kw):
        raise AssertionError("ffmpeg at speed 1.0")

    monkeypatch.setattr(server, "FfmpegStage", no_ffmpeg)
    out = run(serve(server.build_app(server_args(tmp_path)), [
        {"input": "ok", "response_format": "mp3"}, {"input": "ok", "response_format": "mp3", "speed": 1.0},
        {"input": "ok"}, {"input": "ok", "speed": 1.0}]))
    assert [status for status, _ in out] == [200] * 4
    assert out[0][1] == out[1][1] and out[2][1] == out[3][1]


def ffmpeg_children() -> list[int]:
    """This process's ffmpeg children (zombies included)."""
    me, found = os.getpid(), []
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            with open(f"/proc/{pid}/stat") as f:
                fields = f.read().split()
        except OSError:
            continue
        if fields[1] == "(ffmpeg)" and int(fields[3]) == me:
            found.append(int(pid))
    return found


@needs_ffmpeg
@pytest.mark.parametrize("fmt", ["aac", "pcm"])
def test_a_client_gone_mid_stream_ends_ffmpeg(server, monkeypatch, tmp_path, fmt):
    """A client that leaves after the first bytes of a stream through ffmpeg (aac, speed 1.5): its ffmpeg process
    ends, no zombie left."""
    tone_engine(monkeypatch, chunks=200, pause=0.02)  # 20 s of audio, 4 s to generate
    asgi = server.build_app(server_args(tmp_path))
    seen = []

    async def main():
        def on_send(m, gone):
            if m.get("body"):
                seen.extend(ffmpeg_children())
                gone.set()  # the client leaves

        async with running(asgi):
            body = json.dumps({"input": "ok", "response_format": fmt, "speed": 1.5}).encode()
            status, _, _ = await request(asgi, "POST", "/v1/audio/speech", body, on_send=on_send)
            for _ in range(100):
                if not ffmpeg_children():
                    break
                await asyncio.sleep(0.02)
            return status

    assert run(main()) == 200
    assert seen and not ffmpeg_children()


def metric_values(text: str) -> dict[str, float]:
    """Prometheus text: {'name{labels}': value}."""
    return {k: float(v) for k, v in (line.rsplit(" ", 1) for line in text.splitlines() if line and line[0] != "#")}


def test_metrics_and_logs(server, monkeypatch, tmp_path, caplog):
    # every speech request's outcome counted, its audio timed and measured; /metrics behind the API key; a line
    # logged per request refused
    import logging

    caplog.set_level(logging.INFO, logger="openai_server")
    monkeypatch.setattr(nanovllm_voxcpm, "VoxCPM", SlowEngine)
    asgi = server.build_app(server_args(tmp_path, api_key="k", max_streams=1))
    key = [(b"authorization", b"Bearer k")]

    def leave_after_two_chunks(m, gone):
        if m.get("more_body") and m.get("body"):
            leave_after_two_chunks.n += 1
            if leave_after_two_chunks.n == 2:
                gone.set()

    leave_after_two_chunks.n = 0

    async def main():
        async with running(asgi):
            statuses = [(await request(asgi, "POST", "/v1/audio/speech", body, headers=h))[0] for body, h in (
                (speech_body(), key), (speech_body(input="..."), key), (b"{", key), (speech_body(), ()))]
            first = asyncio.ensure_future(request(asgi, "POST", "/v1/audio/speech", speech_body(), headers=key))
            await asyncio.sleep(0.05)
            statuses.append((await request(asgi, "POST", "/v1/audio/speech", speech_body(), headers=key))[0])
            statuses.append((await first)[0])
            await request(asgi, "POST", "/v1/audio/speech", speech_body(), headers=key, on_send=leave_after_two_chunks)
            await asyncio.sleep(0.05)
            return statuses, await request(asgi, "GET", "/metrics"), await request(asgi, "GET", "/metrics",
                                                                                     headers=key)

    statuses, (no_key, _, _), (status, headers, body) = asyncio.run(main())
    assert statuses == [200, 400, 422, 401, 503, 200] and no_key == 401 and status == 200
    assert headers[b"content-type"] == b"text/plain; version=0.0.4; charset=utf-8"
    m = metric_values(body.decode())
    assert {o: m[f'voxcpm_requests_total{{outcome="{o}"}}'] for o in ("ok", "client_error", "server_error", "busy",
                                                                     "cancelled")} == {
        "ok": 2, "client_error": 3, "server_error": 0, "busy": 1, "cancelled": 1}
    assert m["voxcpm_active_streams"] == 0 and m["voxcpm_engine_alive"] == 1
    assert m["voxcpm_first_audio_seconds_count"] == 3  # (the client that left had its first audio)
    assert 0 < m['voxcpm_first_audio_seconds_bucket{le="0.1"}'] and m["voxcpm_first_audio_seconds_sum"] < 1.0
    assert 400 / 48000 < m["voxcpm_audio_seconds_total"] < 500 / 48000  # 2 x 50 chunks of 4 samples, and 2-3 more
    logged = [(r.levelname, r.getMessage()) for r in caplog.records if r.name == "openai_server"]
    assert ("INFO", "speech request of default refused (400): input has nothing to speak (only punctuation, "
                    "symbols or emoji)") in logged
    assert ("WARNING", "speech request of default refused (503): busy: 1 speech requests in progress "
                       "(--max-streams)") in logged
    assert any(level == "INFO" and "(401)" in msg and "/metrics" in msg for level, msg in logged)


def test_audio_cache(server, monkeypatch, tmp_path):
    """--audio-cache-mb: a sentence asked again in the same voice and settings gets the audio it got first (the
    engine's would differ: another load), with no engine request; a seed, another sentence or a voice registered
    again under its name with other audio (other latents): generated. Requests for a sentence in progress share its
    generation. --audio-cache-dir: another server on the directory (a restart, a replica) plays it. Its counts in
    /metrics."""
    pytest.importorskip("multipart")
    import hashlib

    async def generate(self, target_text, **kw):  # other audio on every call, in five chunks 20 ms apart
        self.targets.append(target_text)
        for _ in range(5):
            await asyncio.sleep(0.02)
            yield np.random.randint(-1000, 1000, 960).astype(np.int16)

    async def encode_latents(self, audio, audio_format):  # a voice's latents follow its audio
        return hashlib.sha256(audio).digest()

    monkeypatch.setattr(FakeEngine, "generate", generate)
    monkeypatch.setattr(FakeEngine, "encode_latents", encode_latents)
    cache_dir = str(tmp_path / "audio-cache")

    async def speak(asgi, **kw):
        status, _, body = await request(asgi, "POST", "/v1/audio/speech", json.dumps(kw).encode())
        assert status == 200
        return body

    async def upload(asgi, name, seconds):
        body, ctype = multipart({"name": name}, {"audio_sample": ("clip.wav", wav(seconds))})
        assert (await request(asgi, "POST", "/v1/audio/voices", body, content_type=ctype))[0] == 200

    asgi = server.build_app(server_args(tmp_path, audio_cache_mb=64, audio_cache_dir=cache_dir))

    async def main():
        async with running(asgi):
            out = [await speak(asgi, input="مرحبا بك"), await speak(asgi, input="مرحبا بك", response_format="wav"),
                   await speak(asgi, input="مرحبا بك", seed=7), await speak(asgi, input="أهلا")]
            await upload(asgi, "v", 1.0)
            out += [await speak(asgi, input="مرحبا بك", voice="v"), await speak(asgi, input="مرحبا بك", voice="v")]
            assert (await request(asgi, "DELETE", "/v1/audio/voices/v"))[0] == 200
            await upload(asgi, "v", 1.2)  # the same name, another voice
            out.append(await speak(asgi, input="مرحبا بك", voice="v"))
            out += await asyncio.gather(*(speak(asgi, input="كيف حالك") for _ in range(3)))  # at the same time
            metrics = (await request(asgi, "GET", "/metrics"))[2].decode()
            await asgi.state["cache"].idle()
        return out, metrics

    out, metrics = run(main())
    assert out[1][44:] == out[0] and out[2] != out[0]  # the wav: the same audio behind its header
    assert out[5] == out[4] and out[6] != out[4]
    assert out[7] == out[8] == out[9]
    asked = [t for t in FakeEngine.last.targets if t in ("مرحبا بك", "أهلا", "كيف حالك")]
    assert len(asked) == 6  # 10 requests: 2 from the cache, 2 joined a generation in progress
    values = metric_values(metrics)
    assert values["voxcpm_audio_cache_hits_total"] == 2 and values["voxcpm_audio_cache_misses_total"] == 6
    assert values["voxcpm_audio_cache_joins_total"] == 2

    again = server.build_app(server_args(tmp_path, audio_cache_mb=64, audio_cache_dir=cache_dir))  # a restart

    async def restarted():
        async with running(again):
            return [await speak(again, input="مرحبا بك"), await speak(again, input="كيف حالك")]

    played = run(restarted())
    assert played == [out[0], out[7]] and len(FakeEngine.last.targets) == len(
        [t for t in FakeEngine.last.targets if t not in ("مرحبا بك", "كيف حالك")])  # (no engine request for them)


def replica_engines(monkeypatch, base=FakeEngine) -> list:
    """Each app (replica) its own engine (SlowEngine's requests take 0.5 s)."""
    engines = []

    class Engine(base):
        @classmethod
        def from_pretrained(cls, **kw):
            engines.append(cls())
            return engines[-1]

    monkeypatch.setattr(nanovllm_voxcpm, "VoxCPM", Engine)
    return engines


def test_replicas_share_their_voices_through_the_voice_dir(server, monkeypatch, tmp_path):
    """Replicas on one --voice-dir: a voice registered on A is spoken on B at once (B registers it on the first
    request that names it); deleted on A, it goes from B at B's next sync, not while B speaks with it; a voice left
    half written, or a file that is no voice, is no voice."""
    pytest.importorskip("multipart")
    engines = replica_engines(monkeypatch, SlowEngine)
    voice_dir = tmp_path / "voices"
    a, b = (server.build_app(server_args(tmp_path, voice_dir=str(voice_dir), voice_sync_s=0)) for _ in range(2))

    async def upload(app, name):
        body, ctype = multipart({"name": name}, {"audio_sample": ("clip.wav", wav(1.0))})
        return (await request(app, "POST", "/v1/audio/voices", body, content_type=ctype))[0]

    async def speak(app, voice):
        return (await request(app, "POST", "/v1/audio/speech", speech_body(voice=voice)))[0]

    async def listed(app):
        return sorted(v["name"] for v in json.loads((await request(app, "GET", "/v1/audio/voices"))[2])["voices"])

    async def main():
        out = {}
        async with running(a), running(b):
            out["A registers v"] = await upload(a, "v")
            out["B lists"] = await listed(b)
            out["B speaks v"] = await speak(b, "v")
            out["v in B's engine"] = "v" in engines[1].voices
            out["B registers v too"] = await upload(b, "v")
            speaking = asyncio.ensure_future(speak(b, "v"))
            await asyncio.sleep(0.05)
            out["A deletes v"] = (await request(a, "DELETE", "/v1/audio/voices/v"))[0]
            out["B lists meanwhile"] = await listed(b)
            await b.state["sync_voices"]()
            out["v on B while it speaks"] = "v" in b.state["voices"]
            out["B's request"] = await speaking
            await b.state["sync_voices"]()
            out["v on B after"] = "v" in b.state["voices"] or "v" in engines[1].voices
            out["B speaks v after"] = await speak(b, "v")
            (voice_dir / "half.json").write_text(json.dumps({"name": "half", "text": None}), encoding="utf-8")
            (voice_dir / "bad.json").write_bytes(b"{no json")
            (voice_dir / "bad.latents").write_bytes(b"latents")
            (voice_dir / ".w.json.123.tmp").write_bytes(b"{")
            await b.state["sync_voices"]()
            out["B speaks half"] = await speak(b, "half")
            out["B's voices"] = sorted(b.state["voices"])
            out["B registers w"] = await upload(b, "w")
            await a.state["sync_voices"]()
            out["w on A"] = "w" in a.state["voices"] and "w" in engines[0].voices
            out["A deletes w (B's)"] = (await request(a, "DELETE", "/v1/audio/voices/w"))[0]
            await b.state["sync_voices"]()
            out["w on B after"] = "w" in b.state["voices"]
        return out

    assert run(main()) == {
        "A registers v": 200, "B lists": ["v"], "B speaks v": 200, "v in B's engine": True, "B registers v too": 409,
        "A deletes v": 200, "B lists meanwhile": [], "v on B while it speaks": True, "B's request": 200,
        "v on B after": False, "B speaks v after": 404, "B speaks half": 404, "B's voices": [], "B registers w": 200,
        "w on A": True, "A deletes w (B's)": 200, "w on B after": False}
    assert sorted(p.name for p in voice_dir.iterdir()) == [".w.json.123.tmp", "bad.json", "bad.latents", "half.json"]


def test_a_replica_follows_the_voice_dir_by_itself(server, monkeypatch, tmp_path):
    """--voice-sync-s: a voice another replica registered comes without a request naming it."""
    pytest.importorskip("multipart")
    engines = replica_engines(monkeypatch)
    voice_dir = tmp_path / "voices"
    a = server.build_app(server_args(tmp_path, voice_dir=str(voice_dir), voice_sync_s=0))
    b = server.build_app(server_args(tmp_path, voice_dir=str(voice_dir), voice_sync_s=0.05))

    async def main():
        async with running(a), running(b):
            body, ctype = multipart({"name": "v"}, {"audio_sample": ("clip.wav", wav(1.0))})
            assert (await request(a, "POST", "/v1/audio/voices", body, content_type=ctype))[0] == 200
            for _ in range(100):
                if "v" in b.state["voices"]:
                    break
                await asyncio.sleep(0.02)
            return "v" in b.state["voices"] and "v" in engines[1].voices

    assert run(main())


def test_client_keys_and_limits(server, monkeypatch, tmp_path):
    """--api-keys-file: a key per client, each with its own limits (429 with Retry-After beyond them), /metrics only
    for a client with metrics=true, counts by client; the file read again when it changes."""
    from nanovllm_voxcpm.entrypoints.clients import Clients

    monkeypatch.setattr(nanovllm_voxcpm, "VoxCPM", SlowEngine)
    monkeypatch.setattr(Clients, "RECHECK_S", 0.0)
    keys = tmp_path / "keys.txt"
    keys.write_text("a ka max_streams=1\nb kb rpm=2\nm km metrics=true\n")
    asgi = server.build_app(server_args(tmp_path, api_keys_file=str(keys)))

    def key(k):
        return [(b"authorization", b"Bearer " + k)]

    async def main():
        async with running(asgi):
            speak = lambda k: request(asgi, "POST", "/v1/audio/speech", speech_body(), headers=key(k))  # noqa: E731
            first = asyncio.ensure_future(speak(b"ka"))
            await asyncio.sleep(0.05)
            out = {"a_second": await speak(b"ka"), "b": [await speak(b"kb") for _ in range(3)],
                   "a_first": await first, "unknown": await speak(b"kx")}
            out["metrics_a"] = await request(asgi, "GET", "/metrics", headers=key(b"ka"))
            out["metrics_m"] = await request(asgi, "GET", "/metrics", headers=key(b"km"))
            keys.write_text("c kc\nm km metrics=true\n")  # a and b gone, c new
            out["after"] = [(await speak(k))[0] for k in (b"kc", b"ka")]
        return out

    out = run(main())
    status, headers, body = out["a_second"]
    assert out["a_first"][0] == 200 and status == 429 and headers[b"retry-after"] == b"1" and b"max_streams" in body
    assert [s for s, _, _ in out["b"]] == [200, 200, 429]
    assert 28 <= int(out["b"][2][1][b"retry-after"]) <= 30  # (rpm 2: a token every 30 s, the requests took ~1 s)
    assert out["unknown"][0] == 401 and out["metrics_a"][0] == 403 and out["metrics_m"][0] == 200
    values = metric_values(out["metrics_m"][2].decode())
    assert values['voxcpm_client_requests_total{client="a",outcome="limited"}'] == 1
    assert values['voxcpm_client_requests_total{client="b",outcome="ok"}'] == 2
    assert values['voxcpm_requests_total{outcome="limited"}'] == 2 and b"kb" not in out["metrics_m"][2]
    assert out["after"] == [200, 401]
