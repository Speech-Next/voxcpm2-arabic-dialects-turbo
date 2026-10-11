"""OpenAI-compatible streaming TTS server for VoxCPM2, built around voice cloning.

    python -m nanovllm_voxcpm.entrypoints.openai_server --model <model dir or repo id> \
        --dit <distilled LocDiT, path or repo:file> --voice-dir /path/to/voices --port 8092

(or NANOVLLM_VOXCPM2_TURBO='{...}' for a configuration of its own). A model with an ``adapters/`` directory serves its
adapters (see LoRA below).

Endpoints (request shapes follow vLLM-Omni's, so one client can drive both servers):

    POST   /v1/audio/voices         multipart: audio_sample (file), name, ref_text (optional), consent (ignored)
    GET    /v1/audio/voices
    DELETE /v1/audio/voices/{name}
    POST   /v1/audio/speech         JSON: input, voice, stream, response_format ("pcm" | "wav"), clone_mode,
                                    seed, cfg_value, max_tokens, split
    GET    /v1/models, /health

A voice is encoded to latents once, when it is registered (or preloaded with --voices), and kept inside the engine
process; a request only names it, so reference audio is never re-encoded or re-sent and the engine's per-voice
caches (reference-prefix KV, LocEnc embeddings, primed VAE state) apply. clone_mode: "prompt_ref" (reference prefix + transcript continuation, the default when
the voice has a transcript), "ref" (reference prefix only, the default otherwise) or "prompt" (continuation only).
Audio is 16-bit little-endian mono PCM at the model's output rate (header X-Sample-Rate), streamed per chunk.

Long input is split into sentences (``split_text``) that are generated one after another into the same stream: the
first audio comes from the first sentence, and no request is longer than a sentence. ``--voice-dir`` keeps
registered voices on disk (latents, transcript, original audio) and registers them again at startup without
re-encoding.

LoRA adapters (e.g. one per dialect of a fine-tuned model): ``--lora-dir DIR`` registers every subdirectory with a
``lora_weights.safetensors`` (VoxCPM's LoRA format; ``--lora NAME=PATH`` adds one), all resident at once. A request
picks one with ``"lora"`` (alias ``"dialect"``) or by naming it as ``"model"`` (vLLM's convention; ``/v1/models``
lists them); ``--default-lora`` serves requests that name none. A burst may mix adapters: every row of a step uses
its own request's adapter. The voices' reference-prefix KV cache is kept per adapter (the adapter changes the KV).
"""

import argparse
import asyncio
import io
import json
import os
import re
import time
import wave
from contextlib import asynccontextmanager

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, ValidationError

from nanovllm_voxcpm.dialect_tts import DEFAULT_MODEL, adapters_lora_config, find_adapters, resolve, turbo_config

CLONE_MODES = ("prompt_ref", "ref", "prompt")
VOICE_NAME = re.compile(r"^[A-Za-z0-9._-]{1,64}$")  # voice names are also file names in --voice-dir

# sentence ends: Latin / Arabic-script punctuation followed by whitespace (Arabic ؟ ؛ and the Urdu full stop ۔),
# CJK punctuation anywhere; clause ends: commas (Arabic ، included)
_SENTENCE_END = re.compile(r"(?<=[.!?…;:؟؛۔])\s+|(?<=[。！？；])")
_CLAUSE_END = re.compile(r"(?<=[,，、،])\s*")


def split_text(text: str, max_chars: int = 200, min_chars: int = 40) -> list[str]:
    """Split text into sentence-sized pieces: sentences shorter than ``min_chars`` are merged with the next one,
    longer than ``max_chars`` are broken at commas, then at spaces."""
    text = " ".join(text.split())
    pieces: list[str] = []
    for sentence in (p for p in _SENTENCE_END.split(text) if p):
        while len(sentence) > max_chars:
            cut = max((m.end() for m in _CLAUSE_END.finditer(sentence, 0, max_chars) if m.end() < max_chars),
                      default=0)
            if cut < min_chars:
                cut = sentence.rfind(" ", min_chars, max_chars) + 1 or max_chars
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            pieces.append(sentence)
    out: list[str] = []
    for p in pieces:
        if out and len(out[-1]) < min_chars and len(out[-1]) + len(p) < max_chars:
            out[-1] = _join(out[-1], p)
        else:
            out.append(p)
    if len(out) > 1 and len(out[-1]) < min_chars // 2 and len(out[-2]) + len(out[-1]) < max_chars:
        tail = out.pop()  # a dangling fragment joins the previous piece
        out[-1] = _join(out[-1], tail)
    return out


def _join(a: str, b: str) -> str:
    return a + b if a[-1] in "。！？；，、" else f"{a} {b}"  # no space after CJK punctuation


def save_voice(voice_dir: str, name: str, latents: bytes, text: str | None, audio: bytes, audio_format: str) -> None:
    os.makedirs(voice_dir, exist_ok=True)
    with open(os.path.join(voice_dir, f"{name}.latents"), "wb") as f:
        f.write(latents)
    with open(os.path.join(voice_dir, f"{name}.{audio_format}"), "wb") as f:
        f.write(audio)
    meta = {"name": name, "text": text, "audio": f"{name}.{audio_format}", "created_at": int(time.time())}
    with open(os.path.join(voice_dir, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)


def load_voices(voice_dir: str) -> list[tuple[dict, bytes]]:
    out = []
    if not os.path.isdir(voice_dir):
        return out
    for fn in sorted(os.listdir(voice_dir)):
        if fn.endswith(".json"):
            meta = json.load(open(os.path.join(voice_dir, fn), encoding="utf-8"))
            with open(os.path.join(voice_dir, f"{meta['name']}.latents"), "rb") as f:
                out.append((meta, f.read()))
    return out


def delete_voice_files(voice_dir: str, name: str) -> None:
    for fn in os.listdir(voice_dir):
        if fn.rsplit(".", 1)[0] == name:
            os.remove(os.path.join(voice_dir, fn))


class SpeechRequest(BaseModel):
    input: str
    voice: str | None = None
    model: str | None = None
    stream: bool = True
    stream_format: str | None = None  # accepted for OpenAI / vLLM-Omni compatibility
    response_format: str = "pcm"
    clone_mode: str | None = None
    seed: int | None = None
    cfg_value: float = 2.0
    max_tokens: int | None = None  # latent patches to generate at most per sentence (160 ms of audio each)
    split: bool = True  # long input: sentence by sentence into one stream
    lora: str | None = None  # a registered LoRA adapter (--lora-dir / --lora)
    dialect: str | None = None  # alias of lora


def pcm16(chunk: np.ndarray) -> bytes:
    if chunk.dtype == np.int16:  # engine started with pcm16_output (the default here)
        return chunk.tobytes()
    return (np.clip(chunk, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def build_app(args: argparse.Namespace):
    state: dict = {"voices": {}}

    async def warm_up(names: list[str]) -> None:
        """One short generation per voice and clone mode: fills the voice's caches (prefix KV, LocEnc embeddings,
        primed VAE state) so that its first real request is as fast as the following ones."""
        async def run(name, mode):
            async for _ in state["pool"].generate(target_text="Warm up.", voice=name, clone_mode=mode,
                                                  max_generate_length=2, seed=0):
                pass

        jobs = []
        for name in names:
            modes = CLONE_MODES if state["voices"][name]["text"] else ("ref",)
            jobs += [run(name, m) for m in modes]
        await asyncio.gather(*jobs)

    async def register(name: str, audio: bytes, audio_format: str, text: str | None) -> dict:
        # encoded once and kept inside the engine: a request then sends only the voice name
        latents = await state["pool"].encode_latents(audio, audio_format)
        await state["pool"].register_voice(name, latents, text or None)
        if args.voice_dir:
            save_voice(args.voice_dir, name, latents, text or None, audio, audio_format)
        state["voices"][name] = {"text": text or None, "created_at": int(time.time())}
        await warm_up([name])
        return {"name": name, "has_transcript": bool(text), "created_at": state["voices"][name]["created_at"]}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        from nanovllm_voxcpm import VoxCPM

        if args.kv_blocks:
            os.environ["NANOVLLM_SERVERPOOL_NUM_KVCACHE_BLOCKS"] = str(args.kv_blocks)
        turbo = json.loads(os.environ.get("NANOVLLM_VOXCPM2_TURBO") or "{}")
        if (args.dit or args.dit_cache) and not turbo:  # the low-latency configuration
            turbo = turbo_config(args.dit, args.head_tokens, args.dit_cache)
        turbo.setdefault("pcm16_output", True)  # int16 from the GPU: half the bytes per chunk, no conversion here
        os.environ["NANOVLLM_VOXCPM2_TURBO"] = json.dumps(turbo)
        model_path = resolve(args.model)
        lora_dir = args.lora_dir
        if lora_dir is None and os.path.isdir(os.path.join(model_path, "adapters")):  # e.g. one adapter per dialect
            lora_dir = os.path.join(model_path, "adapters")
        adapters = find_adapters(lora_dir, args.lora)
        if args.default_lora and args.default_lora not in adapters:
            raise SystemExit(f"--default-lora {args.default_lora!r} is not among the adapters {sorted(adapters)}")
        pool = VoxCPM.from_pretrained(
            model=model_path,
            devices=[0],
            inference_timesteps=args.inference_timesteps,
            max_num_seqs=args.max_num_seqs,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_num_batched_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            lora_config=adapters_lora_config(adapters, args.max_lora_rank),
        )
        await pool.wait_for_ready()
        for name, path in adapters.items():
            await pool.register_lora(name, path)
        state["adapters"] = sorted(adapters)
        state["pool"] = pool
        state["sample_rate"] = int((await pool.get_model_info())["sample_rate"])
        if args.voice_dir:  # voices registered in earlier runs: latents are on disk, no re-encoding
            for meta, latents in load_voices(args.voice_dir):
                await pool.register_voice(meta["name"], latents, meta.get("text"))
                state["voices"][meta["name"]] = {"text": meta.get("text"), "created_at": meta.get("created_at")}
            await warm_up(list(state["voices"]))
        if args.voices:
            for v in json.load(open(args.voices, encoding="utf-8")):
                if v["id"] not in state["voices"]:
                    with open(v["wav"], "rb") as f:
                        await register(v["id"], f.read(), os.path.splitext(v["wav"])[1].lstrip(".") or "wav",
                                       v.get("text"))
        print(f"[server] ready: {len(state['voices'])} voices, {len(state['adapters'])} LoRA adapters, "
              f"sample rate {state['sample_rate']}", flush=True)
        import gc

        gc.collect()  # startup objects frozen: no full collection over them while requests stream
        gc.freeze()
        prof = None
        if os.environ.get("NANOVLLM_HTTP_PROFILE"):  # cProfile of the event-loop thread, written at shutdown
            import cProfile

            prof = cProfile.Profile()
            prof.enable()
        try:
            yield
        finally:
            if prof is not None:
                prof.disable()
                prof.dump_stats(os.environ["NANOVLLM_HTTP_PROFILE"])
            await pool.stop()

    app = FastAPI(title="VoxCPM2 (nanovllm-voxcpm-turbo)", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        base = [{"id": args.served_model_name, "object": "model", "owned_by": "local"}]
        loras = [{"id": name, "object": "model", "owned_by": "local", "parent": args.served_model_name}
                 for name in state.get("adapters", [])]
        return {"object": "list", "data": base + loras}

    @app.post("/v1/audio/voices")
    async def upload_voice(
        name: str = Form(...),
        audio_sample: UploadFile = File(...),
        ref_text: str | None = Form(None),
        consent: str | None = Form(None),
    ):
        if not VOICE_NAME.match(name):
            raise HTTPException(400, "voice name: 1-64 characters from A-Z a-z 0-9 . _ -")
        if name in state["voices"]:
            raise HTTPException(409, f"voice '{name}' exists; DELETE /v1/audio/voices/{name} first")
        audio_format = os.path.splitext(audio_sample.filename or "")[1].lstrip(".") or "wav"
        return await register(name, await audio_sample.read(), audio_format, ref_text)

    @app.get("/v1/audio/voices")
    async def list_voices():
        return {"voices": [{"name": k, "has_transcript": bool(v["text"])} for k, v in state["voices"].items()]}

    @app.delete("/v1/audio/voices/{name}")
    async def delete_voice(name: str):
        if state["voices"].pop(name, None) is None:
            raise HTTPException(404, f"voice '{name}' not found")
        await state["pool"].unregister_voice(name)
        if args.voice_dir:
            delete_voice_files(args.voice_dir, name)
        return {"deleted": name}

    def voice_kwargs(req: SpeechRequest) -> dict:
        if req.voice is None:
            if req.clone_mode is not None:
                raise HTTPException(400, "clone_mode needs a voice")
            return {}
        v = state["voices"].get(req.voice)
        if v is None:
            raise HTTPException(404, f"voice '{req.voice}' not found")
        mode = req.clone_mode or ("prompt_ref" if v["text"] else "ref")
        if mode not in CLONE_MODES:
            raise HTTPException(400, f"clone_mode must be one of {CLONE_MODES}")
        if mode != "ref" and not v["text"]:
            raise HTTPException(400, f"clone_mode '{mode}' needs a voice registered with ref_text")
        return {"voice": req.voice, "clone_mode": mode}

    def lora_kwargs(req: SpeechRequest) -> dict:
        if req.lora and req.dialect and req.lora != req.dialect:
            raise HTTPException(400, "lora and dialect name different adapters")
        name = req.lora or req.dialect
        if name is None and req.model in state["adapters"]:
            name = req.model  # vLLM's convention: an adapter is selected as the model
        name = name or args.default_lora
        if name is None:
            return {}
        if name not in state["adapters"]:
            raise HTTPException(404, f"LoRA adapter '{name}' not found; registered: {state['adapters']}")
        return {"lora_name": name}

    trace = open(os.environ["NANOVLLM_HTTP_TRACE"], "a") if os.environ.get("NANOVLLM_HTTP_TRACE") else None

    async def send_json(send, status: int, obj: dict):
        body = json.dumps(obj).encode()
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})

    async def synthesize(req: SpeechRequest, vk: dict):
        """Audio chunks of the whole input: one engine request per sentence, in order."""
        parts = split_text(req.input, args.split_max_chars) if req.split else [req.input]
        for k, part in enumerate(parts):
            gen = state["pool"].generate(
                target_text=part,
                cfg_value=req.cfg_value,
                seed=None if req.seed is None else req.seed + k,
                max_generate_length=req.max_tokens or args.max_tokens,
                **vk,
            )
            try:
                async for chunk in gen:
                    yield chunk
            finally:
                await gen.aclose()

    async def speech(scope, receive, send):
        """POST /v1/audio/speech as a bare ASGI handler: with hundreds of simultaneous requests, FastAPI's
        per-request work (dependency resolution, body handling) on the event loop that also drives the engine
        delays when later requests reach the engine; this path reads the body, validates it with pydantic's
        JSON parser and streams the PCM chunks itself. A failed send (client gone) cancels the request."""
        t_arrive = time.perf_counter()
        body = b""
        while True:
            msg = await receive()
            body += msg.get("body", b"")
            if not msg.get("more_body"):
                break
        try:
            req = SpeechRequest.model_validate_json(body)
            if req.response_format not in ("pcm", "wav"):
                raise HTTPException(400, "response_format must be 'pcm' or 'wav'")
            if not req.input.strip():
                raise HTTPException(400, "input is empty")
            gen = synthesize(req, {**voice_kwargs(req), **lora_kwargs(req)})
            try:
                first = await gen.__anext__()  # errors (e.g. a prompt too long) become HTTP errors
            except StopAsyncIteration:
                first = None
            except ValueError as e:
                raise HTTPException(400, str(e)) from e
        except ValidationError as e:
            return await send_json(send, 422, {"detail": e.errors(include_url=False)})
        except HTTPException as e:
            return await send_json(send, e.status_code, {"detail": e.detail})
        if trace is not None:  # NANOVLLM_HTTP_TRACE=<file>: perf_counter (CLOCK_MONOTONIC) per request
            trace.write(json.dumps({"arrive": t_arrive, "first": time.perf_counter()}) + "\n")
            trace.flush()
        sr = str(state["sample_rate"]).encode()
        if req.stream and req.response_format == "pcm":
            await send({"type": "http.response.start", "status": 200,
                        "headers": [(b"content-type", b"audio/pcm"), (b"x-sample-rate", sr)]})
            try:
                if first is not None:
                    await send({"type": "http.response.body", "body": pcm16(first), "more_body": True})
                async for chunk in gen:
                    await send({"type": "http.response.body", "body": pcm16(chunk), "more_body": True})
                await send({"type": "http.response.body", "body": b"", "more_body": False})
            finally:
                await gen.aclose()
            return
        chunks = [] if first is None else [pcm16(first)]
        async for chunk in gen:
            chunks.append(pcm16(chunk))
        pcm = b"".join(chunks)
        out, ctype = (wav_bytes(pcm, int(sr)), b"audio/wav") if req.response_format == "wav" else (pcm, b"audio/pcm")
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", ctype), (b"content-length", str(len(out)).encode()),
                                (b"x-sample-rate", sr)]})
        await send({"type": "http.response.body", "body": out})

    async def asgi(scope, receive, send):
        if scope["type"] == "http" and scope["method"] == "POST" and scope["path"] == "/v1/audio/speech":
            return await speech(scope, receive, send)
        return await app(scope, receive, send)

    return asgi


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=DEFAULT_MODEL, help="VoxCPM2 model directory or Hugging Face repo id")
    ap.add_argument("--dit", help="distilled LocDiT checkpoint (path or repo:file): the low-latency configuration, "
                                  "unless NANOVLLM_VOXCPM2_TURBO is set")
    ap.add_argument("--head-tokens", type=int, default=8, help="with --dit: head start length in tokens (0 = off)")
    ap.add_argument("--dit-cache", choices=["light", "medium"],
                    help="instead of --dit: the original LocDiT, its middle layers reused on 2 (light) or 4 (medium) "
                         "of its 9 steps")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--voices", help="JSON list of {id, wav, text?} registered at startup (if not in --voice-dir)")
    ap.add_argument("--voice-dir", help="keep registered voices here and register them again at startup")
    ap.add_argument("--split-max-chars", type=int, default=200, help="longest piece of input generated as one request")
    ap.add_argument("--served-model-name", default="voxcpm2")
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--max-num-batched-tokens", type=int, default=2048)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--kv-blocks", type=int, default=600, help="KV-cache blocks (0 = automatic sizing)")
    ap.add_argument("--inference-timesteps", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=600, help="default cap on generated patches per sentence")
    ap.add_argument("--lora-dir", help="register every subdirectory with a lora_weights.safetensors as an adapter "
                                       "(default: the model's adapters/)")
    ap.add_argument("--lora", action="append", default=[], metavar="NAME=PATH", help="register one more adapter")
    ap.add_argument("--default-lora", help="adapter for requests that name none (default: the base model)")
    ap.add_argument("--max-lora-rank", type=int, help="LoRA slot rank (default: the largest adapter rank)")
    args = ap.parse_args()
    os.environ.setdefault("NANOVLLM_SUBMIT_FLUSH_EVERY", "4")  # requests reach the engine as a steady stream
    uvicorn.run(build_app(args), host=args.host, port=args.port, loop="uvloop", http="httptools",
                backlog=4096, timeout_keep_alive=30, log_level="warning")


if __name__ == "__main__":
    main()
