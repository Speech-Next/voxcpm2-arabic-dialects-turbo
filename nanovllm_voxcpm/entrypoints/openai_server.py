"""OpenAI-compatible streaming TTS server for VoxCPM2, built around voice cloning.

    python -m nanovllm_voxcpm.entrypoints.openai_server [--voice-dir /path/to/voices] [--port 8092]

The defaults are the library's: the dialect model, routed, with the distilled 3-step LocDiT and int8 layers
(--dit none: the model's own LocDiT, 10 steps). A model with an ``adapters/`` directory serves its
adapters (see LoRA below).

Endpoints (request shapes follow vLLM-Omni's, so one client can drive both servers):

    POST   /v1/audio/voices         multipart: audio_sample (file), name, ref_text (optional), consent (ignored),
                                    normalize (default true)
    GET    /v1/audio/voices
    DELETE /v1/audio/voices/{name}
    POST   /v1/audio/speech         JSON: input, voice, stream, response_format ("pcm" | "wav"), clone_mode,
                                    seed, cfg_value, max_tokens, split, normalize (default true)
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

A model with a ``dialects.json`` (one model for all dialects, see DialectTags) has no adapters: ``"dialect"`` names a
dialect (``/v1/models`` lists them), whose tag token is put in front of the text.

Dialect routing (the default for a model with ``adapters/``, ``--no-route`` turns it off; see dialect_routing): the
engine serves the adapters the routes use and ``--tagged-model``'s LoRA (teacher-v3) as one more adapter; a request's
``"dialect"`` (or ``"lora"``, or a dialect named as ``"model"``) picks its adapter, or teacher-v3 with the dialect's
tag. ``/v1/models`` lists the dialects.

Arabic input and ref_text are read as the model was trained on them (dialect_tts.read_as: digits, dates, times, amounts
as words; formal when the request's dialect or adapter, aliases resolved, is msa); hindi_english and text without
Arabic script stay as they are. ``"normalize": false`` sends the text as written.

In production: ``--api-key`` (or VOXCPM_API_KEY) makes every endpoint but /health ask for ``Authorization: Bearer
<key>``; ``--max-streams`` answers 503 (Retry-After) beyond that many speech requests at once (one A10: 150);
``--max-input-chars`` and ``--max-voice-mb`` / ``--max-voice-seconds`` bound a request's input and a voice's clip. A
speech request whose client disconnects ends its engine request. /health is 503 once the engine process has died, and
the server then exits with code 1, for its supervisor to start it again.
"""

import argparse
import asyncio
import contextlib
import hmac
import io
import json
import os
import re
import signal
import sys
import time
import wave
from contextlib import asynccontextmanager

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from nanovllm_voxcpm import warmup
from nanovllm_voxcpm.dialect_routing import (DEFAULT_ROUTES, DEFAULT_TAGGED_MODEL, TAGGED, TAGGED_ADAPTER,
                                             resolve_tagged, route_table, routed_lora_config, used_adapters)
from nanovllm_voxcpm.dialect_tts import (DEFAULT_DIT, DEFAULT_MODEL, DialectTags, adapters_lora_config, find_adapters,
                                         read_as, resolve, turbo_config, use_model_calibration)

CLONE_MODES = ("prompt_ref", "ref", "prompt")
VOICE_NAME = re.compile(r"^[A-Za-z0-9._-]{1,64}$")  # voice names are also file names in --voice-dir
MAX_JSON_BODY = 1 << 20  # a speech request's body, bytes

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


def save_voice(voice_dir: str, name: str, latents: bytes, text: str | None, audio: bytes, audio_format: str,
               normalize: bool = True) -> None:
    os.makedirs(voice_dir, exist_ok=True)
    with open(os.path.join(voice_dir, f"{name}.latents"), "wb") as f:
        f.write(latents)
    with open(os.path.join(voice_dir, f"{name}.{audio_format}"), "wb") as f:
        f.write(audio)
    meta = {"name": name, "text": text, "normalize": normalize, "audio": f"{name}.{audio_format}",
            "created_at": int(time.time())}  # text: the transcript as written
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
    cfg_value: float = Field(2.0, ge=0.0, le=10.0)  # (NaN and infinity fail the bounds)
    max_tokens: int | None = Field(None, ge=1)  # latent patches to generate at most per sentence (160 ms of audio each)
    split: bool = True  # long input: sentence by sentence into one stream
    lora: str | None = None  # a registered LoRA adapter (--lora-dir / --lora)
    dialect: str | None = None  # alias of lora; a tagged model's dialect
    normalize: bool = True  # Arabic input read as the model was trained on it (dialect_tts.read_as)


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
    # active: speech requests in progress (--max-streams); registering: voice names being uploaded; dead: the engine
    # process died; on_engine_dead: what stops the server then (main: uvicorn's exit)
    state: dict = {"voices": {}, "tags": None, "route": None, "active": 0, "registering": set(), "dead": False,
                   "on_engine_dead": lambda: os.kill(os.getpid(), signal.SIGTERM)}

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

    async def warm_up_kernels() -> None:
        """nanovllm_voxcpm.warmup's bursts over the dialects, with a throwaway voice (5 s of noise): Triton compiles
        its kernel variants now, not during the first clients' requests (#39)."""
        name = "_warmup"
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(np.random.default_rng(0).normal(0.0, 1600.0, 5 * 16000).astype("<i2").tobytes())
        latents = await state["pool"].encode_latents(buf.getvalue(), "wav")
        await register_voice(name, latents, None, True)
        state["voices"][name] = voice_entry(latents, None, 0)
        dialects = (sorted(state["route"]) if state["route"] is not None else list(state["tags"].names)
                    if state["tags"] is not None else state["adapters"] or [None])

        async def run(i: int, text: str) -> None:
            async for _ in request_audio(SpeechRequest(input=text, voice=name, dialect=dialects[i % len(dialects)])):
                pass

        try:
            await warmup.warm_up(run)
        finally:
            del state["voices"][name]
            await state["pool"].unregister_voice(name)
            if state["tags"] is not None:
                await state["tags"].drop_voice(state["pool"], name)

    def uploaded_clip(audio: bytes) -> bytes:
        """An uploaded reference clip decoded and resampled here (in a worker thread: in the engine process this
        held every stream's next step for as long), at most --max-voice-seconds long (its header checked before it
        is decoded): float32 WAV at the encoder's rate, which the engine reads as it is."""
        import librosa
        import soundfile as sf

        sr = state["encoder_rate"]
        try:
            seconds = sf.info(io.BytesIO(audio)).duration
            y, _ = librosa.load(io.BytesIO(audio), sr=sr, mono=True) if seconds <= args.max_voice_seconds else (None, 0)
        except Exception as e:  # noqa: BLE001 - whatever the decoder says: not a clip it reads
            raise HTTPException(400, f"audio_sample: not an audio file the server reads ({type(e).__name__})") from e
        if y is None:
            raise HTTPException(400, f"audio_sample: {seconds:.1f} s, at most {args.max_voice_seconds:g} s "
                                     "(--max-voice-seconds)")
        if not y.size or not np.isfinite(y).all():
            raise HTTPException(400, "audio_sample: no audio")
        buf = io.BytesIO()
        sf.write(buf, y, sr, format="WAV", subtype="FLOAT")
        return buf.getvalue()

    async def register(name: str, audio: bytes, audio_format: str, text: str | None, normalize: bool = True,
                       upload: bool = False) -> dict:
        # encoded once and kept inside the engine: a request then sends only the voice name
        clip, fmt = (await asyncio.to_thread(uploaded_clip, audio), "wav") if upload else (audio, audio_format)
        latents = await state["pool"].encode_latents(clip, fmt)
        await register_voice(name, latents, text or None, normalize)
        if args.voice_dir:
            save_voice(args.voice_dir, name, latents, text or None, audio, audio_format, normalize)
        state["voices"][name] = voice_entry(latents, text or None, int(time.time()), normalize)
        await warm_up([name])
        return {"name": name, "has_transcript": bool(text), "created_at": state["voices"][name]["created_at"]}

    async def register_voice(name: str, latents: bytes, text: str | None, normalize: bool) -> None:
        """The engine gets the transcript as target text is read (no dialect: colloquial)."""
        await state["pool"].register_voice(name, latents, read_as(text) if text and normalize else text)

    def voice_entry(latents: bytes, text: str | None, created_at, normalize: bool = True) -> dict:
        v = {"text": text, "created_at": created_at, "normalize": normalize}
        if state["tags"] is not None and text:
            v["latents"] = latents  # its per-dialect copies are registered from these
        return v

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        from nanovllm_voxcpm import VoxCPM

        if args.kv_blocks:
            os.environ["NANOVLLM_SERVERPOOL_NUM_KVCACHE_BLOCKS"] = str(args.kv_blocks)
        turbo = json.loads(os.environ.get("NANOVLLM_VOXCPM2_TURBO") or "{}")
        if not turbo:
            turbo = turbo_config(None if args.dit in ("", "none") or args.dit_cache else args.dit, args.head_tokens,
                                 args.dit_cache)
        turbo.setdefault("pcm16_output", True)  # int16 from the GPU: half the bytes per chunk, no conversion here
        os.environ["NANOVLLM_VOXCPM2_TURBO"] = json.dumps(turbo)
        model_path = resolve(args.model)
        if turbo.get("w8a8"):
            use_model_calibration(model_path)
        state["tags"] = DialectTags.of_model(model_path)
        lora_dir = args.lora_dir
        if lora_dir is None and state["tags"] is None and os.path.isdir(os.path.join(model_path, "adapters")):
            lora_dir = os.path.join(model_path, "adapters")
        adapters = find_adapters(lora_dir, args.lora)
        lora_config = None
        if not args.no_route and state["tags"] is None and args.lora_dir is None and not args.lora and adapters:
            # dialect routing: the adapters the routes use and the tagged model's LoRA in this one engine
            tagged_path = resolve_tagged(args.tagged_model)
            with open(os.path.join(tagged_path, "dialects.json"), encoding="utf-8") as f:
                state["tags"] = DialectTags(json.load(f))
            state["route"] = route_table(sorted(adapters), state["tags"], DEFAULT_ROUTES)
            adapters = {a: adapters[a] for a in used_adapters(state["route"])}
            lora_config = routed_lora_config(adapters, tagged_path, args.max_lora_rank)
            adapters[TAGGED_ADAPTER] = tagged_path
            if args.default_lora and args.default_lora not in state["route"]:
                raise SystemExit(f"--default-lora {args.default_lora!r} is not a dialect: {sorted(state['route'])}")
        elif args.default_lora and args.default_lora not in adapters:
            raise SystemExit(f"--default-lora {args.default_lora!r} is not among the adapters {sorted(adapters)}")
        if lora_config is None:
            lora_config = adapters_lora_config(adapters, args.max_lora_rank)
        pool = VoxCPM.from_pretrained(
            model=model_path,
            devices=[0],
            inference_timesteps=args.inference_timesteps,
            max_num_seqs=args.max_num_seqs,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_num_batched_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            lora_config=lora_config,
        )
        await pool.wait_for_ready()
        for name, path in adapters.items():
            await pool.register_lora(name, path)
        state["adapters"] = sorted(adapters)
        state["pool"] = pool
        info = await pool.get_model_info()
        state["sample_rate"] = int(info["sample_rate"])
        state["encoder_rate"] = int(info.get("encoder_sample_rate", 16000))
        if args.voice_dir:  # voices registered in earlier runs: latents are on disk, no re-encoding
            for meta, latents in load_voices(args.voice_dir):
                norm = meta.get("normalize", True)
                await register_voice(meta["name"], latents, meta.get("text"), norm)
                state["voices"][meta["name"]] = voice_entry(latents, meta.get("text"), meta.get("created_at"), norm)
            await warm_up(list(state["voices"]))
        if args.voices:
            for v in json.load(open(args.voices, encoding="utf-8")):
                if v["id"] not in state["voices"]:
                    with open(v["wav"], "rb") as f:
                        await register(v["id"], f.read(), os.path.splitext(v["wav"])[1].lstrip(".") or "wav",
                                       v.get("text"))
        await warm_up_kernels()
        dialects = (f"{len(state['route'])} dialects (routed), " if state["route"] is not None else
                    f"{len(state['tags'].names)} dialects (tagged model), " if state["tags"] else "")
        print(f"[server] ready: {len(state['voices'])} voices, {dialects}{len(state['adapters'])} LoRA adapters, "
              f"sample rate {state['sample_rate']}", flush=True)
        if not args.api_key:
            print(f"[server] no --api-key: whoever reaches {args.host}:{args.port} can use every endpoint", flush=True)
        import gc

        gc.collect()  # startup objects frozen: no full collection over them while requests stream
        gc.freeze()

        async def watch_engine() -> None:
            await pool.wait_dead()
            state["dead"] = True
            print("[server] the engine process died: shutting down (exit code 1)", flush=True)
            state["on_engine_dead"]()

        watcher = asyncio.create_task(watch_engine()) if hasattr(pool, "wait_dead") else None
        try:
            yield
        finally:
            if watcher is not None:
                watcher.cancel()
            await pool.stop()

    app = FastAPI(title="VoxCPM2 (nanovllm-voxcpm-turbo)", lifespan=lifespan)

    def engine_alive() -> bool:
        return not state["dead"] and getattr(state.get("pool"), "alive", True)

    @app.get("/health")
    async def health():
        if not engine_alive():
            return JSONResponse({"status": "engine process died"}, status_code=503)
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        base = [{"id": args.served_model_name, "object": "model", "owned_by": "local"}]
        if state["route"] is not None:
            base[0]["dialects"] = sorted(state["route"])
        elif state["tags"] is not None:
            base[0]["dialects"] = state["tags"].names
        names = sorted(state["route"]) if state["route"] is not None else state.get("adapters", [])
        loras = [{"id": name, "object": "model", "owned_by": "local", "parent": args.served_model_name}
                 for name in names]
        return {"object": "list", "data": base + loras}

    @app.post("/v1/audio/voices")
    async def upload_voice(
        name: str = Form(...),
        audio_sample: UploadFile = File(...),
        ref_text: str | None = Form(None),
        consent: str | None = Form(None),
        normalize: bool = Form(True),
    ):
        if not VOICE_NAME.match(name):
            raise HTTPException(400, "voice name: 1-64 characters from A-Z a-z 0-9 . _ -")
        if name in state["voices"] or name in state["registering"]:
            raise HTTPException(409, f"voice '{name}' exists; DELETE /v1/audio/voices/{name} first")
        limit = int(args.max_voice_mb * (1 << 20))
        if (audio_sample.size or 0) > limit:
            raise HTTPException(413, f"audio_sample: at most {args.max_voice_mb:g} MB (--max-voice-mb)")
        audio_format = os.path.splitext(audio_sample.filename or "")[1].lstrip(".") or "wav"
        state["registering"].add(name)
        try:
            return await register(name, await audio_sample.read(), audio_format, ref_text, normalize, upload=True)
        finally:
            state["registering"].discard(name)

    @app.get("/v1/audio/voices")
    async def list_voices():
        return {"voices": [{"name": k, "has_transcript": bool(v["text"])} for k, v in state["voices"].items()]}

    @app.delete("/v1/audio/voices/{name}")
    async def delete_voice(name: str):
        if state["voices"].pop(name, None) is None:
            raise HTTPException(404, f"voice '{name}' not found")
        await state["pool"].unregister_voice(name)
        if state["tags"] is not None:
            await state["tags"].drop_voice(state["pool"], name)
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

    def routed_kwargs(req: SpeechRequest) -> tuple[dict, str | None, str | None]:
        """Dialect routing: (LoRA kwargs, the tag token in front of the text or None, the dialect the text is read
        in) of the request's dialect (``dialect``, ``lora``, a dialect named as ``model``, or --default-lora)."""
        if req.lora and req.dialect and req.lora != req.dialect:
            raise HTTPException(400, "lora and dialect name different dialects")
        name = req.dialect or req.lora or (req.model if req.model in state["route"] else None) or args.default_lora
        if name is None:
            return {}, None, None  # the base model
        r = state["route"].get(name)
        if r is None:
            raise HTTPException(404, f"unknown dialect {name!r}; dialects: {', '.join(sorted(state['route']))}")
        if r.model == TAGGED:
            return {"lora_name": TAGGED_ADAPTER}, state["tags"].tags[r.dialect], r.dialect
        return {"lora_name": r.dialect}, None, r.dialect

    def dialect_tag(req: SpeechRequest) -> str | None:
        """A tagged model: the token of the request's dialect (None: no dialect)."""
        if state["tags"] is None or req.dialect is None:
            return None
        try:
            return state["tags"].tag(req.dialect)
        except ValueError as e:
            raise HTTPException(404, str(e)) from e

    def lora_kwargs(req: SpeechRequest) -> dict:
        if state["tags"] is None and req.lora and req.dialect and req.lora != req.dialect:
            raise HTTPException(400, "lora and dialect name different adapters")
        name = req.lora if state["tags"] is not None else req.lora or req.dialect
        if name is None and req.model in state["adapters"]:
            name = req.model  # vLLM's convention: an adapter is selected as the model
        name = name or args.default_lora
        if name is None:
            return {}
        if name not in state["adapters"]:
            raise HTTPException(404, f"LoRA adapter '{name}' not found; registered: {state['adapters']}")
        return {"lora_name": name}

    async def send_json(send, status: int, obj: dict):
        body = json.dumps(obj).encode()
        retry = [(b"retry-after", b"1")] if status == 503 else []
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()),
                                *retry]})
        await send({"type": "http.response.body", "body": body})

    def request_dialect(req: SpeechRequest, lk: dict) -> str | None:
        """The dialect the request is read in: its adapter (lora / dialect / model / --default-lora), or a tagged
        model's dialect with aliases resolved."""
        if state["tags"] is not None:
            return state["tags"].canonical(req.dialect) if req.dialect is not None else None
        return lk.get("lora_name")

    async def synthesize(req: SpeechRequest, text: str, vk: dict, tag: str | None = None, dialect: str | None = None):
        """Audio chunks of ``text`` (the input as read): one engine request per sentence, in order. tag: a tagged
        model's dialect token, in front of every sentence (or of the transcript: prompt_ref clones the voice's copy for
        the tag, its transcript read in ``dialect``)."""
        parts = split_text(text, args.split_max_chars) if req.split else [text]
        prefix = ""
        if tag is not None:
            if vk.get("clone_mode") in ("prompt_ref", "prompt"):  # the transcript comes first: the tag leads it
                v = state["voices"][vk["voice"]]
                transcript = read_as(v["text"], dialect) if v["normalize"] else v["text"]
                vk = {**vk, "voice": await state["tags"].voice_copy(state["pool"], vk["voice"], tag, v["latents"],
                                                                    transcript)}
            else:
                prefix = tag
        for k, part in enumerate(parts):
            gen = state["pool"].generate(
                target_text=prefix + part,
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

    def request_audio(req: SpeechRequest):
        """The audio chunks of a speech request: its voice, its adapter or dialect tag, its input as read."""
        vk = voice_kwargs(req)
        if state["route"] is not None:
            lk, tag, dialect = routed_kwargs(req)
        else:
            lk, tag = lora_kwargs(req), dialect_tag(req)
            dialect = request_dialect(req, lk)
        text = read_as(req.input, dialect) if req.normalize else req.input
        if not text.strip():  # after normalization: tatweel or marks alone read as nothing
            raise HTTPException(400, "input is empty")
        return synthesize(req, text, {**vk, **lk}, tag, dialect)

    async def speech(scope, receive, send):
        """POST /v1/audio/speech as a bare ASGI handler: with hundreds of simultaneous requests, FastAPI's
        per-request work (dependency resolution, body handling) on the event loop that also drives the engine
        delays when later requests reach the engine; this path reads the body, validates it with pydantic's
        JSON parser and streams the PCM chunks itself. The response runs against the client's disconnect, which
        cancels it and so its engine request: uvicorn's send() returns quietly once the client is gone, and the
        engine generated the rest of the audio for no one."""
        body = b""
        while True:
            msg = await receive()
            if msg["type"] == "http.disconnect":
                return
            body += msg.get("body", b"")
            if len(body) > MAX_JSON_BODY:
                return await send_json(send, 413, {"detail": f"request body over {MAX_JSON_BODY} bytes"})
            if not msg.get("more_body"):
                break

        async def disconnected() -> None:
            while (await receive())["type"] != "http.disconnect":
                pass

        job, gone = asyncio.ensure_future(respond(body, send)), asyncio.ensure_future(disconnected())
        try:
            await asyncio.wait((job, gone), return_when=asyncio.FIRST_COMPLETED)
        finally:
            gone.cancel()
            if not job.done():
                job.cancel()  # (the client is gone: closing its generator cancels the engine request)
            with contextlib.suppress(asyncio.CancelledError):
                await job

    async def respond(body: bytes, send) -> None:
        """A speech request's response, once its body is read (speech)."""
        try:
            req = SpeechRequest.model_validate_json(body)
            if req.response_format not in ("pcm", "wav"):
                raise HTTPException(400, "response_format must be 'pcm' or 'wav'")
            if len(req.input) > args.max_input_chars:
                raise HTTPException(400, f"input: at most {args.max_input_chars} characters (--max-input-chars)")
            if not engine_alive():
                raise HTTPException(503, "the engine process died; the server restarts")
            if args.max_streams and state["active"] >= args.max_streams:
                raise HTTPException(503, f"busy: {state['active']} speech requests in progress (--max-streams)")
        except ValidationError as e:
            return await send_json(send, 422, {"detail": e.errors(include_url=False)})
        except HTTPException as e:
            return await send_json(send, e.status_code, {"detail": e.detail})
        state["active"] += 1
        try:
            await respond_audio(req, send)
        finally:
            state["active"] -= 1

    async def respond_audio(req: SpeechRequest, send) -> None:
        try:
            gen = request_audio(req)
            try:
                first = await gen.__anext__()  # errors (e.g. a prompt too long) become HTTP errors
            except StopAsyncIteration:
                first = None
            except ValueError as e:  # the request's fault, also when the engine says so (too long, ...)
                raise HTTPException(400, str(e)) from e
            except RuntimeError as e:
                if engine_alive():
                    raise
                raise HTTPException(503, "the engine process died; the server restarts") from e
        except HTTPException as e:
            return await send_json(send, e.status_code, {"detail": e.detail})
        try:
            sr = str(state["sample_rate"]).encode()
            if req.stream and req.response_format == "pcm":
                await send({"type": "http.response.start", "status": 200,
                            "headers": [(b"content-type", b"audio/pcm"), (b"x-sample-rate", sr)]})
                if first is not None:
                    await send({"type": "http.response.body", "body": pcm16(first), "more_body": True})
                async for chunk in gen:
                    await send({"type": "http.response.body", "body": pcm16(chunk), "more_body": True})
                await send({"type": "http.response.body", "body": b"", "more_body": False})
                return
            chunks = [] if first is None else [pcm16(first)]
            async for chunk in gen:
                chunks.append(pcm16(chunk))
            pcm = b"".join(chunks)
            out, ctype = ((wav_bytes(pcm, int(sr)), b"audio/wav") if req.response_format == "wav"
                          else (pcm, b"audio/pcm"))
            await send({"type": "http.response.start", "status": 200,
                        "headers": [(b"content-type", ctype), (b"content-length", str(len(out)).encode()),
                                    (b"x-sample-rate", sr)]})
            await send({"type": "http.response.body", "body": out})
        finally:
            await gen.aclose()

    key = f"Bearer {args.api_key}".encode() if args.api_key else None

    def authorized(scope) -> bool:
        """--api-key: the request's ``Authorization: Bearer <key>`` (/health needs none)."""
        if key is None or scope["path"] == "/health":
            return True
        got = next((v for k, v in scope.get("headers", ()) if k == b"authorization"), b"")
        return hmac.compare_digest(got, key)

    async def asgi(scope, receive, send):
        if scope["type"] == "http" and not authorized(scope):
            return await send_json(send, 401, {"detail": "missing or wrong API key (Authorization: Bearer <key>)"})
        if scope["type"] == "http" and scope["method"] == "POST" and scope["path"] == "/v1/audio/speech":
            return await speech(scope, receive, send)
        return await app(scope, receive, send)

    asgi.state = state  # (main: the engine's death ends the server, which then exits non-zero)
    return asgi


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=DEFAULT_MODEL, help="VoxCPM2 model directory or Hugging Face repo id")
    ap.add_argument("--dit", default=DEFAULT_DIT, help="distilled LocDiT checkpoint (path or repo:file); 'none': the "
                                                       "model's own LocDiT (10 steps, quality mode)")
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
    ap.add_argument("--default-lora", help="adapter for requests that name none (default: the base model); with "
                                           "routing, a dialect")
    ap.add_argument("--no-route", action="store_true", help="no dialect routing: every adapter of the model, no "
                                                             "tagged model (the engine before routing)")
    ap.add_argument("--tagged-model", default=DEFAULT_TAGGED_MODEL, help="routing: the tagged model whose LoRA "
                                                                          "serves the routed-away dialects")
    ap.add_argument("--max-lora-rank", type=int, help="LoRA slot rank (default: the largest adapter rank)")
    ap.add_argument("--api-key", default=os.environ.get("VOXCPM_API_KEY"),
                    help="every endpoint but /health asks for 'Authorization: Bearer <key>' (default: $VOXCPM_API_KEY; "
                         "none: open to whoever reaches the port)")
    ap.add_argument("--max-streams", type=int, default=0, help="speech requests at once; more get 503 with "
                                                                "Retry-After (0: no limit; one A10: 150)")
    ap.add_argument("--max-input-chars", type=int, default=4096, help="longest input of a speech request")
    ap.add_argument("--max-voice-mb", type=float, default=10.0, help="largest uploaded voice clip")
    ap.add_argument("--max-voice-seconds", type=float, default=30.0, help="longest uploaded voice clip")
    args = ap.parse_args()
    os.environ.setdefault("NANOVLLM_SUBMIT_FLUSH_EVERY", "4")  # requests reach the engine as a steady stream
    app = build_app(args)
    server = uvicorn.Server(uvicorn.Config(app, host=args.host, port=args.port, loop="uvloop", http="httptools",
                                           backlog=4096, timeout_keep_alive=30, log_level="warning"))
    app.state["on_engine_dead"] = lambda: setattr(server, "should_exit", True)
    server.run()
    if app.state["dead"]:  # (exit code 1: a supervisor starts the server again)
        sys.exit(1)


if __name__ == "__main__":
    main()
