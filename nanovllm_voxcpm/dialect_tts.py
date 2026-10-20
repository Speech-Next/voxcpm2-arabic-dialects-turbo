"""Voice cloning with a model's dialect adapters, in the low-latency configuration: the unrouted engine (the default
one is RoutedDialectTTS, dialect_routing.py: these adapters and teacher-v3, each dialect on the better one).

One engine holds every adapter of the model's ``adapters/`` directory; a voice is registered once from a reference
clip and then cloned by name, mixed with any dialect in the same batch. A model with a ``dialects.json`` instead
(one model for all dialects) is steered by a dialect tag token in front of the text, with no adapters::

    tts = await DialectTTS.load()                              # VoxCPM-Kernel/VoxCPM2-Arabic-Dialects
    await tts.add_voice("speaker", "reference.wav")            # text="<transcript>": prompt + reference cloning
    audio = await tts.synthesize("هلا والله، كيف حالك؟", dialect="najdi", voice="speaker")  # float32, tts.sample_rate
    async for chunk in tts.stream("...", dialect="saudi", voice="speaker"):          # the same, as it is generated
        ...
    await tts.close()
"""

from __future__ import annotations

import asyncio
import json
import os
import unicodedata
from typing import AsyncIterator

import numpy as np

from nanovllm_voxcpm.arabic_text import has_arabic, normalize as normalize_text

DEFAULT_MODEL = "VoxCPM-Kernel/VoxCPM2-Arabic-Dialects"
# 3 steps, distilled from these adapters; ":student_k2_dmd.pt", 2 steps, is 11 ms faster at 200 requests and less
# natural (the model card)
DEFAULT_DIT = "VoxCPM-Kernel/VoxCPM2-Arabic-Dialects-Distilled-DiT:student_k3.pt"
NOT_ARABIC = frozenset({"hindi_english"})  # dialects whose text is not read as Arabic
_INT16_SCALE = np.float32(1 / 32768)


def read_as(text: str, dialect: str | None = None) -> str:
    """``text`` as the model reads it in ``dialect`` (canonical name, aliases resolved; None: the base model or a
    transcript without a dialect): arabic_text.normalize, formal for "msa", colloquial otherwise. Unchanged for a
    NOT_ARABIC dialect and for text without Arabic script (Hindi, English: their digits are not Arabic words)."""
    if dialect in NOT_ARABIC or not has_arabic(text):
        return text
    return normalize_text(text, "msa" if dialect == "msa" else "colloquial")


def speakable(text: str) -> bool:
    """Whether ``text`` has something to speak: a letter or a digit. Text of punctuation, symbols, emoji, marks or
    tatweel alone was spoken anyway (the model made up sounds for it)."""
    return any(c.isalnum() and unicodedata.category(c) != "Lm" for c in text)


def check_text(text: str, normalized: bool = False) -> None:
    """ValueError for text with nothing to speak (empty, or punctuation / symbols / emoji only)."""
    if not text.strip():
        raise ValueError("text is empty" + (" after normalization" if normalized else ""))
    if not speakable(text):
        raise ValueError("text has nothing to speak (only punctuation, symbols or emoji)")


def resolve(spec: str) -> str:
    """A local path as is; otherwise a Hugging Face repo id (downloaded snapshot) or ``repo:file`` (one file), at the
    repo's pinned revision or ``repo@revision`` (nanovllm_voxcpm.hub)."""
    if os.path.exists(os.path.expanduser(spec)):
        return os.path.expanduser(spec)
    from nanovllm_voxcpm.hub import download

    return download(spec)


def turbo_config(dit: str | None, head_tokens: int = 8, dit_cache: str | dict | None = None, w8a8: bool = True) -> dict:
    """The measured low-latency configuration: distilled 3-step LocDiT (``dit``), int8 layers, head start of
    ``head_tokens`` tokens (0 = off), int16 audio chunks, a stream's leading silence capped at 2 patches (320 ms),
    no new request admitted while a playing stream has less than 100 ms of audio queued at its player, and for
    requests that start together (#28): heads of up to 7 patches, enough audio for their continuations' prefill,
    prefill steps of at most 130 ms while streams play, every prefill step then also decoding the streams short of
    audio (the continuations' and, #31, the heads' of a burst's late arrivals).
    New heads are not held (#31: first audio stays at ~0.1 s at 200 at once): a head whose whole continuation would
    come too late speaks the first 24, then 48 text tokens first (intermediate stages, each reusing the KV of the one
    before; up to 16 patches each, up to 7 fewer by the stream's id: a burst's streams reach the seams apart). A front that sends the audio on in chunks passes their length (stream's
    chunk_ms), so that the deadlines allow for what it holds back, and a paced front its pace (stream's pace, e.g.
    0.9: one chunk per 0.9 x its audio at most), so that they follow its send clock.
    With a distilled LocDiT, a prefill step of new streams solves their first patch in its grid's first step and one
    jump to 0, t = [1.0, 0.7071, 0.0]: the 2-step student's grid (the 3-step one was trained from it and shares its
    first step); 200 requests at once: TTFB -12 ms (#31).
    dit_cache: with the original LocDiT (``dit`` None), reuse its middle layers on some solver steps ("light",
    "medium" or a plan, see models/voxcpm2/dit_cache.py).
    w8a8=False: bf16 layers (quality measurements of a model whose SmoothQuant statistics were not calibrated)."""
    cfg = {"pcm16_output": True, "head_tokens": head_tokens, "lead_silence_patches": 2,
           "stream_guard_ms": 100, "head_patches_max": 7, "mix_decode": True, "mix_heads": True, "step_cap_ms": 130,
           "mid_tokens": [24, 48], "mid_patches_max": 16, "mid_stagger": 8, "stage_kv_reuse": True}
    if w8a8:
        cfg["w8a8"] = True
    if dit and dit_cache:
        raise ValueError("dit_cache is for the original LocDiT (dit=None): the distilled one has no step to spare")
    if dit:
        cfg["dit_checkpoint"] = resolve(dit)
        cfg["dit_first_t_span"] = [1.0, 0.7071, 0.0]
    if dit_cache:
        from nanovllm_voxcpm.models.voxcpm2.dit_cache import plan

        cfg["dit_cache"] = plan(dit_cache)
    return cfg


def use_model_calibration(model_path: str) -> str | None:
    """A model directory's own SmoothQuant statistics (``smooth_stats.pt``, recorded with VOXCPM_KERNELS_CALIBRATE)
    for the int8 layers of the engine started next; VOXCPM_KERNELS_SMOOTH, when set, still wins. Without them the
    engine smooths with base VoxCPM2's statistics, which a fine-tuned model's activations may not match."""
    stats = os.path.join(model_path, "smooth_stats.pt")
    if os.path.exists(stats) and "VOXCPM_KERNELS_SMOOTH" not in os.environ:
        os.environ["VOXCPM_KERNELS_SMOOTH"] = stats  # read by the engine process when it starts
        return stats
    return None


def find_adapters(lora_dir: str | None, extra: list[str] = ()) -> dict[str, str]:
    """Adapter name -> directory: the subdirectories of ``lora_dir`` holding ``lora_weights.safetensors``, plus
    ``NAME=PATH`` items."""
    out: dict[str, str] = {}
    if lora_dir:
        for name in sorted(os.listdir(lora_dir)):
            path = os.path.join(lora_dir, name)
            if os.path.isfile(os.path.join(path, "lora_weights.safetensors")):
                out[name] = path
    for item in extra:
        name, sep, path = item.partition("=")
        if not sep or not name or not os.path.isdir(path):
            raise SystemExit(f"--lora expects NAME=DIRECTORY, got {item!r}")
        out[name] = path
    return out


def adapters_lora_config(adapters: dict[str, str], max_rank: int | None = None):
    """The engine's LoRAConfig for ``adapters`` (None without any): every adapter resident (max_loras = their
    number), the largest rank, and the targets of their ``lora_config.json`` (they must agree)."""
    if not adapters:
        return None
    from nanovllm_voxcpm.models.voxcpm2.config import LoRAConfig

    keys = ("enable_lm", "enable_dit", "enable_proj", "target_modules_lm", "target_modules_dit", "target_proj_modules")
    shared, ranks = None, []
    for name, path in adapters.items():
        cfg = json.load(open(os.path.join(path, "lora_config.json"), encoding="utf-8")).get("lora_config", {})
        ranks.append(int(cfg.get("r", cfg.get("rank", 0)) or 0))
        targets = {k: sorted(cfg[k]) if isinstance(cfg.get(k), list) else cfg.get(k) for k in keys if k in cfg}
        if shared is None:
            shared = targets
        elif targets != shared:
            raise SystemExit(f"LoRA adapter {name!r} targets other modules than the others: {targets} vs {shared}")
    fields = {k: v for k, v in (shared or {}).items() if v is not None}
    # q/k/v in their natural order: the qkv LoRA expands straight into the packed output
    for k in ("target_modules_lm", "target_modules_dit"):
        if k in fields:
            order = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            fields[k] = sorted(fields[k], key=lambda t: order.index(t) if t in order else len(order))
    if not fields.get("enable_proj"):
        fields.pop("target_proj_modules", None)
    return LoRAConfig(max_loras=len(adapters), max_lora_rank=max_rank or max(ranks) or 32, **fields)


async def start_engine(path: str, dit: str | None, adapters: dict[str, str], lora_config, *, max_concurrency: int,
                       head_tokens: int, dit_cache, gpu_memory_utilization: float, kv_blocks: int, device: int,
                       w8a8: bool, smooth_stats: str | None = None, prefill_tokens: int = 16384):
    """(server, sample rate): the engine on the model directory ``path`` in the low-latency configuration
    (turbo_config), with ``adapters`` (name -> directory) registered. smooth_stats: SmoothQuant statistics for its
    int8 layers (default: the model's own, use_model_calibration). prefill_tokens: the most tokens one prefill step
    computes (max_num_batched_tokens)."""
    # the engine process reads its configuration from the environment when it starts
    os.environ["NANOVLLM_VOXCPM2_TURBO"] = json.dumps(turbo_config(dit, head_tokens, dit_cache, w8a8))
    if kv_blocks:
        os.environ["NANOVLLM_SERVERPOOL_NUM_KVCACHE_BLOCKS"] = str(kv_blocks)
    from nanovllm_voxcpm import VoxCPM

    if w8a8:
        if smooth_stats:
            os.environ["VOXCPM_KERNELS_SMOOTH"] = smooth_stats
        else:
            use_model_calibration(path)
    server = VoxCPM.from_pretrained(
        model=path, devices=[device], max_num_seqs=max_concurrency, max_model_len=1024,
        max_num_batched_tokens=prefill_tokens, gpu_memory_utilization=gpu_memory_utilization, lora_config=lora_config,
    )
    await server.wait_for_ready()
    for name, adapter_path in adapters.items():
        await server.register_lora(name, adapter_path)
    sample_rate = int((await server.get_model_info())["sample_rate"])
    # this process's startup objects (torch, transformers, ...) out of the garbage collector's reach: a full
    # collection over them paused a burst's submission for ~165 ms (200 requests: half sent, the rest late)
    import gc

    gc.collect()
    gc.freeze()
    return server, sample_rate


class DialectTags:
    """A tagged model's dialects (its ``dialects.json``: {"tags": {dialect: token}, "aliases": {name: dialect}}).

    The dialect's tag token leads the text. A voice cloned with its transcript (``prompt_ref``, ``prompt``) has it
    before the text, so there the tag leads the transcript instead: one copy of the voice per dialect,
    ``<voice>@<tag>``, registered on first use."""

    def __init__(self, spec: dict):
        self.tags: dict[str, str] = spec["tags"]
        self.aliases: dict[str, str] = spec.get("aliases", {})
        self.names = sorted(self.tags) + sorted(self.aliases)
        self._copies: dict[str, asyncio.Future] = {}  # copy name -> its registration (shared by concurrent requests)

    @classmethod
    def of_model(cls, path: str) -> "DialectTags | None":
        spec = os.path.join(path, "dialects.json")
        return cls(json.load(open(spec, encoding="utf-8"))) if os.path.exists(spec) else None

    def canonical(self, dialect: str) -> str:
        """The dialect an alias names (qatari -> gulf); a dialect itself as is."""
        if dialect not in self.tags and dialect not in self.aliases:
            raise ValueError(f"unknown dialect {dialect!r}; dialects: {', '.join(self.names)}")
        return self.aliases.get(dialect, dialect)

    def tag(self, dialect: str) -> str:
        return self.tags[self.canonical(dialect)]

    async def voice_copy(self, server, voice: str, tag: str, latents: bytes, transcript: str) -> str:
        """The name of ``voice``'s copy whose transcript follows ``tag``, registered with the engine once."""
        copy = f"{voice}@{tag}"
        fut = self._copies.get(copy)
        if fut is None:
            fut = self._copies[copy] = asyncio.ensure_future(server.register_voice(copy, latents, tag + transcript))
        try:
            await asyncio.shield(fut)
        except Exception:
            if self._copies.get(copy) is fut:  # a failed registration is tried again by the next request
                del self._copies[copy]
            raise
        return copy

    async def drop_voice(self, server, voice: str) -> None:
        """Unregister ``voice``'s copies."""
        for copy in [c for c in self._copies if c.startswith(voice + "@")]:
            fut = self._copies.pop(copy)
            try:
                await fut
            except Exception:
                continue
            await server.unregister_voice(copy)


class DialectTTS:
    """A running engine with a model's dialect adapters registered, and the voices added to it."""

    def __init__(self, server, dialects: list[str], sample_rate: int, tags: dict | None = None):
        """tags: a tagged model's dialects.json ({"tags": {dialect: token}, "aliases": {name: dialect}})."""
        self.server = server
        self.sample_rate = sample_rate
        self.tags = DialectTags(tags) if tags else None
        self.dialects = self.tags.names if tags else dialects
        self.voices: dict[str, bool] = {}  # name -> registered with a transcript
        # tagged models: latents, transcript and its normalize flag of prompt_ref voices
        self._voice_src: dict[str, tuple[bytes, str, bool]] = {}

    @classmethod
    async def load(cls, model: str = DEFAULT_MODEL, dit: str | None = DEFAULT_DIT, *, max_concurrency: int = 256,
                   head_tokens: int = 8, dit_cache: str | dict | None = None, gpu_memory_utilization: float = 0.85,
                   kv_blocks: int = 600, device: int = 0, w8a8: bool = True,
                   prefill_tokens: int = 16384) -> "DialectTTS":
        """Start the engine on ``model`` (local directory or Hugging Face repo with ``adapters/<dialect>/``).

        dit: the distilled 3-step LocDiT (path or ``repo:file``; None = the model's own LocDiT, slower).
        head_tokens: the first audio comes from the text's first <= head_tokens tokens (0 = from the whole text).
        dit_cache: with dit=None, "light" or "medium": reuse the LocDiT's middle layers on 2 or 4 of its 9 steps.
        prefill_tokens: the most tokens one prefill step computes (16384: a burst's continuations in fewer steps).
        """
        path = resolve(model)
        tags_path = os.path.join(path, "dialects.json")
        tags = json.load(open(tags_path, encoding="utf-8")) if os.path.exists(tags_path) else None
        adapters = {} if tags else find_adapters(
            os.path.join(path, "adapters") if os.path.isdir(os.path.join(path, "adapters")) else None)
        server, sample_rate = await start_engine(
            path, dit, adapters, adapters_lora_config(adapters), max_concurrency=max_concurrency,
            head_tokens=head_tokens, dit_cache=dit_cache, gpu_memory_utilization=gpu_memory_utilization,
            kv_blocks=kv_blocks, device=device, w8a8=w8a8, prefill_tokens=prefill_tokens)
        return cls(server, sorted(adapters), sample_rate, tags)

    async def add_voice(self, name: str, audio: str | bytes, text: str | None = None, audio_format: str = "wav",
                        normalize: bool = True) -> None:
        """Register a voice from a reference clip (a file path or its bytes). With ``text`` (the clip's transcript)
        the voice is cloned by continuing the reference (``prompt_ref``); without it, from its timbre (``ref``: the
        fastest under load, its KV cached once per voice and dialect). normalize: the transcript is read like target
        text (read_as: colloquial; a tagged model's copy of the voice per dialect in that dialect's style)."""
        if isinstance(audio, str):
            audio_format = os.path.splitext(audio)[1].lstrip(".") or audio_format
            with open(audio, "rb") as f:
                audio = f.read()
        latents = await self.server.encode_latents(audio, audio_format)
        await self.server.register_voice(name, latents, (read_as(text) if normalize else text) if text else None)
        self.voices[name] = bool(text)
        if self.tags is not None and text:
            self._voice_src[name] = (latents, text, normalize)

    async def remove_voice(self, name: str) -> None:
        await self.server.unregister_voice(name)
        self.voices.pop(name, None)
        self._voice_src.pop(name, None)
        if self.tags is not None:
            await self.tags.drop_voice(self.server, name)

    async def stream(self, text: str, dialect: str | None = None, voice: str | None = None, *,
                     clone_mode: str | None = None, seed: int | None = None, cfg_value: float = 2.0,
                     max_patches: int = 600, normalize: bool = True,
                     chunk_ms: float | None = None, pace: float | dict | None = None) -> AsyncIterator[np.ndarray]:
        """Audio chunks (float32 in [-1, 1], ``sample_rate``) of ``text`` in ``dialect`` (None: the base model),
        cloning ``voice`` (None: no cloning). max_patches: at most this many 160 ms patches. normalize: Arabic text
        read as the model was trained on it (read_as: digits, dates, times, amounts as words; formal when the dialect,
        aliases resolved, is msa); hindi_english and text without Arabic script stay as they are. chunk_ms: the
        caller sends the audio on in chunks of this many ms, holding the rest back until a chunk is full (e.g. 128 for
        4096-byte chunks of 16 kHz PCM16): the engine's deadlines allow for it (None: every patch whole).
        pace: the caller paces the chunks, one per pace x chunk_ms at most (a jitter buffer): the deadlines follow its
        send clock (None: not paced)."""
        if dialect is not None and dialect not in self.dialects:
            raise ValueError(f"unknown dialect {dialect!r}; dialects: {', '.join(self.dialects)}")
        canonical = self.tags.canonical(dialect) if self.tags is not None and dialect is not None else dialect
        if normalize:
            text = read_as(text, canonical)
        check_text(text, normalize)
        kw = {}
        if voice is not None:
            if voice not in self.voices:
                raise ValueError(f"voice {voice!r} is not registered (add_voice)")
            kw = {"voice": voice, "clone_mode": clone_mode or ("prompt_ref" if self.voices[voice] else "ref")}
        lora = dialect
        if self.tags is not None:
            lora = None
            if dialect is not None:
                tag = self.tags.tag(dialect)
                if kw.get("clone_mode") in ("prompt_ref", "prompt"):  # the transcript comes first: the tag leads it
                    latents, transcript, norm = self._voice_src[voice]
                    kw["voice"] = await self.tags.voice_copy(self.server, voice, tag, latents,
                                                             read_as(transcript, canonical) if norm else transcript)
                else:
                    text = tag + text
        if chunk_ms is not None:
            kw["client_chunk_ms"] = chunk_ms
        if pace is not None:
            kw["client_pace"] = pace
        async for chunk in self.server.generate(target_text=text, lora_name=lora, seed=seed, cfg_value=cfg_value,
                                                max_generate_length=max_patches, **kw):
            # one pass to float32 (a burst's 200 first chunks reach their streams one after another)
            yield np.multiply(chunk, _INT16_SCALE, dtype=np.float32) if chunk.dtype == np.int16 else chunk

    async def synthesize(self, text: str, dialect: str | None = None, voice: str | None = None, **kw) -> np.ndarray:
        """The whole utterance (float32, ``sample_rate``); keyword arguments as for ``stream``."""
        chunks = [c async for c in self.stream(text, dialect, voice, **kw)]
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    async def close(self) -> None:
        await self.server.stop()

    async def __aenter__(self) -> "DialectTTS":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()
