"""Voice cloning with a model's dialect adapters, in the low-latency configuration.

One engine holds every adapter of the model's ``adapters/`` directory; a voice is registered once from a reference
clip and then cloned by name, mixed with any dialect in the same batch::

    tts = await DialectTTS.load()                              # VoxCPM-Kernel/VoxCPM2-Arabic-Dialects
    await tts.add_voice("speaker", "reference.wav")            # text="<transcript>": prompt + reference cloning
    audio = await tts.synthesize("هلا والله، كيف حالك؟", dialect="najdi", voice="speaker")  # float32, tts.sample_rate
    async for chunk in tts.stream("...", dialect="saudi", voice="speaker"):          # the same, as it is generated
        ...
    await tts.close()
"""

from __future__ import annotations

import json
import os
from typing import AsyncIterator

import numpy as np

DEFAULT_MODEL = "VoxCPM-Kernel/VoxCPM2-Arabic-Dialects"
DEFAULT_DIT = "kadirnar/VoxCPM2-Turbo-distilled-DiT:student_k2_dmd.pt"
_INT16_SCALE = np.float32(1 / 32768)


def resolve(spec: str) -> str:
    """A local path as is; otherwise a Hugging Face repo id (downloaded snapshot) or ``repo:file`` (one file)."""
    if os.path.exists(os.path.expanduser(spec)):
        return os.path.expanduser(spec)
    from huggingface_hub import hf_hub_download, snapshot_download

    if ":" in spec:
        repo, filename = spec.split(":", 1)
        return hf_hub_download(repo, filename)
    return snapshot_download(spec)


def turbo_config(dit: str | None, head_tokens: int = 8, dit_cache: str | dict | None = None) -> dict:
    """The measured low-latency configuration: distilled 2-step LocDiT (``dit``), int8 layers, head start of
    ``head_tokens`` tokens (0 = off), int16 audio chunks. dit_cache: with the original LocDiT (``dit`` None), reuse
    its middle layers on some solver steps ("light", "medium" or a plan, see models/voxcpm2/dit_cache.py)."""
    cfg = {"w8a8": True, "w8a8_targets": ["base_lm", "residual_lm", "feat_encoder", "feat_decoder"],
           "pcm16_output": True, "head_tokens": head_tokens, "head_patches": 3}
    if dit and dit_cache:
        raise ValueError("dit_cache is for the original LocDiT (dit=None): the distilled one has no step to spare")
    if dit:
        cfg["dit_checkpoint"] = resolve(dit)
    if dit_cache:
        from nanovllm_voxcpm.models.voxcpm2.dit_cache import plan

        cfg["dit_cache"] = plan(dit_cache)
    return cfg


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


class DialectTTS:
    """A running engine with a model's dialect adapters registered, and the voices added to it."""

    def __init__(self, server, dialects: list[str], sample_rate: int):
        self.server = server
        self.dialects = dialects
        self.sample_rate = sample_rate
        self.voices: dict[str, bool] = {}  # name -> registered with a transcript

    @classmethod
    async def load(cls, model: str = DEFAULT_MODEL, dit: str | None = DEFAULT_DIT, *, max_concurrency: int = 256,
                   head_tokens: int = 8, dit_cache: str | dict | None = None, gpu_memory_utilization: float = 0.85,
                   kv_blocks: int = 600, device: int = 0) -> "DialectTTS":
        """Start the engine on ``model`` (local directory or Hugging Face repo with ``adapters/<dialect>/``).

        dit: the distilled 2-step LocDiT (path or ``repo:file``; None = the model's own LocDiT, slower).
        head_tokens: the first audio comes from the text's first <= head_tokens tokens (0 = from the whole text).
        dit_cache: with dit=None, "light" or "medium": reuse the LocDiT's middle layers on 2 or 4 of its 9 steps.
        """
        # the engine process reads its configuration from the environment when it starts
        os.environ["NANOVLLM_VOXCPM2_TURBO"] = json.dumps(turbo_config(dit, head_tokens, dit_cache))
        if kv_blocks:
            os.environ["NANOVLLM_SERVERPOOL_NUM_KVCACHE_BLOCKS"] = str(kv_blocks)
        from nanovllm_voxcpm import VoxCPM

        path = resolve(model)
        adapters = find_adapters(os.path.join(path, "adapters") if os.path.isdir(os.path.join(path, "adapters")) else None)
        server = VoxCPM.from_pretrained(
            model=path, devices=[device], max_num_seqs=max_concurrency, max_model_len=1024, max_num_batched_tokens=4096,
            gpu_memory_utilization=gpu_memory_utilization, lora_config=adapters_lora_config(adapters),
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
        return cls(server, sorted(adapters), sample_rate)

    async def add_voice(self, name: str, audio: str | bytes, text: str | None = None, audio_format: str = "wav") -> None:
        """Register a voice from a reference clip (a file path or its bytes). With ``text`` (the clip's transcript)
        the voice is cloned by continuing the reference (``prompt_ref``); without it, from its timbre (``ref``: the
        fastest under load, its KV cached once per voice and dialect)."""
        if isinstance(audio, str):
            audio_format = os.path.splitext(audio)[1].lstrip(".") or audio_format
            with open(audio, "rb") as f:
                audio = f.read()
        latents = await self.server.encode_latents(audio, audio_format)
        await self.server.register_voice(name, latents, text or None)
        self.voices[name] = bool(text)

    async def remove_voice(self, name: str) -> None:
        await self.server.unregister_voice(name)
        self.voices.pop(name, None)

    async def stream(self, text: str, dialect: str | None = None, voice: str | None = None, *,
                     clone_mode: str | None = None, seed: int | None = None, cfg_value: float = 2.0,
                     max_patches: int = 600) -> AsyncIterator[np.ndarray]:
        """Audio chunks (float32 in [-1, 1], ``sample_rate``) of ``text`` in ``dialect`` (None: the base model),
        cloning ``voice`` (None: no cloning). max_patches: at most this many 160 ms patches."""
        if dialect is not None and dialect not in self.dialects:
            raise ValueError(f"unknown dialect {dialect!r}; dialects: {', '.join(self.dialects)}")
        kw = {}
        if voice is not None:
            if voice not in self.voices:
                raise ValueError(f"voice {voice!r} is not registered (add_voice)")
            kw = {"voice": voice, "clone_mode": clone_mode or ("prompt_ref" if self.voices[voice] else "ref")}
        async for chunk in self.server.generate(target_text=text, lora_name=dialect, seed=seed, cfg_value=cfg_value,
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
