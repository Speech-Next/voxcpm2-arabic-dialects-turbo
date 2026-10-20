"""Per-dialect routing (the default engine; DialectTTS is the unrouted one): one engine serves the adapters model's
adapters and a tagged model, each request spoken by the one that serves its dialect better.

The tagged model (VoxCPM-Kernel/VoxCPM2-Arabic-Teacher-v3) is base VoxCPM2 plus a LoRA (base_lm attention and
MLP) and trained embeddings for its dialect tag tokens; its unmerged training state (``lora/``) is served as one more
adapter of the adapters model (VoxCPM-Kernel/VoxCPM2-Arabic-Dialects), its tags as extra embedding rows that only
its requests read. A dialect named in ``routes`` goes to its adapter, every other dialect of the tagged model to the
tagged adapter with its tag, a name only the adapters model knows (hindi_english) to that adapter::

    tts = await RoutedDialectTTS.load(dit=None)                          # quality mode (10 LocDiT steps)
    await tts.add_voice("speaker", "reference.wav")
    audio = await tts.synthesize("شلونك؟", dialect="bahraini", voice="speaker")  # the qatari adapter
    audio = await tts.synthesize("هلا والله", dialect="najdi", voice="speaker")  # teacher-v3, its najdi tag
    await tts.close()

Default routes, from #2 (the held-out set, 17 groups): teacher-v3 is better pooled but worse than the adapters on
Bahraini, Egyptian and Saudi, and less natural on Hejazi (UTMOS) in int8. Names resolve as below; ``tts.table`` holds
the result.

| name | served by | adapter / tag |
|---|---|---|
| bahraini | adapter | qatari |
| egyptian | adapter | egyptian |
| saudi | adapter | saudi |
| hejazi | adapter | hejazi |
| hindi_english | adapter | hindi_english |
| None (no dialect) | base model | none, as DialectTTS |
| msa, lebanese, syrian, iraqi, najdi | teacher-v3 | its own tag |
| yemeni, sudanese, libyan, tunisian, algerian, moroccan | teacher-v3 | its own tag |
| gulf, emirati, qatari, khaliji | teacher-v3 | gulf |
| palestinian, jordanian, ab, levantine | teacher-v3 | palestinian |

An alias follows the tagged model's dialects.json (qatari, emirati and khaliji are gulf there; jordanian, ab and
levantine palestinian), so "qatari" is Gulf Arabic on teacher-v3, and the qatari adapter speaks "bahraini" (the #2
Bahraini group was measured with it). A route keyed by a name wins over one keyed by its dialect:
``routes={**DEFAULT_ROUTES, "qatari": "qatari"}`` sends "qatari" alone to the qatari adapter. Only the adapters the
table uses are loaded.

Cost (one A10, int8, 600 KV blocks): the speed of the unrouted engine and less memory than it (5 adapters instead
of 11, plus teacher-v3's merged MLP weights, 1 GB; README). In int8 the tagged requests run the merged int8 weights
at every batch size, as the adapters do; SmoothQuant statistics are the base model's (teacher-v3's own, max-combined
with them, changed nothing on #2).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import AsyncIterator

import numpy as np

from nanovllm_voxcpm.dialect_tts import (DEFAULT_DIT, DEFAULT_MODEL, DialectTags, check_text, find_adapters, read_as,
                                         resolve, start_engine)

DEFAULT_TAGGED_MODEL = "VoxCPM-Kernel/VoxCPM2-Arabic-Teacher-v3"
# dialect (or name) -> adapter
DEFAULT_ROUTES = {"bahraini": "qatari", "egyptian": "egyptian", "saudi": "saudi", "hejazi": "hejazi"}
ADAPTERS, TAGGED = "adapters", "tagged"
TAGGED_ADAPTER = "@tagged"  # the tagged model's name in the engine (no adapter directory is named so)
# the files of a tagged model a routed engine reads (not its merged weights)
TAGGED_FILES = ["lora/*", "dialects.json", "tokenizer*", "special_tokens_map.json", "config.json"]
_ORDER = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
_MLP = {"gate_proj", "up_proj", "down_proj"}


@dataclass(frozen=True)
class Route:
    model: str  # ADAPTERS or TAGGED
    dialect: str | None  # an adapter of the adapters model, a dialect of the tagged model; None = the base model


def route_table(adapters: list[str], tags: DialectTags, routes: dict[str, str]) -> dict[str, Route]:
    """name -> Route for every name of both models (and of ``routes``). A name is looked up in ``routes`` first, then
    its tagged dialect (aliases resolved): found, that adapter; else the tagged model when it knows the name; else
    the adapters model, the name being one of its adapters."""
    unknown = sorted({a for a in routes.values() if a not in adapters})
    if unknown:
        raise ValueError(f"routes name adapters the adapters model lacks: {unknown}; adapters: {', '.join(adapters)}")
    table = {}
    for name in sorted(set(adapters) | set(tags.names) | set(routes)):
        dialect = tags.canonical(name) if name in tags.names else None
        adapter = routes.get(name, routes.get(dialect) if dialect else None)
        if adapter is not None:
            table[name] = Route(ADAPTERS, adapter)
        elif dialect is not None:
            table[name] = Route(TAGGED, dialect)
        else:
            table[name] = Route(ADAPTERS, name)
    return table


def used_adapters(table: dict[str, Route]) -> list[str]:
    """The adapters the table sends requests to (the engine loads only these)."""
    return sorted({r.dialect for r in table.values() if r.model == ADAPTERS and r.dialect is not None})


def resolve_tagged(spec: str) -> str:
    """A tagged model's directory (local, or the TAGGED_FILES of a Hugging Face repo) with lora/ and dialects.json."""
    from nanovllm_voxcpm.models.voxcpm2.lora_loader import is_tagged_adapter

    path = os.path.expanduser(spec)
    if not os.path.exists(path):
        from huggingface_hub import snapshot_download

        path = snapshot_download(spec, allow_patterns=TAGGED_FILES)
    if not is_tagged_adapter(path):
        raise ValueError(f"{spec}: not a tagged model with its training state (lora/trainable.safetensors, "
                         "lora/train_state.json, dialects.json)")
    return path


def _order(targets) -> list[str]:
    return sorted(targets, key=lambda t: _ORDER.index(t) if t in _ORDER else len(_ORDER))


def routed_lora_config(adapters: dict[str, str], tagged_path: str, max_rank: int | None = None):
    """The engine's LoRAConfig for ``adapters`` (name -> directory) and the tagged model's adapter: the union of their
    target modules per LM (the tagged LoRA trains base_lm only), merged MLP weights for the adapters with MLP LoRA,
    the tags' embedding rows."""
    from nanovllm_voxcpm.models.voxcpm2.config import LoRAConfig
    from nanovllm_voxcpm.models.voxcpm2.lora_loader import tagged_adapter_state

    state = tagged_adapter_state(tagged_path)
    cfgs = [json.load(open(os.path.join(p, "lora_config.json"), encoding="utf-8")).get("lora_config", {})
            for p in adapters.values()]
    tagged = state["lora"]
    trains = set(tagged.get("train_modules") or ["base_lm", "residual_lm"])
    base, residual, dit, n_mlp, ranks = set(), set(), set(), 0, []
    for c, is_tagged in [(c, False) for c in cfgs] + [(tagged, True)]:
        lm = set(c.get("target_modules_lm") or []) if c.get("enable_lm", True) else set()
        lm_base = lm if not is_tagged or "base_lm" in trains else set()
        base |= lm_base
        residual |= lm if not is_tagged or "residual_lm" in trains else set()
        if c.get("enable_dit", True) and (not is_tagged or trains & {"feat_decoder", "dit"}):
            dit |= set(c.get("target_modules_dit") or [])
        n_mlp += bool(_MLP & lm_base)
        ranks.append(int(c.get("r", c.get("rank", 0)) or 0))
    return LoRAConfig(
        enable_lm=bool(base | residual), enable_dit=bool(dit), enable_proj=False, max_loras=len(adapters) + 1,
        max_lora_rank=max_rank or max(ranks) or 32, target_modules_lm=_order(base), target_modules_dit=_order(dit),
        target_modules_residual_lm=_order(residual), max_mlp_loras=n_mlp, tag_rows=len(state["dialects"]))


class RoutedDialectTTS:
    """One running engine with the routed adapters and the tagged model's adapter, behind DialectTTS's interface."""

    def __init__(self, server, adapters: list[str], tags: dict, sample_rate: int, routes: dict[str, str] | None = None,
                 tagged_adapter: str = TAGGED_ADAPTER):
        """adapters: the adapters model's adapter names (all of them: the table resolves its names); tags: the tagged
        model's dialects.json; tagged_adapter: its name in the engine."""
        self.server = server
        self.sample_rate = sample_rate
        self.tags = DialectTags(tags)
        self.tagged_adapter = tagged_adapter
        self.table = route_table(sorted(adapters), self.tags, DEFAULT_ROUTES if routes is None else routes)
        self.dialects = sorted(self.table)
        self.voices: dict[str, bool] = {}  # name -> registered with a transcript
        self._voice_src: dict[str, tuple[bytes, str, bool]] = {}  # prompt_ref voices: latents, transcript, normalize

    @classmethod
    async def load(cls, model: str = DEFAULT_MODEL, tagged_model: str = DEFAULT_TAGGED_MODEL,
                   dit: str | None = DEFAULT_DIT, *, routes: dict[str, str] | None = None, max_concurrency: int = 256,
                   head_tokens: int = 8, dit_cache: str | dict | None = None, gpu_memory_utilization: float = 0.85,
                   kv_blocks: int = 600, device: int = 0, w8a8: bool = True,
                   smooth_stats: str | None = None, prefill_tokens: int = 16384) -> "RoutedDialectTTS":
        """Start the engine on ``model`` (adapters/) with the adapters the routes use and ``tagged_model``'s LoRA
        (a local directory or a Hugging Face repo; only its lora/ and tokenizer files are read). One LocDiT ``dit``
        for every request. Other arguments as DialectTTS.load; smooth_stats: SmoothQuant statistics for the int8
        layers (default: the model's)."""
        path_a, path_t = resolve(model), resolve_tagged(tagged_model)
        tags = json.load(open(os.path.join(path_t, "dialects.json"), encoding="utf-8"))
        adapters_dir = os.path.join(path_a, "adapters")
        all_adapters = find_adapters(adapters_dir if os.path.isdir(adapters_dir) else None)
        table = route_table(sorted(all_adapters), DialectTags(tags), DEFAULT_ROUTES if routes is None else routes)
        adapters = {a: all_adapters[a] for a in used_adapters(table)}
        server, sample_rate = await start_engine(
            path_a, dit, {**adapters, TAGGED_ADAPTER: path_t}, routed_lora_config(adapters, path_t),
            max_concurrency=max_concurrency, head_tokens=head_tokens, dit_cache=dit_cache,
            gpu_memory_utilization=gpu_memory_utilization, kv_blocks=kv_blocks, device=device, w8a8=w8a8,
            smooth_stats=smooth_stats, prefill_tokens=prefill_tokens)
        return cls(server, sorted(all_adapters), tags, sample_rate, routes)

    def route(self, dialect: str | None) -> Route:
        """Where ``dialect`` is spoken (None: the base model, as DialectTTS)."""
        if dialect is None:
            return Route(ADAPTERS, None)
        r = self.table.get(dialect)
        if r is None:
            raise ValueError(f"unknown dialect {dialect!r}; dialects: {', '.join(self.dialects)}")
        return r

    async def add_voice(self, name: str, audio: str | bytes, text: str | None = None, audio_format: str = "wav",
                        normalize: bool = True) -> None:
        """As DialectTTS.add_voice; a voice with a transcript gets a copy per tagged dialect on first use."""
        if isinstance(audio, str):
            audio_format = os.path.splitext(audio)[1].lstrip(".") or audio_format
            with open(audio, "rb") as f:
                audio = f.read()
        latents = await self.server.encode_latents(audio, audio_format)
        await self.server.register_voice(name, latents, (read_as(text) if normalize else text) if text else None)
        self.voices[name] = bool(text)
        if text:
            self._voice_src[name] = (latents, text, normalize)

    async def remove_voice(self, name: str) -> None:
        await self.server.unregister_voice(name)
        self.voices.pop(name, None)
        self._voice_src.pop(name, None)
        await self.tags.drop_voice(self.server, name)

    async def stream(self, text: str, dialect: str | None = None, voice: str | None = None, *,
                     clone_mode: str | None = None, seed: int | None = None, cfg_value: float = 2.0,
                     max_patches: int = 600, normalize: bool = True,
                     chunk_ms: float | None = None, pace: float | dict | None = None) -> AsyncIterator[np.ndarray]:
        """As DialectTTS.stream, through the adapter, or the tagged adapter and tag, that ``dialect`` routes to."""
        r = self.route(dialect)
        if normalize:
            text = read_as(text, r.dialect)
        check_text(text, normalize)
        kw = {}
        if voice is not None:
            if voice not in self.voices:
                raise ValueError(f"voice {voice!r} is not registered (add_voice)")
            kw = {"voice": voice, "clone_mode": clone_mode or ("prompt_ref" if self.voices[voice] else "ref")}
        lora = r.dialect
        if r.model == TAGGED:
            lora, tag = self.tagged_adapter, self.tags.tags[r.dialect]
            if kw.get("clone_mode") in ("prompt_ref", "prompt"):  # the transcript comes first: the tag leads it
                latents, transcript, norm = self._voice_src[voice]
                kw["voice"] = await self.tags.voice_copy(self.server, voice, tag, latents,
                                                         read_as(transcript, r.dialect) if norm else transcript)
            else:
                text = tag + text
        if chunk_ms is not None:
            kw["client_chunk_ms"] = chunk_ms
        if pace is not None:
            kw["client_pace"] = pace
        async for chunk in self.server.generate(target_text=text, lora_name=lora, seed=seed, cfg_value=cfg_value,
                                                max_generate_length=max_patches, **kw):
            yield np.multiply(chunk, np.float32(1 / 32768), dtype=np.float32) if chunk.dtype == np.int16 else chunk

    async def synthesize(self, text: str, dialect: str | None = None, voice: str | None = None, **kw) -> np.ndarray:
        """The whole utterance (float32, ``sample_rate``); keyword arguments as for ``stream``."""
        chunks = [c async for c in self.stream(text, dialect, voice, **kw)]
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    async def close(self) -> None:
        await self.server.stop()

    async def __aenter__(self) -> "RoutedDialectTTS":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()
