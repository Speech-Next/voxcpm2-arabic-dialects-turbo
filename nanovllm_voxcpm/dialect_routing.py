"""Per-dialect routing (opt-in): two dialect models in one process, each request spoken by the one that serves its
dialect better.

The adapters model (``adapters/<dialect>/``, the default VoxCPM-Kernel/VoxCPM2-Arabic-Dialects) and a tagged model
(``dialects.json``, VoxCPM-Kernel/VoxCPM2-Arabic-Teacher-v3) run as two engines on the same GPU. A dialect named in
``routes`` goes to the adapters model with its adapter, every other dialect of the tagged model to the tagged model
with its tag, and a name only the adapters model knows (hindi_english) to the adapters model::

    tts = await RoutedDialectTTS.load(dit=None)                          # both models, quality mode (10 steps)
    await tts.add_voice("speaker", "reference.wav")                      # registered with both engines
    audio = await tts.synthesize("شلونك؟", dialect="bahraini", voice="speaker")  # adapters model, qatari adapter
    audio = await tts.synthesize("هلا والله", dialect="najdi", voice="speaker")  # teacher-v3, its najdi tag
    await tts.close()

Default routes, from #2 (the held-out set, 17 groups): teacher-v3 is better pooled but worse than the adapters on
Bahraini, Egyptian and Saudi. Names resolve as below; ``tts.table`` holds the result.

| name | model | adapter / tag |
|---|---|---|
| bahraini | adapters | qatari |
| egyptian | adapters | egyptian |
| saudi | adapters | saudi |
| hindi_english | adapters | hindi_english |
| None (no dialect) | adapters | none: the base model, as DialectTTS |
| msa, lebanese, syrian, iraqi, najdi, hejazi | teacher-v3 | its own tag |
| yemeni, sudanese, libyan, tunisian, algerian, moroccan | teacher-v3 | its own tag |
| gulf, emirati, qatari, khaliji | teacher-v3 | gulf |
| palestinian, jordanian, ab, levantine | teacher-v3 | palestinian |

An alias follows the tagged model's dialects.json (qatari, emirati and khaliji are gulf there; jordanian, ab and
levantine palestinian), so "qatari" is Gulf Arabic on teacher-v3, and the qatari adapter speaks "bahraini" (the #2
Bahraini group was measured with it). A route keyed by a name wins over one keyed by its dialect:
``routes={**DEFAULT_ROUTES, "qatari": "qatari"}`` sends "qatari" alone to the qatari adapter.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from dataclasses import dataclass
from typing import AsyncIterator

import numpy as np

from nanovllm_voxcpm.dialect_tts import DEFAULT_DIT, DEFAULT_MODEL, DialectTags, DialectTTS, find_adapters, resolve

DEFAULT_TAGGED_MODEL = "VoxCPM-Kernel/VoxCPM2-Arabic-Teacher-v3"
# fast mode on the tagged model until it has a distilled LocDiT of its own (#7), chosen on #2
DEFAULT_TAGGED_DIT = DEFAULT_DIT
# dialect (or name) -> adapter: where the adapters model beats teacher-v3 on #2
DEFAULT_ROUTES = {"bahraini": "qatari", "egyptian": "egyptian", "saudi": "saudi"}
ADAPTERS, TAGGED = "adapters", "tagged"
AUTO = "auto"
# read by an engine's process when it starts (DialectTTS.load sets them): restored after each start, so that the
# next engine does not inherit the previous one's (a model's SmoothQuant statistics, its KV blocks)
_ENGINE_ENV = ("NANOVLLM_VOXCPM2_TURBO", "NANOVLLM_SERVERPOOL_NUM_KVCACHE_BLOCKS", "VOXCPM_KERNELS_SMOOTH")


@dataclass(frozen=True)
class Route:
    model: str  # ADAPTERS or TAGGED
    dialect: str | None  # what that model's DialectTTS takes: an adapter, a tagged dialect; None = the base model


def route_table(adapters: list[str], tags: DialectTags, routes: dict[str, str]) -> dict[str, Route]:
    """name -> Route for every name of both models (and of ``routes``). A name is looked up in ``routes`` first, then
    its tagged dialect (aliases resolved): found, the adapters model with that adapter; else the tagged model when
    it knows the name; else the adapters model, the name being one of its adapters."""
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


def split_kv_blocks(kv_blocks: int | tuple[int, int]) -> tuple[int, int]:
    """(adapters, tagged) KV-cache blocks: a pair as is, a total split evenly (either engine may get a whole burst)."""
    a, t = (kv_blocks - kv_blocks // 2, kv_blocks // 2) if isinstance(kv_blocks, int) else kv_blocks
    if a <= 0 or t <= 0:
        raise ValueError("kv_blocks: both engines need blocks; automatic sizing (0) would give the first engine "
                         "all the free memory")
    return a, t


@contextlib.contextmanager
def _engine_env():
    saved = {k: os.environ.get(k) for k in _ENGINE_ENV}
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class RoutedDialectTTS:
    """Two running DialectTTS engines (an adapters model and a tagged model) behind DialectTTS's interface."""

    def __init__(self, adapters: DialectTTS, tagged: DialectTTS, routes: dict[str, str] | None = None):
        if adapters.tags is not None or tagged.tags is None:
            raise ValueError("RoutedDialectTTS takes an adapters model first and a tagged model (dialects.json) second")
        if adapters.sample_rate != tagged.sample_rate:
            raise ValueError(f"sample rates differ: {adapters.sample_rate} vs {tagged.sample_rate}")
        self.engines = {ADAPTERS: adapters, TAGGED: tagged}
        self.sample_rate = adapters.sample_rate
        self.table = route_table(adapters.dialects, tagged.tags, DEFAULT_ROUTES if routes is None else routes)
        self.dialects = sorted(self.table)
        self.voices: dict[str, bool] = {}  # name -> registered with a transcript

    @classmethod
    async def load(cls, model: str = DEFAULT_MODEL, tagged_model: str = DEFAULT_TAGGED_MODEL,
                   dit: str | None = DEFAULT_DIT, tagged_dit: str | None = AUTO, *,
                   routes: dict[str, str] | None = None, max_concurrency: int = 256, head_tokens: int = 8,
                   dit_cache: str | dict | None = None, gpu_memory_utilization: float = 0.85,
                   kv_blocks: int | tuple[int, int] = 600, device: int = 0, w8a8: bool = True) -> "RoutedDialectTTS":
        """Start both engines, one after the other, on ``device``.

        dit: the adapters model's distilled LocDiT (None = its own, 10 steps: quality mode).
        tagged_dit: the tagged model's; "auto": None with dit=None, else DEFAULT_TAGGED_DIT.
        kv_blocks: the KV cache, a total split evenly or (adapters, tagged); a block is 256 tokens (~9 MB).
        max_concurrency: per engine (a burst may go to one). Other arguments as DialectTTS.load, for both.
        """
        if tagged_dit == AUTO:
            tagged_dit = None if dit is None else DEFAULT_TAGGED_DIT
        kv_a, kv_t = split_kv_blocks(kv_blocks)
        path_a, path_t = resolve(model), resolve(tagged_model)
        tags = DialectTags.of_model(path_t)
        if tags is None or DialectTags.of_model(path_a) is not None:
            raise ValueError(f"model must have adapters/ ({model}) and tagged_model a dialects.json ({tagged_model})")
        adapters = find_adapters(os.path.join(path_a, "adapters") if os.path.isdir(os.path.join(path_a, "adapters"))
                                 else None)
        route_table(sorted(adapters), tags, DEFAULT_ROUTES if routes is None else routes)  # fails before starting
        common = dict(max_concurrency=max_concurrency, head_tokens=head_tokens,
                      gpu_memory_utilization=gpu_memory_utilization, device=device, w8a8=w8a8)
        started = []
        try:
            for path, d, kv in ((path_a, dit, kv_a), (path_t, tagged_dit, kv_t)):
                with _engine_env():
                    started.append(await DialectTTS.load(path, d, dit_cache=None if d else dit_cache, kv_blocks=kv,
                                                         **common))
            return cls(*started, routes=routes)
        except BaseException:
            await asyncio.gather(*(t.close() for t in started), return_exceptions=True)
            raise

    def route(self, dialect: str | None) -> Route:
        """Where ``dialect`` is spoken (None: the adapters model's base model, as DialectTTS)."""
        if dialect is None:
            return Route(ADAPTERS, None)
        r = self.table.get(dialect)
        if r is None:
            raise ValueError(f"unknown dialect {dialect!r}; dialects: {', '.join(self.dialects)}")
        return r

    async def add_voice(self, name: str, audio: str | bytes, text: str | None = None, audio_format: str = "wav",
                        normalize: bool = True) -> None:
        """Register a voice with both engines (arguments as DialectTTS.add_voice)."""
        if isinstance(audio, str):
            audio_format = os.path.splitext(audio)[1].lstrip(".") or audio_format
            with open(audio, "rb") as f:
                audio = f.read()
        engines = list(self.engines.values())
        res = await asyncio.gather(*(e.add_voice(name, audio, text, audio_format, normalize) for e in engines),
                                   return_exceptions=True)
        errors = [r for r in res if isinstance(r, BaseException)]
        if errors:  # not half registered: a request routed to the other engine would fail later
            await asyncio.gather(*(e.remove_voice(name) for e, r in zip(engines, res) if r is None),
                                 return_exceptions=True)
            raise errors[0]
        self.voices[name] = bool(text)

    async def remove_voice(self, name: str) -> None:
        await asyncio.gather(*(e.remove_voice(name) for e in self.engines.values()))
        self.voices.pop(name, None)

    async def stream(self, text: str, dialect: str | None = None, voice: str | None = None, *,
                     clone_mode: str | None = None, seed: int | None = None, cfg_value: float = 2.0,
                     max_patches: int = 600, normalize: bool = True) -> AsyncIterator[np.ndarray]:
        """As DialectTTS.stream, on the engine and adapter or tag that ``dialect`` routes to."""
        r = self.route(dialect)
        async for chunk in self.engines[r.model].stream(text, r.dialect, voice, clone_mode=clone_mode, seed=seed,
                                                        cfg_value=cfg_value, max_patches=max_patches,
                                                        normalize=normalize):
            yield chunk

    async def synthesize(self, text: str, dialect: str | None = None, voice: str | None = None, **kw) -> np.ndarray:
        """The whole utterance (float32, ``sample_rate``); keyword arguments as for ``stream``."""
        chunks = [c async for c in self.stream(text, dialect, voice, **kw)]
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    async def close(self) -> None:
        await asyncio.gather(*(e.close() for e in self.engines.values()))

    async def __aenter__(self) -> "RoutedDialectTTS":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()
