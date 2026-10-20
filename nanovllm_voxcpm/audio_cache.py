"""Generated audio kept by what was asked: the same sentence in the same voice, dialect and settings gets the same
audio at any load, and a repeated one costs no GPU (both servers, ``--audio-cache-mb``).

A seed alone does not give that (#47): a head start's split follows the queue, and the batch and the prefix cache
change the engine's float results, so a seeded request sounds different under another load. The cache keeps each
sentence's first complete generation (its chunks as the engine gave them, before resampling or encoding) and plays
it again; beyond max_bytes the least recently used go first. Requests for the same sentence that run at the same
time each generate, and the first one to finish is kept. A kept sentence plays its first second at once and the rest
at most ``speed`` x real time (when asked): 150 kept sentences at once, all their audio in one go, held the event loop
that also serves the engine and paces the streams (first chunk p99 0.51 s instead of 0.32 s at 150 at once).
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable, Hashable

import numpy as np

_INT16_SCALE = np.float32(1 / 32768)


def _is_pcm16(c: np.ndarray) -> bool:
    """Whether ``c`` is the engine's PCM16 as float32 (DialectTTS.stream: int16 / 32768): kept as that int16, in half
    the memory, _unpacked gives back the very same float32."""
    return c.dtype == np.float32 and np.array_equal(np.multiply(c, 32768, dtype=np.float32).astype(np.int16),
                                                    np.multiply(c, 32768, dtype=np.float32))


def _packed(c: np.ndarray, pcm16: bool) -> np.ndarray:
    """A chunk as kept (a new array: the engine's may be a view of a buffer it reuses)."""
    return np.multiply(c, 32768, dtype=np.float32).astype(np.int16) if pcm16 else np.array(c)


def _unpacked(c: np.ndarray, dtype: np.dtype) -> np.ndarray:
    return np.multiply(c, _INT16_SCALE, dtype=np.float32) if dtype == np.float32 and c.dtype == np.int16 else c


class AudioCache:
    LEAD_S = 1.0  # seconds of a kept sentence played at once, the rest at most ``speed`` x real time

    def __init__(self, max_bytes: int, sample_rate: int = 48000) -> None:
        self.max_bytes = max(0, int(max_bytes))
        self.sample_rate = sample_rate  # the chunks' (the engine's)
        # key -> (the chunks' dtype as generated, the chunks as kept: _packed, read-only)
        self.entries: OrderedDict[Hashable, tuple[np.dtype, tuple[np.ndarray, ...]]] = OrderedDict()
        self.bytes = 0
        self.hits = self.misses = 0  # sentences played from the cache, generated

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses, "bytes": self.bytes, "entries": len(self.entries)}

    def _kept(self, key: Hashable) -> tuple[np.dtype, tuple[np.ndarray, ...]] | None:
        entry = self.entries.get(key)
        if entry is not None:
            self.entries.move_to_end(key)
        return entry

    def get(self, key: Hashable) -> list[np.ndarray] | None:
        """The kept chunks of ``key`` as they were generated, or None."""
        entry = self._kept(key)
        return None if entry is None else [_unpacked(c, entry[0]) for c in entry[1]]

    def put(self, key: Hashable, chunks: list[np.ndarray], dtype: np.dtype | None = None) -> None:
        """Kept (``chunks`` _packed already when ``dtype`` is given: the dtype they were generated in), unless it is
        kept already (the first generation stays) or larger than a quarter of the cache."""
        if dtype is None:
            dtype = chunks[0].dtype if chunks else np.dtype(np.float32)
            chunks = [_packed(c, _is_pcm16(c)) for c in chunks]
        size = sum(c.nbytes for c in chunks)
        if key in self.entries or size > self.max_bytes // 4:
            return
        for c in chunks:
            c.setflags(write=False)
        self.entries[key] = (dtype, tuple(chunks))
        self.bytes += size
        while self.bytes > self.max_bytes:
            _, (_, old) = self.entries.popitem(last=False)
            self.bytes -= sum(c.nbytes for c in old)

    async def through(self, key: Hashable, generate: Callable[[], AsyncIterator[np.ndarray]],
                      speed: float | None = None) -> AsyncIterator[np.ndarray]:
        """The audio chunks of ``key``: the kept ones (after the first LEAD_S seconds at most ``speed`` x real time,
        None: as fast as they are taken, the event loop served between them), else ``generate()``'s, kept once it
        ends (not when it fails, or its consumer stops before the end: a client gone)."""
        hit = self._kept(key)
        if hit is not None:
            self.hits += 1
            dtype, chunks = hit
            t0, played = time.perf_counter(), 0.0
            for c in chunks:  # (each unpacked as it goes: 150 sentences at once, all at the start, held the loop)
                yield _unpacked(c, dtype)
                played += c.shape[-1] / self.sample_rate
                ahead = (played - self.LEAD_S) / speed - (time.perf_counter() - t0) if speed else 0.0
                await asyncio.sleep(max(ahead, 0.0))
            return
        self.misses += 1
        kept: list[np.ndarray] | None = []
        size, dtype, pcm16 = 0, None, None
        async with contextlib.aclosing(generate()) as gen:
            async for c in gen:
                if kept is not None:
                    dtype = dtype or c.dtype
                    if pcm16 is None and c.any():  # (the first sounding chunk tells the rest's form: one engine path)
                        pcm16 = _is_pcm16(c)
                    p = _packed(c, pcm16 is not False)  # (silence before it: int16 zeros, the same either way)
                    size += p.nbytes
                    if size <= self.max_bytes // 4:
                        kept.append(p)
                    else:  # (too long to keep)
                        kept = None
                yield c
        if kept is not None:
            self.put(key, kept, dtype)
