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


class AudioCache:
    LEAD_S = 1.0  # seconds of a kept sentence played at once, the rest at most ``speed`` x real time

    def __init__(self, max_bytes: int, sample_rate: int = 48000) -> None:
        self.max_bytes = max(0, int(max_bytes))
        self.sample_rate = sample_rate  # the chunks' (the engine's)
        self.entries: OrderedDict[Hashable, tuple[np.ndarray, ...]] = OrderedDict()
        self.bytes = 0
        self.hits = self.misses = 0  # sentences played from the cache, generated

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses, "bytes": self.bytes, "entries": len(self.entries)}

    def get(self, key: Hashable) -> tuple[np.ndarray, ...] | None:
        chunks = self.entries.get(key)
        if chunks is not None:
            self.entries.move_to_end(key)
        return chunks

    def put(self, key: Hashable, chunks: list[np.ndarray]) -> None:
        """Kept, unless it is kept already (the first generation stays) or larger than a quarter of the cache."""
        size = sum(c.nbytes for c in chunks)
        if key in self.entries or size > self.max_bytes // 4:
            return
        self.entries[key] = tuple(chunks)
        self.bytes += size
        while self.bytes > self.max_bytes:
            _, old = self.entries.popitem(last=False)
            self.bytes -= sum(c.nbytes for c in old)

    async def through(self, key: Hashable, generate: Callable[[], AsyncIterator[np.ndarray]],
                      speed: float | None = None) -> AsyncIterator[np.ndarray]:
        """The audio chunks of ``key``: the kept ones (read-only arrays; after the first LEAD_S seconds at most
        ``speed`` x real time, None: as fast as they are taken, the event loop served between them), else
        ``generate()``'s, kept once it ends (not when it fails, or its consumer stops before the end: a client
        gone)."""
        hit = self.get(key)
        if hit is not None:
            self.hits += 1
            t0, played = time.perf_counter(), 0.0
            for c in hit:
                yield c
                played += c.shape[-1] / self.sample_rate
                ahead = (played - self.LEAD_S) / speed - (time.perf_counter() - t0) if speed else 0.0
                await asyncio.sleep(max(ahead, 0.0))
            return
        self.misses += 1
        kept: list[np.ndarray] | None = []
        size = 0
        async with contextlib.aclosing(generate()) as gen:
            async for c in gen:
                if kept is not None:
                    c = np.array(c)  # (a copy: the engine's chunk may be a view of a buffer it reuses)
                    c.setflags(write=False)
                    size += c.nbytes
                    if size <= self.max_bytes // 4:
                        kept.append(c)
                    else:  # (too long to keep)
                        kept = None
                yield c
        if kept is not None:
            self.put(key, kept)
