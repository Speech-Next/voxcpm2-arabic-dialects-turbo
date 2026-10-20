"""Generated audio kept by what was asked: the same sentence in the same voice, dialect and settings gets the same
audio at any load, and a repeated one costs no GPU (both servers, ``--audio-cache-mb``).

A seed alone does not give that (#47): a head start's split follows the queue, and the batch and the prefix cache
change the engine's float results, so a seeded request sounds different under another load. The cache keeps each
sentence's first complete generation (its chunks as the engine gave them, before resampling or encoding) and plays
it again; beyond max_bytes the least recently used go first. Requests for the same sentence that run at the same
time each generate, and the first one to finish is kept.
"""

from __future__ import annotations

import contextlib
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable, Hashable

import numpy as np


class AudioCache:
    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max(0, int(max_bytes))
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

    async def through(self, key: Hashable, generate: Callable[[], AsyncIterator[np.ndarray]]
                      ) -> AsyncIterator[np.ndarray]:
        """The audio chunks of ``key``: the kept ones (read-only arrays), else ``generate()``'s, kept once it ends
        (not when it fails, or its consumer stops before the end: a client gone)."""
        hit = self.get(key)
        if hit is not None:
            self.hits += 1
            for c in hit:
                yield c
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
