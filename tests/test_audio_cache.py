"""The audio cache (--audio-cache-mb): the same sentence gets the same audio at any load, a repeated one costs the
engine nothing; only complete generations are kept, least recently used dropped first."""

import asyncio

import numpy as np
import pytest

from nanovllm_voxcpm.audio_cache import AudioCache


def chunks(n: int, v: float = 0.1, size: int = 4) -> list[np.ndarray]:
    return [np.full(size, v + k, np.float32) for k in range(n)]


async def collect(agen) -> list[np.ndarray]:
    return [c async for c in agen]


def engine(arrays: list[np.ndarray], calls: list, fail_at: int | None = None):
    async def generate():
        calls.append(1)
        for k, a in enumerate(arrays):
            if k == fail_at:
                raise RuntimeError("the engine failed")
            yield a

    return generate


def test_kept_once_complete_and_played_again():
    cache, calls = AudioCache(1 << 20), []
    first = asyncio.run(collect(cache.through("k", engine(chunks(3), calls))))
    again = asyncio.run(collect(cache.through("k", engine(chunks(3, v=5.0), calls))))  # (another load: other audio)
    assert len(calls) == 1 and len(again) == 3 and all(np.array_equal(a, b) for a, b in zip(first, again))
    assert not again[0].flags.writeable  # (shared by every request that plays it)
    assert cache.stats() == {"hits": 1, "misses": 1, "bytes": 3 * 16, "entries": 1}


def test_not_kept_when_cut_or_failed():
    cache, calls = AudioCache(1 << 20), []

    async def first_chunk_only():
        agen = cache.through("cut", engine(chunks(3), calls))
        await agen.__anext__()
        await agen.aclose()  # the client gone after its first chunk

    asyncio.run(first_chunk_only())
    with pytest.raises(RuntimeError):
        asyncio.run(collect(cache.through("failed", engine(chunks(3), calls, fail_at=2))))
    assert cache.stats()["entries"] == 0 and cache.bytes == 0


def test_least_recently_used_go_first_and_large_ones_are_not_kept():
    cache = AudioCache(4 * 64)  # 4 entries of 64 bytes; an entry at most a quarter
    for k in range(4):
        cache.put(k, chunks(1, size=16))
    cache.get(0)  # 0 played again: 1 is now the least recently used
    cache.put(4, chunks(1, size=16))
    assert list(cache.entries) == [2, 3, 0, 4] and cache.bytes == 4 * 64
    cache.put(5, chunks(1, size=17))  # larger than a quarter of the cache
    assert 5 not in cache.entries
    cache.put(0, chunks(1, v=9.0, size=16))  # the first generation stays
    assert cache.entries[0][0][0] == np.float32(0.1)

    big, calls = AudioCache(4 * 64), []  # too long to keep: played through, not kept
    out = asyncio.run(collect(big.through("long", engine(chunks(5, size=16), calls))))
    assert len(out) == 5 and big.stats()["entries"] == 0


def test_a_kept_sentence_plays_its_first_second_at_once_then_at_most_speed():
    """150 kept sentences at once, all their audio in one go, held the event loop that serves the engine and paces
    the streams: after LEAD_S, at most ``speed`` x real time (the event loop served between the chunks)."""
    import time

    cache, calls = AudioCache(1 << 20, sample_rate=1000), []
    asyncio.run(collect(cache.through("k", engine(chunks(20, size=100), calls))))  # 2 s of audio, kept

    async def timed(speed):
        t0, at = time.perf_counter(), []
        async for _ in cache.through("k", engine([], calls), speed=speed):
            at.append(time.perf_counter() - t0)
        return at, time.perf_counter() - t0

    at, total = asyncio.run(timed(4.0))
    assert at[9] < 0.05  # the first second at once
    assert 0.2 < total < 0.6  # the second second at 4 x: 0.25 s
    _, fast = asyncio.run(timed(None))
    assert fast < 0.05 and len(calls) == 1
