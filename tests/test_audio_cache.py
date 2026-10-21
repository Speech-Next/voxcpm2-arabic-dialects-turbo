"""The audio cache (--audio-cache-mb, --audio-cache-dir): the same sentence gets the same audio at any load, a
repeated one costs the engine nothing; requests for a sentence in progress share its generation; only complete
generations are kept, least recently used dropped first; a directory keeps them across processes."""

import asyncio
import os

import numpy as np
import pytest

from nanovllm_voxcpm.entrypoints.audio_cache import AudioCache


def chunks(n: int, v: float = 0.1, size: int = 4) -> list[np.ndarray]:
    return [np.full(size, v + k, np.float32) for k in range(n)]


async def collect(agen) -> list[np.ndarray]:
    return [c async for c in agen]


def engine(arrays: list[np.ndarray], calls: list, fail_at: int | None = None, pause: float = 0.0, ended=None):
    """A generation of ``arrays`` (``pause`` s apart); ``calls`` counts them, ``ended`` gets "done" or "closed"."""

    async def generate():
        calls.append(1)
        try:
            for k, a in enumerate(arrays):
                if k == fail_at:
                    raise RuntimeError("the engine failed")
                if pause:
                    await asyncio.sleep(pause)
                yield a
            if ended is not None:
                ended.append("done")
        finally:
            if ended is not None and not ended:
                ended.append("closed")  # (cut: the engine request ends)

    return generate


def test_kept_once_complete_and_played_again():
    cache, calls = AudioCache(1 << 20), []
    first = asyncio.run(collect(cache.through("k", engine(chunks(3), calls))))
    again = asyncio.run(collect(cache.through("k", engine(chunks(3, v=5.0), calls))))  # (another load: other audio)
    assert len(calls) == 1 and len(again) == 3 and all(np.array_equal(a, b) for a, b in zip(first, again))
    assert not again[0].flags.writeable  # (shared by every request that plays it)
    assert cache.stats() == {"hits": 1, "misses": 1, "bytes": 3 * 16, "entries": 1, "disk_hits": 0, "joins": 0}


def test_not_kept_when_cut_or_failed():
    cache, calls, ended = AudioCache(1 << 20), [], []

    async def first_chunk_only():
        agen = cache.through("cut", engine(chunks(3), calls, pause=0.01, ended=ended))
        await agen.__anext__()
        await agen.aclose()  # the client gone after its first chunk, the only one reading it
        await cache.idle()

    asyncio.run(first_chunk_only())
    assert ended == ["closed"] and not cache.inflight
    with pytest.raises(RuntimeError, match="the engine failed"):
        asyncio.run(collect(cache.through("failed", engine(chunks(3), calls, fail_at=2))))
    assert cache.stats()["entries"] == 0 and cache.bytes == 0 and not cache.inflight


def test_requests_at_the_same_time_share_one_generation():
    """Single flight: one engine request, the same chunks for both, the second getting the ones made before it came
    and then the live ones."""
    cache, calls = AudioCache(1 << 20), []

    async def main():
        gen = engine(chunks(6), calls, pause=0.01)
        first = asyncio.create_task(collect(cache.through("k", gen)))
        await asyncio.sleep(0.035)  # (about three chunks made)
        second = await collect(cache.through("k", gen))
        return await first, second

    first, second = asyncio.run(main())
    assert len(calls) == 1 and len(first) == len(second) == 6
    assert all(a is b for a, b in zip(first, second))
    assert cache.stats()["misses"] == 1 and cache.stats()["joins"] == 1 and cache.stats()["entries"] == 1


def test_the_generation_goes_on_while_a_request_still_reads_it():
    cache, calls, ended = AudioCache(1 << 20), [], []

    async def main():
        gen = engine(chunks(6), calls, pause=0.01, ended=ended)
        first = cache.through("k", gen)
        await first.__anext__()
        second = asyncio.create_task(collect(cache.through("k", gen)))
        await asyncio.sleep(0.015)
        await first.aclose()  # the first client gone
        out = await second
        await cache.idle()
        return out

    out = asyncio.run(main())
    assert len(out) == 6 and ended == ["done"] and len(calls) == 1 and cache.stats()["entries"] == 1


def test_every_request_gone_ends_the_generation_and_an_error_reaches_all():
    cache, calls, ended = AudioCache(1 << 20), [], []

    async def gone():
        gen = engine(chunks(6), calls, pause=0.01, ended=ended)
        a, b = cache.through("k", gen), cache.through("k", gen)
        await a.__anext__()
        await b.__anext__()
        await a.aclose()
        await b.aclose()
        await cache.idle()
        # a request after that starts another generation
        return await collect(cache.through("k", engine(chunks(6), calls)))

    assert len(asyncio.run(gone())) == 6 and ended == ["closed"] and len(calls) == 2

    async def failing():
        gen = engine(chunks(4), calls, fail_at=2, pause=0.01)
        return await asyncio.gather(collect(cache.through("bad", gen)), collect(cache.through("bad", gen)),
                                    return_exceptions=True)

    errors = asyncio.run(failing())
    assert [str(e) for e in errors] == ["the engine failed"] * 2 and "bad" not in cache.entries


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
    assert cache.get(0)[0][0] == np.float32(0.1)

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


def test_the_engines_pcm16_kept_in_half_the_memory_played_back_bitwise():
    """DialectTTS.stream gives the engine's int16 audio as float32 (x 1/32768): kept as that int16, half the memory,
    played back as the very same float32."""
    cache, calls = AudioCache(1 << 20), []
    pcm = [np.random.default_rng(k).integers(-32768, 32767, 480).astype(np.int16) for k in range(3)]
    floats = [np.multiply(c, np.float32(1 / 32768), dtype=np.float32) for c in pcm]
    first = asyncio.run(collect(cache.through("k", engine(floats, calls))))
    again = asyncio.run(collect(cache.through("k", engine([], calls))))
    assert cache.bytes == 3 * 480 * 2  # int16
    assert all(a.dtype == np.float32 and a.tobytes() == b.tobytes() for a, b in zip(again, first))
    full = [np.full(4, 1.0, np.float32)]  # 1.0 is no int16 / 32768: kept as float32
    asyncio.run(collect(cache.through("one", engine(full, calls))))
    assert cache.get("one")[0].dtype == np.float32 and cache.get("one")[0][0] == 1.0


def test_a_silent_start_does_not_decide_how_the_sentence_is_kept():
    """Leading silence (zeros) is exact either way: the first sounding chunk decides; float audio that is no PCM16
    stays float (played back bitwise)."""
    cache, calls = AudioCache(1 << 20), []
    audio = [np.zeros(8, np.float32), np.full(8, 0.123456, np.float32)]
    first = asyncio.run(collect(cache.through("k", engine(audio, calls))))
    again = asyncio.run(collect(cache.through("k", engine([], calls))))
    assert all(a.dtype == np.float32 and a.tobytes() == b.tobytes() for a, b in zip(again, first))


def pcm16_floats(n: int, size: int = 480, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [np.multiply(rng.integers(-32768, 32767, size).astype(np.int16), np.float32(1 / 32768),
                        dtype=np.float32) for _ in range(n)]


def test_a_directory_keeps_sentences_for_another_process(tmp_path):
    """--audio-cache-dir: a second cache on the directory (another process, a replica, a restart) plays a sentence
    the first one kept, bitwise, without the engine; another namespace (another model) does not read it."""
    calls = []

    async def first():
        cache = AudioCache(1 << 20, directory=str(tmp_path), namespace="model A")
        out = [await collect(cache.through(("s", k), engine(audio, calls))) for k, audio in
               enumerate([pcm16_floats(3), chunks(2), [np.arange(5, dtype=np.int16)]])]
        await cache.idle()
        return out

    made = asyncio.run(first())
    assert len(os.listdir(tmp_path)) == 3

    async def again(ns):
        cache = AudioCache(1 << 20, directory=str(tmp_path), namespace=ns)
        out = [await collect(cache.through(("s", k), engine(chunks(1, v=7.0), calls))) for k in range(3)]
        await cache.idle()
        return out, cache.stats()

    played, stats = asyncio.run(again("model A"))
    assert len(calls) == 3 and stats["disk_hits"] == 3 and stats["hits"] == 3 and stats["misses"] == 0
    for a, b in zip(made, played):
        assert [c.dtype for c in a] == [c.dtype for c in b] and all(x.tobytes() == y.tobytes() for x, y in zip(a, b))
    _, stats = asyncio.run(again("model B"))
    assert stats["disk_hits"] == 0 and stats["misses"] == 3


def test_the_directory_within_its_size_and_a_bad_file_ignored(tmp_path):
    calls = []

    async def keep(cache, keys):
        for k in keys:
            await collect(cache.through(k, engine(pcm16_floats(2, seed=ord(k)), calls)))
            await cache.idle()

    one = AudioCache(1 << 20, directory=str(tmp_path / "size"))
    asyncio.run(keep(one, ["z"]))
    size = os.path.getsize(one._file("z")[0])
    small = AudioCache(1 << 20, directory=str(tmp_path), dir_max_bytes=int(2.5 * size))  # two files
    for t, k in enumerate("abcde"):  # (each used after the one before)
        asyncio.run(keep(small, [k]))
        os.utime(small._file(k)[0], (1e9 + t * 1e7,) * 2)
    assert {k for k in "abcde" if os.path.exists(small._file(k)[0])} == {"d", "e"}  # (the oldest removed)
    assert small._read("d") is not None  # used now: "e" is the one used longest ago
    asyncio.run(keep(small, ["f"]))
    assert {k for k in "def" if os.path.exists(small._file(k)[0])} == {"d", "f"}

    with open(small._file("f")[0], "r+b") as f:  # cut short
        f.truncate(40)
    fresh = AudioCache(1 << 20, directory=str(tmp_path))
    n = len(calls)
    asyncio.run(keep(fresh, ["f"]))
    assert len(calls) == n + 1 and fresh.stats()["disk_hits"] == 0  # generated again
    assert AudioCache(1 << 20, directory=str(tmp_path))._read("f") is not None  # (and written again, whole)
