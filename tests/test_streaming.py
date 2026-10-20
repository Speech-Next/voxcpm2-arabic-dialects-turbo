"""The paced chunker (nanovllm_voxcpm/streaming.py, #31) on a fake clock: the first chunk at once, the grid, a burst
then a pause, a slow producer, the end of a stream, the reserve; and paced() on a real event loop."""

import asyncio
import time

import pytest

from nanovllm_voxcpm.streaming import ChunkPacer, paced

C = 4096  # 128 ms of 16 kHz PCM16
CS = 0.128


def run(pacer: ChunkPacer, pushes: list[tuple[float, int]], end: float | None) -> list[tuple[float, int]]:
    """Replay pushes (time, bytes) and the end in time order; (send time, bytes) of every chunk."""
    sends, i, ended = [], 0, False
    while not pacer.done:
        due = pacer.due()
        nxt = pushes[i][0] if i < len(pushes) else None
        if nxt is None and not ended:
            assert end is not None, "waiting for audio that never comes"
            nxt = end
        if due is not None and (nxt is None or due <= nxt):
            sends.append((due, len(pacer.pop())))
        elif i < len(pushes):
            pacer.push(b"\0" * pushes[i][1], pushes[i][0])
            i += 1
        else:
            pacer.finish(end)
            ended = True
    return sends


def gaps(sends):
    return [b[0] - a[0] for a, b in zip(sends, sends[1:])]


def test_first_chunk_at_once_then_the_grid():
    s = run(ChunkPacer(C, 32000, alpha=0.95, reserve_ms=None), [(0.1, 5 * C)], end=5.0)
    assert s[0] == (0.1, C)
    assert gaps(s) == pytest.approx([0.95 * CS] * 4)


def test_a_partial_first_chunk_waits_for_its_bytes():
    p = ChunkPacer(C, 32000, reserve_ms=None)
    p.push(b"\0" * 1000, 0.05)
    assert p.due() is None  # not complete, stream not ended
    p.push(b"\0" * 3096, 0.07)
    assert p.due() == 0.07


def test_burst_then_pause_then_late_chunk_restarts_the_grid():
    # 5 chunks at once, then nothing until 2.0 s: the server spends its lead at the pace, the late chunk goes out
    # when it comes and the next one is due a paced gap after it
    s = run(ChunkPacer(C, 32000, alpha=0.95, reserve_ms=None), [(0.0, 5 * C), (2.0, 2 * C)], end=9.0)
    t = [x[0] for x in s]
    assert t[:5] == pytest.approx([0.95 * CS * k for k in range(5)])
    assert t[5] == 2.0
    assert t[6] == pytest.approx(2.0 + 0.95 * CS)


def test_slow_producer_each_chunk_on_arrival():
    pushes = [(0.2 * k, C) for k in range(6)]
    s = run(ChunkPacer(C, 32000, reserve_ms=None), pushes, end=1.2)
    assert [x[0] for x in s] == pytest.approx([0.2 * k for k in range(6)])


def test_end_of_stream_flushes_the_rest_with_the_partial_chunk():
    # 4.5 chunks at once, the stream ends at 0.1: chunk 0 at 0, chunk 1 on the grid (0.1216), the rest at once
    s = run(ChunkPacer(C, 32000, alpha=0.95, reserve_ms=None), [(0.0, 4 * C + C // 2)], end=0.1)
    assert [x[1] for x in s] == [C, C, C, C, C // 2]
    t = [x[0] for x in s]
    assert t[0] == 0.0 and t[1] == pytest.approx(0.1) and t[2:] == pytest.approx([0.1] * 3)
    s = run(ChunkPacer(C, 32000, alpha=0.95, reserve_ms=None), [(0.0, 3 * C + 10)], end=1.0)
    t = [x[0] for x in s]
    assert t[:3] == pytest.approx([0, 0.95 * CS, 2 * 0.95 * CS]) and t[3] == pytest.approx(1.0)
    assert s[-1][1] == 10


def test_a_stream_shorter_than_a_chunk():
    s = run(ChunkPacer(C, 32000), [(0.0, 1000), (0.01, 500)], end=0.02)
    assert s == [(0.02, 1500)]
    p = ChunkPacer(C, 32000)
    p.finish(0.0)
    assert p.done and p.due() is None


def test_reserve_releases_what_is_beyond_it_at_once():
    # 20 chunks at once, a 1 s reserve (7.8 chunks): chunks go out at once while more than chunk + reserve is queued
    # (12 of them), the last 8 at the pace
    s = run(ChunkPacer(C, 32000, alpha=0.95, reserve_ms=1000), [(0.0, 20 * C)], end=10.0)
    t = [x[0] for x in s]
    assert t[:12] == [0.0] * 12
    assert gaps(s)[11:] == pytest.approx([0.95 * CS] * 8)


def test_reserve_release_from_audio_that_arrives_later():
    # chunk 1 is waiting for its grid time when a burst arrives that puts more than the reserve behind it
    p = ChunkPacer(C, 32000, alpha=0.95, reserve_ms=256)  # 2 chunks
    p.push(b"\0" * 2 * C, 0.0)
    assert p.due() == 0.0
    p.pop()
    assert p.due() == pytest.approx(0.95 * CS)
    p.push(b"\0" * 3 * C, 0.05)  # now 4 chunks queued: 3 behind chunk 1 > 2 chunks of reserve
    assert p.due() == pytest.approx(0.05)


def test_the_grid_does_not_drift_when_pops_come_late():
    p = ChunkPacer(C, 32000, alpha=0.95, reserve_ms=None)
    p.push(b"\0" * 6 * C, 0.0)
    planned = []
    for _ in range(6):
        planned.append(p.due())
        p.pop()  # a busy event loop pops each one late: the grid keeps the planned times
    assert planned == pytest.approx([0.95 * CS * k for k in range(6)])


def test_dry_at():
    p = ChunkPacer(C, 32000, alpha=0.95, reserve_ms=None)
    assert p.dry_at() is None
    p.push(b"\0" * (3 * C + 100), 0.0)
    p.pop()
    # chunk 0 was due at 0; 2 more complete chunks queued: the next incomplete one is due 3 paced gaps later
    assert p.dry_at() == pytest.approx(3 * 0.95 * CS)


def test_paced_on_a_real_loop():
    async def main():
        async def source():
            for k in range(12):  # 12 chunks of 10 ms audio in 3 bursts, then the end after the pace is through
                yield bytes([k]) * 320
                if k % 4 == 3:
                    await asyncio.sleep(0.02)
            await asyncio.sleep(0.2)

        pacer = ChunkPacer(chunk_bytes=320, bytes_per_s=32000, alpha=0.9, reserve_ms=None)
        out, ts = [], []
        async for chunk in paced(source(), pacer):
            out.append(chunk)
            ts.append(time.perf_counter())
        return out, ts

    out, ts = asyncio.run(main())
    assert out == [bytes([k]) * 320 for k in range(12)]
    g = gaps([(t, 0) for t in ts])
    assert min(g) > 0.005 and max(g) < 0.05  # the grid's 9 ms, give or take the loop


def test_paced_passes_on_a_source_error():
    async def main():
        async def source():
            yield b"\0" * 5000
            raise ValueError("engine failed")

        async for _ in paced(source(), ChunkPacer(4096, 32000)):
            pass

    with pytest.raises(ValueError, match="engine failed"):
        asyncio.run(main())


def test_paced_closes_its_source_when_the_consumer_stops():
    closed = []

    async def main():
        async def source():
            try:
                for _ in range(100):
                    yield b"\0" * 4096
                    await asyncio.sleep(0.001)
            finally:
                closed.append(True)

        gen = paced(source(), ChunkPacer(4096, 32000, reserve_ms=None))
        async for _ in gen:
            break  # the client went away
        await gen.aclose()

    asyncio.run(main())
    assert closed == [True]


def test_alpha_last_slows_the_chunk_that_empties_the_server():
    # 2 chunks at once: chunk 1 is the last one queued -> alpha_last; with a third queued it goes at alpha
    s = run(ChunkPacer(C, 32000, alpha=0.85, reserve_ms=None, alpha_last=0.97), [(0.0, 2 * C)], end=5.0)
    assert s[1][0] == pytest.approx(0.97 * CS)
    s = run(ChunkPacer(C, 32000, alpha=0.85, reserve_ms=None, alpha_last=0.97), [(0.0, 3 * C)], end=5.0)
    assert [x[0] for x in s] == pytest.approx([0, 0.85 * CS, 0.85 * CS + 0.97 * CS])


def test_alpha_zero_is_unpaced():
    s = run(ChunkPacer(C, 32000, alpha=0.0, reserve_ms=None), [(0.0, 3 * C), (0.5, C + 10)], end=0.6)
    assert s == [(0.0, C), (0.0, C), (0.0, C), (0.5, C), (0.6, 10)]


def test_chunk_stream_pushed_from_callbacks():
    from nanovllm_voxcpm.streaming import ChunkStream

    async def main():
        stream = ChunkStream(ChunkPacer(chunk_bytes=320, bytes_per_s=32000, alpha=0.9, reserve_ms=None))
        loop = asyncio.get_running_loop()
        for k in range(6):  # 6 chunks of 10 ms: two bursts from plain callbacks, then the end
            loop.call_later(0.0 if k < 3 else 0.02, stream.push, bytes([k]) * 320)
        loop.call_later(0.2, stream.finish)
        return [c async for c in stream]

    assert asyncio.run(main()) == [bytes([k]) * 320 for k in range(6)]


def test_stream_sender_batches_streams_in_order_and_finishes_them():
    from nanovllm_voxcpm.streaming import ChunkStream, StreamSender

    async def main():
        sender = StreamSender(window_s=0.005)
        sent = []
        streams = [ChunkStream(ChunkPacer(chunk_bytes=320, bytes_per_s=32000, alpha=0.9, reserve_ms=None))
                   for _ in range(3)]
        futs = [sender.add(s, lambda b, i=i: sent.append((i, b[0], time.perf_counter()))) for i, s in enumerate(streams)]
        for i, s in enumerate(streams):  # 3 chunks each, all at once
            s.push(bytes([i * 10 + k for k in range(3) for _ in range(320)]))
            s.finish()
        failing = ChunkStream(ChunkPacer(chunk_bytes=320, bytes_per_s=32000, alpha=0.9, reserve_ms=None))
        f_fail = sender.add(failing, lambda b: (_ for _ in ()).throw(ConnectionError("gone")))
        failing.push(b"\0" * 320)
        await asyncio.gather(*futs)
        with pytest.raises(ConnectionError):
            await f_fail
        return sent

    sent = asyncio.run(main())
    assert [(i, b) for i, b, _ in sent if b % 10 == 0] == [(0, 0), (1, 10), (2, 20)]  # first chunks, in order
    for i in range(3):
        assert [b for j, b, _ in sent if j == i] == [i * 10, i * 10 + 1, i * 10 + 2]


def test_min_gap_spreads_the_end_flush_and_the_reserve_release():
    s = run(ChunkPacer(C, 32000, alpha=0.95, reserve_ms=None, min_gap_ms=8), [(0.0, 4 * C + C // 2)], end=0.1)
    t = [x[0] for x in s]
    assert t[0] == 0.0 and t[1] == pytest.approx(0.1) and gaps(s)[1:] == pytest.approx([0.008] * 3)
    s = run(ChunkPacer(C, 32000, alpha=0.95, reserve_ms=256, min_gap_ms=8), [(0.0, 10 * C)], end=9.0)
    assert min(gaps(s)) == pytest.approx(0.008)


def test_start_delays_the_first_chunk_and_the_grid_with_it():
    s = run(ChunkPacer(C, 32000, alpha=0.95, reserve_ms=None, start_ms=100), [(0.1, 5 * C)], end=5.0)
    assert s[0] == (pytest.approx(0.2), C)
    assert gaps(s) == pytest.approx([0.95 * CS] * 4)
    p = ChunkPacer(C, 32000, alpha=0.9, reserve_ms=None, start_ms=100)
    p.push(b"\0" * 2 * C, 0.0)
    # the engine's deadline, 100 ms later as well, and already while the first chunk waits (the same as after it)
    assert p.due() == pytest.approx(0.1) and p.dry_at() == pytest.approx(0.1 + 0.9 * 2 * CS)
    p.pop()
    assert p.dry_at() == pytest.approx(0.1 + 0.9 * 2 * CS)
