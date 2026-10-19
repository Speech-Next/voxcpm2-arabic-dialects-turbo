"""Paced delivery of a stream's audio in fixed-size chunks: a jitter buffer on the server (#31).

The engine delivers a stream's audio in bursts: in a burst of requests it pauses the streams that have audio to
spare while it prefills the others, then decodes them again. A front that sends every chunk the moment it is complete
passes the bursts on: the client sees gaps between chunks longer than a chunk's audio although its player never runs
dry (#28's WebSocket benchmark counts each such gap as a stall: ~2,400 at 200 requests at once on one A10). Sent
through a ChunkPacer the same audio arrives at an even pace: the server keeps the lead instead of the client, and a
gap longer than a chunk only remains where the engine is late with the audio itself (the server has nothing left)::

    pacer = ChunkPacer(chunk_bytes=4096, bytes_per_s=32000)  # 128 ms chunks of 16 kHz PCM16
    async for chunk in paced(pcm_bytes(stream), pacer):
        await ws.send(chunk)

Policy (times from the stream's own audio, so that the pace does not drift):
- the first chunk goes out as soon as it is complete (time to first audio as without pacing), or ``start_ms`` after
  that (a jitter buffer's start: every later chunk, and the engine's deadlines, that much later);
- chunk k is due ``alpha`` times chunk k-1's audio after chunk k-1 was due: a grid, so that a late send (the event
  loop busy) does not shift the chunks after it. A chunk complete before it is due waits; one completed after it is
  sent at once and the grid restarts there;
- while more than ``reserve_ms`` of audio waits behind a chunk, it goes out at once (and the grid restarts): the
  server keeps no more lead than that, the client gets the rest as it comes (time to the last chunk stays short when
  the engine is ahead);
- when the stream has ended, what is left goes out at once.

``alpha`` < 1: a gap is a stall when it is longer than the request's average chunk (its audio over its number of
chunks: below a whole chunk's audio by the shorter last one, most for short utterances). ``expect_s``, a lower bound
of the utterance's audio (e.g. from its text), lowers the pace's ratio for short ones so that a gap stays
``margin_ms`` (the event loop's delays) under the shortest such average (``unit_bytes``: the audio's granularity). The pace then runs 1 / ratio as fast as real
time: a stream needs that much audio ahead of its first chunk to never leave the server empty (``dry_at``).
"""

from __future__ import annotations

import asyncio
import bisect
import heapq
import math
import time
from collections.abc import AsyncIterator, Callable

__all__ = ["ChunkPacer", "ChunkStream", "StreamSender", "paced"]


class ChunkPacer:
    """When each chunk_bytes chunk of one stream goes out (see the module docstring). Times are seconds of any clock
    that ``push``, ``finish`` and ``pop`` agree on; ``due`` is never earlier than the times they were given."""

    def __init__(self, chunk_bytes: int = 4096, bytes_per_s: float = 32000.0, alpha: float = 0.95,
                 reserve_ms: float | None = 1000.0, expect_s: float | None = None, margin_ms: float = 4.0,
                 unit_bytes: int = 1, alpha_last: float | None = None, phase_ms: float = 0.0,
                 min_gap_ms: float = 0.0, start_ms: float = 0.0):
        if chunk_bytes <= 0 or bytes_per_s <= 0 or not 0 <= alpha <= 1:
            raise ValueError("chunk_bytes and bytes_per_s must be positive, 0 <= alpha <= 1 (0: no pace)")
        self.chunk_bytes = chunk_bytes
        self.bytes_per_s = float(bytes_per_s)
        cap = 1.0 if expect_s is None else self.ratio_for(expect_s, chunk_bytes, bytes_per_s, margin_ms, unit_bytes)
        self.alpha = min(alpha, cap)
        self.alpha_last = self.alpha if alpha_last is None else min(alpha_last, cap)
        self.reserve = None if reserve_ms is None else int(reserve_ms / 1000 * bytes_per_s)
        self._buf = bytearray()
        self._base = 0  # stream offset of _buf[0]
        self._marks: list[int] = []  # stream offsets at the end of each push, and when they arrived
        self._times: list[float] = []
        self._ended: float | None = None
        self._grid: float | None = None  # when the last chunk was due (None: no chunk sent yet)
        self._last_s = 0.0  # its audio, seconds
        self.sent = 0  # chunks sent
        # the chunk after one sent off the grid (the first, or a late one) goes out this much sooner: streams that
        # start (or are late) together then do not send on the same instants (a client reading 200 streams in one
        # event loop timestamps the last of a burst several ms late); costs the server that much lead each time
        self.phase = phase_ms / 1000
        self._shift = 0.0
        # chunks sent ahead of the grid (above the reserve, or what is left at the end) go out at least this far
        # apart: at 200 streams, ends flushed at once (up to a second's chunks of each of a dozen streams) held the
        # connections' event loop and the client's for 6-7 ms, and other streams' paced chunks came late
        self.min_gap = min_gap_ms / 1000
        self.start = start_ms / 1000  # the first chunk this long after it is complete

    @staticmethod
    def ratio_for(expect_s: float, chunk_bytes: int = 4096, bytes_per_s: float = 32000.0, margin_ms: float = 4.0,
                  unit_bytes: int = 1) -> float:
        """The pace's ratio that keeps every gap ``margin_ms`` under the average chunk (its audio over the number of
        chunks) of any utterance at least ``expect_s`` long whose length is a multiple of ``unit_bytes`` (1: any; the
        engine's audio comes in 40 ms frames). At least 0.5."""
        chunk_s = chunk_bytes / bytes_per_s
        first = max(1, -(-int(expect_s * bytes_per_s) // unit_bytes)) * unit_bytes
        worst = min(n / -(-n // chunk_bytes) for n in range(first, first + 8 * chunk_bytes + 1, unit_bytes))
        return max(0.5, (worst / bytes_per_s - margin_ms / 1000) / chunk_s)

    def push(self, data: bytes, t: float) -> None:
        if self._ended is not None:
            raise RuntimeError("push after finish")
        if data:
            self._buf += data
            self._marks.append(self._base + len(self._buf))
            self._times.append(t)

    def finish(self, t: float) -> None:
        if self._ended is None:
            self._ended = t

    @property
    def done(self) -> bool:
        """The stream has ended and everything was sent."""
        return self._ended is not None and not self._buf

    def _arrival(self, end: int) -> float:
        """When the stream had ``end`` bytes (an offset not past what was pushed)."""
        return self._times[bisect.bisect_left(self._marks, end)]

    def due(self) -> float | None:
        """When the next chunk goes out, None while it is not complete (more audio needed). May be in the past."""
        n = len(self._buf)
        if n >= self.chunk_bytes:
            ready = self._arrival(self._base + self.chunk_bytes)
        elif n and self._ended is not None:
            ready = self._ended
        else:
            return None
        if self._grid is None:
            return ready + self.start
        # the last complete chunk queued (the server is empty after it): as late as the pace allows (alpha_last),
        # the engine's next audio has the most time to come
        a = self.alpha_last if n < 2 * self.chunk_bytes and self._ended is None else self.alpha
        t = self._grid + a * self._last_s - self._shift
        if self._ended is not None:  # what is left goes out at once
            t = min(t, self._ended)
        if self.reserve is not None and n > self.chunk_bytes + self.reserve:  # more than the reserve behind it
            t = min(t, self._arrival(self._base + self.chunk_bytes + self.reserve + 1))
        return max(ready, t, self._grid + self.min_gap)

    def pop(self) -> bytes:
        """Take the next chunk (call once ``due`` has passed); the grid moves to when it was due."""
        t = self.due()
        if t is None:
            raise RuntimeError("no chunk is due")
        on_grid = self._grid is not None and t <= self._grid + self.alpha_last * self._last_s - self._shift + 1e-9
        self._shift = 0.0 if on_grid else self.phase
        k = min(self.chunk_bytes, len(self._buf))
        out = bytes(self._buf[:k])
        del self._buf[:k]
        self._base += k
        i = bisect.bisect_right(self._marks, self._base)  # pushes sent entirely
        del self._marks[:i], self._times[:i]
        self._grid, self._last_s = t, k / self.bytes_per_s
        self.sent += 1
        return out

    def dry_at(self) -> float | None:
        """When the next chunk not yet complete is due if none comes before: the time by which the stream's next
        audio must reach the server for the client to see no gap longer than alpha x a chunk (None before the first
        chunk). With audio queued it is later by alpha x a chunk for each complete chunk waiting. Before the first
        chunk went out but once it is complete (it waits start_ms): as if it had gone out when it is due."""
        if self._grid is None:
            first = self.due()
            if first is None:
                return None
            k = min(self.chunk_bytes, len(self._buf))
            waiting = (len(self._buf) - k) // self.chunk_bytes
            return first + self.alpha * (k + waiting * self.chunk_bytes) / self.bytes_per_s
        waiting = len(self._buf) // self.chunk_bytes
        return self._grid + self.alpha * (self._last_s + waiting * self.chunk_bytes / self.bytes_per_s)


class ChunkStream:
    """One stream's chunks at its pacer's times (alpha 0: each one the moment it is complete). The audio is pushed in
    from any callback or task of the event loop (push, finish, fail); ``async for`` takes the chunks out. The taking
    side wakes only when a chunk is due (a timer) or new audio makes one due sooner: at 200 streams the event loop
    gets a patch of every stream at once after each engine step, and each wake-up then delays the sends due."""

    def __init__(self, pacer: ChunkPacer, clock: Callable[[], float] = time.perf_counter):
        self.pacer, self.clock = pacer, clock
        self._wake = asyncio.Event()
        self._target = math.inf  # when the taking side wakes up anyway
        self._failed: BaseException | None = None
        self.last_due: float | None = None  # when the chunk last taken out was due
        self._sender: "tuple[StreamSender, int] | None" = None  # sent by a StreamSender instead of async for

    def push(self, data: bytes, t: float | None = None) -> None:
        self.pacer.push(data, self.clock() if t is None else t)
        if self._sender is not None:
            self._sender[0]._update(self._sender[1])
            return
        due = self.pacer.due()
        if due is not None and due < self._target:
            self._wake.set()

    def finish(self) -> None:
        self.pacer.finish(self.clock())
        if self._sender is not None:
            self._sender[0]._update(self._sender[1])
        self._wake.set()

    def fail(self, e: BaseException) -> None:
        self._failed = e
        if self._sender is not None:
            self._sender[0]._update(self._sender[1])
        self._wake.set()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        loop = asyncio.get_running_loop()
        pacer = self.pacer
        while True:
            if self._failed is not None:
                raise self._failed
            self._wake.clear()
            due = pacer.due()
            if due is None:
                if pacer.done:
                    return
                self._target = math.inf
                await self._wake.wait()
                continue
            wait = due - self.clock()
            if wait > 0:  # (new audio may make the chunk due sooner: more than the reserve waiting)
                self._target = due
                timer = loop.call_later(wait, self._wake.set)
                await self._wake.wait()
                timer.cancel()
                continue
            self._target = -math.inf
            self.last_due = due
            yield pacer.pop()


class StreamSender:
    """Sends the due chunks of many ChunkStreams from one task, a batch at a time: every chunk due within ``window_s``
    goes out in one pass, in the order the streams were added. One wake-up per batch instead of one per stream and
    chunk, and a stream's place in the batch stays the same from chunk to chunk: at 200 streams, waking each stream's
    own task cost 7-15 ms of jitter between a stream's chunks, against margins of 3-9 ms under the average chunk.
    ``send`` must not block (e.g. a transport's write)."""

    def __init__(self, window_s: float = 0.002, clock: Callable[[], float] = time.perf_counter):
        self.window, self.clock = window_s, clock
        self._streams: dict[int, tuple[ChunkStream, Callable[[bytes], None], asyncio.Future]] = {}
        self._queued: dict[int, float] = {}  # the due time each stream is queued with
        self._ending: set[int] = set()  # streams done or failed, to be finished
        self._heap: list[tuple[float, int]] = []
        self._next = 0
        self._wake = asyncio.Event()
        self._sleep_until = math.inf
        self._task: asyncio.Task | None = None

    def add(self, stream: ChunkStream, send: Callable[[bytes], None]) -> asyncio.Future:
        """Send ``stream``'s chunks with ``send``; the future is done once the stream is (or failed)."""
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self._run())
        sid, self._next = self._next, self._next + 1
        fut = asyncio.get_running_loop().create_future()
        self._streams[sid] = (stream, send, fut)
        stream._sender = (self, sid)
        self._update(sid)
        return fut

    def remove(self, stream: ChunkStream) -> None:
        if stream._sender is not None and stream._sender[0] is self:
            self._streams.pop(stream._sender[1], None)
            self._queued.pop(stream._sender[1], None)
            stream._sender = None

    def _update(self, sid: int) -> None:
        item = self._streams.get(sid)
        if item is None:
            return
        stream, _, fut = item
        if stream._failed is not None or stream.pacer.done:
            self._ending.add(sid)
            self._wake.set()  # (finished in _run)
            return
        due = stream.pacer.due()
        if due is None or self._queued.get(sid) == due:
            return
        self._queued[sid] = due
        heapq.heappush(self._heap, (due, sid))
        if due < self._sleep_until:
            self._wake.set()

    def _finish_done(self) -> None:
        """Only the streams _update saw end: scanning every stream on every pass was a third of the connections'
        process's busy time at 200 at once (cProfile)."""
        while self._ending:
            sid = self._ending.pop()
            item = self._streams.get(sid)
            if item is None:
                continue
            stream, _, fut = item
            if stream._failed is not None or stream.pacer.done:
                del self._streams[sid]
                self._queued.pop(sid, None)
                stream._sender = None
                if not fut.done():
                    if stream._failed is not None:
                        fut.set_exception(stream._failed)
                    else:
                        fut.set_result(None)

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            self._wake.clear()
            self._finish_done()
            now = self.clock()
            batch = set()
            while self._heap and self._heap[0][0] <= now + self.window:
                due, sid = heapq.heappop(self._heap)
                if self._queued.get(sid) == due:  # (else stale: the stream's due time changed since)
                    del self._queued[sid]
                    batch.add(sid)
            if batch:
                for sid in sorted(batch):
                    stream, send, fut = self._streams[sid]
                    stream.last_due = stream.pacer.due()
                    try:
                        send(stream.pacer.pop())
                    except Exception as e:  # noqa: BLE001 - the stream's owner hears it
                        stream._failed = e
                    self._update(sid)
                continue
            self._sleep_until = self._heap[0][0] if self._heap else math.inf
            timer = loop.call_later(self._sleep_until - now, self._wake.set) if self._heap else None
            await self._wake.wait()
            self._sleep_until = -math.inf
            if timer is not None:
                timer.cancel()


async def paced(source: AsyncIterator[bytes], pacer: ChunkPacer,
                clock: Callable[[], float] = time.perf_counter) -> AsyncIterator[bytes]:
    """``source``'s bytes in chunks at ``pacer``'s times (a ChunkStream that a task fills from the source)."""
    stream = ChunkStream(pacer, clock)

    async def read():
        try:
            async for data in source:
                stream.push(data)
        except BaseException as e:  # noqa: BLE001 - re-raised on the taking side
            stream.fail(e)
        else:
            stream.finish()

    reader = asyncio.create_task(read())
    try:
        async for chunk in stream:
            yield chunk
    finally:  # the client gone (or done): stop reading, and close the source (its engine request)
        if not reader.done():
            reader.cancel()
            try:
                await reader
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        aclose = getattr(source, "aclose", None)
        if aclose is not None:
            await aclose()
