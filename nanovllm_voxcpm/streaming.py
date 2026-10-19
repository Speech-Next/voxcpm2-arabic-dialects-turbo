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
- the first chunk goes out as soon as it is complete (time to first audio as without pacing);
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
``margin_ms`` (the event loop's delays) under the shortest such average. The pace then runs 1 / ratio as fast as real
time: a stream needs that much audio ahead of its first chunk to never leave the server empty (``dry_at``).
"""

from __future__ import annotations

import asyncio
import bisect
import time
from collections.abc import AsyncIterator, Callable

__all__ = ["ChunkPacer", "paced"]


class ChunkPacer:
    """When each chunk_bytes chunk of one stream goes out (see the module docstring). Times are seconds of any clock
    that ``push``, ``finish`` and ``pop`` agree on; ``due`` is never earlier than the times they were given."""

    def __init__(self, chunk_bytes: int = 4096, bytes_per_s: float = 32000.0, alpha: float = 0.95,
                 reserve_ms: float | None = 1000.0, expect_s: float | None = None, margin_ms: float = 4.0):
        if chunk_bytes <= 0 or bytes_per_s <= 0 or not 0 < alpha <= 1:
            raise ValueError("chunk_bytes and bytes_per_s must be positive, 0 < alpha <= 1")
        self.chunk_bytes = chunk_bytes
        self.bytes_per_s = float(bytes_per_s)
        self.alpha = alpha if expect_s is None else min(alpha, self.ratio_for(expect_s, chunk_bytes, bytes_per_s,
                                                                              margin_ms))
        self.reserve = None if reserve_ms is None else int(reserve_ms / 1000 * bytes_per_s)
        self._buf = bytearray()
        self._base = 0  # stream offset of _buf[0]
        self._marks: list[int] = []  # stream offsets at the end of each push, and when they arrived
        self._times: list[float] = []
        self._ended: float | None = None
        self._grid: float | None = None  # when the last chunk was due (None: no chunk sent yet)
        self._last_s = 0.0  # its audio, seconds
        self.sent = 0  # chunks sent

    @staticmethod
    def ratio_for(expect_s: float, chunk_bytes: int = 4096, bytes_per_s: float = 32000.0,
                  margin_ms: float = 4.0) -> float:
        """The pace's ratio that keeps every gap ``margin_ms`` under the average chunk of any utterance at least
        ``expect_s`` long (m whole chunks and a last one of a byte: m / (m + 1) of a chunk). At least 0.5."""
        chunk_s = chunk_bytes / bytes_per_s
        m = max(1, int(expect_s / chunk_s))
        return max(0.5, (chunk_s * m / (m + 1) - margin_ms / 1000) / chunk_s)

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
            return ready
        t = self._grid + self.alpha * self._last_s
        if self._ended is not None:  # what is left goes out at once
            t = min(t, self._ended)
        if self.reserve is not None and n > self.chunk_bytes + self.reserve:  # more than the reserve behind it
            t = min(t, self._arrival(self._base + self.chunk_bytes + self.reserve + 1))
        return max(ready, t)

    def pop(self) -> bytes:
        """Take the next chunk (call once ``due`` has passed); the grid moves to when it was due."""
        t = self.due()
        if t is None:
            raise RuntimeError("no chunk is due")
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
        chunk). With audio queued it is later by alpha x a chunk for each complete chunk waiting."""
        if self._grid is None:
            return None
        waiting = len(self._buf) // self.chunk_bytes
        return self._grid + self.alpha * (self._last_s + waiting * self.chunk_bytes / self.bytes_per_s)


async def paced(source: AsyncIterator[bytes], pacer: ChunkPacer,
                clock: Callable[[], float] = time.perf_counter) -> AsyncIterator[bytes]:
    """``source``'s bytes in chunks at ``pacer``'s times: a task reads the source while the chunks wait."""
    arrived = asyncio.Event()
    failed: list[BaseException] = []

    async def read():
        try:
            async for data in source:
                pacer.push(data, clock())
                arrived.set()
        except BaseException as e:  # noqa: BLE001 - re-raised in the consumer
            failed.append(e)
        finally:
            pacer.finish(clock())
            arrived.set()

    reader = asyncio.create_task(read())
    try:
        while True:
            if failed:
                raise failed[0]
            arrived.clear()
            due = pacer.due()
            if due is None:
                if pacer.done:
                    break
                await arrived.wait()
                continue
            wait = due - clock()
            if wait > 0:  # new audio may release the chunk before it is due (more than the reserve waiting)
                try:
                    await asyncio.wait_for(arrived.wait(), wait)
                except (asyncio.TimeoutError, TimeoutError):
                    pass
                continue
            yield pacer.pop()
        if failed:
            raise failed[0]
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
