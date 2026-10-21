"""Generated audio kept by what was asked: the same sentence in the same voice, dialect and settings gets the same
audio at any load, and a repeated one costs no GPU (both servers, ``--audio-cache-mb``, ``--audio-cache-dir``).

A seed alone does not give that (#47): a head start's split follows the queue, and the batch and the prefix cache
change the engine's float results, so a seeded request sounds different under another load. The cache keeps each
sentence's first complete generation (its chunks as the engine gave them, before resampling or encoding) and plays
it again; beyond max_bytes the least recently used go first. Requests for the same sentence that run at the same
time share one generation (single flight): a request that comes while it runs gets the chunks made so far, then the
live ones; the generation goes on while any of them still reads (its first client may leave), and ends when none
does. A kept sentence plays its first second at once and the rest at most ``speed`` x real time (when asked): 150
kept sentences at once, all their audio in one go, held the event loop that also serves the engine and paces the
streams (first chunk p99 0.51 s instead of 0.32 s at 150 at once).

With a ``directory`` (``--audio-cache-dir``) each kept sentence is also a file there: it outlives the server, and
replicas on a shared disk share it. A file is named by a hash of the cache's namespace (the model and the settings
that decide the audio: another model reads other files) and the sentence's key; written whole (a temporary file,
renamed), read on a miss in memory, the ones used longest ago removed beyond ``dir_max_bytes``. A file that does not
read back whole is removed and the sentence generated.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable, Hashable

import numpy as np

_INT16_SCALE = np.float32(1 / 32768)
_TMP_MAX_AGE_S = 3600  # a temporary file older than this was left by a writer that died: removed


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


def namespace(**settings) -> str:
    """A cache directory's namespace: the settings that decide the audio besides a sentence's key (the model, the
    LocDiT, the pinned Hub revisions, ...), so that a file is read only by a server that would make the same audio."""
    return json.dumps(settings, sort_keys=True, default=str)


class _Flight:
    """A sentence's generation in progress, read by every request for it: its chunks so far, then the live ones."""

    def __init__(self) -> None:
        self.chunks: list[np.ndarray] = []
        self.done = False
        self.error: BaseException | None = None
        self.readers = 0
        self.task: asyncio.Task | None = None
        self._event = asyncio.Event()

    def _wake(self) -> None:
        event, self._event = self._event, asyncio.Event()
        event.set()

    def push(self, c: np.ndarray) -> None:
        self.chunks.append(c)
        self._wake()

    def end(self, error: BaseException | None = None) -> None:
        self.done, self.error = True, error
        self._wake()

    async def wait(self) -> None:
        await self._event.wait()


class AudioCache:
    LEAD_S = 1.0  # seconds of a kept sentence played at once, the rest at most ``speed`` x real time

    def __init__(self, max_bytes: int, sample_rate: int = 48000, directory: str | None = None,
                 dir_max_bytes: int = 0, namespace: str = "") -> None:
        self.max_bytes = max(0, int(max_bytes))
        self.sample_rate = sample_rate  # the chunks' (the engine's)
        # key -> (the chunks' dtype as generated, the chunks as kept: _packed, read-only)
        self.entries: OrderedDict[Hashable, tuple[np.dtype, tuple[np.ndarray, ...]]] = OrderedDict()
        self.bytes = 0
        self.inflight: dict[Hashable, _Flight] = {}
        self._tasks: set[asyncio.Task] = set()  # generations (and their files' writes) not ended yet
        self.directory, self.dir_max_bytes, self.namespace = directory, max(0, int(dir_max_bytes)), namespace
        if directory:
            os.makedirs(directory, exist_ok=True)
        # sentences played from the cache (disk_hits of them read from the directory), generated, and requests that
        # joined a generation in progress
        self.hits = self.disk_hits = self.misses = self.joins = 0

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses, "bytes": self.bytes, "entries": len(self.entries),
                "disk_hits": self.disk_hits, "joins": self.joins}

    def _kept(self, key: Hashable) -> tuple[np.dtype, tuple[np.ndarray, ...]] | None:
        entry = self.entries.get(key)
        if entry is not None:
            self.entries.move_to_end(key)
        return entry

    def get(self, key: Hashable) -> list[np.ndarray] | None:
        """The kept chunks of ``key`` as they were generated, or None."""
        entry = self._kept(key)
        return None if entry is None else [_unpacked(c, entry[0]) for c in entry[1]]

    def put(self, key: Hashable, chunks: list[np.ndarray], dtype: np.dtype | None = None) -> bool:
        """Kept (``chunks`` _packed already when ``dtype`` is given: the dtype they were generated in), unless it is
        kept already (the first generation stays) or larger than a quarter of the cache. Whether it was kept."""
        if dtype is None:
            dtype = chunks[0].dtype if chunks else np.dtype(np.float32)
            chunks = [_packed(c, _is_pcm16(c)) for c in chunks]
        size = sum(c.nbytes for c in chunks)
        if key in self.entries or size > self.max_bytes // 4:
            return False
        for c in chunks:
            c.setflags(write=False)
        self.entries[key] = (np.dtype(dtype), tuple(chunks))
        self.bytes += size
        while self.bytes > self.max_bytes:
            _, (_, old) = self.entries.popitem(last=False)
            self.bytes -= sum(c.nbytes for c in old)
        return True

    # --- the directory (--audio-cache-dir)

    def _file(self, key: Hashable) -> tuple[str, str]:
        """(the file of ``key``, its key as written in it): a hash of the namespace and the key's canonical JSON."""
        text = json.dumps([self.namespace, key], ensure_ascii=False, default=str)
        return os.path.join(self.directory, hashlib.sha256(text.encode()).hexdigest()[:40] + ".npz"), text

    def _read(self, key: Hashable) -> tuple[np.dtype, list[np.ndarray]] | None:
        """``key``'s file (in a thread: off the event loop): (dtype, chunks as kept), or None. A file that does not
        read back whole (cut short, another key, not one of ours) is removed."""
        path, text = self._file(key)
        try:
            with np.load(path, allow_pickle=False) as f:
                if str(f["key"]) != text:
                    raise ValueError("another key")
                audio, lengths, dtype = f["audio"], f["lengths"], np.dtype(str(f["dtype"]))
            if audio.ndim != 1 or lengths.ndim != 1 or lengths.min(initial=0) < 0 or int(lengths.sum()) != len(audio):
                raise ValueError("its lengths do not add up")
        except FileNotFoundError:
            return None
        except Exception:  # noqa: BLE001 - cut short, corrupt, another format: not kept
            with contextlib.suppress(OSError):
                os.remove(path)
            return None
        with contextlib.suppress(OSError):
            os.utime(path)  # (used now: the ones used longest ago go first)
        return dtype, list(np.split(audio, np.cumsum(lengths)[:-1])) if len(lengths) else []

    def _write(self, key: Hashable, dtype: np.dtype, chunks: list[np.ndarray]) -> None:
        """``key``'s file, whole (a temporary file renamed), then the directory within dir_max_bytes (in a thread)."""
        if any(c.ndim != 1 for c in chunks) or len({c.dtype for c in chunks}) > 1:
            return  # (mixed forms, e.g. int16 silence then float audio: kept in memory only)
        path, text = self._file(key)
        tmp = f"{path}.{os.getpid()}.{id(chunks)}.tmp"
        try:
            with open(tmp, "wb") as f:
                np.savez(f, audio=np.concatenate(chunks) if chunks else np.zeros(0, np.int16),
                         lengths=np.array([len(c) for c in chunks], np.int64), dtype=np.array(str(dtype)),
                         key=np.array(text))
            os.replace(tmp, path)
        except OSError:  # (a full or read-only disk: kept in memory only)
            with contextlib.suppress(OSError):
                os.remove(tmp)
            return
        self._trim()

    def _trim(self) -> None:
        """The directory within dir_max_bytes: the files used longest ago removed (and stale temporary ones)."""
        files, total, now = [], 0, time.time()
        with contextlib.suppress(OSError), os.scandir(self.directory) as it:
            for e in it:
                with contextlib.suppress(OSError):
                    st = e.stat()
                    if e.name.endswith(".tmp"):
                        if now - st.st_mtime > _TMP_MAX_AGE_S:
                            os.remove(e.path)
                    elif e.name.endswith(".npz"):
                        files.append((st.st_mtime, st.st_size, e.path))
                        total += st.st_size
        if self.dir_max_bytes:
            for _, size, path in sorted(files):
                if total <= self.dir_max_bytes:
                    break
                with contextlib.suppress(OSError):
                    os.remove(path)
                    total -= size

    async def idle(self) -> None:
        """Once every generation started has ended and its file is written (tests, a clean stop)."""
        while self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    # --- requests

    async def _replay(self, chunks, dtype: np.dtype, speed: float | None) -> AsyncIterator[np.ndarray]:
        t0, played = time.perf_counter(), 0.0
        for c in chunks:  # (each unpacked as it goes: 150 sentences at once, all at the start, held the loop)
            yield _unpacked(c, dtype)
            played += c.shape[-1] / self.sample_rate
            ahead = (played - self.LEAD_S) / speed - (time.perf_counter() - t0) if speed else 0.0
            await asyncio.sleep(max(ahead, 0.0))

    async def through(self, key: Hashable, generate: Callable[[], AsyncIterator[np.ndarray]],
                      speed: float | None = None) -> AsyncIterator[np.ndarray]:
        """The audio chunks of ``key``: the kept ones (in memory, else in the directory; after the first LEAD_S
        seconds at most ``speed`` x real time, None: as fast as they are taken, the event loop served between them),
        else those of the generation in progress for it (the ones made already paced as kept ones, then as they come)
        or of a new one, ``generate()``'s, kept once it ends (not when it fails, or every request reading it left)."""
        hit = self._kept(key)
        if hit is None and key not in self.inflight and self.directory:
            read = await asyncio.to_thread(self._read, key)
            hit = self._kept(key)  # (kept meanwhile by a generation that ended)
            if hit is None and read is not None:
                self.disk_hits += 1
                dtype, chunks = read
                hit = self._kept(key) if self.put(key, chunks, dtype) else (np.dtype(dtype), tuple(chunks))
        if hit is not None:
            self.hits += 1
            async with contextlib.aclosing(self._replay(hit[1], hit[0], speed)) as replay:
                async for c in replay:
                    yield c
            return
        flight = self.inflight.get(key)
        if flight is None:
            self.misses += 1
            flight = self.inflight[key] = _Flight()
            flight.task = asyncio.get_running_loop().create_task(self._generate(key, flight, generate))
            self._tasks.add(flight.task)
            flight.task.add_done_callback(self._tasks.discard)
        else:
            self.joins += 1
        flight.readers += 1
        try:
            backlog, i = len(flight.chunks), 0  # (the chunks made before this request came)
            t0, played = time.perf_counter(), 0.0
            while True:
                if i < len(flight.chunks):
                    c = flight.chunks[i]
                    i += 1
                    yield c
                    if i <= backlog and speed:  # (played as a kept sentence's, then as they come)
                        played += c.shape[-1] / self.sample_rate
                        await asyncio.sleep(max((played - self.LEAD_S) / speed - (time.perf_counter() - t0), 0.0))
                elif flight.error is not None:
                    raise flight.error
                elif flight.done:
                    return
                else:
                    await flight.wait()
        finally:
            flight.readers -= 1
            if not flight.readers and not flight.done and flight.task is not None:
                # every request reading it left: its engine request ends, nothing is kept; a request for it from now
                # on starts another
                if self.inflight.get(key) is flight:
                    del self.inflight[key]
                flight.task.cancel()

    async def _generate(self, key: Hashable, flight: _Flight, generate: Callable[[], AsyncIterator[np.ndarray]]
                        ) -> None:
        """A sentence's generation, a task of its own (it goes on while a request still reads it), kept once it
        ends; an error ends every request reading it."""
        kept: list[np.ndarray] | None = []
        size, dtype, pcm16 = 0, None, None
        try:
            async with contextlib.aclosing(generate()) as gen:
                async for c in gen:
                    if kept is not None:
                        dtype = dtype or c.dtype
                        if pcm16 is None and c.any():  # (the first sounding chunk tells the rest's form)
                            pcm16 = _is_pcm16(c)
                        p = _packed(c, pcm16 is not False)  # (silence before it: int16 zeros, the same either way)
                        size += p.nbytes
                        if size <= self.max_bytes // 4:
                            kept.append(p)
                        else:  # (too long to keep)
                            kept = None
                    flight.push(c)
        except asyncio.CancelledError:
            flight.end(RuntimeError("the generation was cancelled"))
            raise
        except Exception as e:  # noqa: BLE001 - every request reading it hears it
            flight.end(e)
            return
        finally:
            if self.inflight.get(key) is flight:
                del self.inflight[key]
        if kept is not None:
            self.put(key, kept, dtype)
        flight.end()
        if kept is not None and self.directory:
            with contextlib.suppress(Exception):  # (the disk is best effort: kept in memory anyway)
                await asyncio.to_thread(self._write, key, np.dtype(dtype or np.float32), kept)
