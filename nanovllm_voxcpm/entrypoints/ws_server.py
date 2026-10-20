"""WebSocket streaming TTS server over RoutedDialectTTS, with the protocol of #28's WebSocket benchmark
(VoxCPM-Kernel/benchmark-test-data-ar), the audio paced on the server so that a client sees no gap between chunks
longer than a chunk's audio where the engine has the audio (#31, nanovllm_voxcpm/streaming.py)::

    python -m nanovllm_voxcpm.entrypoints.ws_server --port 8092 --voice default=reference.wav [--no-pace]

    ws://HOST:PORT/api/tts/stream?audio_format=pcm_16000&chunk_size=4096&segmenting=false&transport=bytes&dialect=..

- An optional first message {"type": "control", "voice": .., "cfg_value": ..} -> {"type": "config_updated"}
  ({"type": "config_error"} for an unknown voice; speed and temperature are accepted and ignored).
- Every other text message is one utterance, spoken in order on its connection: {"type": "start", "sample_rate": R},
  the 16-bit PCM at R (audio_format=pcm_R; the model's 48 kHz resampled with soxr) in chunks of chunk_size bytes (the
  last one shorter), then {"type": "end_of_audio"}. transport=json_base64: {"type": "audio_chunk", "chunk_data":
  <base64>} instead of binary messages. segmenting=true: a long text spoken sentence by sentence into the same stream
  (openai_server.split_text). Errors: {"type": "error", "message": ..}.
- dialect: a name RoutedDialectTTS routes (--dialect: the default). The seed is the text's CRC32 (the same text, the
  same seed; its audio still varies with the load: a head start's split follows the queue).
- Pacing (default; --no-pace sends each chunk the moment it is complete): streaming.ChunkPacer per utterance, the
  first chunk --pace-start-ms (150) after it is complete, the next ones on a grid of --pace-alpha (0.9) x a chunk's audio (--pace-alpha-last, 0.92,
  for the last chunk queued), more than --pace-reserve-ms queued and the rest at the end sooner (at least
  --pace-min-gap-ms, 8, apart: an end flushed at once held both event loops for 6-7 ms at 200 streams); the chunk
  after the first (or a late one) up to --pace-spread-ms sooner, a different amount per utterance. One task sends
  every connection's due chunks as whole frames (StreamSender). A gap stays 10-13 ms under the shortest average
  chunk (124.4 ms with 4096-byte chunks of 16 kHz): the event loops of both sides (the client's too) delayed
  chunks by up to ~10 ms at 150-200 streams. The engine is told the chunk's length (stream's chunk_ms) so that its
  deadlines allow for what the server holds back; the pace runs 1/0.9 as fast as real time.
- Two processes: the connections and their pacers run in a process of their own, the
  engine's client (RoutedDialectTTS) in this one, the audio crossing over socket pairs. At 200 streams the client's
  event loop is busy 10-20 ms after every engine step (a patch of every stream at once), and a paced chunk due
  meanwhile went out that much late: a stall where the server had the audio.
- Production: ``--api-key`` (or VOXCPM_API_KEY): the handshake needs ``Authorization: Bearer <key>`` or
  ``?api_key=<key>``; ``--max-streams`` answers an utterance beyond that many at once with an error (one A10: 150);
  ``--max-text-chars`` bounds a text; an unknown dialect is refused at the handshake; an utterance's error leaves its
  connection open for the next. ``GET /health`` on the same port answers 200 while the server runs; when the engine
  process dies the server exits with code 1, for its supervisor to start it again.
"""

import argparse
import asyncio
import base64
import contextlib
import hmac
import json
import math
import os
import signal
import struct
import sys
import time
import zlib
from collections.abc import AsyncIterator, Callable
from urllib.parse import parse_qs, urlparse

import numpy as np

from nanovllm_voxcpm.streaming import ChunkPacer, ChunkStream, StreamSender

PATH = "/api/tts/stream"
QUEUED_TEXTS = 64  # a connection's texts waiting for their turn; more wait in the socket (backpressure)


def ws_frame(opcode: int, payload: bytes) -> bytes:
    """A whole, unmasked WebSocket frame (server to client; no extensions): opcode 1 text, 2 binary."""
    n = len(payload)
    if n < 126:
        return bytes((0x80 | opcode, n)) + payload
    if n < 65536:
        return struct.pack("!BBH", 0x80 | opcode, 126, n) + payload
    return struct.pack("!BBQ", 0x80 | opcode, 127, n) + payload


def pacer_settings(p: ChunkPacer) -> dict | float:
    """An utterance's pacer settings for the engine, which runs the same pacer (its deadlines); 0.0: not paced."""
    if not p.alpha:
        return 0.0
    return {"alpha": p.alpha, "alpha_last": p.alpha_last,
            "reserve_ms": None if p.reserve is None else p.reserve / p.bytes_per_s * 1000,
            "phase_ms": p.phase * 1000, "min_gap_ms": p.min_gap * 1000, "start_ms": p.start * 1000}


def pcm16(y: np.ndarray) -> bytes:
    if y.dtype == np.int16:
        return y.tobytes()
    return (np.clip(y, -1.0, 1.0) * 32767.0).round().astype("<i2").tobytes()


async def speak(tts, req: dict, yields: list | None = None) -> AsyncIterator[bytes]:
    """One utterance's PCM16 at req["rate"]: req = text, dialect, voice, cfg_value, rate, chunk_ms, pace, segmenting.
    yields: [time, samples from the engine, ms resampling, bytes out] per engine chunk."""
    parts = [req["text"]]
    if req["segmenting"]:
        from nanovllm_voxcpm.entrypoints.openai_server import split_text

        parts = split_text(req["text"]) or parts
    rs = None
    if req["rate"] != tts.sample_rate:
        import soxr

        rs = soxr.ResampleStream(tts.sample_rate, req["rate"], 1, dtype="float32")
    for part in parts:
        seed = zlib.crc32(part.encode()) & 0x7FFFFFFF
        async for c in tts.stream(part, dialect=req["dialect"], voice=req["voice"], seed=seed,
                                  cfg_value=req["cfg_value"], chunk_ms=req["chunk_ms"], pace=req.get("pace")):
            t = time.perf_counter()
            out = pcm16(rs.resample_chunk(c, last=False) if rs is not None else c)
            if yields is not None:
                yields.append([t, int(c.size), (time.perf_counter() - t) * 1000, len(out)])
            yield out
    if rs is not None:
        out = pcm16(rs.resample_chunk(np.zeros(0, np.float32), last=True))
        if yields is not None:
            yields.append([time.perf_counter(), 0, 0.0, len(out)])
        yield out


Feed = Callable[[dict, list, ChunkStream], Callable[[], None]]


def local_feed(tts) -> Feed:
    """An utterance's audio from the engine's client in this process: a task fills the stream (speak); the
    returned callable stops it (the client gone: its engine request ends)."""

    def feed(req: dict, yields: list, stream: ChunkStream) -> Callable[[], None]:
        async def read():
            gen = speak(tts, req, yields)
            try:
                async for data in gen:
                    stream.push(data)
                stream.finish()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - the client hears why
                stream.fail(e)
            finally:
                await gen.aclose()

        return asyncio.create_task(read()).cancel

    return feed


def message(e: BaseException) -> str:
    """An error as its client reads it: the message, no traceback or exception repr."""
    return str(e) or type(e).__name__


class Server:
    """The connections. voices: the registered voice names; feed(req, yields, stream): fills an utterance's
    ChunkStream with its PCM16 and returns what stops it; dialects: the names a connection may ask for (None: not
    checked here)."""

    def __init__(self, voices, feed: Feed, args, dialects=None):
        self.voices, self.feed, self.args, self.dialects = voices, feed, args, dialects
        self.log = open(args.log, "a") if args.log else None
        self.conn = 0
        # paced chunks of every connection from one task (StreamSender), written to the transports as whole frames
        self.sender = StreamSender(0.0) if args.pace else None
        self.utterances = 0
        self.active = 0  # utterances in progress (--max-streams)

    def process_request(self, connection, request):
        """Before a handshake: GET /health answered here (this process runs only while the engine's does); with
        --api-key, a client without it (``Authorization: Bearer <key>`` or ``?api_key=<key>``, browsers) gets 401."""
        from http import HTTPStatus

        u = urlparse(request.path)
        if u.path == "/health":
            return connection.respond(HTTPStatus.OK, "ok\n")
        key = getattr(self.args, "api_key", None)
        if key:
            bearer = request.headers.get("Authorization", "").encode()
            query = parse_qs(u.query).get("api_key", [""])[0].encode()
            if not (hmac.compare_digest(bearer, f"Bearer {key}".encode()) or hmac.compare_digest(query, key.encode())):
                return connection.respond(HTTPStatus.UNAUTHORIZED, "missing or wrong API key\n")
        return None

    def control(self, cfg: dict, obj: dict) -> dict:
        """A control message applied to its connection's settings (all or nothing): the reply."""
        voice = obj.get("voice")
        if voice is not None and (not isinstance(voice, str) or voice not in self.voices):
            return {"type": "config_error", "message": f"voice {voice!r}"}
        cfg_value = obj.get("cfg_value")
        if cfg_value is not None:
            try:
                cfg_value = float(cfg_value)
            except (TypeError, ValueError):
                cfg_value = math.nan
            if not 0.0 <= cfg_value <= 10.0:  # (NaN fails it too)
                return {"type": "config_error", "message": f"cfg_value {obj['cfg_value']!r}: a number from 0 to 10"}
            cfg["cfg_value"] = cfg_value
        if "voice" in obj:  # (null: the model's own voice)
            cfg["voice"] = voice
        return {"type": "config_updated"}

    def pacer(self, n: int, rate: int) -> ChunkPacer:
        a = self.args
        if not a.pace:  # each chunk the moment it is complete
            return ChunkPacer(n, 2 * rate, alpha=0.0, reserve_ms=None)
        self.utterances += 1  # each its own phase in [0, --pace-spread-ms) (golden-ratio sequence)
        phase = getattr(a, "pace_spread_ms", 0.0) * ((self.utterances * 0.6180339887) % 1.0)
        return ChunkPacer(n, 2 * rate, a.pace_alpha, a.pace_reserve_ms if a.pace_reserve_ms >= 0 else None,
                          alpha_last=a.pace_alpha_last, phase_ms=phase, min_gap_ms=getattr(a, "pace_min_gap_ms", 0.0),
                          start_ms=getattr(a, "pace_start_ms", 0.0))

    async def handler(self, ws):
        u = urlparse(ws.request.path)
        if u.path != PATH:
            await ws.close(1008, "unknown path")
            return
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        fmt = q.get("audio_format", "pcm_16000")
        try:
            rate = int(fmt.removeprefix("pcm_")) if fmt.startswith("pcm_") else 0
            n = int(q.get("chunk_size", 4096))
        except ValueError:
            rate = n = 0
        transport, segmenting = q.get("transport", "bytes"), q.get("segmenting", "false").lower()
        # (chunks under ~8 ms of audio streamed slower than real time; a huge one held the whole utterance back)
        if (not 8000 <= rate <= 48000 or n % 2 or not 0.02 <= n / (2 * rate) <= 10.0
                or transport not in ("bytes", "json_base64") or segmenting not in ("true", "false")):
            await ws.send(json.dumps({"type": "error", "message": "audio_format pcm_<8000-48000>, chunk_size even "
                                      "and 20 ms to 10 s of audio, transport bytes or json_base64, segmenting true "
                                      "or false"}))
            await ws.close()
            return
        dialect = q.get("dialect", self.args.dialect)
        if self.dialects is not None and dialect not in self.dialects:
            await ws.send(json.dumps({"type": "error", "message": f"unknown dialect {dialect!r}; dialects: "
                                                                  f"{', '.join(sorted(self.dialects))}"}))
            await ws.close()
            return
        cfg = dict(rate=rate, chunk_size=n, transport=transport, segmenting=segmenting == "true",
                   dialect=dialect, voice=self.args.default_voice, cfg_value=2.0)
        self.conn += 1
        conn = self.conn
        texts: asyncio.Queue = asyncio.Queue(QUEUED_TEXTS)

        async def reader():
            try:
                async for msg in ws:
                    if not isinstance(msg, str):
                        continue
                    try:
                        obj = json.loads(msg)
                    except json.JSONDecodeError:
                        obj = None
                    if isinstance(obj, dict) and obj.get("type") == "control":
                        await ws.send(json.dumps(self.control(cfg, obj)))
                        continue
                    if isinstance(obj, dict):  # (another protocol message, e.g. a ping: not a text to speak)
                        await ws.send(json.dumps({"type": "error", "message": f"unknown message {obj.get('type')!r}"}))
                        continue
                    await texts.put((time.perf_counter(), msg))
            finally:  # (a full queue: the texts in it end the loop, their sends failing on the closed connection)
                with contextlib.suppress(asyncio.QueueFull):
                    texts.put_nowait(None)

        rd = asyncio.create_task(reader())
        try:
            while (item := await texts.get()) is not None:
                try:
                    await self.utterance(ws, conn, cfg, *item)
                except Exception as e:  # noqa: BLE001 - the client hears why; its next text is still spoken
                    try:
                        await ws.send(json.dumps({"type": "error", "message": message(e)}))
                    except Exception:  # noqa: BLE001 - the client is gone
                        break
        finally:
            rd.cancel()

    async def utterance(self, ws, conn, cfg, t_recv, text):
        """One text spoken (speak_text), within --max-text-chars and --max-streams."""
        if not text.strip():
            raise ValueError("text is empty")
        limit = getattr(self.args, "max_text_chars", 0)
        if limit and len(text) > limit:
            raise ValueError(f"text: at most {limit} characters (--max-text-chars)")
        most = getattr(self.args, "max_streams", 0)
        if most and self.active >= most:
            raise ValueError(f"busy: {self.active} utterances in progress (--max-streams), try again")
        self.active += 1
        try:
            await self.speak_text(ws, conn, cfg, t_recv, text)
        finally:
            self.active -= 1

    async def speak_text(self, ws, conn, cfg, t_recv, text):
        n, rate, b64 = cfg["chunk_size"], cfg["rate"], cfg["transport"] == "json_base64"
        await ws.send(json.dumps({"type": "start", "sample_rate": rate}))
        pacer = self.pacer(n, rate)
        # the engine runs the same pacer (chunk_ms, pace: its deadlines are the pacer's)
        req = dict(text=text, dialect=cfg["dialect"], voice=cfg["voice"], cfg_value=cfg["cfg_value"], rate=rate,
                   chunk_ms=n / (2 * rate) * 1000, pace=pacer_settings(pacer), segmenting=cfg["segmenting"])
        yields, sends = [], []
        stream = ChunkStream(pacer)
        stop = self.feed(req, yields, stream)
        try:
            if self.sender is not None and not ws.protocol.extensions:
                from websockets.protocol import State

                transport = ws.transport

                def write(b: bytes) -> None:  # (in the sender's pass: no await, the frame written whole)
                    if ws.state is not State.OPEN:
                        raise ConnectionError("the client is gone")
                    t = time.perf_counter()
                    transport.write(ws_frame(1, json.dumps({"type": "audio_chunk", "chunk_data": base64.b64encode(
                        b).decode()}).encode()) if b64 else ws_frame(2, b))
                    sends.append([time.perf_counter(), len(b), stream.last_due, t])

                try:
                    await self.sender.add(stream, write)
                finally:
                    self.sender.remove(stream)
            else:
                async for b in stream:
                    t = time.perf_counter()
                    await (ws.send(json.dumps({"type": "audio_chunk", "chunk_data": base64.b64encode(b).decode()}))
                           if b64 else ws.send(b))
                    sends.append([time.perf_counter(), len(b), stream.last_due, t])
        finally:
            stop()
        await ws.send(json.dumps({"type": "end_of_audio"}))
        if self.log is not None:  # perf_counter (CLOCK_MONOTONIC) times: comparable with the engine's profiles
            # yields: [t, samples from the engine, ms resampling, bytes out] (two processes: [t received, 0, 0,
            # bytes]); sends: [t after the send, bytes, when it was due, t before the send]
            self.log.write(json.dumps({"conn": conn, "crc": zlib.crc32(text.encode()), "t_recv": t_recv,
                                       "t_end": time.perf_counter(), "chars": len(text), "dialect": cfg["dialect"],
                                       "chunk_size": n, "rate": rate,
                                       "paced": self.args.pace, "yields": yields, "sends": sends}) + "\n")
            self.log.flush()


# --- two processes: the engine's client here, the connections there, length-prefixed frames over a socket pair ---
_HDR = struct.Struct("<BII")  # type, utterance, payload bytes
READY, LISTENING, START, CANCEL, DATA, END, ERROR = range(7)


def frame(kind: int, utt: int, payload: bytes = b"") -> bytes:
    return _HDR.pack(kind, utt, len(payload)) + payload


async def read_frame(reader: asyncio.StreamReader) -> tuple[int, int, bytes]:
    kind, utt, n = _HDR.unpack(await reader.readexactly(_HDR.size))
    return kind, utt, await reader.readexactly(n) if n else b""


async def serve_engine(tts, reader: asyncio.StreamReader, writers, log=None) -> None:
    """The engine's side: START (an utterance's request, JSON) -> DATA frames, then END (or ERROR); CANCEL stops
    one. Returns when the connections' side closes. writers: the link's sockets (a StreamWriter or a list): an
    utterance's frames all go to writers[utt % len]. log: a file, one jsonl line per utterance (perf_counter times:
    the START frame read, the task started, the yields [t, samples, ms resampling, bytes])."""
    jobs: dict[int, asyncio.Task] = {}
    writers = writers if isinstance(writers, list) else [writers]

    async def run(utt: int, req: dict, t_start: float):
        t_task, yields = time.perf_counter(), []
        writer = writers[utt % len(writers)]
        try:
            async for out in speak(tts, req, yields if log is not None else None):
                writer.write(frame(DATA, utt, out))
            writer.write(frame(END, utt))
            if log is not None:
                log.write(json.dumps({"utt": utt, "crc": zlib.crc32(req["text"].encode()), "t_start": t_start,
                                      "t_task": t_task, "yields": yields, "t_end": time.perf_counter()}) + "\n")
        except asyncio.CancelledError:
            pass
        except Exception as e:  # noqa: BLE001 - the client hears why
            writer.write(frame(ERROR, utt, message(e).encode()))
        finally:
            jobs.pop(utt, None)

    try:
        while True:
            kind, utt, payload = await read_frame(reader)
            if kind == START:
                jobs[utt] = asyncio.create_task(run(utt, json.loads(payload), time.perf_counter()))
            elif kind == CANCEL and utt in jobs:
                jobs.pop(utt).cancel()
    except (asyncio.IncompleteReadError, ConnectionError):
        pass
    finally:
        for job in list(jobs.values()):
            job.cancel()


class EngineLink(asyncio.Protocol):
    """The connections' side of the two processes' link: each frame's audio goes straight into its utterance's
    ChunkStream as the bytes arrive (no task or queue per frame: after each engine step a patch of every stream
    arrives at once). start(req, yields, stream) is a Feed. More sockets (``primary``: the first one's link, whose
    utterances they share) carry more of a step's audio at once: a socket pair holds ~340 KB, a step of 200
    streams ~1 MB, and the rest waited in the engine client until its event loop got round to the socket again
    (the process hop's delay p99 7-9 ms at 200 at once)."""

    def __init__(self, primary: "EngineLink | None" = None):
        loop = asyncio.get_running_loop()
        self.buf = bytearray()
        self.transport = None
        if primary is None:
            self.ready, self.closed = loop.create_future(), loop.create_future()
            self.sinks: dict[int, tuple[ChunkStream, list | None]] = {}
            self.next_utt = 0
        else:
            self.ready, self.closed, self.sinks = primary.ready, primary.closed, primary.sinks

    def connection_made(self, transport):
        self.transport = transport

    def data_received(self, data: bytes) -> None:
        """Frames parsed from views of the bytes as they came (no copy but the pacer's own): at 200 streams an
        engine step's ~1 MB arrives at once, and every microsecond here delays the chunks waiting behind it."""
        src = data
        if self.buf:  # (the rest of a frame from the last read)
            self.buf += data
            src = self.buf
        t, off, size, end = time.perf_counter(), 0, _HDR.size, len(src)
        with memoryview(src) as mv:
            while end - off >= size:
                kind, utt, n = _HDR.unpack_from(src, off)
                if end - off - size < n:
                    break
                payload = mv[off + size:off + size + n]
                off += size + n
                if kind == READY:
                    if not self.ready.done():
                        self.ready.set_result(json.loads(payload.tobytes()))
                elif (sink := self.sinks.get(utt)) is not None:
                    stream, yields = sink
                    if kind == DATA:
                        if yields is not None:
                            yields.append([t, 0, 0.0, n])
                        stream.push(payload, t)  # (the pacer copies it)
                    else:
                        del self.sinks[utt]
                        if kind == END:
                            stream.finish()
                        else:
                            stream.fail(RuntimeError(payload.tobytes().decode(errors="replace")))
                payload.release()
        if src is self.buf:
            del self.buf[:off]
        elif off < end:
            self.buf += data[off:]

    def connection_lost(self, exc) -> None:
        for stream, _ in self.sinks.values():
            stream.fail(RuntimeError("the engine's process is gone"))
        self.sinks.clear()
        for f in (self.ready, self.closed):
            if not f.done():
                f.set_result(None)

    def send(self, kind: int, utt: int = 0, payload: bytes = b"") -> None:
        if not self.transport.is_closing():
            self.transport.write(frame(kind, utt, payload))

    def start(self, req: dict, yields: list, stream: ChunkStream) -> Callable[[], None]:
        self.next_utt += 1
        utt = self.next_utt
        self.sinks[utt] = (stream, yields)
        self.send(START, utt, json.dumps(req).encode())

        def stop():
            if self.sinks.pop(utt, None) is not None:  # not ended: the client gone, the engine request ends
                self.send(CANCEL, utt)

        return stop


async def load_tts(args):
    from nanovllm_voxcpm import RoutedDialectTTS
    from nanovllm_voxcpm.warmup import warm_up

    kw = {k: v for k, v in dict(model=args.model, tagged_model=args.tagged_model, head_tokens=args.head_tokens,
                                max_concurrency=args.max_concurrency, kv_blocks=args.kv_blocks).items()
          if v is not None}
    if args.dit is not None:
        kw["dit"] = None if args.dit in ("", "none") else args.dit
    tts = await RoutedDialectTTS.load(**kw)
    for spec in args.voice:
        name, _, path = spec.partition("=")
        await tts.add_voice(name, path)
    await warm_up(lambda i, text: tts.synthesize(text, dialect=args.dialect, voice=args.default_voice))
    return tts


async def serve_ws(server: Server, args, until: asyncio.Future, on_listening: Callable[[], None]):
    from websockets.asyncio.server import serve

    # client messages are texts and control messages: at most 1 MiB each (else close 1009), 16 frames buffered
    # (then TCP backpressure); a ping every --ping-interval s finds clients gone without a close (else their engine
    # requests ran to the end)
    ping = getattr(args, "ping_interval", 0) or None
    async with serve(server.handler, args.host, args.port, max_size=1 << 20, compression=None, ping_interval=ping,
                     ping_timeout=ping, max_queue=16, process_request=server.process_request) as ws_server:
        on_listening()
        await until
        ws_server.close()


def announce(args):
    if args.ready:
        with open(args.ready, "w") as f:
            f.write(str(os.getpid()))
    print(f"[ws_server] ws://{args.host}:{args.port}{PATH} ready (pacing "
          f"{'alpha %.2f, reserve %g ms' % (args.pace_alpha, args.pace_reserve_ms) if args.pace else 'off'}, "
          f"{', uvloop' if 'uvloop' in type(asyncio.get_running_loop()).__module__ else ''})", flush=True)


def stop_signal() -> asyncio.Future:
    """Set on SIGTERM / SIGINT (then the engine is closed: its process holds the GPU)."""
    stop = asyncio.get_running_loop().create_future()
    for sig in (signal.SIGTERM, signal.SIGINT):
        asyncio.get_running_loop().add_signal_handler(sig, lambda: stop.done() or stop.set_result(None))
    return stop


async def main_engine(args, socks, front_pid: int) -> int:
    """This process: the engine's client, serving the connections' process. Returns the exit code: 1 when the
    engine process died (every utterance would fail: the supervisor starts the server again)."""
    conns = [await asyncio.open_connection(sock=s) for s in socks]
    reader, writer = conns[0]
    stop = stop_signal()
    try:
        tts = await load_tts(args)
        async with tts:
            writer.write(frame(READY, 0, json.dumps({"voices": sorted(tts.voices),
                                                     "dialects": sorted(tts.dialects)}).encode()))
            kind, _, _ = await read_frame(reader)  # LISTENING
            if kind != LISTENING:
                raise RuntimeError("the connections' process did not start")
            announce(args)
            log = open(args.log + ".engine", "a", buffering=1) if args.log else None
            engine = asyncio.create_task(serve_engine(tts, reader, [w for _, w in conns], log))
            dead = asyncio.ensure_future(tts.server.wait_dead())
            await asyncio.wait([stop, engine, dead], return_when=asyncio.FIRST_COMPLETED)
            engine.cancel()
            died = dead.done()
            dead.cancel()
            if died:
                print("[ws_server] the engine process died: exiting (code 1)", flush=True)
                return 1
            return 0
    finally:
        try:
            os.kill(front_pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        await asyncio.to_thread(os.waitpid, front_pid, 0)


async def main_front(args, socks):
    """The connections' process."""
    loop = asyncio.get_running_loop()
    _, link = await loop.create_connection(EngineLink, sock=socks[0])
    for s in socks[1:]:
        await loop.create_connection(lambda: EngineLink(link), sock=s)
    info = await link.ready  # once the engine is up
    if info is None:
        return
    server = Server(set(info["voices"]), link.start, args, set(info["dialects"]) if info.get("dialects") else None)
    stop = stop_signal()
    until = asyncio.ensure_future(asyncio.wait([stop, link.closed], return_when=asyncio.FIRST_COMPLETED))
    await serve_ws(server, args, until, lambda: link.send(LISTENING))


def cli():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--model", default=None, help="the adapters model (default: RoutedDialectTTS's)")
    ap.add_argument("--tagged-model", default=None)
    ap.add_argument("--dit", default=None, help="distilled LocDiT (default: RoutedDialectTTS's; none: the model's own)")
    ap.add_argument("--head-tokens", type=int, default=None)
    ap.add_argument("--max-concurrency", type=int, default=None)
    ap.add_argument("--kv-blocks", type=int, default=None)
    ap.add_argument("--dialect", default="egyptian", help="the dialect of a connection that names none")
    ap.add_argument("--voice", action="append", default=[], metavar="NAME=PATH", help="register a voice at startup")
    ap.add_argument("--default-voice", default=None, help="the voice of a connection that names none (default: the "
                                                          "first --voice; none: the model's own)")
    ap.add_argument("--no-pace", dest="pace", action="store_false", help="send each chunk the moment it is complete")
    ap.add_argument("--pace-alpha", type=float, default=0.9, help="a paced gap, in chunks of audio")
    ap.add_argument("--pace-alpha-last", type=float, default=0.92,
                    help="the pace of the last complete chunk queued (the server empty after it): the engine's next "
                         "audio has the longest to come")
    ap.add_argument("--pace-reserve-ms", type=float, default=1000.0, help="audio queued beyond this goes out at "
                                                                          "once (-1: never)")
    ap.add_argument("--pace-spread-ms", type=float, default=10.0,
                    help="the chunk after the first (or a late one) goes out up to this much sooner, a different "
                         "amount per utterance: streams that start together do not send on the same instants")
    ap.add_argument("--pace-start-ms", type=float, default=150.0,
                    help="the first chunk this long after it is complete (later chunks and the engine's deadlines "
                         "with it): more time for the engine, as much later first audio at the client (150: no "
                         "stalls up to 150 streams at once on one A10, #31)")
    ap.add_argument("--pace-min-gap-ms", type=float, default=8.0,
                    help="chunks sent ahead of the pace (above the reserve, the rest at the end) at least this far apart")
    ap.add_argument("--link-sockets", type=int, default=8, help="socket pairs between the two processes (an engine "
                                                                    "step of 200 streams is ~1 MB of audio)")
    ap.add_argument("--log", default=None, help="jsonl, one line per utterance: the engine's yields and the sends")
    ap.add_argument("--ready", default=None, help="a file written once the server accepts connections")
    ap.add_argument("--api-key", default=os.environ.get("VOXCPM_API_KEY"),
                    help="the handshake needs 'Authorization: Bearer <key>' or ?api_key=<key> (default: "
                         "$VOXCPM_API_KEY; none: open to whoever reaches the port)")
    ap.add_argument("--max-streams", type=int, default=0, help="utterances at once; more get an error (0: no limit; "
                                                                "one A10: 150)")
    ap.add_argument("--max-text-chars", type=int, default=4096, help="longest text of an utterance")
    ap.add_argument("--ping-interval", type=float, default=20.0,
                    help="seconds between keepalive pings (a client that does not answer in as long is closed, and "
                         "its utterance's engine request ends; 0: off)")
    args = ap.parse_args()
    if args.default_voice is None and args.voice:
        args.default_voice = args.voice[0].partition("=")[0]
    try:  # a faster event loop: at 150-200 streams the loop's delays decide whether a paced chunk is late
        import uvloop

        factory = uvloop.new_event_loop
    except ImportError:
        factory = None
    import socket

    pairs = [socket.socketpair() for _ in range(args.link_sockets)]
    pid = os.fork()  # before the engine (CUDA, threads) starts: the connections' process
    if pid == 0:
        for a, _ in pairs:
            a.close()
        import gc

        # what this process inherited (imports) out of the collector's reach: a full collection over it paused the
        # connections' event loop for 29 ms at 150 at once, and paced chunks due meanwhile went out late
        gc.collect()
        gc.freeze()
        try:
            asyncio.run(main_front(args, [b for _, b in pairs]), loop_factory=factory)
        finally:
            os._exit(0)
    for _, b in pairs:
        b.close()
    if not args.api_key:
        print(f"[ws_server] no --api-key: whoever reaches {args.host}:{args.port} can use it", flush=True)
    sys.exit(asyncio.run(main_engine(args, [a for a, _ in pairs], pid), loop_factory=factory))


if __name__ == "__main__":
    cli()
