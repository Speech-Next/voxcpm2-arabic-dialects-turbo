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
  same audio).
- Pacing (default; --no-pace sends each chunk the moment it is complete): streaming.ChunkPacer per utterance, the
  first chunk at once, the next ones on a grid of --pace-alpha x a chunk's audio (--pace-alpha-last for the last
  chunk queued), more than --pace-reserve-ms queued at once, the rest at the end at once. The engine is told the
  chunk's length (stream's chunk_ms) so that its deadlines allow for what the server holds back.
- Two processes (default; --procs 1: one): the connections and their pacers run in a process of their own, the
  engine's client (RoutedDialectTTS) in this one, the audio crossing over a socket pair. At 200 streams the client's
  event loop is busy 10-20 ms after every engine step (a patch of every stream at once), and a paced chunk due
  meanwhile went out that much late: a stall where the server had the audio.
"""

import argparse
import asyncio
import base64
import json
import os
import signal
import struct
import time
import zlib
from collections.abc import AsyncIterator, Callable
from urllib.parse import parse_qs, urlparse

import numpy as np

from nanovllm_voxcpm.streaming import ChunkPacer, ChunkStream, StreamSender

PATH = "/api/tts/stream"
FRAME_S = 0.04  # the AudioVAE's frame: a stream's audio is a whole number of them


def ws_frame(opcode: int, payload: bytes) -> bytes:
    """A whole, unmasked WebSocket frame (server to client; no extensions): opcode 1 text, 2 binary."""
    n = len(payload)
    if n < 126:
        return bytes((0x80 | opcode, n)) + payload
    if n < 65536:
        return struct.pack("!BBH", 0x80 | opcode, 126, n) + payload
    return struct.pack("!BBQ", 0x80 | opcode, 127, n) + payload


def pcm16(y: np.ndarray) -> bytes:
    if y.dtype == np.int16:
        return y.tobytes()
    return (np.clip(y, -1.0, 1.0) * 32767.0).round().astype("<i2").tobytes()


async def speak(tts, req: dict, yields: list | None = None) -> AsyncIterator[bytes]:
    """One utterance's PCM16 at req["rate"]: req = text, dialect, voice, cfg_value, rate, chunk_ms, segmenting.
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
                                  cfg_value=req["cfg_value"], chunk_ms=req["chunk_ms"]):
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


class Server:
    """The connections. voices: the registered voice names; feed(req, yields, stream): fills an utterance's
    ChunkStream with its PCM16 and returns what stops it."""

    def __init__(self, voices, feed: Feed, args):
        self.voices, self.feed, self.args = voices, feed, args
        self.log = open(args.log, "a") if args.log else None
        self.conn = 0
        # paced chunks of every connection from one task (StreamSender), written to the transports as whole frames
        self.sender = (StreamSender(getattr(args, "send_window_ms", 2.0) / 1000)
                       if args.pace and getattr(args, "batch_sends", True) else None)
        self.utterances = 0

    def pacer(self, n: int, rate: int, chars: int) -> ChunkPacer:
        a = self.args
        if not a.pace:  # each chunk the moment it is complete
            return ChunkPacer(n, 2 * rate, alpha=0.0, reserve_ms=None)
        expect = a.expect_per_char * chars if a.expect_per_char > 0 else None
        unit = max(2, round(FRAME_S * rate) * 2)  # the engine's audio comes in 40 ms frames
        self.utterances += 1  # each its own phase in [0, --pace-spread-ms) (golden-ratio sequence)
        phase = getattr(a, "pace_spread_ms", 0.0) * ((self.utterances * 0.6180339887) % 1.0)
        return ChunkPacer(n, 2 * rate, a.pace_alpha, a.pace_reserve_ms if a.pace_reserve_ms >= 0 else None,
                          expect_s=expect, margin_ms=a.pace_margin_ms, unit_bytes=unit, alpha_last=a.pace_alpha_last,
                          phase_ms=phase)

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
        if not 8000 <= rate <= 48000 or n <= 0 or n % 2:
            await ws.send(json.dumps({"type": "error", "message": "audio_format pcm_<8000-48000>, chunk_size even"}))
            await ws.close()
            return
        cfg = dict(rate=rate, chunk_size=n, transport=q.get("transport", "bytes"),
                   segmenting=q.get("segmenting", "false").lower() == "true",
                   dialect=q.get("dialect", self.args.dialect), voice=self.args.default_voice, cfg_value=2.0)
        self.conn += 1
        conn = self.conn
        texts: asyncio.Queue = asyncio.Queue()

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
                        if obj.get("voice") is not None and obj["voice"] not in self.voices:
                            await ws.send(json.dumps({"type": "config_error", "message": f"voice {obj['voice']!r}"}))
                            continue
                        cfg["voice"] = obj.get("voice", cfg["voice"])
                        if obj.get("cfg_value") is not None:
                            cfg["cfg_value"] = float(obj["cfg_value"])
                        await ws.send(json.dumps({"type": "config_updated"}))
                        continue
                    await texts.put((time.perf_counter(), msg))
            finally:
                await texts.put(None)

        rd = asyncio.create_task(reader())
        try:
            while (item := await texts.get()) is not None:
                await self.utterance(ws, conn, cfg, *item)
        except Exception as e:  # noqa: BLE001 - the client hears why
            try:
                await ws.send(json.dumps({"type": "error", "message": repr(e)}))
            except Exception:  # noqa: BLE001
                pass
        finally:
            rd.cancel()

    async def utterance(self, ws, conn, cfg, t_recv, text):
        n, rate, b64 = cfg["chunk_size"], cfg["rate"], cfg["transport"] == "json_base64"
        await ws.send(json.dumps({"type": "start", "sample_rate": rate}))
        req = dict(text=text, dialect=cfg["dialect"], voice=cfg["voice"], cfg_value=cfg["cfg_value"], rate=rate,
                   chunk_ms=n / (2 * rate) * 1000, segmenting=cfg["segmenting"])
        yields, sends = [], []
        stream = ChunkStream(self.pacer(n, rate, len(text)))
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


async def serve_engine(tts, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, log=None) -> None:
    """The engine's side: START (an utterance's request, JSON) -> DATA frames, then END (or ERROR); CANCEL stops
    one. Returns when the connections' side closes. log: a file, one jsonl line per utterance (perf_counter times:
    the START frame read, the task started, the yields [t, samples, ms resampling, bytes])."""
    jobs: dict[int, asyncio.Task] = {}

    async def run(utt: int, req: dict, t_start: float):
        t_task, yields = time.perf_counter(), []
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
            writer.write(frame(ERROR, utt, repr(e).encode()))
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
    arrives at once). start(req, yields, stream) is a Feed."""

    def __init__(self):
        loop = asyncio.get_running_loop()
        self.ready, self.closed = loop.create_future(), loop.create_future()
        self.buf = bytearray()
        self.sinks: dict[int, tuple[ChunkStream, list | None]] = {}
        self.next_utt = 0
        self.transport = None

    def connection_made(self, transport):
        self.transport = transport

    def data_received(self, data: bytes) -> None:
        self.buf += data
        t, off, buf, size = time.perf_counter(), 0, self.buf, _HDR.size
        while len(buf) - off >= size:
            kind, utt, n = _HDR.unpack_from(buf, off)
            if len(buf) - off - size < n:
                break
            payload = bytes(buf[off + size:off + size + n])
            off += size + n
            if kind == READY:
                if not self.ready.done():
                    self.ready.set_result(json.loads(payload))
                continue
            sink = self.sinks.get(utt)
            if sink is None:
                continue
            stream, yields = sink
            if kind == DATA:
                if yields is not None:
                    yields.append([t, 0, 0.0, len(payload)])
                stream.push(payload, t)
            else:
                del self.sinks[utt]
                if kind == END:
                    stream.finish()
                else:
                    stream.fail(RuntimeError(payload.decode(errors="replace")))
        del buf[:off]

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


async def lag_monitor(path: str, every: float = 0.002):
    """The event loop's lag (a 2 ms sleep's overshoot), one jsonl line a second: p50, p99, max (ms)."""
    with open(path, "a") as f:
        lags, t_line = [], time.perf_counter()
        while True:
            t = time.perf_counter()
            await asyncio.sleep(every)
            now = time.perf_counter()
            lags.append((now - t - every) * 1000)
            if now - t_line >= 1.0:
                lags.sort()
                f.write(json.dumps({"t": now, "p50": lags[len(lags) // 2], "p99": lags[int(len(lags) * 0.99)],
                                    "max": lags[-1], "n": len(lags)}) + "\n")
                f.flush()
                lags, t_line = [], now


async def load_tts(args):
    from nanovllm_voxcpm import RoutedDialectTTS

    kw = {k: v for k, v in dict(model=args.model, tagged_model=args.tagged_model, head_tokens=args.head_tokens,
                                max_concurrency=args.max_concurrency, kv_blocks=args.kv_blocks).items()
          if v is not None}
    if args.dit is not None:
        kw["dit"] = args.dit or None
    tts = await RoutedDialectTTS.load(**kw)
    for spec in args.voice:
        name, _, path = spec.partition("=")
        await tts.add_voice(name, path)
    await tts.synthesize("أهلاً بيك، إزيك النهارده؟", dialect=args.dialect, voice=args.default_voice)  # warm up
    return tts


async def serve_ws(server: Server, args, until: asyncio.Future, on_listening: Callable[[], None]):
    from websockets.asyncio.server import serve

    lag = asyncio.create_task(lag_monitor(args.lag_log)) if args.lag_log else None
    async with serve(server.handler, args.host, args.port, max_size=None, compression=None, ping_interval=None,
                     max_queue=None) as ws_server:
        on_listening()
        await until
        ws_server.close()
    if lag is not None:
        lag.cancel()


def announce(args):
    if args.ready:
        with open(args.ready, "w") as f:
            f.write(str(os.getpid()))
    print(f"[ws_server] ws://{args.host}:{args.port}{PATH} ready (pacing "
          f"{'alpha %.2f, reserve %g ms' % (args.pace_alpha, args.pace_reserve_ms) if args.pace else 'off'}, "
          f"{args.procs} process{'es' if args.procs > 1 else ''}"
          f"{', uvloop' if 'uvloop' in type(asyncio.get_running_loop()).__module__ else ''})", flush=True)


def stop_signal() -> asyncio.Future:
    """Set on SIGTERM / SIGINT (then the engine is closed: its process holds the GPU)."""
    stop = asyncio.get_running_loop().create_future()
    for sig in (signal.SIGTERM, signal.SIGINT):
        asyncio.get_running_loop().add_signal_handler(sig, lambda: stop.done() or stop.set_result(None))
    return stop


async def main_one(args):
    """--procs 1: the connections on the engine client's event loop."""
    tts = await load_tts(args)
    async with tts:
        server = Server(set(tts.voices), local_feed(tts), args)
        await serve_ws(server, args, stop_signal(), lambda: announce(args))


async def main_engine(args, sock, front_pid: int):
    """--procs 2, this process: the engine's client, serving the connections' process."""
    reader, writer = await asyncio.open_connection(sock=sock)
    stop = stop_signal()
    try:
        tts = await load_tts(args)
        async with tts:
            writer.write(frame(READY, 0, json.dumps({"voices": sorted(tts.voices)}).encode()))
            kind, _, _ = await read_frame(reader)  # LISTENING
            if kind != LISTENING:
                raise RuntimeError("the connections' process did not start")
            announce(args)
            lag = asyncio.create_task(lag_monitor(args.lag_log + ".engine")) if args.lag_log else None
            log = open(args.log + ".engine", "a", buffering=1) if args.log else None
            engine = asyncio.create_task(serve_engine(tts, reader, writer, log))
            await asyncio.wait([stop, engine], return_when=asyncio.FIRST_COMPLETED)
            engine.cancel()
            if lag is not None:
                lag.cancel()
    finally:
        try:
            os.kill(front_pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        await asyncio.to_thread(os.waitpid, front_pid, 0)


async def main_front(args, sock):
    """--procs 2, the connections' process."""
    _, link = await asyncio.get_running_loop().create_connection(EngineLink, sock=sock)
    info = await link.ready  # once the engine is up
    if info is None:
        return
    server = Server(set(info["voices"]), link.start, args)
    stop = stop_signal()
    until = asyncio.ensure_future(asyncio.wait([stop, link.closed], return_when=asyncio.FIRST_COMPLETED))
    await serve_ws(server, args, until, lambda: link.send(LISTENING))


def cli():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--model", default=None, help="the adapters model (default: RoutedDialectTTS's)")
    ap.add_argument("--tagged-model", default=None)
    ap.add_argument("--dit", default=None, help="distilled LocDiT (default: RoutedDialectTTS's; '' = the original)")
    ap.add_argument("--head-tokens", type=int, default=None)
    ap.add_argument("--max-concurrency", type=int, default=None)
    ap.add_argument("--kv-blocks", type=int, default=None)
    ap.add_argument("--dialect", default="egyptian", help="the dialect of a connection that names none")
    ap.add_argument("--voice", action="append", default=[], metavar="NAME=PATH", help="register a voice at startup")
    ap.add_argument("--default-voice", default=None, help="the voice of a connection that names none (default: the "
                                                          "first --voice; none: the model's own)")
    ap.add_argument("--no-pace", dest="pace", action="store_false", help="send each chunk the moment it is complete")
    ap.add_argument("--pace-alpha", type=float, default=0.93, help="a paced gap, in chunks of audio")
    ap.add_argument("--pace-alpha-last", type=float, default=0.95,
                    help="the pace of the last complete chunk queued (the server empty after it): the engine's next "
                         "audio has the longest to come")
    ap.add_argument("--pace-reserve-ms", type=float, default=1000.0, help="audio queued beyond this goes out at "
                                                                          "once (-1: never)")
    ap.add_argument("--pace-margin-ms", type=float, default=4.0)
    ap.add_argument("--expect-per-char", type=float, default=0.0,
                    help="> 0: an utterance lasts at least this x its characters (s): a slower pace's ratio for "
                         "short ones keeps their gaps under their shorter average chunk (ChunkPacer.ratio_for)")
    ap.add_argument("--send-window-ms", type=float, default=2.0, help="paced chunks due within this go out together")
    ap.add_argument("--pace-spread-ms", type=float, default=0.0,
                    help="the chunk after the first (or a late one) goes out up to this much sooner, a different "
                         "amount per utterance: streams that start together do not send on the same instants")
    ap.add_argument("--per-stream-sends", dest="batch_sends", action="store_false",
                    help="each connection sends its own paced chunks (default: one task sends every due chunk)")
    ap.add_argument("--procs", type=int, choices=(1, 2), default=2,
                    help="2: the connections in a process of their own (default), 1: on the engine client's loop")
    ap.add_argument("--no-uvloop", dest="uvloop", action="store_false", help="asyncio's own event loop even when "
                                                                              "uvloop is installed")
    ap.add_argument("--log", default=None, help="jsonl, one line per utterance: the engine's yields and the sends")
    ap.add_argument("--lag-log", default=None, help="jsonl: the event loop's lag once a second")
    ap.add_argument("--ready", default=None, help="a file written once the server accepts connections")
    args = ap.parse_args()
    if args.default_voice is None and args.voice:
        args.default_voice = args.voice[0].partition("=")[0]
    factory = None
    if args.uvloop:
        try:  # a faster event loop: at 200 streams the loop's delays decide whether a paced chunk is late
            import uvloop

            factory = uvloop.new_event_loop
        except ImportError:
            pass
    if args.procs == 1:
        asyncio.run(main_one(args), loop_factory=factory)
        return
    import socket

    a, b = socket.socketpair()
    pid = os.fork()  # before the engine (CUDA, threads) starts: the connections' process
    if pid == 0:
        a.close()
        import gc

        # what this process inherited (imports) out of the collector's reach: a full collection over it paused the
        # connections' event loop for 29 ms at 150 at once, and paced chunks due meanwhile went out late
        gc.collect()
        gc.freeze()
        try:
            asyncio.run(main_front(args, b), loop_factory=factory)
        finally:
            os._exit(0)
    b.close()
    asyncio.run(main_engine(args, a, pid), loop_factory=factory)


if __name__ == "__main__":
    cli()
