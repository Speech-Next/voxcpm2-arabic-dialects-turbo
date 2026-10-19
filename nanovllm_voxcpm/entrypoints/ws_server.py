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
  first chunk at once, the next ones on a grid of --pace-alpha x a chunk's audio, more than --pace-reserve-ms queued
  at once, the rest at the end at once. With 4096-byte chunks of 16 kHz the engine is told the chunk's length
  (stream's chunk_ms) so that its deadlines allow for what the server holds back.
"""

import argparse
import asyncio
import base64
import json
import os
import signal
import time
import zlib
from collections.abc import AsyncIterator
from urllib.parse import parse_qs, urlparse

import numpy as np

from nanovllm_voxcpm.streaming import ChunkPacer, paced

PATH = "/api/tts/stream"
FRAME_S = 0.04  # the AudioVAE's frame: a stream's audio is a whole number of them


def pcm16(y: np.ndarray) -> bytes:
    if y.dtype == np.int16:
        return y.tobytes()
    return (np.clip(y, -1.0, 1.0) * 32767.0).round().astype("<i2").tobytes()


async def rechunk(source: AsyncIterator[bytes], n: int) -> AsyncIterator[bytes]:
    """Unpaced: each chunk of n bytes the moment it is complete, the rest at the end."""
    buf = bytearray()
    async for data in source:
        buf += data
        while len(buf) >= n:
            yield bytes(buf[:n])
            del buf[:n]
    while buf:
        yield bytes(buf[:n])
        del buf[:n]


class Server:
    def __init__(self, tts, args):
        self.tts, self.args = tts, args
        self.log = open(args.log, "a") if args.log else None
        self.conn = 0

    def chunker(self, source: AsyncIterator[bytes], n: int, rate: int, chars: int) -> AsyncIterator[bytes]:
        a = self.args
        if not a.pace:
            return rechunk(source, n)
        expect = a.expect_per_char * chars if a.expect_per_char > 0 else None
        unit = max(2, round(FRAME_S * rate) * 2)  # the engine's audio comes in 40 ms frames
        return paced(source, ChunkPacer(n, 2 * rate, a.pace_alpha, a.pace_reserve_ms if a.pace_reserve_ms >= 0
                                        else None, expect_s=expect, margin_ms=a.pace_margin_ms, unit_bytes=unit,
                                        alpha_last=a.pace_alpha_last))

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
                        if obj.get("voice") is not None and obj["voice"] not in self.tts.voices:
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
        parts = [text]
        if cfg["segmenting"]:
            from nanovllm_voxcpm.entrypoints.openai_server import split_text

            parts = split_text(text) or [text]
        await ws.send(json.dumps({"type": "start", "sample_rate": rate}))
        rs = None
        if rate != self.tts.sample_rate:
            import soxr

            rs = soxr.ResampleStream(self.tts.sample_rate, rate, 1, dtype="float32")
        yields, sends = [], []
        chunk_ms = n / (2 * rate) * 1000

        async def pcm():
            for part in parts:
                seed = zlib.crc32(part.encode()) & 0x7FFFFFFF
                async for c in self.tts.stream(part, dialect=cfg["dialect"], voice=cfg["voice"], seed=seed,
                                               cfg_value=cfg["cfg_value"], chunk_ms=chunk_ms):
                    t = time.perf_counter()
                    out = pcm16(rs.resample_chunk(c, last=False) if rs is not None else c)
                    yields.append([t, int(c.size), (time.perf_counter() - t) * 1000, len(out)])
                    yield out
            if rs is not None:
                out = pcm16(rs.resample_chunk(np.zeros(0, np.float32), last=True))
                yields.append([time.perf_counter(), 0, 0.0, len(out)])
                yield out

        async for b in self.chunker(pcm(), n, rate, len(text)):
            await (ws.send(json.dumps({"type": "audio_chunk", "chunk_data": base64.b64encode(b).decode()}))
                   if b64 else ws.send(b))
            sends.append([time.perf_counter(), len(b)])
        await ws.send(json.dumps({"type": "end_of_audio"}))
        if self.log is not None:  # perf_counter (CLOCK_MONOTONIC) times: comparable with the engine's profiles
            # yields: [t, samples from the engine, ms resampling, bytes out]; sends: [t after the send, bytes]
            self.log.write(json.dumps({"conn": conn, "t_recv": t_recv, "t_end": time.perf_counter(),
                                       "chars": len(text), "dialect": cfg["dialect"], "chunk_size": n, "rate": rate,
                                       "paced": self.args.pace, "yields": yields, "sends": sends}) + "\n")
            self.log.flush()


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


async def main(args):
    from websockets.asyncio.server import serve

    from nanovllm_voxcpm import RoutedDialectTTS

    kw = {k: v for k, v in dict(model=args.model, tagged_model=args.tagged_model, head_tokens=args.head_tokens,
                                max_concurrency=args.max_concurrency, kv_blocks=args.kv_blocks).items()
          if v is not None}
    if args.dit is not None:
        kw["dit"] = args.dit or None
    tts = await RoutedDialectTTS.load(**kw)
    async with tts:
        for spec in args.voice:
            name, _, path = spec.partition("=")
            await tts.add_voice(name, path)
        if args.default_voice is None and args.voice:
            args.default_voice = args.voice[0].partition("=")[0]
        await tts.synthesize("أهلاً بيك، إزيك النهارده؟", dialect=args.dialect, voice=args.default_voice)  # warm up
        server = Server(tts, args)
        lag = asyncio.create_task(lag_monitor(args.lag_log)) if args.lag_log else None
        async with serve(server.handler, args.host, args.port, max_size=None, compression=None, ping_interval=None,
                         max_queue=None) as ws_server:
            announce(args)
            await stopped()
            ws_server.close()
        if lag is not None:
            lag.cancel()


def announce(args):
    if args.ready:
        with open(args.ready, "w") as f:
            f.write(str(os.getpid()))
    print(f"[ws_server] ws://{args.host}:{args.port}{PATH} ready (pacing "
          f"{'alpha %.2f, reserve %g ms' % (args.pace_alpha, args.pace_reserve_ms) if args.pace else 'off'}"
          f"{', uvloop' if 'uvloop' in type(asyncio.get_running_loop()).__module__ else ''})", flush=True)


async def stopped():
    """Until SIGTERM / SIGINT (then the engine is closed: its process holds the GPU)."""
    stop = asyncio.get_running_loop().create_future()
    for sig in (signal.SIGTERM, signal.SIGINT):
        asyncio.get_running_loop().add_signal_handler(sig, lambda: stop.done() or stop.set_result(None))
    await stop


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
    ap.add_argument("--pace-alpha", type=float, default=0.95, help="a paced gap, in chunks of audio")
    ap.add_argument("--pace-alpha-last", type=float, default=None,
                    help="the pace of the last complete chunk queued (the server empty after it; default: --pace-alpha)")
    ap.add_argument("--pace-reserve-ms", type=float, default=1000.0, help="audio queued beyond this goes out at "
                                                                          "once (-1: never)")
    ap.add_argument("--pace-margin-ms", type=float, default=4.0)
    ap.add_argument("--expect-per-char", type=float, default=0.0,
                    help="> 0: an utterance lasts at least this x its characters (s): a slower pace's ratio for "
                         "short ones keeps their gaps under their shorter average chunk (ChunkPacer.ratio_for)")
    ap.add_argument("--no-uvloop", dest="uvloop", action="store_false", help="asyncio's own event loop even when "
                                                                              "uvloop is installed")
    ap.add_argument("--log", default=None, help="jsonl, one line per utterance: the engine's yields and the sends")
    ap.add_argument("--lag-log", default=None, help="jsonl: the event loop's lag once a second")
    ap.add_argument("--ready", default=None, help="a file written once the server accepts connections")
    args = ap.parse_args()
    factory = None
    if args.uvloop:
        try:  # a faster event loop: at 200 streams the loop's delays decide whether a paced chunk is late
            import uvloop

            factory = uvloop.new_event_loop
        except ImportError:
            pass
    asyncio.run(main(args), loop_factory=factory)


if __name__ == "__main__":
    cli()
