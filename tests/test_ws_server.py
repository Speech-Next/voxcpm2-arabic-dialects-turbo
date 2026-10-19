"""The WebSocket server (entrypoints/ws_server.py) on a fake engine over a real localhost connection: the protocol of
#28's benchmark (start, chunks of chunk_size bytes, end_of_audio; control; json_base64), paced and unpaced, with
the engine in the same event loop and across the two processes' link (a socket pair)."""

import argparse
import asyncio
import base64
import json

import numpy as np
import pytest

websockets = pytest.importorskip("websockets")
pytest.importorskip("soxr")

from nanovllm_voxcpm.entrypoints.ws_server import EngineLink, Server, serve_engine, speak  # noqa: E402

PATCH = 7680  # 160 ms at 48 kHz


class FakeTTS:
    sample_rate = 48000

    def __init__(self, patches=6, burst=3, pause=0.05):
        self.voices = {"default": True}
        self.patches, self.burst, self.pause = patches, burst, pause
        self.calls = []

    async def stream(self, text, dialect=None, voice=None, **kw):
        self.calls.append((text, dialect, voice, kw))
        for k in range(self.patches):
            if k and k % self.burst == 0:
                await asyncio.sleep(self.pause)  # the engine busy with other streams
            yield np.full(PATCH, 0.1, np.float32)


def args(**kw):
    a = dict(pace=True, pace_alpha=0.95, pace_alpha_last=None, pace_reserve_ms=1000.0, pace_margin_ms=4.0,
             expect_per_char=0.0, dialect="egyptian", default_voice="default", log=None)
    a.update(kw)
    return argparse.Namespace(**a)


async def session(tts, a, query="audio_format=pcm_16000&chunk_size=4096&segmenting=false&transport=bytes",
                  texts=("مرحبا",), control=None, path="/api/tts/stream"):
    from websockets.asyncio.client import connect
    from websockets.asyncio.server import serve

    srv = Server(set(tts.voices), lambda req, y: speak(tts, req, y), a)
    async with serve(srv.handler, "127.0.0.1", 0) as s:
        port = s.sockets[0].getsockname()[1]
        msgs = []
        async with connect(f"ws://127.0.0.1:{port}{path}?{query}", max_size=None) as ws:
            if control is not None:
                await ws.send(json.dumps(control))
                msgs.append(json.loads(await ws.recv()))
            for t in texts:
                await ws.send(t)
            ends = 0
            while ends < len(texts):
                m = await ws.recv()
                if isinstance(m, str):
                    obj = json.loads(m)
                    msgs.append(obj)
                    ends += obj.get("type") in ("end_of_audio", "error")
                else:
                    msgs.append(m)
    return msgs


@pytest.mark.parametrize("pace", [True, False])
def test_protocol_chunks_and_end(pace):
    tts = FakeTTS()
    msgs = asyncio.run(session(tts, args(pace=pace)))
    assert msgs[0] == {"type": "start", "sample_rate": 16000}
    assert msgs[-1] == {"type": "end_of_audio"}
    chunks = msgs[1:-1]
    total = 6 * PATCH // 3 * 2  # 16 kHz PCM16
    assert sum(map(len, chunks)) == total
    assert all(len(c) == 4096 for c in chunks[:-1]) and 0 < len(chunks[-1]) <= 4096
    assert tts.calls[0][1:3] == ("egyptian", "default") and tts.calls[0][3]["chunk_ms"] == 128.0


def test_control_dialect_and_json_base64():
    tts = FakeTTS(patches=2)
    msgs = asyncio.run(session(tts, args(), query="audio_format=pcm_16000&chunk_size=2048&transport=json_base64"
                                                  "&dialect=najdi",
                               control={"type": "control", "voice": "default", "cfg_value": 1.5},
                               texts=("a", "b")))
    assert msgs[0] == {"type": "config_updated"}
    assert [m["type"] for m in msgs if m["type"] != "audio_chunk"] == ["config_updated", "start", "end_of_audio",
                                                                         "start", "end_of_audio"]
    data = [base64.b64decode(m["chunk_data"]) for m in msgs if m["type"] == "audio_chunk"]
    assert sum(map(len, data)) == 2 * (2 * PATCH // 3 * 2)
    assert [c[0] for c in tts.calls] == ["a", "b"] and tts.calls[0][1] == "najdi"
    assert tts.calls[0][3]["cfg_value"] == 1.5


def test_unknown_voice_is_a_config_error():
    msgs = asyncio.run(session(FakeTTS(patches=1), args(), control={"type": "control", "voice": "nobody"}))
    assert msgs[0]["type"] == "config_error"


def test_bad_format_is_an_error():
    async def main():
        from websockets.asyncio.client import connect
        from websockets.asyncio.server import serve

        tts = FakeTTS()
        async with serve(Server(set(tts.voices), lambda req, y: speak(tts, req, y), args()).handler, "127.0.0.1",
                         0) as s:
            port = s.sockets[0].getsockname()[1]
            async with connect(f"ws://127.0.0.1:{port}/api/tts/stream?audio_format=mp3") as ws:
                return json.loads(await ws.recv())

    assert asyncio.run(main())["type"] == "error"


def test_paced_gaps_stay_under_a_chunk_through_the_engine_pause():
    # 3 patches at once, a 50 ms pause, 3 more: unpaced, the client sees the pause as one long gap after a burst;
    # paced, the server spreads the lead so that no gap is longer than a chunk (128 ms)
    import time

    async def timed(a):
        from websockets.asyncio.client import connect
        from websockets.asyncio.server import serve

        tts = FakeTTS(patches=6, burst=3, pause=0.3)
        async with serve(Server(set(tts.voices), lambda req, y: speak(tts, req, y), a).handler, "127.0.0.1", 0) as s:
            port = s.sockets[0].getsockname()[1]
            ts = []
            async with connect(f"ws://127.0.0.1:{port}/api/tts/stream?audio_format=pcm_16000&chunk_size=4096") as ws:
                await ws.send("x")
                while True:
                    m = await ws.recv()
                    if isinstance(m, bytes):
                        ts.append(time.perf_counter())
                    elif json.loads(m)["type"] == "end_of_audio":
                        break
        return [b - a for a, b in zip(ts, ts[1:])]

    unpaced = asyncio.run(timed(args(pace=False)))
    paced_gaps = asyncio.run(timed(args(pace=True)))
    assert max(unpaced) > 0.25
    assert max(paced_gaps[:-2]) < 0.128  # (the end: what is left goes out at once)



def test_through_the_two_processes_link():
    # serve_engine on one end of a socket pair, EngineLink on the other (in one loop here; two processes in the
    # server): the same protocol, and a client that leaves cancels its engine request
    import socket

    tts = FakeTTS(patches=6, burst=3, pause=0.02)
    cancelled = []
    orig = tts.stream

    async def stream(*a, **kw):
        try:
            async for c in orig(*a, **kw):
                yield c
        except BaseException:
            cancelled.append(True)
            raise

    tts.stream = stream

    async def main():
        from websockets.asyncio.client import connect
        from websockets.asyncio.server import serve

        sa, sb = socket.socketpair()
        ra, wa = await asyncio.open_connection(sock=sa)
        rb, wb = await asyncio.open_connection(sock=sb)
        engine = asyncio.create_task(serve_engine(tts, ra, wa))
        link = EngineLink(rb, wb)
        async with serve(Server({"default"}, link.stream, args()).handler, "127.0.0.1", 0) as s:
            port = s.sockets[0].getsockname()[1]
            url = f"ws://127.0.0.1:{port}/api/tts/stream?chunk_size=4096"
            async with connect(url) as ws:
                await ws.send("x")
                msgs = []
                while not (msgs and isinstance(msgs[-1], str) and "end_of_audio" in msgs[-1]):
                    msgs.append(await ws.recv())
            tts.patches = 1000  # a long one, left after its first chunk
            async with connect(url) as ws:
                await ws.send("y")
                while not isinstance(await ws.recv(), bytes):
                    pass
            for _ in range(100):
                if cancelled:
                    break
                await asyncio.sleep(0.01)
        wb.close()
        await asyncio.wait_for(engine, 2)
        return msgs

    msgs = asyncio.run(main())
    assert sum(len(m) for m in msgs if isinstance(m, bytes)) == 6 * PATCH // 3 * 2
    assert cancelled == [True]
