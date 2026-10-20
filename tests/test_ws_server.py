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

from nanovllm_voxcpm.entrypoints.ws_server import PATH, EngineLink, Server, local_feed, serve_engine  # noqa: E402

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
    a = dict(pace=True, pace_alpha=0.95, pace_alpha_last=None, pace_reserve_ms=1000.0,
             dialect="egyptian", default_voice="default", log=None)
    a.update(kw)
    return argparse.Namespace(**a)


async def session(tts, a, query="audio_format=pcm_16000&chunk_size=4096&segmenting=false&transport=bytes",
                  texts=("مرحبا",), control=None, path="/api/tts/stream"):
    from websockets.asyncio.client import connect
    from websockets.asyncio.server import serve

    srv = Server(set(tts.voices), local_feed(tts), a)
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
    sent = tts.calls[0][3]["pace"]  # the engine runs the same pacer
    assert (sent["alpha"] == 0.95 and sent["reserve_ms"] == pytest.approx(1000.0)) if pace else sent == 0.0


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
        async with serve(Server(set(tts.voices), local_feed(tts), args()).handler, "127.0.0.1",
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
        async with serve(Server(set(tts.voices), local_feed(tts), a).handler, "127.0.0.1", 0) as s:
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

        pairs = [socket.socketpair() for _ in range(3)]  # utterances striped over 3 sockets
        conns = [await asyncio.open_connection(sock=a) for a, _ in pairs]
        engine = asyncio.create_task(serve_engine(tts, conns[0][0], [w for _, w in conns]))
        loop = asyncio.get_running_loop()
        tb, link = await loop.create_connection(EngineLink, sock=pairs[0][1])
        others = [(await loop.create_connection(lambda: EngineLink(link), sock=b))[0] for _, b in pairs[1:]]
        async with serve(Server({"default"}, link.start, args()).handler, "127.0.0.1", 0) as s:
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
        tb.close()
        for t in others:
            t.close()
        await asyncio.wait_for(engine, 2)
        return msgs

    msgs = asyncio.run(main())
    assert sum(len(m) for m in msgs if isinstance(m, bytes)) == 6 * PATCH // 3 * 2
    assert cancelled == [True]


# --- production: API key and /health at the handshake, dialects, control values, limits, errors per utterance ---


class PickyTTS(FakeTTS):
    """Fails the text "bad" (as an engine's ValueError would), slow otherwise (busy)."""

    async def stream(self, text, dialect=None, voice=None, **kw):
        if text == "bad":
            raise ValueError("Request may exceed max_model_len")
        async for c in super().stream(text, dialect, voice, **kw):
            yield c


async def exchange(srv, query="chunk_size=4096", texts=(), control=None, headers=None, path=PATH):
    """The messages a client gets (audio as its byte count) for ``texts``, one at a time."""
    from websockets.asyncio.client import connect
    from websockets.asyncio.server import serve

    async with serve(srv.handler, "127.0.0.1", 0, process_request=srv.process_request) as s:
        port = s.sockets[0].getsockname()[1]
        out = []
        async with connect(f"ws://127.0.0.1:{port}{path}?{query}", additional_headers=headers) as ws:
            if control is not None:
                await ws.send(json.dumps(control))
                out.append(json.loads(await ws.recv()))
            for t in texts:
                await ws.send(t)
                while True:
                    m = await ws.recv()
                    if isinstance(m, bytes):
                        out.append(len(m))
                        continue
                    out.append(json.loads(m))
                    if out[-1]["type"] in ("end_of_audio", "error"):
                        break
        return out



def kinds(msgs):
    return [m["type"] for m in msgs if isinstance(m, dict)]


def test_api_key_and_health():
    import urllib.request

    from websockets.exceptions import InvalidStatus

    tts = FakeTTS(patches=1)
    srv = Server(set(tts.voices), local_feed(tts), args(api_key="k"))

    async def main():
        from websockets.asyncio.server import serve

        out = []
        for headers, query in ((None, "chunk_size=4096"), ({"Authorization": "Bearer wrong"}, "chunk_size=4096"),
                               ({"Authorization": "Bearer k"}, "chunk_size=4096"), (None, "chunk_size=4096&api_key=k")):
            try:
                out.append(kinds(await exchange(srv, query, ["x"], headers=headers)))
            except InvalidStatus as e:
                out.append(e.response.status_code)
        async with serve(srv.handler, "127.0.0.1", 0, process_request=srv.process_request) as s:
            url = f"http://127.0.0.1:{s.sockets[0].getsockname()[1]}/health"
            health = await asyncio.to_thread(lambda: urllib.request.urlopen(url, timeout=5).status)
        return out, health

    out, health = asyncio.run(main())
    assert out == [401, 401, ["start", "end_of_audio"], ["start", "end_of_audio"]] and health == 200


def test_unknown_dialect_is_refused_at_the_handshake():
    tts = FakeTTS(patches=1)
    srv = Server(set(tts.voices), local_feed(tts), args(), dialects={"egyptian", "najdi"})
    from websockets.asyncio.client import connect
    from websockets.asyncio.server import serve

    async def main():
        async with serve(srv.handler, "127.0.0.1", 0) as s:
            port = s.sockets[0].getsockname()[1]
            async with connect(f"ws://127.0.0.1:{port}{PATH}?dialect=klingon") as ws:
                return json.loads(await ws.recv())

    err = asyncio.run(main())
    assert err["type"] == "error" and "klingon" in err["message"] and "najdi" in err["message"]


def test_control_values_are_checked():
    tts = FakeTTS(patches=1)
    srv = Server(set(tts.voices), local_feed(tts), args())
    for control in ({"type": "control", "cfg_value": "abc"}, {"type": "control", "cfg_value": float("nan")},
                    {"type": "control", "cfg_value": 1e9}, {"type": "control", "voice": ["default"]}):
        msgs = asyncio.run(exchange(srv, texts=["x"], control=control))
        assert kinds(msgs) == ["config_error", "start", "end_of_audio"], control  # (the connection goes on)
    msgs = asyncio.run(exchange(srv, texts=["x"], control={"type": "control", "cfg_value": 3, "voice": "default"}))
    assert kinds(msgs)[0] == "config_updated" and tts.calls[-1][3]["cfg_value"] == 3.0


def test_an_utterance_error_leaves_the_connection_open():
    tts = PickyTTS(patches=2)
    srv = Server(set(tts.voices), local_feed(tts), args(max_text_chars=5))
    msgs = asyncio.run(exchange(srv, texts=["bad", "x" * 6, "ok"]))
    errors = [m["message"] for m in msgs if isinstance(m, dict) and m["type"] == "error"]
    assert errors == ["Request may exceed max_model_len", "text: at most 5 characters (--max-text-chars)"]
    assert kinds(msgs)[-2:] == ["start", "end_of_audio"]  # "ok" spoken after both errors
    assert not any("Traceback" in e or "ValueError(" in e for e in errors)


def test_busy_beyond_max_streams():
    tts = FakeTTS(patches=6, burst=1, pause=0.05)
    srv = Server(set(tts.voices), local_feed(tts), args(max_streams=1))

    async def main():
        first = asyncio.ensure_future(exchange(srv, texts=["one"]))
        await asyncio.sleep(0.1)
        return await exchange(srv, texts=["two"]), await first

    second, first = asyncio.run(main())
    assert kinds(first) == ["start", "end_of_audio"]
    assert kinds(second) == ["error"] and "busy" in second[0]["message"]


def test_other_json_messages_are_not_spoken_and_the_query_is_checked():
    tts = FakeTTS(patches=1)
    srv = Server(set(tts.voices), local_feed(tts), args())

    async def ping_then_text():
        from websockets.asyncio.client import connect
        from websockets.asyncio.server import serve

        async with serve(srv.handler, "127.0.0.1", 0) as s:
            async with connect(f"ws://127.0.0.1:{s.sockets[0].getsockname()[1]}{PATH}?chunk_size=4096") as ws:
                await ws.send(json.dumps({"type": "ping"}))
                first = json.loads(await ws.recv())
                await ws.send("   ")
                empty = json.loads(await ws.recv())
            return first, empty

    first, empty = asyncio.run(ping_then_text())
    assert first["type"] == "error" and "ping" in first["message"] and not tts.calls
    assert empty == {"type": "error", "message": "text is empty"}  # (before any "start")

    async def first_message(query):
        from websockets.asyncio.client import connect
        from websockets.asyncio.server import serve

        async with serve(srv.handler, "127.0.0.1", 0) as s:
            async with connect(f"ws://127.0.0.1:{s.sockets[0].getsockname()[1]}{PATH}?{query}") as ws:
                return json.loads(await ws.recv())

    for query in ("chunk_size=2", "chunk_size=1000000000", "chunk_size=4096&transport=foo",
                  "chunk_size=4096&segmenting=yes"):
        assert asyncio.run(first_message(query))["type"] == "error", query


def test_nothing_to_speak_is_an_error_and_the_connection_goes_on():
    tts = FakeTTS(patches=1)
    srv = Server(set(tts.voices), local_feed(tts), args())
    msgs = asyncio.run(exchange(srv, texts=["...", "🙂!", "ok"]))
    errors = [m["message"] for m in msgs if isinstance(m, dict) and m["type"] == "error"]
    assert len(errors) == 2 and all("nothing to speak" in e for e in errors)
    assert kinds(msgs)[-2:] == ["start", "end_of_audio"] and [c[0] for c in tts.calls] == ["ok"]


def test_shutdown_drains_the_utterances_in_progress():
    # SIGTERM: no new utterance or connection; the one in progress ends whole (within --shutdown-timeout), then its
    # connection closes (1001, going away)
    from websockets.asyncio.client import connect
    from websockets.asyncio.server import serve

    tts = FakeTTS(patches=6, burst=1, pause=0.05)
    srv = Server(set(tts.voices), local_feed(tts), args())

    async def main():
        async with serve(srv.handler, "127.0.0.1", 0) as s:
            port = s.sockets[0].getsockname()[1]
            msgs = []
            async with connect(f"ws://127.0.0.1:{port}{PATH}") as ws:
                await ws.send("one")
                await asyncio.sleep(0.1)  # (speaking)
                drain = asyncio.ensure_future(srv.drain(s, 5.0))
                await asyncio.sleep(0)
                await ws.send("two")
                try:
                    async for m in ws:
                        msgs.append(json.loads(m)["type"] if isinstance(m, str) else len(m))
                except Exception:  # noqa: BLE001
                    pass
            await drain
            return msgs, ws.close_code

    msgs, code = asyncio.run(main())
    kinds_ = [m for m in msgs if isinstance(m, str)]
    assert kinds_ == ["start", "end_of_audio"] and code == 1001  # ("two" not spoken: the client goes elsewhere)
    assert sum(m for m in msgs if isinstance(m, int)) == 6 * PATCH // 3 * 2


def test_shutdown_cuts_what_does_not_end_in_time():
    from websockets.asyncio.client import connect
    from websockets.asyncio.server import serve

    tts = FakeTTS(patches=100, burst=1, pause=0.05)  # 5 s of engine time
    srv = Server(set(tts.voices), local_feed(tts), args())
    cut = []

    async def main():
        async with serve(srv.handler, "127.0.0.1", 0) as s:
            async with connect(f"ws://127.0.0.1:{s.sockets[0].getsockname()[1]}{PATH}") as ws:
                await ws.send("long")
                await asyncio.sleep(0.2)
                t = asyncio.get_running_loop().time()
                await srv.drain(s, 0.3, lambda: cut.append(True))
                return asyncio.get_running_loop().time() - t, srv.active

    took, active = asyncio.run(main())
    assert cut == [True] and 0.3 <= took < 1.0 and active == 1
