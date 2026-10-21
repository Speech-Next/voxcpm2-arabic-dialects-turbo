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


TWO_PROCESSES = '''
import asyncio, sys
import numpy as np
from nanovllm_voxcpm.entrypoints import ws_server


class Pool:
    async def wait_dead(self):
        await asyncio.Future()


class TTS:  # 20 patches, 50 ms apart: 1 s of engine time per utterance
    sample_rate, voices, dialects, server = 48000, {"default": True}, ["egyptian"], Pool()

    async def stream(self, text, **kw):
        for _ in range(20):
            await asyncio.sleep(0.05)
            yield np.zeros(7680, np.float32)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        pass


async def load_tts(args):
    return TTS()


ws_server.load_tts = load_tts
sys.argv = ["ws_server", "--port", sys.argv[1], "--voice", "default=x.wav", "--link-sockets", "2"]
ws_server.cli()
'''


def test_sigterm_drains_and_exits_in_time(tmp_path):
    # SIGTERM took 31-41 s, more than Kubernetes' 30 s grace period: now the utterance in progress ends whole, then
    # both processes exit at once (idle: at once)
    import os
    import signal
    import socket
    import subprocess
    import sys
    import time
    from pathlib import Path

    from websockets.asyncio.client import connect

    script = tmp_path / "two.py"
    script.write_text(TWO_PROCESSES)
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1]), "CUDA_VISIBLE_DEVICES": ""}

    def start():
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        p = subprocess.Popen([sys.executable, str(script), str(port)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             env=env, text=True)
        for line in p.stdout:
            if "ready (pacing" in line:
                return p, port
        raise AssertionError("the server did not start")

    def stop(p) -> float:
        t = time.monotonic()
        p.send_signal(signal.SIGTERM)
        p.wait(timeout=30)
        return time.monotonic() - t

    p, _ = start()
    try:
        assert stop(p) < 2.0 and p.returncode == 0
    finally:
        p.kill()

    async def speak_then_stop(p, port):
        msgs = []
        async with connect(f"ws://127.0.0.1:{port}{PATH}") as ws:
            await ws.send("one")
            await ws.recv()  # start
            stopping = asyncio.ensure_future(asyncio.to_thread(stop, p))
            async for m in ws:
                msgs.append(json.loads(m)["type"] if isinstance(m, str) else len(m))
        return msgs, ws.close_code, await stopping

    p, port = start()
    try:
        msgs, code, took = asyncio.run(speak_then_stop(p, port))
    finally:
        p.kill()
    assert msgs[-1] == "end_of_audio" and sum(m for m in msgs if isinstance(m, int)) == 20 * PATCH // 3 * 2
    assert code == 1001 and took < 3.0 and p.returncode == 0


def metric_values(text: str) -> dict[str, float]:
    """Prometheus text: {'name{labels}': value}."""
    return {k: float(v) for k, v in (line.rsplit(" ", 1) for line in text.splitlines() if line and line[0] != "#")}


def test_metrics_and_logs(caplog):
    # every utterance's outcome counted, its audio timed and measured; /metrics on the same port, behind the API key;
    # a line logged per connection opened and closed and per utterance not spoken
    import logging
    import urllib.error
    import urllib.request

    caplog.set_level(logging.INFO, logger="ws_server")
    tts = PickyTTS(patches=2)
    srv = Server(set(tts.voices), local_feed(tts), args(api_key="k"))

    async def main():
        from websockets.asyncio.server import serve

        await exchange(srv, texts=["ok", "bad", "..."], headers={"Authorization": "Bearer k"})
        async with serve(srv.handler, "127.0.0.1", 0, process_request=srv.process_request) as s:
            url = f"http://127.0.0.1:{s.sockets[0].getsockname()[1]}/metrics"

            def get(headers):
                try:
                    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=5) as r:
                        return r.status, r.headers["Content-Type"], r.read().decode()
                except urllib.error.HTTPError as e:
                    return e.code, None, ""

            return await asyncio.to_thread(get, {}), await asyncio.to_thread(get, {"Authorization": "Bearer k"})

    (no_key, _, _), (status, ctype, text) = asyncio.run(main())
    assert no_key == 401 and status == 200 and ctype == "text/plain; version=0.0.4; charset=utf-8"
    m = metric_values(text)
    assert m['voxcpm_requests_total{outcome="ok"}'] == 1 and m['voxcpm_requests_total{outcome="client_error"}'] == 2
    assert m["voxcpm_connections_total"] == 1 and m["voxcpm_connections"] == 0 and m["voxcpm_active_streams"] == 0
    assert m["voxcpm_first_audio_seconds_count"] == 1 and m["voxcpm_first_chunk_seconds_count"] == 1
    assert m["voxcpm_audio_seconds_total"] == pytest.approx(2 * PATCH / 48000, abs=0.01)  # (16 kHz PCM16 sent)
    assert m["voxcpm_engine_alive"] == 1
    logged = [r.getMessage() for r in caplog.records if r.name == "ws_server"]
    assert any(msg.startswith("conn 1 open: 127.0.0.1:") for msg in logged)
    assert "conn 1: utterance refused: Request may exceed max_model_len" in logged
    assert "conn 1: utterance refused: text has nothing to speak (only punctuation, symbols or emoji)" in logged
    assert any(msg.startswith("conn 1 closed: ") and msg.endswith("1 utterances spoken, 2 failed") for msg in logged)


def test_an_engine_value_error_is_one_across_the_link():
    # the request's fault (e.g. a text too long for the model) crosses the two processes' link as a ValueError: the
    # client hears the same message, and the connections' process counts it as the request's fault
    import socket

    tts = PickyTTS(patches=1)

    async def main():
        a, b = socket.socketpair()
        reader, writer = await asyncio.open_connection(sock=a)
        engine = asyncio.create_task(serve_engine(tts, reader, writer))
        tb, link = await asyncio.get_running_loop().create_connection(EngineLink, sock=b)
        srv = Server({"default"}, link.start, args())
        msgs = await exchange(srv, texts=["bad", "x"])
        tb.close()
        await asyncio.wait_for(engine, 2)
        return msgs, metric_values(srv.metrics.render())

    msgs, m = asyncio.run(main())
    assert [x["message"] for x in msgs if isinstance(x, dict) and x["type"] == "error"] == [
        "Request may exceed max_model_len"]
    assert m['voxcpm_requests_total{outcome="client_error"}'] == 1 and m['voxcpm_requests_total{outcome="ok"}'] == 1
    assert m['voxcpm_requests_total{outcome="server_error"}'] == 0


def test_audio_cache_the_same_sentence_the_same_audio():
    """--audio-cache-mb: a sentence spoken again (another utterance, here segmented with another one) is played from
    the cache: the audio it got first, though the engine's would differ (another load), with no engine request."""
    from nanovllm_voxcpm.entrypoints.audio_cache import AudioCache
    from nanovllm_voxcpm.entrypoints.ws_server import speak

    tts, cache = FakeTTS(), AudioCache(1 << 24)

    async def stream(text, dialect=None, voice=None, **kw):  # other audio on every call
        tts.calls.append((text, dialect, voice, kw))
        for _ in range(3):
            yield np.random.uniform(-0.5, 0.5, PATCH).astype(np.float32)

    tts.stream = stream
    s1 = "النهارده الجو حلو قوي، تعالى نتمشى شوية على الكورنيش."  # (split_text: sentences of 40+ characters)
    s2 = "وبعدين نروح نشرب حاجة ساقعة في الكافيه اللي جنب البحر."
    req = dict(text=s1, dialect="egyptian", voice="default", cfg_value=2.0, rate=16000, chunk_ms=128, pace=None,
               segmenting=True)

    async def utter(**kw):
        return b"".join([b async for b in speak(tts, {**req, **kw}, cache=cache)])

    first, again = asyncio.run(utter()), asyncio.run(utter())
    assert first == again and len(first) > 0 and len(tts.calls) == 1
    both = asyncio.run(utter(text=f"{s1} {s2}"))  # its first sentence kept, the second generated
    assert both[:len(first) - 2000] == first[:-2000] and len(tts.calls) == 2  # (but the resampler's flushed tail)
    asyncio.run(utter(dialect="najdi"))  # another dialect: generated
    assert len(tts.calls) == 3
    assert cache.stats()["hits"] == 2 and cache.stats()["misses"] == 3


def test_audio_cache_across_the_link_in_the_metrics():
    """Two processes: the cache is the engine's (speak in serve_engine); an END frame tells its stats to the
    connections' process, whose /metrics exports them."""
    import socket

    from nanovllm_voxcpm.entrypoints.audio_cache import AudioCache

    tts = FakeTTS(patches=3)

    async def main():
        from websockets.asyncio.client import connect
        from websockets.asyncio.server import serve

        a, b = socket.socketpair()
        reader, writer = await asyncio.open_connection(sock=a)
        engine = asyncio.create_task(serve_engine(tts, reader, writer, cache=AudioCache(1 << 24)))
        loop = asyncio.get_running_loop()
        tb, link = await loop.create_connection(EngineLink, sock=b)
        srv = Server({"default"}, link.start, args(), cache_stats=lambda: link.cache_stats)
        async with serve(srv.handler, "127.0.0.1", 0) as s:
            port = s.sockets[0].getsockname()[1]
            for _ in range(2):
                async with connect(f"ws://127.0.0.1:{port}/api/tts/stream?chunk_size=4096") as ws:
                    await ws.send("x")
                    while True:
                        m = await ws.recv()
                        if isinstance(m, str) and "end_of_audio" in m:
                            break
        tb.close()
        await asyncio.wait_for(engine, 2)
        return srv.metrics.render()

    text = asyncio.run(main())
    assert len(tts.calls) == 1
    assert "voxcpm_audio_cache_hits_total 1" in text and "voxcpm_audio_cache_misses_total 1" in text
    assert "voxcpm_audio_cache_entries 1" in text and "voxcpm_audio_cache_joins_total 0" in text


def test_audio_cache_one_generation_for_utterances_at_the_same_time_and_a_directory(tmp_path):
    """Two utterances of the same sentence at the same time: one engine request, the same bytes; another process on
    --audio-cache-dir (a restart, a replica) plays it from the file; a voice is known by its file's content."""
    from nanovllm_voxcpm.entrypoints.audio_cache import AudioCache
    from nanovllm_voxcpm.entrypoints.ws_server import speak

    tts = FakeTTS()

    async def stream(text, dialect=None, voice=None, **kw):  # other audio on every call, 20 ms apart
        tts.calls.append((text, dialect, voice, kw))
        for _ in range(4):
            await asyncio.sleep(0.02)
            yield np.random.uniform(-0.5, 0.5, PATCH).astype(np.float32)

    tts.stream = stream
    req = dict(text="مرحبا", dialect="egyptian", voice="default", cfg_value=2.0, rate=16000, chunk_ms=128, pace=None,
               segmenting=False)

    async def utter(cache, ids=None):
        return b"".join([b async for b in speak(tts, req, cache=cache, voice_ids=ids)])

    async def first():
        cache = AudioCache(1 << 24, directory=str(tmp_path), namespace="ws")
        both = await asyncio.gather(utter(cache, {"default": "a1"}), utter(cache, {"default": "a1"}))
        await cache.idle()
        return both, cache.stats()

    (a, b), stats = asyncio.run(first())
    assert a == b and len(a) > 0 and len(tts.calls) == 1 and stats["misses"] == 1 and stats["joins"] == 1

    async def restarted(ids):
        cache = AudioCache(1 << 24, directory=str(tmp_path), namespace="ws")
        out = await utter(cache, ids)
        await cache.idle()
        return out, cache.stats()

    played, stats = asyncio.run(restarted({"default": "a1"}))
    assert played == a and len(tts.calls) == 1 and stats["disk_hits"] == 1
    other, stats = asyncio.run(restarted({"default": "b2"}))  # "default" now another file: another voice
    assert other != a and len(tts.calls) == 2 and stats["misses"] == 1


def test_client_keys_and_limits(tmp_path):
    """--api-keys-file: an unknown key is refused at the handshake (401); an utterance beyond its client's limits is
    an error (the connection goes on), another client's are not touched; counts by client in /metrics."""
    from websockets.exceptions import InvalidStatus

    keys = tmp_path / "keys.txt"
    keys.write_text("a ka max_streams=1\nb kb rpm=1\nm km metrics=true\n")
    tts = FakeTTS(patches=6, burst=1, pause=0.05)
    srv = Server(set(tts.voices), local_feed(tts), args(api_keys_file=str(keys)))

    async def main():
        out = {}
        try:
            await exchange(srv, "chunk_size=4096&api_key=kx", ["x"])
        except InvalidStatus as e:
            out["unknown"] = e.response.status_code
        first = asyncio.ensure_future(exchange(srv, texts=["one"], headers={"Authorization": "Bearer ka"}))
        await asyncio.sleep(0.1)
        out["a_second"] = await exchange(srv, texts=["two"], headers={"Authorization": "Bearer ka"})
        out["b"] = await exchange(srv, "chunk_size=4096&api_key=kb", ["one", "two", "three"])
        out["a_first"] = await first
        return out

    out = asyncio.run(main())
    assert out["unknown"] == 401 and kinds(out["a_first"]) == ["start", "end_of_audio"]
    assert kinds(out["a_second"]) == ["error"] and "client a" in out["a_second"][0]["message"]
    assert kinds(out["b"]) == ["start", "end_of_audio", "error", "error"]  # rpm=1: one; the connection goes on
    assert "rpm" in out["b"][-1]["message"] and "try again in" in out["b"][-1]["message"]
    text = srv.metrics.render()
    assert 'voxcpm_client_requests_total{client="a",outcome="limited"} 1' in text
    assert 'voxcpm_client_requests_total{client="b",outcome="ok"} 1' in text
    assert 'voxcpm_client_requests_total{client="b",outcome="limited"} 2' in text


def test_a_key_removed_from_the_file_ends_its_open_connection_s_utterances(tmp_path, monkeypatch):
    """A connection keeps its handshake's client, as the keys file has it now: a key removed (revoked) while its
    connection is open speaks no more."""
    from nanovllm_voxcpm.entrypoints.clients import Clients

    monkeypatch.setattr(Clients, "RECHECK_S", 0.0)
    keys = tmp_path / "keys.txt"
    keys.write_text("a ka\n")
    tts = FakeTTS(patches=1)
    srv = Server(set(tts.voices), local_feed(tts), args(api_keys_file=str(keys)))
    stream = tts.stream

    async def revoked_after(text, *a, **kw):
        async for c in stream(text, *a, **kw):
            yield c
        keys.write_text("z kz-new\n")

    tts.stream = revoked_after
    out = asyncio.run(exchange(srv, "chunk_size=4096&api_key=ka", ["one", "two"]))
    assert kinds(out) == ["start", "end_of_audio", "error"] and "no longer valid" in out[-1]["message"]
