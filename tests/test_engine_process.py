"""The engine process's link to its server (model/server.py), CPU only: the engine ends with the process that
started it, also when that one is killed (SIGKILL: the OOM killer, a container's stop timeout), instead of staying
behind with the GPU; an engine command's error reaches the caller as a ValueError (the request's fault) or a
RuntimeError without the engine's traceback."""

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

from nanovllm_voxcpm.model.server import _engine_exception

REPO = Path(__file__).resolve().parents[1]

PARENT = textwrap.dedent("""
    import multiprocessing as mp, os, sys, time
    from types import SimpleNamespace


    def engine(q_in, q_out):  # main_loop on an engine with no model: it waits for commands
        os.environ["NANOVLLM_BUSY_POLL_S"] = "0"
        from nanovllm_voxcpm.model import server

        class Impl:
            def __init__(self, *a, **k):
                self.llm = SimpleNamespace(model_runner=SimpleNamespace())

        server.VoxCPM2ServerImpl = Impl
        server.main_loop(q_in, q_out, (), {})


    if __name__ == "__main__":
        from nanovllm_voxcpm.model.server import _Channel

        ctx = mp.get_context("spawn")
        q_in, q_out = _Channel(ctx), _Channel(ctx)
        p = ctx.Process(target=engine, args=(q_in, q_out), daemon=True)
        p.start()
        assert q_out.get(timeout=120)["type"] == "init_ok"
        print(p.pid, flush=True)
        time.sleep(600)  # (until killed)
""")


def gone(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/status") as f:
            return any(line.startswith("State:") and "Z" in line.split()[1] for line in f)  # (a zombie)
    except FileNotFoundError:
        return True


def test_engine_ends_with_its_killed_parent(tmp_path):
    script = tmp_path / "parent.py"
    script.write_text(PARENT)
    env = {**os.environ, "PYTHONPATH": f"{REPO}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
    parent = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                              env=env, text=True)
    try:
        engine = int(parent.stdout.readline())
        assert not gone(engine)
        parent.kill()  # SIGKILL: no cleanup of its own
        parent.wait()
        deadline = time.time() + 10
        while not gone(engine) and time.time() < deadline:
            time.sleep(0.05)
        assert gone(engine), "the engine process outlived the process that started it"
    finally:
        parent.kill()


def test_engine_errors_for_the_caller():
    e = _engine_exception({"type": "error", "error": "Traceback ...", "kind": "ValueError", "message": "too long"})
    assert type(e) is ValueError and str(e) == "too long"
    e = _engine_exception({"type": "error", "error": "Traceback ...", "kind": "KeyError", "message": "'x'"})
    assert type(e) is RuntimeError and "Traceback" not in str(e) and "KeyError" in str(e)
    assert str(_engine_exception({"type": "error", "error": "old format"})) == "old format"


def test_a_second_cancel_leaves_no_stream_behind():
    # generate()'s finally sends the cancel and waits for the engine's answer; cancelled again meanwhile (a shutdown,
    # a timeout around the consumer), its stream's queue stayed in stream_table, filling with the stream's audio
    import asyncio
    from types import SimpleNamespace

    import numpy as np

    from nanovllm_voxcpm.model.server import AsyncVoxCPM2Server

    async def main():
        srv = object.__new__(AsyncVoxCPM2Server)
        srv.op_table, srv._pending_cmds, srv.stream_table = {}, [], {}
        sent = []
        srv.queue_in = SimpleNamespace(put_nowait=sent.append)

        async def consume():
            async for _ in srv.generate("text"):
                pass

        task = asyncio.ensure_future(consume())
        while not sent:
            await asyncio.sleep(0)
        srv._handle({"type": "response", "id": sent[0]["id"], "data": None})
        seq_id = sent[0]["args"][0]
        srv._handle({"type": "stream", "id": seq_id, "data": np.zeros(4, np.int16)})
        for _ in range(5):
            await asyncio.sleep(0)
        task.cancel()  # the client goes away
        while len(sent) < 2:
            await asyncio.sleep(0)
        task.cancel()  # and again, while the cancel waits for the engine
        try:
            await task
        except asyncio.CancelledError:
            pass
        return sent[1]["type"], seq_id in srv.stream_table

    assert asyncio.run(main()) == ("cancel", False)


FLOOD = textwrap.dedent("""
    import asyncio, multiprocessing as mp, os, sys, time
    from queue import Empty

    N = 1000  # commands of ~2 KB: ~2 MB in one event-loop turn


    def engine(q_in, q_out):  # main_loop's way: a step's output written blocking, commands read between steps
        q_in._q._writer.close()
        q_out._q._reader.close()
        got = 0
        while got < N:
            while True:
                try:
                    q_in.get_nowait()
                except Empty:
                    break
                got += 1
            q_out.put({"type": "audio", "data": b"x" * 20000})
        q_out.put({"type": "done", "got": got})


    async def client(q_in, q_out, blocking):
        from nanovllm_voxcpm.model.server import _PipeWriter

        loop = asyncio.get_running_loop()
        done, reader = loop.create_future(), q_out._q._reader

        def on_readable():
            while reader.poll():
                m = q_out.get()
                if m["type"] == "done" and not done.done():
                    done.set_result(m["got"])

        loop.add_reader(reader.fileno(), on_readable)
        os.set_blocking(reader.fileno(), True)
        time.sleep(1.0)  # the loop busy for a while: the engine fills its pipe and waits
        put = q_in.put if blocking else _PipeWriter(loop, q_in._q._writer).put
        for _ in range(N):  # a burst flushed before the loop reads again
            put({"type": "cmd", "pad": b"y" * 2000})
        return await done


    if __name__ == "__main__":
        from nanovllm_voxcpm.model.server import _Channel

        ctx = mp.get_context("spawn")
        q_in, q_out = _Channel(ctx), _Channel(ctx)
        p = ctx.Process(target=engine, args=(q_in, q_out), daemon=True)
        p.start()
        print(asyncio.run(client(q_in, q_out, sys.argv[1:] == ["blocking"])), flush=True)
""")


def test_both_pipes_full_is_no_deadlock(tmp_path):
    # the client writing its pipe blocking from its event loop (the engine's output unread meanwhile) and the engine
    # writing its own blocking waited for each other for good once both pipes were full
    script = tmp_path / "flood.py"
    script.write_text(FLOOD)
    env = {**os.environ, "PYTHONPATH": f"{REPO}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env, timeout=120)
    assert out.stdout.strip() == "1000", out.stderr[-2000:]


def _watched(open_streams: set, hang_timeout: float | None):
    """A client whose engine process never answers (e.g. a CUDA hang), watched for ~0.8 s: (killed, dead, what
    its stream's consumer got)."""
    import asyncio
    from types import SimpleNamespace

    from nanovllm_voxcpm.model.server import AsyncVoxCPM2Server

    class Process:
        exitcode, killed = None, False

        def kill(self):
            self.killed = True

    async def main():
        srv = object.__new__(AsyncVoxCPM2Server)
        srv.op_table, srv._pending_cmds, srv.stream_table = {}, [], {"s": asyncio.Queue()}
        srv.process, srv.dead, srv._reader = Process(), asyncio.Event(), SimpleNamespace(poll=lambda: False)
        srv._open, srv._last_msg, srv.hang_timeout = open_streams, time.monotonic(), hang_timeout
        srv._init_fut = asyncio.get_running_loop().create_future()
        srv._init_fut.set_result(None)  # (started)
        try:
            await asyncio.wait_for(srv._watch_engine(), 0.8)
        except asyncio.TimeoutError:
            pass
        q = srv.stream_table["s"]
        return srv.process.killed, srv.dead.is_set(), q.get_nowait() if q.qsize() else None

    return asyncio.run(main())


def test_a_hung_engine_is_killed_and_its_requests_fail():
    killed, dead, got = _watched({"s"}, 0.3)
    assert killed and dead and isinstance(got, RuntimeError) and "hung" in str(got)


def test_no_hang_without_work_owed_or_before_watch_hangs():
    # a stream whose end the engine sent, its consumer slow (e.g. an HTTP client reading at its own pace): the
    # engine owes nothing; and before watch_hangs (warm-up: kernels compiling) nothing is watched
    assert _watched(set(), 0.3) == (False, False, None)
    assert _watched({"s"}, None) == (False, False, None)
