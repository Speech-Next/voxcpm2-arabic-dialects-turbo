"""The engine process's link to its server (models/voxcpm2/server.py), CPU only: the engine ends with the process that
started it, also when that one is killed (SIGKILL: the OOM killer, a container's stop timeout), instead of staying
behind with the GPU; an engine command's error reaches the caller as a ValueError (the request's fault) or a
RuntimeError without the engine's traceback."""

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

from nanovllm_voxcpm.models.voxcpm2.server import _engine_exception

REPO = Path(__file__).resolve().parents[1]

PARENT = textwrap.dedent("""
    import multiprocessing as mp, os, sys, time
    from types import SimpleNamespace


    def engine(q_in, q_out):  # main_loop on an engine with no model: it waits for commands
        os.environ["NANOVLLM_BUSY_POLL_S"] = "0"
        from nanovllm_voxcpm.models.voxcpm2 import server

        class Impl:
            def __init__(self, *a, **k):
                self.llm = SimpleNamespace(model_runner=SimpleNamespace())

        server.VoxCPM2ServerImpl = Impl
        server.main_loop(q_in, q_out, (), {})


    if __name__ == "__main__":
        from nanovllm_voxcpm.models.voxcpm2.server import _Channel

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

    from nanovllm_voxcpm.models.voxcpm2.server import AsyncVoxCPM2Server

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
