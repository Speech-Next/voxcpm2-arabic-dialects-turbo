"""The client reads the engine's output in its event loop (AsyncVoxCPM2Server._on_readable, #31): a message larger
than the pipe's buffer arrives whole also with an event loop whose add_reader makes the descriptor non-blocking
(uvloop, as ws_server runs): it failed half-read and the reader was dropped, the engine blocked on its write."""

import asyncio
import multiprocessing as mp
import os
import threading

from nanovllm_voxcpm.models.voxcpm2.server import AsyncVoxCPM2Server, _Channel


class _NonBlockingReaderLoop(asyncio.SelectorEventLoop):
    """As uvloop: add_reader sets the descriptor non-blocking."""

    def add_reader(self, fd, callback, *args):
        super().add_reader(fd, callback, *args)
        os.set_blocking(fd, False)


def test_a_message_larger_than_the_pipe_arrives_whole():
    ch = _Channel(mp.get_context("spawn"))
    got = []
    srv = AsyncVoxCPM2Server.__new__(AsyncVoxCPM2Server)  # (only the reading side: no engine)
    srv.queue_out, srv._convert, srv._handle = ch, (lambda res: res), got.append
    big = {"type": "tokenizer", "tokenizer": "x" * (8 << 20), "version": 1}

    def write():  # 8 MB through a 64 KB pipe: the reader finds it empty mid-message many times
        ch.put(big)
        ch.put({"type": "done"})

    async def main():
        srv._attach_reader(asyncio.get_running_loop(), ch._q._reader)
        threading.Thread(target=write, daemon=True).start()
        while len(got) < 2:
            await asyncio.sleep(0.01)

    asyncio.run(asyncio.wait_for(main(), 30), loop_factory=_NonBlockingReaderLoop)
    assert got[0] == big and got[1] == {"type": "done"}
