"""API clients (--api-key, --api-keys-file): a key each, under a name, with per-client limits (max_streams, rpm),
the file read again when it changes."""

import logging
import os

import pytest

from nanovllm_voxcpm.entrypoints.clients import Client, Clients, parse


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def keys_file(tmp_path, text: str, name: str = "keys.txt") -> str:
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def test_parse_lines_and_json():
    lines = parse("# name key [limits]\nacme sk-acme max_streams=2 rpm=60  # a comment\n\nbeta sk-beta metrics=true\n")
    as_json = parse('[{"name": "acme", "key": "sk-acme", "max_streams": 2, "rpm": 60}, '
                    '{"name": "beta", "key": "sk-beta", "metrics": true}]')
    assert lines == as_json == [Client("acme", b"sk-acme", 2, 60.0), Client("beta", b"sk-beta", 0, 0.0, True)]
    assert "sk-acme" not in repr(lines)  # (a key never in a log line or an error message)
    assert parse("ops sk-ops admin=true")[0].admin and not lines[0].admin
    assert Clients("k").identify(b"k").admin  # (--api-key's client: the operator's key)
    for bad in ("acme", "acme sk-1 fast", "acme sk-1 burst=3", "acme sk-1\nacme sk-2", "acme sk-1\nbeta sk-1",
                "acme sk-1 rpm=-1", '[{"name": "acme"}]'):
        with pytest.raises(ValueError):
            parse(bad)


def test_identify():
    clients = Clients("k")
    assert clients.enabled and clients.identify(b"k").name == "default" and clients.identify(b"K") is None
    assert clients.identify(b"") is None and Clients().identify(b"") is None and not Clients().enabled


def test_max_streams_and_rpm_per_client(tmp_path):
    clock = Clock()
    clients = Clients(keys_file=keys_file(tmp_path, "a ka max_streams=1\nb kb rpm=2\n"), clock=clock)
    a, b = clients.identify(b"ka"), clients.identify(b"kb")
    assert clients.acquire(a) == (None, 0.0)
    why, retry = clients.acquire(a)  # a's one stream in progress
    assert "max_streams" in why and retry == 1.0
    assert clients.acquire(b) == (None, 0.0) and clients.acquire(b) == (None, 0.0)  # (a's limit is not b's)
    why, retry = clients.acquire(b)  # b's 2 a minute used up: the next token in 30 s
    assert "rpm" in why and retry == pytest.approx(30.0)
    clients.release(a)
    assert clients.acquire(a) == (None, 0.0)
    clock.t += 15.0  # half a token back
    assert clients.acquire(b)[1] == pytest.approx(15.0)
    clock.t += 15.0
    assert clients.acquire(b) == (None, 0.0)
    clock.t += 3600.0  # a full bucket at most: 2
    assert [clients.acquire(b)[0] is None for _ in range(3)] == [True, True, False]
    assert clients.acquire(None) == (None, 0.0)  # (no keys: no client, no limits)


def test_the_keys_file_read_again_when_it_changes(tmp_path, caplog):
    clock = Clock()
    path = keys_file(tmp_path, "a ka max_streams=1\n")
    clients = Clients("k", path, clock=clock, log=logging.getLogger("test_clients"))
    a = clients.identify(b"ka")
    assert clients.acquire(a) == (None, 0.0) and clients.identify(b"k").name == "default"  # (--api-key too)
    with open(path + ".new", "w") as f:  # (replaced whole: written, then renamed)
        f.write("a ka2 max_streams=1\nc kc\n")
    os.replace(path + ".new", path)
    assert clients.identify(b"kc") is None  # (looked at no more than once a second)
    clock.t += Clients.RECHECK_S
    assert clients.identify(b"kc").name == "c" and clients.identify(b"ka") is None
    assert "max_streams" in clients.acquire(clients.identify(b"ka2"))[0]  # (a's stream in progress still counts)
    with open(path, "w") as f:
        f.write("a ka2 fast\n")
    clock.t += Clients.RECHECK_S
    with caplog.at_level(logging.ERROR, "test_clients"):
        assert clients.identify(b"kc").name == "c"  # (a file that does not parse: the clients read before stay)
    assert "the clients read before stay" in caplog.text and "ka2" not in caplog.text
    with pytest.raises(ValueError):
        Clients(keys_file=keys_file(tmp_path, "a\n", "bad.txt"))  # (at startup: an error)
