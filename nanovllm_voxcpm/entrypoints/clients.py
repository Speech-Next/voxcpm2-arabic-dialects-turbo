"""API clients of both servers: who may call them (an API key each, under a name) and how much (per-client limits).

``--api-key`` (or VOXCPM_API_KEY) is one client, named "default". ``--api-keys-file`` names a client per key, each
with optional limits, in lines (``#`` comments)::

    # name   key          [max_streams=N] [rpm=N] [metrics=true]
    acme     sk-acme-1    max_streams=20  rpm=600
    beta     sk-beta-2    rpm=60
    monitor  sk-prom-3    metrics=true

or JSON: ``[{"name": "acme", "key": "sk-acme-1", "max_streams": 20, "rpm": 600}, ...]``.

- max_streams: the client's requests (utterances) at once; more get 429 (an error for that utterance).
- rpm: its requests a minute, a token bucket of rpm tokens (a minute's worth) refilled at rpm / 60 a second; more get
  429 with the seconds until the next one (Retry-After).
- metrics: whether its key reads ``GET /metrics`` (the clients' names and counts); --api-key's client does.

The file is read again when it changes (its modification time and size, looked at no more than once a second, on a
request): keys rotate and limits change without a restart (replace the file whole, e.g. write and rename). A file
that does not parse keeps the clients read before (logged); a client's requests in progress keep counting across a
reload. Keys are compared in constant time and never logged: logs and metrics carry the client's name.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import time
from dataclasses import dataclass, field



@dataclass
class Client:
    name: str
    key: bytes = field(repr=False)  # (never in a log line or an error message)
    max_streams: int = 0  # at once (0: no limit)
    rpm: float = 0.0  # a minute (0: no limit)
    metrics: bool = False  # may read /metrics


@dataclass
class _Use:
    """A client's use, kept across reloads of the keys file (by name)."""

    active: int = 0
    tokens: float = -1.0  # (-1: a full bucket, set at the first request)
    stamp: float = 0.0


def parse(text: str) -> list[Client]:
    """The clients of a keys file (JSON, or lines: name key [max_streams=N] [rpm=N] [metrics=true])."""
    text = text.strip()
    if text.startswith("["):
        rows = json.loads(text)
    else:
        rows = []
        for n, line in enumerate(text.splitlines(), 1):
            words = line.split("#", 1)[0].split()
            if not words:
                continue
            if len(words) < 2:
                raise ValueError(f"line {n}: name and key")
            row: dict = {"name": words[0], "key": words[1]}
            for w in words[2:]:
                k, eq, v = w.partition("=")
                if not eq:
                    raise ValueError(f"line {n}: {k!r}: max_streams=N, rpm=N or metrics=true")
                row[k] = v
            rows.append(row)
    clients, names, keys = [], set(), set()
    for row in rows:
        unknown = set(row) - {"name", "key", "max_streams", "rpm", "metrics"}
        if unknown or not row.get("name") or not row.get("key"):
            raise ValueError(f"client {row.get('name')!r}: name, key, max_streams, rpm, metrics "
                             f"(not {sorted(unknown)})")
        c = Client(str(row["name"]), str(row["key"]).encode(), int(row.get("max_streams", 0)),
                   float(row.get("rpm", 0)), str(row.get("metrics", False)).lower() in ("true", "1", "yes"))
        if c.name in names or c.key in keys or c.max_streams < 0 or c.rpm < 0 or not c.name.isprintable():
            raise ValueError(f"client {c.name!r}: a name or key given twice, or a negative limit")
        names.add(c.name)
        keys.add(c.key)
        clients.append(c)
    return clients


class Clients:
    """The clients a server knows (``api_key``: one named "default"; ``keys_file``: one per key, read again when it
    changes). No key at all: anyone may call (``enabled`` is False; identify() finds no client, acquire(None) lets
    every request through). ``log``: the server's logger (a reload, a file that does not parse)."""

    RECHECK_S = 1.0  # the keys file looked at no more often

    def __init__(self, api_key: str | None = None, keys_file: str | None = None, clock=time.monotonic,
                 log: logging.Logger | None = None):
        self.api_key, self.keys_file, self.clock = api_key, keys_file, clock
        self.log = log or logging.getLogger(__name__)
        self.clients: list[Client] = []
        self.use: dict[str, _Use] = {}
        self._seen: tuple | None = None  # the keys file's (mtime, size) when read
        self._checked = -1e9
        self._load(force=True)

    @property
    def enabled(self) -> bool:
        return bool(self.api_key or self.keys_file)

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.clients]

    def _load(self, force: bool = False) -> None:
        if not self.keys_file:
            self.clients = [Client("default", self.api_key.encode(), metrics=True)] if self.api_key else []
            return
        now = self.clock()
        if not force and now - self._checked < self.RECHECK_S:
            return
        self._checked = now
        try:
            st = os.stat(self.keys_file)
            seen = (st.st_mtime_ns, st.st_size)
            if seen == self._seen:
                return
            with open(self.keys_file, encoding="utf-8") as f:
                clients = parse(f.read())
        except (OSError, ValueError) as e:  # (json's errors are ValueErrors)
            if force and self._seen is None:
                raise ValueError(f"--api-keys-file {self.keys_file}: {e}") from e
            self.log.error("--api-keys-file %s: %s: the clients read before stay", self.keys_file, e)
            return
        if self.api_key:  # (--api-key too: one more client)
            clients.append(Client("default", self.api_key.encode(), metrics=True))
        if self._seen is not None:
            self.log.info("--api-keys-file read again: %d clients (%s)", len(clients),
                          ", ".join(c.name for c in clients))
        self.clients, self._seen = clients, seen

    def identify(self, presented: bytes) -> Client | None:
        """The client whose key was presented, else None. Every key is compared, in constant time: how long it
        takes tells nothing about which one matched or how much of it."""
        self._load()
        found = None
        for c in self.clients:
            if hmac.compare_digest(presented, c.key) and found is None:
                found = c
        return found

    def acquire(self, client: Client | None) -> tuple[str | None, float]:
        """A request of ``client`` (None: no keys, no limits) starts: (None, 0) when its limits let it, counted
        (release() when it ends); else (why, seconds until it may try again)."""
        if client is None:
            return None, 0.0
        use = self.use.setdefault(client.name, _Use())
        if client.max_streams and use.active >= client.max_streams:
            return f"{use.active} requests in progress, at most {client.max_streams} (max_streams)", 1.0
        if client.rpm:
            now = self.clock()
            rate = client.rpm / 60.0
            use.tokens = client.rpm if use.tokens < 0 else min(client.rpm, use.tokens + (now - use.stamp) * rate)
            use.stamp = now
            if use.tokens < 1.0:
                return f"at most {client.rpm:g} requests a minute (rpm)", (1.0 - use.tokens) / rate
            use.tokens -= 1.0
        use.active += 1
        return None, 0.0

    def release(self, client: Client | None) -> None:
        if client is not None:
            self.use[client.name].active -= 1
