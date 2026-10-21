"""The servers' logs and metrics (ws_server, openai_server), with no dependency.

- Logs: a server's log lines go through a queue to a thread of their own, which formats and writes them; the event
  loop that paces 150 streams' chunks never waits on stdout (a full pipe to a slow log collector held a print).
- Metrics: counters, gauges and histograms in Prometheus' text format (``GET /metrics``), updated from the event
  loop's thread only (no lock: an increment is a dict update) and read there when scraped.
"""

import bisect
import logging
import logging.handlers
import queue
import sys
from collections.abc import Callable, Iterator

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
OUTCOMES = ("ok", "client_error", "server_error", "busy", "limited", "cancelled")
# time to first audio, seconds: the engine's is 25-150 ms up to 150 streams at once (README), a paced first chunk
# --pace-start-ms later
FIRST_AUDIO_BUCKETS = (0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0, 2.0, 5.0)


class _Handoff(logging.handlers.QueueHandler):
    """The record as it is: formatted by the writer thread, not by the event loop that logs it."""

    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        return record


def setup_logging(name: str, level: str = "info", stream=None) -> logging.handlers.QueueListener:
    """``name``'s logger (and its children's) written by a thread of its own, to ``stream`` (stdout). Returns the
    started listener: its stop() writes what is still queued (call it before the process exits)."""
    q: queue.SimpleQueue = queue.SimpleQueue()
    handler = logging.StreamHandler(sys.stdout if stream is None else stream)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    logger = logging.getLogger(name)
    logger.handlers = [_Handoff(q)]
    logger.setLevel(level.upper())
    logger.propagate = False
    listener = logging.handlers.QueueListener(q, handler)
    listener.start()
    return listener


def _num(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else repr(float(v))


def _labels(names: tuple[str, ...], values: tuple) -> str:
    if not names:
        return ""
    return "{" + ",".join(f'{k}="{v}"' for k, v in zip(names, values)) + "}"


class Counter:
    """A counter per tuple of label values (in ``labels``' order)."""

    def __init__(self, name: str, help: str, labels: tuple[str, ...] = (), known: tuple = ()):
        self.name, self.help, self.labels = name, help, labels
        # exported from the start, at 0: the known label values (e.g. every outcome), or the one unlabeled value
        self.values: dict[tuple, float] = {(v,): 0.0 for v in known} if labels else {(): 0.0}

    def inc(self, *values: str, by: float = 1.0) -> None:
        self.values[values] = self.values.get(values, 0.0) + by

    def lines(self) -> Iterator[str]:
        yield f"# HELP {self.name} {self.help}"
        yield f"# TYPE {self.name} counter"
        for values, v in sorted(self.values.items()):
            yield f"{self.name}{_labels(self.labels, values)} {_num(v)}"


class Gauge:
    """A value read when scraped."""

    def __init__(self, name: str, help: str, read: Callable[[], float]):
        self.name, self.help, self.read = name, help, read

    def lines(self) -> Iterator[str]:
        yield f"# HELP {self.name} {self.help}"
        yield f"# TYPE {self.name} gauge"
        yield f"{self.name} {_num(self.read())}"


class ReadCounter(Gauge):
    """A counter kept elsewhere (e.g. in the engine's process), read when scraped."""

    def lines(self) -> Iterator[str]:
        yield f"# HELP {self.name} {self.help}"
        yield f"# TYPE {self.name} counter"
        yield f"{self.name} {_num(self.read())}"


class Histogram:
    def __init__(self, name: str, help: str, buckets: tuple[float, ...]):
        self.name, self.help, self.buckets = name, help, tuple(sorted(buckets))
        self.counts = [0] * (len(self.buckets) + 1)  # per bucket, the last one above them all
        self.sum = 0.0

    def observe(self, v: float) -> None:
        self.counts[bisect.bisect_left(self.buckets, v)] += 1  # (the first bucket with v <= le)
        self.sum += v

    def lines(self) -> Iterator[str]:
        yield f"# HELP {self.name} {self.help}"
        yield f"# TYPE {self.name} histogram"
        n = 0
        for le, c in zip(self.buckets, self.counts):
            n += c
            yield f'{self.name}_bucket{{le="{le:g}"}} {n}'
        n += self.counts[-1]
        yield f'{self.name}_bucket{{le="+Inf"}} {n}'
        yield f"{self.name}_sum {_num(self.sum)}"
        yield f"{self.name}_count {n}"


class Metrics:
    """A server's series, rendered for ``GET /metrics``."""

    def __init__(self):
        self.series: list[Counter | Gauge | Histogram] = []

    def add(self, series):
        self.series.append(series)
        return series

    def render(self) -> str:
        return "\n".join(line for s in self.series for line in s.lines()) + "\n"


def add_audio_cache(metrics: Metrics, stats: Callable[[], dict | None]) -> None:
    """The audio cache's series (--audio-cache-mb): ``stats()`` its AudioCache.stats() (None: none yet, 0)."""

    def read(key: str) -> Callable[[], float]:
        return lambda: float((stats() or {}).get(key, 0))

    metrics.add(ReadCounter("voxcpm_audio_cache_hits_total", "sentences played from the audio cache", read("hits")))
    metrics.add(ReadCounter("voxcpm_audio_cache_misses_total", "sentences generated (not in the audio cache)",
                            read("misses")))
    metrics.add(ReadCounter("voxcpm_audio_cache_disk_hits_total", "sentences of the hits read from --audio-cache-dir",
                            read("disk_hits")))
    metrics.add(ReadCounter("voxcpm_audio_cache_joins_total", "requests that joined a generation of their sentence "
                            "in progress (single flight)", read("joins")))
    metrics.add(Gauge("voxcpm_audio_cache_bytes", "audio kept in the audio cache, bytes", read("bytes")))
    metrics.add(Gauge("voxcpm_audio_cache_entries", "sentences kept in the audio cache", read("entries")))


class ServerMetrics(Metrics):
    """The series both servers export. ``what``: what a request is there (speech request, utterance); ``active``
    and ``alive``: read when scraped (requests in progress, whether the engine process runs)."""

    def __init__(self, what: str, active: Callable[[], float], alive: Callable[[], bool]):
        super().__init__()
        self.requests = self.add(Counter(
            "voxcpm_requests_total", f"{what}s by outcome: ok, client_error (the request's fault), server_error "
            "(the engine failed), busy (--max-streams, shutting down), limited (over its client's max_streams or rpm, "
            "--api-keys-file), cancelled (the client left)", ("outcome",),
            OUTCOMES))
        self.add(Gauge("voxcpm_active_streams", f"{what}s in progress", active))
        self.first_audio = self.add(Histogram(
            "voxcpm_first_audio_seconds", f"time to first audio: from a {what}'s turn to the engine's first audio "
            "at the server", FIRST_AUDIO_BUCKETS))
        self.audio = self.add(Counter("voxcpm_audio_seconds_total", "seconds of audio returned to the clients"))
        self.add(Gauge("voxcpm_engine_alive", "1 while the engine process runs, else 0", lambda: float(alive())))
