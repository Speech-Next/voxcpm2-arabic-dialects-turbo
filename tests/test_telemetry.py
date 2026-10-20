"""The servers' telemetry (entrypoints/telemetry.py): log lines written by a thread of their own, never by the event
loop that logs them; counters, gauges and histograms in Prometheus' text format."""

import logging
import logging.handlers
import threading
import time

from nanovllm_voxcpm.entrypoints.telemetry import (OUTCOMES, Counter, Gauge, Histogram, Metrics, ServerMetrics,
                                                   setup_logging)


class SlowStream:
    """stdout behind a slow log collector: every write takes 20 ms."""

    def __init__(self):
        self.lines, self.threads = [], set()

    def write(self, s):
        time.sleep(0.02)
        self.lines.append(s)
        self.threads.add(threading.current_thread().name)

    def flush(self):
        pass


def test_log_lines_are_written_by_a_thread_of_their_own():
    stream = SlowStream()
    listener = setup_logging("telemetry_test", "info", stream)
    logger = logging.getLogger("telemetry_test")
    try:
        t = time.perf_counter()
        for i in range(20):
            logger.info("line %d", i)
        logger.debug("not written")
        took = time.perf_counter() - t
    finally:
        listener.stop()  # (writes what is queued)
    assert took < 0.1  # 20 writes of 20 ms each: the logging calls did not wait for them
    text = "".join(stream.lines)
    assert all(f"line {i}\n" in text for i in range(20)) and "not written" not in text
    assert "INFO [telemetry_test] line 0" in text
    assert threading.current_thread().name not in stream.threads
    assert logger.handlers and all(isinstance(h, logging.handlers.QueueHandler) for h in logger.handlers)


def test_prometheus_text():
    m = Metrics()
    c = m.add(Counter("x_total", "things", ("outcome",), ("ok", "busy")))
    c.inc("ok")
    c.inc("ok")
    c.inc("busy", by=0.5)
    m.add(Gauge("y", "a gauge", lambda: 3))
    h = m.add(Histogram("z_seconds", "times", (0.125, 1.0)))
    for v in (0.0625, 0.125, 0.5, 2.0):  # (le is inclusive: 0.125 counts in the 0.125 bucket)
        h.observe(v)
    assert m.render().splitlines() == [
        "# HELP x_total things", "# TYPE x_total counter", 'x_total{outcome="busy"} 0.5', 'x_total{outcome="ok"} 2',
        "# HELP y a gauge", "# TYPE y gauge", "y 3",
        "# HELP z_seconds times", "# TYPE z_seconds histogram", 'z_seconds_bucket{le="0.125"} 2',
        'z_seconds_bucket{le="1"} 3', 'z_seconds_bucket{le="+Inf"} 4', "z_seconds_sum 2.6875", "z_seconds_count 4"]


def test_server_metrics_start_at_zero():
    text = ServerMetrics("utterance", lambda: 0, lambda: True).render()
    assert all(f'voxcpm_requests_total{{outcome="{o}"}} 0' in text for o in OUTCOMES)
    assert "voxcpm_active_streams 0" in text and "voxcpm_audio_seconds_total 0" in text
    assert "voxcpm_engine_alive 1" in text and 'voxcpm_first_audio_seconds_bucket{le="+Inf"} 0' in text
