"""Opt-in per-step timing.

Set ``NANOVLLM_PROFILE=/path/to/steps.jsonl`` to append one JSON record per
engine step with CUDA-synchronized stage durations (ms). Unset, every call is a
no-op and nothing is synchronized. ``NANOVLLM_PROFILE_SYNC=0`` keeps the records but skips the
synchronizations (the engine then runs exactly as unprofiled; stage times become CPU launch times, while
step boundaries stay accurate because every step ends with a device-to-host copy).
``NANOVLLM_PROFILE_GPU=1`` (with ``SYNC=0``) also records a CUDA event at the engine step's entry (begin_step) and at
every mark: ``gpu_<mark>`` is the GPU's time from the step's entry to that point (when the GPU got there: the
first model kernel starts right after ``gpu_prep``), read once the step has synchronized at its end.
"""

import json
import os
import time

import torch

_PATH = os.environ.get("NANOVLLM_PROFILE")
_SYNC = os.environ.get("NANOVLLM_PROFILE_SYNC", "1") != "0"
_GPU = bool(_PATH) and os.environ.get("NANOVLLM_PROFILE_GPU", "0") != "0"
_current: "StepProfile | None" = None
_begin: tuple | None = None  # (host perf_counter, CUDA event) at the engine step's entry


def begin_step() -> None:
    """The engine step's entry (before scheduling): the reference of the step record's t_begin and gpu_* times."""
    global _begin
    if not StepProfile.enabled:
        return
    ev = None
    if _GPU and torch.cuda.is_available():
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
    _begin = (time.perf_counter(), ev)


class StepProfile:
    enabled = bool(_PATH)

    def __init__(self, **meta):
        self.rec = dict(meta) if self.enabled else None
        self._t = None
        self._events: list = []  # (mark, CUDA event)

    def start(self) -> "StepProfile":
        global _current
        if self.enabled:
            if _SYNC:
                torch.cuda.synchronize()
            self._t = time.perf_counter()
            self.rec["t"] = self._t  # perf_counter at the start, comparable with the server's step records
            if _begin is not None:
                self.rec["t_begin"] = _begin[0]
            _current = self
        return self

    def mark(self, name: str) -> None:
        """Record the time since the previous mark under ``name`` (accumulates on repeats)."""
        if not self.enabled or self._t is None:
            return
        if _SYNC:
            torch.cuda.synchronize()
        elif _GPU and _begin is not None and _begin[1] is not None:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._events.append((name, ev))
        now = time.perf_counter()
        self.rec[name] = self.rec.get(name, 0.0) + (now - self._t) * 1000
        self._t = now

    def write(self) -> None:
        global _current, _begin
        if not self.enabled:
            return
        _current = None
        if self._events and _begin is not None:
            self._events[-1][1].synchronize()
            for name, ev in self._events:  # the last time the GPU reached a repeated mark
                self.rec["gpu_" + name] = _begin[1].elapsed_time(ev)
        _begin = None
        with open(_PATH, "a") as f:
            f.write(json.dumps(self.rec) + "\n")


def mark(name: str) -> None:
    """Mark on the step currently being profiled, if any."""
    if _current is not None:
        _current.mark(name)


def log(**rec) -> None:
    if StepProfile.enabled:
        with open(_PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")
