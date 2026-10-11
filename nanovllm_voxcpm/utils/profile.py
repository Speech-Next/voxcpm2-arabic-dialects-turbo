"""Opt-in per-step timing.

Set ``NANOVLLM_PROFILE=/path/to/steps.jsonl`` to append one JSON record per
engine step with CUDA-synchronized stage durations (ms). Unset, every call is a
no-op and nothing is synchronized. ``NANOVLLM_PROFILE_SYNC=0`` keeps the records but skips the
synchronizations (the engine then runs exactly as unprofiled; stage times become CPU launch times, while
step boundaries stay accurate because every step ends with a device-to-host copy).
"""

import json
import os
import time

import torch

_PATH = os.environ.get("NANOVLLM_PROFILE")
_SYNC = os.environ.get("NANOVLLM_PROFILE_SYNC", "1") != "0"
_current: "StepProfile | None" = None


class StepProfile:
    enabled = bool(_PATH)

    def __init__(self, **meta):
        self.rec = dict(meta) if self.enabled else None
        self._t = None

    def start(self) -> "StepProfile":
        global _current
        if self.enabled:
            if _SYNC:
                torch.cuda.synchronize()
            self._t = time.perf_counter()
            self.rec["t"] = self._t  # perf_counter at the start, comparable with the server's step records
            _current = self
        return self

    def mark(self, name: str) -> None:
        """Record the time since the previous mark under ``name`` (accumulates on repeats)."""
        if not self.enabled or self._t is None:
            return
        if _SYNC:
            torch.cuda.synchronize()
        now = time.perf_counter()
        self.rec[name] = self.rec.get(name, 0.0) + (now - self._t) * 1000
        self._t = now

    def write(self) -> None:
        global _current
        if not self.enabled:
            return
        _current = None
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
