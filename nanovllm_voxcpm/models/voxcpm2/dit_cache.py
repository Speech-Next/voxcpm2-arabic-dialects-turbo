"""Block caching across the LocDiT solver's steps: cache-dit's DBCache with a static step mask, plus TaylorSeer.

Every evaluated step runs the decoder's first ``fn`` and last ``bn`` layers. On a step the mask marks as cached, the
middle layers are skipped: their contribution to the residual stream (output minus input of the middle layers) is
taken from the last computed step and, with ``taylor``, extrapolated linearly over the step index from the last two
computed steps (TaylorSeer, order 1). The mask is static, the same for every request and every patch: a batch of
requests then keeps one shape and the CUDA graphs that replay the solver stay valid. (cache-dit's dynamic mode decides
per step from the residual change, which here would differ per request in the batch and need a host round trip.)

TurboConfig.dit_cache = {"fn": 2, "bn": 2, "mask": [1, 1, 0, 1, 0, 1, 0, 1, 1], "taylor": true}: mask[i] is the i-th
DiT evaluation of a patch (the original sampler evaluates 9 of its 10 steps, the distilled one 2); 1 = compute every
layer, 0 = reuse the middle; steps past the mask and the first step are always computed.
"""

from __future__ import annotations

from typing import Any

import torch

# Plans for the original LocDiT (9 evaluations per patch), measured on 200 voice-cloned Arabic requests on an A10:
# "light" reuses the middle layers on 2 of the 9 evaluations, "medium" on 4. The distilled 2-step LocDiT has no step
# to spare: caching its second step made the speech unintelligible (WER 54 %).
PRESETS: dict[str, dict[str, Any]] = {
    "light": {"fn": 1, "bn": 1, "mask": [1, 1, 1, 1, 1, 0, 1, 0, 1]},
    "medium": {"fn": 1, "bn": 1, "mask": [1, 1, 0, 1, 0, 1, 0, 1, 0]},
}


def plan(spec: str | dict[str, Any]) -> dict[str, Any]:
    """A preset's name or a plan dict -> the plan dict (TurboConfig.dit_cache)."""
    if isinstance(spec, dict):
        return spec
    if spec not in PRESETS:
        raise ValueError(f"unknown dit_cache preset {spec!r}; presets: {', '.join(PRESETS)}")
    return dict(PRESETS[spec])


class DiTCache:
    """The cache of one solver call (one patch of a batch)."""

    def __init__(self, fn: int = 2, bn: int = 2, mask: list[int] | None = None, taylor: bool = True) -> None:
        if fn < 1 or bn < 0:
            raise ValueError("dit_cache: fn >= 1 and bn >= 0")
        self.fn, self.bn, self.taylor = fn, bn, taylor
        self.mask = [bool(m) for m in (mask or [])]
        self.step = 0
        self.last: torch.Tensor | None = None  # middle layers' contribution at the last computed step
        self.prev: torch.Tensor | None = None  # ... and at the computed step before it
        self.last_step = self.prev_step = -1

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "DiTCache":
        return cls(fn=int(cfg.get("fn", 2)), bn=int(cfg.get("bn", 2)), mask=cfg.get("mask"),
                   taylor=bool(cfg.get("taylor", True)))

    def _compute(self) -> bool:
        return self.last is None or self.step >= len(self.mask) or self.mask[self.step]

    def _estimate(self, rows: int) -> torch.Tensor:
        """The middle layers' contribution for the current step (the first ``rows`` rows: a CFG step may be
        followed by conditional-only ones, whose rows are the first half)."""
        last = self.last[:rows]
        if not self.taylor or self.prev is None:
            return last
        slope = (self.step - self.last_step) / (self.last_step - self.prev_step)
        return last + (last - self.prev[:rows]) * slope

    def run(self, layers, positions: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """The decoder's layer stack on ``hidden`` for this step: the residual stream after the last layer."""
        n = len(layers)
        fn, bn = min(self.fn, n), min(self.bn, n - min(self.fn, n))
        residual = None
        for layer in layers[:fn]:
            hidden, residual = layer(positions, hidden, residual)
        if self._compute():
            start = hidden
            for layer in layers[fn:n - bn]:
                hidden, residual = layer(positions, hidden, residual)
            if self.taylor:
                self.prev, self.prev_step = self.last, self.last_step
            self.last, self.last_step = hidden - start, self.step
        else:
            hidden = hidden + self._estimate(hidden.shape[0])
            residual = None
        for layer in layers[n - bn:]:
            hidden, residual = layer(positions, hidden, residual)
        self.step += 1
        return hidden
