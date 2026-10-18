"""Predicted engine step times, fitted online to the measured steps (the scheduler's time model).

A decode step takes ``a + b * rows`` seconds, a prefill step (or a mixed one: prefill rows and decode rows)
``a + b * rows + c * tokens``. Each is a least-squares fit with exponential forgetting over the steps seen so far,
pulled toward a prior (one A10, the routed default: 15.6 ms + 0.365 ms per row; 19.6 ms + 0.23 ms per row + 0.036 ms
per token) so the first predictions are sane and a run of one-sized steps cannot tilt the slopes into nonsense."""

import numpy as np

_DECODE_PRIOR = (0.0156, 0.000365)
_PREFILL_PRIOR = (0.0196, 0.00023, 0.0000363)
# where the prior speaks: (rows,) / (rows, tokens) pseudo-observations, each worth one measured step
_DECODE_AT = [(1,), (64,), (256,)]
_PREFILL_AT = [(1, 64), (32, 1500), (64, 4096), (256, 2048)]


class _Fit:
    def __init__(self, prior, at, decay: float):
        self.decay = decay
        x = np.array([(1.0, *p) for p in at])
        y = x @ np.array(prior)
        self.prior_xx, self.prior_xy = x.T @ x, x.T @ y
        self.xx = np.zeros_like(self.prior_xx)
        self.xy = np.zeros_like(self.prior_xy)
        self.coef = tuple(float(c) for c in prior)

    def observe(self, x: tuple, seconds: float) -> None:
        v = np.array((1.0, *x))
        self.xx = self.decay * self.xx + np.outer(v, v)
        self.xy = self.decay * self.xy + v * seconds
        coef = np.linalg.solve(self.prior_xx + self.xx, self.prior_xy + self.xy)
        self.coef = tuple(float(c) for c in coef)


class StepCost:
    def __init__(self, decay: float = 0.995):
        self.decode = _Fit(_DECODE_PRIOR, _DECODE_AT, decay)
        self.prefill = _Fit(_PREFILL_PRIOR, _PREFILL_AT, decay)

    def seconds(self, rows: int, tokens: int, prefill: bool) -> float:
        if prefill:
            a, b, c = self.prefill.coef
            return a + b * rows + c * tokens
        a, b = self.decode.coef
        return a + b * rows

    def observe(self, rows: int, tokens: int, prefill: bool, seconds: float) -> None:
        if rows <= 0 or not 0.0 < seconds < 10.0:  # (an empty step; a stall of the host, not of the step)
            return
        if prefill:
            self.prefill.observe((rows, tokens), seconds)
        else:
            self.decode.observe((rows,), seconds)
