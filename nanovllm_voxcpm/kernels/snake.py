"""Snake activation, x + sin(alpha * x)^2 / alpha, as one Triton kernel.

An optional per-channel affine (x * scale + bias) is applied first, so the
AudioVAE's sample-rate conditioning layer folds into the snake that follows it.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _snake_kernel(
    x_ptr,
    out_ptr,
    alpha_ptr,
    inv_alpha_ptr,
    scale_ptr,
    bias_ptr,
    n_elements,
    C,
    T,
    HAS_AFFINE: tl.constexpr,
    FAST_SIN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    c = (offs // T) % C
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    if HAS_AFFINE:
        x = x * tl.load(scale_ptr + c, mask=mask, other=1.0) + tl.load(bias_ptr + c, mask=mask, other=0.0)
    ax = tl.load(alpha_ptr + c, mask=mask, other=1.0) * x
    if FAST_SIN:
        s = libdevice.fast_sinf(ax)
    else:
        s = tl.sin(ax)
    y = x + s * s * tl.load(inv_alpha_ptr + c, mask=mask, other=1.0)
    tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


class Snake:
    """Precomputed per-channel parameters (fp32) for the Triton snake."""

    def __init__(
        self,
        alpha: torch.Tensor,
        scale: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        fast_sin: bool = False,
    ):
        self.fast_sin = fast_sin
        alpha = alpha.detach().reshape(-1).float()
        self.alpha = alpha.contiguous()
        self.inv_alpha = (alpha + 1e-9).reciprocal().contiguous()
        self.scale = None if scale is None else scale.detach().reshape(-1).float().contiguous()
        self.bias = None if bias is None else bias.detach().reshape(-1).float().contiguous()
        self.channels = alpha.numel()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()
        out = torch.empty_like(x)
        n = x.numel()
        affine = self.scale is not None
        block = 1024
        _snake_kernel[(triton.cdiv(n, block),)](
            x,
            out,
            self.alpha,
            self.inv_alpha,
            self.scale if affine else self.alpha,
            self.bias if affine else self.alpha,
            n,
            self.channels,
            x.size(-1),
            HAS_AFFINE=affine,
            FAST_SIN=self.fast_sin,
            BLOCK=block,
        )
        return out


def snake_reference(x, alpha, scale=None, bias=None):
    """Eager reference in fp32 (alpha/scale/bias shaped [C])."""
    x = x.float()
    if scale is not None:
        x = x * scale.float()[None, :, None] + bias.float()[None, :, None]
    a = alpha.float()[None, :, None]
    return x + (a + 1e-9).reciprocal() * torch.sin(a * x).pow(2)
