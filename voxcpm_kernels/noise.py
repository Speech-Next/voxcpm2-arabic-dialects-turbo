"""Per-row seeded Gaussian noise in one launch (Philox via tl.randn).

Each row r gets randn(seed=seeds[r], offsets 0..n-1), so a request's noise depends
only on its own (seed, step) — not on its batch neighbours — like a per-row
torch.Generator, but without B generator constructions and launches per step.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _seeded_randn(seed_ptr, out_ptr, n_per_row, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    seed = tl.load(seed_ptr + row)
    offs = tl.arange(0, BLOCK)
    z = tl.randn(seed, offs)
    tl.store(out_ptr + row.to(tl.int64) * n_per_row + offs, z.to(out_ptr.dtype.element_ty), mask=offs < n_per_row)


def seeded_randn(seeds, row_shape, dtype=torch.float32, device="cuda") -> torch.Tensor:
    """seeds: sequence of non-negative ints (< 2**63), one per row -> tensor [len(seeds), *row_shape]."""
    n = 1
    for d in row_shape:
        n *= d
    seed_t = torch.tensor([int(s) & 0x7FFFFFFFFFFFFFFF for s in seeds], dtype=torch.int64, device="cpu", pin_memory=True)
    seed_t = seed_t.to(device, non_blocking=True)
    out = torch.empty(len(seeds), *row_shape, device=device, dtype=dtype)
    _seeded_randn[(len(seeds),)](seed_t, out, n, BLOCK=triton.next_power_of_2(n))
    return out
