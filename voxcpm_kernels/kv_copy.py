"""Copy the first n token slots of KV-cache blocks, for every causal layer, in one launch.

The engine keeps all layers' K/V in one tensor ``[layers, 2, blocks, block_size, kv_heads, head_dim]``, so a
block's first n slots are one contiguous run of ``n * kv_heads * head_dim`` elements in each (layer, k/v) plane.
One program copies one chunk of that run for one (src, dst, n) entry and one plane: read once, written once,
instead of a gather into a temporary and a scatter out of it per layer and per K/V (2 x 72 launches), and each
entry copies only its own n slots.
"""

import torch
import triton
import triton.language as tl

_CHUNK = 4096  # elements per program (8 KB of bf16)


@triton.jit
def _copy_block_prefix(kv_ptr, pairs_ptr, per_token, plane_stride, block_stride, CHUNK: tl.constexpr):
    pair, plane, chunk = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    src = tl.load(pairs_ptr + 3 * pair).to(tl.int64)
    dst = tl.load(pairs_ptr + 3 * pair + 1).to(tl.int64)
    n_elems = tl.load(pairs_ptr + 3 * pair + 2) * per_token
    if chunk * CHUNK < n_elems:
        offs = chunk * CHUNK + tl.arange(0, CHUNK)
        mask = offs < n_elems
        base = kv_ptr + plane.to(tl.int64) * plane_stride
        tl.store(base + dst * block_stride + offs, tl.load(base + src * block_stride + offs, mask=mask), mask=mask)


def copy_kv_block_prefix(kv: torch.Tensor, entries: list[tuple[int, int, int]]) -> None:
    """kv: contiguous [layers, 2, blocks, block_size, heads, dim]; entries: (src block, dst block, n tokens)."""
    assert kv.is_contiguous()
    layers, kvs, blocks, block_size = kv.shape[:4]
    entries = [(s, d, min(n, block_size)) for s, d, n in entries if n > 0]
    if not entries:
        return
    per_token = kv.shape[4] * kv.shape[5]
    pairs = torch.tensor(entries, dtype=torch.int32, pin_memory=True).to(kv.device, non_blocking=True)
    grid = (len(entries), layers * kvs, triton.cdiv(max(n for _, _, n in entries) * per_token, _CHUNK))
    _copy_block_prefix[grid](kv, pairs, per_token, blocks * block_size * per_token, block_size * per_token,
                             CHUNK=_CHUNK, num_warps=4)
