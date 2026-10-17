"""The segmented MLP GEMMs of an adapter with MLP LoRA (a tagged model served as an adapter), on a GPU: each row
segment's gate_up (SwiGLU epilogue) and down weights picked through the remap table equal the plain int8 kernels on
that segment's weights, and remapped-to-base rows equal the base path bit for bit. Skipped without CUDA."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("M", [3, 64, 200, 300])
def test_seg_swiglu_and_down_follow_the_remap(M):
    from voxcpm_kernels.fused_layer import _gemm, swiglu_gemm_quant, swiglu_weights
    from voxcpm_kernels.int8_linear import W8A8Linear, quantize_rows
    from voxcpm_kernels.seg_gemm import build_tiles, seg_gemm, seg_swiglu_gemm_quant

    torch.manual_seed(M)
    H, I, slots = 256, 384, 3
    base_gu, base_dn = torch.randn(2 * I, H, device="cuda") / H**0.5, torch.randn(H, I, device="cuda") / I**0.5
    other_gu, other_dn = base_gu + 0.1 * torch.randn_like(base_gu), base_dn + 0.1 * torch.randn_like(base_dn)
    lins = [(swiglu_weights(W8A8Linear(gu), W8A8Linear(dn)), W8A8Linear(dn))
            for gu, dn in ((base_gu, base_dn), (other_gu, other_dn))]
    w_gu = torch.stack([g.w_q for g, _ in lins]); s_gu = torch.stack([g.w_scale for g, _ in lins])
    w_dn = torch.stack([d.w_q for _, d in lins]); s_dn = torch.stack([d.w_scale for _, d in lins])
    remap = torch.zeros(slots + 1, device="cuda", dtype=torch.int32)
    remap[2 + 1] = 1  # slot 2 has the MLP LoRA (entry 1); slots 0, 1 and the base (-1) use entry 0
    row_slots = sorted(torch.randint(-1, slots, (M,)).tolist(), key=lambda s: (s < 0, s))  # grouped by adapter
    x = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)
    xq = quantize_rows(x)
    tiles = build_tiles(row_slots, 128 if M > 256 else 64)
    seg = (torch.from_numpy(tiles).cuda(), tiles.shape[0], 128 if M > 256 else 64)

    hq, hs = seg_swiglu_gemm_quant(*xq, w_gu, s_gu, torch.bfloat16, *seg, remap=remap)
    out = seg_gemm(hq, hs, w_dn, s_dn, None, torch.bfloat16, *seg, residual=x, remap=remap)
    for entry in (0, 1):
        rows = torch.tensor([i for i, s in enumerate(row_slots) if (entry == 1) == (s == 2)], device="cuda")
        if rows.numel() == 0:
            continue
        g, d = lins[entry]
        ref_hq, ref_hs = swiglu_gemm_quant(*quantize_rows(x[rows]), g, torch.bfloat16)
        assert torch.equal(hq[rows], ref_hq) and torch.equal(hs[rows], ref_hs), entry
        ref = _gemm(ref_hq, ref_hs, d, torch.bfloat16, residual=x[rows])
        assert torch.equal(out[rows], ref), entry
