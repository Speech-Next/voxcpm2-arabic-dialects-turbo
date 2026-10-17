"""The 2:4 sparse FFN GEMMs (voxcpm_kernels/sparse24.py) against the dense W8A8 kernels on the same pruned weights:
bit-identical outputs at base_lm's FFN shapes (gate_up 2048 -> 2 x 6144 with the SwiGLU epilogue, down 6144 -> 2048
with the residual add), for every tile configuration, and under CUDA-graph replay. Skipped without CUDA.

    CUDA_VISIBLE_DEVICES=0 python -m pytest -q tests/gpu/test_sparse24_gemm.py
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

HIDDEN, INTER = 2048, 6144
ROWS = [1, 2, 7, 8, 9, 16, 17, 31, 32, 33, 64, 65, 128, 200, 256, 257, 512, 1300, 2030]


@pytest.fixture(scope="module")
def ffn():
    from voxcpm_kernels import fused_layer as F
    from voxcpm_kernels import sparse24 as S
    from voxcpm_kernels.int8_linear import W8A8Linear

    if not S.available():
        pytest.skip("sparse24 extension unavailable")
    g = torch.Generator(device="cuda").manual_seed(0)
    w_gu = S.prune_24(torch.randn(2 * INTER, HIDDEN, device="cuda", generator=g) * 0.02)
    w_dn = S.prune_24(torch.randn(HIDDEN, INTER, device="cuda", generator=g) * 0.02)
    # SmoothQuant factors as in the engine: down's 1/s folded into the up rows' scales, s into down's columns
    smooth_dn = torch.rand(INTER, device="cuda", generator=g) + 0.5
    down = W8A8Linear(w_dn, None, smooth_dn)
    gate_up = F.swiglu_weights(W8A8Linear(w_gu), down)
    assert S.is_24(gate_up.w_q) and S.is_24(down.w_q)
    sp_gu = S.Sparse24Linear(gate_up, S.swiglu_rows(gate_up.out_features, "cuda"))
    sp_dn = S.Sparse24Linear(down)
    return F, S, gate_up, down, sp_gu, sp_dn


def _inputs(m, k, seed):
    from voxcpm_kernels.int8_linear import quantize_rows

    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, device="cuda", generator=g, dtype=torch.float32).to(torch.bfloat16)
    return quantize_rows(x)


def _dense_h(F, q, s, lin):
    """h and the per-row max |h| of the dense SwiGLU GEMM (fused_layer._w8a8_gemm_swiglu), before quantization."""
    import triton

    from voxcpm_kernels.int8_linear import _tiles

    m, n = q.shape[0], lin.out_features
    bm, bn, bk, warps, stages = _tiles(m, n, "swiglu")
    nt = triton.cdiv(n, bn)
    h = torch.empty(m, n // 2, device="cuda", dtype=torch.bfloat16)
    amax = torch.empty(m, nt, device="cuda", dtype=torch.float32)
    F._w8a8_gemm_swiglu[(triton.cdiv(m, bm) * nt,)](q, lin.w_q, h, amax, s, lin.w_scale, m, n, q.shape[1], nt,
                                                     BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8,
                                                     num_warps=warps, num_stages=stages)
    return h, amax.amax(1)


def test_packed_weight_is_the_dense_one(ffn):
    F, S, gate_up, down, sp_gu, sp_dn = ffn
    assert torch.equal(S.decompress_24(sp_gu.wf, sp_gu.mf), gate_up.w_q[S.swiglu_rows(gate_up.out_features, "cuda")])
    assert torch.equal(S.decompress_24(sp_dn.wf, sp_dn.mf), down.w_q)


@pytest.mark.parametrize("m", ROWS)
def test_swiglu_bit_identical(ffn, m):
    F, S, gate_up, down, sp_gu, sp_dn = ffn
    q, s = _inputs(m, HIDDEN, m)
    h_ref, amax_ref = _dense_h(F, q, s, gate_up)
    h, amax = S.gemm_swiglu(q, s, sp_gu, torch.bfloat16)
    assert torch.equal(h.view(torch.int16), h_ref.view(torch.int16))
    assert torch.equal(amax.amax(1), amax_ref)
    hq_ref, scale_ref = F.swiglu_gemm_quant(q, s, gate_up, torch.bfloat16)
    hq, scale = S.swiglu_gemm_quant(q, s, sp_gu, torch.bfloat16)
    assert torch.equal(hq, hq_ref) and torch.equal(scale, scale_ref)


@pytest.mark.parametrize("m", ROWS)
def test_residual_bit_identical(ffn, m):
    F, S, gate_up, down, sp_gu, sp_dn = ffn
    q, s = _inputs(m, INTER, 1000 + m)
    res = torch.randn(m, HIDDEN, device="cuda", dtype=torch.bfloat16)
    ref = F._gemm(q, s, down, torch.bfloat16, residual=res)
    out = S.gemm_res(q, s, sp_dn, torch.bfloat16, res)
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))


@pytest.mark.parametrize("cfg", sorted(__import__("voxcpm_kernels.sparse24", fromlist=["CONFIGS"]).CONFIGS))
@pytest.mark.parametrize("m", [1, 37, 300])
def test_every_tile_configuration(ffn, cfg, m):
    F, S, gate_up, down, sp_gu, sp_dn = ffn
    q, s = _inputs(m, HIDDEN, 7 * m + cfg)
    h_ref, amax_ref = _dense_h(F, q, s, gate_up)
    h, amax = S.gemm_swiglu(q, s, sp_gu, torch.bfloat16, cfg=cfg)
    assert torch.equal(h.view(torch.int16), h_ref.view(torch.int16)) and torch.equal(amax.amax(1), amax_ref)
    if cfg in S.SWIGLU_ONLY:
        return
    q2, s2 = _inputs(m, INTER, 9 * m + cfg)
    res = torch.randn(m, HIDDEN, device="cuda", dtype=torch.bfloat16)
    ref = F._gemm(q2, s2, down, torch.bfloat16, residual=res)
    assert torch.equal(S.gemm_res(q2, s2, sp_dn, torch.bfloat16, res, cfg=cfg).view(torch.int16),
                       ref.view(torch.int16))


@pytest.mark.parametrize("m", [8, 200])
def test_cuda_graph_replay(ffn, m):
    """The FFN (sparse SwiGLU GEMM -> quant -> sparse residual GEMM) captured once, replayed on new inputs."""
    F, S, gate_up, down, sp_gu, sp_dn = ffn
    q, s = _inputs(m, HIDDEN, 1)
    res = torch.randn(m, HIDDEN, device="cuda", dtype=torch.bfloat16)

    def ffn_sparse():
        return S.gemm_res(*S.swiglu_gemm_quant(q, s, sp_gu, torch.bfloat16), sp_dn, torch.bfloat16, res)

    ffn_sparse()  # warm-up (builds, sets the shared-memory attributes)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = ffn_sparse()
    for seed in (2, 3):
        q2, s2 = _inputs(m, HIDDEN, seed)
        q.copy_(q2)
        s.copy_(s2)
        res.normal_()
        graph.replay()
        ref = F._gemm(*F.swiglu_gemm_quant(q, s, gate_up, torch.bfloat16), down, torch.bfloat16, residual=res)
        assert torch.equal(out.view(torch.int16), ref.view(torch.int16))
