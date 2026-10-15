"""Packing of 2:4 sparse int8 weights for the mma.sp FFN kernels (voxcpm_kernels/sparse24.py), on the CPU. The GPU
kernels themselves: tests/gpu/test_sparse24_gemm.py."""

import pytest
import torch

from voxcpm_kernels import sparse24 as S


@pytest.mark.parametrize("n,k", [(16, 128), (64, 256), (32, 6144)])
def test_pack_round_trip(n, k):
    torch.manual_seed(n + k)
    w = S.prune_24(torch.randint(-127, 128, (n, k), dtype=torch.int8))
    w[0, :8] = 0  # groups with one and with no non-zero
    w[1, 4:6] = 0
    assert S.is_24(w)
    wf, mf = S.compress_24(w)
    assert wf.shape == (n // 16, k // 64, 32, 16) and wf.dtype == torch.uint8
    assert mf.shape == (n // 16, k // 64, 128) and mf.dtype == torch.uint8
    assert torch.equal(S.decompress_24(wf, mf), w)


def test_fragment_order_of_one_tile():
    """Lane (g, t) of a 16 x 64 tile: values a0..a3 and the metadata nibbles as mma.sp reads them."""
    n, k = 16, 128
    w = torch.zeros(n, k, dtype=torch.int8)
    for r in range(n):  # row r keeps positions (1, 3) of every group, values 2 G + j + r (mod 100)
        for G in range(k // 4):
            w[r, 4 * G + 1], w[r, 4 * G + 3] = (2 * G + r) % 100 + 1, (2 * G + 1 + r) % 100 + 1
    wf, mf = S.compress_24(w)
    for lane in range(32):
        g, t = lane // 4, lane % 4
        regs = wf[0, 0, lane].view(torch.int8).view(4, 4)  # a0..a3, 4 bytes each
        for reg, (row, c0) in enumerate(((g, 0), (g + 8, 0), (g, 16), (g + 8, 16))):
            cols = torch.arange(c0 + 4 * t, c0 + 4 * t + 4)  # compressed columns 2 G + j
            assert regs[reg].tolist() == [((c + row) % 100) + 1 for c in cols.tolist()]
        word = int(mf[0, 0].view(torch.int32)[lane]) & 0xFFFFFFFF
        assert word == sum((1 | (3 << 2)) << (4 * i) for i in range(8))  # every nibble: positions 1 and 3


def test_not_24_is_rejected():
    w = torch.ones(16, 128, dtype=torch.int8)
    assert not S.is_24(w)
    with pytest.raises(ValueError):
        S.compress_24(w)


def test_swiglu_rows():
    assert S.swiglu_rows(32).tolist() == [0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15,
                                          16, 18, 20, 22, 24, 26, 28, 30, 17, 19, 21, 23, 25, 27, 29, 31]


def test_prune_keeps_the_two_largest():
    w = torch.tensor([[1.0, -5.0, 3.0, 0.5, 2.0, 2.5, -3.0, 0.0]])
    assert S.prune_24(w).tolist() == [[0.0, -5.0, 3.0, 0.0, 0.0, 2.5, -3.0, 0.0]]


@pytest.mark.parametrize("m", [1, 16, 64, 200, 256])
def test_res_fma_follows_the_dense_epilogue(m):
    """Up to 256 rows the dense residual GEMM's tiles are below 128 x 128: single-pass epilogue, one fma."""
    assert S.res_fma(m, 2048)
