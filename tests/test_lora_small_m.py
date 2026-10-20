"""The small-M LoRA shrink kernel (decode steps and CUDA-graph capture with < 32 rows) against torch, on a GPU.

With triton 3.4 its tl.dot of one row failed to compile (M, N, K >= 16): every bf16 engine with adapters died while
capturing its CUDA graphs. Skipped without CUDA.
"""

import pytest
import torch

from nanovllm_voxcpm.lora_ops.triton_ops.lora_shrink_op import _use_small_m_shrink_path, lora_shrink

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("K", [64, 200, 1024, 2048, 6144])  # 6144: split-K with atomic adds
@pytest.mark.parametrize("M", [1, 2, 5, 31])
@pytest.mark.parametrize("slices", [1, 3])
def test_small_m_shrink_matches_torch(M, K, slices):
    torch.manual_seed(M * 7 + K + slices)
    max_loras, N, scaling = 4, 32, 2.0
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    a = [torch.randn(max_loras, N, K, device="cuda", dtype=torch.bfloat16) / K**0.5 for _ in range(slices)]
    mapping = torch.randint(-1, max_loras, (M,), device="cuda", dtype=torch.int32)
    mapping[0] = 1  # at least one row with an adapter
    out = torch.full((slices, M, N), float("nan"), device="cuda", dtype=torch.bfloat16)
    assert _use_small_m_shrink_path(M=M, N=N, num_active_loras=max_loras, num_slices=slices)

    dummy = torch.zeros(max_loras + 1, device="cuda", dtype=torch.int32)
    lora_shrink(x, a, out, mapping, dummy, dummy, dummy, dummy, False, max_loras, scaling)
    torch.cuda.synchronize()

    ref = torch.zeros(slices, M, N, device="cuda", dtype=torch.float32)
    for s in range(slices):
        for m in range(M):
            if (lid := int(mapping[m])) >= 0:
                ref[s, m] = scaling * (a[s][lid].float() @ x[m].float())
    # split-K (K = 6144) adds its partial sums into the bf16 output with atomics, in no fixed order, each add rounded
    # to bf16: the error varies run to run, up to 0.032 over 0.02 |ref| in 1,200 runs (2e-2 failed now and then)
    torch.testing.assert_close(out.float(), ref, atol=6e-2 if K > 2048 else 2e-2, rtol=2e-2)
