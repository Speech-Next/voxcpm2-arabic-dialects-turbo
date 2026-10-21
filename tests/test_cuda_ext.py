"""The CUDA kernels' architecture: the GPU's own, or VOXCPM_CUDA_ARCH's without a visible GPU (docker build)."""

import re

import pytest
import torch

from nanovllm_voxcpm.kernels import cuda_ext


def test_the_gpu_decides_and_without_one_voxcpm_cuda_arch(monkeypatch):
    monkeypatch.setenv("VOXCPM_CUDA_ARCH", "8.6, 8.9,9.0")
    assert cuda_ext.archs() == [(8, 6), (8, 9), (9, 0)]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert cuda_ext.capability() == (8, 6)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (8, 9))
    assert cuda_ext.capability() == (8, 9)  # the GPU's own build, whatever the image was built for
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.delenv("VOXCPM_CUDA_ARCH")
    with pytest.raises(RuntimeError, match="VOXCPM_CUDA_ARCH"):
        cuda_ext.capability()


def test_a_build_for_an_architecture_without_a_gpu(monkeypatch, tmp_path):
    """The build's name and nvcc's -gencode follow the architecture (torch's build itself is not run here)."""
    import torch.utils.cpp_extension as cpp_extension

    calls = []
    monkeypatch.setattr(cpp_extension, "load", lambda **kw: calls.append(kw) or "module")
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("VOXCPM_CUDA_ARCH", "9.0,8.6")
    load = cuda_ext.load.__wrapped__  # (not functools.cache's: no module of this test kept for the others)
    assert load("resunit_fused") == "module" and load("resunit_fused", arch=(8, 6)) == "module"
    assert [re.search(r"_sm(\d+)_", kw["name"]).group(1) for kw in calls] == ["90", "86"]
    assert "-gencode=arch=compute_90,code=sm_90" in calls[0]["extra_cuda_cflags"]
    assert calls[0]["sources"][0].startswith(str(tmp_path))  # the copy in its build directory (_stable_source)
