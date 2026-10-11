"""CUDA C++ kernels of voxcpm_kernels (cuda/*.cu), compiled on first use with torch.utils.cpp_extension.

The build is cached (TORCH_EXTENSIONS_DIR, default ~/.cache/torch_extensions), so only the first import on a machine
pays the nvcc time. Needs nvcc and ninja; ``available()`` is False when the build fails, and callers fall back to the
Triton kernels.
"""

import functools
import hashlib
import os

import torch

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuda")


@functools.cache
def load(name: str, sweep: bool = False):
    """``sweep`` also compiles the tile variants that only the benchmark sweeps use (-D<NAME>_SWEEP)."""
    from torch.utils.cpp_extension import load as _load

    major, minor = torch.cuda.get_device_capability()
    src = os.path.join(_DIR, f"{name}.cu")
    # the source hash in the name: two versions of the kernels (A/B in one process) get two builds and modules
    digest = hashlib.sha1(open(src, "rb").read()).hexdigest()[:10]
    return _load(
        name=f"voxcpm_{name}_sm{major}{minor}_{digest}" + ("_sweep" if sweep else ""),
        sources=[src],
        extra_cuda_cflags=["-O3", "-std=c++17", f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}",
                           "--expt-relaxed-constexpr"] + ([f"-D{name.split('_')[0].upper()}_SWEEP"] if sweep else []),
        extra_cflags=["-O3", "-std=c++17"],
        verbose=os.environ.get("VOXCPM_KERNELS_BUILD_VERBOSE") == "1",
    )


def available(name: str) -> bool:
    try:
        load(name)
        return True
    except Exception as e:  # no nvcc / ninja, or a compile error: the Triton path stays
        print(f"[voxcpm_kernels] CUDA extension {name!r} unavailable: {str(e).splitlines()[-1][:200]}")
        return False
