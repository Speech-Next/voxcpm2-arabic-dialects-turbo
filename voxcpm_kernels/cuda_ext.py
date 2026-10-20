"""CUDA C++ kernels of voxcpm_kernels (cuda/*.cu), compiled on first use with torch.utils.cpp_extension.

The build is cached (TORCH_EXTENSIONS_DIR, default ~/.cache/torch_extensions): setup.sh builds both kernels
(``python -m voxcpm_kernels.cuda_ext``), so the first start does not wait for nvcc. Needs nvcc and ninja;
``available()`` is False when the build fails, and callers fall back to the Triton kernels.
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


@functools.cache  # a failed build is tried once per process, not by every layer that asks
def available(name: str) -> bool:
    try:
        load(name)
        return True
    except Exception as e:  # no nvcc / ninja, or a compile error: the Triton path stays
        last = (str(e).strip().splitlines() or [type(e).__name__])[-1]
        print(f"[voxcpm_kernels] CUDA extension {name!r} unavailable: {last[:200]}")
        return False


if __name__ == "__main__":  # setup.sh: every kernel built now, side by side, instead of on the first start
    from concurrent.futures import ProcessPoolExecutor

    names = sorted(f[:-3] for f in os.listdir(_DIR) if f.endswith(".cu"))
    with ProcessPoolExecutor(len(names)) as pool:
        for name, ok in zip(names, pool.map(available, names)):
            print(f"[voxcpm_kernels] {name}: " + ("built" if ok else "not built, the Triton kernels are used"))
