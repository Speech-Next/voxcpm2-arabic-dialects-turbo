"""CUDA C++ kernels of voxcpm_kernels (cuda/*.cu), compiled on first use with torch.utils.cpp_extension.

The build is cached (TORCH_EXTENSIONS_DIR, default ~/.cache/torch_extensions): setup.sh builds both kernels
(``python -m voxcpm_kernels.cuda_ext``), so the first start does not wait for nvcc. The cache holds one build per
source version and GPU architecture, shared by every checkout and worktree of the repo (_stable_source). Needs nvcc
and ninja; ``available()`` is False when the build fails, and callers fall back to the Triton kernels.

Without a visible GPU (``docker build``) the kernels are built for VOXCPM_CUDA_ARCH, the compute capabilities as
``8.6`` or ``8.6,8.9,9.0`` (A10, L4, H100); a start on a GPU then loads its own architecture's build, no nvcc.
"""

import functools
import hashlib
import os
import shutil
import sys

import torch

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuda")


def archs() -> list[tuple[int, int]]:
    """VOXCPM_CUDA_ARCH's compute capabilities (``8.6,8.9`` -> [(8, 6), (8, 9)]), in its order."""
    spec = os.environ.get("VOXCPM_CUDA_ARCH", "")
    return [(int(a.split(".")[0]), int(a.split(".")[1])) for a in spec.replace(" ", "").split(",") if a]


def capability() -> tuple[int, int]:
    """The architecture to build and load for: the GPU's; without a visible GPU, VOXCPM_CUDA_ARCH's first."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_capability()
    if not archs():
        raise RuntimeError("no GPU visible: set VOXCPM_CUDA_ARCH (e.g. 8.6) to build the CUDA kernels for")
    return archs()[0]


@functools.cache
def load(name: str, sweep: bool = False, arch: tuple[int, int] | None = None):
    """``sweep`` also compiles the tile variants that only the benchmark sweeps use (-D<NAME>_SWEEP); ``arch``: the
    compute capability to build for (default: capability())."""
    from torch.utils.cpp_extension import load as _load

    major, minor = arch or capability()
    src = os.path.join(_DIR, f"{name}.cu")
    # the source hash in the name: two versions of the kernels (A/B in one process) get two builds and modules
    digest = hashlib.sha1(open(src, "rb").read()).hexdigest()[:10]
    ext = f"voxcpm_{name}_sm{major}{minor}_{digest}" + ("_sweep" if sweep else "")
    # torch adds nvcc -gencode flags of its own, TORCH_CUDA_ARCH_LIST's or else the visible GPUs' (none, at docker
    # build: an IndexError); this build's architecture as torch derives it on that GPU (8.6+PTX on an A10) keeps the
    # build the same with or without a GPU, so a start finds the image's build up to date
    arch_list, path = os.environ.get("TORCH_CUDA_ARCH_LIST"), os.environ.get("PATH", "")
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list or f"{major}.{minor}+PTX"
    # the ninja next to this Python (pip's, with torch) before the shell's: another ninja version (/usr/bin's 1.11
    # against the venv's 1.13) reads the build's log as stale and compiles everything again (~3 min at a start)
    os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + path
    try:
        return _load(
            name=ext,
            sources=[_stable_source(src, ext)],
            extra_cuda_cflags=["-O3", "-std=c++17", f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}",
                               "--expt-relaxed-constexpr"]
            + ([f"-D{name.split('_')[0].upper()}_SWEEP"] if sweep else []),
            extra_cflags=["-O3", "-std=c++17"],
            verbose=os.environ.get("VOXCPM_KERNELS_BUILD_VERBOSE") == "1",
        )
    finally:
        os.environ["PATH"] = path
        if arch_list is None:
            del os.environ["TORCH_CUDA_ARCH_LIST"]


def _stable_source(src: str, ext: str) -> str:
    """``src`` copied into the build directory of ``ext`` (its name holds the source's hash), the path the build
    compiles: ninja's command line holds the source's absolute path, so a start from another checkout or worktree of
    the same source compiled everything again (~160 s of nvcc), and so did the next start from the first one."""
    from torch.utils.cpp_extension import _get_build_directory

    path = os.path.join(_get_build_directory(ext, verbose=False), os.path.basename(src))
    if not os.path.exists(path):
        tmp = f"{path}.{os.getpid()}.tmp"
        shutil.copyfile(src, tmp)
        os.replace(tmp, path)  # (whole: another process may build the same source meanwhile)
    return path


@functools.cache  # a failed build is tried once per process, not by every layer that asks
def available(name: str, arch: tuple[int, int] | None = None) -> bool:
    try:
        load(name) if arch is None else load(name, arch=arch)
        return True
    except Exception as e:  # no nvcc / ninja, or a compile error: the Triton path stays
        last = (str(e).strip().splitlines() or [type(e).__name__])[-1]
        print(f"[voxcpm_kernels] CUDA extension {name!r} unavailable: {last[:200]}")
        return False


if __name__ == "__main__":  # setup.sh, the Dockerfile: every kernel built now, side by side, not on the first start
    from concurrent.futures import ProcessPoolExecutor

    names = sorted(f[:-3] for f in os.listdir(_DIR) if f.endswith(".cu"))
    jobs = [(name, arch) for arch in (archs() or [capability()]) for name in names]  # VOXCPM_CUDA_ARCH, else the GPU
    with ProcessPoolExecutor(len(jobs)) as pool:
        for (name, (major, minor)), ok in zip(jobs, pool.map(available, *zip(*jobs))):
            print(f"[voxcpm_kernels] {name} (sm{major}{minor}): " +
                  ("built" if ok else "not built, the Triton kernels are used"))
