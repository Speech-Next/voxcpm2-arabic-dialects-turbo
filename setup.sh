#!/bin/bash
# Everything in one step: uv, a .venv (Python 3.12), torch 2.8.0 for CUDA 12.8, the prebuilt flash-attn wheel for that
# torch (no compiling), this package with the server, and its two CUDA kernels (nvcc, ~1.5 min). Models and voices
# are downloaded on first use.
set -euo pipefail
cd "$(dirname "$0")"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv venv --python 3.12 .venv
. .venv/bin/activate
uv pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
uv pip install ninja -e ".[server]"
python -m nanovllm_voxcpm.kernels.cuda_ext
python -c "import torch, flash_attn, nanovllm_voxcpm; print('ready: torch', torch.__version__, 'GPU', torch.cuda.get_device_name())"
