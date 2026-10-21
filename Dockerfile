# syntax=docker/dockerfile:1
# Either server, one GPU per container (docs/details.md, Docker):
#   docker build -t voxcpm2-turbo --build-arg CUDA_ARCH=8.6 .
#   docker run -d --gpus '"device=0"' --shm-size=1g --stop-timeout 30 -p 8092:8092 -e HF_TOKEN -e VOXCPM_API_KEY \
#     -v voxcpm-cache:/cache -v voxcpm-voices:/voices voxcpm2-turbo http --max-streams 150 --voice-dir /voices
# The same steps as setup.sh; nvcc (a devel image) builds the two CUDA kernels.
ARG CUDA_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu24.04
FROM ${CUDA_IMAGE}

# the GPU architectures the CUDA kernels are built for now (no GPU at docker build), e.g. 8.6 (A10) or 8.6,8.9,9.0
# (A10, L4 / RTX 4090, H100); another GPU builds its own at its first start (~2 min)
ARG CUDA_ARCH=8.6
ARG USER_UID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    TORCH_EXTENSIONS_DIR=/opt/torch_extensions \
    HF_HOME=/cache/huggingface \
    TRITON_CACHE_DIR=/cache/triton \
    TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
    PORT=8092

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl ffmpeg g++ tini \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.12.7 /uv /uvx /bin/

# Python 3.12, torch 2.8.0 for CUDA 12.8, the prebuilt flash-attn wheel for that torch (no compiling)
RUN uv venv --python 3.12 /opt/venv \
    && uv pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128 \
    && uv pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

# the package's dependencies before its source: a code change does not install them again
WORKDIR /app
COPY pyproject.toml README.md LICENSE THIRD_PARTY_NOTICES.md ./
RUN uv pip install ninja -r pyproject.toml --extra server

COPY nanovllm_voxcpm nanovllm_voxcpm
COPY examples examples
RUN uv pip install --no-deps -e . && python -m compileall -q nanovllm_voxcpm examples

# the CUDA kernels for CUDA_ARCH (nanovllm_voxcpm.kernels.cuda_ext, VOXCPM_CUDA_ARCH): a start loads them, no nvcc
RUN VOXCPM_CUDA_ARCH=${CUDA_ARCH} python -m nanovllm_voxcpm.kernels.cuda_ext > /tmp/kernels.log 2>&1; status=$?; \
    cat /tmp/kernels.log; [ $status -eq 0 ] && ! grep -q "not built" /tmp/kernels.log && rm /tmp/kernels.log

COPY --chmod=755 docker/entrypoint.sh /usr/local/bin/voxcpm-entrypoint
# a non-root user (the ubuntu image's own uid 1000 user made way); /cache: the models (HF_HOME), the compiled Triton
# kernels and torch.compile's graphs, /voices: --voice-dir, both kept on volumes
RUN (userdel -r ubuntu 2>/dev/null || true) \
    && useradd --create-home --uid ${USER_UID} voxcpm \
    && mkdir -p /cache /voices \
    && chown -R voxcpm /cache /voices /opt/torch_extensions
USER voxcpm

EXPOSE 8092
# the first start downloads the models and compiles the Triton kernels (~3 min, more on a slow link)
HEALTHCHECK --interval=30s --timeout=5s --start-period=15m --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1
# tini: SIGTERM reaches the server (its drain: --shutdown-timeout), its exited processes are reaped
ENTRYPOINT ["/usr/bin/tini", "--", "voxcpm-entrypoint"]
CMD ["http"]
