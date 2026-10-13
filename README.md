# VoxCPM2 Arabic Dialects (Turbo)

Fast Arabic dialect TTS with voice cloning, built on [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2).

| requests at once (one A10) | first audio, avg | first audio, p95 | whole sentence, avg |
|---:|---:|---:|---:|
| 1 | 31 ms | 69 ms | 0.37 s |
| 64 | 44 ms | 44 ms | 0.98 s |
| 200 | 97 ms | 98 ms | 1.96 s |
| 256 | 123 ms | 124 ms | 2.43 s |

The configuration measured:

| | |
|---|---|
| model | `VoxCPM-Kernel/VoxCPM2-Arabic-Dialects` (VoxCPM2 + dialect LoRAs, mixed in one batch) |
| LocDiT | `kadirnar/VoxCPM2-Turbo-distilled-DiT:student_k2_dmd.pt`: 2 steps, no CFG |
| precision | int8 W8A8 (`w8a8=True`) |
| head start | first audio from the first 8 text tokens (`head_tokens=8`) |
| requests | voice cloning from reference audio of 5 voices, one sentence each (~7 words) |
| command | `NANOVLLM_BUSY_POLL_S=30 python examples/benchmark.py --voice-dir voices/` |
| hardware | one NVIDIA A10, nothing else on the GPU |

## Setup

**Requirements**
- Linux and an NVIDIA GPU of the Ampere generation or newer (flash-attn); tested: A10 24 GB, driver 570.86.
- Python 3.10-3.12 (tested 3.12.3).
- The CUDA toolkit 12.x with `nvcc` (tested 12.9) and `ninja`: a CUDA kernel is compiled on the first start.
- About 12 GB of disk for the models.
- A Hugging Face account with read access to the model repos below (they are private).

Tested versions: torch 2.9.1+cu128, torchaudio 2.9.1, triton 3.5.1, flash-attn 2.8.3, transformers 5.19,
huggingface_hub 1.33.

**1. Install**

```bash
git clone https://github.com/Speech-Next/voxcpm2-arabic-dialects-turbo
cd voxcpm2-arabic-dialects-turbo
uv venv --python 3.12 && source .venv/bin/activate
uv pip install torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install -e ".[server]"      # also builds flash-attn against this torch (several minutes)
```

With pip instead of uv, run `pip install ninja && pip install flash-attn --no-build-isolation` first, then
`pip install -e ".[server]"`.

**2. Hugging Face login** (once)

```bash
hf auth login                      # or set HF_TOKEN in the environment
```

**3. Models** (downloaded on first use; to download them ahead of time, into the Hugging Face cache)

```bash
hf download VoxCPM-Kernel/VoxCPM2-Arabic-Dialects
hf download kadirnar/VoxCPM2-Turbo-distilled-DiT student_k2_dmd.pt
hf download VoxCPM-Kernel/VoxCPM2-Arabic-Teacher-v2          # optional, the 17-dialect model
```

**4. Reference voice:** a `.wav` of 5-10 s of one speaker, without music. Any sample rate works.

**5. First start:** compiling the kernel and capturing CUDA graphs takes 1-2 minutes. The kernel is then cached in
`~/.cache/torch_extensions`. The engine takes 85 % of the GPU memory.

## Usage

```python
import asyncio
from nanovllm_voxcpm import DialectTTS

async def main():
    tts = await DialectTTS.load(
        "VoxCPM-Kernel/VoxCPM2-Arabic-Dialects",                     # VoxCPM2 + one LoRA per dialect
        dit="kadirnar/VoxCPM2-Turbo-distilled-DiT:student_k2_dmd.pt",  # distilled LocDiT: 2 steps, no CFG
    )
    await tts.add_voice("ahmed", "ahmed.wav")
    audio = await tts.synthesize("هلا والله، كيف حالك اليوم؟", dialect="najdi", voice="ahmed")  # 48 kHz
    await tts.close()

if __name__ == "__main__":
    asyncio.run(main())
```

| model id | `dit` | LocDiT steps | first audio p95, 200 req. | WER |
|---|---|---|---:|---:|
| `VoxCPM-Kernel/VoxCPM2-Arabic-Dialects` | `kadirnar/VoxCPM2-Turbo-distilled-DiT:student_k2_dmd.pt` | 2, no CFG | 98 ms | 17.5 % |
| `VoxCPM-Kernel/VoxCPM2-Arabic-Dialects` | `None` | 10, with CFG | 294 ms | 13.6 % |
| `VoxCPM-Kernel/VoxCPM2-Arabic-Teacher-v2` (also pass `w8a8=False`) | `None` | 10, with CFG | not measured | not measured |

- **VoxCPM2-Arabic-Dialects:** ab, egyptian, emirati, hejazi, hindi_english, iraqi, jordanian, najdi, palestinian,
  qatari, saudi.
- **VoxCPM2-Arabic-Teacher-v2:** 17 dialects, msa to moroccan (see its model card).
- **Streaming:** `tts.stream(...)` yields the audio while it is generated.

## Server

```bash
NANOVLLM_BUSY_POLL_S=600 python -m nanovllm_voxcpm.entrypoints.openai_server \
    --model VoxCPM-Kernel/VoxCPM2-Arabic-Dialects \
    --dit kadirnar/VoxCPM2-Turbo-distilled-DiT:student_k2_dmd.pt     # 2 steps; without --dit: --inference-timesteps 10
curl -F name=ahmed -F audio_sample=@ahmed.wav localhost:8092/v1/audio/voices
curl localhost:8092/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input": "هلا والله", "dialect": "najdi", "voice": "ahmed", "response_format": "wav"}' > out.wav
```

- **Port:** 8092 on 127.0.0.1 (`--host 0.0.0.0` to listen on all interfaces).
- **`NANOVLLM_BUSY_POLL_S=600`:** keeps the engine awake for 10 minutes after a request, so the first audio stays fast
  after idle periods. It costs one CPU core.

## Benchmark

```bash
NANOVLLM_BUSY_POLL_S=30 python examples/benchmark.py --voice-dir voices/   # a folder of reference .wav clips
```

This prints the table at the top: per level, one warm-up burst and 3 measured bursts. Run it on an otherwise idle GPU.

License: MIT. VoxCPM2 weights: Apache-2.0.
