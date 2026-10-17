# VoxCPM2 Arabic Dialects (Turbo)

Fast Arabic dialect text-to-speech with voice cloning, on [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) with one
LoRA per dialect.

## Results

One NVIDIA A10, installed with `./setup.sh`. The test clones the 32 voices of
[`VoxCPM-Kernel/Arabic-Hindi-Reference-Voices`](https://huggingface.co/datasets/VoxCPM-Kernel/Arabic-Hindi-Reference-Voices),
each in its own dialect's LoRA.

| requests at once | first audio, avg | first audio, p95 | whole sentence, avg |
|---:|---:|---:|---:|
| 1 | 25 ms | 25 ms | 0.41 s |
| 16 | 28 ms | 29 ms | 0.55 s |
| 64 | 40 ms | 40 ms | 0.95 s |
| 128 | 67 ms | 68 ms | 1.56 s |
| 200 | 103 ms | 103 ms | 2.17 s |
| 256 | 131 ms | 133 ms | 2.67 s |

| mode | LocDiT steps | first audio p95, 200 requests | CER | WER | UTMOS |
|---|---|---:|---:|---:|---:|
| fast (default) | 3, distilled for these adapters | 103 ms | 5.3 % | 15.9 % | 2.23 |
| faster (`dit=...:student_k2_dmd.pt`) | 2, distilled for these adapters | 92 ms | 5.4 % | 15.9 % | 2.14 |
| quality (`dit=None`) | 10, with CFG | 270 ms | 5.3 % | 15.5 % | 2.40 |

- Quality: the held-out evaluation set, 17 dialects × 40 sentences × 2 seeds (1,320 clips), each voice cloned from
  another clip of its speaker, default settings (int8). Per dialect: the
  [distilled LocDiT's card](https://huggingface.co/VoxCPM-Kernel/VoxCPM2-Arabic-Dialects-Distilled-DiT).
- CER / WER: the better of Whisper large-v3 and Cohere Transcribe Arabic; lower is better. UTMOS: predicted
  naturalness, 1-5.

## Install

Needs Linux, an NVIDIA GPU (Ampere or newer) and the CUDA toolkit (`nvcc`).

```bash
git clone https://github.com/Speech-Next/voxcpm2-arabic-dialects-turbo && cd voxcpm2-arabic-dialects-turbo
./setup.sh                      # Python env, torch, flash-attn and this package, in one step
source .venv/bin/activate
hf auth login                   # only for the benchmark voices (gated dataset, request access first)
```

Models and voices are downloaded automatically on first use. The first start takes 1-2 minutes, to compile a kernel
and capture CUDA graphs.

## Run

```python
import asyncio
from nanovllm_voxcpm import DialectTTS

async def main():
    tts = await DialectTTS.load(
        "VoxCPM-Kernel/VoxCPM2-Arabic-Dialects",                       # VoxCPM2 + dialect LoRAs
        dit="VoxCPM-Kernel/VoxCPM2-Arabic-Dialects-Distilled-DiT:student_k3.pt",  # 3 steps (the default)
        # dit="VoxCPM-Kernel/VoxCPM2-Arabic-Dialects-Distilled-DiT:student_k2_dmd.pt": 2 steps, faster; None: 10 (quality)
    )
    await tts.add_voice("ahmed", "ahmed.wav")
    audio = await tts.synthesize("هلا والله، كيف حالك اليوم؟", dialect="najdi", voice="ahmed")  # 48 kHz
    await tts.close()

if __name__ == "__main__":
    asyncio.run(main())
```

Dialects: ab, egyptian, emirati, hejazi, hindi_english, iraqi, jordanian, najdi, palestinian, qatari, saudi.

```bash
python examples/voice_clone.py --voice ahmed.wav --dialect najdi --text "هلا والله" --out out.wav   # one file
NANOVLLM_BUSY_POLL_S=30 python examples/benchmark.py                                             # the speed table
NANOVLLM_BUSY_POLL_S=30 python examples/benchmark.py --dit none --levels 200 --kv-blocks 400     # quality mode
python -m nanovllm_voxcpm.entrypoints.openai_server --model VoxCPM-Kernel/VoxCPM2-Arabic-Dialects \
    --dit VoxCPM-Kernel/VoxCPM2-Arabic-Dialects-Distilled-DiT:student_k3.pt                          # HTTP server, :8092
```

License: MIT. VoxCPM2 weights: Apache-2.0.
