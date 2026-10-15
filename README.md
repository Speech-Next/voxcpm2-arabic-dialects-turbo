# VoxCPM2 Arabic Dialects (Turbo)

Fast Arabic dialect text-to-speech with voice cloning, on [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) with one
LoRA per dialect.

## Results

One NVIDIA A10, installed with `./setup.sh`. The test clones the 32 voices of
[`VoxCPM-Kernel/Arabic-Dialects-and-Hindi-Voice-Samples`](https://huggingface.co/datasets/VoxCPM-Kernel/Arabic-Dialects-and-Hindi-Voice-Samples),
each in its own dialect's LoRA.

| requests at once | first audio, avg | first audio, p95 | whole sentence, avg |
|---:|---:|---:|---:|
| 1 | 24 ms | 26 ms | 0.50 s |
| 16 | 28 ms | 29 ms | 0.46 s |
| 64 | 39 ms | 40 ms | 0.86 s |
| 128 | 64 ms | 66 ms | 1.36 s |
| 200 | 98 ms | 99 ms | 1.94 s |
| 256 | 124 ms | 125 ms | 2.40 s |

| mode | LocDiT steps | first audio p95, 200 requests | CER | WER | UTMOS |
|---|---|---:|---:|---:|---:|
| fast (default) | 2 | 99 ms | 4.6 % | 13.5 % | 2.41 |
| quality (`dit=None`) | 10, with CFG | 277 ms | 4.2 % | 12.5 % | 2.77 |

| dialect (fast mode) | voices | CER | WER | UTMOS |
|---|---:|---:|---:|---:|
| jordanian | 17 | 5.8 % | 15.8 % | 2.34 |
| palestinian | 5 | 3.4 % | 10.4 % | 2.25 |
| qatari | 3 | 1.8 % | 7.8 % | 2.80 |
| saudi | 3 | 2.6 % | 11.4 % | 2.65 |
| hindi_english | 4 | 8.9 % | 13.5 % | 2.02 |

- CER / WER: the better of Whisper large-v3 and Cohere Transcribe Arabic (Whisper only for Hindi).
- 6 sentences per voice. The mode table covers the Arabic voices only.

## Install

Needs Linux, an NVIDIA GPU (Ampere or newer) and the CUDA toolkit (`nvcc`).

```bash
git clone https://github.com/Speech-Next/voxcpm2-arabic-dialects-turbo && cd voxcpm2-arabic-dialects-turbo
./setup.sh                      # Python env, torch, flash-attn and this package, in one step
source .venv/bin/activate
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
        dit="VoxCPM-Kernel/VoxCPM2-Turbo-distilled-DiT:student_k2_dmd.pt",  # 2 steps; None = 10 steps (quality)
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
python -m nanovllm_voxcpm.entrypoints.openai_server --model VoxCPM-Kernel/VoxCPM2-Arabic-Dialects \
    --dit VoxCPM-Kernel/VoxCPM2-Turbo-distilled-DiT:student_k2_dmd.pt                                # HTTP server, :8092
```

License: MIT. VoxCPM2 weights: Apache-2.0.
