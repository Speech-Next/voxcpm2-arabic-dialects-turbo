# VoxCPM2 Arabic Dialects (Turbo)

Fast Arabic dialect text-to-speech with voice cloning, on [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2): one LoRA
per dialect, and [teacher-v3](https://huggingface.co/VoxCPM-Kernel/VoxCPM2-Arabic-Teacher-v3) (one model, a tag per
dialect) served as one more adapter of the same engine. Each dialect is spoken by the one that is better at it.

## Results

One NVIDIA A10, installed with `./setup.sh`. The test clones the 32 voices of
[`VoxCPM-Kernel/Arabic-Hindi-Reference-Voices`](https://huggingface.co/datasets/VoxCPM-Kernel/Arabic-Hindi-Reference-Voices),
each in its own dialect.

| engine | CER | WER | UTMOS | dialect ID | first audio p95, 200 requests | GPU memory |
|---|---:|---:|---:|---:|---:|---:|
| routed (the default, `RoutedDialectTTS`) | 4.6 % | 13.9 % | 2.22 | 64 % | 107 ms | 21.2 GiB |
| adapters only (`DialectTTS`) | 5.3 % | 16.0 % | 2.23 | 67 % | 105 ms | 22.0 GiB |

| requests at once | first audio p95, routed | adapters only | whole sentence, routed | adapters only |
|---:|---:|---:|---:|---:|
| 1 | 25 ms | 27 ms | 0.42 s | 0.41 s |
| 64 | 43 ms | 42 ms | 0.96 s | 0.96 s |
| 128 | 71 ms | 68 ms | 1.50 s | 1.56 s |
| 200 | 107 ms | 105 ms | 2.14 s | 2.17 s |
| 256 | 135 ms | 132 ms | 2.65 s | 2.67 s |

- Quality: the held-out evaluation set, 17 dialects × 40 sentences × 2 seeds (1,320 clips), each voice cloned from
  another clip of its speaker, default settings (int8, the 3-step distilled LocDiT). Routed vs adapters only, paired:
  CER −0.7 and WER −2.0 points (significant), UTMOS and DNSMOS unchanged, dialect ID −3.6 points.
- CER / WER: the better of Whisper large-v3 and Cohere Transcribe Arabic; lower is better. UTMOS: predicted
  naturalness, 1-5. Speed: two alternated runs of each engine, averaged.
- Routing: bahraini (the qatari adapter), egyptian, saudi, hejazi and hindi_english on their adapters, the other 13
  dialects of teacher-v3 (and their aliases, e.g. gulf / qatari / emirati, palestinian / jordanian) on teacher-v3.
- Other LocDiTs: `dit="VoxCPM-Kernel/VoxCPM2-Arabic-Dialects-Distilled-DiT:student_k2_dmd.pt"` (2 steps, ~10 ms faster
  at 200 requests, less natural), `dit=None` (10 steps with CFG, quality mode, ~270 ms).

Streaming, with requests arriving at random (Poisson, 20 s per rate) and a player that starts at the first audio
(`python examples/streaming_benchmark.py`):

| requests per second | first audio p95 | first word p95 | streams with a gap > 50 ms |
|---:|---:|---:|---:|
| 30 | 80 ms | 456 ms | 0 % |
| 40 | 97 ms | 460 ms | 0 % |
| 50 | 118 ms | 488 ms | 0 % |
| 60 | 142 ms | 509 ms | 0 % |

- First word: when the player reaches the speech's start (the median sentence starts speaking 170 ms into its audio).
- A stream's leading silence is capped at 320 ms, and no new request starts while a playing stream has less than
  150 ms of audio queued. Without these, 0.5 / 7.4 / 85 / 93 % of the streams had a gap.
- Requests sent at the same instant: gap-free up to 160. 200 or more at once exceed one A10: gaps of 0.1-0.5 s after
  the first half second.

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
from nanovllm_voxcpm import RoutedDialectTTS

async def main():
    tts = await RoutedDialectTTS.load()   # VoxCPM2-Arabic-Dialects + teacher-v3, the 3-step LocDiT (dit=None: quality)
    await tts.add_voice("ahmed", "ahmed.wav")
    audio = await tts.synthesize("هلا والله، كيف حالك اليوم؟", dialect="najdi", voice="ahmed")  # 48 kHz
    await tts.close()

if __name__ == "__main__":
    asyncio.run(main())
```

Dialects (`tts.dialects`): msa, egyptian, palestinian, lebanese, syrian, iraqi, saudi, najdi, hejazi, gulf, bahraini,
yemeni, sudanese, libyan, tunisian, algerian, moroccan, hindi_english, and the aliases ab, emirati, jordanian,
khaliji, levantine, qatari. `DialectTTS.load()` (same methods) is the engine with the adapters only.

```bash
python examples/voice_clone.py --voice ahmed.wav --dialect najdi --text "هلا والله" --out out.wav   # one file
NANOVLLM_BUSY_POLL_S=30 python examples/benchmark.py                                             # the speed table
NANOVLLM_BUSY_POLL_S=30 python examples/benchmark.py --dit none --levels 200 --kv-blocks 400     # quality mode
python -m nanovllm_voxcpm.entrypoints.openai_server --model VoxCPM-Kernel/VoxCPM2-Arabic-Dialects \
    --dit VoxCPM-Kernel/VoxCPM2-Arabic-Dialects-Distilled-DiT:student_k3.pt                          # HTTP server, :8092
```

Every command routes by default; `--no-route` runs the adapters only.

License: MIT. VoxCPM2 weights: Apache-2.0.
