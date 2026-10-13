# VoxCPM2 Arabic Dialects (Turbo)

Fast Arabic dialect TTS with voice cloning, built on [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2).

| requests at once (one A10) | first audio, avg | first audio, p95 | whole sentence, avg |
|---:|---:|---:|---:|
| 1 | 31 ms | 69 ms | 0.37 s |
| 64 | 44 ms | 44 ms | 0.98 s |
| 200 | 97 ms | 98 ms | 1.96 s |
| 256 | 123 ms | 124 ms | 2.43 s |

## Install

Needs Linux, an NVIDIA GPU, Python 3.10-3.12 and the CUDA toolkit (`nvcc`, `ninja`). Tested with an A10, torch 2.9.1
(cu128) and flash-attn 2.8.3.

```bash
git clone https://github.com/Speech-Next/voxcpm2-arabic-dialects-turbo && cd voxcpm2-arabic-dialects-turbo
uv venv --python 3.12 && source .venv/bin/activate
uv pip install torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install -e ".[server]"
hf auth login        # the model repos are private
```

## Usage

```python
import asyncio
from nanovllm_voxcpm import DialectTTS

async def main():
    async with await DialectTTS.load() as tts:
        await tts.add_voice("ahmed", "ahmed.wav")
        audio = await tts.synthesize("هلا والله، كيف حالك اليوم؟", dialect="najdi", voice="ahmed")  # 48 kHz

if __name__ == "__main__":
    asyncio.run(main())
```

- **Model:** `VoxCPM-Kernel/VoxCPM2-Arabic-Dialects` by default, with one LoRA per dialect.
  - Dialects: ab, egyptian, emirati, hejazi, hindi_english, iraqi, jordanian, najdi, palestinian, qatari, saudi.
  - The 17-dialect model: `DialectTTS.load("VoxCPM-Kernel/VoxCPM2-Arabic-Teacher-v2", dit=None, w8a8=False)`.
- **Quality mode:** `DialectTTS.load(dit=None)`, with lower WER (13.6 % vs 17.5 %) but a first-audio p95 of 294 ms at
  200 requests.
- **Streaming:** `tts.stream(...)` yields the audio while it is generated.

## Server

```bash
NANOVLLM_BUSY_POLL_S=600 python -m nanovllm_voxcpm.entrypoints.openai_server --dit kadirnar/VoxCPM2-Turbo-distilled-DiT:student_k2_dmd.pt
curl -F name=ahmed -F audio_sample=@ahmed.wav localhost:8092/v1/audio/voices
curl localhost:8092/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input": "هلا والله", "dialect": "najdi", "voice": "ahmed", "response_format": "wav"}' > out.wav
```

## Benchmark

```bash
NANOVLLM_BUSY_POLL_S=30 python examples/benchmark.py --voice-dir voices/   # a folder of reference .wav clips
```

The first start takes 1-2 minutes, to compile a kernel and capture CUDA graphs.

License: MIT. VoxCPM2 weights: Apache-2.0.
