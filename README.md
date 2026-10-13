# VoxCPM2 Arabic Dialects (Turbo)

Fast Arabic dialect TTS with voice cloning, built on [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2).

## Results

Tested with the 32 voices of
[`VoxCPM-Kernel/Arabic-Dialects-and-Hindi-Voice-Samples`](https://huggingface.co/datasets/VoxCPM-Kernel/Arabic-Dialects-and-Hindi-Voice-Samples).
Each voice is cloned in its own dialect's LoRA of `VoxCPM-Kernel/VoxCPM2-Arabic-Dialects`:

| voices | LoRA |
|---|---|
| 17 Jordanian | `jordanian` |
| 5 Palestinian | `palestinian` |
| 3 Qatari | `qatari` |
| 3 Saudi | `saudi` |
| 4 Hindi | `hindi_english` |

Everything runs on one NVIDIA A10 with int8 W8A8 and an 8-token head start.

**Speed, fast mode:** distilled LocDiT, 2 steps, no CFG. All requests are sent at the same instant, each one sentence
of about 7 words.

| requests at once | first audio, avg | first audio, p95 | whole sentence, avg |
|---:|---:|---:|---:|
| 1 | 31 ms | 77 ms | 0.36 s |
| 16 | 28 ms | 35 ms | 0.46 s |
| 64 | 37 ms | 39 ms | 0.86 s |
| 128 | 62 ms | 64 ms | 1.34 s |
| 200 | 92 ms | 94 ms | 1.92 s |
| 256 | 119 ms | 120 ms | 2.37 s |

- At 1 request the median is 21 ms; its p95 is the slowest of 10 measurements.
- Quality mode at 200 requests: first audio p95 277 ms, whole sentence 5.47 s.

**Quality, fast mode:** 6 sentences per voice.

| dialect | CER | WER | naturalness (UTMOS) | speaker similarity |
|---|---:|---:|---:|---:|
| jordanian | 5.8 % | 15.8 % | 2.34 | 0.95 |
| palestinian | 3.4 % | 10.4 % | 2.25 | 0.91 |
| qatari | 1.8 % | 7.8 % | 2.80 | 0.94 |
| saudi | 2.6 % | 11.4 % | 2.65 | 0.97 |
| **Arabic, all** | **4.6 %** | **13.5 %** | **2.41** | **0.95** |
| hindi_english | 8.9 % | 13.5 % | 2.02 | 0.95 |

**Quality, quality mode** (`dit=None`): the model's own LocDiT, 10 steps with CFG.

| dialect | CER | WER | naturalness (UTMOS) | speaker similarity |
|---|---:|---:|---:|---:|
| jordanian | 5.3 % | 14.6 % | 2.76 | 0.96 |
| palestinian | 3.3 % | 10.4 % | 2.59 | 0.93 |
| qatari | 2.5 % | 9.9 % | 2.97 | 0.93 |
| saudi | 1.5 % | 6.8 % | 2.97 | 0.97 |
| **Arabic, all** | **4.2 %** | **12.5 %** | **2.77** | **0.95** |
| hindi_english | 10.9 % | 13.9 % | 2.22 | 0.94 |

How the quality was measured:
- **Arabic CER / WER:** the better of Whisper large-v3 and Cohere Transcribe Arabic. The sentences are held-out
  dialect sentences (Levantine for jordanian / palestinian, Gulf for qatari, Saudi for saudi).
- **Hindi:** Whisper large-v3 on 6 simple Hindi sentences (`examples/hi_sentences.txt`), a rough number.
- **Speaker similarity:** WavLM-SV against the reference clip.
- **Small samples:** each voice has 6 sentences, so differences between single voices are noise.

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
| `VoxCPM-Kernel/VoxCPM2-Arabic-Dialects` | `kadirnar/VoxCPM2-Turbo-distilled-DiT:student_k2_dmd.pt` | 2, no CFG | 94 ms | 13.5 % |
| `VoxCPM-Kernel/VoxCPM2-Arabic-Dialects` | `None` | 10, with CFG | 277 ms | 12.5 % |
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

The speed table:

```bash
hf download VoxCPM-Kernel/Arabic-Dialects-and-Hindi-Voice-Samples --repo-type dataset --local-dir voices
NANOVLLM_BUSY_POLL_S=30 python examples/benchmark.py --voice-dir voices/ --levels 1,16,64,128,200,256
NANOVLLM_BUSY_POLL_S=30 python examples/benchmark.py --voice-dir voices/ --levels 200 --dit none --gpu-memory-utilization 0.8
```

- **Dialects:** a voice whose file name starts with a dialect (`Jordanian…`, `Saudi…`, `Hindi…`) speaks in it.
- **Run:** per level, one warm-up burst, then 3 measured bursts.
- **Conditions:** run it on an otherwise idle GPU. Results vary by a few ms between runs.

License: MIT. VoxCPM2 weights: Apache-2.0.
