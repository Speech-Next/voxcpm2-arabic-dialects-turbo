# VoxCPM2 Arabic Dialects (Turbo)

Fast Arabic dialect text-to-speech with voice cloning, built on [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2). One
NVIDIA A10 answers 200 simultaneous requests with the first audio in under 100 ms.

## Results

Voice cloning on one NVIDIA A10. Each request is one Arabic dialect sentence of about 7 words, and all requests are
sent at the same instant.

| requests at once | first audio, average | first audio, p95 | whole sentence ready, average |
|---:|---:|---:|---:|
| 1 | 31 ms | 69 ms | 0.37 s |
| 16 | 28 ms | 28 ms | 0.65 s |
| 64 | 44 ms | 44 ms | 0.98 s |
| 128 | 67 ms | 68 ms | 1.45 s |
| 200 | 97 ms | 98 ms | 1.96 s |
| 256 | 123 ms | 124 ms | 2.43 s |

- The median at 1 request is 22 ms. Its p95 is the slowest of 10 measurements.
- A sentence is about 3.7 s of audio, so even at 256 requests it is generated faster than it plays.
- For comparison, base VoxCPM2 inference needs 2.2 s (4.5 s with voice cloning) for the first audio at 200 requests.

Two modes, chosen with `dit` (see Usage):

| mode | first audio p95, 200 requests | WER | CER | naturalness (UTMOS) |
|---|---:|---:|---:|---:|
| fast (default) | 98 ms | 17.5 % | 7.6 % | 2.49 |
| quality (`dit=None`) | 294 ms | 13.6 % | 5.5 % | 2.80 |

WER / CER: Whisper large-v3 over 473 outputs, 5 cloned voices.

## Models

| model | what | dialects |
|---|---|---|
| `VoxCPM-Kernel/VoxCPM2-Arabic-Dialects` (default) | VoxCPM2 + one LoRA adapter per dialect | ab, egyptian, emirati, hejazi, hindi_english, iraqi, jordanian, najdi, palestinian, qatari, saudi |
| `VoxCPM-Kernel/VoxCPM2-Arabic-Teacher-v2` | one model, the dialect chosen by a tag token | msa, egyptian, palestinian, lebanese, syrian, iraqi, saudi, najdi, hejazi, gulf, bahraini, yemeni, sudanese, libyan, tunisian, algerian, moroccan |
| `kadirnar/VoxCPM2-Turbo-distilled-DiT` | the 2-step distilled LocDiT of the fast mode | — |

Run `VoxCPM2-Arabic-Teacher-v2` in quality mode (`dit=None, w8a8=False`): the fast mode is not tuned for it yet.

## Requirements

- Linux with an NVIDIA GPU (tested: A10 24 GB, driver 570).
- Python 3.10-3.12 (tested 3.12).
- The CUDA toolkit with `nvcc` (tested 12.9) and `ninja`: one CUDA kernel is compiled the first time the engine
  starts.
- About 12 GB of disk for the models.
- Read access to the Hugging Face repos above (they are private).

Tested versions: torch 2.9.1 (CUDA 12.8 build), triton 3.5.1, flash-attn 2.8.3, transformers 5.19.

## Install

```bash
git clone https://github.com/Speech-Next/voxcpm2-arabic-dialects-turbo
cd voxcpm2-arabic-dialects-turbo

uv venv --python 3.12 && source .venv/bin/activate
uv pip install torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install -e ".[server]"      # builds flash-attn against the installed torch (several minutes)
```

With plain pip, install flash-attn first with `pip install flash-attn --no-build-isolation`, then
`pip install -e ".[server]"`.

Log in to Hugging Face once, with an account that can read the private repos:

```bash
hf auth login        # or: export HF_TOKEN=...
```

Models are downloaded on first use. To keep them in a local folder instead (then pass the folder as `model` / `dit`):

```bash
hf download VoxCPM-Kernel/VoxCPM2-Arabic-Dialects --local-dir models/VoxCPM2-Arabic-Dialects
hf download kadirnar/VoxCPM2-Turbo-distilled-DiT student_k2_dmd.pt --local-dir models/distilled-dit
```

## Usage

```python
import asyncio
from nanovllm_voxcpm import DialectTTS

async def main():
    async with await DialectTTS.load() as tts:               # fast mode, VoxCPM2-Arabic-Dialects
        await tts.add_voice("ahmed", "ahmed.wav")             # the voice to clone, once

        audio = await tts.synthesize("هلا والله، كيف حالك اليوم؟", dialect="najdi", voice="ahmed")

        async for chunk in tts.stream("إزيك عامل إيه؟", dialect="egyptian", voice="ahmed"):
            ...                                               # audio while it is generated

if __name__ == "__main__":      # required: the engine runs in a separate process
    asyncio.run(main())
```

- **Audio:** 48 kHz mono float32.
- **Dialect:** `dialect` picks the dialect. Requests in different dialects run together in the same batch.
  `dialect=None` speaks plain VoxCPM2.
- **Numbers:** digits, times, amounts and percentages are read as Arabic words; `normalize=False` turns this off.

Options of `DialectTTS.load`:

| option | default | |
|---|---|---|
| `model` | `VoxCPM-Kernel/VoxCPM2-Arabic-Dialects` | Hugging Face id or local folder |
| `dit` | the distilled LocDiT | `None`: the model's own LocDiT (quality mode) |
| `w8a8` | `True` | `False`: bf16 instead of int8 |
| `head_tokens` | `8` | `0`: the first audio waits for the whole text |
| `dit_cache` | `None` | with `dit=None`: `"light"` or `"medium"`, a faster quality mode |
| `max_concurrency` | `256` | most requests at once |

The tag model:

```python
tts = await DialectTTS.load("VoxCPM-Kernel/VoxCPM2-Arabic-Teacher-v2", dit=None, w8a8=False)
```

## Command line

```bash
python examples/voice_clone.py --voice ahmed.wav --dialect najdi --text "هلا والله، كيف حالك اليوم؟" --out out.wav
```

## Server

An OpenAI-style HTTP server on port 8092:

```bash
NANOVLLM_BUSY_POLL_S=600 python -m nanovllm_voxcpm.entrypoints.openai_server \
    --dit kadirnar/VoxCPM2-Turbo-distilled-DiT:student_k2_dmd.pt

curl -F name=ahmed -F audio_sample=@ahmed.wav localhost:8092/v1/audio/voices
curl localhost:8092/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input": "هلا والله، كيف حالك اليوم؟", "dialect": "najdi", "voice": "ahmed", "response_format": "wav"}' > out.wav
```

`NANOVLLM_BUSY_POLL_S=600` keeps the engine awake for 10 minutes after the last request, so latency stays low after
idle periods. It uses one extra CPU core.

## Benchmark

The results table comes from:

```bash
NANOVLLM_BUSY_POLL_S=30 python examples/benchmark.py --voice-dir voices/
```

- **`voices/`:** a folder of reference clips (`.wav`, 5-10 s each); the requests clone them in turn. The table above
  used 5 Arabic voices.
- **Run:** per level, one warm-up burst, then 3 measured bursts (10 at 1 request).
- **Output:** the table, in Markdown.
- **Options:**
  - `--levels 1,16,200` picks the levels.
  - `--dit none` measures the quality mode.
- **Conditions:** the GPU must be otherwise idle.

## Notes

- **First start:** compiling the CUDA kernel and capturing CUDA graphs takes 1-2 minutes. Later starts are faster
  (the kernel is cached in `~/.cache/torch_extensions`).
- **Memory:** the engine takes 85 % of the GPU memory by default (`gpu_memory_utilization`).
- **Script names:** do not name a script `concurrent.py`; it hides Python's own `concurrent` module.

## License

MIT, based on Nano-vLLM-VoxCPM and Nano-vLLM. VoxCPM2 weights: Apache-2.0.
