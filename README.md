# VoxCPM2 Arabic Dialects

Fast voice-cloning text-to-speech for 11 Arabic dialects, built on [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2)
and [`VoxCPM-Kernel/VoxCPM2-Arabic-Dialects`](https://huggingface.co/VoxCPM-Kernel/VoxCPM2-Arabic-Dialects).

## Results

200 different Arabic dialect sentences sent at the same time, one NVIDIA A10:

| | TTFA, 1 request | TTFA p95, 200 at once | WER | CER | speaker similarity | UTMOS |
|---|---:|---:|---:|---:|---:|---:|
| base inference | 93 ms | 2174 ms | 16.0 % | 6.3 % | – | 2.67 |
| this library | 22 ms | 96 ms | 16.8 % | 6.9 % | – | 2.58 |
| base inference, voice cloning | 94 ms | 4464 ms | 17.1 % | 6.3 % | 0.95 | 2.96 |
| **this library, voice cloning** | **23 ms** | **96 ms** | **14.7 %** | **6.0 %** | **0.92** | **2.95** |
| this library, voice cloning with the reference transcript | 22 ms | 430 ms | 18.5 % | 7.7 % | 0.94 | 2.82 |

TTFA: time to first audio. WER / CER: Whisper large-v3 on the 200 outputs (rows without cloning: mean of 4-8 runs).
Speaker similarity: WavLM-SV to the reference clip; the cloning rows clone the same voice. Base inference:
`VoxCPM-Kernel/voxcpm-inference` (dq_engine, bf16, 6 CFG steps) on nano-vllm-voxcpm 2.0.4; with cloning it needed
smaller batches to fit 200 requests in memory.

## Install

```bash
pip install -e ".[server]"
```

## Voice cloning

```python
import asyncio
from nanovllm_voxcpm import DialectTTS

async def main():
    async with await DialectTTS.load() as tts:
        await tts.add_voice("speaker", "reference.wav")
        audio = await tts.synthesize("هلا والله، كيف حالك اليوم؟", dialect="najdi", voice="speaker")

if __name__ == "__main__":  # the engine runs in a spawned process
    asyncio.run(main())
```

`audio` is 48 kHz mono float32; `tts.stream(...)` yields it chunk by chunk as it is generated. Dialects: `ab`,
`egyptian`, `emirati`, `hejazi`, `hindi_english`, `iraqi`, `jordanian`, `najdi`, `palestinian`, `qatari`, `saudi`.

From the command line:

```bash
python examples/voice_clone.py --voice reference.wav --dialect najdi --text "هلا والله، كيف حالك اليوم؟"
```

## Server

```bash
python -m nanovllm_voxcpm.entrypoints.openai_server --dit kadirnar/VoxCPM2-Turbo-distilled-DiT:student_k2_dmd.pt

curl -F name=speaker -F audio_sample=@reference.wav localhost:8092/v1/audio/voices
curl localhost:8092/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input": "هلا والله، كيف حالك اليوم؟", "dialect": "najdi", "voice": "speaker", "response_format": "wav"}' > out.wav
```

Set `NANOVLLM_BUSY_POLL_S=600` to keep latency low after idle periods (uses one extra CPU core).

## Samples

50 voice-cloned samples with Whisper large-v3 WER / CER:
[`VoxCPM-Kernel/VoxCPM2-Arabic-Dialects-Samples`](https://huggingface.co/datasets/VoxCPM-Kernel/VoxCPM2-Arabic-Dialects-Samples).

## License

MIT, based on Nano-vLLM-VoxCPM and Nano-vLLM. VoxCPM2 weights: Apache-2.0.
