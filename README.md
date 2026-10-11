# VoxCPM2 Arabic Dialects

Fast voice-cloning text-to-speech for 11 Arabic dialects, built on [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2)
and [`VoxCPM-Kernel/VoxCPM2-Arabic-Dialects`](https://huggingface.co/VoxCPM-Kernel/VoxCPM2-Arabic-Dialects).

## Speed

200 requests sent at the same time, each a different sentence, dialect and cloned voice (one NVIDIA A10):

| | time to first audio (p95) |
|---|---:|
| base VoxCPM2 inference | 2174 ms |
| **this library, voice cloning** | **~96 ms** |

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
