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
| this library, voice cloning, without the distilled LocDiT | 30 ms | 294 ms | 14.4 % | 6.1 % | 0.93 | 3.13 |
| the same + cache-dit light | 30 ms | 258 ms | 14.0 % | 5.6 % | 0.92 | 3.13 |
| the same + cache-dit medium | 27 ms | 210 ms | 16.8 % | 7.6 % | 0.92 | 3.09 |
| this library, voice cloning with the reference transcript | 22 ms | 430 ms | 18.5 % | 7.7 % | 0.94 | 2.82 |

TTFA: time to first audio. WER / CER: Whisper large-v3 on the 200 outputs (rows without cloning: mean of 4-8 runs).
Speaker similarity: WavLM-SV to the reference clip; the cloning rows clone the same voice. Base inference:
`VoxCPM-Kernel/voxcpm-inference` (dq_engine, bf16, 6 CFG steps) on nano-vllm-voxcpm 2.0.4; with cloning it needed
smaller batches to fit 200 requests in memory.

Time to first audio (p95) with voice cloning, by the number of requests sent at the same time:

| simultaneous requests | 1 | 8 | 16 | 64 | 128 | 256 |
|---|---:|---:|---:|---:|---:|---:|
| base inference | 93 ms | 188 ms | 364 ms | 1455 ms | 2838 ms | 5708 ms |
| **this library** | **21 ms** | **25 ms** | **29 ms** | **43 ms** | **66 ms** | **122 ms** |
| without the distilled LocDiT, cache-dit medium | 27 ms | 38 ms | 53 ms | 87 ms | 140 ms | 254 ms |
| without the distilled LocDiT, cache-dit light | 30 ms | 44 ms | 62 ms | 103 ms | 166 ms | 297 ms |
| without the distilled LocDiT | 30 ms | 49 ms | 72 ms | 118 ms | 194 ms | 344 ms |

Quality of the same runs (all 473 outputs of the six bursts):

| | WER | CER | speaker similarity | UTMOS | mean stall, 128 / 256 requests |
|---|---:|---:|---:|---:|---:|
| this library | 17.5 % | 7.6 % | 0.91 | 2.49 | 0.0 / 0.3 s |
| without the distilled LocDiT, cache-dit medium | 13.9 % | 5.5 % | 0.93 | 2.79 | 0.1 / 1.8 s |
| without the distilled LocDiT, cache-dit light | 13.6 % | 5.3 % | 0.93 | 2.83 | 0.2 / 2.8 s |
| without the distilled LocDiT | 13.6 % | 5.5 % | 0.93 | 2.80 | 0.6 / 3.7 s |

Median over the measured rounds after a first warm-up round; this library clones 5 voices mixed, the base inference
one voice. Stall: time a stream waited for audio after it started playing. The distilled 2-step LocDiT (the default)
is the only configuration that keeps the first audio under 100 ms at 200 requests; it costs quality, more with some
voices than others (with the voice of the 200-request table its WER was within noise of the original LocDiT, over these
5 voices it is 4 points higher). Without it (`DialectTTS.load(dit=None)`: the original LocDiT, 10 steps with CFG),
cache-dit brings the first audio 14 % (light) or 26 % (medium) earlier and shortens the stalls; light keeps the
original's quality, medium cost 2 points of WER with one voice and nothing measurable over the 5.

### cache-dit

Block caching across the original LocDiT's solver steps, after [cache-dit](https://github.com/vipshop/cache-dit)
(DBCache + TaylorSeer): the first and last decoder layers run on every step; on the cached steps the middle layers are
skipped and their contribution is extrapolated from the last two computed steps. The plan is fixed (light: 2 of the 9
steps cached, medium: 4), the same for every request, so batches and CUDA graphs are unaffected. It does not combine
with the distilled LocDiT, which has no step to spare (WER 54 %).

```python
tts = await DialectTTS.load(dit=None, dit_cache="medium")
```

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
# or, without the distilled LocDiT: --dit-cache medium

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
