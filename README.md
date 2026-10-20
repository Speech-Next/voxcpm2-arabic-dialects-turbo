# VoxCPM2 Arabic Dialects (Turbo)

Fast Arabic dialect text-to-speech with voice cloning, built on [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2).
Each dialect is spoken by its own LoRA adapter or by
[teacher-v3](https://huggingface.co/VoxCPM-Kernel/VoxCPM2-Arabic-Teacher-v3), all in one engine.

## Results

One NVIDIA A10 (24 GB), default settings.

**WebSocket server**, measured with the benchmark of
[`VoxCPM-Kernel/benchmark-test-data-ar`](https://huggingface.co/datasets/VoxCPM-Kernel/benchmark-test-data-ar)
(its scripts and `text.txt`, unchanged):

WS_RESULTS_TABLE

- TTFA: time to first audio, from the text arriving at the server to the engine's first audio.
- TTFB: time to the first 4096-byte chunk (128 ms of audio) at the client, as the benchmark measures it. The server
  holds the first chunk 150 ms (`--pace-start-ms`); this buys the engine the time for no stalls.
- Stalls: gaps between two chunks longer than one chunk's audio. Underruns: a player starting at the first chunk
  runs dry for more than 50 ms.
- 150 requests at once is the limit of one A10: at 200 at once there were 24-84 stalls per run (no underruns).

**First audio of the engine** (`python examples/benchmark.py`, no network, p99 of 3 runs):

| requests at once | 1 | 64 | 128 | 150 | 200 | 256 |
|---|---:|---:|---:|---:|---:|---:|
| first audio p99 | 24-36 ms | 39-41 ms | 64-71 ms | 75 ms | 94-99 ms | 121-133 ms |

**Quality**, 17 dialects x 40 sentences x 2 seeds (1,320 clips), each voice cloned from another clip of its speaker:

| engine | CER | WER | UTMOS | dialect ID | GPU memory |
|---|---:|---:|---:|---:|---:|
| routed (default, `RoutedDialectTTS`) | 4.6 % | 13.9 % | 2.22 | 64 % | 21.2 GiB |
| adapters only (`DialectTTS`) | 5.3 % | 16.0 % | 2.23 | 67 % | 22.0 GiB |

CER / WER: the better of Whisper large-v3 and Cohere Transcribe Arabic (lower is better). UTMOS: predicted
naturalness, 1-5.

## Install

Linux, an NVIDIA GPU (Ampere or newer) and the CUDA toolkit (`nvcc`).

```bash
git clone https://github.com/Speech-Next/voxcpm2-arabic-dialects-turbo && cd voxcpm2-arabic-dialects-turbo
./setup.sh                  # Python 3.12 env, torch, flash-attn, this package with the servers
source .venv/bin/activate
hf auth login               # only for the gated benchmark voices
```

Models download on first use. The first start takes 1-2 minutes (kernels compile, CUDA graphs are captured).

## Python

```python
import asyncio
from nanovllm_voxcpm import RoutedDialectTTS

async def main():
    tts = await RoutedDialectTTS.load()
    await tts.add_voice("ahmed", "ahmed.wav")            # optional text="<transcript>": closer cloning
    audio = await tts.synthesize("هلا والله، كيف حالك اليوم؟", dialect="najdi", voice="ahmed")  # float32, 48 kHz
    async for chunk in tts.stream("أهلا وسهلا", dialect="egyptian", voice="ahmed"):            # streaming
        ...
    await tts.close()

asyncio.run(main())
```

Dialects (`tts.dialects`): msa, egyptian, palestinian, lebanese, syrian, iraqi, saudi, najdi, hejazi, gulf,
bahraini, yemeni, sudanese, libyan, tunisian, algerian, moroccan, hindi_english, and the aliases ab, emirati,
jordanian, khaliji, levantine, qatari.

## WebSocket server

```bash
python -m nanovllm_voxcpm.entrypoints.ws_server --voice default=reference.wav --dialect egyptian
# ws://127.0.0.1:8092/api/tts/stream?audio_format=pcm_16000&chunk_size=4096&segmenting=false&transport=bytes&dialect=egyptian
```

| message | |
|---|---|
| client: text | one utterance; several are spoken in order |
| client: `{"type": "control", "voice": .., "cfg_value": ..}` | optional, answered with `config_updated` |
| server: `{"type": "start", "sample_rate": 16000}` | then the 16-bit PCM in `chunk_size` binary messages |
| server: `{"type": "end_of_audio"}` | after each utterance |

`transport=json_base64` sends `{"type": "audio_chunk", "chunk_data": ..}` instead of binary messages;
`segmenting=true` speaks a long text sentence by sentence. The benchmark:

```bash
hf download VoxCPM-Kernel/benchmark-test-data-ar --repo-type dataset --local-dir bench
python bench/bench_voxcpm2_texts_underruns.py --levels 1,10,40,80,100,150 --dialect egyptian --texts-file bench/text.txt
```

## HTTP server (OpenAI-compatible)

```bash
python -m nanovllm_voxcpm.entrypoints.openai_server          # http://127.0.0.1:8092
curl -F name=ahmed -F audio_sample=@ahmed.wav http://127.0.0.1:8092/v1/audio/voices
curl http://127.0.0.1:8092/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input": "هلا والله", "voice": "ahmed", "dialect": "najdi", "response_format": "wav"}' -o out.wav
```

## Examples

```bash
python examples/voice_clone.py --voice ahmed.wav --dialect najdi --text "هلا والله" --out out.wav
python examples/benchmark.py                 # first audio per level (the table above)
python examples/streaming_benchmark.py       # requests arriving at random
```

## Options

| option | default | other |
|---|---|---|
| engine | `RoutedDialectTTS`: each dialect on its best model | `DialectTTS` / `--no-route`: the adapters only |
| LocDiT (`dit=`, `--dit`) | distilled, 3 steps | `None` / `none`: the model's own, 10 steps (slower) |
| precision | int8 | `--precision bf16` (benchmark.py) |
| first chunk delay (ws_server) | `--pace-start-ms 150` | `0`: TTFB p99 0.20-0.22 s at 150 at once, rare stalls |

License: MIT. VoxCPM2 weights: Apache-2.0.
