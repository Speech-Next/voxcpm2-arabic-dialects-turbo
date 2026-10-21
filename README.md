# VoxCPM2 Arabic Dialects (Turbo)

Fast Arabic dialect text-to-speech with voice cloning, built on [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2).

## Results

| requests at once | TTFA p99 | TTFB p95 | TTFB max | stalls | underruns |
|---:|---:|---:|---:|---:|---:|
| 1 | 27-29 ms | 30-33 ms | 30-33 ms | 0 | 0 |
| 10 | 36-71 ms | 45-76 ms | 45-76 ms | 0 | 0 |
| 40 | 58-81 ms | 73-97 ms | 74-108 ms | 0 | 0 |
| 80 | 85-104 ms | 111-155 ms | 112-156 ms | 0 | 0 |
| 100 | 99-126 ms | 119-180 ms | 119-181 ms | 0 | 0 |
| 150 | 131-160 ms | 165-228 ms | 166-253 ms | 0 | 0 |

TTFA: the text at the server to the engine's first audio. TTFB: the text sent to the first 128 ms of audio at the
client.

| engine | CER | WER | UTMOS | dialect ID | GPU memory |
|---|---:|---:|---:|---:|---:|
| routed (default, `RoutedDialectTTS`) | 4.6 % | 13.9 % | 2.22 | 64 % | 21.2 GiB |
| adapters only (`DialectTTS`) | 5.3 % | 16.0 % | 2.23 | 67 % | 22.0 GiB |

17 dialects x 40 sentences x 2 seeds. CER / WER: lower is better; UTMOS: naturalness, 1-5.

## LoRA

Each dialect is spoken by one LoRA: an adapter of
[VoxCPM2-Arabic-Dialects](https://huggingface.co/VoxCPM-Kernel/VoxCPM2-Arabic-Dialects), or
[teacher-v3](https://huggingface.co/VoxCPM-Kernel/VoxCPM2-Arabic-Teacher-v3) (one LoRA for every dialect, with the
dialect's tag). The default is the better of the two on the quality set:

| dialect | default LoRA |
|---|---|
| egyptian, saudi, hejazi, hindi_english | its adapter |
| bahraini | the qatari adapter |
| msa, palestinian, lebanese, syrian, iraqi, najdi, gulf, yemeni, sudanese, libyan, tunisian, algerian, moroccan | teacher-v3 |
| emirati, qatari, khaliji (as gulf); jordanian, ab, levantine (as palestinian) | teacher-v3 |

The adapters: ab, egyptian, emirati, hejazi, hindi_english, iraqi, jordanian, najdi, palestinian, qatari, saudi.

## Usage

```bash
git clone https://github.com/Speech-Next/voxcpm2-arabic-dialects-turbo && cd voxcpm2-arabic-dialects-turbo
./setup.sh && source .venv/bin/activate
hf auth login   # the models are gated on Hugging Face: request access first
```

```python
import asyncio
from nanovllm_voxcpm import RoutedDialectTTS

async def main():
    tts = await RoutedDialectTTS.load()
    await tts.add_voice("ahmed", "ahmed.wav")
    audio = await tts.synthesize("هلا والله، كيف حالك اليوم؟", dialect="najdi", voice="ahmed")  # float32, 48 kHz
    await tts.close()

if __name__ == "__main__":
    asyncio.run(main())
```

```bash
# WebSocket server: ws://127.0.0.1:8092/api/tts/stream?dialect=egyptian
python -m nanovllm_voxcpm.entrypoints.ws_server --voice default=ahmed.wav --dialect egyptian

# HTTP server (OpenAI-compatible): http://127.0.0.1:8092
python -m nanovllm_voxcpm.entrypoints.openai_server
curl -F name=ahmed -F audio_sample=@ahmed.wav http://127.0.0.1:8092/v1/audio/voices
curl http://127.0.0.1:8092/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input": "هلا والله", "voice": "ahmed", "dialect": "najdi", "response_format": "wav"}' -o out.wav

# Docker (HF_TOKEN in .env)
cp .env.example .env && docker compose --profile http up -d

# Choose a dialect's LoRA (both servers): an adapter (DIALECT=ADAPTER) or teacher-v3 (DIALECT=teacher)
python -m nanovllm_voxcpm.entrypoints.ws_server --voice default=ahmed.wav --route najdi=najdi --route egyptian=teacher
# The adapters only, no teacher-v3
python -m nanovllm_voxcpm.entrypoints.openai_server --no-route
```

In Python: `RoutedDialectTTS.load(routes={**DEFAULT_ROUTES, "najdi": "najdi", "egyptian": "teacher"})`
(`DEFAULT_ROUTES` from `nanovllm_voxcpm.dialect_routing`), or `DialectTTS.load()` for the adapters only. Only the
adapters in use are loaded; all 11 with teacher-v3 need `--kv-blocks 500` on an A10.

Dialects: msa, egyptian, palestinian, lebanese, syrian, iraqi, saudi, najdi, hejazi, gulf, bahraini, yemeni, sudanese,
libyan, tunisian, algerian, moroccan, hindi_english. Every option: `--help`, and [docs/details.md](docs/details.md).
