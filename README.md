# VoxCPM2 Arabic Dialects (Turbo)

Fast Arabic dialect text-to-speech with voice cloning, built on [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2).
Each dialect is spoken by its own LoRA adapter or by
[teacher-v3](https://huggingface.co/VoxCPM-Kernel/VoxCPM2-Arabic-Teacher-v3), all in one engine.

## Results

One NVIDIA A10 (24 GB), default settings.

**WebSocket server**, measured with the benchmark of
[`VoxCPM-Kernel/benchmark-test-data-ar`](https://huggingface.co/datasets/VoxCPM-Kernel/benchmark-test-data-ar)
(its scripts and `text.txt`, unchanged):

| requests at once | TTFA p99 | TTFB p95 | TTFB max | stalls | underruns |
|---:|---:|---:|---:|---:|---:|
| 1 | 24-29 ms | 178-185 ms | 178-185 ms | 0 | 0 |
| 10 | 29-36 ms | 187-195 ms | 187-196 ms | 0 | 0 |
| 40 | 58-70 ms | 219-239 ms | 219-245 ms | 0 | 0 |
| 80 | 69-73 ms | 241-256 ms | 241-256 ms | 0 | 0 |
| 100 | 76-81 ms | 257-270 ms | 257-270 ms | 0 | 0 |
| 150 | 94-148 ms | 275-330 ms | 288-330 ms | 0 | 0 |

Three runs (both scripts of the dataset), the range over the runs. The script prints no p99 for TTFB; its max is the
upper bound. Results and the check against the server's own timestamps:
[`results/`](https://huggingface.co/datasets/VoxCPM-Kernel/benchmark-test-data-ar/tree/main/results).

- TTFA: time to first audio, from the text arriving at the server to the engine's first audio (server side).
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
./setup.sh                  # Python 3.12 env, torch, flash-attn, this package with the servers, its CUDA kernels
source .venv/bin/activate
hf auth login               # the models are gated: request access first (below)
```

The models are gated on Hugging Face, and access is approved by hand: request it on the pages of
[VoxCPM2-Arabic-Dialects](https://huggingface.co/VoxCPM-Kernel/VoxCPM2-Arabic-Dialects),
[VoxCPM2-Arabic-Teacher-v3](https://huggingface.co/VoxCPM-Kernel/VoxCPM2-Arabic-Teacher-v3) and
[VoxCPM2-Arabic-Dialects-Distilled-DiT](https://huggingface.co/VoxCPM-Kernel/VoxCPM2-Arabic-Dialects-Distilled-DiT)
(and of [Arabic-Hindi-Reference-Voices](https://huggingface.co/datasets/VoxCPM-Kernel/Arabic-Hindi-Reference-Voices)
for the examples' voices).

`setup.sh` builds the two CUDA kernels (nvcc, ~1.5 minutes). Models download on first use. The first start takes
~3 minutes (the Triton kernels compile, CUDA graphs are captured), later starts ~1 minute.

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

if __name__ == "__main__":   # needed: the engine runs in a process of its own
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

Before it listens, the server synthesizes bursts of 1 to 150 requests with the default voice, so that every kernel
variant is compiled before the first client (no stalls on a fresh machine): it is ready after ~75 s, ~5.5 minutes
the first time.

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

## Production

| | WebSocket server | HTTP server |
|---|---|---|
| API key (`--api-key` or `VOXCPM_API_KEY`) | `Authorization: Bearer <key>` or `?api_key=<key>` | `Authorization: Bearer <key>` (all but `/health`) |
| requests at once (`--max-streams`, one A10: 150) | more get `{"type": "error", "message": "busy: .."}` | more get 503 with `Retry-After` |
| health | `GET /health` on the same port | `GET /health`, 503 once the engine process died |
| input | `--max-text-chars` (4096), messages up to 1 MiB, an unknown dialect refused at the handshake | `--max-input-chars` (4096), `cfg_value` 0-10, voice clips up to `--max-voice-mb` (10) and `--max-voice-seconds` (30) |
| text with nothing to speak (punctuation, symbols or emoji only) | an error for that utterance | 400 |
| a client that leaves | its engine request ends (pings every `--ping-interval`, 20 s, find dead connections) | its engine request ends |
| the engine process dies, or hangs (sends nothing for `--hang-timeout`, 60 s, while requests wait) | the server exits with code 1 | the server exits with code 1 |
| SIGTERM | no new connection or utterance; those in progress get `--shutdown-timeout` (10 s) to end, then are cut; the connections close with 1001 | no new connection; requests in progress get `--shutdown-timeout` (10 s) to end, then are cut |
| `DELETE /v1/audio/voices/{name}` | | 409 while a speech request uses the voice |

Run a server under a supervisor that starts it again when it exits (systemd `Restart=always`, Docker
`--restart unless-stopped`, Kubernetes), and send traffic once `/health` answers (~75 s after start). A stop
(SIGTERM) takes at most `--shutdown-timeout` plus a few seconds, within Kubernetes' default 30 s grace period. The
engine process ends with its server, also when the server is killed (`kill -9`, the OOM killer). Both servers listen on
127.0.0.1 by default: put them behind a TLS proxy, or listen on 0.0.0.0 only with an API key. After the first
download, `HF_HUB_OFFLINE=1` keeps a restart on the models already downloaded. Voices registered over HTTP live in
that server's process; several replicas share them with `--voice-dir` on a shared disk (read at startup).

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

## License

| | license |
|---|---|
| this code | [PolyForm Noncommercial 1.0.0](LICENSE): free for noncommercial use; commercial use needs the author's permission |
| the code it builds on (Nano-vLLM-VoxCPM, Nano-vLLM; vLLM's LoRA kernels) | MIT; Apache-2.0: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) |
| [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2), VoxCPM2-Arabic-Dialects (the adapters) | Apache-2.0 |
| VoxCPM2-Arabic-Teacher-v3, VoxCPM2-Arabic-Dialects-Distilled-DiT (both used by default) | CC BY-NC-SA 4.0: non-commercial |
