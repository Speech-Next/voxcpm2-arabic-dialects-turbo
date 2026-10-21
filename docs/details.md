# Details

The [README](../README.md) has the results tables and the usage; here is how they were measured, every option,
production and Docker.

## Benchmark

The README's WebSocket table: one NVIDIA A10 (24 GB), default settings, the WebSocket server; the scripts and `text.txt` of
[`VoxCPM-Kernel/benchmark-test-data-ar`](https://huggingface.co/datasets/VoxCPM-Kernel/benchmark-test-data-ar),
unchanged; four runs (three of the underruns script, one of the original), a fresh server each, the range over the
runs, without the audio cache (`--audio-cache-mb 0`). The script prints no p99 for TTFB; its max is the upper bound.
Earlier results (with the 150 ms first-chunk hold) and their check against the server's own timestamps:
[`results/`](https://huggingface.co/datasets/VoxCPM-Kernel/benchmark-test-data-ar/tree/main/results).

- TTFA: time to first audio, from the text arriving at the server to the engine's first audio (server side).
- TTFB: time to the first 4096-byte chunk (128 ms of audio) at the client, as the benchmark measures it. The server
  sends the first chunk as soon as it is complete. `--pace-start-ms 150` (the default until #79) holds it 150 ms:
  TTFB 150 ms later, and with the same stalls (0) and quality (Whisper CER / WER, UTMOS at 100 and 150 at once, #79).
- Stalls: gaps between two chunks longer than one chunk's audio. Underruns: a player starting at the first chunk
  runs dry for more than 50 ms.
- 150 requests at once is the limit of one A10: at 200 at once there were 24-84 stalls per run (no underruns; measured
  with the 150 ms hold).

**First audio of the engine** (`python examples/benchmark.py`): no network, every request submitted at the same
instant in one process, so a best case; the WebSocket server's TTFA (the README) is what a server sees. p99 of 3 runs:

| requests at once | 1 | 64 | 128 | 150 | 200 | 256 |
|---|---:|---:|---:|---:|---:|---:|
| first audio p99 | 24-36 ms | 39-41 ms | 64-71 ms | 75 ms | 94-99 ms | 121-133 ms |

**Quality** (the README's second table): 17 dialects x 40 sentences x 2 seeds (1,320 clips), each voice cloned from
another clip of its speaker. CER / WER: the better of Whisper large-v3 and Cohere Transcribe Arabic (lower is
better). UTMOS: predicted naturalness, 1-5.

## Install

Linux, an NVIDIA GPU (Ampere or newer) and the CUDA toolkit (`nvcc`); or the Docker image ([Docker](#docker)).

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

`setup.sh` builds the two CUDA kernels (nvcc, ~1.5 minutes); the build is cached per kernel source, one for every
checkout and worktree. Models download on first use, at the commits the numbers here were measured with
(`nanovllm_voxcpm/hub.py`): a new upload to a model repo changes nothing until it is pinned there. `repo@revision`
(e.g. `--model VoxCPM-Kernel/VoxCPM2-Arabic-Dialects@main`) takes another commit, branch or tag, and
`VOXCPM_HUB_LATEST=1` the main branch of every repo. The first start takes ~3 minutes (the Triton kernels compile,
CUDA graphs are captured), later starts ~1 minute.

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

`response_format`: `pcm` (the default: raw 16-bit, rate in `X-Sample-Rate`), `wav` (sent whole), and `mp3`
(constant bitrate, ~96 kbit/s), `opus` (Ogg) and `flac`, which are streamed as they are encoded (a streamed flac carries
no length). OpenAI's default is mp3. With `ffmpeg` on the server's PATH (the Docker image has it) also `aac` (ADTS,
streamed as it is encoded), and `speed` 0.25-4 for every format, by ffmpeg's `atempo`: the pitch stays, streamed too.
Through ffmpeg the first audio comes ~50 ms later (its process starts); speed 1.0 never runs it. Without ffmpeg, `aac`
and a speed other than 1.0 are a 400. `cfg_value` has no effect with the distilled LocDiT (the default), only with
`--dit none`.

## Production

| | WebSocket server | HTTP server |
|---|---|---|
| API key (`--api-key` or `VOXCPM_API_KEY`) | `Authorization: Bearer <key>` or `?api_key=<key>` | `Authorization: Bearer <key>` (all but `/health`) |
| clients (`--api-keys-file`): a key each, with its own `max_streams` and `rpm` | an utterance beyond them gets `{"type": "error", "message": "client <name>: .., try again in <s> s"}`, its connection goes on | beyond them: 429 with `Retry-After` |
| requests at once (`--max-streams`, one A10: 150) | more get `{"type": "error", "message": "busy: .."}` | more get 503 with `Retry-After` |
| health | `GET /health` on the same port | `GET /health`, 503 once the engine process died |
| input | `--max-text-chars` (4096), messages up to 1 MiB, an unknown dialect refused at the handshake | `--max-input-chars` (4096), `cfg_value` 0-10, voice clips up to `--max-voice-mb` (10) and `--max-voice-seconds` (30) |
| text with nothing to speak (punctuation, symbols or emoji only) | an error for that utterance | 400 |
| a client that leaves | its engine request ends (pings every `--ping-interval`, 20 s, find dead connections) | its engine request ends |
| the engine process dies, or hangs (sends nothing for `--hang-timeout`, 60 s, while requests wait) | the server exits with code 1 | the server exits with code 1 |
| SIGTERM | no new connection or utterance; those in progress get `--shutdown-timeout` (10 s) to end, then are cut; the connections close with 1001 | no new connection; requests in progress get `--shutdown-timeout` (10 s) to end, then are cut |
| `DELETE /v1/audio/voices/{name}` | | 409 while a speech request uses the voice; with `--api-keys-file`, 403 for another client's voice (an `admin=true` client may delete any) |
| logs (`--log-level`, `info`) | a line per connection opened and closed (code, reason) and per utterance not spoken; shutdown, engine death | a line per speech request refused or failed (status, why) and per voice registered or deleted; shutdown, engine death |
| metrics | `GET /metrics` on the same port, behind the API key (with `--api-keys-file`: a client with `metrics=true`) | `GET /metrics`, behind the API key (with `--api-keys-file`: a client with `metrics=true`) |
| audio cache (`--audio-cache-mb`, 512; `0`: off) | each sentence's first audio, by sentence, dialect, voice (its file's content) and `cfg_value` | each sentence's first audio, by sentence as read, voice (its latents and transcript), dialect, `seed`, `cfg_value` and `max_tokens` |
| audio cache on disk (`--audio-cache-dir`, `--audio-cache-dir-mb` 4096) | the kept sentences as files: they outlive restarts, replicas on a shared disk share them | the same |

Run a server under a supervisor that starts it again when it exits (systemd `Restart=always`, Docker
`--restart unless-stopped`, Kubernetes), and send traffic once `/health` answers (~75 s after start). A stop
(SIGTERM) takes at most `--shutdown-timeout` plus a few seconds, within Kubernetes' default 30 s grace period. The
engine process ends with its server, also when the server is killed (`kill -9`, the OOM killer). Both servers listen on
127.0.0.1 by default: put them behind a TLS proxy, or listen on 0.0.0.0 only with an API key. After the first
download, `HF_HUB_OFFLINE=1` keeps a restart on the models already downloaded. Voices registered over HTTP live in
that server's process, and in `--voice-dir` when it is given.

**More than 150 streams: one replica per GPU.** A server runs one engine on one GPU (one A10: 150 streams at once).
Run one server per GPU, each with its own GPU (`CUDA_VISIBLE_DEVICES=0`, `=1`, ...) and port, behind a load balancer
(nginx, HAProxy, a Kubernetes Service) that sends each request or WebSocket connection to any replica: a request
holds no state between replicas, so no sticky sessions are needed. For the HTTP server, give every replica the same
`--voice-dir` on a shared disk: a voice registered on one replica is spoken on any other at once (it registers the
voice on the first request that names it), and every `--voice-sync-s` seconds (10) each replica picks up the voices
the others registered and drops the ones they deleted; a voice in use is dropped once its requests end. The voice's
files are written whole, its `.json` last, so a replica never reads half a voice. The WebSocket server's voices are
its `--voice` options, the same on every replica. The audio cache is each replica's own, unless the replicas
share a cache directory.

`--api-keys-file` gives each client its own key and limits, in lines (or JSON, `nanovllm_voxcpm/entrypoints/clients.py`):

```
# name    key              limits and rights (optional)
acme      sk-acme-7f3a...  max_streams=20 rpm=600
beta      sk-beta-91c2...  rpm=60
monitor   sk-prom-0d4e...  metrics=true
ops       sk-ops-5b1f...   admin=true
```

- `max_streams`: the client's requests at once. `rpm`: its requests a minute, a token bucket of a minute's worth.
  Beyond either, its request is refused (HTTP 429 with `Retry-After`; WebSocket: an error for that utterance). The
  server's own `--max-streams` (503) comes first.
- `metrics=true`: the key reads `/metrics`, which holds every client's name and counts. `--api-key` (the client
  `default`) works next to the file and reads `/metrics`.
- The file is read again when it changes (looked at no more than once a second): replace it whole (write, then
  rename) to add, remove or rotate keys and change limits without a restart; an open WebSocket connection's key is
  looked up again at each utterance (a removed key speaks no more). A file that does not parse keeps the clients
  read before, and logs an error.
- Keys are compared in constant time and never logged. Logs and `voxcpm_client_requests_total{client,outcome}` carry
  the client's name.
- A voice registered over HTTP belongs to the client that registered it (its `owner`, kept in `--voice-dir` too, so
  every replica knows it). Every client may speak with it, unless it was uploaded with `private=true`: then only its
  owner and the admins may, and to every other client it does not exist (404, not in `GET /v1/audio/voices`). Only
  its owner and the admins may delete it (others: 403). `admin=true` makes a client an admin, and so is `--api-key`'s
  `default`; a voice with no owner (registered before `--api-keys-file`, or by `--voices`) is the admins' to delete.
  The listing shows a client its own voices' (an admin: every voice's) `owner` and `private`. Without
  `--api-keys-file` nothing changes: every voice is every caller's.

`/metrics` is Prometheus' text format; with an API key, scrape it with the key as a bearer token
(`authorization: {credentials: <key>}` in the scrape config). Both servers export:
- `voxcpm_requests_total{outcome}`: speech requests (HTTP) or utterances (WebSocket). The outcomes are `ok`,
  `client_error` (the request's fault), `server_error` (the engine failed), `busy` (`--max-streams`, shutting down),
  `limited` (over its client's `max_streams` or `rpm`) and `cancelled` (the client left);
- `voxcpm_active_streams`;
- `voxcpm_first_audio_seconds`, a histogram from a request's turn to the engine's first audio at the server;
- `voxcpm_audio_seconds_total`;
- `voxcpm_engine_alive`.

The WebSocket server also exports `voxcpm_connections`, `voxcpm_connections_total` and `voxcpm_first_chunk_seconds`
(to the first chunk sent, the pacer's `--pace-start-ms` hold included). Log lines are written by a thread of their
own, never by the event loop that paces the audio.

The audio cache gives a sentence the same audio every time it is asked for, at any load, and a repeated sentence
costs no GPU. A seed alone does not: a head start's split follows the queue, and the batch and the prefix cache change
the engine's float results. It keeps each sentence's first complete generation in the server's memory (cut or failed
ones are not kept) and drops the least recently used beyond `--audio-cache-mb`: 512 MB hold ~780 sentences of 7 s
(the engine's 48 kHz PCM16, 96 KB a second). A kept sentence plays its first second at once and the rest at 4 x real
time (a whole HTTP response at once). Requests for the same sentence that run at the same time share one generation:
a request that comes while it runs gets the audio made so far, then the rest as it comes; the generation goes on while
any of them still listens (its first client may leave) and ends when none does.

With `--audio-cache-dir` each kept sentence is also a file there (written whole, the files used longest ago removed
beyond `--audio-cache-dir-mb`): it outlives a restart, and replicas on a shared disk share it. A file is read only by a
server with the same model and settings (they are part of its name), and a file that does not read back whole is
removed and the sentence generated again. `voxcpm_audio_cache_hits_total` (of which
`voxcpm_audio_cache_disk_hits_total` from the directory), `voxcpm_audio_cache_misses_total` (sentences generated),
`voxcpm_audio_cache_joins_total` (requests that joined a generation in progress), `voxcpm_audio_cache_bytes` and
`voxcpm_audio_cache_entries` are in `/metrics`. Benchmark the engine with `--audio-cache-mb 0`: a benchmark that
repeats its sentences measures the cache.

## Docker

```bash
cp .env.example .env                 # HF_TOKEN (the gated models), VOXCPM_API_KEY, COMPOSE_PROFILES
docker compose build                 # torch, flash-attn, ffmpeg and the CUDA kernels for CUDA_ARCH (8.6: A10)
docker compose --profile http up -d  # the HTTP server on :8092; --profile ws: the WebSocket server on :8093
docker compose logs -f http          # "ready" ~3 min after the first start (models download, Triton compiles)
```

Without compose:

```bash
docker build -t voxcpm2-turbo --build-arg CUDA_ARCH=8.6 .
docker run -d --gpus '"device=0"' --shm-size=1g --stop-timeout 30 -p 8092:8092 -e HF_TOKEN -e VOXCPM_API_KEY \
  -v voxcpm-cache:/cache -v voxcpm-voices:/voices voxcpm2-turbo http --max-streams 150 --voice-dir /voices
```

- **The image:**
  - Based on CUDA 12.8 devel, the same steps as `setup.sh`: Python 3.12, torch 2.8.0, flash-attn, plus ffmpeg.
  - It runs as a non-root user under tini.
  - Its command is `http` or `ws`, followed by that server's options. It listens on `0.0.0.0:$PORT` (8092) unless `--host` or `--port` are given; set `PORT`, not `--port`, so that the HEALTHCHECK follows. Any other command runs as given, e.g. `bash`.
- **Volumes:**
  - `/cache`: `HF_HOME` with the models, plus `TRITON_CACHE_DIR` and `TORCHINDUCTOR_CACHE_DIR` with the compiled Triton kernels and `torch.compile` graphs. A restart downloads and compiles nothing, so later starts take ~1 minute.
  - `/voices`: the HTTP server's `--voice-dir`.
  - After the first download, `-e HF_HUB_OFFLINE=1` keeps a start off the Hub.
- **GPU:**
  - The CUDA kernels are built at `docker build` for `--build-arg CUDA_ARCH`, a list such as `8.6,8.9,9.0` (A10, L4 / RTX 4090, H100). There is no GPU at build time, so `voxcpm_kernels.cuda_ext` builds for `VOXCPM_CUDA_ARCH`; a start then loads its GPU's build without nvcc.
  - Another GPU builds its own at its first start, ~2 min (nvcc is in the image).
  - One GPU per container: an A10 holds one engine. `docker-compose.yml` has a commented second replica on a second GPU.
- **Shared memory:** the engine streams audio through 128 MiB of shared memory, so `--shm-size=1g`. Docker's default of 64 MiB is too small.
- **Stop:** `docker stop` waits 10 s by default. The servers drain for `--shutdown-timeout` (10 s) and then stop the engine, so give them 30 s (`--stop-timeout 30`, compose `stop_grace_period: 30s`).
- **Health:** the image's HEALTHCHECK polls `/health`, with a start period of 15 min for the first start's downloads.
- **Kubernetes:**
  - one GPU per pod (`nvidia.com/gpu: 1`);
  - `readinessProbe` and `livenessProbe` on `GET /health`, with a `startupProbe` long enough for the first start;
  - `terminationGracePeriodSeconds` ≥ 30;
  - `/dev/shm` as an `emptyDir` with `medium: Memory` (`sizeLimit: 1Gi`);
  - `/cache` on a PersistentVolumeClaim;
  - `HF_TOKEN` and `VOXCPM_API_KEY` from a Secret.

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
| a dialect's LoRA | the README's LoRA table | `--route DIALECT=ADAPTER` or `DIALECT=teacher` (both servers, `voice_clone.py`; Python: `routes=`); only the adapters in use are loaded |
| LocDiT (`dit=`, `--dit`) | distilled, 3 steps | `None` / `none`: the model's own, 10 steps (slower) |
| precision | int8 | `--precision bf16` (benchmark.py) |
| first chunk delay (ws_server) | `--pace-start-ms 0` | `150`: the first chunk 150 ms later, more time for the engine (the default until #79) |

## License

| | license |
|---|---|
| this code | [PolyForm Noncommercial 1.0.0](../LICENSE): free for noncommercial use; commercial use needs the author's permission |
| the code it builds on (Nano-vLLM-VoxCPM, Nano-vLLM; vLLM's LoRA kernels) | MIT; Apache-2.0: [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) |
| [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2), VoxCPM2-Arabic-Dialects (the adapters) | Apache-2.0 |
| VoxCPM2-Arabic-Teacher-v3, VoxCPM2-Arabic-Dialects-Distilled-DiT (both used by default) | CC BY-NC-SA 4.0: non-commercial |
