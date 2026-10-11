# VoxCPM2 Arabic Dialects: low-latency inference

Inference engine for [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) with 11 Arabic dialect LoRA adapters
([`VoxCPM-Kernel/VoxCPM2-Arabic-Dialects`](https://huggingface.co/VoxCPM-Kernel/VoxCPM2-Arabic-Dialects)) and voice
cloning. Every adapter is resident in one engine, a batch mixes dialects and voices, audio streams as it is generated.
One NVIDIA A10 answers **200 simultaneous requests with p95 time to first audio below 100 ms**.

## Time to first audio, 200 simultaneous requests

200 different Arabic dialect sentences submitted at the same instant, request i on adapter i % 11 (and voice i % 5
when cloning), NVIDIA A10 (24 GB, 150 W):

| | p95 TTFA |
|---|---:|
| base inference (`dq_engine`, bf16, 6 CFG steps) | 2174 ms |
| this engine, no cloning | **95.7-96.9 ms** (10 bursts, median 96.5) |
| this engine, voice cloning, `clone_mode="ref"` (5 Arabic reference voices) | **94.8-99.4 ms** (8 bursts, median 96.4) |
| this engine, voice cloning with the reference transcript, `clone_mode="prompt_ref"` | 445-483 ms |

`examples/burst.py` reproduces the measurement (with cloning: 93.9-98.1 ms over 6 bursts). In `ref` mode a voice's
reference KV is cached per voice and dialect and a request prefills only its own text; in `prompt_ref` mode the
reference audio follows the request's text and is prefilled for every request. With 200 users in conversation (each
speaking ~46 % of the time, no cloning) TTFA is p50 50 ms, p95 136 ms.

## Quality

Whisper large-v3 on the 200 outputs of a burst (Arabic-normalized text), mean over repeated runs:

| | CER | WER | DNSMOS | UTMOS | speaker similarity |
|---|---:|---:|---:|---:|---:|
| base inference, no cloning (4 runs) | 6.3 % | | 2.84 | 2.67 | |
| this engine, no cloning (8 runs) | 6.9 % | | 2.79 | 2.58 | |
| this engine, voice cloning `ref` (200 utterances) | 6.7 % | 16.4 % | 2.84 | 2.57 | 0.912 |

50 voice-cloned samples with per-utterance transcripts, WER and CER:
[`VoxCPM-Kernel/VoxCPM2-Arabic-Dialects-Samples`](https://huggingface.co/datasets/VoxCPM-Kernel/VoxCPM2-Arabic-Dialects-Samples).
Most of the low-latency configuration's CER cost (~0.6 point) comes from head start (below): the first word or two
are generated before the rest of the sentence is read.

## Install

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[server]"        # torch, flash-attn, Triton; the CUDA kernels compile on first use
```

Weights: the model (VoxCPM2 + `adapters/<dialect>/`) and the distilled 2-step LocDiT
(`kadirnar/VoxCPM2-Turbo-distilled-DiT`, file `student_k2_dmd.pt`). The examples download both from the Hugging Face
Hub (log in with `hf auth login`) or take local paths.

## Generate

```bash
python examples/arabic_dialects.py --dialect najdi --text "هلا والله، كيف حالك اليوم؟ إن شاء الله تمام." \
    --voice reference.wav --out out.wav                 # --voice-text "<transcript>": prompt + reference cloning
```

Dialects: `ab`, `egyptian`, `emirati`, `hejazi`, `hindi_english`, `iraqi`, `jordanian`, `najdi`, `palestinian`,
`qatari`, `saudi`. From Python (in an async function, run from the repository root):

```python
from examples.arabic_dialects import start_engine, register_voice

server, dialects = await start_engine("VoxCPM-Kernel/VoxCPM2-Arabic-Dialects",
                                      "kadirnar/VoxCPM2-Turbo-distilled-DiT:student_k2_dmd.pt", max_num_seqs=256)
await register_voice(server, "speaker1", "reference.wav")
async for chunk in server.generate(target_text="...", lora_name="saudi", voice="speaker1", clone_mode="ref"):
    ...                                                 # int16 PCM chunks, 48 kHz
```

200 simultaneous requests:

```bash
NANOVLLM_BUSY_POLL_S=30 python examples/burst.py --n 200 --voice-dir <dir of reference .wav files>
```

## Serve (OpenAI-compatible HTTP)

```bash
export NANOVLLM_VOXCPM2_TURBO='{"dit_checkpoint": "<student_k2_dmd.pt>", "w8a8": true,
  "w8a8_targets": ["base_lm", "residual_lm", "feat_encoder", "feat_decoder"], "head_tokens": 8, "head_patches": 3}'
NANOVLLM_BUSY_POLL_S=600 python -m nanovllm_voxcpm.entrypoints.openai_server \
    --model <model snapshot> --lora-dir <model snapshot>/adapters --voice-dir voices/ --port 8092

curl -F name=speaker1 -F audio_sample=@reference.wav localhost:8092/v1/audio/voices
curl localhost:8092/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input": "المكيف في الغرفة ما يبرد زين.", "dialect": "saudi", "voice": "speaker1"}' > out.pcm
```

The response streams 16-bit mono PCM at 48 kHz (`"response_format": "wav"` for a WAV file). Request fields:
`input`, `dialect` (or `lora` / `model`), `voice`, `clone_mode` (`ref` | `prompt_ref` | `prompt`; `prompt_ref` when the
voice was registered with `ref_text`), `seed`, `cfg_value`, `max_tokens`. `GET /v1/models` lists the dialects,
`GET /v1/audio/voices` the voices; `--voice-dir` keeps registered voices across restarts.

## Settings

| | |
|---|---|
| `NANOVLLM_VOXCPM2_TURBO` (JSON) | `dit_checkpoint`: the distilled LocDiT; `w8a8` / `w8a8_targets`: int8 layers; `head_tokens` / `head_patches`: head start (0 = off); `pcm16_output`: int16 chunks |
| `NANOVLLM_BUSY_POLL_S=s` | the engine and the client keep polling for s seconds after the last request instead of sleeping (one CPU core each): on hosts whose CPU governor slows idle cores, a burst after a quiet period otherwise spends ~10 ms more on the host |
| `NANOVLLM_SERVERPOOL_NUM_KVCACHE_BLOCKS` | KV-cache blocks (the examples use 600, as measured) |
| `VOXCPM_KERNELS_SMOOTH` | SmoothQuant calibration of the int8 layers (default: `voxcpm_kernels/smooth_stats.pt`; `0` = off) |

## How it gets there

- **Head start**: a request first generates 3 patches (480 ms of audio) from the first <= 8 tokens of its text, then
  continues with the whole text after them; the first audio waits only for that short prefill.
- **Adapters merged into int8 weights**: q/k/v/o of every adapter merged into per-adapter int8 weights, rows grouped by
  adapter and run by segmented GEMMs (`voxcpm_kernels/seg_gemm.py`), no LoRA kernels in the hot path.
- **Distilled 2-step LocDiT, W8A8 with SmoothQuant**, fused decoder layers (norm + quantization, RoPE and residual in
  GEMM epilogues), CUDA graphs.
- **Short-sequence attention** with the output quantized in the kernel (`voxcpm_kernels/short_attn.py`).
- **First audio = the first VAE frame** (40 ms), decoded from a CUDA graph before the rest of the patch.
- **Host path**: requests registered in chunks while the client is still sending, the engine's output read in the
  client's event loop, audio through shared memory, garbage collection only while no request waits.

## Layout

| | |
|---|---|
| `nanovllm_voxcpm/` | the engine (scheduler, model runner, VoxCPM2 model and LoRA, streaming server process, HTTP server) |
| `voxcpm_kernels/` | Triton / CUDA kernels: int8 GEMMs, segmented adapter GEMMs, attention, streaming VAE |
| `examples/` | generation with a dialect and a cloned voice; the 200-request burst |

## License

MIT (see `LICENSE`), based on Nano-vLLM-VoxCPM and Nano-vLLM. The VoxCPM2 weights are Apache-2.0.
