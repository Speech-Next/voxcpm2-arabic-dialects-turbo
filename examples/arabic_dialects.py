"""Arabic dialect speech, optionally cloning a voice: one engine holds every dialect adapter.

    python examples/arabic_dialects.py --dialect najdi --text "هلا والله، كيف حالك اليوم؟" \
        --voice reference.wav --out out.wav

--model: a local snapshot or the Hugging Face repo id of VoxCPM2 + dialect adapters (adapters/<name>/);
--dit: the distilled 2-step LocDiT checkpoint (a local file, or repo:file on the Hugging Face Hub).
--voice: a reference clip whose timbre is cloned; with --voice-text (its transcript) the reference is also continued.
"""

import argparse
import asyncio
import json
import os

import numpy as np
import soundfile as sf

TURBO = {  # the measured low-latency configuration
    "w8a8": True,
    "w8a8_targets": ["base_lm", "residual_lm", "feat_encoder", "feat_decoder"],
    "pcm16_output": True,
    "head_tokens": 8,
    "head_patches": 3,
}


def local_path(spec: str) -> str:
    """A local path, or repo id (directory snapshot) / repo:file (one file) on the Hugging Face Hub."""
    if os.path.exists(spec):
        return spec
    from huggingface_hub import hf_hub_download, snapshot_download

    if ":" in spec:
        repo, filename = spec.split(":", 1)
        return hf_hub_download(repo, filename)
    return snapshot_download(spec)


async def start_engine(model: str, dit: str, max_num_seqs: int):
    """The engine with the turbo configuration and every adapter of ``model``/adapters registered."""
    os.environ.setdefault("NANOVLLM_VOXCPM2_TURBO", json.dumps({**TURBO, "dit_checkpoint": local_path(dit)}))
    os.environ.setdefault("NANOVLLM_SERVERPOOL_NUM_KVCACHE_BLOCKS", "600")  # KV cache next to the adapters' weights
    from nanovllm_voxcpm import VoxCPM
    from nanovllm_voxcpm.models.voxcpm2.config import LoRAConfig

    path = local_path(model)
    adapters = sorted(d for d in os.listdir(os.path.join(path, "adapters"))
                      if os.path.exists(os.path.join(path, "adapters", d, "lora_weights.safetensors")))
    server = VoxCPM.from_pretrained(
        model=path, devices=[0], max_num_seqs=max_num_seqs, max_model_len=1024, max_num_batched_tokens=4096,
        gpu_memory_utilization=0.85, lora_config=LoRAConfig(max_loras=len(adapters), max_lora_rank=64),
    )
    await server.wait_for_ready()
    for name in adapters:
        await server.register_lora(name, os.path.join(path, "adapters", name))
    return server, adapters


async def register_voice(server, name: str, wav_path: str, text: str | None = None) -> None:
    with open(wav_path, "rb") as f:
        latents = await server.encode_latents(f.read(), "wav")
    await server.register_voice(name, latents, text)


async def main(args):
    server, adapters = await start_engine(args.model, args.dit, max_num_seqs=16)
    try:
        if args.dialect not in adapters:
            raise SystemExit(f"unknown dialect {args.dialect!r}; adapters: {', '.join(adapters)}")
        kw = {}
        if args.voice:
            await register_voice(server, "voice", args.voice, args.voice_text)
            kw = {"voice": "voice", "clone_mode": "prompt_ref" if args.voice_text else "ref"}
        sample_rate = int((await server.get_model_info())["sample_rate"])
        chunks = [c async for c in server.generate(target_text=args.text, lora_name=args.dialect, cfg_value=2.0,
                                                   max_generate_length=600, **kw)]
        audio = np.concatenate(chunks)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        sf.write(args.out, audio, sample_rate)
        print(f"{args.out}: {len(audio) / sample_rate:.2f} s, {args.dialect}" + (f", voice of {args.voice}" if args.voice else ""))
    finally:
        await server.stop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="VoxCPM-Kernel/VoxCPM2-Arabic-Dialects")
    ap.add_argument("--dit", default="kadirnar/VoxCPM2-Turbo-distilled-DiT:student_k2_dmd.pt")
    ap.add_argument("--dialect", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--voice", help="reference clip (wav) to clone")
    ap.add_argument("--voice-text", help="the reference clip's transcript (prompt + reference cloning)")
    ap.add_argument("--out", default="out.wav")
    asyncio.run(main(ap.parse_args()))
