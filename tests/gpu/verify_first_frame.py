"""GPU checks of the first-frame split (#10) on the real AudioVAE decoder (VoxCPM-Kernel/VoxCPM2-Arabic-Dialects).
Not collected by pytest: run as a script; exits 0 when every check passes (skips without CUDA).

    PYTHONPATH=<worktree> .venv/bin/python <worktree>/tests/gpu/verify_first_frame.py

Streams decoded the usual way (the first frame, then frames 1-3, then a whole patch) are compared with streams fed the
same latents whose first frame goes through decode_first + decode_first_rest:
  1 the first part is bit-identical to the first samples of the whole frame
  2 the parts together, and the later chunks, match the usual decode to fp16 rounding (the causal states round at the
    split, as at every chunk boundary): SNR above 50 dB
  3 CUDA graphs and eager runs give the same audio; batches of 200 (a padded bucket) and 3
"""

import os
import sys

import torch

__test__ = False

MODEL = "VoxCPM-Kernel/VoxCPM2-Arabic-Dialects"
PHASES = 4  # the default NANOVLLM_FIRST_AUDIO_MS=20


def build(graphs: bool):
    import json

    from huggingface_hub import snapshot_download
    from torch.nn.utils import remove_weight_norm

    from nanovllm_voxcpm.layers.audio_vae_v2 import AudioVAEConfigV2, AudioVAEV2
    from voxcpm_kernels import SlotStreamingVAEDecoder

    path = snapshot_download(MODEL, allow_patterns=["config.json", "audiovae.pth"])
    cfg = json.load(open(os.path.join(path, "config.json")))["audio_vae_config"]
    torch.set_default_dtype(torch.float32)
    vae = AudioVAEV2(config=AudioVAEConfigV2(**cfg))
    vae.load_state_dict(torch.load(os.path.join(path, "audiovae.pth"))["state_dict"])
    for m in vae.decoder.modules():
        if hasattr(m, "weight_g"):
            remove_weight_norm(m)
    vae = vae.cuda()
    vae.decoder.to(dtype=torch.float16)
    dec = SlotStreamingVAEDecoder(vae.decoder, max_streams=512, latent_dim=64, chunk_frames=4, fused=True,
                                  extra_chunk_frames=(1, 3), first_phases=PHASES, use_cuda_graph=graphs)
    dec.warmup(64, 4)
    torch.set_default_dtype(torch.bfloat16)
    return dec


def snr(ref: torch.Tensor, x: torch.Tensor) -> float:
    ref, x = ref.float(), x.float()
    return float(10 * torch.log10(ref.pow(2).sum() / (x - ref).pow(2).sum().clamp_min(1e-30)))


def run(dec, B: int, tag: str, zs) -> dict:
    """The usual decode on streams ('u', i) and the split one on ('s', i), same latents."""
    usual, split = [(tag, "u", i) for i in range(B)], [(tag, "s", i) for i in range(B)]
    out = {"u0": dec.decode_chunks(zs[0][..., :1].contiguous(), usual).clone()}
    out["sa"] = dec.decode_first(zs[0][..., :1].contiguous(), split).clone()
    out["sb"] = dec.decode_first_rest(split).clone()
    for name, ids in (("u", usual), ("s", split)):
        out[name + "13"] = dec.decode_chunks(zs[0][..., 1:].contiguous(), ids).clone()
        out[name + "p1"] = dec.decode_chunks(zs[1], ids).clone()
    for sid in usual + split:
        dec.release(sid)
    return out


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA: skipped")
        return 0
    rows = []

    def check(name, ok, detail=""):
        rows.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}", flush=True)

    g = torch.Generator(device="cuda").manual_seed(10)
    batches = {B: [torch.randn(B, 64, 4, device="cuda", dtype=torch.float16, generator=g) for _ in range(2)]
               for B in (200, 3)}
    results = {}
    for graphs in (True, False):
        dec = build(graphs)
        check(f"split enabled (graphs={graphs})", dec.split_phases == PHASES, f"split_phases={dec.split_phases}")
        for B, zs in batches.items():
            r = results[(graphs, B)] = run(dec, B, f"g{graphs}", zs)
            n = r["sa"].size(-1)
            check(f"1 first part = the frame's first {n} samples (B={B}, graphs={graphs})",
                  torch.equal(r["sa"], r["u0"][..., :n]))
            whole = torch.cat([r["sa"], r["sb"]], -1)
            for name, a, b in (("first frame", r["u0"], whole), ("frames 1-3", r["u13"], r["s13"]),
                               ("next patch", r["up1"], r["sp1"])):
                s = snr(a, b)
                check(f"2 {name} matches the usual decode (B={B}, graphs={graphs})", s > 50, f"SNR {s:.1f} dB")
        del dec
        torch.cuda.empty_cache()
    for B in batches:
        same = all(torch.equal(results[(True, B)][k], results[(False, B)][k]) for k in results[(True, B)])
        check(f"3 graphs and eager give the same audio (B={B})", same)
    print(f"{sum(rows)} passed, {len(rows) - sum(rows)} failed")
    return 0 if all(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
