"""GPU check of stage KV reuse (TurboConfig.stage_kv_reuse, #31) on the real model (VoxCPM-Kernel/VoxCPM2-Arabic-
Dialects, the distilled LocDiT, int8). Not collected by pytest: run it as a script once the GPU is free; it exits 0 when
the checks pass (and skips without CUDA).

    PYTHONPATH=<worktree> .venv/bin/python <worktree>/tests/gpu/verify_stage_reuse.py

An in-process engine (head start on, no reuse) runs a few requests until their heads end; the KV their continuations
then compute for the prefix they share with their heads (the head's text) is compared with the heads' own KV for it,
which a continuation with reuse would copy. They agree to rounding, not bit for bit: prefill attention over a longer
query rounds differently, and a step of fewer than w8a8_min_rows rows (here the heads') runs the bf16 layers instead
of the int8 ones (--min-rows 0: int8 for both). Checks, per request:
  1 the shared prefix is the head's text (its ids), at the positions after the voice
  2 every layer's K and V agree to int8 rounding: cosine > 0.99 (a wrong block or position is far below; measured:
    0.9986-0.9993 with --min-rows 0, 0.9947-0.9963 with the heads' step in bf16); max |diff| is reported
"""

import argparse
import json
import os
import sys

__test__ = False


def main(args) -> int:
    import torch

    if not torch.cuda.is_available():
        print("[SKIP] no CUDA")
        return 0
    from nanovllm_voxcpm.dialect_tts import DEFAULT_DIT, DEFAULT_MODEL, resolve, turbo_config

    path = resolve(DEFAULT_MODEL)
    turbo = {**turbo_config(resolve(DEFAULT_DIT), head_tokens=8), "head_hold": False, "stage_kv_reuse": False,
             "mid_tokens": 0}
    if args.min_rows is not None:  # 0: int8 layers for every step size (the head's small prefill as well)
        turbo["w8a8_min_rows"] = args.min_rows
    os.environ["NANOVLLM_VOXCPM2_TURBO"] = json.dumps(turbo)
    os.environ.setdefault("NANOVLLM_SERVERPOOL_NUM_KVCACHE_BLOCKS", "64")
    from nanovllm_voxcpm.models.voxcpm2.server import VoxCPM2ServerImpl

    srv = VoxCPM2ServerImpl(path, max_num_batched_tokens=4096, max_num_seqs=16, max_model_len=1024, devices=[0])
    eng = srv.llm
    kv = eng.model_runner.kv_cache  # [layers, 2, blocks, block_size, heads, dim]
    texts = [s.strip() for s in open(os.path.join(os.path.dirname(__file__), "..", "..", "examples",
                                                  "ar_dialects.txt"), encoding="utf-8") if s.strip()]
    texts = [t for t in texts if len(t.split()) >= 8][: args.n]
    heads = {}
    for i, t in enumerate(texts):
        heads[f"r{i}"] = eng.add_request(f"r{i}", t, seed=100 + i, max_generate_length=200)
    snap = {}
    for _ in range(20):  # until every head has ended: snapshot its first block's KV right after its last step
        blocks = {sid: (h.block_table[0] if h.block_table else None) for sid, h in heads.items()}
        seqs = eng.step()
        for s in seqs:
            if s.seq_id in heads and s is heads[s.seq_id] and s.stoped and s.seq_id not in snap:
                n = min(s.num_prompt_tokens, eng.block_size)
                snap[s.seq_id] = (kv[:, :, blocks[s.seq_id], :n].clone(), list(s.token_ids[:n]))
        if len(snap) == len(heads):
            break
    fails = []
    conts = {s.seq_id: s for s in eng.scheduler.waiting if s.seq_id in heads}
    if set(conts) != set(heads):
        print(f"[FAIL] continuations queued for {sorted(conts)} of {sorted(heads)}")
        return 1
    eng.step()  # the continuations' prefill: they compute the shared prefix themselves (no reuse)
    for sid, c in conts.items():
        head_kv, head_ids = snap[sid]
        k = 0
        while k < len(head_ids) and k < len(c.token_ids) and c.token_ids[k] == head_ids[k]:
            k += 1
        text = len(heads[sid].custom_payload.chain["text_ids"])
        ok1 = 0 < k < len(head_ids)
        print(f"[{'PASS' if ok1 else 'FAIL'}] 1 {sid}: shares {k} of the head's {len(head_ids)} prompt positions "
              f"(its whole text: {text} ids)")
        if not ok1:
            fails.append("prefix")
            continue
        mine = kv[:, :, c.block_table[0], :k].float()
        theirs = head_kv[:, :, :k].float()
        worst_cos, worst_rel = 1.0, 0.0
        for layer in range(mine.shape[0]):
            for j in range(2):
                a, b = mine[layer, j].ravel(), theirs[layer, j].ravel()
                cos = float(a @ b / (a.norm() * b.norm() + 1e-12))
                rel = float((a - b).abs().max() / (b.abs().max() + 1e-12))
                worst_cos, worst_rel = min(worst_cos, cos), max(worst_rel, rel)
        ok2 = worst_cos > 0.99
        print(f"[{'PASS' if ok2 else 'FAIL'}] 2 {sid}: K/V over the shared prefix, worst layer cosine {worst_cos:.6f}, "
              f"max |diff| {worst_rel * 100:.2f} % of the layer's max")
        if not ok2:
            fails.append("kv")
    print("FAILED: " + ", ".join(sorted(set(fails))) if fails else "all checks passed")
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--min-rows", type=int, help="TurboConfig.w8a8_min_rows (default: turbo_config's, 128)")
    sys.exit(main(ap.parse_args()))
