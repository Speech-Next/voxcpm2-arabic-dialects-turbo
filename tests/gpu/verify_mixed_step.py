"""GPU check of mixed steps (TurboConfig.mix_decode) on the real model (VoxCPM-Kernel/VoxCPM2-Arabic-Dialects, the
distilled LocDiT, int8): a running sequence's next patch computed through the prefill path (as a mixed step decodes
it) against the decode path. Not collected by pytest: run it as a script once the GPU is free; it exits 0 when the
checks pass (and skips without CUDA).

    PYTHONPATH=<worktree> .venv/bin/python <worktree>/tests/gpu/verify_mixed_step.py

An in-process engine (no adapters) starts --n seeded requests and decodes them for a few steps. Then the same decode
step runs twice, through the decode path and through the prefill path (the step's VAE state advances twice; the
latents and stop flags do not depend on it), and once more as a mixed step: a prefill step that resumes a stream (here
a preempted one; mixed steps are only for resuming steps) with the running ones riding along (Scheduler.mixed), every
stream reporting an empty buffer. Checks:
  1 the prefill path's latents match the decode path's (cosine median > 0.999 and min > 0.97 over the rows; both read
    the same K/V, only the attention kernels differ, and a few rows' LocDiT solves amplify that: main 9029009 gave a
    min of 0.9993 over 6 rows but 0.9775 over 16) and its stop flags are the same
  2 a mixed step returns one patch per sequence, the resumed stream's next and every running one's next, and the
    engine keeps going (every request ends)
"""

import argparse
import json
import os
import sys

import numpy as np

__test__ = False


def main(args) -> int:
    import torch

    if not torch.cuda.is_available():
        print("[SKIP] no CUDA")
        return 0
    from nanovllm_voxcpm.dialect_tts import DEFAULT_DIT, DEFAULT_MODEL, resolve, turbo_config

    path = resolve(DEFAULT_MODEL)
    turbo = {**turbo_config(resolve(DEFAULT_DIT), head_tokens=0), "mix_decode": True, "stream_guard_ms": 0}
    if args.first_t_span:  # the first patch's own LocDiT grid: the prefill path below must still solve as decode does
        turbo["dit_first_t_span"] = [float(x) for x in args.first_t_span.split(",")]
    os.environ["NANOVLLM_VOXCPM2_TURBO"] = json.dumps(turbo)
    os.environ.setdefault("NANOVLLM_SERVERPOOL_NUM_KVCACHE_BLOCKS", "64")
    from nanovllm_voxcpm.models.voxcpm2.server import VoxCPM2ServerImpl

    srv = VoxCPM2ServerImpl(path, max_num_batched_tokens=4096, max_num_seqs=16, max_model_len=1024, devices=[0])
    eng = srv.llm
    texts = [s.strip() for s in open(os.path.join(os.path.dirname(__file__), "..", "..", "examples",
                                                  "ar_dialects.txt"), encoding="utf-8") if s.strip()]
    for i in range(args.n):
        eng.add_request(f"r{i}", texts[i], seed=100 + i, max_generate_length=200)
    for _ in range(4):  # the prefill, then a few decode steps
        eng.step()
    fails = []
    seqs, prefill = eng.scheduler.schedule()  # a decode step (its KV slots allocated), run by hand twice
    assert not prefill and seqs
    tasks = [eng.preprocess_seq(s, False) for s in seqs]
    out_d = eng.model_runner.call("run", tasks, False)
    out_p = eng.model_runner.call("run", tasks, True, False)  # (a mixed step's riders: not a first patch)
    for s, o in zip(seqs, out_d):  # the decode path's results, as the engine's step would take them
        eng.postprocess_seq(s, o, False)
    for s in seqs:
        if s.stoped:
            eng.scheduler.finish(s)
    cos = []
    for a, b in zip(out_d, out_p):
        x, y = a["latents"].ravel().astype(np.float64), b["latents"].ravel().astype(np.float64)
        cos.append(float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12)))
        if a["stop_flag"] != b["stop_flag"]:
            fails.append("stop flag")
    if args.first_t_span:  # for scale: the same rows through the first patch's grid
        out_f = eng.model_runner.call("run", tasks, True, True)
        cf = [float(a["latents"].ravel() @ b["latents"].ravel() / (np.linalg.norm(a["latents"]) * np.linalg.norm(
            b["latents"]) + 1e-12)) for a, b in zip(out_d, out_f)]
        print(f"[info] the first patch's grid instead: latents cosine min {min(cf):.6f} (not a check)")
    ok = min(cos) > 0.97 and float(np.median(cos)) > 0.999
    rows = "" if ok else f" (rows: {', '.join(f'{c:.5f}' for c in cos)})"
    print(f"[{'PASS' if ok else 'FAIL'}] prefill path = decode path: latents cosine median {np.median(cos):.6f}, "
          f"min {min(cos):.6f} over {len(cos)} rows{rows}")
    if not ok:
        fails.append("latents")
    # (the last token's KV was written twice, with the same values); now a mixed step
    sch = eng.scheduler
    resumed = sch.running.pop()
    sch.preempt(resumed)
    eng.stream_buffer = lambda seq: 0.0  # every stream is short of audio: all of them ride along
    before = {s.seq_id: len(s.custom_payload.generated_waveforms) for s in [*sch.running, resumed]}
    stepped = eng.step()
    del eng.stream_buffer
    mixed = [s.seq_id for s in sch.mixed]
    after = {s.seq_id: len(s.custom_payload.generated_waveforms) for s in stepped}
    ok = (sorted(mixed) == sorted(k for k in before if k != resumed.seq_id)
          and all(after.get(k) == v + 1 for k, v in before.items()))
    print(f"[{'PASS' if ok else 'FAIL'}] a mixed step: the resumed stream's next patch and {len(mixed)} running "
          f"ones' next")
    if not ok:
        fails.append("mixed step")
    n = 0
    while not eng.is_finished() and n < 400:
        eng.step()
        n += 1
    ok = eng.is_finished()
    print(f"[{'PASS' if ok else 'FAIL'}] every request ends ({n} more steps)")
    if not ok:
        fails.append("end")
    print("FAILED: " + ", ".join(fails) if fails else "all checks passed")
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--first-t-span", default="", help="e.g. 1.0,0.7071,0.0: TurboConfig.dit_first_t_span")
    sys.exit(main(ap.parse_args()))
