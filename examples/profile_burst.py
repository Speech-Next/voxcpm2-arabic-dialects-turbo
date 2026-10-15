"""Timeline of a voice-cloned burst's host path, from the engine's NANOVLLM_PROFILE records (issue #9).

    NANOVLLM_BUSY_POLL_S=30 python examples/profile_burst.py                    # 200 requests, 5 measured bursts
    NANOVLLM_BUSY_POLL_S=30 python examples/profile_burst.py --levels 1,8,64,200 --out profile.json

For every measured burst, relative to its submission (t = 0, as TTFA): the client's sends (one per chunk), each
registration batch done, the admission window's prepare calls, the head step's start (server; runner), its host
preparation (context, inputs + the one host-to-device copy, noise) up to the launch of the first model kernel, the
GPU reaching that kernel, the first audio leaving the engine, and the time to first audio (TTFA) at the client.
The engine runs unsynchronized (NANOVLLM_PROFILE_SYNC=0, as unprofiled; stage times are host times) with CUDA
events (NANOVLLM_PROFILE_GPU=1) for the GPU side; --sync instead synchronizes at every mark (GPU stage durations,
slower). Requests, voices, dialects and seeds are examples/benchmark.py's; a burst is the same every round.

Bit-identity: --hashes FILE stores the SHA-256 of every request's whole audio (per level and round); --compare FILE
checks a run against such a file (the same requests and seeds: every round of a level must match the reference's
first round). Record on main, compare on the branch, e.g.:

    cd /workspace/voxcpm2-arabic-dialects-turbo   # main checkout (the editable install imports it)
    NANOVLLM_BUSY_POLL_S=30 .venv/bin/python <worktree>/examples/profile_burst.py --levels 1,8,64,200 --hashes main.json
    NANOVLLM_BUSY_POLL_S=30 PYTHONPATH=<worktree> .venv/bin/python <worktree>/examples/profile_burst.py \
        --levels 1,8,64,200 --compare main.json
"""

import argparse
import asyncio
import hashlib
import json
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _setup_env(args) -> str:
    """The engine process reads these when it starts (before DialectTTS.load)."""
    path = args.records or os.path.join(tempfile.mkdtemp(prefix="profile_burst_"), "steps.jsonl")
    os.environ["NANOVLLM_PROFILE"] = path
    os.environ["NANOVLLM_PROFILE_SYNC"] = "1" if args.sync else "0"
    os.environ["NANOVLLM_PROFILE_GPU"] = "0" if args.sync else "1"
    return path


def _read(path: str, offset: int) -> list[dict]:
    with open(path) as f:
        f.seek(offset)
        out = []
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:  # a line still being written
                pass
        return out


def timeline(recs: list[dict], t0: float, ttfa_ms: list[float]) -> dict:
    """One burst's milestones (ms after t0, its submission: TTFA counts from there) from its profile records."""
    ms = lambda t: (t - t0) * 1000  # noqa: E731
    flush = sorted((r for r in recs if r.get("kind") == "client_flush"), key=lambda r: r["t"])
    regs = sorted((r for r in recs if r.get("kind") == "batch_recv"), key=lambda r: r["t"])
    prep = [r for r in recs if r.get("kind") == "prepare"]
    steps = sorted((r for r in recs if r.get("kind") == "prefill"), key=lambda r: r["t"])
    servers = sorted((r for r in recs if r.get("kind") == "server"), key=lambda r: r["t"])
    emits = sorted((r for r in recs if r.get("kind") == "emit"), key=lambda r: r["t"])
    out: dict = {"requests": len(ttfa_ms)}
    if flush:
        out["client_sends"] = [round(ms(r["t"] + r.get("put_ms", 0) / 1000), 3) for r in flush]
    if regs:
        out["registered"] = [round(ms(r["t"] + r["register_ms"] / 1000), 3) for r in regs]
        out["registered_n"] = [r["n"] for r in regs]
        out["register_ms"] = [round(r["register_ms"], 3) for r in regs]
    if prep:
        out["prepare_calls"], out["prepare_ms"] = len(prep), round(sum(r["ms"] for r in prep), 3)
    if steps:
        head = steps[0]
        start = head.get("t_begin", head["t"])
        # the server's record of the head step: written after it, with t = the step's start (before scheduling)
        srv = max((r for r in servers if r["t"] <= head["t"]), key=lambda r: r["t"], default=None)
        host = head.get("ctx", 0.0) + head.get("h2d", 0.0) + head.get("prep", 0.0)
        out.update(
            head_seqs=head.get("bs"), head_tokens=head.get("tokens"),
            step_start=round(ms(srv["t"] if srv is not None else start), 3),  # server: before scheduling
            runner_start=round(ms(head["t"]), 3),  # after scheduling + preprocessing
            sched_preprocess_ms=round((head["t"] - start) * 1000, 3) if "t_begin" in head else None,
            ctx_ms=round(head.get("ctx", 0.0), 3), h2d_ms=round(head.get("h2d", 0.0), 3),
            noise_ms=round(head.get("prep", 0.0), 3),
            first_kernel_launch=round(ms(head["t"]) + host, 3),
        )
        if "gpu_prep" in head and "t_begin" in head:
            out["first_kernel_gpu"] = round(ms(head["t_begin"]) + head["gpu_prep"], 3)
        if regs:
            out["gap_last_registration_to_step"] = round(out["step_start"] - out["registered"][-1], 3)
            out["last_registration_to_first_kernel"] = round(
                out.get("first_kernel_gpu", out["first_kernel_launch"]) - out["registered"][-1], 3)
    if emits:
        out["first_audio_out"] = round(ms(emits[0]["t"]), 3)
    if ttfa_ms:
        a = np.asarray(ttfa_ms)
        out.update(ttfa_min=round(float(a.min()), 3), ttfa_p50=round(float(np.percentile(a, 50)), 3),
                   ttfa_p95=round(float(np.percentile(a, 95)), 3), ttfa_max=round(float(a.max()), 3))
    return out


def show(level: int, k: int, tl: dict) -> None:
    f = lambda key: tl.get(key, float("nan"))  # noqa: E731
    sends = tl.get("client_sends", [])
    regs = tl.get("registered", [])
    print(f"[{level} requests, burst {k}] t = 0 when the burst is submitted (TTFA counts from there)")
    print(f"  client sends      {len(sends)} chunks, done at " + ", ".join(f"+{x:.2f}" for x in sends) + " ms")
    print(f"  registered        {len(regs)} batches {tl.get('registered_n', [])}, at "
          + ", ".join(f"+{x:.2f}" for x in regs) + f" ms ({sum(tl.get('register_ms', [])):.2f} ms of work)")
    if "prepare_calls" in tl:
        print(f"  prepared          {tl['prepare_calls']} idle moments, {tl['prepare_ms']:.2f} ms")
    print(f"  head step         {tl.get('head_seqs')} sequences, {tl.get('head_tokens')} tokens; starts +{f('step_start'):.2f}"
          f" ms (gap {f('gap_last_registration_to_step'):.2f} ms after the last registration)")
    if tl.get("sched_preprocess_ms") is not None:
        print(f"  schedule+preprocess {tl['sched_preprocess_ms']:.2f} ms -> runner +{f('runner_start'):.2f} ms")
    print(f"  step preparation  context {f('ctx_ms'):.2f}, inputs + H2D {f('h2d_ms'):.2f}, noise {f('noise_ms'):.2f} ms"
          f" -> first kernel launched +{f('first_kernel_launch'):.2f} ms"
          + (f", on the GPU +{tl['first_kernel_gpu']:.2f} ms" if "first_kernel_gpu" in tl else ""))
    print(f"  first audio out   +{f('first_audio_out'):.2f} ms; TTFA p50 {f('ttfa_p50'):.2f} / p95 {f('ttfa_p95'):.2f} ms")


async def burst(tts, reqs, hashes: bool):
    """(t0, [ttfa ms], [sha256 of each request's audio] or None): all requests submitted at once, as benchmark.py."""
    t0 = time.perf_counter()

    async def one(i, voice, dialect, text):
        first, h = None, hashlib.sha256() if hashes else None
        async for chunk in tts.stream(text, dialect=dialect, voice=voice, seed=1000 + i, max_patches=400):
            if first is None:
                first = (time.perf_counter() - t0) * 1000
            if h is not None:  # the whole audio's float32 bytes, chunk by chunk
                h.update(np.ascontiguousarray(chunk, dtype=np.float32).tobytes())
        return first, h.hexdigest() if h is not None else None

    res = await asyncio.gather(*[one(i, *r) for i, r in enumerate(reqs)])
    return t0, [r[0] for r in res], [r[1] for r in res] if hashes else None


async def main(args):
    path = _setup_env(args)
    from benchmark import requests, voice_folder

    from nanovllm_voxcpm import DialectTTS
    from nanovllm_voxcpm.dialect_tts import DEFAULT_DIT, DEFAULT_MODEL

    levels = [int(x) for x in args.levels.split(",")]
    hashes = bool(args.hashes or args.compare)
    report: dict = {"records": path, "sync": args.sync, "levels": {}}
    stored: dict = {}
    ref = json.load(open(args.compare)) if args.compare else None
    mismatches = 0
    async with await DialectTTS.load(args.model or DEFAULT_MODEL, DEFAULT_DIT if args.dit is None else (args.dit or None),
                                     max_concurrency=max(max(levels), 1),
                                     gpu_memory_utilization=args.gpu_memory_utilization) as tts:
        folder = voice_folder(args.voices)
        voices = sorted(f[:-4] for f in os.listdir(folder) if f.endswith(".wav"))
        for v in voices:
            await tts.add_voice(v, os.path.join(folder, f"{v}.wav"))
        await burst(tts, requests(len(voices), voices, tts.dialects, shift=3), False)  # every voice once (cache)
        import gc

        for n in levels:
            await burst(tts, requests(n, voices, tts.dialects, shift=7), False)  # warm-up, as benchmark.py
            reqs = requests(n, voices, tts.dialects)
            runs = []
            for k in range(args.rounds):
                gc.collect()
                gc.freeze()
                gc.disable()  # (benchmark-style: no client collection inside a burst)
                offset = os.path.getsize(path) if os.path.exists(path) else 0
                try:
                    t0, ttfa, digests = await burst(tts, reqs, hashes)
                finally:
                    gc.enable()
                await asyncio.sleep(0.05)  # the last step records
                tl = timeline(_read(path, offset), t0, ttfa)
                runs.append(tl)
                if not args.quiet:
                    show(n, k, tl)
                if digests is not None:
                    stored.setdefault(str(n), []).append(digests)
                    if ref is not None:
                        want = ref.get(str(n), [None])[0]
                        bad = sum(a != b for a, b in zip(digests, want)) if want is not None else len(digests)
                        mismatches += bad
                        print(f"  bit-identity vs {args.compare}: {len(digests) - bad}/{len(digests)} requests equal")
            keys = ("step_start", "gap_last_registration_to_step", "sched_preprocess_ms", "ctx_ms", "h2d_ms",
                    "noise_ms", "first_kernel_launch", "first_kernel_gpu", "last_registration_to_first_kernel",
                    "first_audio_out", "ttfa_p50", "ttfa_p95")
            med = {key: round(float(np.median([r[key] for r in runs if r.get(key) is not None])), 3)
                   for key in keys if any(r.get(key) is not None for r in runs)}
            report["levels"][str(n)] = {"median": med, "bursts": runs}
            print(f"== {n} requests, median of {args.rounds} bursts: " + ", ".join(f"{k} {v}" for k, v in med.items()),
                  flush=True)
    if args.hashes:
        json.dump(stored, open(args.hashes, "w"), indent=1)
    if args.out:
        json.dump(report, open(args.out, "w"), indent=1)
    if ref is not None:
        print("bit-identical" if mismatches == 0 else f"{mismatches} requests differ from {args.compare}")
        sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="200", help="requests sent at once, per level")
    ap.add_argument("--rounds", type=int, default=5, help="measured bursts per level")
    ap.add_argument("--voices", default="VoxCPM-Kernel/Arabic-Dialects-and-Hindi-Voice-Samples")
    ap.add_argument("--model", default=None)
    ap.add_argument("--dit", default=None, help="'' for the model's own LocDiT")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--sync", action="store_true", help="synchronize at every mark (GPU stage durations)")
    ap.add_argument("--records", default="", help="the NANOVLLM_PROFILE file (default: a temporary one)")
    ap.add_argument("--out", default="", help="JSON with every burst's timeline and the medians")
    ap.add_argument("--hashes", default="", help="store each request's audio SHA-256 (bit-identity reference)")
    ap.add_argument("--compare", default="", help="compare each request's audio with a --hashes file")
    ap.add_argument("--quiet", action="store_true", help="medians only")
    asyncio.run(main(ap.parse_args()))
