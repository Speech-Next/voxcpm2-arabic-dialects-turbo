"""GPU checks of the engine robustness fixes on the real model (VoxCPM-Kernel/VoxCPM2-Arabic-Dialects, its 11 dialect
adapters, the distilled LocDiT, head start on). Not collected by pytest (no test_ functions): run it as a script once
the GPU is free; it exits 0 when every check passes (and skips without CUDA).

    # 1. reference audio of plain seeded requests, recorded on main (the editable install imports the main checkout):
    cd /workspace/voxcpm2-arabic-dialects-turbo
    .venv/bin/python <worktree>/tests/gpu/verify_engine_fixes.py --record
    # 2. this branch: every check, the reference audio compared bit for bit
    PYTHONPATH=<worktree> .venv/bin/python <worktree>/tests/gpu/verify_engine_fixes.py

Engines (each a fresh DialectTTS: max_concurrency 16, a small KV pool):
  A  reference requests (bit-identical to --record) | a long request alone (baseline of B) | 1 mixed batch: base +
     every adapter | 6 over-long text with head start must raise | 4 cancels mid-stream, then a full batch of
     max_concurrency streams (leaked VAE slots would kill it) | 5 1.5 x max_concurrency streams at once
  B  2 the long request in adapter A1, then in A2 (same voice, text, seed): A2 must equal its baseline from A (a
     cross-adapter prefix-cache hit would reuse A1's KV)
  C  7 a tiny shared-memory audio ring and a stalled client: the lapped stream must fail, not end quietly
On main, A's reference passes and the other checks fail (the first engine death ends that engine's checks).
"""

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
import traceback

import numpy as np

__test__ = False  # pytest: nothing to collect here

HERE = os.path.dirname(os.path.abspath(__file__))
VOICES = "VoxCPM-Kernel/Arabic-Hindi-Reference-Voices"
TEXTS = os.path.join(HERE, "..", "..", "examples", "ar_dialects.txt")
# (name, dialect, voice clip, line of ar_dialects.txt, seed): plain single requests, run one after another
REFERENCE = [("najdi_ref", "najdi", "SaudiReem_38.wav", 0, 11), ("base_ref", None, "SaudiReem_38.wav", 1, 12),
             ("egyptian_ref", "egyptian", "QatariJassim.wav", 2, 13)]
A1, A2 = "najdi", "egyptian"
LONG_VOICE = "JordanianAnas.wav"  # 75 patches + ~5 lines of text (~215 tokens): the prompt's first KV block is full


def sha(audio: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(audio, dtype=np.float32).tobytes()).hexdigest()


class Checks:
    def __init__(self):
        self.rows: list[tuple[str, bool, str]] = []

    def add(self, name: str, ok: bool, detail: str = "") -> bool:
        self.rows.append((name, bool(ok), detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}", flush=True)
        return bool(ok)

    def failed(self) -> list[str]:
        return [n for n, ok, _ in self.rows if not ok]


def lines() -> list[str]:
    with open(TEXTS, encoding="utf-8") as f:
        return [s.strip() for s in f if s.strip()]


def voice_file(name: str) -> str:
    from huggingface_hub import snapshot_download

    return os.path.join(snapshot_download(VOICES, repo_type="dataset", allow_patterns=["*.wav"]), name)


async def load(args):
    from nanovllm_voxcpm import DialectTTS

    return await DialectTTS.load(model=args.model, max_concurrency=args.max_concurrency, kv_blocks=args.kv_blocks,
                                 gpu_memory_utilization=args.gpu_mem, device=args.device)


async def synth(tts, text, dialect, voice, seed, max_patches):
    return await tts.synthesize(text, dialect, voice, seed=seed, max_patches=max_patches, normalize=False)


def engine_dead(tts) -> bool:
    return any(getattr(s, "_engine_error", None) is not None for s in tts.server.servers)


async def alive(tts, texts) -> bool:
    try:
        return (await synth(tts, texts[3], None, None, 1, 20)).size > 0
    except Exception:
        return False


async def guarded(checks: Checks, name: str, tts, coro) -> bool:
    """Run one check; an exception fails it (and a dead engine ends the engine's remaining checks)."""
    try:
        return await coro
    except Exception as e:
        checks.add(name, False, f"{type(e).__name__}: {str(e).strip().splitlines()[-1][:300] if str(e).strip() else ''}")
        traceback.print_exc()
        return False


async def engine_a(args, checks: Checks, results: dict, texts: list[str]) -> None:
    tts = await load(args)
    try:
        voices = {clip: clip.rsplit(".", 1)[0] for clip in {r[2] for r in REFERENCE} | {LONG_VOICE}}
        for clip, name in voices.items():
            await tts.add_voice(name, voice_file(clip))  # no transcript: "ref" cloning
        # R: plain seeded requests, first thing in a fresh engine
        ref = {}
        for name, dialect, clip, line, seed in REFERENCE:
            audio = await synth(tts, texts[line], dialect, voices[clip], seed, 150)
            ref[name] = {"sha256": sha(audio), "samples": int(audio.size)}
        if args.record:
            with open(args.ref, "w") as f:
                json.dump(ref, f, indent=1)
            print(f"recorded the reference audio of {len(ref)} requests to {args.ref}")
            return
        if os.path.exists(args.ref):
            with open(args.ref) as f:
                want = json.load(f)
            for name, got in ref.items():
                checks.add(f"R bit-identical audio: {name}", want.get(name) == got,
                           "" if want.get(name) == got else f"recorded {want.get(name)}, now {got}")
        else:
            checks.add("R reference recorded on main", False, f"{args.ref} missing: run with --record on main first")

        long_text = " ".join(texts[4:9])
        results["alone"] = await synth(tts, long_text, A2, voices[LONG_VOICE], 21, 300)  # B's baseline

        async def mixed():  # 1: base rows + all 11 adapters in one batch (CUDA graph bs 16: padding rows)
            dialects = [None] + list(tts.dialects)
            outs = await asyncio.gather(*(synth(tts, texts[10 + i], d, "QatariJassim", 100 + i, 60)
                                          for i, d in enumerate(dialects)))
            return checks.add("1 mixed batch (base + every adapter)", all(o.size for o in outs),
                              f"{len(dialects)} streams")

        async def over_long():  # 6: the whole request is checked, not just its head
            try:
                audio = await synth(tts, texts[0], A1, "SaudiReem_38", 5, 1000)
            except RuntimeError as e:
                return checks.add("6 over-long head-start request raises", "max_model_len" in str(e), "max_model_len")
            return checks.add("6 over-long head-start request raises", False,
                              f"no error: {audio.size} samples (the stream ended after its head)")

        async def cancels():  # 4: cancels during steps, then max_concurrency streams (all VAE slots)
            srv = tts.server.servers[0]
            adapters = list(tts.dialects)

            async def cancelled(i, k):
                agen = srv.generate(texts[20 + i], max_generate_length=200, cfg_value=2.0, seed=200 + i,
                                    lora_name=adapters[i % len(adapters)], voice="QatariJassim",
                                    clone_mode="ref")
                got = 0
                async for _ in agen:
                    got += 1
                    if got >= k:
                        break
                await agen.aclose()  # its finally sends the cancel
                return got

            await asyncio.gather(*(cancelled(i, 1 + i % 4) for i in range(10)),
                                 *(synth(tts, texts[40 + i], adapters[i], "QatariJassim", 250 + i, 60)
                                   for i in range(4)))
            outs = await asyncio.gather(*(synth(tts, texts[50 + i], adapters[i % len(adapters)],
                                                "QatariJassim", 260 + i, 40)
                                          for i in range(args.max_concurrency)))
            return checks.add("4 cancels mid-stream leave no VAE slot behind", all(o.size for o in outs),
                              f"10 cancelled, then {args.max_concurrency} concurrent streams")

        async def over_capacity():  # 5: more live streams than VAE slots: the rest wait
            n = args.max_concurrency * 3 // 2
            adapters = [None] + list(tts.dialects)
            outs = await asyncio.gather(*(synth(tts, texts[70 + i], adapters[i % len(adapters)],
                                                "QatariJassim", 400 + i, 40) for i in range(n)))
            return checks.add("5 more streams than max_concurrency", all(o.size for o in outs), f"{n} streams")

        for name, check in (("1 mixed batch (base + every adapter)", mixed), ("6 over-long head-start request", over_long),
                            ("4 cancels mid-stream", cancels), ("5 more streams than max_concurrency", over_capacity)):
            await guarded(checks, name, tts, check())
            if engine_dead(tts):
                checks.add("engine A alive", False, f"died in check {name!r}")
                return
        checks.add("engine A alive after the checks", await alive(tts, texts))
    finally:
        await tts.close()


async def engine_b(args, checks: Checks, results: dict, texts: list[str]) -> None:
    tts = await load(args)
    try:
        voice = LONG_VOICE.rsplit(".", 1)[0]
        await tts.add_voice(voice, voice_file(LONG_VOICE))
        long_text = " ".join(texts[4:9])

        async def cross():  # 2: equal voice and text in two adapters must not share prefix KV
            a1 = await synth(tts, long_text, A1, voice, 21, 300)
            a2 = await synth(tts, long_text, A2, voice, 21, 300)
            alone = results.get("alone")
            if alone is None:
                return checks.add("2 no cross-adapter prefix cache hit", False, "no baseline (engine A failed)")
            same = a2.size == alone.size and sha(a2) == sha(alone)
            n = min(a2.size, alone.size)
            diff = float(np.abs(a2[:n] - alone[:n]).max()) if n else float("nan")
            checks.add("2 no cross-adapter prefix cache hit", same,
                       f"{A2} after {A1} vs {A2} alone: {a2.size} / {alone.size} samples, max |diff| {diff:.3g}")
            return checks.add("2 the two adapters' audio differs", sha(a1) != sha(a2))

        await guarded(checks, "2 no cross-adapter prefix cache hit", tts, cross())
    finally:
        await tts.close()


async def engine_c(args, checks: Checks, texts: list[str]) -> None:
    os.environ["NANOVLLM_SHM_AUDIO_MB"] = str(args.ring_mb)  # read by the engine process at its start
    try:
        tts = await load(args)
    finally:
        os.environ.pop("NANOVLLM_SHM_AUDIO_MB", None)
    try:
        voice = "QatariJassim"
        await tts.add_voice(voice, voice_file(voice + ".wav"))
        adapters = list(tts.dialects)

        async def consume(i):
            chunks = 0
            try:
                async for _ in tts.stream(texts[100 + i], adapters[i % len(adapters)], voice, seed=300 + i,
                                          max_patches=120, normalize=False):
                    chunks += 1
                    if i == 0 and chunks == 1:
                        time.sleep(args.ring_stall_s)  # the client's loop stalls: the engine laps the ring
                return None
            except RuntimeError as e:
                return str(e)

        async def lapped():  # 7
            errors = [e for e in await asyncio.gather(*(consume(i) for i in range(8))) if e]
            return checks.add("7 overwritten audio fails the stream", any("overwritten" in e for e in errors),
                              f"{len(errors)} of 8 streams failed")

        await guarded(checks, "7 overwritten audio fails the stream", tts, lapped())
        checks.add("engine C alive after the lapped streams", not engine_dead(tts) and await alive(tts, texts))
    finally:
        await tts.close()


async def run(args) -> int:
    import nanovllm_voxcpm

    print(f"nanovllm_voxcpm from {os.path.dirname(nanovllm_voxcpm.__file__)}", flush=True)
    texts = lines()
    checks, results = Checks(), {}
    engines = "a" if args.record else args.engines
    if "a" in engines:
        await engine_a(args, checks, results, texts)
    if "b" in engines:
        await engine_b(args, checks, results, texts)
    if "c" in engines:
        await engine_c(args, checks, texts)
    if args.record:
        return 0
    failed = checks.failed()
    print(f"\n{len(checks.rows) - len(failed)} passed, {len(failed)} failed" + (": " + "; ".join(failed) if failed else ""))
    return 1 if failed else 0


def main() -> int:
    from nanovllm_voxcpm.dialect_tts import DEFAULT_MODEL

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--record", action="store_true", help="only record the reference audio (run this on main)")
    ap.add_argument("--ref", default=os.path.join(HERE, "engine_fixes_reference.json"), help="reference audio file")
    ap.add_argument("--engines", default="abc", help="which engines to run (a, b, c; B needs A's baseline)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-concurrency", type=int, default=16)
    ap.add_argument("--kv-blocks", type=int, default=128)
    ap.add_argument("--gpu-mem", type=float, default=0.5, help="gpu_memory_utilization (the KV pool is --kv-blocks)")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--ring-mb", type=float, default=0.25, help="engine C: shared audio ring size")
    ap.add_argument("--ring-stall-s", type=float, default=3.0, help="engine C: how long the client's loop stalls")
    args = ap.parse_args()
    import torch

    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return 0
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
