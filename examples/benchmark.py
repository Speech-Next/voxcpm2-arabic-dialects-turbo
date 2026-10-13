"""Time to first audio (TTFB) and time to the whole sentence, by the number of requests sent at the same instant.

    NANOVLLM_BUSY_POLL_S=30 python examples/benchmark.py --voice-dir voices/

Per level: one warm-up burst, then --rounds bursts of N voice-cloned requests sent at once (10 for N = 1). Request i
speaks sentence i of ar_dialects.txt, in dialect i (in turn), cloning voice i (in turn) from the .wav files of
--voice-dir. The warm-up pairs each sentence with another dialect and voice: the engine caches a voice's first
request, text included, and repeating it would measure the cache. Prints, per level, TTFB average / p50 / p95 and the
average time until the last audio chunk.
"""

import argparse
import asyncio
import os
import time

import numpy as np

from nanovllm_voxcpm import DialectTTS
from nanovllm_voxcpm.dialect_tts import DEFAULT_DIT, DEFAULT_MODEL

TEXTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ar_dialects.txt")


async def burst(tts, texts, voices, shift=0):
    """(first audio s, last audio s) of every request, from the instant all were submitted."""
    t0 = time.perf_counter()

    async def one(i, text):
        k = i + shift
        first = last = None
        async for _ in tts.stream(text, dialect=tts.dialects[k % len(tts.dialects)], voice=voices[k % len(voices)],
                                  seed=1000 + i, max_patches=400):
            last = time.perf_counter() - t0
            first = first if first is not None else last
        return first, last

    return await asyncio.gather(*[one(i, t) for i, t in enumerate(texts)])


async def main(args):
    sentences = [s.strip() for s in open(TEXTS, encoding="utf-8") if s.strip()]
    levels = [int(x) for x in args.levels.split(",")]
    async with await DialectTTS.load(args.model, args.dit, max_concurrency=max(levels)) as tts:
        voices = sorted(f[:-4] for f in os.listdir(args.voice_dir) if f.endswith(".wav"))
        for v in voices:
            await tts.add_voice(v, os.path.join(args.voice_dir, f"{v}.wav"))
        print("| requests at once | TTFB avg | TTFB p50 | TTFB p95 | whole sentence, avg |")
        print("|---:|---:|---:|---:|---:|")
        for n in levels:
            texts = [sentences[i % len(sentences)] for i in range(n)]
            await burst(tts, texts, voices, shift=1)  # warm-up, see above
            res = []
            for _ in range(10 if n == 1 else args.rounds):
                res += await burst(tts, texts, voices)
            ttfb = np.array([r[0] for r in res]) * 1000
            total = np.array([r[1] for r in res])
            print(f"| {n} | {ttfb.mean():.1f} ms | {np.percentile(ttfb, 50):.1f} ms | {np.percentile(ttfb, 95):.1f} ms | "
                  f"{total.mean():.2f} s |", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice-dir", required=True, help="reference clips (.wav) of the voices to clone")
    ap.add_argument("--levels", default="1,8,16,32,64,128,200,256", help="requests sent at once, per level")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dit", default=DEFAULT_DIT, help="'none' for the model's own LocDiT")
    args = ap.parse_args()
    args.dit = None if args.dit == "none" else args.dit
    asyncio.run(main(args))
