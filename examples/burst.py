"""N voice-cloned requests submitted at the same instant: time to first audio.

    NANOVLLM_BUSY_POLL_S=30 python examples/burst.py --voice-dir voices/ --n 200

Request i: sentence i of ar_dialects.txt, dialect i % 11, voice i % voices (the .wav files of --voice-dir). The first
burst warms the engine up; it pairs each sentence with the next request's dialect and voice, because the engine
caches the first request of every voice and dialect, text included, and repeating it would prefill almost nothing.
NANOVLLM_BUSY_POLL_S keeps the engine's and this client's loops from sleeping between bursts.
"""

import argparse
import asyncio
import os
import time

import numpy as np

from nanovllm_voxcpm import DialectTTS
from nanovllm_voxcpm.dialect_tts import DEFAULT_DIT, DEFAULT_MODEL

TEXTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ar_dialects.txt")


async def burst(tts, texts, voices, shift=0) -> np.ndarray:
    """Milliseconds from submitting every request to each request's first audio."""
    t0 = time.perf_counter()

    async def one(i, text):
        first = None
        k = i + shift
        async for _ in tts.stream(text, dialect=tts.dialects[k % len(tts.dialects)], voice=voices[k % len(voices)],
                                  seed=1000 + i, max_patches=400):
            first = first or time.perf_counter() - t0
        return first

    return np.array(await asyncio.gather(*[one(i, t) for i, t in enumerate(texts)])) * 1000


async def main(args):
    sentences = [s.strip() for s in open(TEXTS, encoding="utf-8") if s.strip()]
    texts = [sentences[i % len(sentences)] for i in range(args.n)]
    async with await DialectTTS.load(args.model, args.dit, max_concurrency=max(args.n, 16)) as tts:
        voices = sorted(f[:-4] for f in os.listdir(args.voice_dir) if f.endswith(".wav"))
        for v in voices:
            await tts.add_voice(v, os.path.join(args.voice_dir, f"{v}.wav"))
        await burst(tts, texts, voices, shift=1)  # warm-up, see above
        for r in range(args.rounds):
            t = await burst(tts, texts, voices)
            print(f"round {r}: {args.n} requests, TTFA p50 {np.percentile(t, 50):.1f} ms, p95 {np.percentile(t, 95):.1f} ms")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice-dir", required=True, help="reference clips (.wav) of the voices to clone")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dit", default=DEFAULT_DIT)
    asyncio.run(main(ap.parse_args()))
