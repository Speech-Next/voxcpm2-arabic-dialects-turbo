"""Streaming under load: requests arriving at random (Poisson) at each --rates rate for --duration seconds.

    NANOVLLM_BUSY_POLL_S=30 python examples/streaming_benchmark.py --rates 30,40,50,60

Each request is streamed and played from its first chunk by a model player that waits when it has nothing to play.
Prints, per rate, the requests, the most streams generating at once, TTFB p50 / p95 / p99, the first word (when the
player reaches the speech's start, see benchmark.first_word) p50 / p95 / p99, and the share of streams with a gap of
more than 50 ms after their speech began (the player waiting for a late chunk), with the largest gap. Requests are
benchmark.py's (voice i, its dialect, sentence i), cloned from --voices.
"""

import argparse
import asyncio
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark import VOICES, first_word, requests, speech_start, voice_folder, burst  # noqa: E402

from nanovllm_voxcpm import DialectTTS, RoutedDialectTTS  # noqa: E402
from nanovllm_voxcpm.dialect_routing import DEFAULT_TAGGED_MODEL  # noqa: E402
from nanovllm_voxcpm.dialect_tts import DEFAULT_DIT, DEFAULT_MODEL  # noqa: E402


def gaps_after_speech(chunks: list[tuple[float, np.ndarray]], sr: int) -> list[float]:
    """The player's waits (s) for a late chunk after the speech began."""
    s = speech_start(np.concatenate([c for _, c in chunks]), sr)
    t = start = 0.0
    gaps = []
    for arrival, c in chunks:
        if start > s and arrival > t:
            gaps.append(arrival - t)
        t = max(t, arrival) + len(c) / sr
        start += len(c)
    return gaps


async def one(tts, i, voice, dialect, text, active):
    t0 = time.perf_counter()
    active[0] += 1
    active[1] = max(active[1], active[0])
    chunks = []
    async for c in tts.stream(text, dialect=dialect, voice=voice, seed=1000 + i, max_patches=400):
        chunks.append((time.perf_counter() - t0, c))
    active[0] -= 1
    gaps = gaps_after_speech(chunks, tts.sample_rate)
    return chunks[0][0], first_word(chunks, tts.sample_rate), max(gaps, default=0.0)


async def main(args):
    kw = dict(max_concurrency=args.max_concurrency, kv_blocks=args.kv_blocks)
    if not args.no_route:
        tts = await RoutedDialectTTS.load(args.model, args.tagged_model, args.dit, **kw)
    else:
        tts = await DialectTTS.load(args.model, args.dit, **kw)
    async with tts:
        folder = voice_folder(args.voices)
        voices = sorted(f[:-4] for f in os.listdir(folder) if f.endswith(".wav"))
        for v in voices:
            await tts.add_voice(v, os.path.join(folder, f"{v}.wav"))
        await burst(tts, requests(len(voices), voices, tts.dialects, shift=3))  # every voice once (its reference)
        reqs = requests(len(voices) * 64, voices, tts.dialects)
        print("| requests/s | requests | streams at once, max | TTFB p50 | TTFB p95 | TTFB p99 | first word p50 | "
              "first word p95 | first word p99 | streams with a gap > 50 ms | largest gap |")
        print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for rate in [float(x) for x in args.rates.split(",")]:
            rng = np.random.default_rng(int(rate))
            active, tasks, t_start, t_next, i = [0, 0], [], time.perf_counter(), 0.0, 0
            while t_next < args.duration:
                await asyncio.sleep(max(0.0, t_start + t_next - time.perf_counter()))
                tasks.append(asyncio.create_task(one(tts, i, *reqs[i % len(reqs)], active)))
                i += 1
                t_next += rng.exponential(1.0 / rate)
            res = np.array(await asyncio.gather(*tasks)) * 1000
            ttfb, word, gap = res[:, 0], res[:, 1], res[:, 2]
            p = lambda x, q: f"{np.percentile(x, q):.0f} ms"  # noqa: E731
            print(f"| {rate:g} | {len(res)} | {active[1]} | {p(ttfb, 50)} | {p(ttfb, 95)} | {p(ttfb, 99)} | "
                  f"{p(word, 50)} | {p(word, 95)} | {p(word, 99)} | {np.mean(gap > 50) * 100:.1f} % | "
                  f"{gap.max():.0f} ms |", flush=True)
            await asyncio.sleep(3)  # the last streams end before the next rate


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", default=VOICES, help="reference clips (.wav) to clone: a folder or a Hugging Face "
                                                     "dataset id")
    ap.add_argument("--rates", default="30,40,50,60", help="requests per second, one measurement each")
    ap.add_argument("--duration", type=float, default=20.0, help="seconds of arrivals per rate")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dit", default=DEFAULT_DIT, help="'none' for the model's own LocDiT")
    ap.add_argument("--kv-blocks", type=int, default=600)
    ap.add_argument("--max-concurrency", type=int, default=256)
    ap.add_argument("--no-route", action="store_true", help="the unrouted engine (DialectTTS)")
    ap.add_argument("--tagged-model", default=DEFAULT_TAGGED_MODEL)
    args = ap.parse_args()
    args.dit = None if args.dit == "none" else args.dit
    asyncio.run(main(args))
