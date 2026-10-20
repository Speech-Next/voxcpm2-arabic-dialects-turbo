"""Time to first audio (TTFB) and time to the whole sentence, by the number of requests sent at the same instant.

    NANOVLLM_BUSY_POLL_S=30 python examples/benchmark.py

The model, the distilled LocDiT and the reference voices are downloaded from Hugging Face on first use.

Per level: one warm-up burst, then --rounds bursts of N voice-cloned requests sent at once (10 for N = 1). Request i
clones voice i (in turn) of the .wav files of --voices (a folder or a Hugging Face dataset id). A voice whose file name starts with a dialect
("JordanianAnas.wav", "SaudiReem.wav", "HindiPooja.wav" for hindi_english) speaks in that dialect; other voices take
the dialects in turn. Arabic requests speak sentence i of ar_dialects.txt, hindi_english ones hi_sentences.txt. The
warm-up pairs each request with another voice: the engine caches a voice's first request, text included, and
repeating it would measure the cache. Prints, per level, TTFB average / p50 / p95 / p99 / max, the first word
(when a player that starts at the first audio plays the speech's start: the first 10 ms frame from which 50 ms stay
within 25 dB of the request's loudest frame; later if a chunk arrives after its play time) p50 / p95 / p99 / max,
and the average time until the last
audio chunk.

The engine is the default, routed one (RoutedDialectTTS: the adapters the routes use and --tagged-model's LoRA in one
engine; the voices' dialects follow its routing table, e.g. Saudi and Hindi voices on the adapters, Jordanian,
Palestinian and Qatari ones on teacher-v3); --no-route: the unrouted engine (DialectTTS, every adapter).
"""

import argparse
import asyncio
import os
import time

import numpy as np

from nanovllm_voxcpm import DialectTTS, RoutedDialectTTS
from nanovllm_voxcpm.dialect_routing import DEFAULT_TAGGED_MODEL
from nanovllm_voxcpm.dialect_tts import DEFAULT_DIT, DEFAULT_MODEL

HERE = os.path.dirname(os.path.abspath(__file__))
VOICES = "VoxCPM-Kernel/Arabic-Hindi-Reference-Voices"


def voice_folder(spec: str) -> str:
    """A local folder as is; otherwise a Hugging Face dataset id (its pinned revision or id@revision), downloaded
    (its .wav files) on first use."""
    if os.path.isdir(spec):
        return spec
    from nanovllm_voxcpm.hub import download

    return download(spec, repo_type="dataset", allow_patterns=["*.wav"])


def lines(name):
    return [s.strip() for s in open(os.path.join(HERE, name), encoding="utf-8") if s.strip()]


def requests(n, voices, dialects, shift=0):
    """(voice, dialect, text) of request i = 0..n-1."""
    ar, hi = lines("ar_dialects.txt"), lines("hi_sentences.txt")
    out = []
    for i in range(n):
        v = voices[(i + shift) % len(voices)]
        d = next((d for d in dialects if v.lower().startswith(d.split("_")[0])), dialects[i % len(dialects)])
        out.append((v, d, (hi if d == "hindi_english" else ar)[i % len(hi if d == "hindi_english" else ar)]))
    return out


def speech_start(audio: np.ndarray, sr: int) -> int:
    """The sample where speech starts: the first 10 ms frame from which 5 frames stay within 25 dB of the loudest."""
    h = sr // 100
    if len(audio) < h:
        return 0
    frames = audio[: len(audio) // h * h].reshape(-1, h)
    db = 20 * np.log10(np.sqrt((frames.astype(np.float64) ** 2).mean(1)) + 1e-9)
    on = np.convolve((db > db.max() - 25).astype(int), np.ones(5, int), "valid") == 5
    return int(np.argmax(on)) * h if on.any() else 0


def first_word(chunks: list[tuple[float, np.ndarray]], sr: int) -> float:
    """When a player that starts at the first chunk's arrival (and waits for late chunks) plays the speech's start."""
    s = speech_start(np.concatenate([c for _, c in chunks]), sr)
    t = start = 0.0
    for arrival, c in chunks:
        t = max(t, arrival)  # this chunk starts playing
        if start + len(c) > s:
            return t + (s - start) / sr
        t += len(c) / sr
        start += len(c)
    return t


async def burst(tts, reqs):
    """(first audio s, last audio s, first word s) of every request, from the instant all were submitted."""
    t0 = time.perf_counter()

    async def one(i, voice, dialect, text):
        chunks = []
        async for c in tts.stream(text, dialect=dialect, voice=voice, seed=1000 + i, max_patches=400):
            chunks.append((time.perf_counter() - t0, c))
        return chunks[0][0], chunks[-1][0], first_word(chunks, tts.sample_rate)

    return await asyncio.gather(*[one(i, *r) for i, r in enumerate(reqs)])


async def main(args):
    levels = [int(x) for x in args.levels.split(",")]
    kw = dict(max_concurrency=args.max_concurrency or max(levels), kv_blocks=args.kv_blocks,
              w8a8=args.precision == "int8")
    if not args.no_route:
        tts = await RoutedDialectTTS.load(args.model, args.tagged_model, args.dit, **kw)
    else:
        tts = await DialectTTS.load(args.model, args.dit, **kw)
    async with tts:
        folder = voice_folder(args.voices)
        voices = sorted(f[:-4] for f in os.listdir(folder) if f.endswith(".wav"))
        for v in voices:
            await tts.add_voice(v, os.path.join(folder, f"{v}.wav"))
        if not args.no_route:
            routes = [tts.route(d) for _, d, _ in requests(len(voices), voices, tts.dialects)]
            n_adapters = sum(r.model == "adapters" for r in routes)
            print(f"routing: {n_adapters} of {len(voices)} voices on the adapters model, the others on the tagged one")
        # every voice once: a voice's first request also encodes its reference, which no measured burst should pay
        await burst(tts, requests(len(voices), voices, tts.dialects, shift=3))
        print("| requests at once | TTFB avg | TTFB p50 | TTFB p95 | TTFB p99 | TTFB max | first word p50 | "
              "first word p95 | first word p99 | first word max | whole sentence, avg |")
        print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for n in levels:
            await burst(tts, requests(n, voices, tts.dialects, shift=7))  # warm-up, see above
            res = []
            for _ in range(10 if n == 1 else args.rounds):
                res += await burst(tts, requests(n, voices, tts.dialects))
            ttfb = np.array([r[0] for r in res]) * 1000
            total = np.array([r[1] for r in res])
            word = np.array([r[2] for r in res]) * 1000
            q = lambda x: [np.percentile(x, 50), np.percentile(x, 95), np.percentile(x, 99), x.max()]  # noqa: E731
            t50, t95, t99, tmax = q(ttfb)
            print(f"| {n} | {ttfb.mean():.1f} ms | {t50:.1f} ms | {t95:.1f} ms | {t99:.1f} ms | {tmax:.1f} ms | "
                  + " | ".join(f"{v:.0f} ms" for v in q(word)) + f" | {total.mean():.2f} s |", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", default=VOICES, help="reference clips (.wav) to clone: a folder or a Hugging Face "
                                                    "dataset id")
    ap.add_argument("--levels", default="1,8,16,32,64,128,200,256", help="requests sent at once, per level")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dit", default=DEFAULT_DIT, help="'none' for the model's own LocDiT")
    ap.add_argument("--kv-blocks", type=int, default=600, help="KV cache blocks of 256 tokens; 400 for --dit none at "
                                                               "200 requests (its CUDA graphs are larger)")
    ap.add_argument("--max-concurrency", type=int, help="requests the engine runs at once (default: the largest level)")
    ap.add_argument("--precision", choices=["int8", "bf16"], default="int8", help="the layers")
    ap.add_argument("--no-route", action="store_true", help="the unrouted engine (DialectTTS: every adapter, no "
                                                             "teacher-v3)")
    ap.add_argument("--tagged-model", default=DEFAULT_TAGGED_MODEL)
    args = ap.parse_args()
    args.dit = None if args.dit == "none" else args.dit
    asyncio.run(main(args))
