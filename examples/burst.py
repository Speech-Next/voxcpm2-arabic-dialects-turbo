"""N requests submitted at the same instant (one sentence each, dialects and voices mixed): time to first audio.

    NANOVLLM_BUSY_POLL_S=30 python examples/burst.py --n 200 --voice-dir voices/

Request i: sentence i of ar_dialects.txt, adapter i % adapters, voice i % voices (<name>.wav files in --voice-dir,
cloned in "ref" mode), seed 1000 + i. The first burst warms the engine up (CUDA graphs, caches); the bursts after it
are measured. NANOVLLM_BUSY_POLL_S keeps the engine's and this client's loops from blocking between bursts.
The warm-up pairs each sentence with the next request's adapter and voice: the engine caches the KV of the first
request of every voice and adapter, text included, and a later request with the same voice, adapter and text would
prefill almost nothing (13 ms less p95 TTFA for 200 requests here; not what distinct requests see).
"""

import argparse
import asyncio
import os
import time

import numpy as np

from arabic_dialects import register_voice, start_engine

TEXTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ar_dialects.txt")


async def burst(server, texts, kw_of) -> list[float]:
    """Seconds from submitting all requests to each request's first audio chunk."""
    t0 = time.perf_counter()

    async def one(i, text):
        first = None
        async for chunk in server.generate(target_text=text, seed=1000 + i, cfg_value=2.0, max_generate_length=400,
                                           **kw_of(i)):
            if first is None and len(chunk):
                first = time.perf_counter() - t0
        return first

    return await asyncio.gather(*[one(i, t) for i, t in enumerate(texts)])


async def main(args):
    sentences = [s.strip() for s in open(TEXTS, encoding="utf-8") if s.strip()]
    texts = [sentences[i % len(sentences)] for i in range(args.n)]
    server, adapters = await start_engine(args.model, args.dit, max_num_seqs=max(args.n, 16))
    try:
        voices = []
        if args.voice_dir:
            for f in sorted(os.listdir(args.voice_dir)):
                if f.endswith(".wav"):
                    await register_voice(server, f[:-4], os.path.join(args.voice_dir, f))
                    voices.append(f[:-4])

        def kw_of(i):
            kw = {"lora_name": adapters[i % len(adapters)]}
            if voices:
                kw.update(voice=voices[i % len(voices)], clone_mode="ref")
            return kw

        await burst(server, texts, lambda i: kw_of(i + 1))  # warm-up, see above
        for r in range(args.rounds):
            t = np.array(await burst(server, texts, kw_of)) * 1000
            print(f"round {r}: {args.n} requests, TTFA p50 {np.percentile(t, 50):.1f} ms, "
                  f"p95 {np.percentile(t, 95):.1f} ms, max {t.max():.1f} ms", flush=True)
    finally:
        await server.stop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="VoxCPM-Kernel/VoxCPM2-Arabic-Dialects")
    ap.add_argument("--dit", default="kadirnar/VoxCPM2-Turbo-distilled-DiT:student_k2_dmd.pt")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--voice-dir", help="<name>.wav reference clips to clone (none: no cloning)")
    asyncio.run(main(ap.parse_args()))
