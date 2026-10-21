"""Clone a voice from a reference clip and speak Arabic in one of the dialects.

    python examples/voice_clone.py --voice reference.wav --dialect najdi --text "هلا والله، كيف حالك اليوم؟"

The engine is the routed one (RoutedDialectTTS: each dialect on its adapter or on teacher-v3); --route
DIALECT=ADAPTER or DIALECT=teacher changes its table (--route najdi=najdi: Najdi on its adapter); --no-route: the
unrouted engine (DialectTTS, the model's adapters only). --voice-text "<the reference's transcript>" clones by
continuing the reference (prompt_ref) instead of from its timbre alone (ref). --model / --dit: local paths or Hugging
Face ids (defaults: the dialect model and the distilled LocDiT). --dit-cache light|medium: the original LocDiT with
block caching instead of the distilled one.
"""

import argparse
import asyncio

import soundfile as sf

from nanovllm_voxcpm import DialectTTS, RoutedDialectTTS
from nanovllm_voxcpm.dialect_routing import DEFAULT_TAGGED_MODEL, parse_routes
from nanovllm_voxcpm.dialect_tts import DEFAULT_DIT, DEFAULT_MODEL


async def main(args):
    dit = None if args.dit_cache or args.dit == "none" else args.dit
    kw = dict(dit_cache=args.dit_cache, max_concurrency=16)
    tts = await (DialectTTS.load(args.model, dit, **kw) if args.no_route
                 else RoutedDialectTTS.load(args.model, args.tagged_model, dit, routes=parse_routes(args.route), **kw))
    async with tts:
        await tts.add_voice("speaker", args.voice, text=args.voice_text)
        audio = await tts.synthesize(args.text, dialect=args.dialect, voice="speaker")
        sf.write(args.out, audio, tts.sample_rate)
        print(f"{args.out}: {len(audio) / tts.sample_rate:.2f} s, dialect {args.dialect}, voice of {args.voice}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", required=True, help="reference clip (wav) of the voice to clone")
    ap.add_argument("--voice-text", help="its transcript (optional)")
    ap.add_argument("--dialect", required=True, help="e.g. najdi, gulf, bahraini, egyptian, msa (routed: the 17 "
                                                     "dialects of teacher-v3, its aliases and hindi_english)")
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", default="out.wav")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tagged-model", default=DEFAULT_TAGGED_MODEL)
    ap.add_argument("--dit", default=DEFAULT_DIT, help="'none': the model's own LocDiT")
    ap.add_argument("--dit-cache", choices=["light", "medium"])
    ap.add_argument("--route", action="append", metavar="DIALECT=ADAPTER",
                    help="speak DIALECT with that adapter, or with teacher-v3 (DIALECT=teacher), instead of the "
                         "default table (repeatable)")
    ap.add_argument("--no-route", action="store_true", help="the unrouted engine (the model's adapters only)")
    asyncio.run(main(ap.parse_args()))
