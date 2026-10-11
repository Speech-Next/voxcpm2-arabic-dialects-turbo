"""Clone a voice from a reference clip and speak Arabic in one of the dialects.

    python examples/voice_clone.py --voice reference.wav --dialect najdi --text "هلا والله، كيف حالك اليوم؟"

--voice-text "<the reference's transcript>" clones by continuing the reference (prompt_ref) instead of from its
timbre alone (ref). --model / --dit: local paths or Hugging Face ids (defaults: the dialect model and the distilled
LocDiT).
"""

import argparse
import asyncio

import soundfile as sf

from nanovllm_voxcpm import DialectTTS
from nanovllm_voxcpm.dialect_tts import DEFAULT_DIT, DEFAULT_MODEL


async def main(args):
    async with await DialectTTS.load(args.model, args.dit, max_concurrency=16) as tts:
        await tts.add_voice("speaker", args.voice, text=args.voice_text)
        audio = await tts.synthesize(args.text, dialect=args.dialect, voice="speaker")
        sf.write(args.out, audio, tts.sample_rate)
        print(f"{args.out}: {len(audio) / tts.sample_rate:.2f} s, dialect {args.dialect}, voice of {args.voice}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", required=True, help="reference clip (wav) of the voice to clone")
    ap.add_argument("--voice-text", help="its transcript (optional)")
    ap.add_argument("--dialect", required=True, help="ab, egyptian, emirati, hejazi, hindi_english, iraqi, jordanian, "
                                                     "najdi, palestinian, qatari, saudi")
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", default="out.wav")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dit", default=DEFAULT_DIT)
    asyncio.run(main(ap.parse_args()))
