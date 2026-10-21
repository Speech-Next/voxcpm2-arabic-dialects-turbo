"""The servers' start-up warm-up (#39): bursts of the sizes a server meets, texts of three lengths, so that Triton
compiles its kernel variants (tile sizes per row count, the adapters' segmented GEMMs, the voice prefix's paged
attention) before the first client, not while the first streams play."""

import asyncio
from collections.abc import Awaitable, Callable

BURSTS = (1, 10, 40, 100, 150)
TEXTS = (
    "أهلاً بيك، إزيك النهارده؟",
    "النهارده الجو حلو قوي، تعالى نتمشى شوية على الكورنيش ونشرب حاجة ساقعة.",
    "لو عندك وقت بكره الصبح، ممكن نعدي على المكتب سوا ونخلص الورق اللي فاضل، وبعدين نروح نتغدى في المطعم اللي "
    "جنب المحطة.",
)


async def warm_up(run: Callable[[int, str], Awaitable[object]]) -> None:
    """run(i, text): request i of a burst, done when its audio is."""
    for n in BURSTS:
        await asyncio.gather(*(run(i, TEXTS[i % len(TEXTS)]) for i in range(n)))
