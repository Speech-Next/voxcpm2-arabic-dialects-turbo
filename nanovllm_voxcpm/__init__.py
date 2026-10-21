"""DialectTTS, RoutedDialectTTS and VoxCPM load on first use (PEP 562): ``import nanovllm_voxcpm.arabic_text`` needs
no torch or flash-attn (the training repo and the tests import the text normalizer alone)."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nanovllm_voxcpm.dialect_routing import RoutedDialectTTS
    from nanovllm_voxcpm.dialect_tts import DialectTTS
    from nanovllm_voxcpm.model.llm import VoxCPM

try:
    from nanovllm_voxcpm._version import version as __version__
except Exception:
    try:
        import importlib.metadata

        __version__ = importlib.metadata.version("voxcpm2-arabic-dialects-turbo")
    except Exception:
        __version__ = "0.0.0"

_LAZY = {"DialectTTS": "nanovllm_voxcpm.dialect_tts", "RoutedDialectTTS": "nanovllm_voxcpm.dialect_routing",
         "VoxCPM": "nanovllm_voxcpm.model.llm"}

__all__ = [
    "DialectTTS",
    "RoutedDialectTTS",
    "VoxCPM",
    "__version__",
]


def __getattr__(name: str):
    if name in _LAZY:
        value = getattr(importlib.import_module(_LAZY[name]), name)
        globals()[name] = value  # the next lookup skips __getattr__
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
