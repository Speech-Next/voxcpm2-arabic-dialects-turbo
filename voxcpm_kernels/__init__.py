"""Custom kernels for fast batched VoxCPM2 inference."""

from .int8_linear import W8A8Linear
from .noise import seeded_randn
from .snake import Snake, snake_reference
from .vae_stream import SlotStreamingVAEDecoder

__all__ = ["Snake", "SlotStreamingVAEDecoder", "W8A8Linear", "seeded_randn", "snake_reference"]
__version__ = "0.1.0"
