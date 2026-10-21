"""VoxCPM2's pure helpers: the LocDiT's Euler solver, RoPE, attention sizes, LoRA targets, the tokenizer."""

import math
from dataclasses import dataclass
from typing import List, Protocol

import torch
from transformers import PreTrainedTokenizer


# --- the LocDiT's Euler solver (CFM)

@dataclass(frozen=True, slots=True)
class EulerSolverInputs:
    x: torch.Tensor
    t_span: torch.Tensor
    mu: torch.Tensor
    cond: torch.Tensor
    cfg_value: torch.Tensor


@dataclass(frozen=True, slots=True)
class EulerSolverConfig:
    in_channels: int
    mean_mode: bool
    cfg_steps: int = -1  # evaluated steps that use CFG; -1 = all, later steps run conditional-only
    zero_init_steps: int = -1  # -1 = the original 4 % rule; 0 = evaluate every step


class EulerEstimator(Protocol):
    def __call__(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor: ...


class EulerSolverOps(Protocol):
    estimator: EulerEstimator

    def optimized_scale(self, positive_flat: torch.Tensor, negative_flat: torch.Tensor) -> torch.Tensor: ...


def compute_optimized_scale(positive_flat: torch.Tensor, negative_flat: torch.Tensor) -> torch.Tensor:
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
    squared_norm = torch.sum(negative_flat**2, dim=1, keepdim=True) + 1e-8
    return dot_product / squared_norm


def build_cfm_t_span(inference_timesteps: int, device: torch.device | None = None) -> torch.Tensor:
    t_span = torch.linspace(1, 0, inference_timesteps + 1, device=device, dtype=torch.float32)
    return t_span + (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)


def compute_zero_init_steps(t_span_len: int) -> int:
    return max(1, int(t_span_len * 0.04))


def solve_euler(inputs: EulerSolverInputs, config: EulerSolverConfig, ops: EulerSolverOps) -> torch.Tensor:
    """Integrate the CFM trajectory while reusing fully overwritten buffers."""
    x = inputs.x
    t_span = inputs.t_span
    t, dt = t_span[0], t_span[0] - t_span[1]
    zero_init_steps = (
        compute_zero_init_steps(len(t_span)) if config.zero_init_steps < 0 else config.zero_init_steps
    )
    bsz = x.size(0)
    x_in = torch.empty([2 * bsz, config.in_channels, x.size(2)], device=x.device, dtype=x.dtype)
    mu_in = torch.empty([2 * bsz, inputs.mu.size(1)], device=x.device, dtype=x.dtype)
    t_in = torch.empty([2 * bsz], device=x.device, dtype=x.dtype)
    dt_in = torch.empty([2 * bsz], device=x.device, dtype=x.dtype)
    cond_in = torch.empty([2 * bsz, config.in_channels, x.size(2)], device=x.device, dtype=x.dtype)
    evaluated = 0
    for step in range(1, len(t_span)):
        if step <= zero_init_steps:
            dphi_dt = 0.0
        elif 0 <= config.cfg_steps <= evaluated:
            # Conditional branch only: half the DiT batch, no guidance.
            evaluated += 1
            x_in[:bsz] = x
            mu_in[:bsz] = inputs.mu
            t_in[:bsz] = t.unsqueeze(0)
            dt_in[:bsz] = dt.unsqueeze(0)
            if not config.mean_mode:
                dt_in[:bsz].zero_()
            cond_in[:bsz] = inputs.cond
            dphi_dt = ops.estimator(x_in[:bsz], mu_in[:bsz], t_in[:bsz], cond_in[:bsz], dt_in[:bsz])
        else:
            evaluated += 1
            x_in[:bsz], x_in[bsz:] = x, x
            mu_in[:bsz] = inputs.mu
            mu_in[bsz:].zero_()
            t_in[:bsz] = t.unsqueeze(0)
            t_in[bsz:] = t.unsqueeze(0)
            dt_in[:bsz] = dt.unsqueeze(0)
            dt_in[bsz:] = dt.unsqueeze(0)
            if not config.mean_mode:
                dt_in.zero_()
            cond_in[:bsz], cond_in[bsz:] = inputs.cond, inputs.cond
            dphi_dt = ops.estimator(x_in, mu_in, t_in, cond_in, dt_in)
            dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
            st_star = ops.optimized_scale(dphi_dt.view(bsz, -1), cfg_dphi_dt.view(bsz, -1))
            st_star = st_star.view(bsz, *([1] * (len(dphi_dt.shape) - 1)))
            dphi_dt = cfg_dphi_dt * st_star + inputs.cfg_value[:, None, None] * (dphi_dt - cfg_dphi_dt * st_star)
        x = x - dt * dphi_dt
        t = t - dt
        sol = x
        if step < len(t_span) - 1:
            dt = t - t_span[step + 1]
    return sol


# --- RoPE and positional embeddings: RoPE and positional-embedding helpers for VoxCPM2
#
# These helpers are pure CPU math and are kept separate from the broader
# VoxCPM2 utility module so the config and CFG helpers stay under the file-size
# ceiling.

def compute_rope_scaling_factor(
    max_position_embeddings: int,
    original_max_position_embeddings: int,
) -> float:
    """Compute the LongRoPE amplitude-scaling factor."""
    scale = max_position_embeddings / original_max_position_embeddings
    return math.sqrt(1 + math.log(scale) / math.log(original_max_position_embeddings))


def build_rope_inv_freq(dim: int, base: float) -> torch.Tensor:
    """Compute inverse-frequency buffer for rotary embeddings."""
    return 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))


def build_rope_cos_sin_cache(
    seq_len: int,
    inv_freq: torch.Tensor,
    scaling_factor: float,
    short_factor: list[float],
    long_factor: list[float],
    original_max_position_embeddings: int,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the cos/sin embedding caches used by MiniCPMLongRoPE."""
    t = torch.arange(seq_len, dtype=inv_freq.dtype)
    ext_factors = (
        torch.tensor(long_factor, dtype=torch.float32)
        if seq_len > original_max_position_embeddings
        else torch.tensor(short_factor, dtype=torch.float32)
    )
    freqs = torch.mul(torch.outer(t, 1.0 / ext_factors), inv_freq.to(dtype=torch.float32))
    emb = torch.cat((freqs, freqs), dim=-1)
    cos_cache = emb.cos().to(dtype) * scaling_factor
    sin_cache = emb.sin().to(dtype) * scaling_factor
    return cos_cache, sin_cache


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary position embeddings to a tensor."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    x1, x2 = torch.chunk(x, 2, dim=-1)
    rotate_half_x = torch.cat((-x2, x1), dim=-1)
    result = x * cos.to(torch.float32) + rotate_half_x * sin.to(torch.float32)
    return result.to(orig_dtype)


def sinusoidal_pos_emb(x: torch.Tensor, dim: int, scale: float = 1000.0) -> torch.Tensor:
    """Compute sinusoidal positional embeddings."""
    if x.ndim < 1:
        x = x.unsqueeze(0)
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=x.dtype, device=x.device) * -emb)
    emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
    return torch.cat((emb.sin(), emb.cos()), dim=-1)


# --- attention sizes and LoRA targets
#
# All functions here are free of CUDA / flash-attn / Triton dependencies and
# can be exercised in a standard CPU pytest session.

# ---------------------------------------------------------------------------
# Attention size calculations
# ---------------------------------------------------------------------------


def compute_attention_sizes(
    hidden_size: int,
    total_num_heads: int,
    total_num_kv_heads: int,
    head_dim: int | None,
    tp_size: int,
) -> dict[str, int]:
    """Derive per-rank attention tensor sizes from model config values.

    This mirrors the attribute math in Cpm4Attention.__init__ so the logic
    can be unit-tested without constructing the full nn.Module.

    Args:
        hidden_size: Model hidden dimension.
        total_num_heads: Total query attention heads across all TP ranks.
        total_num_kv_heads: Total key/value heads across all TP ranks.
        head_dim: Explicit head dimension; if ``None`` derived from
            ``hidden_size // total_num_heads``.
        tp_size: Tensor-parallel world size (typically 1 for unit tests).

    Returns:
        Dict with keys ``num_heads``, ``num_kv_heads``, ``head_dim``,
        ``q_size``, ``kv_size``, ``scaling``.
    """
    num_heads = total_num_heads // tp_size
    num_kv_heads = total_num_kv_heads // tp_size
    resolved_head_dim = head_dim if head_dim is not None else hidden_size // total_num_heads
    q_size = num_heads * resolved_head_dim
    kv_size = num_kv_heads * resolved_head_dim
    scaling = resolved_head_dim**-0.5
    return {
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": resolved_head_dim,
        "q_size": q_size,
        "kv_size": kv_size,
        "scaling": scaling,
    }


# ---------------------------------------------------------------------------
# LoRA target-module parsing helpers
# ---------------------------------------------------------------------------


def parse_qkv_lora_targets(lora_targets: list[str]) -> list[str]:
    """Derive the qkv_proj LoRA sub-target list from raw target_modules_lm.

    Cpm4Attention strips the ``_proj`` suffix and keeps only q/k/v entries.

    Args:
        lora_targets: Raw ``target_modules_lm`` list (e.g. ``["q_proj", "o_proj"]``).

    Returns:
        Filtered list, e.g. ``["q", "k", "v"]`` (subset, order preserved).
    """
    return [t.replace("_proj", "") for t in lora_targets if t in ("q_proj", "k_proj", "v_proj")]


def parse_gate_up_lora_targets(lora_targets: list[str]) -> list[int]:
    """Derive the gate_up_proj LoRA index list from raw target_modules_lm.

    Cpm4MLP uses index 0 for gate_proj and index 1 for up_proj.

    Args:
        lora_targets: Raw ``target_modules_lm`` list.

    Returns:
        Sorted list of integer indices present in ``lora_targets``.
    """
    result: list[int] = []
    if "gate_proj" in lora_targets:
        result.append(0)
    if "up_proj" in lora_targets:
        result.append(1)
    return result


# ---------------------------------------------------------------------------
# VoxCPM2Model config-derivation helpers
# ---------------------------------------------------------------------------


# --- the tokenizer

def mask_multichar_chinese_tokens(tokenizer: PreTrainedTokenizer):
    multichar_tokens = {
        token for token in tokenizer.vocab.keys() if len(token) >= 2 and all("\u4e00" <= c <= "\u9fff" for c in token)
    }

    class CharTokenizerWrapper:
        def __init__(self, base_tokenizer: PreTrainedTokenizer) -> None:
            self.tokenizer = base_tokenizer
            self.multichar_tokens = multichar_tokens

        def tokenize(self, text: str, **kwargs) -> List[str]:
            if not isinstance(text, str):
                raise TypeError(f"Expected string input, got {type(text)}")

            tokens = self.tokenizer.tokenize(text, **kwargs)
            processed = []
            for token in tokens:
                clean_token = token.replace("▁", "")
                if clean_token in self.multichar_tokens:
                    processed.extend(list(clean_token))
                else:
                    processed.append(token)
            return processed

        def __call__(self, text: str, **kwargs) -> List[int]:
            try:
                tokens = self.tokenize(text, **kwargs)
                return self.tokenizer.convert_tokens_to_ids(tokens)
            except Exception as e:
                raise ValueError(f"Tokenization failed: {str(e)}") from e

    return CharTokenizerWrapper(tokenizer)
