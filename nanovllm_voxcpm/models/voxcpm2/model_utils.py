"""Pure CPU-testable utilities extracted from voxcpm2/model.py.

All functions here are free of CUDA / flash-attn / Triton dependencies and
can be exercised in a standard CPU pytest session.
"""

from nanovllm_voxcpm.models.voxcpm2 import model_utils_cfm as _cfm
from nanovllm_voxcpm.models.voxcpm2 import model_utils_rope as _rope

EulerSolverConfig = _cfm.EulerSolverConfig
EulerSolverInputs = _cfm.EulerSolverInputs
apply_rotary_emb = _rope.apply_rotary_emb
build_cfm_t_span = _cfm.build_cfm_t_span
build_rope_cos_sin_cache = _rope.build_rope_cos_sin_cache
build_rope_inv_freq = _rope.build_rope_inv_freq
compute_optimized_scale = _cfm.compute_optimized_scale
compute_rope_scaling_factor = _rope.compute_rope_scaling_factor
compute_zero_init_steps = _cfm.compute_zero_init_steps
sinusoidal_pos_emb = _rope.sinusoidal_pos_emb
solve_euler = _cfm.solve_euler

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


