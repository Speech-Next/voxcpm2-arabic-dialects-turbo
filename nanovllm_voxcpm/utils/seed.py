_UINT64_MASK = (1 << 64) - 1
_SPLITMIX64_INCREMENT = 0x9E3779B97F4A7C15
_SPLITMIX64_MUL1 = 0xBF58476D1CE4E5B9
_SPLITMIX64_MUL2 = 0x94D049BB133111EB


def derive_step_seeds(seeds, seed_steps):
    """derive_step_seed for arrays (numpy uint64, wrapping like the masks above): a step's rows in one pass."""
    import numpy as np

    u = np.uint64
    seeds = np.asarray(seeds, dtype=np.uint64)
    steps = np.asarray(seed_steps, dtype=np.uint64)
    value = seeds ^ (steps + u(_SPLITMIX64_INCREMENT))
    value = value + u(_SPLITMIX64_INCREMENT)
    value = (value ^ (value >> u(30))) * u(_SPLITMIX64_MUL1)
    value = (value ^ (value >> u(27))) * u(_SPLITMIX64_MUL2)
    return value ^ (value >> u(31))


def derive_step_seed(seed: int, seed_step: int) -> int:
    """Derive a deterministic per-step seed from a request seed."""

    value = (int(seed) & _UINT64_MASK) ^ ((int(seed_step) + _SPLITMIX64_INCREMENT) & _UINT64_MASK)
    value = (value + _SPLITMIX64_INCREMENT) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * _SPLITMIX64_MUL1) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * _SPLITMIX64_MUL2) & _UINT64_MASK
    return (value ^ (value >> 31)) & _UINT64_MASK
