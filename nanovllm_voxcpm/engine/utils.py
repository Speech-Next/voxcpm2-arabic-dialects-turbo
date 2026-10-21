"""The engine's helpers: the forward context, the weight loader, NANOVLLM_PROFILE step records, seeds."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from glob import glob
from typing import Any, Callable, cast, Protocol

import torch
import torch.distributed as dist
from safetensors import safe_open
from torch import nn


# --- the forward context

LM_LORA_DOMAIN = "lm_domain"
PROJ_LORA_DOMAIN = "proj_domain"
DIT_LORA_DOMAIN = "dit_domain"


@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    # a prefill step whose sequences all start their stream (no resumed one, no decode riding along): the LocDiT
    # solves with the first patch's settings (TurboConfig.dit_first_t_span)
    first_patch: bool = True


@dataclass
class LoRAContext:
    token_to_slot: torch.Tensor | None = None
    token_indices_sorted_by_slot: torch.Tensor | None = None
    active_slot_ids: torch.Tensor | None = None
    num_tokens_per_slot: torch.Tensor | None = None
    slot_start_offsets: torch.Tensor | None = None
    no_lora_flag: bool = True
    num_active_loras: int = 0
    host_token_to_slot: list[int] | None = None
    host_token_indices_sorted_by_slot: list[int] | None = None
    host_active_slot_ids: list[int] | None = None
    host_num_tokens_per_slot: list[int] | None = None
    # Cached `LoRAMetadata` view of this context. Layers all build an identical
    # metadata object from the same context per step; we materialize it lazily
    # once per context instance and reuse across every LoRA-enabled layer in
    # the model. Cleared by ``set_lora_context`` (i.e. fresh each step).
    _cached_metadata: object | None = field(default=None, repr=False, compare=False)

    @property
    def token_count(self) -> int:
        if self.token_to_slot is not None:
            return self.token_to_slot.numel()
        return len(self.host_token_to_slot or ())


@dataclass
class LoRAContexts:
    domains: dict[str, LoRAContext] = field(default_factory=dict)


_CONTEXT = Context()
_LORA_CONTEXTS = LoRAContexts()


def get_context():
    return _CONTEXT


def set_context(
    is_prefill,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=None,
    context_lens=None,
    block_tables=None,
):
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        slot_mapping,
        context_lens,
        block_tables,
    )


def get_lora_context(domain: str = LM_LORA_DOMAIN):
    return _LORA_CONTEXTS.domains.get(domain, LoRAContext())


def build_lora_context_from_token_to_slot(
    token_to_slot: torch.Tensor | None,
) -> LoRAContext:
    if token_to_slot is None or token_to_slot.numel() == 0:
        return LoRAContext(
            token_to_slot=token_to_slot,
            no_lora_flag=True,
            num_active_loras=0,
        )

    token_to_slot = token_to_slot.to(dtype=torch.int32)
    active_slot_ids = torch.unique(token_to_slot[token_to_slot >= 0], sorted=True).to(dtype=torch.int32)
    no_lora_flag = active_slot_ids.numel() == 0
    if no_lora_flag:
        return LoRAContext(
            token_to_slot=token_to_slot,
            no_lora_flag=True,
            num_active_loras=0,
        )

    token_indices_by_slot = []
    token_counts = []
    for slot_id in active_slot_ids.tolist():
        indices = torch.nonzero(token_to_slot == slot_id, as_tuple=False).flatten().to(dtype=torch.int32)
        token_indices_by_slot.append(indices)
        token_counts.append(indices.numel())

    num_tokens_per_slot = torch.tensor(token_counts, dtype=torch.int32, device=token_to_slot.device)
    slot_start_offsets = torch.zeros(active_slot_ids.numel() + 1, dtype=torch.int32, device=token_to_slot.device)
    slot_start_offsets[1:] = torch.cumsum(num_tokens_per_slot, dim=0)
    token_indices_sorted_by_slot = torch.cat(token_indices_by_slot).to(device=token_to_slot.device, dtype=torch.int32)
    return LoRAContext(
        token_to_slot=token_to_slot,
        token_indices_sorted_by_slot=token_indices_sorted_by_slot,
        active_slot_ids=active_slot_ids,
        num_tokens_per_slot=num_tokens_per_slot,
        slot_start_offsets=slot_start_offsets,
        no_lora_flag=False,
        num_active_loras=active_slot_ids.numel(),
    )


def set_lora_context(
    lora_context: LoRAContext,
    *,
    domain: str = LM_LORA_DOMAIN,
):
    # Defensively invalidate any per-step metadata view cached by layers in a
    # prior step. Engines in this repo build a fresh ``LoRAContext`` per step,
    # so this is normally a no-op, but it keeps ``_cached_metadata`` honest if
    # a caller mutates and reuses the same instance across steps.
    lora_context._cached_metadata = None
    _LORA_CONTEXTS.domains[domain] = lora_context


def reset_context():
    global _CONTEXT
    _CONTEXT = Context()


def reset_lora_context(domain: str | None = None):
    global _LORA_CONTEXTS
    if domain is None:
        _LORA_CONTEXTS = LoRAContexts()
    else:
        _LORA_CONTEXTS.domains.pop(domain, None)


def reset_all_contexts():
    reset_context()
    reset_lora_context()


# --- tensor-parallel ranks

def get_tp_world_size() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_tp_rank() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


# --- weight loading

ShardId = str | int

try:
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_state_dict_into(model: nn.Module, state_dict: dict[str, torch.Tensor], prefix: str = "") -> int:
    """Load an unpacked (q/k/v, gate/up) state dict into a packed nanovllm model; returns #tensors."""
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    count = 0
    for weight_name, tensor in state_dict.items():
        if not weight_name.startswith(prefix):
            continue
        for k in packed_modules_mapping:
            if k in weight_name:
                v, shard_id = packed_modules_mapping[k]
                param = model.get_parameter(weight_name.replace(k, v))
                getattr(param, "weight_loader")(param, tensor, shard_id)
                break
        else:
            param = model.get_parameter(weight_name)
            getattr(param, "weight_loader", default_weight_loader)(param, tensor)
        count += 1
    return count


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

    visited_param_names = set()
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        visited_param_names.add(param_name)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
                    visited_param_names.add(weight_name)

    missing_param_names = []
    for name, _ in model.named_parameters():
        if name not in visited_param_names:
            # Skip LoRA parameters (they are optional)
            if "lora_" in name:
                continue
            missing_param_names.append(name)

    if missing_param_names:
        raise ValueError(f"Missing parameters: {missing_param_names}")


# --- step records (NANOVLLM_PROFILE): Opt-in per-step timing
#
# Set ``NANOVLLM_PROFILE=/path/to/steps.jsonl`` to append one JSON record per
# engine step with CUDA-synchronized stage durations (ms). Unset, every call is a
# no-op and nothing is synchronized. ``NANOVLLM_PROFILE_SYNC=0`` keeps the records but skips the
# synchronizations (the engine then runs exactly as unprofiled; stage times become CPU launch times, while
# step boundaries stay accurate because every step ends with a device-to-host copy).
# ``NANOVLLM_PROFILE_GPU=1`` (with ``SYNC=0``) also records a CUDA event at the engine step's entry (begin_step) and at
# every mark: ``gpu_<mark>`` is the GPU's time from the step's entry to that point (when the GPU got there: the
# first model kernel starts right after ``gpu_prep``), read once the step has synchronized at its end.

_PATH = os.environ.get("NANOVLLM_PROFILE")
_SYNC = os.environ.get("NANOVLLM_PROFILE_SYNC", "1") != "0"
_GPU = bool(_PATH) and os.environ.get("NANOVLLM_PROFILE_GPU", "0") != "0"
_current: "StepProfile | None" = None
_begin: tuple | None = None  # (host perf_counter, CUDA event) at the engine step's entry


def begin_step() -> None:
    """The engine step's entry (before scheduling): the reference of the step record's t_begin and gpu_* times."""
    global _begin
    if not StepProfile.enabled:
        return
    ev = None
    if _GPU and torch.cuda.is_available():
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
    _begin = (time.perf_counter(), ev)


class StepProfile:
    enabled = bool(_PATH)

    def __init__(self, **meta):
        self.rec = dict(meta) if self.enabled else None
        self._t = None
        self._events: list = []  # (mark, CUDA event)

    def start(self) -> "StepProfile":
        global _current
        if self.enabled:
            if _SYNC:
                torch.cuda.synchronize()
            self._t = time.perf_counter()
            self.rec["t"] = self._t  # perf_counter at the start, comparable with the server's step records
            if _begin is not None:
                self.rec["t_begin"] = _begin[0]
            _current = self
        return self

    def mark(self, name: str) -> None:
        """Record the time since the previous mark under ``name`` (accumulates on repeats)."""
        if not self.enabled or self._t is None:
            return
        if _SYNC:
            torch.cuda.synchronize()
        elif _GPU and _begin is not None and _begin[1] is not None:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._events.append((name, ev))
        now = time.perf_counter()
        self.rec[name] = self.rec.get(name, 0.0) + (now - self._t) * 1000
        self._t = now

    def write(self) -> None:
        global _current, _begin
        if not self.enabled:
            return
        _current = None
        if self._events and _begin is not None:
            self._events[-1][1].synchronize()
            for name, ev in self._events:  # the last time the GPU reached a repeated mark
                self.rec["gpu_" + name] = _begin[1].elapsed_time(ev)
        _begin = None
        with open(_PATH, "a") as f:
            f.write(json.dumps(self.rec) + "\n")


def mark(name: str) -> None:
    """Mark on the step currently being profiled, if any."""
    if _current is not None:
        _current.mark(name)


_host: dict = {}  # host time per engine-step phase (ms), the server's record takes it (host_take)
_host_t = [0.0]


def host_begin() -> None:
    """NANOVLLM_PROFILE: the host phases of an engine step are timed from here (host_mark)."""
    if StepProfile.enabled:
        _host_t[0] = time.perf_counter()


def host_mark(name: str) -> None:
    if StepProfile.enabled:
        now = time.perf_counter()
        _host["h_" + name] = _host.get("h_" + name, 0.0) + (now - _host_t[0]) * 1000
        _host_t[0] = now


def host_take() -> dict:
    out = dict(_host)
    _host.clear()
    return out


def log(**rec) -> None:
    if StepProfile.enabled:
        with open(_PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")


# --- seeds

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


# --- parameter helpers

class _ParamWithWeightLoader(Protocol):
    # The codebase attaches a dynamic attribute used by weight-loading code.
    weight_loader: Callable[..., Any]


def set_weight_loader(param: nn.Parameter, loader: Callable[..., Any]) -> None:
    cast(_ParamWithWeightLoader, param).weight_loader = loader
