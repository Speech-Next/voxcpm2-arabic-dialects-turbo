"""nanovllm_voxcpm.engine.model_runner

This module defines the GPU execution abstraction used by the engine.

The high-level runtime separates concerns:
- :mod:`nanovllm_voxcpm.engine.scheduler` decides *what to run* (which sequences)
  and manages KV-cache block allocation.
- :mod:`nanovllm_voxcpm.engine.llm_engine` orchestrates the step loop and
  converts between request objects and runner tasks.
- This module executes the model forward pass on GPU(s) given a batch of
  lightweight :class:`RunnerTask` objects.

RunnerTask
----------
:class:`RunnerTask` is a minimal, picklable view of a sequence needed to build
GPU inputs:
- ``block_table``: physical KV-cache block ids for this request.
- ``seq_length``: logical length (prompt + generated tokens so far).
- ``num_cached_tokens``: cached prefix tokens (prefill only).
- ``custom_payload``: model-specific inputs (e.g. token tensors, sampling params).

BaseModelRunner
---------------
:class:`BaseModelRunner` owns the actual ``torch.nn.Module`` and the KV-cache
tensors stored inside causal :class:`~nanovllm_voxcpm.layers.attention.Attention`
modules. Key responsibilities:

- Initialize NCCL process group and set the CUDA device for the current rank.
- Load and warm up the model (used to measure peak memory).
- Allocate the KV-cache block pool based on available GPU memory and
  ``gpu_memory_utilization``.
- Prepare attention metadata ("context") for flash-attn kernels via
  :func:`nanovllm_voxcpm.utils.context.set_context`.
  * Prefill context supports prefix caching by distinguishing query length
    (new tokens) vs key length (full context).
  * Decode context writes one token per sequence into the KV cache.
- Optional CUDA Graph capture for decode to reduce launch overhead
  (disabled with ``enforce_eager``).

Multi-GPU execution model
-------------------------
Tensor-parallel ranks are spawned as separate processes. Rank 0 acts as the
"driver" and broadcasts method calls to other ranks through shared memory +
``multiprocessing.Event``. Non-zero ranks run :meth:`loop`, which blocks on an
event, reads the serialized method call, and executes it.

Model-specific runners
----------------------
Concrete model families subclass :class:`BaseModelRunner` and implement:
- model construction / weight loading (:meth:`init_model`)
- building inputs/outputs for warmup/graph capture (:meth:`make_dummy_inputs`,
  :meth:`make_dummy_outputs`)
- the actual per-step execution logic (:meth:`run`) which typically:
  1) builds tensors from ``RunnerTask.custom_payload``
  2) calls :meth:`prepare_prefill_context` or :meth:`prepare_decode_context`
  3) runs the model via :meth:`run_model`
  4) returns Python-friendly outputs for engine postprocessing.

Concrete example: VoxCPM
------------------------
``nanovllm_voxcpm/models/voxcpm2/runner.py`` shows a typical implementation:

- Prefill: the engine slices away ``num_cached_tokens`` and sends the remaining
  prompt segment (text tokens + audio features + masks) to the runner.
- Decode: the engine sends only the last step (length 1) and sets
  ``RunnerTask.num_cached_tokens = seq_length - 1`` so the runner builds a
  decode context (query length 1, key length = full context).
- The runner concatenates per-sequence numpy arrays into a packed token-major
  batch, runs the model, then converts outputs back to numpy.
- Besides model outputs (e.g. ``latents`` and ``stop_flag``), VoxCPMRunner also
  decodes the generated latents into waveform chunks via an AudioVAE and returns
  them to be streamed.
"""

import math
import os
import sys
import pickle
import tempfile
import numpy as np
import torch

from nanovllm_voxcpm.utils import profile as _profile
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm_voxcpm.config import Config
from nanovllm_voxcpm.engine.lora_manager import (
    LoRAModelPayload,
    LoRARuntime,
    build_lora_context_from_batch_plan,
    build_lora_context_from_slot_list,
    materialize_lora_context,
    stage_lora_context,
)
from nanovllm_voxcpm.layers.attention import Attention
from nanovllm_voxcpm.layers.lora import iter_lora_modules
from nanovllm_voxcpm.lora import is_available as is_lora_available
from nanovllm_voxcpm.utils.context import (
    DIT_LORA_DOMAIN,
    LM_LORA_DOMAIN,
    PROJ_LORA_DOMAIN,
    LoRAContext,
    build_lora_context_from_token_to_slot,
    get_context,
    get_lora_context,
    reset_all_contexts,
    set_context,
    set_lora_context,
)
from typing import Generic, TypeVar

PlayloadType = TypeVar("PlayloadType")
LORA_DOMAINS = (LM_LORA_DOMAIN, PROJ_LORA_DOMAIN, DIT_LORA_DOMAIN)


def select_lora_payload_for_rank(payload, rank: int):
    if isinstance(payload, (list, tuple)):
        if rank >= len(payload):
            raise ValueError(f"Missing rank-local LoRA payload for rank {rank}")
        return payload[rank]
    return payload


_RPC_FILE_SENTINEL = "__rpc_file__"
_NUM_KVCACHE_BLOCKS_ENV = "NANOVLLM_SERVERPOOL_NUM_KVCACHE_BLOCKS"


def _env_int(name: str) -> int | None:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value.strip() == "":
        return None
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc


class RunnerTask(Generic[PlayloadType]):
    def __init__(
        self,
        block_table: list[int],
        seq_length: int,
        num_cached_tokens: int,
        block_size: int,
        custom_payload: PlayloadType = None,
        adapter_id: int | None = None,
        seq_id: str | None = None,
    ):
        self.block_table = block_table
        self.seq_length = seq_length
        self.num_cached_tokens = num_cached_tokens
        self.custom_payload = custom_payload
        self.block_size = block_size
        self.adapter_id = adapter_id
        self.seq_id = seq_id

    @property
    def num_blocks(self):
        return (self.seq_length + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.seq_length - (self.num_blocks - 1) * self.block_size


def cut_inputs(inputs, bs):
    return {k: v[:bs] for k, v in inputs.items()}


def assign_outputs(inputs, outputs, bs):
    for k in outputs.keys():
        if k not in inputs:
            raise KeyError(f"Input {k} is required")
        outputs[k][:bs] = inputs[k]


def expand_dit_lora_slots(
    sample_to_slot: list[int],
    sequence_length: int,
    cfg_branches: int,
    padded_batch_size: int | None = None,
) -> list[int]:
    """Expand sample-level slots in the branch-major order used by CFG.

    Diffusion constructs its estimator batch as all positive samples followed
    by all negative samples. CUDA graph batch padding must therefore be
    inserted at the end of each CFG branch rather than once at the end.
    """
    batch_size = len(sample_to_slot)
    padded_batch_size = batch_size if padded_batch_size is None else padded_batch_size
    if padded_batch_size < batch_size:
        raise ValueError("padded_batch_size cannot be smaller than the real batch size")
    if padded_batch_size * sequence_length <= 128:  # a few rows: the comprehension beats numpy's fixed cost
        padded = list(sample_to_slot) + [-1] * (padded_batch_size - batch_size)
        return [slot for _ in range(cfg_branches) for slot in padded for _ in range(sequence_length)]
    padded_slots = np.full(padded_batch_size, -1, dtype=np.int64)
    padded_slots[:batch_size] = sample_to_slot
    # each slot sequence_length times, the padded batch once per CFG branch (numpy: ~2300 rows at 200 requests)
    return np.tile(np.repeat(padded_slots, max(sequence_length, 0)), max(cfg_branches, 0)).tolist()


_TORCH_DTYPES = {np.dtype(d): t for d, t in (
    (np.int64, torch.int64), (np.int32, torch.int32), (np.int16, torch.int16), (np.int8, torch.int8),
    (np.uint8, torch.uint8), (np.bool_, torch.bool), (np.float64, torch.float64), (np.float32, torch.float32),
    (np.float16, torch.float16))}


class PinnedStaging:
    """One page-locked buffer that every prefill step's HostPack writes into (grown when a step needs more), in
    place of a fresh page-locked allocation per step: the caching host allocator hands a block out again only once
    nothing views it, and the streams' audio rows (views of their step's page-locked copies) keep blocks alive for
    whole requests; a burst's head step's upload took 1.7 instead of 0.5 ms in some bursts at 200 at once. Before
    the buffer is written again, the previous upload's copy out of it must be done (queued a step earlier: the wait
    is a check)."""

    def __init__(self) -> None:
        self.buf: torch.Tensor | None = None
        self.copied: torch.cuda.Event | None = None

    def get(self, nbytes: int) -> torch.Tensor:
        if self.copied is not None:
            self.copied.synchronize()
        if self.buf is None or self.buf.numel() < nbytes:
            grow = 2 * self.buf.numel() if self.buf is not None else 0
            self.buf = torch.empty(max(nbytes, grow), dtype=torch.uint8, device="cpu", pin_memory=True)
        return self.buf[:nbytes]

    def copied_out(self) -> None:
        """The upload's copy is queued: the buffer is not written again before it is done."""
        ev = torch.cuda.Event()
        ev.record()
        self.copied = ev


class HostPack:
    """A step's host inputs staged in one page-locked buffer and copied to the GPU in one transfer. Every small H2D
    copy costs a pinned allocation, a device allocation and a cudaMemcpyAsync: a prefill step made ~17 of them
    before its first kernel. add() / add_concat() return the index of the tensor upload() hands back: a view into
    the one device buffer with the array's dtype, shape and values. Parts start at multiples of 256 bytes, as fresh
    allocations do, so kernels that specialize on pointer alignment see the alignment they saw before."""

    ALIGN = 256

    def __init__(self, device="cuda", staging: PinnedStaging | None = None):
        self.device = torch.device(device)
        self.parts: list[tuple[tuple, np.dtype, object]] = []  # (shape, dtype, array or arrays to concatenate)
        self.staging = staging  # (a step's page-locked buffer: PinnedStaging, else a fresh allocation)

    def add(self, array) -> int:
        a = np.asarray(array)  # (np.ascontiguousarray would make a 0-d array 1-d)
        self.parts.append((a.shape, a.dtype, a))
        return len(self.parts) - 1

    def add_concat(self, arrays: list[np.ndarray]) -> int:
        """np.concatenate(arrays) (the first array's dtype), written straight into the staging buffer."""
        first = arrays[0]
        self.parts.append(((sum(a.shape[0] for a in arrays), *first.shape[1:]), first.dtype, arrays))
        return len(self.parts) - 1

    def upload(self) -> list[torch.Tensor]:
        counts = [math.prod(shape) for shape, _, _ in self.parts]
        offsets, total = [], 0
        for (_, dtype, _), n in zip(self.parts, counts):
            offsets.append(total)
            total += -(-n * dtype.itemsize // self.ALIGN) * self.ALIGN
        total = max(total, self.ALIGN)
        on_gpu = self.device.type == "cuda"
        staged = on_gpu and self.staging is not None and not torch.cuda.is_current_stream_capturing()
        host = self.staging.get(total) if staged else torch.empty(total, dtype=torch.uint8, device="cpu",
                                                                  pin_memory=on_gpu)
        buf = host.numpy()
        typed_host: dict = {}  # the buffer viewed once per dtype (offsets are multiples of every item size)
        for (shape, dtype, src), off, n in zip(self.parts, offsets, counts):
            view = typed_host.get(dtype)
            if view is None:
                view = typed_host[dtype] = buf.view(dtype)
            dst = view[off // dtype.itemsize : off // dtype.itemsize + n]
            if len(shape) != 1:
                dst = dst.reshape(shape)
            if isinstance(src, list):
                np.concatenate(src, axis=0, out=dst)
            else:
                dst[...] = src
        dev = host.to(self.device, non_blocking=True) if on_gpu else host
        if staged:
            self.staging.copied_out()
        typed: dict = {}
        out = []
        for (shape, dtype, _), off, n in zip(self.parts, offsets, counts):
            t = typed.get(dtype)
            if t is None:
                t = typed[dtype] = dev.view(_TORCH_DTYPES[dtype])
            v = t[off // dtype.itemsize : off // dtype.itemsize + n]
            out.append(v if len(shape) == 1 else v.view(shape))
        return out


def _clear_lora_slot_modules(modules, slot_id: int, module_names: list[str] | None = None) -> None:
    """Zero out LoRA weights for ``slot_id`` across ``modules``.

    ``module_names`` (when provided) restricts the walk to just the modules
    previously written into this slot. This avoids iterating the entire model
    graph — and issuing dozens of tiny ``zero_()`` kernels per slot admission —
    for the common case where each LoRA only populates a small subset of
    modules. Passing ``None`` preserves the legacy "clear everything" behavior
    (used by tests).
    """
    if module_names is None:
        iterable = modules.values()
    else:
        iterable = (modules[name] for name in module_names if name in modules)
    for module in iterable:
        clear_slot_lora = getattr(module, "clear_slot_lora", None)
        if clear_slot_lora is not None:
            clear_slot_lora(slot_id)


class BaseModelRunner:
    dit_lora_seq_len_offset = 0
    cfg_branches = 2
    patch_size: int

    model: torch.nn.Module

    def __init__(
        self,
        config: Config,
        rank: int,
        device_idx: int,
        distributed_port: int | None,
        event: Event | list[Event],
    ):
        self._config = config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        if sys.platform == "win32":
            import torch._dynamo as dynamo_mod

            dynamo_mod.config.disable = True
            self.enforce_eager = True
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.max_lora_rank = max(1, getattr(config.lora_config, "max_lora_rank", 1) if config.lora_config else 1)
        self.max_loras = max(0, getattr(config.lora_config, "max_loras", 0) if config.lora_config else 0)
        self.lora_runtime = LoRARuntime(max_loras=self.max_loras, max_lora_rank=self.max_lora_rank)
        # Lazy cache of ``dict(self.model.named_modules())`` — walking the full
        # VoxCPM module tree is surprisingly expensive (~ms per call on a
        # real-sized model) and was called on every LoRA slot admission and
        # validation. Populated by ``_lora_model_modules()`` on first use.
        self._lora_model_modules_cache: dict[str, torch.nn.Module] | None = None
        # Track which module names each GPU slot currently holds LoRA weights
        # for, so evict/clear can skip the no-op "zero already-zero weights"
        # walk across the entire model graph.
        self._lora_slot_modules: dict[int, list[str]] = {}

        if self.world_size > 1:
            assert distributed_port is not None
            if sys.platform == "win32":
                raise NotImplementedError(
                    "Tensor parallelism (world_size > 1) is currently not supported on Windows "
                    "because NCCL is unavailable and Gloo lacks direct CUDA tensor collectives support. "
                    "Please run with a single GPU on Windows or use a Linux environment."
                )
            else:
                dist.init_process_group(
                    "nccl",
                    f"tcp://localhost:{distributed_port}",
                    world_size=self.world_size,
                    rank=rank,
                )
        torch.cuda.set_device(device_idx)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.dtype)
        torch.set_default_device("cuda")
        self.init_model(self._config.model_config, self._config.model)
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        # None, not "cpu": a "cpu" default still installs a DeviceContext torch-function mode that runs Python on every
        # torch call (~5 ms of host time per prefill step of 200 sequences)
        torch.set_default_device(None)
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name=f"nanovllm-{distributed_port}", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name=f"nanovllm-{distributed_port}")
                self.loop()

    @property
    def dtype(self) -> torch.dtype:
        raise NotImplementedError()

    def init_model(self, model_config, model_path: str):
        raise NotImplementedError()

    def make_dummy_inputs(self, batch_size: int, length: int) -> torch.Tensor:
        raise NotImplementedError()

    def make_dummy_outputs(
        self,
        batch_size: int,
    ) -> torch.Tensor:
        raise NotImplementedError()

    def run(self, seqs: list[RunnerTask], is_prefill: bool, first_patch: bool = True):
        raise NotImplementedError()

    def _dit_lora_sequence_length(self) -> int:
        lora_config = getattr(self, "lora_config", None)
        if not (lora_config and getattr(lora_config, "enable_dit", False)):
            return 0
        return self.dit_lora_seq_len_offset + 2 * self.patch_size

    def _dit_lora_rows_per_sample(self) -> int:
        return self.cfg_branches * self._dit_lora_sequence_length()

    def _build_lora_contexts(
        self,
        seqs: list[RunnerTask],
        token_counts: list[int],
        *,
        materialize_domains: set[str] | None = None,
    ) -> dict[str, LoRAContext]:
        if materialize_domains is None:
            materialize_domains = set(LORA_DOMAINS)
        adapter_ids = [seq.adapter_id for seq in seqs]
        if not any(adapter_id is not None for adapter_id in adapter_ids):
            # No active LoRA anywhere in this batch. Build just the LM
            # ``token_to_slot=[-1,...]`` tensor; the PROJ and DIT domains can
            # share the "all -1" sentinel since every sample gets slot=-1.
            # Layers short-circuit on ``no_lora_flag=True`` before reading any
            # of the other fields, so PROJ/DIT don't even need a device
            # tensor.
            empty_ctx = LoRAContext(no_lora_flag=True, num_active_loras=0)
            return {
                LM_LORA_DOMAIN: build_lora_context_from_slot_list(
                    [-1] * sum(token_counts),
                    materialize_device=LM_LORA_DOMAIN in materialize_domains,
                ),
                PROJ_LORA_DOMAIN: empty_ctx,
                DIT_LORA_DOMAIN: empty_ctx,
            }

        plan = self.lora_runtime.build_batch_plan(adapter_ids, token_counts, self._load_lora_slot)
        sample_to_slot = [
            plan.adapter_to_slot.get(adapter_id, -1) if adapter_id is not None else -1 for adapter_id in adapter_ids
        ]
        padded_batch_size = len(sample_to_slot)
        if not getattr(self, "enforce_eager", True) and hasattr(self, "graph_bs"):
            padded_batch_size = next(
                (graph_bs for graph_bs in self.graph_bs if graph_bs >= len(sample_to_slot)),
                len(sample_to_slot),
            )
        dit_token_to_slot = expand_dit_lora_slots(
            sample_to_slot,
            sequence_length=self._dit_lora_sequence_length(),
            cfg_branches=self.cfg_branches,
            padded_batch_size=padded_batch_size,
        )
        return {
            LM_LORA_DOMAIN: build_lora_context_from_batch_plan(
                plan,
                materialize_device=LM_LORA_DOMAIN in materialize_domains,
            ),
            PROJ_LORA_DOMAIN: build_lora_context_from_slot_list(
                sample_to_slot,
                materialize_device=PROJ_LORA_DOMAIN in materialize_domains,
            ),
            DIT_LORA_DOMAIN: build_lora_context_from_slot_list(
                dit_token_to_slot,
                materialize_device=DIT_LORA_DOMAIN in materialize_domains,
            ),
        }

    def _lora_model_modules(self) -> dict[str, torch.nn.Module]:
        """Memoize ``dict(self.model.named_modules())``.

        The dict is only invalidated by topology changes to the model; LoRA
        admission/validation never mutates the module graph, so it's safe to
        cache for the lifetime of the runner.
        """
        cache = getattr(self, "_lora_model_modules_cache", None)
        if cache is None:
            cache = dict(self.model.named_modules())
            self._lora_model_modules_cache = cache
        return cache

    def validate_lora_payload(
        self, payload: LoRAModelPayload | list[LoRAModelPayload] | tuple[LoRAModelPayload, ...]
    ) -> None:
        rank_payload = select_lora_payload_for_rank(payload, self.rank)
        if rank_payload.rank <= 0:
            raise ValueError(f"LoRA payload rank must be > 0, got {rank_payload.rank}")
        if not rank_payload.modules:
            raise ValueError("LoRA payload must contain at least one target module")

        modules = self._lora_model_modules()
        for module_name, module_payload in rank_payload.modules.items():
            try:
                module = modules[module_name]
            except KeyError as exc:
                raise ValueError(f"Unknown LoRA target module '{module_name}'") from exc
            validate_payload = getattr(module, "validate_slot_lora_payload", None)
            if validate_payload is None:
                raise ValueError(f"Module '{module_name}' does not support LoRA slots")
            validate_payload(
                module_payload.lora_a,
                module_payload.lora_b,
                module_payload.effective_rank,
                module_payload.scaling,
            )

    def register_lora(
        self,
        adapter_id: int,
        name: str,
        payload: LoRAModelPayload | list[LoRAModelPayload] | tuple[LoRAModelPayload, ...],
    ) -> None:
        rank_payload = select_lora_payload_for_rank(payload, self.rank)
        self.validate_lora_payload(rank_payload)
        registered_adapter_id = self.lora_runtime.register_lora(name, rank_payload, adapter_id=adapter_id)
        if registered_adapter_id != adapter_id:
            raise RuntimeError(f"Runner LoRA adapter id mismatch: expected {adapter_id}, got {registered_adapter_id}")

    def unregister_lora(self, adapter_id: int) -> None:
        entry = self.lora_runtime.get_entry(adapter_id)
        self.lora_runtime.unregister_lora(entry.name)

    def lora_on_sequence_enqueued(self, adapter_id: int | None) -> None:
        self.lora_runtime.on_sequence_enqueued(adapter_id)

    def lora_on_sequences_started(self, adapter_ids: list[int | None]) -> None:
        for adapter_id in adapter_ids:
            self.lora_runtime.on_sequence_started(adapter_id)

    def lora_on_sequence_preempted(self, adapter_id: int | None) -> None:
        self.lora_runtime.on_sequence_preempted(adapter_id)

    def lora_on_sequence_finished(self, adapter_id: int | None, was_running: bool) -> None:
        self.lora_runtime.on_sequence_finished(adapter_id, was_running=was_running)

    def release_sequence_state(self, seq_id: str) -> None:
        vae_decoder = getattr(self, "vae_streaming_decoder", None)
        if vae_decoder is not None:
            vae_decoder.release(seq_id)

    def _load_lora_slot(self, slot_id: int, payload: LoRAModelPayload) -> None:
        modules = self._lora_model_modules()
        # Only clear modules that the previous occupant of this slot actually
        # populated. This avoids issuing one ``zero_()`` kernel per LoRA-capable
        # layer in the entire model on every admission, which dominated the
        # ~0.18s LoRA TTFB regression.
        slot_modules = getattr(self, "_lora_slot_modules", None)
        if slot_modules is None:
            slot_modules = {}
            self._lora_slot_modules = slot_modules
        previously_loaded = slot_modules.get(slot_id)
        if previously_loaded is not None:
            incoming_modules = set(payload.modules)
            removed_modules = [name for name in previously_loaded if name not in incoming_modules]
            _clear_lora_slot_modules(modules, slot_id, module_names=removed_modules)
        for module_name, module_payload in payload.modules.items():
            try:
                module = modules[module_name]
            except KeyError as exc:
                raise ValueError(f"Unknown LoRA target module '{module_name}'") from exc
            set_slot_lora = getattr(module, "set_slot_lora", None)
            if set_slot_lora is None:
                raise ValueError(f"Module '{module_name}' does not support LoRA slots")
            set_slot_lora(
                slot_id=slot_id,
                lora_a=module_payload.lora_a.to(device="cuda", non_blocking=True),
                lora_b=(
                    [tensor.to(device="cuda", non_blocking=True) for tensor in module_payload.lora_b]
                    if isinstance(module_payload.lora_b, list)
                    else module_payload.lora_b.to(device="cuda", non_blocking=True)
                ),
                effective_rank=module_payload.effective_rank,
                scaling=module_payload.scaling,
            )
        slot_modules[slot_id] = list(payload.modules.keys())
        for fused in self._fused_lora_layers():
            fused.refresh_lora(slot_id, payload)

    def _fused_lora_layers(self) -> list:
        """Fused decoder layers holding merged per-adapter weights (voxcpm_kernels FusedCpm4Layer.stacks)."""
        cache = getattr(self, "_fused_lora_cache", None)
        if cache is None:
            model = getattr(self, "model", None)
            if model is None:
                return []
            cache = [m.forward for m in model.modules()
                     if getattr(getattr(m, "forward", None), "stacks", None) is not None]
            self._fused_lora_cache = cache
        return cache

    @torch.inference_mode()
    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = (
            self._config.max_num_batched_tokens,
            self._config.max_model_len,
        )
        num_seqs = min(max_num_batched_tokens // max_model_len, self._config.max_num_seqs)
        seqs = [
            RunnerTask(
                block_table=[],
                seq_length=max_model_len,
                num_cached_tokens=0,
                block_size=self.block_size,
                custom_payload=None,
            )
            for _ in range(num_seqs)
        ]
        inputs = {"positions": self.prepare_prefill_context(seqs)}
        inputs.update(self.make_dummy_inputs(num_seqs, max_model_len))
        _ = self.model(**inputs)

        # If LoRA is enabled, run additional warmup prefills with a fake
        # active slot so the Triton shrink/expand kernels JIT-compile and
        # autotune for prefill-shaped inputs during startup. Without this,
        # the first real request pays the JIT cost (hundreds of ms) on its
        # critical path and TTFB regresses significantly.
        #
        # Slot 0 weights are zero at this point; the kernel still runs and
        # contributes 0 to the output, which is exactly what we need for a
        # compile-only warmup.
        #
        # We exercise two shapes because `get_lora_op_configs` picks a
        # different shrink config at M<128 vs M>=128 (different split_k and
        # block_k), which instantiates distinct Triton kernels. Warming only
        # one regime still leaves the other to JIT on the first real request.
        if self.max_loras > 0 and is_lora_available():
            short_len = min(64, max_model_len)
            shape_candidates = []
            if max_model_len >= 128:
                shape_candidates.append((num_seqs, max_model_len))
            if short_len < 128:
                shape_candidates.append((1, short_len))
            # Deduplicate while preserving order.
            seen = set()
            shapes = [s for s in shape_candidates if not (s in seen or seen.add(s))]
            for warmup_num_seqs, warmup_len in shapes:
                warmup_seqs = [
                    RunnerTask(
                        block_table=[],
                        seq_length=warmup_len,
                        num_cached_tokens=0,
                        block_size=self.block_size,
                        custom_payload=None,
                    )
                    for _ in range(warmup_num_seqs)
                ]
                warmup_inputs = {"positions": self.prepare_prefill_context(warmup_seqs)}
                warmup_inputs.update(self.make_dummy_inputs(warmup_num_seqs, warmup_len))
                # Override LoRA contexts with "slot 0 active for every row".
                total_rows = warmup_num_seqs * warmup_len
                dit_rows_per_sample = self._dit_lora_rows_per_sample()
                lm_ctx = build_lora_context_from_slot_list([0] * total_rows)
                proj_ctx = build_lora_context_from_slot_list([0] * warmup_num_seqs)
                dit_ctx = build_lora_context_from_slot_list([0] * (warmup_num_seqs * dit_rows_per_sample))
                set_lora_context(lm_ctx, domain=LM_LORA_DOMAIN)
                set_lora_context(proj_ctx, domain=PROJ_LORA_DOMAIN)
                set_lora_context(dit_ctx, domain=DIT_LORA_DOMAIN)
                _ = self.model(**warmup_inputs)

        reset_all_contexts()
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        total_attention_block_size = 0
        for module in self.model.modules():
            if isinstance(module, Attention) and module.is_causal:
                total_attention_block_size += (
                    2 * self.block_size * module.num_kv_heads * module.head_dim * self.dtype.itemsize
                )

        # Manual overrides are explicit escape hatches. Otherwise, auto sizing
        # remains fail-fast when the measured memory budget cannot safely hold
        # any KV-cache blocks.
        env_num_blocks = _env_int(_NUM_KVCACHE_BLOCKS_ENV)
        if env_num_blocks is not None:
            if env_num_blocks <= 0:
                raise ValueError(f"{_NUM_KVCACHE_BLOCKS_ENV} must be greater than 0")
            self._config.num_kvcache_blocks = env_num_blocks
            allocated_mb = (self._config.num_kvcache_blocks * total_attention_block_size) / (1024**2)
            print(
                f"\n[VoxCPM-Warning] Using manual KV cache override from {_NUM_KVCACHE_BLOCKS_ENV}: "
                f"{self._config.num_kvcache_blocks} blocks ({allocated_mb:.2f} MB). "
                "This bypasses automatic memory sizing and may cause CUDA OOM.",
                file=sys.stderr,
                flush=True,
            )
        elif self._config.num_kvcache_blocks <= 0:
            free, total = torch.cuda.mem_get_info()
            peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
            current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
            reserved = torch.cuda.memory_reserved()

            available_budget = total * self._config.gpu_memory_utilization - peak
            available_physical = free + (reserved - current) - (peak - current)
            available_for_kv = min(available_budget, available_physical)
            self._config.num_kvcache_blocks = int(available_for_kv) // total_attention_block_size

            if self._config.num_kvcache_blocks <= 0:
                raise RuntimeError(
                    f"KV cache calculation resulted in {self._config.num_kvcache_blocks} blocks. "
                    "There is no safe memory budget to allocate KV cache. "
                    "Please lower 'max_model_len', 'max_num_batched_tokens', 'max_num_seqs', or "
                    "'gpu_memory_utilization'. Advanced users may set "
                    f"{_NUM_KVCACHE_BLOCKS_ENV} to bypass automatic KV cache sizing."
                )

        assert self._config.num_kvcache_blocks > 0

        causal = [m for m in self.model.modules() if isinstance(m, Attention) and m.is_causal]
        head_shapes = {(m.num_kv_heads, m.head_dim) for m in causal}
        self.kv_cache = None
        if len(head_shapes) == 1:
            # one allocation, [layer, k/v, block, slot, head, dim]: every layer keeps the per-layer layout flash-attn
            # reads, and a block's KV across all layers is reachable from one base pointer (one-launch block copy)
            heads, dim = head_shapes.pop()
            self.kv_cache = torch.empty(len(causal), 2, self._config.num_kvcache_blocks, self.block_size, heads, dim)
            for i, module in enumerate(causal):
                module.k_cache, module.v_cache = self.kv_cache[i, 0], self.kv_cache[i, 1]
            return
        for module in causal:
            module.k_cache = torch.empty(
                self._config.num_kvcache_blocks,
                self.block_size,
                module.num_kv_heads,
                module.head_dim,
            )
            module.v_cache = torch.empty(
                self._config.num_kvcache_blocks,
                self.block_size,
                module.num_kv_heads,
                module.head_dim,
            )

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
            dist.destroy_process_group()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
            if hasattr(self, "prefill_diffusion_graphs"):
                del self.prefill_diffusion_graphs, self.prefill_diffusion_graph_vars
        torch.cuda.synchronize()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            method = getattr(self, method_name, None)
            error = None
            try:
                method(*args)
            except Exception as exc:
                error = exc
            self._synchronize_rpc_result(method_name, error)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4 : n + 4])
        self.event.clear()
        if method_name == _RPC_FILE_SENTINEL:
            with open(args[0], "rb") as f:
                method_name, *args = pickle.load(f)
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        overflow_path = None
        if len(data) + 4 > self.shm.size:
            fd, overflow_path = tempfile.mkstemp(prefix="nanovllm-rpc-", suffix=".pkl")
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            data = pickle.dumps([_RPC_FILE_SENTINEL, overflow_path])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4 : n + 4] = data
        for event in self.event:
            event.set()
        return overflow_path

    def call(self, method_name, *args):
        overflow_path = None
        if self.world_size > 1 and self.rank == 0:
            overflow_path = self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        result = None
        error = None
        try:
            result = method(*args)
        except Exception as exc:
            error = exc
        try:
            self._synchronize_rpc_result(method_name, error)
            return result
        finally:
            if overflow_path is not None:
                try:
                    os.remove(overflow_path)
                except FileNotFoundError:
                    pass

    def _synchronize_rpc_result(self, method_name: str, error: Exception | None) -> None:
        if self.world_size <= 1 or method_name == "exit":
            if error is not None:
                raise error
            return
        failure = torch.tensor(
            [0 if error is None else 1], dtype=torch.int32, device="cuda" if torch.cuda.is_available() else "cpu"
        )
        dist.all_reduce(failure, op=dist.ReduceOp.MAX)
        if error is not None:
            raise error
        if int(failure.item()) != 0:
            raise RuntimeError(f"Distributed RPC '{method_name}' failed on another rank")

    def prepare_block_tables(self, seqs: list[RunnerTask]) -> torch.Tensor:
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables_list: list[list[int]] = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        return torch.tensor(block_tables_list, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

    def prepare_prefill_context(self, seqs: list[RunnerTask]):
        pack = HostPack(getattr(self, "pack_device", "cuda"))
        finish = self.stage_prefill_context(seqs, pack)
        return finish(pack.upload())

    def stage_prefill_context(self, seqs: list[RunnerTask], pack: HostPack):
        """The prefill attention context and LoRA contexts, built on the host with numpy (one pass per array instead
        of a Python loop over every token: ~1.5 -> ~0.4 ms for 200 cloned requests) and staged in ``pack``. Returns
        ``finish(uploaded)``: sets the contexts from the uploaded tensors and returns the positions. Values, dtypes
        and shapes are those of the per-token loop it replaces (tests/test_burst_host_path.py compares both)."""
        n = len(seqs)
        seqlen_k = np.fromiter((seq.seq_length for seq in seqs), dtype=np.int64, count=n)
        cached = np.fromiter((seq.num_cached_tokens for seq in seqs), dtype=np.int64, count=n)
        seqlen_q = seqlen_k - cached
        cu_seqlens_q = np.zeros(n + 1, dtype=np.int64)
        cu_seqlens_k = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(seqlen_q, out=cu_seqlens_q[1:])
        np.cumsum(seqlen_k, out=cu_seqlens_k[1:])
        max_seqlen_q = int(seqlen_q.max(initial=0))
        max_seqlen_k = int(seqlen_k.max(initial=0))
        total_q = int(cu_seqlens_q[-1])
        seq_of_token = np.repeat(np.arange(n), seqlen_q)
        # position of every query token: from its sequence's cached prefix to its length
        positions = np.arange(total_q, dtype=np.int64) - np.repeat(cu_seqlens_q[:-1] - cached, seqlen_q)
        lens = [len(seq.block_table) for seq in seqs]
        max_blocks = max(lens, default=0)
        flat: list[int] = []
        for seq, m in zip(seqs, lens):
            flat.extend(seq.block_table)
            if m < max_blocks:
                flat.extend([-1] * (max_blocks - m))
        tables = np.array(flat, dtype=np.int64).reshape(n, max_blocks)
        # token-granular slots (a cached prefix may end inside a block: ref_prefix_cache); warmup tasks have no
        # block table and no slots
        bs = self.block_size
        with_table = np.asarray(lens, dtype=np.int64)[seq_of_token] > 0
        tok_seq, tok_pos = seq_of_token[with_table], positions[with_table]
        slot_mapping = (tables[tok_seq, tok_pos // bs] * bs + tok_pos % bs) if tok_seq.size else np.zeros(0, np.int64)
        i_pos = pack.add(positions)
        i_cu_q = pack.add(cu_seqlens_q.astype(np.int32))
        i_cu_k = pack.add(cu_seqlens_k.astype(np.int32))
        i_slot = pack.add(slot_mapping.astype(np.int32))
        # prefix cache, or the paged prefill graphs (which read every sequence's K/V through block tables)
        i_bt = None
        if cu_seqlens_k[-1] > cu_seqlens_q[-1] or getattr(self, "prefill_backbone_graphs", None):
            i_bt = pack.add(tables.astype(np.int32))
        use_diffusion_graph = (
            not getattr(self, "enforce_eager", True)
            and hasattr(self, "prefill_diffusion_graphs")
            and hasattr(self.model, "forward_backbone")
        )
        materialize_domains = {LM_LORA_DOMAIN, PROJ_LORA_DOMAIN} if use_diffusion_graph else set(LORA_DOMAINS)
        # host-only contexts; the domains that get device metadata stage it in the pack (no copies of their own)
        lora = [
            (domain, stage_lora_context(ctx, pack.add) if domain in materialize_domains else (lambda views, c=ctx: c))
            for domain, ctx in self._build_lora_contexts(seqs, seqlen_q.tolist(), materialize_domains=set()).items()
        ]

        def finish(views: list[torch.Tensor]) -> torch.Tensor:
            set_context(True, views[i_cu_q], views[i_cu_k], max_seqlen_q, max_seqlen_k, views[i_slot], None,
                        views[i_bt] if i_bt is not None else None)
            for domain, make in lora:
                set_lora_context(make(views), domain=domain)
            return views[i_pos]

        return finish

    def prepare_decode_context(self, seqs: list[RunnerTask]):
        positions_list: list[int] = []
        slot_mapping_list: list[int] = []
        context_lens_list: list[int] = []
        for seq in seqs:
            positions_list.append(seq.seq_length - 1)
            context_lens_list.append(seq.seq_length)
            slot_mapping_list.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
        positions = torch.tensor(positions_list, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping_list, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens_list, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )
        materialize_domains = set(LORA_DOMAINS) if getattr(self, "enforce_eager", True) else set()
        for domain, lora_context in self._build_lora_contexts(
            seqs,
            [1 for _ in seqs],
            materialize_domains=materialize_domains,
        ).items():
            set_lora_context(lora_context, domain=domain)
        return positions

    def _make_graph_domain_buffers(self, max_rows: int, max_lora_buckets: int) -> dict:
        sizes = (
            max_rows,
            max_rows,
            max_lora_buckets,
            max_lora_buckets,
            max_lora_buckets + 1,
        )
        total_size = sum(sizes)
        packed = torch.empty(total_size, dtype=torch.int32)
        is_cuda_buffer = packed.device.type == "cuda"
        host_staging = (
            torch.empty(total_size, dtype=torch.int32, device="cpu", pin_memory=True) if is_cuda_buffer else packed
        )
        offsets = [0]
        for size in sizes:
            offsets.append(offsets[-1] + size)

        def views(buffer: torch.Tensor):
            return (
                buffer[offsets[0] : offsets[1]],
                buffer[offsets[1] : offsets[2]],
                buffer[offsets[2] : offsets[3]],
                buffer[offsets[3] : offsets[4]],
                buffer[offsets[4] : offsets[5]],
            )

        token_to_slot, token_indices, active_slot_ids, num_tokens, slot_offsets = views(packed)
        host_token_to_slot, host_token_indices, host_active_slot_ids, host_num_tokens, host_slot_offsets = views(
            host_staging
        )
        host_token_to_slot.fill_(-1)
        host_token_indices.copy_(torch.arange(max_rows, dtype=torch.int32))
        host_active_slot_ids.copy_(torch.arange(-1, max_lora_buckets - 1, dtype=torch.int32))
        host_num_tokens.zero_()
        host_slot_offsets.zero_()

        buffers = {
            "packed": packed,
            "token_to_slot": token_to_slot,
            "token_indices_sorted_by_slot": token_indices,
            "active_slot_ids": active_slot_ids,
            "num_tokens_per_slot": num_tokens,
            "slot_start_offsets": slot_offsets,
        }
        if is_cuda_buffer:
            buffers.update(
                {
                    "host_staging": host_staging,
                    "copy_event": torch.cuda.Event(),
                    "copy_pending": False,
                    "host_token_to_slot": host_token_to_slot,
                    "host_token_indices_sorted_by_slot": host_token_indices,
                    "host_active_slot_ids": host_active_slot_ids,
                    "host_num_tokens_per_slot": host_num_tokens,
                    "host_slot_start_offsets": host_slot_offsets,
                }
            )
        return buffers

    def _copy_lora_domain_to_graph_vars(
        self,
        graph_vars: dict,
        domain: str,
        context: LoRAContext,
    ) -> None:
        """Update per-domain graph-captured LoRA metadata buffers in place.

        Hot path: this runs 3× per decode step (once per LoRA domain), and
        previously issued ~10 tiny kernel launches per domain (fill_, two
        narrow-slice copies including a fresh ``torch.arange`` allocation,
        zero_, scatter_, zero_, cumsum). That was ~30 launches / step of pure
        metadata shuffling, dominating the LoRA RTF regression.

        We now precompute the final-form slices on CPU (int32), pack them into
        one pinned buffer, and issue a single H2D copy for the entire domain.
        The same packed transfer restores the sentinel state for batches
        without active LoRA adapters.
        """
        domain_vars = graph_vars["lora_domains"][domain]
        token_to_slot_buf: torch.Tensor = domain_vars["token_to_slot"]
        token_indices_buf: torch.Tensor = domain_vars["token_indices_sorted_by_slot"]
        num_tokens_buf: torch.Tensor = domain_vars["num_tokens_per_slot"]
        slot_start_buf: torch.Tensor = domain_vars["slot_start_offsets"]

        token_count = context.token_count
        has_host_metadata = context.host_token_to_slot is not None
        if "host_staging" in domain_vars and (context.no_lora_flag or has_host_metadata):
            if domain_vars["copy_pending"]:
                domain_vars["copy_event"].synchronize()

            host_token_to_slot = domain_vars["host_token_to_slot"]
            host_token_indices = domain_vars["host_token_indices_sorted_by_slot"]
            host_num_tokens = domain_vars["host_num_tokens_per_slot"]
            host_slot_offsets = domain_vars["host_slot_start_offsets"]
            host_token_to_slot.fill_(-1)
            host_num_tokens.zero_()
            host_slot_offsets.zero_()

            if not context.no_lora_flag:
                host_slots = context.host_token_to_slot or []
                host_indices = context.host_token_indices_sorted_by_slot or []
                host_active_ids = context.host_active_slot_ids or []
                host_counts = context.host_num_tokens_per_slot or []
                if len(host_slots) > host_token_to_slot.numel():
                    raise ValueError("LoRA token metadata exceeds captured CUDA graph capacity")
                host_token_to_slot[: len(host_slots)].copy_(torch.tensor(host_slots, dtype=torch.int32))
                host_token_indices[: len(host_indices)].copy_(torch.tensor(host_indices, dtype=torch.int32))
                fixed_counts = [0] * host_num_tokens.numel()
                for slot_id, count in zip(host_active_ids, host_counts):
                    fixed_counts[slot_id + 1] = count
                fixed_offsets = [0]
                for count in fixed_counts:
                    fixed_offsets.append(fixed_offsets[-1] + count)
                host_num_tokens.copy_(torch.tensor(fixed_counts, dtype=torch.int32))
                host_slot_offsets.copy_(torch.tensor(fixed_offsets, dtype=torch.int32))

            domain_vars["packed"].copy_(domain_vars["host_staging"], non_blocking=True)
            domain_vars["copy_event"].record()
            domain_vars["copy_pending"] = True
            return

        if context.no_lora_flag or context.token_to_slot is None:
            # Kernels bail out on no_lora; we only need token_to_slot to be
            # all -1 so downstream sanity checks still see a stable state.
            token_to_slot_buf.fill_(-1)
            num_tokens_buf.zero_()
            slot_start_buf.zero_()
            # token_indices buffer already contains arange(...) from capture
            # time; no kernel needed to restore it since no_lora short-circuits
            # before it's read.
            return

        buf_size = token_to_slot_buf.size(0)
        device = token_to_slot_buf.device

        # 1. token_to_slot: prefix from context, rest -1.
        token_to_slot_buf[:token_count].copy_(context.token_to_slot, non_blocking=True)
        if token_count < buf_size:
            token_to_slot_buf[token_count:].fill_(-1)

        # 2. token_indices_sorted_by_slot: prefix from context, rest stays at
        # whatever arange value it had from capture (kernels only read the
        # first ``token_count`` entries via num_tokens_per_slot+slot_start).
        if context.token_indices_sorted_by_slot is not None:
            token_indices_buf[: context.token_indices_sorted_by_slot.size(0)].copy_(
                context.token_indices_sorted_by_slot, non_blocking=True
            )

        # 3. num_tokens_per_slot: zero, then scatter. Single scatter kernel —
        # unavoidable when we need to honor active_slot_ids ordering.
        num_tokens_buf.zero_()
        if context.active_slot_ids is not None and context.num_tokens_per_slot is not None:
            bucket_indices = context.active_slot_ids.to(device=device, dtype=torch.int64) + 1
            num_tokens_buf.scatter_(0, bucket_indices, context.num_tokens_per_slot.to(device=device))

        # 4. slot_start_offsets: cumsum of num_tokens_per_slot, with a leading
        # zero. Done as one cumsum kernel into the [1:] view.
        slot_start_buf[0] = 0
        torch.cumsum(num_tokens_buf, dim=0, out=slot_start_buf[1:])

    def _set_graph_lora_context(self, graph_vars: dict, domain: str, context: LoRAContext) -> None:
        self._copy_lora_domain_to_graph_vars(graph_vars, domain, context)
        if self._fused_lora_layers():  # segmented GEMMs' tile tables for this step
            from voxcpm_kernels.seg_gemm import SEG_TABLES

            rows = graph_vars["lora_domains"][domain]["token_to_slot"].numel()
            if context.no_lora_flag:  # a LoRA graph replayed for another domain's adapters reads these tables too
                SEG_TABLES.update_base(domain, rows, self.max_loras + 1)
            else:
                slots = list(context.host_token_to_slot or [])
                SEG_TABLES.update(domain, slots + [-1] * (rows - len(slots)), self.max_loras + 1)
        domain_vars = graph_vars["lora_domains"][domain]
        token_count = context.token_count
        num_lora_buckets = domain_vars["active_slot_ids"].size(0)
        set_lora_context(
            LoRAContext(
                token_to_slot=domain_vars["token_to_slot"][:token_count],
                token_indices_sorted_by_slot=domain_vars["token_indices_sorted_by_slot"][:token_count],
                active_slot_ids=domain_vars["active_slot_ids"],
                num_tokens_per_slot=domain_vars["num_tokens_per_slot"],
                slot_start_offsets=domain_vars["slot_start_offsets"],
                no_lora_flag=context.no_lora_flag,
                num_active_loras=num_lora_buckets,
            ),
            domain=domain,
        )

    def _set_graph_lora_contexts(self, graph_vars: dict, contexts: dict[str, LoRAContext]) -> None:
        for domain in LORA_DOMAINS:
            self._set_graph_lora_context(graph_vars, domain, contexts[domain])

    def _materialize_lora_contexts(self, contexts: dict[str, LoRAContext]) -> None:
        for domain, context in contexts.items():
            materialized = materialize_lora_context(context)
            contexts[domain] = materialized
            set_lora_context(materialized, domain=domain)

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self._config
        max_bs = min(config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        max_dit_lora_rows = self._dit_lora_rows_per_sample() * max_bs
        positions = torch.zeros(max_bs, dtype=torch.int64)
        inputs = {
            "positions": positions,
        }
        inputs.update(self.make_dummy_inputs(max_bs, 1))

        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        max_lora_buckets = self.max_loras + 1
        lora_domains = {
            LM_LORA_DOMAIN: self._make_graph_domain_buffers(max_bs, max_lora_buckets),
            PROJ_LORA_DOMAIN: self._make_graph_domain_buffers(max_bs, max_lora_buckets),
            DIT_LORA_DOMAIN: self._make_graph_domain_buffers(max_dit_lora_rows, max_lora_buckets),
        }
        outputs = self.make_dummy_outputs(max_bs)

        graph_bs_candidates = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16)) + [max_bs]
        self.graph_bs = sorted({bs for bs in graph_bs_candidates if 1 <= bs <= max_bs})
        if self._fused_lora_layers():  # static tile tables of the segmented GEMMs, before any capture
            from voxcpm_kernels.seg_gemm import SEG_TABLES

            SEG_TABLES.reserve(LM_LORA_DOMAIN, max_bs, max_lora_buckets)
            SEG_TABLES.reserve(DIT_LORA_DOMAIN, max_dit_lora_rows, max_lora_buckets,
                               segments=max_lora_buckets * self.cfg_branches)  # CFG: each branch grouped by adapter
        self.graphs = {"base": {}, "lora": {}}
        self.graph_pool = None
        capture_lora_graphs = bool(config.lora_config is not None and is_lora_available())
        if capture_lora_graphs:
            for module in iter_lora_modules(self.model):
                prime_lora_cache = getattr(module, "prime_lora_cache", None)
                if prime_lora_cache is not None:
                    prime_lora_cache()

        for bs in reversed(self.graph_bs):
            base_graph = torch.cuda.CUDAGraph()
            set_context(
                False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
            )
            self._set_graph_lora_contexts(
                {"lora_domains": lora_domains},
                {
                    LM_LORA_DOMAIN: build_lora_context_from_slot_list([-1] * bs),
                    PROJ_LORA_DOMAIN: build_lora_context_from_slot_list([-1] * bs),
                    DIT_LORA_DOMAIN: build_lora_context_from_slot_list([-1] * (self._dit_lora_rows_per_sample() * bs)),
                },
            )

            if isinstance(outputs, torch.Tensor):
                outputs[:bs] = self.model(**cut_inputs(inputs, bs))  # warmup
            else:
                assign_outputs(self.model(**cut_inputs(inputs, bs)), outputs, bs)

            with torch.cuda.graph(base_graph, self.graph_pool):
                if isinstance(outputs, torch.Tensor):
                    outputs[:bs] = self.model(**cut_inputs(inputs, bs))  # capture
                else:
                    assign_outputs(self.model(**cut_inputs(inputs, bs)), outputs, bs)

            if self.graph_pool is None:
                self.graph_pool = base_graph.pool()
            self.graphs["base"][bs] = base_graph

            if capture_lora_graphs:
                lora_graph = torch.cuda.CUDAGraph()
                dummy_sample_to_slot = [0 for _ in range(bs)]
                dummy_contexts = {
                    LM_LORA_DOMAIN: build_lora_context_from_slot_list([0 for _ in range(bs)]),
                    PROJ_LORA_DOMAIN: build_lora_context_from_slot_list(dummy_sample_to_slot),
                    DIT_LORA_DOMAIN: build_lora_context_from_slot_list(
                        [slot for slot in dummy_sample_to_slot for _ in range(self._dit_lora_rows_per_sample())]
                    ),
                }
                set_context(
                    False,
                    slot_mapping=slot_mapping[:bs],
                    context_lens=context_lens[:bs],
                    block_tables=block_tables[:bs],
                )
                self._set_graph_lora_contexts({"lora_domains": lora_domains}, dummy_contexts)
                if isinstance(outputs, torch.Tensor):
                    outputs[:bs] = self.model(**cut_inputs(inputs, bs))
                else:
                    assign_outputs(self.model(**cut_inputs(inputs, bs)), outputs, bs)
                with torch.cuda.graph(lora_graph, self.graph_pool):
                    if isinstance(outputs, torch.Tensor):
                        outputs[:bs] = self.model(**cut_inputs(inputs, bs))
                    else:
                        assign_outputs(self.model(**cut_inputs(inputs, bs)), outputs, bs)
                self.graphs["lora"][bs] = lora_graph
            torch.cuda.synchronize()
            reset_all_contexts()

        self.graph_vars = dict(
            inputs=inputs,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            lora_domains=lora_domains,
            outputs=outputs,
        )
        self.capture_prefill_diffusion_cudagraph()

    @torch.inference_mode()
    def capture_prefill_diffusion_cudagraph(self) -> None:
        make_dummy_inputs = getattr(self.model, "make_dummy_diffusion_inputs", None)
        forward_diffusion = getattr(self.model, "forward_diffusion", None)
        forward_backbone = getattr(self.model, "forward_backbone", None)
        if not callable(make_dummy_inputs) or not callable(forward_diffusion) or not callable(forward_backbone):
            return

        max_bs = max(self.graph_bs)
        inputs = make_dummy_inputs(max_bs)
        self._set_first_patch(True)
        cond = inputs["cond"]
        outputs = torch.zeros(max_bs, self.patch_size, cond.size(1), dtype=cond.dtype, device=cond.device)
        dit_rows_per_sample = self._dit_lora_rows_per_sample()
        max_lora_buckets = self.max_loras + 1
        graph_vars = {
            "inputs": inputs,
            "outputs": outputs,
            "lora_domains": {
                DIT_LORA_DOMAIN: self._make_graph_domain_buffers(
                    dit_rows_per_sample * max_bs,
                    max_lora_buckets,
                )
            },
        }
        # graphs of the first patch's LocDiT solve ("base" / "lora"); when the first patch has settings of its own
        # (dit_first_t_span, first_steps), also of the others' ("rest_base" / "rest_lora"), for prefill steps that
        # resume streams or decode running ones (Context.first_patch False)
        self.prefill_diffusion_graphs: dict[str, dict[int, torch.cuda.CUDAGraph]] = {"base": {}, "lora": {}}
        capture_lora_graphs = bool(
            self._config.lora_config is not None
            and getattr(self._config.lora_config, "enable_dit", False)
            and is_lora_available()
        )
        differs = getattr(self.model, "first_patch_differs", None)
        firsts = [True, False] if callable(differs) and differs() else [True]

        for first in firsts:
            prefix = "" if first else "rest_"
            self.prefill_diffusion_graphs.setdefault(prefix + "base", {})
            self.prefill_diffusion_graphs.setdefault(prefix + "lora", {})
            self._set_first_patch(first)
            for bs in reversed(self.graph_bs):
                dit_rows = dit_rows_per_sample * bs
                base_context = build_lora_context_from_slot_list([-1] * dit_rows)
                self._set_graph_lora_context(graph_vars, DIT_LORA_DOMAIN, base_context)
                outputs[:bs] = forward_diffusion(**cut_inputs(inputs, bs))

                base_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(base_graph, self.graph_pool):
                    outputs[:bs] = forward_diffusion(**cut_inputs(inputs, bs))
                self.prefill_diffusion_graphs[prefix + "base"][bs] = base_graph

                if capture_lora_graphs:
                    lora_context = build_lora_context_from_slot_list([0] * dit_rows)
                    self._set_graph_lora_context(graph_vars, DIT_LORA_DOMAIN, lora_context)
                    outputs[:bs] = forward_diffusion(**cut_inputs(inputs, bs))

                    lora_graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(lora_graph, self.graph_pool):
                        outputs[:bs] = forward_diffusion(**cut_inputs(inputs, bs))
                    self.prefill_diffusion_graphs[prefix + "lora"][bs] = lora_graph

                torch.cuda.synchronize()
                reset_all_contexts()

        self.prefill_diffusion_graph_vars = graph_vars
        self._set_first_patch(False)
        self._capture_prefill_backbone_graphs()

    _prefill_has_audio = True  # set per prefill step by runners that know their payloads

    @torch.inference_mode()
    def _capture_prefill_backbone_graphs(self) -> None:
        """CUDA-graph the prefill backbone for a few token buckets.

        Eager prefill is launch-bound (hundreds of small kernels per step), which dominates a voice-clone request's
        first chunk. The graphs read K/V through block tables (paged attention), so a prefix taken from the cache
        (ref_prefix_cache) works, and take the audio positions' LocEnc embeddings as an input, computed (or
        fetched from the LocEnc cache) before the replay. Each bucket takes up to ``nseq_max`` sequences:
        cu_seqlens are padded with zero-length sequences and padded tokens get slot -1 (no KV write).
        """
        self.prefill_backbone_graphs = {}
        turbo = getattr(getattr(self._config, "model_config", None), "turbo", None)
        if not getattr(turbo, "prefill_graph", False) or not hasattr(self.model, "prefill_no_audio"):
            return
        if self.max_loras > 0:
            return
        cfg = self._config
        t_cap = min(cfg.max_num_batched_tokens, getattr(turbo, "prefill_graph_max_tokens", 2048), 2048)
        buckets = sorted({b for b in (64, 128, 256, 384, 512, 768, 1024, 1536, 2048) if b <= t_cap} | {t_cap})
        nseq_max = min(cfg.max_num_seqs, 256)
        t_max = buckets[-1]
        max_blocks = (cfg.max_model_len + self.block_size - 1) // self.block_size
        hidden = self.model.config.lm_config.hidden_size
        gv = {
            "positions": torch.zeros(t_max, dtype=torch.int64),
            "text_tokens": torch.zeros(t_max, dtype=torch.int64),
            "feat": torch.zeros(t_max, self.patch_size, self.feat_dim),
            "feat_mask": torch.zeros(t_max, dtype=torch.bool),
            "feat_embeds": torch.zeros(t_max, hidden),
            "temperature": torch.zeros(nseq_max),
            "cfg_value": torch.zeros(nseq_max),
            "z_noise": torch.zeros(nseq_max, self.feat_dim, self.patch_size),
            "cu": torch.zeros(nseq_max + 1, dtype=torch.int32),
            "cu_k": torch.zeros(nseq_max + 1, dtype=torch.int32),
            "block_tables": torch.zeros(nseq_max, max_blocks, dtype=torch.int32),
            "slot": torch.full((t_max,), -1, dtype=torch.int32),
        }
        pool = None
        try:
            for tb in reversed(buckets):
                # capture shape: the bucket split into sequences of at most max_model_len tokens
                bounds = list(range(0, tb, cfg.max_model_len)) + [tb]
                gv["cu"].fill_(tb)
                gv["cu"][: len(bounds)] = torch.tensor(bounds, dtype=torch.int32)
                gv["cu_k"].copy_(gv["cu"])
                gv["positions"][:tb] = torch.arange(tb) % cfg.max_model_len
                max_q = min(tb, cfg.max_model_len)
                args = {
                    "positions": gv["positions"][:tb],
                    "text_tokens": gv["text_tokens"][:tb],
                    "feat": gv["feat"][:tb],
                    "feat_mask": gv["feat_mask"][:tb],
                    "temperature": gv["temperature"],
                    "cfg_value": gv["cfg_value"],
                    "z_noise": gv["z_noise"],
                    "feat_embeds_in": gv["feat_embeds"][:tb],
                }
                set_context(True, gv["cu"], gv["cu_k"], max_q, cfg.max_model_len, gv["slot"][:tb], None,
                            gv["block_tables"])
                self.model.forward_backbone(**args)  # warm-up (Triton JIT, cuBLAS handles)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool):
                    out = self.model.forward_backbone(**args)
                pool = graph.pool() if pool is None else pool
                self.prefill_backbone_graphs[tb] = (graph, out, max_q)
                reset_all_contexts()
        finally:
            reset_all_contexts()
        gv["positions"].zero_()
        gv["cu"].zero_()
        gv["cu_k"].zero_()
        self.prefill_backbone_vars = gv
        self.prefill_backbone_buckets = buckets
        torch.cuda.synchronize()

    def _prefill_feat_embeds(self, inputs: dict[str, torch.Tensor], out: torch.Tensor) -> None:
        """Audio positions' LocEnc embeddings into ``out`` (zeros elsewhere), outside the graph."""
        n_tok = inputs["positions"].size(0)
        out[:n_tok].zero_()
        model = self.model
        if inputs.get("enc_cache") is not None:  # LocEnc cache: encode first-seen patches, gather the rest
            pos, rows, miss, cache = inputs["enc_pos"], inputs["enc_rows"], inputs["enc_miss"], inputs["enc_cache"]
            if miss.numel() > 0:
                cache[rows[miss]] = model.enc_to_lm_proj(model.feat_encoder(inputs["feat"][pos[miss]]))
            if pos.numel() > 0:
                out[pos] = cache[rows]
        elif self._prefill_has_audio:
            idx = inputs["feat_mask"].nonzero(as_tuple=True)[0]
            if idx.numel() > 0:
                out[idx] = model.enc_to_lm_proj(model.feat_encoder(inputs["feat"][idx]))

    def _try_prefill_backbone_graph(self, inputs: dict[str, torch.Tensor], lora_contexts) -> tuple | None:
        graphs = getattr(self, "prefill_backbone_graphs", None)
        if not graphs:
            return None
        if any(not c.no_lora_flag and c.token_count > 0 for c in lora_contexts.values()):
            return None
        ctx = get_context()
        if ctx.block_tables is None:
            return None
        gv = self.prefill_backbone_vars
        n_tok = inputs["positions"].size(0)
        n_seq = inputs["temperature"].size(0)
        tb = next((b for b in self.prefill_backbone_buckets if b >= n_tok), None)
        if tb is None or n_seq > gv["temperature"].size(0) or ctx.block_tables.size(1) > gv["block_tables"].size(1):
            return None
        graph, (diffusion_out, other_out), max_q = graphs[tb]
        if ctx.max_seqlen_q > max_q:
            return None
        for name in ("positions", "text_tokens", "feat", "feat_mask"):
            gv[name][:n_tok].copy_(inputs[name])
        self._prefill_feat_embeds(inputs, gv["feat_embeds"])
        if tb > n_tok:
            gv["positions"][n_tok:tb].zero_()
            gv["text_tokens"][n_tok:tb].zero_()
            gv["feat_mask"][n_tok:tb].zero_()
            gv["feat_embeds"][n_tok:tb].zero_()
            gv["slot"][n_tok:tb].fill_(-1)
        gv["slot"][:n_tok].copy_(ctx.slot_mapping)
        gv["cu"][: n_seq + 1].copy_(ctx.cu_seqlens_q)
        gv["cu"][n_seq + 1 :].fill_(n_tok)
        gv["cu_k"][: n_seq + 1].copy_(ctx.cu_seqlens_k)
        gv["cu_k"][n_seq + 1 :].copy_(ctx.cu_seqlens_k[-1])
        bt = gv["block_tables"]
        bt[:n_seq].zero_()
        bt[:n_seq, : ctx.block_tables.size(1)].copy_(ctx.block_tables)
        for name in ("temperature", "cfg_value", "z_noise"):
            gv[name][:n_seq].copy_(inputs[name])
        graph.replay()
        return (
            {k: (v[:n_seq] if v is not None else None) for k, v in diffusion_out.items()},
            {k: v[:n_seq] for k, v in other_out.items()},
        )

    def _set_first_patch(self, enabled: bool) -> None:
        fn = getattr(self.model, "set_first_patch", None)
        if callable(fn):
            fn(enabled)

    def _run_prefill_with_diffusion_graph(
        self,
        inputs: dict[str, torch.Tensor],
        lora_contexts: dict[str, LoRAContext],
    ):
        forward_backbone = getattr(self.model, "forward_backbone", None)
        if not callable(forward_backbone):
            raise RuntimeError("diffusion graph replay requires model.forward_backbone")
        replayed = self._try_prefill_backbone_graph(inputs, lora_contexts)
        diffusion_inputs, outputs = replayed if replayed is not None else forward_backbone(**inputs)
        _profile.mark("backbone")
        batch_size = diffusion_inputs["mu"].size(0)
        dit_context = lora_contexts[DIT_LORA_DOMAIN]
        first = get_context().first_patch  # (Context.first_patch: no resumed stream, no decode riding along)

        def run_diffusion_eager():
            eager_dit_context = materialize_lora_context(dit_context)
            set_lora_context(eager_dit_context, domain=DIT_LORA_DOMAIN)
            sequence_length = self._dit_lora_sequence_length()
            token_to_slot = eager_dit_context.token_to_slot
            if not eager_dit_context.no_lora_flag and token_to_slot is not None and sequence_length > 0:
                rows_per_branch = token_to_slot.numel() // self.cfg_branches
                padded_batch_size = rows_per_branch // sequence_length
                if padded_batch_size > batch_size:
                    real_rows_per_branch = batch_size * sequence_length
                    token_to_slot = torch.cat(
                        [
                            token_to_slot[branch * rows_per_branch : branch * rows_per_branch + real_rows_per_branch]
                            for branch in range(self.cfg_branches)
                        ]
                    )
                    set_lora_context(
                        build_lora_context_from_token_to_slot(token_to_slot),
                        domain=DIT_LORA_DOMAIN,
                    )
            self._set_first_patch(first)
            try:
                outputs["latents"] = self.model.forward_diffusion(**diffusion_inputs)
            finally:
                self._set_first_patch(False)
            return outputs

        z_noise = diffusion_inputs.get("z_noise")
        if batch_size > self.graph_bs[-1] or z_noise is None:
            return run_diffusion_eager()

        has_active_dit_lora = not dit_context.no_lora_flag and dit_context.token_count > 0
        graph_kind = "lora" if has_active_dit_lora else "base"
        if not first and "rest_" + graph_kind in self.prefill_diffusion_graphs:
            graph_kind = "rest_" + graph_kind
        graphs = self.prefill_diffusion_graphs[graph_kind]
        if not graphs:
            return run_diffusion_eager()

        graph_bs = next(bs for bs in self.graph_bs if bs >= batch_size)
        graph_vars = self.prefill_diffusion_graph_vars
        for name, value in diffusion_inputs.items():
            buffer = graph_vars["inputs"][name]
            buffer[:batch_size].copy_(value)
            if batch_size < graph_bs:
                buffer[batch_size:graph_bs].zero_()

        self._set_graph_lora_context(graph_vars, DIT_LORA_DOMAIN, dit_context)
        graphs[graph_bs].replay()
        outputs["latents"] = graph_vars["outputs"][:batch_size]
        return outputs

    @torch.inference_mode()
    def run_model(self, inputs: dict, is_prefill: bool):
        lora_contexts = {domain: get_lora_context(domain) for domain in LORA_DOMAINS}
        has_active_lora = any(
            not context.no_lora_flag and context.token_count > 0 for context in lora_contexts.values()
        )
        has_lora_graph = has_active_lora and bool(getattr(self, "graphs", {}).get("lora"))
        try:
            if (
                is_prefill
                and not self.enforce_eager
                and hasattr(self, "prefill_diffusion_graphs")
                and hasattr(self.model, "forward_backbone")
            ):
                return self._run_prefill_with_diffusion_graph(inputs, lora_contexts)
            if (
                is_prefill
                or self.enforce_eager
                or inputs["positions"].size(0) > 512
                or (has_active_lora and not has_lora_graph)
            ):
                self._materialize_lora_contexts(lora_contexts)
                return self.model(**inputs)

            bs = inputs["positions"].size(0)
            context = get_context()
            graph_kind = "lora" if has_active_lora else "base"
            graph = self.graphs[graph_kind][next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            for kw in graph_vars["inputs"].keys():
                if kw not in inputs:
                    raise ValueError(f"Input {kw} is required")
                graph_vars["inputs"][kw][:bs] = inputs[kw]
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, : context.block_tables.size(1)] = context.block_tables
            self._set_graph_lora_contexts(graph_vars, lora_contexts)
            graph.replay()
            if isinstance(graph_vars["outputs"], torch.Tensor):
                return graph_vars["outputs"][:bs]
            else:
                return cut_inputs(graph_vars["outputs"], bs)
        finally:
            reset_all_contexts()
