from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np
import torch

from nanovllm_voxcpm.lora.backend import assert_available as assert_lora_available
from nanovllm_voxcpm.engine.utils import LoRAContext


class LoRALifecycleState(str, Enum):
    REGISTERED = "REGISTERED"
    ACTIVE = "ACTIVE"
    DRAINING = "DRAINING"
    REMOVED = "REMOVED"


class LoRAResidentState(str, Enum):
    ACTIVE = "ACTIVE"
    IDLE = "IDLE"


@dataclass
class LoRAModulePayload:
    lora_a: torch.Tensor
    lora_b: torch.Tensor | list[torch.Tensor]
    effective_rank: int
    scaling: float


@dataclass
class LoRAModelPayload:
    modules: dict[str, LoRAModulePayload]
    rank: int
    alpha: float
    # a tagged model served as an adapter (model/lora_loader.py): its dialect tags' embedding rows [n, dim],
    # written at ids tag_row_start.. (extra rows after the vocabulary), and exact_merge: the fused int8 layers merge it
    # as the merged model was exported (fp32 W + B A, rounded to the weight dtype) from these fp32 tensors
    tag_embed: torch.Tensor | None = None
    tag_row_start: int = -1
    exact_merge: bool = False


@dataclass
class LoRAControlEntry:
    name: str
    adapter_id: int
    state: LoRALifecycleState
    rank: int
    alpha: float
    scaling: float
    cpu_ref_count: int = 0
    gpu_running_ref_count: int = 0
    slot_id: int | None = None
    last_used_ts: int = 0


@dataclass
class LoRARuntimeEntry(LoRAControlEntry):
    model_payload: LoRAModelPayload | None = None


@dataclass
class LoRASlot:
    slot_id: int
    adapter_id: int | None = None
    resident_state: LoRAResidentState = LoRAResidentState.IDLE
    last_used_ts: int = 0


@dataclass
class LoRABatchPlan:
    adapter_to_slot: dict[int, int]
    token_to_slot: list[int]
    token_indices_sorted_by_slot: list[int]
    active_slot_ids: list[int]
    num_tokens_per_slot: list[int]
    slot_start_offsets: list[int]


_LOOP_MAX_ROWS = 512  # LoRA metadata: the per-token loop up to here, numpy above (measured crossover)


def _build_lora_metadata_from_token_to_slot(
    token_to_slot,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """(token indices grouped by slot, ascending within a slot; active slots ascending; tokens per slot; offsets) of
    the tokens with a slot (>= 0). numpy, one pass: the per-token Python loop took ~0.7 ms of a 200-request prefill
    step's preparation (its LM, projection and LocDiT domains: ~1300 + 200 + ~2300 rows). Up to _LOOP_MAX_ROWS rows
    (every decode step, small prefills) the loop is faster: numpy's fixed cost is ~15 us, the loop's ~1 us per 16
    rows."""
    if len(token_to_slot) <= _LOOP_MAX_ROWS:
        slot_to_indices: dict[int, list[int]] = {}
        for token_idx, slot_id in enumerate(token_to_slot):
            if slot_id >= 0:
                slot_to_indices.setdefault(int(slot_id), []).append(token_idx)
        active_slot_ids = sorted(slot_to_indices)
        num_tokens_per_slot = [len(slot_to_indices[slot_id]) for slot_id in active_slot_ids]
        token_indices_sorted_by_slot = [i for slot_id in active_slot_ids for i in slot_to_indices[slot_id]]
        slot_start_offsets = [0]
        for count in num_tokens_per_slot:
            slot_start_offsets.append(slot_start_offsets[-1] + count)
        return token_indices_sorted_by_slot, active_slot_ids, num_tokens_per_slot, slot_start_offsets
    s = np.asarray(token_to_slot, dtype=np.int64).reshape(-1)
    idx = np.flatnonzero(s >= 0)
    if idx.size == 0:
        return [], [], [], [0]
    slots = s[idx]
    active, counts = np.unique(slots, return_counts=True)
    order = idx[np.argsort(slots, kind="stable")]  # stable: token order within each slot
    return order.tolist(), active.tolist(), counts.tolist(), [0] + np.cumsum(counts).tolist()


def build_lora_batch_plan_from_token_to_slot(
    adapter_to_slot: dict[int, int],
    token_to_slot: list[int],
) -> LoRABatchPlan:
    (
        token_indices_sorted_by_slot,
        active_slot_ids,
        num_tokens_per_slot,
        slot_start_offsets,
    ) = _build_lora_metadata_from_token_to_slot(token_to_slot)
    return LoRABatchPlan(
        adapter_to_slot=adapter_to_slot,
        token_to_slot=token_to_slot,
        token_indices_sorted_by_slot=token_indices_sorted_by_slot,
        active_slot_ids=active_slot_ids,
        num_tokens_per_slot=num_tokens_per_slot,
        slot_start_offsets=slot_start_offsets,
    )


def _cuda_int_tensor(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32, device="cpu", pin_memory=True).cuda(non_blocking=True)


def _pack_int_tensors_to_cuda(*lists: list[int]) -> list[torch.Tensor]:
    """Concatenate several int lists into one pinned CPU tensor and issue a
    single H2D copy, then split on GPU into views.

    Per LoRA-enabled step we previously made up to 15 small H2D copies (5 int
    tensors × 3 LoRA domains), each paying a non-trivial fixed launch cost.
    Coalescing them removes most of that overhead while keeping the
    per-tensor view semantics the rest of the pipeline expects.
    """
    sizes = [len(lst) for lst in lists]
    flat: list[int] = []
    for lst in lists:
        flat.extend(lst)
    if not flat:
        # Caller will short-circuit on empty buckets; return empty views so
        # downstream slicing math still works.
        empty = torch.empty(0, dtype=torch.int32, device="cuda")
        return [empty for _ in lists]
    packed = torch.tensor(flat, dtype=torch.int32, device="cpu", pin_memory=True).cuda(non_blocking=True)
    out: list[torch.Tensor] = []
    offset = 0
    for size in sizes:
        out.append(packed[offset : offset + size])
        offset += size
    return out


def build_lora_context_from_batch_plan(
    plan: LoRABatchPlan,
    *,
    materialize_device: bool = True,
) -> LoRAContext:
    if not materialize_device:
        return LoRAContext(
            no_lora_flag=not plan.active_slot_ids,
            num_active_loras=len(plan.active_slot_ids),
            host_token_to_slot=plan.token_to_slot,
            host_token_indices_sorted_by_slot=plan.token_indices_sorted_by_slot,
            host_active_slot_ids=plan.active_slot_ids,
            host_num_tokens_per_slot=plan.num_tokens_per_slot,
        )

    if not plan.active_slot_ids:
        token_to_slot_tensor = _cuda_int_tensor(plan.token_to_slot)
        return LoRAContext(
            token_to_slot=token_to_slot_tensor,
            no_lora_flag=True,
            num_active_loras=0,
            host_token_to_slot=plan.token_to_slot,
            host_token_indices_sorted_by_slot=plan.token_indices_sorted_by_slot,
            host_active_slot_ids=plan.active_slot_ids,
            host_num_tokens_per_slot=plan.num_tokens_per_slot,
        )

    (
        token_to_slot_tensor,
        token_indices_sorted_by_slot_tensor,
        active_slot_ids_tensor,
        num_tokens_per_slot_tensor,
        slot_start_offsets_tensor,
    ) = _pack_int_tensors_to_cuda(
        plan.token_to_slot,
        plan.token_indices_sorted_by_slot,
        plan.active_slot_ids,
        plan.num_tokens_per_slot,
        plan.slot_start_offsets,
    )

    return LoRAContext(
        token_to_slot=token_to_slot_tensor,
        token_indices_sorted_by_slot=token_indices_sorted_by_slot_tensor,
        active_slot_ids=active_slot_ids_tensor,
        num_tokens_per_slot=num_tokens_per_slot_tensor,
        slot_start_offsets=slot_start_offsets_tensor,
        no_lora_flag=False,
        num_active_loras=len(plan.active_slot_ids),
        host_token_to_slot=plan.token_to_slot,
        host_token_indices_sorted_by_slot=plan.token_indices_sorted_by_slot,
        host_active_slot_ids=plan.active_slot_ids,
        host_num_tokens_per_slot=plan.num_tokens_per_slot,
    )


def build_lora_context_from_slot_list(
    token_to_slot: list[int],
    *,
    materialize_device: bool = True,
) -> LoRAContext:
    return build_lora_context_from_batch_plan(
        build_lora_batch_plan_from_token_to_slot({}, token_to_slot),
        materialize_device=materialize_device,
    )


def stage_lora_context(context: LoRAContext, add: Callable[[np.ndarray], int]) -> Callable[[list], LoRAContext]:
    """materialize_lora_context through a step's HostPack: ``add`` stages an int32 array (returns its index); the
    returned function builds, from the uploaded tensors, the context materialize_lora_context would have built (the
    same five device arrays, flags and host lists), without an H2D copy of its own."""
    if context.token_to_slot is not None or context.host_token_to_slot is None:
        return lambda views: context
    host_slots = context.host_token_to_slot
    indices = context.host_token_indices_sorted_by_slot or []
    active = context.host_active_slot_ids or []
    counts = context.host_num_tokens_per_slot or []
    host = dict(host_token_to_slot=host_slots, host_token_indices_sorted_by_slot=indices,
                host_active_slot_ids=active, host_num_tokens_per_slot=counts)
    if not active:
        i = add(np.asarray(host_slots, dtype=np.int32))
        return lambda views: LoRAContext(token_to_slot=views[i], no_lora_flag=True, num_active_loras=0, **host)
    offsets = [0]
    for c in counts:
        offsets.append(offsets[-1] + c)
    ids = [add(np.asarray(values, dtype=np.int32)) for values in (host_slots, indices, active, counts, offsets)]
    return lambda views: LoRAContext(
        token_to_slot=views[ids[0]],
        token_indices_sorted_by_slot=views[ids[1]],
        active_slot_ids=views[ids[2]],
        num_tokens_per_slot=views[ids[3]],
        slot_start_offsets=views[ids[4]],
        no_lora_flag=False,
        num_active_loras=len(active),
        **host,
    )


def materialize_lora_context(context: LoRAContext) -> LoRAContext:
    """Create device metadata for a host-only graph context on eager fallback."""
    if context.token_to_slot is not None or context.host_token_to_slot is None:
        return context

    host_counts = context.host_num_tokens_per_slot or []
    slot_start_offsets = [0]
    for count in host_counts:
        slot_start_offsets.append(slot_start_offsets[-1] + count)
    plan = LoRABatchPlan(
        adapter_to_slot={},
        token_to_slot=context.host_token_to_slot,
        token_indices_sorted_by_slot=context.host_token_indices_sorted_by_slot or [],
        active_slot_ids=context.host_active_slot_ids or [],
        num_tokens_per_slot=host_counts,
        slot_start_offsets=slot_start_offsets,
    )
    return build_lora_context_from_batch_plan(plan)


class _LoRAStateBase:
    def __init__(self, max_loras: int, max_lora_rank: int | None = None):
        self.max_loras = max(0, max_loras)
        self.max_lora_rank = max_lora_rank
        self._next_adapter_id = 0
        self._clock = 0
        self._name_to_adapter_id: dict[str, int] = {}
        self._entries: dict[int, LoRAControlEntry | LoRARuntimeEntry] = {}
        self._slots = [LoRASlot(slot_id=i) for i in range(self.max_loras)]

    @property
    def enabled(self) -> bool:
        return self.max_loras > 0

    def peek_next_adapter_id(self) -> int:
        return self._next_adapter_id

    def _validate_registration(self, name: str, payload: LoRAModelPayload, adapter_id: int | None) -> int:
        if not self.enabled:
            raise RuntimeError("LoRA runtime is disabled for this engine")
        assert_lora_available()
        if name in self._name_to_adapter_id:
            raise ValueError(f"LoRA '{name}' is already registered")
        if self.max_lora_rank is not None and payload.rank > self.max_lora_rank:
            raise ValueError(f"LoRA rank {payload.rank} exceeds max_lora_rank={self.max_lora_rank}")

        if adapter_id is None:
            adapter_id = self._next_adapter_id
        elif adapter_id in self._entries:
            raise ValueError(f"LoRA adapter id {adapter_id} is already registered")
        self._next_adapter_id = max(self._next_adapter_id, adapter_id + 1)
        return adapter_id

    def resolve_adapter(self, name: str | None) -> int | None:
        if name is None:
            return None
        try:
            adapter_id = self._name_to_adapter_id[name]
        except KeyError as exc:
            raise ValueError(f"LoRA '{name}' is not registered") from exc
        entry = self._entries[adapter_id]
        if entry.state in (LoRALifecycleState.DRAINING, LoRALifecycleState.REMOVED):
            raise ValueError(f"LoRA '{name}' is draining and cannot accept new requests")
        return adapter_id

    def list_loras(self) -> list[LoRAControlEntry | LoRARuntimeEntry]:
        return sorted(self._entries.values(), key=lambda entry: entry.adapter_id)

    def get_entry(self, adapter_id: int) -> LoRAControlEntry | LoRARuntimeEntry:
        return self._entries[adapter_id]

    def unregister_lora(self, name: str) -> None:
        adapter_id = self.resolve_known_adapter(name)
        entry = self._entries[adapter_id]
        entry.state = LoRALifecycleState.DRAINING
        if entry.cpu_ref_count == 0:
            self._remove_entry(adapter_id)

    def resolve_known_adapter(self, name: str) -> int:
        try:
            return self._name_to_adapter_id[name]
        except KeyError as exc:
            raise ValueError(f"LoRA '{name}' is not registered") from exc

    def on_sequence_enqueued(self, adapter_id: int | None) -> None:
        if adapter_id is None:
            return
        entry = self._entries[adapter_id]
        entry.cpu_ref_count += 1
        if entry.state == LoRALifecycleState.REGISTERED:
            entry.state = LoRALifecycleState.ACTIVE

    def on_sequence_started(self, adapter_id: int | None) -> None:
        if adapter_id is None:
            return
        entry = self._entries[adapter_id]
        entry.gpu_running_ref_count += 1
        if entry.state == LoRALifecycleState.REGISTERED:
            entry.state = LoRALifecycleState.ACTIVE
        self._refresh_entry_slot(entry)

    def on_sequence_preempted(self, adapter_id: int | None) -> None:
        if adapter_id is None:
            return
        entry = self._entries[adapter_id]
        if entry.gpu_running_ref_count <= 0:
            raise RuntimeError(f"LoRA adapter {adapter_id} GPU ref count underflow")
        entry.gpu_running_ref_count -= 1
        self._refresh_entry_slot(entry)

    def on_sequence_finished(self, adapter_id: int | None, *, was_running: bool) -> None:
        if adapter_id is None:
            return
        entry = self._entries[adapter_id]
        if was_running:
            if entry.gpu_running_ref_count <= 0:
                raise RuntimeError(f"LoRA adapter {adapter_id} GPU ref count underflow")
            entry.gpu_running_ref_count -= 1
        if entry.cpu_ref_count <= 0:
            raise RuntimeError(f"LoRA adapter {adapter_id} CPU ref count underflow")
        entry.cpu_ref_count -= 1
        self._refresh_slot_states()
        if entry.state == LoRALifecycleState.DRAINING and entry.cpu_ref_count == 0:
            self._remove_entry(adapter_id)
        elif entry.cpu_ref_count == 0 and entry.state != LoRALifecycleState.DRAINING:
            entry.state = LoRALifecycleState.REGISTERED

    def can_schedule(self, running_adapter_ids: set[int], candidate_adapter_id: int | None) -> bool:
        """Can every adapter of ``running_adapter_ids`` and the candidate's be resident at once: the required adapters
        without a slot (missing) fit into free slots plus idle slots of adapters not required (reclaimable). One pass
        over the slots, no sets built (the scheduler asks once per admitted request)."""
        if not self.enabled or candidate_adapter_id is None:
            return True
        missing = len(running_adapter_ids) + (candidate_adapter_id not in running_adapter_ids)  # |required|
        reclaimable = 0
        resident_required = None
        for slot in self._slots:
            adapter_id = slot.adapter_id
            if adapter_id is None:
                reclaimable += 1
            elif adapter_id == candidate_adapter_id or adapter_id in running_adapter_ids:
                if resident_required is None:
                    resident_required = set()
                resident_required.add(adapter_id)
            elif slot.resident_state == LoRAResidentState.IDLE:
                reclaimable += 1
        if resident_required is not None:
            missing -= len(resident_required)
        return missing <= reclaimable

    def _remove_entry(self, adapter_id: int) -> None:
        entry = self._entries[adapter_id]
        if entry.slot_id is not None:
            slot = self._slots[entry.slot_id]
            slot.adapter_id = None
            slot.resident_state = LoRAResidentState.IDLE
            entry.slot_id = None
        entry.state = LoRALifecycleState.REMOVED
        self._name_to_adapter_id.pop(entry.name, None)
        self._entries.pop(adapter_id, None)

    def _refresh_entry_slot(self, entry: LoRAControlEntry) -> None:
        """_refresh_slot_states after a change of ``entry``'s GPU ref count, for its slot only: every other slot is up
        to date already (each count change refreshes its own slot; a removed entry clears its slot in _remove_entry;
        build_batch_plan refreshes all slots before it reads any state). A slot holds an adapter exactly when the
        adapter's entry names it (_assign_slot / _evict_slot / _remove_entry set both), and the control plane
        (LoRAManager) never assigns slots at all: walking every slot twice per admitted request was ~0.3 ms of a
        200-request prefill step."""
        if entry.slot_id is None:
            return
        slot = self._slots[entry.slot_id]
        if slot.adapter_id != entry.adapter_id:  # not reachable (see above); the full refresh stays correct
            self._refresh_slot_states()
            return
        slot.resident_state = LoRAResidentState.ACTIVE if entry.gpu_running_ref_count > 0 else LoRAResidentState.IDLE

    def _refresh_slot_states(self) -> None:
        for slot in self._slots:
            if slot.adapter_id is None:
                continue
            entry = self._entries.get(slot.adapter_id)
            if entry is None:
                slot.adapter_id = None
                slot.resident_state = LoRAResidentState.IDLE
                continue
            slot.resident_state = (
                LoRAResidentState.ACTIVE if entry.gpu_running_ref_count > 0 else LoRAResidentState.IDLE
            )

    def _tick(self) -> int:
        self._clock += 1
        return self._clock


class LoRAManager(_LoRAStateBase):
    def register_lora(self, name: str, payload: LoRAModelPayload, *, adapter_id: int | None = None) -> int:
        adapter_id = self._validate_registration(name, payload, adapter_id)
        scaling = payload.alpha / payload.rank if payload.rank > 0 else 0.0
        self._name_to_adapter_id[name] = adapter_id
        self._entries[adapter_id] = LoRAControlEntry(
            name=name,
            adapter_id=adapter_id,
            state=LoRALifecycleState.REGISTERED,
            rank=payload.rank,
            alpha=payload.alpha,
            scaling=scaling,
        )
        return adapter_id


class LoRARuntime(_LoRAStateBase):
    def register_lora(self, name: str, payload: LoRAModelPayload, *, adapter_id: int | None = None) -> int:
        adapter_id = self._validate_registration(name, payload, adapter_id)
        scaling = payload.alpha / payload.rank if payload.rank > 0 else 0.0
        self._name_to_adapter_id[name] = adapter_id
        self._entries[adapter_id] = LoRARuntimeEntry(
            name=name,
            adapter_id=adapter_id,
            state=LoRALifecycleState.REGISTERED,
            rank=payload.rank,
            alpha=payload.alpha,
            scaling=scaling,
            model_payload=payload,
        )
        return adapter_id

    def get_entry(self, adapter_id: int) -> LoRARuntimeEntry:
        return self._entries[adapter_id]  # type: ignore[return-value]

    def build_batch_plan(
        self,
        adapter_ids: list[int | None],
        token_counts: list[int],
        load_lora: Callable[[int, LoRAModelPayload], None],
    ) -> LoRABatchPlan:
        if len(adapter_ids) != len(token_counts):
            raise ValueError("adapter_ids and token_counts must have the same length")
        if not self.enabled:
            return LoRABatchPlan({}, [-1 for count in token_counts for _ in range(count)], [], [], [], [0])

        self._refresh_slot_states()
        distinct_adapter_ids = []
        for adapter_id in adapter_ids:
            if adapter_id is not None and adapter_id not in distinct_adapter_ids:
                distinct_adapter_ids.append(adapter_id)

        adapter_to_slot: dict[int, int] = {}
        for adapter_id in distinct_adapter_ids:
            slot_id = self._ensure_slot(adapter_id, load_lora)
            adapter_to_slot[adapter_id] = slot_id
            entry = self.get_entry(adapter_id)
            entry.last_used_ts = self._tick()
            slot = self._slots[slot_id]
            slot.last_used_ts = entry.last_used_ts
            slot.resident_state = (
                LoRAResidentState.ACTIVE if entry.gpu_running_ref_count > 0 else LoRAResidentState.IDLE
            )

        token_to_slot: list[int] = []
        for adapter_id, token_count in zip(adapter_ids, token_counts):
            slot_id = adapter_to_slot.get(adapter_id, -1) if adapter_id is not None else -1
            token_to_slot.extend([slot_id] * token_count)
        return build_lora_batch_plan_from_token_to_slot(adapter_to_slot, token_to_slot)

    def _ensure_slot(self, adapter_id: int, load_lora: Callable[[int, LoRAModelPayload], None]) -> int:
        entry = self.get_entry(adapter_id)
        if entry.slot_id is not None:
            return entry.slot_id

        empty_slot = next((slot for slot in self._slots if slot.adapter_id is None), None)
        if empty_slot is not None:
            self._assign_slot(empty_slot, entry, load_lora)
            return empty_slot.slot_id

        victim = self._select_victim({adapter_id})
        if victim is None:
            raise RuntimeError("No idle LoRA GPU slot available for admission")
        self._evict_slot(victim)
        self._assign_slot(victim, entry, load_lora)
        return victim.slot_id

    def _select_victim(self, protected_adapter_ids: set[int]) -> LoRASlot | None:
        idle_slots = [
            slot
            for slot in self._slots
            if slot.adapter_id is not None
            and slot.adapter_id not in protected_adapter_ids
            and slot.resident_state == LoRAResidentState.IDLE
        ]
        if not idle_slots:
            return None
        return min(idle_slots, key=lambda slot: (slot.last_used_ts, slot.slot_id))

    def _assign_slot(
        self, slot: LoRASlot, entry: LoRARuntimeEntry, load_lora: Callable[[int, LoRAModelPayload], None]
    ) -> None:
        if entry.model_payload is None:
            raise RuntimeError(f"LoRA runtime entry {entry.adapter_id} is missing model payload")
        load_lora(slot.slot_id, entry.model_payload)
        slot.adapter_id = entry.adapter_id
        slot.last_used_ts = self._tick()
        slot.resident_state = LoRAResidentState.ACTIVE if entry.gpu_running_ref_count > 0 else LoRAResidentState.IDLE
        entry.slot_id = slot.slot_id
        entry.last_used_ts = slot.last_used_ts

    def _evict_slot(self, slot: LoRASlot) -> None:
        if slot.adapter_id is None:
            return
        entry = self.get_entry(slot.adapter_id)
        if entry.gpu_running_ref_count > 0:
            raise RuntimeError("Cannot evict ACTIVE LoRA slot")
        entry.slot_id = None
        slot.adapter_id = None
        slot.resident_state = LoRAResidentState.IDLE
