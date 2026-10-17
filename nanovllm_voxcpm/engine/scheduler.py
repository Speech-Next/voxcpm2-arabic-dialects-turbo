"""nanovllm_voxcpm.engine.scheduler

This module implements the batching/scheduling policy for the inference runtime.

The scheduler owns two queues of :class:`~nanovllm_voxcpm.engine.sequence.Sequence`:
- ``waiting``: admitted requests that are not currently executing.
- ``running``: requests that have KV-cache allocated and participate in decode.

It is responsible for choosing *which* sequences run on the next engine step and
for enforcing resource limits:
- ``max_num_seqs``: maximum number of sequences per batch.
- ``max_num_batched_tokens``: maximum number of tokens computed in a prefill.
- KV-cache capacity: enforced via :class:`~nanovllm_voxcpm.engine.block_manager.BlockManager`.

Two-phase scheduling
--------------------
The scheduler operates in two modes (returned as ``is_prefill``):

1) Prefill phase (``is_prefill=True``)
   - Pull from ``waiting`` in FIFO order.
   - Admit a sequence only if:
     * batched tokens would not exceed ``max_num_batched_tokens``
     * :meth:`BlockManager.can_allocate` is true for the full prompt length
     * it holds stream state already (preempted, a head start's continuation), or fewer than ``max_num_seqs``
       sequences do (``num_streams``): the runner's per-stream state (VAE slots) is sized for ``max_num_seqs``, so
       ``running`` never exceeds it and every running sequence decodes each step.
   - Allocate KV blocks (:meth:`BlockManager.allocate`) and move the sequence to
     ``running``.
   - If prefix caching hits, ``Sequence.num_cached_tokens`` may be > 0, and only
     the remaining prompt tokens (uncached portion) count toward the batch.
   - While the engine is idle, :meth:`admit` runs this admission ahead of the step
     (for the requests registered so far); :meth:`schedule` continues it.

2) Decode phase (``is_prefill=False``)
   - Round-robin over ``running``.
   - Before decoding one step for a sequence, ensure there is KV space for the
     *current last token* (see :meth:`BlockManager.can_append`).
   - If KV space is insufficient, preempt other sequences: move them back to
     ``waiting`` and free their blocks (:meth:`preempt`).
   - Once capacity is ensured, prepare KV bookkeeping for the step
     (:meth:`BlockManager.may_append`) and include the sequence in the decode batch.

Decode bookkeeping detail
-------------------------
The engine appends newly generated tokens during postprocessing (after the model
returns). Therefore, on the *next* decode step, the sequence already contains a
new "last token" whose KV state has not been written yet. ``may_append`` is
called before executing the decode step to ensure there is a physical KV slot
(possibly a new block) where that last token will be stored.

Concrete example: VoxCPM
------------------------
VoxCPM appends a newly generated latent patch as ``bytes`` into
``Sequence.token_ids`` in ``VoxCPMEngine.postprocess_seq``. On the next step,
the scheduler will:
- potentially allocate a new KV block if the new token starts a new block;
- then batch the sequence for decode so the runner can compute KV for that last
  latent patch and predict the next one.

Interaction with the engine loop
--------------------------------
The engine calls :meth:`Scheduler.schedule` once per step, then executes the
returned sequences via the model runner. Model-specific postprocessing sets
``seq.stoped`` when an EOS/stop condition is met, after which the engine calls
:meth:`Scheduler.finish` to deallocate KV resources and remove the request.
"""

from collections import deque

from nanovllm_voxcpm.config import Config
from nanovllm_voxcpm.engine.sequence import Sequence, SequenceStatus
from nanovllm_voxcpm.engine.block_manager import BlockManager


def _by_adapter(seqs: list[Sequence]) -> list[Sequence]:
    """A step's sequences grouped by LoRA adapter (stable; base-model sequences last): every adapter's rows are
    one contiguous segment, which the segmented GEMMs of merged adapter weights (voxcpm_kernels.seg_gemm) need.
    Base rows last: a CUDA graph's padding rows (base, -1) then extend the base segment instead of adding one, so a
    step with base rows and all max_loras adapters stays within the captured max_loras + 1 segments (2x for CFG)."""
    if len(seqs) < 2 or all(seq.adapter_id is None for seq in seqs):
        return seqs
    return sorted(seqs, key=lambda seq: (seq.adapter_id is None, seq.adapter_id or 0))


class Scheduler:
    def __init__(self, config: Config, callbacks=None):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        turbo = getattr(config.model_config, "turbo", None)
        self.max_prefill_seqs = getattr(turbo, "prefill_max_seqs", 0) or self.max_num_seqs
        # streams before new requests (TurboConfig.stream_guard_ms, 0 = off): see _admit_loop
        self.stream_guard = (getattr(turbo, "stream_guard_ms", 0) or 0) / 1000
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.callbacks = callbacks

        self._id_to_seq: dict[str, Sequence] = {}
        # sequences holding stream state (Sequence.has_stream: running, preempted, head start continuations); the
        # VAE decoder has max_num_seqs slots, so a new request waits while they are all taken
        self.num_streams = 0
        # the next prefill step's sequences admitted ahead of it (admit(), while the engine is idle) and its budget so
        # far (sequences, batched tokens): schedule() continues the same admission
        self._admitted: list[Sequence] = []
        self._budget = [0, 0]

    def is_finished(self):
        return not self.waiting and not self.running

    @property
    def num_queued(self) -> int:
        """Sequences waiting for a prefill step: in the waiting queue or admitted ahead of it."""
        return len(self.waiting) + len(self._admitted)

    def add(self, seq: Sequence, front: bool = False):
        """front: ahead of the queue (a head start's continuation, which carries on a stream already playing)."""
        self._id_to_seq[seq.seq_id] = seq
        self.num_streams += seq.has_stream
        if self.callbacks is not None:
            self.callbacks.on_seq_added(seq)

        if front:
            self.waiting.appendleft(seq)
        else:
            self.waiting.append(seq)

    def cancel(self, seq_id: str):
        try:
            seq = self._id_to_seq.pop(seq_id)
        except KeyError:
            return

        if seq.status == SequenceStatus.RUNNING and any(s is seq for s in self._admitted):
            # admitted ahead of its prefill step: that step never had it (its budget returns)
            self._admitted = [s for s in self._admitted if s is not seq]
            self._budget[0] -= 1
            self._budget[1] -= len(seq) - seq.num_cached_tokens
        self.block_manager.deallocate(seq)
        self.num_streams -= seq.has_stream
        was_running = seq.status == SequenceStatus.RUNNING
        if seq.status == SequenceStatus.RUNNING:
            self.running.remove(seq)
        elif seq.status == SequenceStatus.WAITING:
            self.waiting.remove(seq)
        if self.callbacks is not None:
            self.callbacks.on_seq_removed(seq, was_running=was_running)
        return

    def admit(self) -> list[Sequence]:
        """Admit waiting sequences into the next prefill step now (only while no step runs before it: the engine is
        idle, e.g. in the server's admission window); schedule() continues the same admission and returns them. The
        admitted set and order are the ones schedule() would choose: every refusal below only gets stricter as a
        step's admissions grow (budget, blocks, stream slots, adapters). Returns the newly admitted sequences."""
        return self._admit()

    def _admit(self) -> list[Sequence]:
        admitted: list[Sequence] = []
        deferred_waiting: deque[Sequence] = deque()
        try:
            self._admit_loop(admitted, deferred_waiting)
        finally:
            while deferred_waiting:
                self.waiting.appendleft(deferred_waiting.pop())
            self._admitted.extend(admitted)
            on_seqs_running = getattr(self.callbacks, "on_seqs_running", None)
            if admitted and on_seqs_running is not None:  # the pass's bookkeeping in one call (LoRA counts)
                on_seqs_running(admitted)
        return admitted

    def _admit_loop(self, admitted: list[Sequence], deferred_waiting: deque[Sequence]) -> None:
        waiting, running, bm, callbacks = self.waiting, self.running, self.block_manager, self.callbacks
        budget = self._budget  # the next prefill step's sequences and batched tokens so far, updated per admission
        max_seqs = min(self.max_num_seqs, self.max_prefill_seqs)
        # model-level prefix caches (VoxCPM2 ref_prefix_cache) shrink the prefill; optional callback
        prefix_hint = getattr(callbacks, "cached_prefix_tokens", None)
        running_ids = None  # adapters of the running sequences, built once and kept up to date in this pass
        # Streams before new requests: while a running stream's player has less than stream_guard of audio queued,
        # a prefill step (which pauses every stream) admits only waiting streams whose players have even less (a
        # head start's continuation, a preempted sequence: their own gap grows while they wait); the decode step
        # that follows refills the others. New requests wait for the next step with enough slack.
        slack = None
        buffer_of = getattr(callbacks, "stream_buffer", None) if self.stream_guard > 0 else None
        if buffer_of is not None and running:
            slack = min((b for b in map(buffer_of, running) if b is not None), default=None)
            if slack is not None and slack >= self.stream_guard:
                slack = None
        for _ in range(len(waiting)):
            if not waiting or budget[0] >= max_seqs:
                break
            seq = waiting.popleft()
            if slack is not None:
                b = buffer_of(seq) if seq.has_stream else None
                if b is None or b > slack:  # a new request, or a stream with more audio left than a running one
                    waiting.appendleft(seq)
                    break
            if not seq.has_stream and self.num_streams >= self.max_num_seqs:
                # every stream slot is taken: a new request waits, a resuming one (preempted, continuation) may run
                deferred_waiting.append(seq)
                if self.num_streams == len(running):  # none waits to resume
                    break
                continue
            hint = prefix_hint(seq) if prefix_hint is not None else 0
            if budget[1] + seq.num_tokens - hint > self.max_num_batched_tokens or not bm.can_allocate(seq):
                waiting.appendleft(seq)
                break
            if callbacks is not None:
                if running_ids is None:  # (recomputing it per candidate was O(running): ~0.6 ms for 200 requests)
                    running_ids = self._running_adapter_ids()
                if not callbacks.can_schedule(running_ids, seq):
                    deferred_waiting.append(seq)
                    continue
            bm.allocate(seq)
            if not seq.has_stream:
                seq.has_stream = True
                self.num_streams += 1
            seq.status = SequenceStatus.RUNNING
            running.append(seq)
            if running_ids is not None and seq.adapter_id is not None:
                running_ids.add(seq.adapter_id)
            if callbacks is not None:
                callbacks.on_seq_running(seq)
            # always >= 1: the block manager and the model-level prefix cache both leave the last token to compute
            # (it produces the step's patch), so an admitted sequence is always prefilled
            budget[0] += 1
            budget[1] += seq.num_tokens - seq.num_cached_tokens
            admitted.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        self._admit()
        if self._admitted:
            scheduled_seqs, self._admitted, self._budget = self._admitted, [], [0, 0]
            return _by_adapter(scheduled_seqs), True

        # decode
        scheduled_seqs = []
        num_seqs = 0
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        self.running.extendleft(reversed(scheduled_seqs))
        return _by_adapter(scheduled_seqs), False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
        if self.callbacks is not None:
            self.callbacks.on_seq_waiting(seq)

    def finish(self, seq: Sequence):
        if self._id_to_seq.get(seq.seq_id) is not seq:  # cancelled already
            return
        seq.status = SequenceStatus.FINISHED
        self.block_manager.deallocate(seq)
        self.num_streams -= seq.has_stream
        self.running.remove(seq)
        self._id_to_seq.pop(seq.seq_id)
        if self.callbacks is not None:
            self.callbacks.on_seq_removed(seq, was_running=True)

    def _running_adapter_ids(self) -> set[int]:
        return {seq.adapter_id for seq in self.running if seq.adapter_id is not None}
