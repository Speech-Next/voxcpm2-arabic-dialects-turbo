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

from nanovllm_voxcpm.engine.config import Config
from nanovllm_voxcpm.engine.sequence import Sequence, SequenceStatus
from nanovllm_voxcpm.engine.block_manager import BlockManager


# step_cap: what a prefill step may leave of the least playing buffer at the least (s), for the time model's error
_PAUSE_MARGIN = 0.03
# mix_decode: a running stream with less audio than a resuming prefill step's predicted time plus this rides along (s;
# a client that plays whole chunks has up to one of them, e.g. 128 ms, less than the engine streamed)
_RIDE_MARGIN = 0.15


def _by_adapter(seqs: list[Sequence]) -> list[Sequence]:
    """A step's sequences grouped by LoRA adapter (stable; base-model sequences last): every adapter's rows are
    one contiguous segment, which the segmented GEMMs of merged adapter weights (nanovllm_voxcpm.kernels.seg_gemm) need.
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
        # streams before new requests (TurboConfig.stream_guard_ms, 0 = off): see _admit_loop
        self.stream_guard = (getattr(turbo, "stream_guard_ms", 0) or 0) / 1000
        # while streams play: prefill steps that decode the playing streams too (TurboConfig.mix_decode), steps of
        # bounded length (step_cap_ms, 0 = off), judged with the callbacks' predicted step times (step_seconds): see
        # _admit_loop
        self.step_cap = (getattr(turbo, "step_cap_ms", 0) or 0) / 1000
        self.mix_decode = bool(getattr(turbo, "mix_decode", False))
        self.mix_heads = self.mix_decode and bool(getattr(turbo, "mix_heads", False))  # (TurboConfig.mix_heads)
        # the last step: (is_prefill, rows, tokens computed), for the callbacks' step time model; mixed: the running
        # sequences a prefill step decodes as well (mix_decode)
        self.last_step = (False, 0, 0)
        self.mixed: list[Sequence] = []
        self._playing = False  # the last admission pass saw a playing stream
        self._resumes = False  # the next prefill step resumes a stream (a continuation, a preempted sequence)
        self.resuming = False  # the last prefill step resumed one (no first-patch LocDiT settings for it)
        self._paused = False  # the last step was a prefill step that paused playing streams (no mixed rows)
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
            self._add_front(seq)
        else:
            self.waiting.append(seq)

    def _add_front(self, seq: Sequence) -> None:
        """Ahead of the new requests, among the streams waiting to resume by their queued audio (earliest deadline
        first, callbacks.stream_buffer): a continuation whose head left less audio (e.g. muted leading silence) is
        prefilled before the others that ended with it."""
        buffer_of = getattr(self.callbacks, "stream_buffer", None)
        b = buffer_of(seq) if buffer_of is not None else None
        if b is None:
            self.waiting.appendleft(seq)
            return
        i = 0
        for other in self.waiting:
            if not other.has_stream:
                break
            o = buffer_of(other)
            if o is None or o > b:
                break
            i += 1
        self.waiting.insert(i, seq)

    def holds_unwritten_blocks(self, seq_id: str) -> bool:
        """``seq_id`` is admitted to the next prefill step ahead of it (admit) and published full blocks to the prefix
        cache whose KV that step writes: freed before it, they stayed findable with that KV never written, by the
        next request with the same prefix and by those admitted with it that found them already."""
        seq = self._id_to_seq.get(seq_id) if self._admitted else None
        if seq is None or not any(s is seq for s in self._admitted):
            return False
        bm = self.block_manager
        return any(bm.blocks[b].hash != -1 for b in seq.block_table[seq.num_cached_tokens // bm.block_size:])

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
        # model-level prefix caches (VoxCPM2 ref_prefix_cache) shrink the prefill; optional callback
        prefix_hint = getattr(callbacks, "cached_prefix_tokens", None)
        running_ids = None  # adapters of the running sequences, built once and kept up to date in this pass
        # Streams before new requests: while a running stream's player has less than stream_guard of audio queued,
        # a prefill step (which pauses every stream) admits only waiting streams whose players have even less (a
        # head start's continuation, a preempted sequence: their own gap grows while they wait); the decode step
        # that follows refills the others. New requests wait for the next step with enough slack.
        slack = least = None
        buffer_of = getattr(callbacks, "stream_buffer", None)
        if buffer_of is not None and running and (self.stream_guard > 0 or self.step_cap > 0 or self.mix_decode):
            least = min((b for b in map(buffer_of, running) if b is not None), default=None)
        # While streams play (least: the least audio queued at a running stream's player):
        # - step_cap: a prefill step takes sequences only while its predicted time (callbacks.step_seconds(rows,
        #   tokens, prefill)) stays under step_cap; its first one also when that keeps the least buffer above a
        #   margin. A prefill step pauses every playing stream; the cap bounds the pause (and the wait between two
        #   chunks of a stream: #28's WebSocket benchmark counts every wait longer than a chunk's audio).
        # - mix_decode: a prefill step that resumes streams (a burst's continuations) also decodes the running ones
        #   short of audio (their next patch, as a decode step would), so it is no pause for them: the cap then bounds
        #   the step, those rows included, and the stream guard no longer holds the resuming streams back.
        cost = getattr(callbacks, "step_seconds", None)
        self._playing = least is not None
        # a prefill step that resumes a stream also decodes the running ones short of audio (mix_decode: _ride); one
        # of new requests only does not, so first audio stays as fast as it was
        mixable = self.mix_decode and least is not None
        patch = getattr(callbacks, "patch_seconds", 0.16)
        others = len(running) - budget[0]  # running sequences besides this step's
        cap = None
        if self.step_cap > 0 and cost is not None and least is not None:
            cap = self.step_cap
        if cap is not None and not mixable and self._paused:
            return  # two pauses in a row would make one longer than the cap: a decode step comes between
        if self.stream_guard > 0 and least is not None and least < self.stream_guard:
            slack = least
        for _ in range(len(waiting)):
            if not waiting or budget[0] >= self.max_num_seqs:
                break
            seq = waiting.popleft()
            mixed = mixable and (self._resumes or seq.has_stream or self.mix_heads)  # the short streams ride too
            if slack is not None and not (mixed and seq.has_stream):
                b = buffer_of(seq) if seq.has_stream else None
                if b is None or b > slack:  # a new request, or a stream with more audio left than a running one
                    waiting.appendleft(seq)
                    break
            hint = prefix_hint(seq) if prefix_hint is not None else 0
            ride = others if mixed else 0  # (at most: _ride takes those short of audio)
            late = False
            if cap is not None:
                t = cost(budget[0] + 1 + ride, budget[1] + seq.num_tokens - hint + ride, True)
                first = max(cap, least - _PAUSE_MARGIN + (patch if mixed else 0.0))
                late = t > (first if budget[0] == 0 else cap)
            if late:
                waiting.appendleft(seq)
                break
            if not seq.has_stream and self.num_streams >= self.max_num_seqs:
                # every stream slot is taken: a new request waits, a resuming one (preempted, continuation) may run
                deferred_waiting.append(seq)
                if self.num_streams == len(running):  # none waits to resume
                    break
                continue
            if budget[1] + seq.num_tokens - hint + ride > self.max_num_batched_tokens or not bm.can_allocate(seq):
                waiting.appendleft(seq)
                break
            if callbacks is not None:
                if running_ids is None:  # (recomputing it per candidate was O(running): ~0.6 ms for 200 requests)
                    running_ids = self._running_adapter_ids()
                if not callbacks.can_schedule(running_ids, seq):
                    deferred_waiting.append(seq)
                    continue
            bm.allocate(seq)
            self._resumes |= seq.has_stream
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
        self.mixed = []
        if self._admitted:
            scheduled_seqs, self._admitted, (_, tokens), self._budget = self._admitted, [], self._budget, [0, 0]
            if self.mix_decode and self._playing and (self._resumes or self.mix_heads):
                self._ride(scheduled_seqs, tokens)
            self.resuming, self._resumes = self._resumes, False
            self.last_step = (True, len(scheduled_seqs) + len(self.mixed), tokens + len(self.mixed))
            self._paused = self._playing and not self.mixed
            return _by_adapter(scheduled_seqs + self.mixed), True

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
        self.last_step = (False, len(scheduled_seqs), len(scheduled_seqs))
        self._paused = False
        return _by_adapter(scheduled_seqs), False

    def _ride(self, scheduled_seqs: list[Sequence], tokens: int) -> None:
        """mix_decode: a prefill step that resumes streams (a burst's continuations, the long ones) also decodes the
        running streams that would run out of audio during it (their buffer under its predicted time plus a margin;
        every one without a time model). The others sit it out; steps of new requests only (first audio) stay
        plain."""
        buffer_of = getattr(self.callbacks, "stream_buffer", None)
        cost = getattr(self.callbacks, "step_seconds", None)
        limit = None
        if buffer_of is not None and cost is not None:
            n = len(self.running)
            limit = cost(len(scheduled_seqs) + n, tokens + n, True) + _RIDE_MARGIN
        fresh = {id(seq) for seq in scheduled_seqs}
        for seq in self.running:
            if id(seq) in fresh or not self.block_manager.can_append(seq):
                continue
            if limit is not None:
                b = buffer_of(seq)
                if b is not None and b >= limit:
                    continue
            self.block_manager.may_append(seq)
            self.mixed.append(seq)

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
