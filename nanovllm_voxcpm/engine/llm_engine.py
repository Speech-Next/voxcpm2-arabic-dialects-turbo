"""nanovllm_voxcpm.engine.llm_engine

This module orchestrates the end-to-end inference runtime loop.

The core class :class:`LLMEngineBase` wires together three subsystems:

- Request state: :class:`~nanovllm_voxcpm.engine.sequence.Sequence`
  (prompt, generated tokens, KV bookkeeping).
- Scheduling + KV allocation: :class:`~nanovllm_voxcpm.engine.scheduler.Scheduler`
  (batching policy) and :class:`~nanovllm_voxcpm.engine.block_manager.BlockManager`
  (KV block pool + prefix caching).
- GPU execution: :class:`~nanovllm_voxcpm.engine.model_runner.BaseModelRunner`
  (build tensors, set attention context, run the model).

Lifecycle (one engine step)
---------------------------
1) :meth:`Scheduler.schedule` selects a list of sequences to run and whether the
   step is prefill or decode.
2) The model-specific engine converts each selected :class:`Sequence` into a
   :class:`~nanovllm_voxcpm.engine.model_runner.RunnerTask` via
   :meth:`preprocess_seq`.
3) :meth:`BaseModelRunner.call` executes :meth:`BaseModelRunner.run` on rank 0
   (and on other ranks if tensor parallelism is enabled).
4) Outputs are merged back into each :class:`Sequence` via
   :meth:`postprocess_seq` (typically appending a token and checking stop/EOS).
5) Sequences that set ``seq.stoped`` are finalized with :meth:`Scheduler.finish`
   (KV blocks are deallocated and the request is removed).

Multi-process / tensor parallel
-------------------------------
If ``tensor_parallel_size > 1``, this engine spawns additional worker processes
using the "spawn" start method. Rank 0 is created in-process; ranks 1..N-1 are
separate processes. Communication uses NCCL for model collectives (inside the
runner) plus a small shared-memory RPC mechanism to broadcast method calls.

Extending the engine
--------------------
:class:`LLMEngineBase` is intentionally model-agnostic. Model families implement
the interface by subclassing and providing:
- :meth:`preprocess_seq` to build model inputs from a :class:`Sequence`
- :meth:`postprocess_seq` to update the :class:`Sequence` from runner outputs

See the model packages under ``nanovllm_voxcpm/models/*`` for concrete
implementations.

Concrete example: VoxCPM
------------------------
``nanovllm_voxcpm/models/voxcpm2/engine.py`` implements an audio generation loop
on top of this base engine:

- ``Sequence.token_ids`` is used as a hashable prefix stream (ints for text token
  ids, bytes for latent patches) so KV prefix caching can work across requests.
- ``Sequence.custom_payload`` carries the real model inputs:
  ``text_tokens`` (list[int]), ``feats`` (list[np.ndarray]), ``feat_masks`` and
  sampling controls (temperature / cfg_value), plus streaming buffers.
- ``postprocess_seq`` appends the predicted latent patch, updates a decode-pad
  window used by the VAE decoder, and sets ``seq.stoped`` when ``stop_flag`` is
  emitted or a max length is reached.
"""

import atexit
import contextlib
import sys
from collections import deque

import torch.multiprocessing as mp

from nanovllm_voxcpm.config import Config
from nanovllm_voxcpm.engine.lora_manager import LoRAManager, LoRAModelPayload
from nanovllm_voxcpm.engine.sequence import Sequence
from nanovllm_voxcpm.engine.scheduler import Scheduler
from nanovllm_voxcpm.engine.model_runner import RunnerTask, BaseModelRunner, select_lora_payload_for_rank
from nanovllm_voxcpm.utils import profile as _profile
import socket
import torch


def get_distributed_port():
    # find a free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


class LLMEngineBase:
    model_runner: BaseModelRunner
    scheduler: Scheduler
    # cancels that arrive during a step (the runner's overlap hook runs commands while the GPU computes) wait for
    # its end: the step still decodes, postprocesses and finishes the sequences it scheduled
    _step_depth = 0
    _deferred_cancels: deque | None = None
    _exited = False

    def __init__(
        self,
        RunnerType: type[BaseModelRunner],
        config: Config,
        tensor_parallel_size: int,
    ):
        if sys.platform == "win32" and tensor_parallel_size > 1:
            raise NotImplementedError(
                "Tensor parallelism (tensor_parallel_size > 1) is currently not supported on Windows "
                "because NCCL is unavailable and Gloo is not a CUDA tensor-collective replacement for this path. "
                "Please run with a single GPU on Windows or use a Linux environment."
            )

        self.distributed_port = get_distributed_port() if tensor_parallel_size > 1 else None

        if config.devices is None or len(config.devices) == 0:
            n_devices = torch.cuda.device_count()
            if tensor_parallel_size > n_devices:
                raise ValueError(
                    f"Tensor parallel size {tensor_parallel_size} is greater than the number of available devices {n_devices}"
                )
            config.devices = list(range(tensor_parallel_size))

        if len(config.devices) != tensor_parallel_size:
            raise ValueError(
                f"Number of devices {len(config.devices)} is not equal to tensor parallel size {tensor_parallel_size}"
            )

        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(
                target=RunnerType,
                args=(config, i, config.devices[i], self.distributed_port, event),
            )
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = RunnerType(config, 0, config.devices[0], self.distributed_port, self.events)
        max_loras = getattr(config.lora_config, "max_loras", 0) if config.lora_config else 0
        max_lora_rank = getattr(config.lora_config, "max_lora_rank", None) if config.lora_config else None
        self.lora_manager = LoRAManager(max_loras=max_loras, max_lora_rank=max_lora_rank)
        self.scheduler = Scheduler(config, callbacks=self)
        atexit.register(self.exit)

    def exit(self):
        if self._exited:  # also registered with atexit
            return
        self._exited = True
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_sequence(self, seq: Sequence, front: bool = False):
        self.scheduler.add(seq, front=front)

    def cancel_sequence(self, seq_id: str):
        # mid-step: after the step. Admitted to the next prefill step ahead of it (prepare_prefill): after that step
        # too, which computes the KV of the full blocks it published at admission; freed before, they stayed in the
        # prefix cache with that KV never computed, and a later request with the same prefix read them as cached
        if self._step_depth or self.scheduler.is_admitted(seq_id):
            if self._deferred_cancels is None:
                self._deferred_cancels = deque()
            self._deferred_cancels.append(seq_id)
            return
        self.scheduler.cancel(seq_id)

    @contextlib.contextmanager
    def stepping(self):
        """Around a step (nestable: a model engine's step wraps this one): cancels wait for the outermost end."""
        self._step_depth += 1
        try:
            yield
        finally:
            self._step_depth -= 1
            while not self._step_depth and self._deferred_cancels:
                self.scheduler.cancel(self._deferred_cancels.popleft())

    def register_lora(
        self, name: str, payload: LoRAModelPayload | list[LoRAModelPayload] | tuple[LoRAModelPayload, ...]
    ) -> int:
        adapter_id = self.lora_manager.peek_next_adapter_id()
        local_payload = select_lora_payload_for_rank(payload, 0)
        self.model_runner.call("validate_lora_payload", payload)
        try:
            self.model_runner.call("register_lora", adapter_id, name, payload)
            self.lora_manager.register_lora(name, local_payload, adapter_id=adapter_id)
        except Exception:  # runner and manager stay in step: neither keeps the adapter
            try:
                self.model_runner.call("unregister_lora", adapter_id)
            except Exception:
                pass
            raise
        return adapter_id

    def unregister_lora(self, name: str) -> None:
        adapter_id = self.lora_manager.resolve_known_adapter(name)
        self.model_runner.call("unregister_lora", adapter_id)
        self.lora_manager.unregister_lora(name)

    def list_loras(self):
        return self.lora_manager.list_loras()

    def resolve_lora(self, lora_name: str | None) -> int | None:
        if lora_name is None:
            return None
        return self.lora_manager.resolve_adapter(lora_name)

    def can_schedule(self, running_adapter_ids: set[int], seq: Sequence) -> bool:
        return self.lora_manager.can_schedule(running_adapter_ids, seq.adapter_id)

    def on_seq_added(self, seq: Sequence) -> None:
        self.lora_manager.on_sequence_enqueued(seq.adapter_id)
        self.model_runner.call("lora_on_sequence_enqueued", seq.adapter_id)

    def cached_prefix_tokens(self, seq: Sequence) -> int:
        """Prompt tokens a model-level prefix cache will supply for ``seq`` (claimed in on_seq_running)."""
        return 0

    def on_seq_running(self, seq: Sequence) -> None:
        """A sequence was admitted (blocks allocated); its LoRA bookkeeping follows in on_seqs_running."""

    def on_seqs_running(self, seqs: list[Sequence]) -> None:
        """The LoRA bookkeeping of a scheduling pass's admitted sequences, at the end of the pass: one runner call
        instead of one per sequence (with tensor parallelism, each was a broadcast and an all-reduce). Nothing reads
        these counts during the pass: the control plane's admission test does not depend on them (it assigns no
        slots), the runtime's slot states are refreshed and read only when a step's batch plan is built."""
        ids = [seq.adapter_id for seq in seqs if seq.adapter_id is not None]  # None: a no-op on both sides
        if not ids:
            return
        for adapter_id in ids:
            self.lora_manager.on_sequence_started(adapter_id)
        self.model_runner.call("lora_on_sequences_started", ids)

    def on_seq_waiting(self, seq: Sequence) -> None:
        self.lora_manager.on_sequence_preempted(seq.adapter_id)
        self.model_runner.call("lora_on_sequence_preempted", seq.adapter_id)

    def on_seq_removed(self, seq: Sequence, *, was_running: bool) -> None:
        if self._prepared:
            self._prepared.pop(seq, None)  # cancelled after prepare_prefill
        self.lora_manager.on_sequence_finished(seq.adapter_id, was_running=was_running)
        self.model_runner.call("lora_on_sequence_finished", seq.adapter_id, was_running)
        self.model_runner.call("release_sequence_state", seq.seq_id)

    def on_prefill_admitted(self, seqs: list[Sequence]) -> None:
        """Hook: ``seqs`` were admitted into the next prefill step, before their runner inputs are built."""

    _prepared: dict | None = None  # sequence -> its prefill RunnerTask, built by prepare_prefill ahead of the step

    def prepare_prefill(self) -> int:
        """Admit waiting sequences into the next prefill step and build their runner inputs now. Only while the engine
        is idle and the next step is theirs (the server calls it in its admission window, when no request is pending):
        step() then starts from them instead of scheduling and preparing every request after the last one arrived.
        Returns how many were admitted."""
        seqs = self.scheduler.admit()
        if seqs:
            self.on_prefill_admitted(seqs)
            if self._prepared is None:
                self._prepared = {}
            for seq in seqs:
                self._prepared[seq] = self.preprocess_seq(seq, True)
        return len(seqs)

    def step(self):
        with self.stepping():
            _profile.begin_step()  # NANOVLLM_PROFILE: the step record's t_begin (and GPU reference)
            seqs, is_prefill = self.scheduler.schedule()
            _profile.host_mark("schedule")
            if not seqs:
                return seqs

            prepared = self._prepared
            # a mixed step (TurboConfig.mix_decode): a prefill step that also decodes these running sequences; the
            # runner's prefill path takes their decode inputs (the last patch, everything before it cached)
            ride = {id(seq) for seq in self.scheduler.mixed} if is_prefill and self.scheduler.mixed else ()
            fresh = [seq for seq in seqs if id(seq) not in ride] if ride else seqs
            if is_prefill:
                self.on_prefill_admitted([seq for seq in fresh if not prepared or seq not in prepared])
            if prepared and is_prefill:
                runner_tasks = [self.preprocess_seq(seq, False) if id(seq) in ride
                                else prepared.pop(seq, None) or self.preprocess_seq(seq, True) for seq in seqs]
                prepared.clear()
            else:
                runner_tasks = [self.preprocess_seq(seq, is_prefill and id(seq) not in ride) for seq in seqs]
            # a prefill step of only new streams: their first patch, with the first patch's LocDiT settings
            resuming = getattr(self.scheduler, "resuming", False)
            first = is_prefill and not ride and not resuming
            _profile.host_mark("preprocess")
            outputs = self.model_runner.call("run", runner_tasks, is_prefill, first)
            _profile.host_mark("run")

            for seq, output in zip(seqs, outputs):
                self.postprocess_seq(seq, output, is_prefill and id(seq) not in ride)
            _profile.host_mark("postprocess")
            if is_prefill:
                self.after_prefill(fresh)

            for seq in seqs:
                if seq.stoped:
                    self.scheduler.finish(seq)
            _profile.host_mark("finish")

            return seqs

    def after_prefill(self, seqs) -> None:
        """Hook for model engines (e.g. to cache prefix KV); runs before finished sequences are freed."""

    def is_finished(self):
        return self.scheduler.is_finished()

    def preprocess_seq(self, seq: Sequence, is_prefill: bool) -> RunnerTask:
        """Convert a scheduled :class:`~nanovllm_voxcpm.engine.sequence.Sequence` into a runner input.

        The engine loop is model-agnostic, so each model family must implement
        how to build a :class:`~nanovllm_voxcpm.engine.model_runner.RunnerTask`
        (plus any model-specific payload) from a scheduled ``Sequence``.

        What you typically need to do
        -----------------------------
        - **Pass through KV mapping**:
          The returned :class:`RunnerTask` must carry ``seq.block_table`` and
          ``seq.block_size`` so the runner can map logical token positions to
          physical KV-cache slots.
        - **Set lengths correctly**:
          ``RunnerTask.seq_length`` should reflect the logical sequence length
          the model should consider for this step.
        - **Set cached-prefix semantics**:
          ``RunnerTask.num_cached_tokens`` tells the runner how many tokens are
          already present in KV (for prefill) or equivalently where the "query"
          begins (for decode). The runner uses this to build attention context:
          prefill may have ``Q < K`` when prefix caching is active; decode is
          typically ``Q = 1``.
        - **Build model-specific tensors**:
          Convert ``seq.custom_payload`` (Python/numpy objects) into a payload
          structure that your runner knows how to turn into GPU tensors.
          The base engine never interprets this payload.

        Concrete example: VoxCPM
        ------------------------
        See ``nanovllm_voxcpm/models/voxcpm2/engine.py``:

        - Prefill (``is_prefill=True``):
          ``RunnerTask.num_cached_tokens = seq.num_cached_tokens`` and the engine
          slices prompt inputs to *only* send the uncached tail:
          ``text_tokens[seq.num_cached_tokens:]``, ``feats[seq.num_cached_tokens:]``,
          ``feat_masks[seq.num_cached_tokens:]``.
          This avoids recomputing KV for the cached prefix blocks.
        - Decode (``is_prefill=False``):
          VoxCPM sends only the last step (length 1) as inputs, and sets
          ``RunnerTask.num_cached_tokens = len(seq) - 1`` so the runner builds a
          decode context (keys over full context, query over the last token).

        Returns
        -------
        A :class:`RunnerTask` whose ``custom_payload`` is specific to the model
        runner implementation.
        """
        raise NotImplementedError()

    def postprocess_seq(self, seq: Sequence, outputs: dict, is_prefill: bool):
        """Merge runner outputs back into the :class:`~nanovllm_voxcpm.engine.sequence.Sequence`.

        This method is the model-specific "state update" step. The base engine
        calls it once per scheduled sequence after ``BaseModelRunner.run``
        returns.

        What you typically need to do
        -----------------------------
        - **Append new token(s)**:
          Update ``seq.token_ids`` via :meth:`Sequence.append_token` so the
          scheduler/block manager sees the sequence length grow. This is crucial
          for correct KV block allocation on the next decode step.
        - **Update model-side payload state**:
          Extend/refresh fields inside ``seq.custom_payload`` (e.g. decoded text,
          generated audio chunks, streaming buffers, sampling state, etc.). Keep
          any parallel arrays in sync with the logical sequence length.
        - **Stop condition**:
          Set ``seq.stoped = True`` when the request should finish (EOS token,
          stop_flag, max tokens, user-defined stopping criteria, etc.).
          The base engine will call ``scheduler.finish(seq)`` in the same step.

        What you should NOT do
        ----------------------
        - Do not allocate/free KV blocks here. KV lifecycle is owned by
          :class:`~nanovllm_voxcpm.engine.scheduler.Scheduler` /
          :class:`~nanovllm_voxcpm.engine.block_manager.BlockManager`.

        Concrete example: VoxCPM
        ------------------------
        In ``nanovllm_voxcpm/models/voxcpm2/engine.py``:

        - The runner returns ``latents`` (one patch), ``stop_flag``, and a decoded
          waveform chunk ``waveforms``.
        - The engine appends ``latents.tobytes()`` into ``seq.token_ids`` so prefix
          caching and KV block accounting see the new token.
        - It appends the same latent patch into ``custom_payload.feats`` and
          updates ``text_tokens``/``feat_masks`` so the next step has consistent
          aligned inputs.
        - It updates a small ``decode_pad`` window used by the AudioVAE decoder.
        - It sets ``seq.stoped`` when ``stop_flag == 1`` or when
          ``generated_waveforms`` reaches ``max_generate_length``.
        """
        raise NotImplementedError()
