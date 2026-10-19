import asyncio
import contextlib
import io
import itertools
import os
import threading
import time
from collections import deque

from nanovllm_voxcpm.utils import profile as _profile
import traceback
import uuid
from queue import Empty
from typing import Any, AsyncGenerator, List, Optional, cast

import librosa
import numpy as np
import torch
import torch.multiprocessing as mp
from numpy.typing import NDArray
from typing_extensions import Literal, TypedDict

from nanovllm_voxcpm.config import Config
from nanovllm_voxcpm.models.voxcpm2.config import LoRAConfig, TurboConfig, VoxCPM2Config
from nanovllm_voxcpm.models.voxcpm2.engine import VoxCPM2Engine
from nanovllm_voxcpm.models.voxcpm2.runner import VoxCPM2Runner

Waveform = NDArray[np.float32]



# NANOVLLM_SUBMIT_FLUSH_EVERY=k: also send pending commands to the engine as soon as k are queued, marked as
# followed by more (0 = once per event-loop pass). The engine registers a burst's first commands while the client is
# still building the rest: 200 requests took ~3 ms to submit and ~5 ms to register, one after the other.
_FLUSH_EVERY = int(os.environ.get("NANOVLLM_SUBMIT_FLUSH_EVERY", "48") or 0)

# NANOVLLM_CLIENT_TOKENIZE (default 1): the client tokenizes a batch of requests' texts (the engine's own tokenizer,
# sent with init_ok) as it sends them, and the engine registers them with those ids. Registering a burst is serial
# work in the engine process, which first audio waits for: tokenizing was ~21 of its ~58 us per request (200 at once:
# ~4.5 of ~12 ms), while the client built 48 requests in ~1 ms and then waited. The two processes now share the work
# (200 at once: registered ~1.5 ms sooner). Same ids and offsets: the same library and tokenizer; texts the engine
# tokenizes differently (CJK: the wrapper's character split) and single requests are left to it.
_CLIENT_TOKENIZE = os.environ.get("NANOVLLM_CLIENT_TOKENIZE", "1") != "0"
# NANOVLLM_SUBMIT_FLUSH_FIRST=k (default 16): a burst's first flush comes after k commands (then every
# NANOVLLM_SUBMIT_FLUSH_EVERY): the engine starts registering while the client builds and tokenizes the rest
_FLUSH_FIRST = int(os.environ.get("NANOVLLM_SUBMIT_FLUSH_FIRST", "16") or 0)
# NANOVLLM_EMIT_BLOCK (default 1): a prefill step's first audio (equal-length chunks, one per stream) goes through the
# shared ring as one block and one message item; the client splits it into the streams' chunks. Per-chunk writes,
# items and reads for 200 streams were ~0.8 ms in the engine and ~0.8 ms in the client, before the first audio's
# dispatch; one block saves ~0.5 ms of it.
_EMIT_BLOCK = os.environ.get("NANOVLLM_EMIT_BLOCK", "1") != "0"


def _block_item(ring, seq_ids, waves) -> dict | None:
    """Equal-length chunks of several streams written to the ring as one block: the message item (NANOVLLM_EMIT_BLOCK),
    or None (no ring, one stream, or chunks that differ)."""
    if ring is None or len(waves) < 2 or not isinstance(waves[0], np.ndarray) or not waves[0].size:
        return None
    shape, dtype = waves[0].shape, waves[0].dtype
    if not all(isinstance(w, np.ndarray) and w.shape == shape and w.dtype == dtype for w in waves):
        return None
    block = np.stack(waves)
    off, end = ring.write(block)
    return {"type": "stream_block", "ids": list(seq_ids), "off": off, "nb": block.nbytes, "end": end,
            "dtype": block.dtype.str, "shape": block.shape}


class _TokenizerSync:
    """The engine's tokenizer as its client has it (NANOVLLM_CLIENT_TOKENIZE). The tokenizer can change after start
    (registering a tagged adapter installs one that reads its dialect tags as single tokens): the client then gets
    the new one (message "tokenizer", with a new version), and ids tokenized with another version are not used."""

    def __init__(self, llm) -> None:
        self.llm, self.version = llm, 0
        self.backend = self._current() if _CLIENT_TOKENIZE and hasattr(llm, "pretokenize") else None

    def _current(self):
        return getattr(getattr(getattr(self.llm, "tokenizer", None), "tokenizer", None), "_tokenizer", None)

    def init_fields(self) -> dict:
        return {"tokenizer": self.backend.to_str() if self.backend is not None else None, "tokenizer_version": 0}

    def changed(self) -> dict | None:
        """The message for the client when the engine's tokenizer was replaced since the last call, else None."""
        if self.backend is None:  # (no client tokenizing: off, or no fast tokenizer at start)
            return None
        b = self._current()
        if b is self.backend:
            return None
        self.backend, self.version = b, self.version + 1
        return {"type": "tokenizer", "tokenizer": b.to_str() if b is not None else None, "version": self.version}

    def accept(self, tok) -> tuple | None:
        """(ids, offsets) of a command's client tokenization if it used the engine's current tokenizer, else None."""
        if tok is None or self.backend is None or tok[0] != self.version:
            return None
        return tok[1], tok[2]


def _request_text(cmd: dict) -> str | None:
    """The text the engine tokenizes for an add_request command (prompt_text + target_text), else None."""
    if cmd.get("type") != "add_request":
        return None
    a = cmd["args"]
    if len(a) >= 4 and isinstance(a[1], str):
        return (a[3] if isinstance(a[3], str) else "") + a[1]
    return None

class HealthResponse(TypedDict):
    status: Literal["ok"]


class ModelInfoResponse(TypedDict):
    sample_rate: int
    encoder_sample_rate: int
    output_sample_rate: int
    channels: int
    feat_dim: int
    patch_size: int
    model_path: str


class LoRAInfo(TypedDict):
    name: str


class RegisterLoRAResponse(TypedDict):
    name: str


class UnregisterLoRAResponse(TypedDict):
    name: str


_ID_PREFIX = uuid.uuid4().hex[:16]
_ID_COUNTER = itertools.count()


def gen_uuid() -> str:
    """A process-unique id: a random per-process prefix and a counter (uuid4 reads os.urandom per call: two ids per
    request were ~5 us of a burst's submission each)."""
    return f"{_ID_PREFIX}{next(_ID_COUNTER):016x}"  # 32 hex digits, as uuid4().hex


def _least_loaded(loads: list[int]) -> int:
    return min(range(len(loads)), key=loads.__getitem__)  # np.argmin converts the list on every request


class VoxCPM2ServerImpl:
    def __init__(
        self,
        model_path: str,
        inference_timesteps: int = 10,
        max_num_batched_tokens: int = 16384,
        max_num_seqs: int = 512,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.9,
        enforce_eager: bool = False,
        devices: List[int] = [],
        lora_config: Optional[LoRAConfig] = None,
    ):
        model_config = VoxCPM2Config.model_validate_json(open(os.path.join(model_path, "config.json")).read())
        model_config.inference_timesteps = inference_timesteps
        turbo_env = os.environ.get("NANOVLLM_VOXCPM2_TURBO")
        if turbo_env:
            model_config.turbo = TurboConfig.model_validate_json(turbo_env)
        self.lora_config = lora_config
        self.model_path = model_path

        engine_config = Config(
            model=model_path,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            model_config=model_config,
            devices=devices,
            lora_config=lora_config,
        )
        self.llm = VoxCPM2Engine(engine_config)
        model_runner = cast(VoxCPM2Runner, self.llm.model_runner)
        self.encoder_sample_rate = model_runner.vae.sample_rate
        self.output_sample_rate = model_runner.vae.out_sample_rate

    def health(self) -> HealthResponse:
        return HealthResponse(status="ok")

    def get_model_info(self) -> ModelInfoResponse:
        return ModelInfoResponse(
            sample_rate=int(self.output_sample_rate),
            encoder_sample_rate=int(self.encoder_sample_rate),
            output_sample_rate=int(self.output_sample_rate),
            channels=1,
            feat_dim=int(self.llm.feat_dim),
            patch_size=int(self.llm.patch_size),
            model_path=str(self.model_path),
        )

    def encode_latents(self, wav: bytes, wav_format: str) -> bytes:
        wav_np, _ = librosa.load(io.BytesIO(wav), sr=self.encoder_sample_rate, mono=False)
        wav_tensor = torch.from_numpy(wav_np)
        if wav_tensor.ndim == 1:
            wav_tensor = wav_tensor.unsqueeze(0)
        if wav_tensor.size(0) > 1:
            wav_tensor = wav_tensor.mean(dim=0, keepdim=True)
        latents = self.llm.encode_latents(wav_tensor)
        assert latents.shape[0] % self.llm.patch_size == 0
        return latents.tobytes()

    def add_request(
        self,
        seq_id: str,
        target_text: str,
        prompt_latents: bytes | None = None,
        prompt_text: str = "",
        max_generate_length: int = 2000,
        temperature: float = 1.0,
        cfg_value: float = 1.0,
        ref_audio_latents: bytes | None = None,
        lora_name: str | None = None,
        seed: int | None = None,
        voice: str | None = None,
        clone_mode: str | None = None,
    ) -> None:
        if voice is not None:
            self.llm.add_request(
                seq_id=seq_id,
                target_text=target_text,
                max_generate_length=max_generate_length,
                temperature=temperature,
                cfg_value=cfg_value,
                lora_name=lora_name,
                seed=seed,
                voice=voice,
                clone_mode=clone_mode,
            )
            return

        if prompt_latents is None:
            if len(prompt_text) > 0:
                raise ValueError("Prompt text is not allowed when prompt latents are not provided")
            self.llm.add_request(
                seq_id=seq_id,
                target_text=target_text,
                prompt_text="",
                ref_audio_latents=(
                    np.frombuffer(ref_audio_latents, dtype=np.float32).reshape(-1, self.llm.feat_dim)
                    if ref_audio_latents is not None
                    else None
                ),
                max_generate_length=max_generate_length,
                temperature=temperature,
                cfg_value=cfg_value,
                lora_name=lora_name,
                seed=seed,
            )
            return

        if len(prompt_text) == 0:
            raise ValueError("Prompt text is required when prompt latents are provided")

        prompt_latents_arr = np.frombuffer(prompt_latents, dtype=np.float32).reshape(-1, self.llm.feat_dim)
        self.llm.add_request(
            seq_id=seq_id,
            target_text=target_text,
            prompt_text=prompt_text,
            prompt_latents=prompt_latents_arr,
            ref_audio_latents=(
                np.frombuffer(ref_audio_latents, dtype=np.float32).reshape(-1, self.llm.feat_dim)
                if ref_audio_latents is not None
                else None
            ),
            max_generate_length=max_generate_length,
            temperature=temperature,
            cfg_value=cfg_value,
            lora_name=lora_name,
            seed=seed,
        )

    def register_voice(self, voice_id: str, latents: bytes, text: str | None = None) -> None:
        self.llm.register_voice(voice_id, np.frombuffer(latents, dtype=np.float32).reshape(-1, self.llm.feat_dim), text)

    def unregister_voice(self, voice_id: str) -> None:
        self.llm.unregister_voice(voice_id)

    def register_lora(self, name: str, path: str) -> RegisterLoRAResponse:
        self.llm.register_lora(name, path)
        return RegisterLoRAResponse(name=name)

    def unregister_lora(self, name: str) -> UnregisterLoRAResponse:
        self.llm.unregister_lora(name)
        return UnregisterLoRAResponse(name=name)

    def list_loras(self) -> list[LoRAInfo]:
        return [LoRAInfo(name=entry.name) for entry in self.llm.list_loras()]

    def cancel(self, seq_id: str):
        self.llm.cancel_sequence(seq_id)

    def step(self):
        return self.llm.step()

    def is_finished(self):
        return self.llm.is_finished()


def _host_sample() -> tuple[float, int, int]:
    """(thread CPU s, minor page faults, CPU number) for NANOVLLM_PROFILE records."""
    import resource

    try:
        cpu = int(open("/proc/thread-self/stat").read().rsplit(")", 1)[1].split()[36])
    except Exception:
        cpu = -1
    return time.thread_time(), resource.getrusage(resource.RUSAGE_SELF).ru_minflt, cpu


def _host_fields(before: tuple[float, int, int], wall_s: float) -> dict:
    after = _host_sample()
    return {"cpu_ms": (after[0] - before[0]) * 1000, "wall_ms": wall_s * 1000, "minflt": after[1] - before[1],
            "core": after[2], "cpu_abs": after[0], "minflt_abs": after[1]}


def _get_by(queue_in, t_end: float):
    """The next command if one arrives by t_end (perf_counter), else None; a command already there is taken even past
    t_end. A pipe poll waits whole milliseconds (a shorter timeout is rounded up to 1 ms), so a sub-millisecond wait,
    such as an admission gap's rest after prepare(), would last a whole millisecond: whole milliseconds are polled,
    the rest is spun."""
    ms = int((t_end - time.perf_counter()) * 1000)
    if ms > 0:
        try:
            return queue_in.get(timeout=(ms - 0.5) / 1000)  # poll rounds it up to ms
        except Empty:
            pass
    while True:
        try:
            return queue_in.get_nowait()
        except Empty:
            if time.perf_counter() >= t_end:
                return None


def main_loop(queue_in: mp.Queue, queue_out: mp.Queue, args, kwargs):
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # The fast tokenizer's worker pool (rayon) defaults to one thread per CPU: on this 96-CPU host a chunk of 48
    # texts took 0.9 ms hot / 1.9 ms after idle with 96 workers, 0.4 / 1.0 ms with 16 (set before the pool starts).
    os.environ.setdefault("RAYON_NUM_THREADS", "16")
    # Admission window when the engine is idle: keep collecting requests while they are still arriving
    # (gaps < NANOVLLM_QUEUE_GAP_MS) until one prefill group is waiting, for at most NANOVLLM_QUEUE_COALESCE_MS.
    # A burst sent over HTTP reaches the engine over several milliseconds; starting the first prefill step on its
    # first few requests would make everybody else wait for that small step. An isolated request waits one gap.
    try:
        coalesce_ms = float(os.environ.get("NANOVLLM_QUEUE_COALESCE_MS", "10"))
    except ValueError:
        coalesce_ms = 10.0
    if coalesce_ms > 0:
        coalesce_ms = min(coalesce_ms, 50.0)
    try:
        gap_ms = float(os.environ.get("NANOVLLM_QUEUE_GAP_MS", "1"))
    except ValueError:
        gap_ms = 1.0
    # A client's turn that ends with at least this many requests in the window ends the window at once: the client
    # sends what it was handed in one event-loop turn (in chunks marked "more", then the rest), so a burst that
    # arrived whole does not wait a further gap for stragglers. While the client promises more, the window waits
    # past the gap and the deadline, up to NANOVLLM_QUEUE_MORE_MS.
    batch_go = int(os.environ.get("NANOVLLM_QUEUE_BATCH_GO", "64"))
    more_ms = float(os.environ.get("NANOVLLM_QUEUE_MORE_MS", "20"))

    try:
        srv = VoxCPM2ServerImpl(*args, **kwargs)
    except Exception:
        queue_out.put({"type": "init_error", "error": traceback.format_exc()})
        return
    else:
        ring = None
        if os.environ.get("NANOVLLM_SHM_AUDIO", "1") != "0":
            try:
                ring = _AudioRing(size=int(float(os.environ.get("NANOVLLM_SHM_AUDIO_MB", "128")) * 2**20))
                import atexit

                atexit.register(ring.close)
            except Exception:  # no shared memory: chunks travel pickled
                ring = None
        tok_sync = _TokenizerSync(getattr(srv, "llm", None))
        queue_out.put({"type": "init_ok", "shm": ring.shm.name if ring is not None else None,
                       **tok_sync.init_fields()})
        if os.environ.get("NANOVLLM_GC_FREEZE", "1") != "0":
            # the model's objects live in a frozen generation and generation 0 collects less often: full
            # collections over them otherwise land inside steps (tens of ms in a burst)
            import gc

            gc.collect()
            gc.freeze()
            # no automatic collection at all: generation-0 collections landed inside the registration of a burst
            # (4.7 -> 8-10 ms every other burst of 200); the loop collects young objects when no request waits
            gc.disable()

    states = {"is_stoped": False}

    def method_call(cmd):
        opid = cmd.get("id", "")
        try:
            method_name = cmd["type"]
            args = cmd["args"]
            kwargs = cmd["kwargs"]
            if method_name == "stop":
                states["is_stoped"] = True
                return {"type": "response", "id": opid, "data": None}
            ret = getattr(srv, method_name)(*args, **kwargs)
            return {"type": "response", "id": opid, "data": ret}
        except Exception:
            return {"type": "error", "id": opid, "error": traceback.format_exc()}

    # NANOVLLM_CPROFILE=<file>: cProfile of the first steps and of the registrations, dumped at exit
    _cprof, _cprof_left = None, [0]
    if os.environ.get("NANOVLLM_CPROFILE"):
        import cProfile

        _cprof, _cprof_left = cProfile.Profile(), [int(os.environ.get("NANOVLLM_CPROFILE_STEPS", "400"))]
        import atexit

        atexit.register(lambda: _cprof.dump_stats(os.environ["NANOVLLM_CPROFILE"]))

    # Commands received but not yet executed. A burst arrives as one "batch"; only the first
    # FIRST_CHUNK requests are registered before the next step starts, the rest are registered by
    # the runner's overlap hook while the GPU computes that step.
    deferred: deque = deque()
    first_chunk = int(os.environ.get("NANOVLLM_FIRST_CHUNK", "96"))

    reg_parts = [0.0, 0.0, 0.0]  # the last registration's ms: pretokenize, add_request calls, responses out

    def run_deferred(limit=None):
        items = []
        pretokenize = getattr(getattr(srv, "llm", None), "pretokenize", None)
        t0 = time.perf_counter()
        if pretokenize is not None and deferred:  # one batched tokenizer call for the requests about to register
            n = len(deferred) if limit is None else min(limit, len(deferred))
            texts, done = [], {}
            for k in range(n):
                cmd = deferred[k]
                text = _request_text(cmd)
                if text is not None:
                    texts.append(text)
                    tok = tok_sync.accept(cmd.pop("tok", None))  # tokenized by the client (NANOVLLM_CLIENT_TOKENIZE)
                    if tok is not None:
                        done[text] = tok
            if texts:
                pretokenize(texts, done)
        t1 = time.perf_counter()
        while deferred and (limit is None or len(items) < limit):
            items.append(method_call(deferred.popleft()))
        t2 = time.perf_counter()
        if items:
            put(tok_sync.changed())  # (ahead of the responses: the client tokenizes its next requests with it)
            queue_out.put({"type": "batch", "items": items})
        reg_parts[:] = [(t1 - t0) * 1000, (t2 - t1) * 1000, (time.perf_counter() - t2) * 1000]

    sched = getattr(getattr(srv, "llm", None), "scheduler", None)
    group = min(sched.max_num_seqs, getattr(sched, "max_prefill_seqs", sched.max_num_seqs)) if sched else first_chunk

    def n_waiting() -> int:  # waiting for a prefill step, also when admitted to the next one already (prepare)
        if sched is None:
            return 0
        return sched.num_queued if hasattr(sched, "num_queued") else len(sched.waiting)

    # In the admission window the engine is idle and the next step is a prefill of what registers there. Whenever
    # no request is pending, it admits what has registered so far into that step and builds their runner inputs
    # (and starts their reference-prefix KV copies): the step that follows the last request only does that for the
    # requests after the last idle moment. Requests that trickle in (HTTP) leave idle moments between them; a burst
    # handed over at once does not, and runs exactly as before.
    prepare_prefill = getattr(getattr(srv, "llm", None), "prepare_prefill", None)

    def prepare() -> None:
        if prepare_prefill is None or sched is None or not sched.waiting:
            return
        t0 = time.perf_counter()
        n = prepare_prefill()
        if _profile.StepProfile.enabled and n:
            _profile.log(kind="prepare", t=t0, n=n, ms=(time.perf_counter() - t0) * 1000)

    window = {"n": 0, "end": False}  # commands of the current admission window, last message ended a client turn

    def handle(cmd):
        is_batch = cmd.get("type") == "batch"
        window["n"] += len(cmd["cmds"]) if is_batch else 1
        window["end"] = not (is_batch and cmd.get("more"))
        if is_batch:
            t_recv = time.perf_counter()
            h0 = _host_sample() if _profile.StepProfile.enabled else None
            deferred.extend(cmd["cmds"])
            # register now only what the next prefill step can take; a burst that arrives in several batches
            # (HTTP) would otherwise register every batch's first chunk before the first step starts
            if _cprof is not None:
                _cprof.enable()
            run_deferred(min(first_chunk, max(1, group - n_waiting())))
            if _cprof is not None:
                _cprof.disable()
            if _profile.StepProfile.enabled:
                _profile.log(kind="batch_recv", t=t_recv, n=len(cmd["cmds"]), register_ms=(time.perf_counter() - t_recv) * 1000,
                             pretok_ms=reg_parts[0], add_ms=reg_parts[1], resp_ms=reg_parts[2],
                             **_host_fields(h0, time.perf_counter() - t_recv))
            return None
        res = method_call(cmd)
        put(tok_sync.changed())  # (e.g. a tagged adapter's registration replaced it; ahead of the response)
        return res

    def overlap_hook():
        # register (during this step's GPU work) only what the next prefill step can take: registering a whole
        # burst here would stretch a small step far beyond its GPU time and hold everybody behind it
        while not states["is_stoped"]:
            try:
                cmd = queue_in.get_nowait()
            except Empty:
                break
            res = handle(cmd)
            if res is not None:
                queue_out.put(res)
        run_deferred(max(group - n_waiting(), 8))

    srv.llm.model_runner.overlap_hook = overlap_hook

    def stream_item(seq_id, wave) -> dict:
        if ring is not None and isinstance(wave, np.ndarray) and wave.size:
            off, end = ring.write(wave)
            return {"type": "stream_shm", "id": seq_id, "off": off, "nb": wave.nbytes, "end": end,
                    "dtype": wave.dtype.str, "n": wave.size}
        return {"type": "stream", "id": seq_id, "data": wave}

    def emit_hook(seq_ids, waves) -> None:
        """Audio that leaves before its step ends (the first half of a prefill step's patch)."""
        t_in = time.perf_counter()
        block = _block_item(ring, seq_ids, waves) if _EMIT_BLOCK else None
        items = [block] if block is not None else [stream_item(i, w) for i, w in zip(seq_ids, waves)]
        queue_out.put({"type": "batch", "items": items})
        if _profile.StepProfile.enabled:
            _profile.log(kind="emit", t=time.perf_counter(), t_in=t_in, n=len(seq_ids))

    srv.llm.model_runner.emit_hook = emit_hook

    def put(res):
        if res is not None:
            queue_out.put(res)

    import gc

    def collect_if_idle() -> None:
        """Young-generation garbage collection, only while no request waits for a prefill."""
        if gc.isenabled():
            return
        n = gc.get_count()[0]
        if n > 2_000_000 or (n > 20_000 and not deferred and n_waiting() == 0):  # never unbounded under load
            gc.collect(1)

    # NANOVLLM_BUSY_POLL_S=s: wait for requests spinning (not blocked) for s seconds after the last activity. The
    # host's cpufreq governor clocks an idle core down: the same host work ran 3.6x slower after 0.5 s idle, so a
    # burst's registration and first step's host work ran at the low clock.
    busy_poll_s = float(os.environ.get("NANOVLLM_BUSY_POLL_S", "0") or 0)
    last_active = [time.perf_counter()]

    def wait_cmd():
        while busy_poll_s > 0 and time.perf_counter() - last_active[0] < busy_poll_s:
            try:
                return queue_in.get_nowait()
            except Empty:
                pass
        return queue_in.get()

    while not states["is_stoped"]:
        if not deferred:
            collect_if_idle()
            cmd = wait_cmd()
            window.update(n=0, end=False)
            put(handle(cmd))

        if coalesce_ms > 0:
            t_open = time.perf_counter()
            deadline, hard = t_open + coalesce_ms / 1000.0, t_open + max(coalesce_ms, more_ms) / 1000.0
            while not states["is_stoped"] and n_waiting() + len(deferred) < group \
                    and not (window["end"] and window["n"] >= batch_go):
                more = not window["end"]
                remaining = (hard if more else deadline) - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    cmd = queue_in.get_nowait()
                except Empty:  # idle until the next request: prepare the step for the ones registered so far
                    gap_end = time.perf_counter() + gap_ms / 1000.0  # the gap counts from here, as before
                    prepare()
                    cmd = _get_by(queue_in, hard if more else min(deadline, gap_end))
                    if cmd is None:
                        break
                put(handle(cmd))

        if srv.is_finished():
            run_deferred(first_chunk)
        while not srv.is_finished() and not states["is_stoped"]:
            while not states["is_stoped"]:
                try:
                    cmd = queue_in.get_nowait()
                    put(handle(cmd))
                except Empty:
                    break
            if states["is_stoped"]:
                break

            t_step = time.perf_counter()
            if _cprof is not None and _cprof_left[0] > 0:  # NANOVLLM_CPROFILE=<file>: the first steps' host time
                _cprof.enable()
                output = srv.step()
                _cprof.disable()
                _cprof_left[0] -= 1
                if _cprof_left[0] == 0:
                    _cprof.dump_stats(os.environ["NANOVLLM_CPROFILE"])
            else:
                output = srv.step()
            t_put = time.perf_counter()
            items = []
            for seq in output:
                if not seq.custom_payload.muted:  # (leading silence past TurboConfig.lead_silence_patches)
                    items.append(stream_item(seq.seq_id, seq.custom_payload.generated_waveforms[-1]))
                if seq.is_finished and not getattr(seq.custom_payload, "chained", False):  # head start continues
                    items.append({"type": "stream", "id": seq.seq_id, "data": None})
            if items:
                queue_out.put({"type": "batch", "items": items})
            collect_if_idle()
            last_active[0] = time.perf_counter()
            if _profile.StepProfile.enabled:
                t_end = time.perf_counter()
                _profile.log(
                    kind="server", n=len(output), t=t_step,
                    step_ms=(t_put - t_step) * 1000, put_ms=(t_end - t_put) * 1000,
                )


class _AudioRing:
    """Streamed PCM through shared memory: the engine writes each step's chunks into a ring and sends only their
    offsets; the client's receiving thread copies them out on arrival. Pickling ~3 MB of int16 (200 streams' first
    chunk) and pushing it through the pipe took ~9 ms per step. The header holds the writer's byte count, so a
    reader that fell a whole ring behind gets an error instead of overwritten audio."""

    HEADER = 64

    def __init__(self, name: str | None = None, size: int = 0) -> None:
        from multiprocessing import shared_memory

        if name is None:
            self.shm = shared_memory.SharedMemory(create=True, size=self.HEADER + size)
            self.owner = True
        else:
            # the engine is spawned from this process and shares its resource tracker, which the owner's unlink
            # updates: an attachment here needs no unregister of its own
            self.shm = shared_memory.SharedMemory(name=name)
            self.owner = False
        self.size = self.shm.size - self.HEADER
        self.total = np.ndarray((1,), dtype=np.int64, buffer=self.shm.buf[:8])
        self.data = np.ndarray((self.size,), dtype=np.uint8, buffer=self.shm.buf[self.HEADER:])
        self.pos = 0
        if self.owner:
            self.total[0] = 0

    def write(self, arr: np.ndarray) -> tuple[int, int]:
        """Copy ``arr`` in; returns (offset, the writer's byte count after it)."""
        b = np.ascontiguousarray(arr).view(np.uint8).reshape(-1)
        n = b.size
        if n > self.size:
            raise ValueError("audio chunk larger than the shared ring")
        if self.pos + n > self.size:
            self.total[0] += self.size - self.pos  # the tail is skipped
            self.pos = 0
        off = self.pos
        self.data[off: off + n] = b
        self.pos += n
        self.total[0] += n
        return off, int(self.total[0])

    def read(self, off: int, n_bytes: int, end: int, dtype: str, count: int) -> np.ndarray:
        out = self.data[off: off + n_bytes].copy().view(np.dtype(dtype))[:count]
        if int(self.total[0]) - end > self.size - n_bytes:
            raise RuntimeError("streamed audio was overwritten before it was read (shared ring too small)")
        return out

    def close(self) -> None:
        del self.total, self.data
        self.shm.close()
        if self.owner:
            self.shm.unlink()


_BRIDGE_POLL_S = float(os.environ.get("NANOVLLM_BRIDGE_POLL_S", "0.1"))  # idle wake-up of the receiving thread
# NANOVLLM_BUSY_POLL_S=s: the client's event loop does not block for s seconds after its last request or engine message
# (the engine's main loop spins the same way): a burst's submission and its first audio's dispatch run at full clock
_BUSY_POLL_S = float(os.environ.get("NANOVLLM_BUSY_POLL_S", "0") or 0)


def _channel(ctx):
    """_Channel for a multiprocessing context; the context's own Queue otherwise (e.g. threads in tests)."""
    return _Channel(ctx) if hasattr(ctx, "SimpleQueue") else ctx.Queue()


class _Channel:
    """A process-to-process queue that sends in the calling thread (multiprocessing.SimpleQueue) with the
    Queue methods the server uses. multiprocessing.Queue hands every message to a feeder thread, which then waits
    for the GIL while the sender's main thread runs on: a step's audio (200 streams) reached the client ~20 ms
    after the step, a burst's requests reached the engine ~15 ms after they were submitted."""

    def __init__(self, ctx) -> None:
        self._q = ctx.SimpleQueue()

    def put(self, obj) -> None:
        self._q.put(obj)

    put_nowait = put

    def get(self, timeout: float | None = None):
        if timeout is not None and not self._q._reader.poll(max(timeout, 0.0)):
            raise Empty
        return self._q.get()

    def get_nowait(self):
        return self.get(timeout=0.0)

    def empty(self) -> bool:
        return self._q.empty()


class AsyncVoxCPM2Server:
    def __init__(
        self,
        model_path: str,
        inference_timesteps: int = 10,
        max_num_batched_tokens: int = 16384,
        max_num_seqs: int = 512,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.9,
        enforce_eager: bool = False,
        devices: List[int] = [],
        lora_config: Optional[LoRAConfig] = None,
        **kwargs,
    ) -> None:
        if len(kwargs) > 0:
            raise ValueError(f"Unknown kwargs: {kwargs}")
        ctx = mp.get_context("spawn")
        self.queue_in = _channel(ctx)
        self.queue_out = _channel(ctx)
        self.process = ctx.Process(
            target=main_loop,
            args=(
                self.queue_in,
                self.queue_out,
                (
                    model_path,
                    inference_timesteps,
                    max_num_batched_tokens,
                    max_num_seqs,
                    max_model_len,
                    gpu_memory_utilization,
                    enforce_eager,
                    devices,
                    lora_config,
                ),
                {},
            ),
            daemon=True,
        )
        self.process.start()
        loop = asyncio.get_running_loop()
        self._init_fut: asyncio.Future[None] = loop.create_future()
        self.op_table: dict[str, asyncio.Future[Any]] = {}
        self._pending_cmds: list[dict[str, Any]] = []
        self.stream_table: dict[str, asyncio.Queue[Waveform | None]] = {}
        self._queue_out_async: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._queue_out_stop = threading.Event()
        self._last_activity = time.perf_counter()
        self._wake = asyncio.Event()
        self._spin_task = asyncio.create_task(self._keep_warm()) if _BUSY_POLL_S > 0 else None
        # The engine's output is read in the event loop's own thread when the channel exposes its pipe: a receiving
        # thread hands every message over through call_soon_threadsafe and must take the GIL from the loop thread
        # first (a burst's first audio waited 23 ms behind a busy loop thread). NANOVLLM_CLIENT_READER=thread: the
        # receiving thread.
        reader = getattr(getattr(self.queue_out, "_q", None), "_reader", None)
        if reader is not None and os.environ.get("NANOVLLM_CLIENT_READER", "loop") != "thread":
            self._reader = reader
            loop.add_reader(reader.fileno(), self._on_readable)
            self.recv_task: asyncio.Task = asyncio.create_task(self._watch_engine())
            return
        self._queue_out_thread = threading.Thread(
            target=self._queue_out_bridge,
            args=(loop,),
            name="voxcpm2-queue-out",
            daemon=True,
        )
        self._queue_out_thread.start()
        self.recv_task = asyncio.create_task(self.recv_queue())

    _reader = None
    _ring = None
    _tokenizer = None  # the engine's tokenizer (init_ok, "tokenizer" messages), NANOVLLM_CLIENT_TOKENIZE
    _tok_version = 0  # its version: the engine uses only ids tokenized with its current one
    _first_flush = True  # the next flush is a burst's first (NANOVLLM_SUBMIT_FLUSH_FIRST)
    _more_sent = False  # this event-loop turn sent commands marked "more": its end must reach the engine
    _spin_task = None
    _wake = None

    def _touch(self) -> None:
        self._last_activity = time.perf_counter()
        if self._wake is not None:
            self._wake.set()

    async def _keep_warm(self) -> None:
        """NANOVLLM_BUSY_POLL_S: yield to the loop without blocking while requests are recent, then sleep until one
        comes (the host's cpufreq governor clocks an idle core down: the same work ran 3.6x slower after 0.5 s)."""
        try:
            while True:
                if time.perf_counter() - self._last_activity < _BUSY_POLL_S:
                    await asyncio.sleep(0)
                else:
                    self._wake.clear()
                    await self._wake.wait()
        except asyncio.CancelledError:
            return

    def _on_readable(self) -> None:
        """Every message waiting in the engine's output pipe, handled in the loop thread."""
        try:
            while self._reader.poll():
                if _profile.StepProfile.enabled:  # (NANOVLLM_PROFILE: when a batch of audio reaches the client)
                    t_in = time.perf_counter()
                    res = self.queue_out.get()
                    t_got = time.perf_counter()
                    res = self._convert(res)
                    t_conv = time.perf_counter()
                    self._handle(res)
                    items = res.get("items", [res]) if isinstance(res, dict) else []
                    n_audio = sum(1 for it in items if isinstance(it, dict) and it.get("type") == "stream")
                    if n_audio:
                        _profile.log(kind="client_recv", t=t_in, t_got=t_got, t_conv=t_conv,
                                     t_done=time.perf_counter(), n=n_audio)
                    continue
                self._handle(self._convert(self.queue_out.get()))
        except (EOFError, OSError, ValueError):  # the pipe is gone: the engine watch reports the exit
            with contextlib.suppress(Exception):
                asyncio.get_running_loop().remove_reader(self._reader.fileno())

    async def _watch_engine(self) -> None:
        try:
            while True:
                await asyncio.sleep(_BRIDGE_POLL_S)
                proc = getattr(self, "process", None)
                if proc is not None and proc.exitcode is not None:  # the engine died: fail everything waiting
                    self._handle({"type": "engine_dead", "exitcode": proc.exitcode})
                    return
        except asyncio.CancelledError:
            return

    def _convert(self, res: dict[str, Any]) -> dict[str, Any]:
        """Attach the engine's audio ring on init_ok; copy chunks sent through the ring out of it, on arrival."""
        if res.get("type") == "init_ok" and res.get("shm"):
            self._ring = _AudioRing(name=res["shm"])
        elif self._ring is not None:
            items = res["items"] if res.get("type") == "batch" else [res]
            blocks = False
            for k, item in enumerate(items):
                if item.get("type") == "stream_shm":
                    try:
                        data = self._ring.read(item["off"], item["nb"], item["end"], item["dtype"], item["n"])
                    except RuntimeError as e:  # overwritten: the stream fails (and is cancelled), no wrong audio
                        data = e
                    items[k] = {"type": "stream", "id": item["id"], "data": data}
                elif item.get("type") == "stream_block":  # (NANOVLLM_EMIT_BLOCK) one chunk per stream, one row each
                    blocks = True
            if blocks:
                out = []
                for item in items:
                    if item.get("type") != "stream_block":
                        out.append(item)
                        continue
                    shape = tuple(item["shape"])
                    try:
                        block = self._ring.read(item["off"], item["nb"], item["end"], item["dtype"],
                                                int(np.prod(shape))).reshape(shape)
                        out.extend({"type": "stream", "id": i, "data": block[r]} for r, i in enumerate(item["ids"]))
                    except RuntimeError as e:
                        out.extend({"type": "stream", "id": i, "data": e} for i in item["ids"])
                res["items"] = out
            if res.get("type") == "stream_shm":
                res = items[0]
        return res

    def _queue_out_bridge(self, loop: asyncio.AbstractEventLoop) -> None:
        while not self._queue_out_stop.is_set():
            try:
                res = self.queue_out.get(timeout=_BRIDGE_POLL_S)
            except Empty:
                proc = getattr(self, "process", None)
                if proc is not None and proc.exitcode is not None:  # the engine died: fail everything waiting
                    try:
                        loop.call_soon_threadsafe(self._queue_out_async.put_nowait,
                                                  {"type": "engine_dead", "exitcode": proc.exitcode})
                    except RuntimeError:
                        pass
                    return
                continue
            except (EOFError, OSError, ValueError):
                return
            res = self._convert(res)
            try:
                loop.call_soon_threadsafe(self._queue_out_async.put_nowait, res)
            except RuntimeError:
                return

    async def recv_queue(self) -> None:
        try:
            while True:
                self._handle(await self._queue_out_async.get())
        except asyncio.CancelledError:
            return

    def _handle(self, res: dict[str, Any]) -> None:
        if self._spin_task is not None:
            self._touch()
        if res.get("type") == "init_ok":
            self._set_tokenizer(res.get("tokenizer"), res.get("tokenizer_version", 0))
            if not self._init_fut.done():
                self._init_fut.set_result(None)
            return
        if res.get("type") == "tokenizer":  # the engine's tokenizer was replaced (e.g. a tagged adapter's)
            self._set_tokenizer(res.get("tokenizer"), res["version"])
            return
        if res.get("type") == "init_error":
            if not self._init_fut.done():
                self._init_fut.set_exception(RuntimeError(res.get("error", "unknown init error")))
            return
        if res.get("type") == "engine_dead":
            err = RuntimeError(f"VoxCPM2 engine process exited (code {res['exitcode']})")
            self._engine_error = err
            if not self._init_fut.done():
                self._init_fut.set_exception(err)
            for fut in list(self.op_table.values()):
                if not fut.done():
                    fut.set_exception(err)
            self.op_table.clear()
            for q in self.stream_table.values():
                q.put_nowait(err)
            return
        for item in res["items"] if res["type"] == "batch" else (res,):
            self._dispatch(item)

    def _dispatch(self, res: dict[str, Any]) -> None:
        if res["type"] == "stream":
            if res["id"] in self.stream_table:
                self.stream_table[res["id"]].put_nowait(res["data"])
        elif res["id"] in self.op_table:
            fut = self.op_table.pop(res["id"])
            if not fut.done():
                if res["type"] == "response":
                    fut.set_result(res["data"] if "data" in res else None)
                else:
                    fut.set_exception(RuntimeError(res["error"]))
            # else: future was cancelled (wait_for timeout etc) — skip set_result silently

    def _set_tokenizer(self, spec: str | None, version: int) -> None:
        """The engine's tokenizer (its JSON) for NANOVLLM_CLIENT_TOKENIZE; None: the engine tokenizes."""
        self._tokenizer, self._tok_version = None, version
        if spec and _CLIENT_TOKENIZE:
            try:
                from tokenizers import Tokenizer

                tok = Tokenizer.from_str(spec)
                tok.no_truncation()
                tok.no_padding()
                self._tokenizer = tok
            except Exception:  # (the engine tokenizes, as without it)
                pass

    def _tokenize(self, cmds: list[dict[str, Any]]) -> None:
        """The add_request commands' tokenizer version, ids and character offsets ("tok"), in one batched call
        (NANOVLLM_CLIENT_TOKENIZE); texts with CJK characters are left to the engine (its wrapper splits
        multi-character Chinese tokens)."""
        from nanovllm_voxcpm.models.voxcpm2.engine import _CJK

        todo = [(c, t) for c in cmds for t in (_request_text(c),) if t is not None and not _CJK.search(t)]
        if not todo:
            return
        try:
            encs = self._tokenizer.encode_batch([t for _, t in todo], add_special_tokens=False)
        except Exception:  # (the engine tokenizes them)
            return
        for (c, _), e in zip(todo, encs):
            c["tok"] = (self._tok_version, e.ids, e.offsets)

    def _flush_submits(self, more: bool = False) -> None:
        """Send the pending commands; ``more``: more commands of this event-loop turn follow (an early flush)."""
        cmds, self._pending_cmds = self._pending_cmds, []
        self._first_flush = not more  # (the next turn's commands start a new burst)
        t = time.perf_counter()
        h0 = _host_sample() if _profile.StepProfile.enabled else None
        if len(cmds) == 1 and not more:
            self.queue_in.put_nowait(cmds[0])
        elif cmds or (self._more_sent and not more):
            # (no commands: the turn's last early flush took them all; the empty batch ends the engine's admission
            # window, which otherwise waits NANOVLLM_QUEUE_MORE_MS for the rest of a turn that has ended)
            if self._tokenizer is not None:  # (a single command registers at once: the engine tokenizes it)
                self._tokenize(cmds)
            t_tok = time.perf_counter()
            self.queue_in.put_nowait({"type": "batch", "cmds": cmds, **({"more": True} if more else {})})
        if cmds or not more:
            self._more_sent = more
        if cmds and _profile.StepProfile.enabled:
            _profile.log(kind="client_flush", t=t, n=len(cmds), put_ms=(time.perf_counter() - t) * 1000,
                         tok_ms=(t_tok - t) * 1000 if len(cmds) > 1 or more else 0.0,
                         **_host_fields(h0, time.perf_counter() - t))

    _engine_error: BaseException | None = None  # set when the engine process died

    async def submit(self, cmd: str, *args: object, **kwargs: object) -> Any:
        if self._engine_error is not None:
            raise self._engine_error
        if self._spin_task is not None:
            self._touch()
        op_id = gen_uuid()
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[Any] = loop.create_future()
        self.op_table[op_id] = fut
        # queue_in is unbounded (maxsize=0); put_nowait() is instant and does not
        # consume a thread pool slot. asyncio.to_thread() here starves recv_queue
        # under high concurrent load. Commands submitted in the same event-loop
        # tick (e.g. a burst of requests) travel as one message.
        if not self._pending_cmds:
            loop.call_soon(self._flush_submits)
        self._pending_cmds.append({"id": op_id, "type": cmd, "args": args, "kwargs": kwargs})
        if _FLUSH_EVERY and len(self._pending_cmds) >= (_FLUSH_FIRST if self._first_flush and _FLUSH_FIRST
                                                        else _FLUSH_EVERY):
            # the engine registers these while the rest of the turn's requests are built; requests that trickle in
            # (an HTTP front-end parses them one by one) reach it as a steady stream instead of a few large clumps
            self._flush_submits(more=True)
        return await fut

    async def health(self) -> HealthResponse:
        return await self.submit("health")

    async def get_model_info(self) -> ModelInfoResponse:
        return await self.submit("get_model_info")

    async def wait_for_ready(self) -> None:
        while not self._init_fut.done():
            if self.process.exitcode is not None:
                if not self._init_fut.done():
                    self._init_fut.set_exception(
                        RuntimeError(f"server process exited early: exitcode={self.process.exitcode}")
                    )
                break
            await asyncio.sleep(0.05)
        await self._init_fut

    async def encode_latents(self, wav: bytes, wav_format: str) -> bytes:
        return await self.submit("encode_latents", wav, wav_format)

    async def register_voice(self, voice_id: str, latents: bytes, text: str | None = None) -> None:
        await self.submit("register_voice", voice_id, latents, text)

    async def unregister_voice(self, voice_id: str) -> None:
        await self.submit("unregister_voice", voice_id)

    async def stop(self) -> None:
        graceful_stop = False
        if self.process.exitcode is None and self.process.is_alive():
            try:
                await asyncio.wait_for(self.submit("stop"), timeout=2.0)
                graceful_stop = True
            except Exception:
                pass

        self.recv_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self.recv_task
        if self._spin_task is not None:
            self._spin_task.cancel()
        if self._reader is not None:
            with contextlib.suppress(Exception):
                asyncio.get_running_loop().remove_reader(self._reader.fileno())
        self._queue_out_stop.set()
        thread = getattr(self, "_queue_out_thread", None)
        if thread is not None:
            thread.join(timeout=1.0)
        if graceful_stop and self.process.is_alive():
            await asyncio.to_thread(self.process.join, 5.0)
        if self.process.is_alive():
            self.process.terminate()
            await asyncio.to_thread(self.process.join, 2.0)
        if self.process.is_alive():
            kill = getattr(self.process, "kill", None)
            if callable(kill):
                kill()
                await asyncio.to_thread(self.process.join, 2.0)
        for q in (getattr(self, "queue_in", None), getattr(self, "queue_out", None)):
            if q is None:
                continue
            with contextlib.suppress(Exception):
                q.close()
            with contextlib.suppress(Exception):
                q.join_thread()

    async def register_lora(self, name: str, path: str) -> RegisterLoRAResponse:
        return await self.submit("register_lora", name, path)

    async def unregister_lora(self, name: str) -> UnregisterLoRAResponse:
        return await self.submit("unregister_lora", name)

    async def list_loras(self) -> list[LoRAInfo]:
        return await self.submit("list_loras")

    async def generate(
        self,
        target_text: str,
        prompt_latents: bytes | None = None,
        prompt_text: str = "",
        max_generate_length: int = 2000,
        temperature: float = 1.0,
        cfg_value: float = 2.0,
        ref_audio_latents: bytes | None = None,
        lora_name: str | None = None,
        seed: int | None = None,
        voice: str | None = None,
        clone_mode: str | None = None,
    ) -> AsyncGenerator[Waveform, None]:
        seq_id = gen_uuid()
        self.stream_table[seq_id] = asyncio.Queue()
        is_normal_exit = False
        try:
            await self.submit(
                "add_request",
                seq_id,
                target_text,
                prompt_latents,
                prompt_text,
                max_generate_length,
                temperature,
                cfg_value,
                ref_audio_latents,
                lora_name,
                seed,
                *((voice, clone_mode) if voice is not None else ()),
            )
            while True:
                data = await self.stream_table[seq_id].get()
                if isinstance(data, BaseException):
                    # the engine is gone: nothing to cancel; otherwise (lost audio) the finally cancels the request
                    is_normal_exit = data is self._engine_error
                    raise data
                if data is None:
                    is_normal_exit = True
                    break
                yield data
        finally:
            if not is_normal_exit:
                await self.submit("cancel", seq_id)
            del self.stream_table[seq_id]


class AsyncVoxCPM2ServerPool:
    def __init__(
        self,
        model_path: str,
        inference_timesteps: int = 10,
        max_num_batched_tokens: int = 16384,
        max_num_seqs: int = 512,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.9,
        enforce_eager: bool = False,
        devices: List[int] = [],
        lora_config: Optional[LoRAConfig] = None,
        **kwargs,
    ):
        if len(kwargs) > 0:
            raise ValueError(f"Unknown kwargs: {kwargs}")
        self.servers = [
            AsyncVoxCPM2Server(
                model_path=model_path,
                inference_timesteps=inference_timesteps,
                max_num_batched_tokens=max_num_batched_tokens,
                max_num_seqs=max_num_seqs,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                enforce_eager=enforce_eager,
                devices=[device_idx],
                lora_config=lora_config,
            )
            for device_idx in devices
        ]
        self.servers_load = np.zeros(len(self.servers), dtype=np.int32)
        self._prompt_pool = {}
        self._registered_loras: set[str] = set()
        self._draining_loras: set[str] = set()

    async def wait_for_ready(self):
        await asyncio.gather(*[server.wait_for_ready() for server in self.servers])

    async def stop(self):
        await asyncio.gather(*[server.stop() for server in self.servers])

    async def encode_latents(self, wav: bytes, wav_format: str):
        min_load_server_idx = _least_loaded(self.servers_load)
        return await self.servers[min_load_server_idx].encode_latents(wav, wav_format)

    async def get_model_info(self) -> ModelInfoResponse:
        if len(self.servers) == 0:
            raise RuntimeError("server pool is empty")
        return await self.servers[0].get_model_info()

    async def add_prompt(self, wav: bytes, wav_format: str, prompt_text: str):
        prompt_id = gen_uuid()
        prompt_latents = await self.encode_latents(wav, wav_format)
        self._prompt_pool[prompt_id] = {"latents": prompt_latents, "text": prompt_text}
        return prompt_id

    async def remove_prompt(self, prompt_id: str):
        del self._prompt_pool[prompt_id]

    async def register_voice(self, voice_id: str, latents: bytes, text: str | None = None) -> None:
        """Register encoded latents (``encode_latents``) as a named voice in every engine; ``generate(voice=...)``
        then sends only the name. With a transcript the voice supports the prompt (continuation) modes."""
        await asyncio.gather(*[server.register_voice(voice_id, latents, text) for server in self.servers])

    async def unregister_voice(self, voice_id: str) -> None:
        await asyncio.gather(*[server.unregister_voice(voice_id) for server in self.servers])

    async def register_lora(self, name: str, path: str) -> RegisterLoRAResponse:
        if name in self._registered_loras or name in self._draining_loras:
            raise ValueError(f"LoRA '{name}' is already registered")
        registered_servers: list[AsyncVoxCPM2Server] = []
        try:
            for server in self.servers:
                await server.register_lora(name, path)
                registered_servers.append(server)
        except Exception:
            for server in reversed(registered_servers):
                with contextlib.suppress(Exception):
                    await server.unregister_lora(name)
            raise
        self._registered_loras.add(name)
        return RegisterLoRAResponse(name=name)

    async def unregister_lora(self, name: str) -> UnregisterLoRAResponse:
        if name not in self._registered_loras:
            raise ValueError(f"LoRA '{name}' is not registered")
        if name in self._draining_loras:
            raise ValueError(f"LoRA '{name}' is already draining")
        self._draining_loras.add(name)
        try:
            for server in self.servers:
                await server.unregister_lora(name)
        except Exception:
            raise
        self._draining_loras.remove(name)
        self._registered_loras.remove(name)
        return UnregisterLoRAResponse(name=name)

    async def list_loras(self) -> list[LoRAInfo]:
        visible_names = sorted(name for name in self._registered_loras if name not in self._draining_loras)
        return [LoRAInfo(name=name) for name in visible_names]

    async def generate(
        self,
        target_text: str,
        prompt_latents: bytes | None = None,
        prompt_text: str = "",
        prompt_id: str | None = None,
        max_generate_length: int = 2000,
        temperature: float = 1.0,
        cfg_value: float = 2.0,
        ref_audio_latents: bytes | None = None,
        lora_name: str | None = None,
        seed: int | None = None,
        voice: str | None = None,
        clone_mode: str | None = None,
    ):
        if prompt_id is not None:
            if prompt_id not in self._prompt_pool:
                raise ValueError(f"Prompt with id {prompt_id} not found")
            if prompt_latents is not None:
                raise ValueError("Prompt latents and prompt id cannot be provided at the same time")
            if len(prompt_text) > 0:
                raise ValueError("Prompt text and prompt id cannot be provided at the same time")
            prompt_info = self._prompt_pool[prompt_id]
            prompt_latents = prompt_info["latents"]
            prompt_text = prompt_info["text"]

        if lora_name is not None and (lora_name not in self._registered_loras or lora_name in self._draining_loras):
            raise ValueError(f"LoRA '{lora_name}' is not registered")

        min_load_server_idx = _least_loaded(self.servers_load)
        self.servers_load[min_load_server_idx] += 1
        server = self.servers[min_load_server_idx]
        try:
            async for data in server.generate(
                target_text,
                prompt_latents,
                prompt_text,
                max_generate_length,
                temperature,
                cfg_value,
                ref_audio_latents,
                lora_name,
                seed,
                **({"voice": voice, "clone_mode": clone_mode} if voice is not None else {}),
            ):
                yield data
        finally:
            self.servers_load[min_load_server_idx] -= 1


class SyncVoxCPM2ServerPool:
    def __init__(
        self,
        model_path: str,
        inference_timesteps: int = 10,
        max_num_batched_tokens: int = 16384,
        max_num_seqs: int = 512,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.9,
        enforce_eager: bool = False,
        devices: List[int] = [],
        lora_config: Optional[LoRAConfig] = None,
        **kwargs,
    ):
        async def init_async_server_pool():
            return AsyncVoxCPM2ServerPool(
                model_path=model_path,
                inference_timesteps=inference_timesteps,
                max_num_batched_tokens=max_num_batched_tokens,
                max_num_seqs=max_num_seqs,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                enforce_eager=enforce_eager,
                devices=devices,
                lora_config=lora_config,
                **kwargs,
            )

        self.loop = asyncio.new_event_loop()
        self.server_pool = self.loop.run_until_complete(init_async_server_pool())
        self.loop.run_until_complete(self.server_pool.wait_for_ready())

    def stop(self):
        assert self.loop is not None
        self.loop.run_until_complete(self.server_pool.stop())
        self.loop.close()
        self.loop = None

    def encode_latents(self, wav: bytes, wav_format: str):
        assert self.loop is not None
        return self.loop.run_until_complete(self.server_pool.encode_latents(wav, wav_format))

    def get_model_info(self) -> ModelInfoResponse:
        assert self.loop is not None
        return self.loop.run_until_complete(self.server_pool.get_model_info())

    def register_lora(self, name: str, path: str) -> RegisterLoRAResponse:
        assert self.loop is not None
        return self.loop.run_until_complete(self.server_pool.register_lora(name, path))

    def unregister_lora(self, name: str) -> UnregisterLoRAResponse:
        assert self.loop is not None
        return self.loop.run_until_complete(self.server_pool.unregister_lora(name))

    def list_loras(self) -> list[LoRAInfo]:
        assert self.loop is not None
        return self.loop.run_until_complete(self.server_pool.list_loras())

    def add_prompt(self, wav: bytes, wav_format: str, prompt_text: str):
        assert self.loop is not None
        return self.loop.run_until_complete(self.server_pool.add_prompt(wav, wav_format, prompt_text))

    def remove_prompt(self, prompt_id: str):
        assert self.loop is not None
        return self.loop.run_until_complete(self.server_pool.remove_prompt(prompt_id))

    def generate(
        self,
        target_text: str,
        prompt_latents: bytes | None = None,
        prompt_text: str = "",
        prompt_id: str | None = None,
        max_generate_length: int = 2000,
        temperature: float = 1.0,
        cfg_value: float = 2.0,
        ref_audio_latents: bytes | None = None,
        lora_name: str | None = None,
        seed: int | None = None,
    ):
        assert self.loop is not None
        async_gen = self.server_pool.generate(
            target_text,
            prompt_latents,
            prompt_text,
            prompt_id,
            max_generate_length,
            temperature,
            cfg_value,
            ref_audio_latents,
            lora_name,
            seed,
        )
        try:
            while True:
                yield self.loop.run_until_complete(async_gen.__anext__())
        except StopAsyncIteration:
            return
