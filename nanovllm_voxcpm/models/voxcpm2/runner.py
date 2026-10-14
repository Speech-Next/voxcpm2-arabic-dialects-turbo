import os
import random
from collections import OrderedDict
from dataclasses import dataclass
from multiprocessing.synchronize import Event

import numpy as np
import torch
import xxhash
from torch.nn.utils import remove_weight_norm

from nanovllm_voxcpm.config import Config
from nanovllm_voxcpm.engine.model_runner import BaseModelRunner, HostPack, RunnerTask
from nanovllm_voxcpm.layers.audio_vae_v2 import AudioVAEV2
from nanovllm_voxcpm.models.voxcpm2.config import VoxCPM2Config
from nanovllm_voxcpm.models.voxcpm2.model import VoxCPM2Model
from nanovllm_voxcpm.utils.loader import load_model, load_state_dict_into
from nanovllm_voxcpm.utils.profile import StepProfile
from nanovllm_voxcpm.utils.seed import _UINT64_MASK, derive_step_seed, derive_step_seeds


def _to_host(t: torch.Tensor) -> np.ndarray:
    """Copy into page-locked memory and wait: a DMA at full PCIe speed, where a pageable destination is staged by
    the driver at a fraction of it (the 256-stream decode step's int16 audio, 3.9 MB: ~3.3 -> ~0.3 ms). The returned
    array keeps its pinned tensor alive, so the caching host allocator cannot hand the buffer out again before."""
    if not t.is_cuda:
        return t.numpy()
    out = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
    out.copy_(t, non_blocking=True)
    torch.cuda.current_stream().synchronize()
    return out.numpy()


def _pinned_concat(arrays: list[np.ndarray]) -> torch.Tensor:
    """np.concatenate straight into page-locked memory, so the copy to the GPU is an async DMA (a pageable source
    is staged through a driver buffer, synchronously). The caching host allocator keeps the buffer until the copy
    has run."""
    first = arrays[0]
    rows = sum(a.shape[0] for a in arrays)
    out = torch.empty((rows, *first.shape[1:]), dtype=torch.from_numpy(first[:0]).dtype,
                      pin_memory=torch.cuda.is_available())
    np.concatenate(arrays, axis=0, out=out.numpy())
    return out


# NANOVLLM_KERNEL_TRACE=<file>: a torch.profiler kernel table of prefill steps with at least
# NANOVLLM_KERNEL_TRACE_MIN_BS sequences (default 128), the first NANOVLLM_KERNEL_TRACE_SKIP (default 2) skipped
_TRACE_PATH = os.environ.get("NANOVLLM_KERNEL_TRACE")
_trace_seen = [0]


def _kernel_trace(is_prefill: bool, bs: int):
    if not _TRACE_PATH or not is_prefill or bs < int(os.environ.get("NANOVLLM_KERNEL_TRACE_MIN_BS", "128")):
        return None
    _trace_seen[0] += 1
    skip = int(os.environ.get("NANOVLLM_KERNEL_TRACE_SKIP", "2"))
    if _trace_seen[0] <= skip or _trace_seen[0] > skip + 1:
        return None
    from torch.profiler import ProfilerActivity, profile

    return profile(activities=[ProfilerActivity.CUDA])


def _dump_kernel_trace(trace) -> None:
    with open(_TRACE_PATH, "w") as f:
        f.write(trace.key_averages().table(sort_by="cuda_time_total", row_limit=60, max_name_column_width=70))
    trace.export_chrome_trace(_TRACE_PATH + ".json")


@dataclass
class VoxCPM2Payload:
    # prefill: the uncached tail's ids / masks / LocEnc keys as lists or arrays (the runner converts the whole step's
    # at once); decode: arrays
    text_tokens: np.ndarray | list | None = None
    feats: np.ndarray | None = None
    feat_masks: np.ndarray | list | None = None
    temperature: float = 1.0
    cfg_value: float = 1.0
    padding_decode: np.ndarray | None = None
    seed: int | None = None
    seed_step: int = 0
    prefix_src_block: int | None = None  # ref_prefix_cache hit: block holding the prefix KV
    prefix_len: int = 0
    feat_keys: np.ndarray | list | None = None  # uint64 per position, LocEnc cache keys (0 = not cacheable)


_SILENCE_RMS = 10 ** (-50 / 20)  # -50 dBFS


def _patch_silent(wave) -> bool:
    """A patch's audio (float in [-1, 1] or int16 PCM) below -50 dBFS RMS."""
    x = np.asarray(wave)
    if x.size == 0:
        return True
    x = x.astype(np.float32) / (32768.0 if x.dtype == np.int16 else 1.0)
    return float(np.sqrt(np.mean(x * x))) < _SILENCE_RMS


# NANOVLLM_FIRST_AUDIO_FRAMES=k: a prefill step's patch is decoded and sent as its first k latent frames (40 ms of
# audio each), then the rest; 0 = in one piece
_FIRST_AUDIO_FRAMES = int(os.environ.get("NANOVLLM_FIRST_AUDIO_FRAMES", "1"))


class VoxCPM2Runner(BaseModelRunner):
    _fast_noise = False  # turbo.fast_noise (set in __init__)
    _pcm16 = False  # turbo.pcm16_output
    model: VoxCPM2Model
    dit_lora_seq_len_offset = 3
    vae_dtype = torch.float16

    def __init__(
        self,
        config: Config[VoxCPM2Config],
        rank: int,
        device_idx: int,
        distributed_port: int | None,
        event: Event | list[Event],
    ):
        self.inference_timesteps = config.model_config.inference_timesteps
        self.feat_dim = config.model_config.feat_dim
        self.patch_size = config.model_config.patch_size
        self.lora_config = config.lora_config
        self._fast_noise = config.model_config.turbo.fast_noise
        self._pcm16 = config.model_config.turbo.pcm16_output
        super().__init__(config, rank, device_idx, distributed_port, event)

    @property
    def dtype(self) -> torch.dtype:
        return torch.bfloat16

    def _set_cfg_branches(self) -> None:
        """The LocDiT batch per sample: 2 rows (CFG) or 1 (a distilled estimator: every step conditional-only). The
        DiT LoRA token slots (``expand_dit_lora_slots``) are laid out for it, so steps must not mix the two."""
        cfm = self.model.feat_decoder
        steps = (cfm.cfg_steps, cfm.first_cfg_steps)  # -1: CFG on every step, 0: on none, k: on the first k
        if steps == (0, 0):
            self.cfg_branches = 1
        elif steps != (-1, -1) and self.lora_config is not None and self.lora_config.enable_dit:
            raise NotImplementedError(
                f"LocDiT LoRA needs CFG on every step or on none (cfg_steps, first_cfg_steps = {steps}): "
                "its token slots are laid out for one DiT batch size"
            )

    def init_model(self, model_config: VoxCPM2Config, model_path: str):
        self.model = VoxCPM2Model(model_config, self.inference_timesteps, lora_config=self.lora_config)
        load_model(self.model, model_path)
        turbo = model_config.turbo
        if turbo.dit_checkpoint:
            ckpt = torch.load(turbo.dit_checkpoint, map_location="cpu", weights_only=True)
            if ckpt.get("dit_layers"):  # a student with fewer decoder layers (some of the teacher's, renumbered)
                dec = self.model.feat_decoder.estimator.decoder
                dec.layers = torch.nn.ModuleList(list(dec.layers)[: int(ckpt["dit_layers"])])
            n = load_state_dict_into(self.model, ckpt["state_dict"], prefix="feat_decoder.estimator.")
            if turbo.dit_t_span is None and "schedule" in ckpt:
                t_span = [float(ckpt["schedule"][0][0])] + [float(t - dt) for t, dt in ckpt["schedule"]]
                self.model.feat_decoder.set_t_span(t_span)
            if ckpt.get("mean_mode"):  # a student trained with its step sizes as input (the estimator's dt)
                self.model.feat_decoder.mean_mode = True
            print(f"[turbo] loaded {n} distilled LocDiT tensors from {turbo.dit_checkpoint}", flush=True)
        if turbo.dit_first_t_span is not None:
            self.model.feat_decoder.first_t_span_override = torch.tensor(turbo.dit_first_t_span, dtype=torch.float32)
        self._set_cfg_branches()
        if turbo.fused_layers:
            from voxcpm_kernels.fused_layer import fuse_cpm4_layers

            fused, quantized = fuse_cpm4_layers(
                self.model,
                quant_prefixes=turbo.w8a8_targets if turbo.w8a8 else [],
                fuse_prefixes=["base_lm", "residual_lm", "feat_encoder", "feat_decoder"],
                min_rows=turbo.w8a8_min_rows,
            )
            print(f"[turbo] fused {fused} decoder layers ({quantized} W8A8 for rows >= {turbo.w8a8_min_rows})", flush=True)
        elif turbo.w8a8:
            from voxcpm_kernels import quantize_linears

            n = quantize_linears(self.model, turbo.w8a8_targets, min_rows=turbo.w8a8_min_rows)
            print(f"[turbo] W8A8 on {n} linear layers (rows >= {turbo.w8a8_min_rows})", flush=True)

        torch.set_default_dtype(torch.float32)
        self.vae = AudioVAEV2(config=model_config.audio_vae_config)
        vae_state_dict = torch.load(os.path.join(model_path, "audiovae.pth"))["state_dict"]
        self.vae.load_state_dict(vae_state_dict)
        self._prepare_vae_decoder_for_inference()
        if model_config.turbo.vae_kernel:
            from voxcpm_kernels import SlotStreamingVAEDecoder

            self.vae_streaming_decoder = SlotStreamingVAEDecoder(
                self.vae.decoder,
                max_streams=self._config.max_num_seqs,
                latent_dim=self.feat_dim,
                chunk_frames=self.patch_size,
                fused=model_config.turbo.vae_fused,
                # the first audio of a prefill step's patch and the rest of it, graphed too
                extra_chunk_frames=(_FIRST_AUDIO_FRAMES, self.patch_size - _FIRST_AUDIO_FRAMES)
                if 0 < _FIRST_AUDIO_FRAMES < self.patch_size else (),
            )
        else:
            self.vae_streaming_decoder = self.vae.streaming_decoder(self._config.max_num_seqs)
        self.vae_streaming_decoder.warmup(self.feat_dim, self.patch_size)
        torch.set_default_dtype(torch.bfloat16)

        # LocEnc embedding cache: rows [0, size) keyed by patch hash (LRU), then scratch rows for uncached patches
        self._enc_cache = None
        if turbo.locenc_cache_size > 0 and turbo.skip_text_locenc:
            # one step's distinct patches must fit without evicting each other
            self._enc_cache_size = max(turbo.locenc_cache_size, self._config.max_num_batched_tokens)
            self._enc_rows: OrderedDict[int, int] = OrderedDict()
            self._enc_cache = torch.zeros(
                self._enc_cache_size + self._config.max_num_batched_tokens,
                model_config.lm_config.hidden_size,
                dtype=torch.bfloat16,
                device="cuda",
            )

    def _prepare_vae_decoder_for_inference(self) -> None:
        """Fold weight norm and cast only the inference decoder to FP16."""
        for module in self.vae.decoder.modules():
            if hasattr(module, "weight_g"):
                remove_weight_norm(module)
        self.vae.decoder.to(dtype=self.vae_dtype)

    def make_dummy_inputs(self, batch_size: int, length: int) -> dict[str, torch.Tensor]:
        return {
            "text_tokens": torch.zeros(batch_size * length, dtype=torch.int64),
            "feat": torch.zeros(batch_size * length, self.patch_size, self.feat_dim),
            "feat_mask": torch.zeros(batch_size * length, dtype=torch.bool),
            "temperature": torch.zeros(batch_size),
            "cfg_value": torch.zeros(batch_size),
            "z_noise": torch.zeros(batch_size, self.feat_dim, self.patch_size, dtype=self.dtype),
        }

    def make_dummy_outputs(self, batch_size: int) -> dict[str, torch.Tensor]:
        return {
            "latents": torch.zeros(batch_size, self.patch_size, self.feat_dim, dtype=self.dtype),
            "stop_flag": torch.zeros(batch_size, dtype=torch.int64),
        }

    def encode_latents(self, wav: torch.Tensor) -> np.ndarray:
        assert wav.ndim == 2, "Invalid shape of wav"
        wav = wav.to(torch.float32).cuda()
        return (
            self.vae.encode(wav, self.vae.sample_rate)
            .permute(0, 2, 1)
            .view(-1, self.feat_dim)
            .to(torch.float32)
            .cpu()
            .numpy()
        )

    def _kv_caches(self) -> list[torch.Tensor]:
        caches = getattr(self, "_kv_cache_list", None)
        if caches is None:
            from nanovllm_voxcpm.layers.attention import Attention

            caches = []
            for module in self.model.modules():
                if isinstance(module, Attention) and module.is_causal and module.k_cache.numel():
                    caches += [module.k_cache, module.v_cache]
            self._kv_cache_list = caches
        return caches

    def copy_kv_blocks(self, pairs: list[tuple[int, int]], n_tokens: int | list[int]) -> None:
        """For every (src, dst) block pair copy the first n_tokens slots (one count, or one per pair) of each
        causal layer's K/V."""
        if not pairs:
            return
        counts = n_tokens if isinstance(n_tokens, list) else [n_tokens] * len(pairs)
        if getattr(self, "kv_cache", None) is not None:  # all layers in one tensor: one launch
            from voxcpm_kernels.kv_copy import copy_kv_block_prefix

            copy_kv_block_prefix(self.kv_cache, [(s, d, n) for (s, d), n in zip(pairs, counts)])
            return
        n_tokens = max(counts)
        src = torch.tensor([p[0] for p in pairs], dtype=torch.long, device="cuda")
        dst = torch.tensor([p[1] for p in pairs], dtype=torch.long, device="cuda")
        n = min(n_tokens, self.block_size)
        for cache in self._kv_caches():
            cache[dst, :n] = cache[src, :n]

    def _locenc_inputs(self, seqs: list[RunnerTask[VoxCPM2Payload]]) -> dict[str, torch.Tensor]:
        """Cache rows for every audio position of a prefill batch; LocEnc runs only for first-seen patches."""
        pack = HostPack(getattr(self, "pack_device", "cuda"))
        finish = self._stage_locenc_inputs(seqs, pack)
        return finish(pack.upload())

    @staticmethod
    def _step_feat_keys(seqs: list[RunnerTask[VoxCPM2Payload]]) -> np.ndarray:
        """The step's LocEnc keys, one per position (0 where a sequence has none)."""
        keys: list = []
        for seq in seqs:
            payload = seq.custom_payload
            keys.extend(payload.feat_keys if payload.feat_keys is not None else [0] * len(payload.feat_masks))
        return np.array(keys, dtype=np.uint64)

    def _stage_locenc_inputs(self, seqs: list[RunnerTask[VoxCPM2Payload]], pack: HostPack, feat_mask=None):
        """_locenc_inputs' index arrays staged in ``pack``; returns finish(uploaded) -> the inputs. feat_mask: the
        step's concatenated masks (built here when not given)."""
        if feat_mask is None:
            feat_mask = np.array([m for seq in seqs for m in seq.custom_payload.feat_masks], dtype=np.bool_)
        pos = np.flatnonzero(feat_mask)  # every audio position of the step
        keys = self._step_feat_keys(seqs)[pos].tolist()
        cache = self._enc_rows
        for k in keys:  # hits first: they must survive this step's evictions
            if k in cache:
                cache.move_to_end(k)
        rows, miss, fresh, scratch = [], [], {}, self._enc_cache_size
        for i, k in enumerate(keys):
            row = cache.get(k) if k else None
            if row is None and k in fresh:
                row = fresh[k]
            if row is None:
                if not k:
                    row, scratch = scratch, scratch + 1
                elif len(cache) < self._enc_cache_size:
                    row = len(cache)
                else:
                    _, row = cache.popitem(last=False)
                if k:
                    cache[k] = fresh[k] = row
                miss.append(i)
            rows.append(row)

        i_pos = pack.add(pos.astype(np.int64))
        i_rows = pack.add(np.asarray(rows, dtype=np.int64))
        i_miss = pack.add(np.asarray(miss, dtype=np.int64))
        return lambda views: {"enc_pos": views[i_pos], "enc_rows": views[i_rows], "enc_miss": views[i_miss],
                              "enc_cache": self._enc_cache}

    def run(self, seqs: list[RunnerTask[VoxCPM2Payload]], is_prefill: bool):
        trace = _kernel_trace(is_prefill, len(seqs))
        if trace is None:
            return self._run(seqs, is_prefill)
        with trace:
            out = self._run(seqs, is_prefill)
            torch.cuda.synchronize()
        _dump_kernel_trace(trace)
        return out

    def _noise_seeds(self, seqs: list[RunnerTask[VoxCPM2Payload]]) -> np.ndarray:
        """Every row's noise seed for seeded_randn (& SEED_MASK, int64): derive_step_seed(seed, step) for seeded
        requests (numpy, one pass: the per-row Python version was ~0.2 ms at 200 rows), random bits otherwise (drawn
        in row order, as before)."""
        from voxcpm_kernels.noise import SEED_MASK

        payloads = [seq.custom_payload for seq in seqs]
        seeded = [i for i, p in enumerate(payloads) if p.seed is not None and p.seed >= 0]
        out = np.zeros(len(payloads), dtype=np.uint64)
        if seeded:
            out[seeded] = derive_step_seeds([int(payloads[i].seed) & _UINT64_MASK for i in seeded],
                                            [int(payloads[i].seed_step) for i in seeded])
        if len(seeded) < len(payloads):
            ok = set(seeded)
            for i in range(len(payloads)):
                if i not in ok:
                    out[i] = random.getrandbits(63)
        return (out & np.uint64(SEED_MASK)).astype(np.int64)

    def _run(self, seqs: list[RunnerTask[VoxCPM2Payload]], is_prefill: bool):
        if is_prefill:
            return self._run_prefill(seqs)
        prof = StepProfile(kind="decode", bs=len(seqs)).start()
        positions = self.prepare_decode_context(seqs)
        prof.mark("ctx")
        inputs = {"positions": positions}

        text_tokens = []
        feats = []
        feat_masks = []
        temperatures = []
        cfg_values = []
        for seq in seqs:
            payload = seq.custom_payload
            assert payload.text_tokens.shape[0] == payload.feats.shape[0]
            assert payload.text_tokens.shape[0] == payload.feat_masks.shape[0]
            text_tokens.append(payload.text_tokens)
            feats.append(payload.feats)
            feat_masks.append(payload.feat_masks)
            temperatures.append(payload.temperature)
            cfg_values.append(payload.cfg_value)

        inputs["text_tokens"] = _pinned_concat(text_tokens).cuda(non_blocking=True)
        inputs["feat"] = _pinned_concat(feats).cuda(non_blocking=True).to(self.dtype)
        inputs["feat_mask"] = _pinned_concat(feat_masks).cuda(non_blocking=True)
        inputs["temperature"] = (
            torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True).to(self.dtype)
        )
        inputs["cfg_value"] = (
            torch.tensor(cfg_values, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True).to(self.dtype)
        )
        prof.mark("h2d")
        return self._with_noise(seqs, inputs, False, prof)

    def _run_prefill(self, seqs: list[RunnerTask[VoxCPM2Payload]]):
        """A prefill step: the attention and LoRA contexts and every input (tokens, features, masks, sampling
        parameters, LocEnc indices, noise seeds) are built on the host into one page-locked buffer and reach the GPU
        in one copy (the per-tensor copies were ~17 before the first kernel). Same values, dtypes and shapes as the
        per-tensor path: each input is a view into the one device buffer."""
        prof = StepProfile(kind="prefill", bs=len(seqs)).start()
        hits = [(s.custom_payload.prefix_src_block, s.block_table[0], s.custom_payload.prefix_len)
                for s in seqs if s.custom_payload.prefix_src_block is not None]
        if hits:  # cached reference prefixes -> this step's sequences, before they attend to them (unless the engine
            # copied them at admission already: VoxCPM2Engine.on_prefill_admitted)
            self.copy_kv_blocks([(src, dst) for src, dst, _ in hits], [n for _, _, n in hits])
        pack = HostPack(getattr(self, "pack_device", "cuda"))
        finish_ctx = self.stage_prefill_context(seqs, pack)
        prof.mark("ctx")

        # the tails' ids and masks of the whole step in one conversion each (the engine hands them over as lists)
        text_tokens, feat_masks, feats, temperatures, cfg_values = [], [], [], [], []
        for seq in seqs:
            payload = seq.custom_payload
            n = len(payload.text_tokens)
            assert n == payload.feats.shape[0] and n == len(payload.feat_masks)
            text_tokens.extend(payload.text_tokens)
            feat_masks.extend(payload.feat_masks)
            feats.append(payload.feats)
            temperatures.append(payload.temperature)
            cfg_values.append(payload.cfg_value)
        feat_mask = np.array(feat_masks, dtype=np.bool_)
        self._prefill_has_audio = bool(feat_mask.any())
        i_tok = pack.add(np.array(text_tokens, dtype=np.int64))
        i_feat, i_mask = pack.add_concat(feats), pack.add(feat_mask)
        i_temp = pack.add(np.asarray(temperatures, dtype=np.float32))
        i_cfg = pack.add(np.asarray(cfg_values, dtype=np.float32))
        finish_enc = None
        if self._prefill_has_audio and getattr(self, "_enc_cache", None) is not None:
            finish_enc = self._stage_locenc_inputs(seqs, pack, feat_mask)
        seeds = i_seed = None
        if self._fast_noise:
            seeds = self._noise_seeds(seqs)
            i_seed = pack.add(seeds)
        views = pack.upload()
        inputs = {"positions": finish_ctx(views)}
        inputs["text_tokens"] = views[i_tok]
        inputs["feat"] = views[i_feat].to(self.dtype)
        inputs["feat_mask"] = views[i_mask]
        inputs["temperature"] = views[i_temp].to(self.dtype)
        inputs["cfg_value"] = views[i_cfg].to(self.dtype)
        if finish_enc is not None:
            inputs.update(finish_enc(views))
        prof.mark("h2d")
        return self._with_noise(seqs, inputs, True, prof, seeds, views[i_seed] if i_seed is not None else None)

    def _with_noise(self, seqs, inputs, is_prefill, prof, seeds=None, seed_t=None):
        """The step's LocDiT noise (seeded per row), then the model."""
        bsz = len(seqs)
        if self._fast_noise:
            from voxcpm_kernels import seeded_randn

            if seeds is None:
                seeds = self._noise_seeds(seqs)
            if seed_t is None:
                seed_t = torch.from_numpy(seeds).pin_memory().cuda(non_blocking=True)
            inputs["z_noise"] = seeded_randn(seeds, (self.feat_dim, self.patch_size), dtype=self.dtype, seed_t=seed_t)
            return self._run_with_inputs(seqs, inputs, is_prefill, prof)
        seeded_rows = [
            (i, int(seq.custom_payload.seed), seq.custom_payload.seed_step)
            for i, seq in enumerate(seqs)
            if seq.custom_payload.seed is not None and seq.custom_payload.seed >= 0
        ]
        if len(seeded_rows) == bsz:
            z_noise = torch.empty((bsz, self.feat_dim, self.patch_size), dtype=self.dtype, device="cuda")
        else:
            z_noise = torch.randn((bsz, self.feat_dim, self.patch_size), dtype=self.dtype, device="cuda")

        for i, seed_val, seed_step in seeded_rows:
            generator = torch.Generator(device="cuda").manual_seed(derive_step_seed(seed_val, seed_step))
            z_noise[i] = torch.randn(
                (self.feat_dim, self.patch_size), generator=generator, dtype=self.dtype, device="cuda"
            )

        inputs["z_noise"] = z_noise
        return self._run_with_inputs(seqs, inputs, is_prefill, prof)

    def _wave_rows(self, vae_decoded: torch.Tensor) -> list:
        """Decoded audio [B, 1, T] -> one host array per row (int16 PCM with pcm16_output)."""
        wav = vae_decoded[:, 0, :]
        if self._pcm16:
            wav = (wav.float().clamp_(-1.0, 1.0) * 32767.0).round_().to(torch.int16)
        out = _to_host(wav.to(torch.float32 if not self._pcm16 else torch.int16))
        return [out[i] for i in range(out.shape[0])]

    def _run_with_inputs(self, seqs, inputs, is_prefill, prof):
        if prof.enabled:
            prof.rec["tokens"] = int(inputs["text_tokens"].numel())
        prof.mark("prep")

        outputs = self.run_model(inputs, is_prefill)
        latents = outputs["latents"]
        hook = getattr(self, "overlap_hook", None)
        if prof.enabled:  # marks synchronize, so profiled steps report the hook separately
            prof.mark("dit" if is_prefill else "model")
            if hook is not None:
                hook()
                prof.mark("hook")
        elif hook is not None:
            hook()  # CPU work (e.g. registering a burst of requests) while the GPU runs this step

        full_waveforms = None  # the whole patch's audio when the step streams it in parts
        if all(getattr(seq, "seq_id", None) is not None for seq in seqs):
            dec = self.vae_streaming_decoder
            has_stream = getattr(dec, "has_stream", None)  # voxcpm_kernels decoder: contexts only for new streams
            initial_contexts, context_keys = [], []
            for seq in seqs:
                padding_decode = seq.custom_payload.padding_decode
                if padding_decode is None or (has_stream is not None and has_stream(seq.seq_id)):
                    initial_contexts.append(None)
                    context_keys.append(None)
                    continue
                initial_contexts.append(
                    torch.from_numpy(padding_decode)
                    .to(device="cuda", dtype=self.vae_dtype, non_blocking=True)
                    .transpose(0, 1)
                    .unsqueeze(0)
                )
                # equal prompt-audio tails (same cloned voice) prime the decoder to the same state
                context_keys.append(xxhash.xxh64_intdigest(np.ascontiguousarray(padding_decode).tobytes()))
            kw = {"context_keys": context_keys} if has_stream is not None else {}
            ids = [seq.seq_id for seq in seqs]
            z = latents.to(self.vae_dtype).permute(0, 2, 1)
            emit = getattr(self, "emit_hook", None)
            if is_prefill and emit is not None and has_stream is not None and 0 < _FIRST_AUDIO_FRAMES < self.patch_size:
                # Time to first audio: a prefill step's patch is decoded in two pieces (the first
                # NANOVLLM_FIRST_AUDIO_FRAMES frames, then the rest) and the first piece's audio leaves at once
                # (emit_hook). The VAE is causal and keeps each stream's state, so the pieces are the patch's audio;
                # one frame (40 ms of audio) costs a quarter of the patch's VAE time before the first audio.
                half = _FIRST_AUDIO_FRAMES
                first = self._wave_rows(dec.decode_chunks(z[..., :half].contiguous(), ids, initial_contexts, **kw))
                emit(ids, first)
                prof.mark("vae_first_half")
                vae_decoded = dec.decode_chunks(z[..., half:].contiguous(), ids)
            else:
                first = None
                vae_decoded = dec.decode_chunks(z, ids, initial_contexts, **kw)
            prof.mark("vae")
            ret_waveforms = self._wave_rows(vae_decoded)
            # the whole patch's audio, for checks on it (head start's silence test); the step streams only the rest
            full_waveforms = ret_waveforms if first is None else [np.concatenate(p) for p in zip(first, ret_waveforms)]
            prof.mark("d2h")
        else:
            pad_lengths = [
                seq.custom_payload.padding_decode.shape[0] if seq.custom_payload.padding_decode is not None else 0
                for seq in seqs
            ]
            max_pad_decode = max(pad_lengths) + self.patch_size
            vae_decoder_inputs = torch.zeros(
                len(seqs),
                max_pad_decode,
                self.feat_dim,
                dtype=self.vae_dtype,
                device="cuda",
            )
            for i, seq in enumerate(seqs):
                pad_len = pad_lengths[i]
                if pad_len > 0:
                    vae_decoder_inputs[i, :pad_len] = torch.from_numpy(seq.custom_payload.padding_decode).cuda(
                        non_blocking=True
                    )
                vae_decoder_inputs[i, pad_len : pad_len + self.patch_size] = latents[i].to(self.vae_dtype)
            vae_decoder_outputs = self.vae.decode(vae_decoder_inputs.permute(0, 2, 1))[:, 0, :]
            if isinstance(vae_decoder_outputs, torch.Tensor):
                vae_decoder_outputs = vae_decoder_outputs.to(torch.float32)
            vae_decoder_outputs = vae_decoder_outputs.cpu().numpy()
            ret_waveforms = []
            for i, pad_len in enumerate(pad_lengths):
                ret_waveforms.append(
                    vae_decoder_outputs[
                        i,
                        pad_len
                        * self.vae.decoder_chunk_size : (pad_len + self.patch_size)
                        * self.vae.decoder_chunk_size,
                    ]
                )

        stop_flag = _to_host(outputs["stop_flag"]).tolist()

        np_latents = _to_host(latents.to(torch.float32))
        prof.mark("tail")
        prof.write()
        full = full_waveforms or ret_waveforms
        return [
            {"latents": np_latents[i], "stop_flag": stop_flag[i], "waveforms": ret_waveforms[i],
             "patch_silent": _patch_silent(full[i])}
            for i in range(len(seqs))
        ]
