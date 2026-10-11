"""Slot-based streaming decoder for the VoxCPM2 AudioVAE (CUDA-graph friendly).

The reference streaming decoder keeps causal-conv state in per-layer Python
dicts keyed by stream id, and concatenates/slices state with a Python loop per
stream and layer (~40 layers x B streams of tiny kernels per step). Here every
stateful layer owns one state tensor ``[max_slots + 1, C, pad]``; a stream gets
a slot index, and a whole batch gathers / scatters its state with one
``index_select`` / ``index_copy_`` per layer. Shapes are static, so each batch
size bucket is captured once as a CUDA graph and replayed.

The last slot is a scratch slot that padded rows of a bucket read and write.
"""

import os
from collections import OrderedDict
from collections.abc import Hashable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .resunit import (CudaResidualUnit, DirectResidualUnit, FusedResidualUnit, OnePassResidualUnit, SnakeConvTanhHead,
                      SplitResidualUnit, TwoPassResidualUnit)
from .snake import Snake
from .tconv import FusedSnakeTransposeConv


class _CausalConv:
    def __init__(self, mod: nn.Conv1d, slots: int):
        self.w = mod.weight.detach()
        self.b = None if mod.bias is None else mod.bias.detach()
        self.dilation = mod.dilation[0]
        self.groups = mod.groups
        self.pad = mod._CausalConv1d__padding * 2 - getattr(mod, "_CausalConv1d__output_padding", 0)
        assert mod.stride[0] == 1 and mod.padding[0] == 0
        self.state = (
            torch.zeros(slots, mod.in_channels, self.pad, device=self.w.device, dtype=self.w.dtype)
            if self.pad > 0
            else None
        )

    def __call__(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        if self.state is not None:
            x = torch.cat([self.state.index_select(0, idx), x], dim=-1)
            self.state.index_copy_(0, idx, x[..., -self.pad :])
        return F.conv1d(x, self.w, self.b, dilation=self.dilation, groups=self.groups)


class _CausalTransposeConv:
    def __init__(self, mod: nn.ConvTranspose1d, slots: int):
        self.w = mod.weight.detach()
        self.b = None if mod.bias is None else mod.bias.detach()
        self.stride = mod.stride[0]
        self.groups = mod.groups
        self.trim = mod._CausalTransposeConv1d__padding * 2 - mod._CausalTransposeConv1d__output_padding
        self.ctx = mod.kernel_size[0] // mod.stride[0] - 1
        assert mod.padding[0] == 0 and mod.output_padding[0] == 0 and mod.dilation[0] == 1
        self.state = (
            torch.zeros(slots, mod.in_channels, self.ctx, device=self.w.device, dtype=self.w.dtype)
            if self.ctx > 0
            else None
        )

    def __call__(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        if self.state is not None:
            full = torch.cat([self.state.index_select(0, idx), x], dim=-1)
            self.state.index_copy_(0, idx, x[..., -self.ctx :])
        else:
            full = x
        out = F.conv_transpose1d(full, self.w, self.b, stride=self.stride, groups=self.groups)
        left = self.ctx * self.stride
        return out[..., left : out.size(-1) - self.trim]


class _SnakeOp:
    def __init__(self, mod: nn.Module, affine=None):
        self.fn = Snake(mod.alpha, *(affine or (None, None)))

    def __call__(self, x, idx):
        return self.fn(x)


class _ResidualUnit:
    def __init__(self, mod: nn.Module, slots: int):
        snake1, conv7, snake2, conv1 = list(mod.block)
        self.ops = [_SnakeOp(snake1), _CausalConv(conv7, slots), _SnakeOp(snake2), _CausalConv(conv1, slots)]

    def __call__(self, x, idx):
        y = x
        for op in self.ops:
            y = op(y, idx)
        return x + y


class _Tanh:
    def __call__(self, x, idx):
        return torch.tanh(x)


# Residual-unit implementation per channel count (tuned on an A10, B=256, power cap): "onepass" (depthwise
# conv, snakes, 1x1 conv and residual in one Triton kernel) for the narrow late blocks, "twopass" (depthwise conv +
# snakes, then a Triton 1x1 conv with bias and residual in its epilogue) from 128 channels on. "direct" (the same
# depthwise kernel, then cuBLAS baddbmm into an output pre-set to x + bias) and the others stay for A/B.
UNIT_KIND = {32: "onepass", 64: "onepass", 128: "twopass", 256: "twopass", 512: "twopass", 1024: "twopass"}
if os.environ.get("VOXCPM_KERNELS_ONEPASS", "1") == "0":  # A/B: the earlier split units everywhere
    UNIT_KIND = {c: "direct" for c in UNIT_KIND}
# "cuda": one-pass CUDA C++ unit (cuda/resunit_fused.cu, smem-staged snake + depthwise conv, tensor-core 1x1 conv) for
# these channel counts; falls back to the Triton units when the extension cannot be built
for _c in filter(None, os.environ.get("VOXCPM_KERNELS_RESUNIT_CUDA", "32,64,128,256").split(",")):
    UNIT_KIND[int(_c)] = "cuda"


# Snake of a transposed conv in the epilogue of the residual unit before it; VOXCPM_KERNELS_POST_SNAKE=0 keeps it apart.
POST_SNAKE = os.environ.get("VOXCPM_KERNELS_POST_SNAKE", "1") == "1"

# Fused output head (snake -> 1-channel conv -> tanh); VOXCPM_KERNELS_HEAD=0 keeps the three ops.
HEAD_FUSED = os.environ.get("VOXCPM_KERNELS_HEAD", "1") == "1"

# Fused snake + transposed conv (tconv.py); VOXCPM_KERNELS_TCONV=0 falls back to snake + cuDNN.
TCONV_FUSED = os.environ.get("VOXCPM_KERNELS_TCONV", "1") == "1"


def _make_residual_unit(unit: nn.Module, slots: int, fused: bool):
    kind = UNIT_KIND.get(unit.block[1].in_channels, "plain") if fused else "plain"
    if kind == "fused":
        return FusedResidualUnit(unit, slots)
    if kind == "split":
        return SplitResidualUnit(unit, slots)
    if kind == "direct":
        return DirectResidualUnit(unit, slots)
    if kind == "onepass":
        return OnePassResidualUnit(unit, slots)
    if kind == "twopass":
        return TwoPassResidualUnit(unit, slots)
    if kind == "cuda":
        from .cuda_ext import available

        if available("resunit_fused"):
            return CudaResidualUnit(unit, slots)
        return TwoPassResidualUnit(unit, slots) if unit.block[1].in_channels >= 128 else OnePassResidualUnit(unit, slots)
    return _ResidualUnit(unit, slots)


def _fuse_post_snakes(ops: list) -> None:
    """A residual unit whose output only feeds a fused (affine +) snake + transposed conv applies that snake in its
    epilogue (saves a read and a write of the activation); the conv then takes its input as already snaked."""
    if not POST_SNAKE:
        return
    for prev, op in zip(ops, ops[1:]):
        if isinstance(op, FusedSnakeTransposeConv) and isinstance(prev, (OnePassResidualUnit, TwoPassResidualUnit, CudaResidualUnit)) \
                and op.snake.fast_sin:
            sn = op.snake
            scale = sn.scale if sn.scale is not None else torch.ones_like(sn.alpha)
            bias = sn.bias if sn.bias is not None else torch.zeros_like(sn.alpha)
            prev.post_snake = (sn.alpha, sn.inv_alpha, scale.contiguous(), bias.contiguous())
            op.pre_snaked = True


def _build_ops(decoder: nn.Module, slots: int, fused: bool = True) -> list:
    """Flatten a CausalDecoder into streaming ops, folding sample-rate scale/bias into snakes."""
    layers = list(decoder.model)
    conds = list(decoder.sr_cond_model) if getattr(decoder, "sr_bin_boundaries", None) is not None else [None] * len(layers)
    ops, affine = [], None
    for layer, cond in zip(layers, conds):
        if cond is not None:
            if cond.cond_type not in ("scale_bias", "scale_bias_init") or not isinstance(cond.out_layer, nn.Identity):
                raise NotImplementedError(f"sample-rate conditioning {cond.cond_type!r} is not supported")
            sr = decoder.default_sr_idx
            affine = (cond.scale_embed.weight[sr], cond.bias_embed.weight[sr])
        name = type(layer).__name__
        if name == "CausalConv1d":
            ops.append(_CausalConv(layer, slots))
        elif name == "CausalDecoderBlock":
            parts = list(layer.block)
            if type(parts[0]).__name__ != "Snake1d" or type(parts[1]).__name__ != "CausalTransposeConv1d":
                raise NotImplementedError("unexpected decoder block layout")
            if fused and TCONV_FUSED:
                ops.append(FusedSnakeTransposeConv(parts[0], parts[1], slots, affine))
            else:
                ops.append(_SnakeOp(parts[0], affine))
                ops.append(_CausalTransposeConv(parts[1], slots))
            affine = None
            for unit in parts[2:]:
                if type(unit).__name__ != "CausalResidualUnit":
                    raise NotImplementedError(f"unsupported layer {type(unit).__name__} (noise block?)")
                ops.append(_make_residual_unit(unit, slots, fused))
        elif name == "Snake1d":
            ops.append(_SnakeOp(layer))
        elif name == "Tanh":
            ops.append(_Tanh())
        else:
            raise NotImplementedError(f"unsupported decoder layer {name}")
        if affine is not None and name != "CausalDecoderBlock" and cond is None:
            raise NotImplementedError("sample-rate affine not followed by a decoder block")
    _fuse_post_snakes(ops)
    # output head: snake -> 1-channel causal conv -> tanh as one kernel
    if fused and HEAD_FUSED and len(layers) >= 3 and [type(m).__name__ for m in layers[-3:]] == ["Snake1d", "CausalConv1d", "Tanh"] \
            and layers[-2].out_channels == 1 and layers[-2].groups == 1 and layers[-2].dilation[0] == 1 \
            and isinstance(ops[-3], _SnakeOp) and ops[-3].fn.scale is None and ops[-3].fn.bias is None:
        ops[-3:] = [SnakeConvTanhHead(layers[-3], layers[-2], slots)]
    return ops


def _state_tensors(ops) -> list[torch.Tensor]:
    out = []
    for op in ops:
        inner = op.ops if isinstance(op, _ResidualUnit) else [op]  # fused ops hold .state themselves
        out += [o.state for o in inner if getattr(o, "state", None) is not None]
    return out


class SlotStreamingVAEDecoder:
    """Drop-in replacement for nanovllm's BatchedStreamingVAEDecoder.

    ``decode_chunks`` returns a view into a graph-owned buffer; consume it
    before the next call.
    """

    def __init__(
        self,
        decoder: nn.Module,
        max_streams: int,
        latent_dim: int = 64,
        chunk_frames: int = 4,
        graph_batch_sizes: Sequence[int] = (1, 2, 4, 8, *range(16, 257, 16)),
        use_cuda_graph: bool = True,
        fused: bool = False,
        context_cache_size: int = 64,
        extra_chunk_frames: Sequence[int] = (),
    ):
        """``extra_chunk_frames``: chunk lengths besides ``chunk_frames`` that also get CUDA graphs (a patch decoded
        in two pieces, e.g. 1 + 3 frames for an early first audio); other lengths run eagerly."""
        self.max_streams = max_streams
        self.dummy_slot = max_streams
        self.latent_dim = latent_dim
        self.chunk_frames = chunk_frames
        self.ops = _build_ops(decoder, max_streams + 1, fused=fused)
        self.states = _state_tensors(self.ops)
        self.device = self.states[0].device
        self.dtype = self.states[0].dtype
        self.graph_batch_sizes = tuple(sorted(b for b in graph_batch_sizes if b <= max_streams)) or (max_streams,)
        self.max_graph_batch = self.graph_batch_sizes[-1]
        self.use_cuda_graph = use_cuda_graph
        self._slot_of: dict[Hashable, int] = {}
        self._free = list(range(max_streams - 1, -1, -1))
        # graphs and their outputs by (batch bucket, chunk frames); one input buffer per chunk length
        self._graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
        self._graph_out: dict[tuple[int, int], torch.Tensor] = {}
        frames = sorted({chunk_frames, *(t for t in extra_chunk_frames if 0 < t)})
        self._zs = {t: torch.zeros(self.max_graph_batch, latent_dim, t, device=self.device, dtype=self.dtype)
                    for t in frames}
        self._z = self._zs[chunk_frames]
        # primed causal state per context key (e.g. one per cloned voice): rows of a copy of every state tensor
        self.context_cache_size = context_cache_size
        self._ctx_rows: OrderedDict[Hashable, int] = OrderedDict()
        self._ctx_cache: list[torch.Tensor] | None = None
        self._idx = torch.full((self.max_graph_batch,), self.dummy_slot, device=self.device, dtype=torch.long)

    # -- graph-captured core -------------------------------------------------
    def _forward(self, z: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        x = z
        for op in self.ops:
            x = op(x, idx)
        return x

    @torch.inference_mode()
    def warmup(self, latent_channels: int | None = None, chunk_size: int | None = None) -> None:
        """Compile the Triton kernels and capture one CUDA graph per batch-size bucket."""
        self._idx.fill_(self.dummy_slot)
        self._forward(self._z[:1], self._idx[:1])
        if not self.use_cuda_graph:
            return
        pool = None
        for t in sorted(self._zs, reverse=True):  # the longest chunk first: the shared pool's largest graph
            z = self._zs[t]
            for b in reversed(self.graph_batch_sizes):
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    self._forward(z[:b], self._idx[:b])
                torch.cuda.current_stream().wait_stream(side)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool):
                    out = self._forward(z[:b], self._idx[:b])
                pool = graph.pool() if pool is None else pool
                self._graphs[(b, t)] = graph
                self._graph_out[(b, t)] = out
        torch.cuda.synchronize()
        for s in self.states:
            s.zero_()

    # -- slot management ------------------------------------------------------
    def _slots_for(self, stream_ids: Sequence[Hashable]) -> list[int]:
        slots, fresh = [], []
        for sid in stream_ids:
            slot = self._slot_of.get(sid)
            if slot is None:
                if not self._free:
                    raise RuntimeError(f"more than {self.max_streams} concurrent VAE streams")
                slot = self._free.pop()
                self._slot_of[sid] = slot
                fresh.append(slot)
            slots.append(slot)
        if fresh:
            new = self._index(fresh)  # pinned + non_blocking: a pageable copy would wait for the queued GPU work
            for s in self.states:
                s.index_fill_(0, new, 0)
        return slots

    def has_stream(self, stream_id: Hashable) -> bool:
        """True once the stream has a slot (its initial context, if any, is no longer needed)."""
        return stream_id in self._slot_of

    def release(self, stream_id: Hashable) -> None:
        slot = self._slot_of.pop(stream_id, None)
        if slot is not None:
            self._free.append(slot)

    def clear(self) -> None:
        self._slot_of.clear()
        self._free = list(range(self.max_streams - 1, -1, -1))

    def _prime(self, slots, initial_contexts, new_streams, context_keys=None) -> None:
        """Run new streams' context latents [1, D, T] through the decoder to set their causal state.

        Streams with the same context length are primed together in one batched pass (output discarded).
        With ``context_keys`` (equal key = equal context, e.g. the same cloned voice), the primed state is kept
        and later streams with that key copy it instead of running the decoder.
        """
        groups: dict[int, list[tuple[int, torch.Tensor, Hashable]]] = {}
        hits: list[tuple[int, Hashable]] = []  # (slot, key) resolved from the cache after priming
        primed: set[Hashable] = set()
        if context_keys is not None:
            keys = {k for k, c, n in zip(context_keys, initial_contexts, new_streams) if n and c is not None}
            keys.discard(None)
            if len(keys) > self.context_cache_size:  # this call alone would evict its own entries
                context_keys = None
            else:
                for k in keys & self._ctx_rows.keys():  # protect this call's hits from eviction below
                    self._ctx_rows.move_to_end(k)
        for i, (slot, ctx, is_new) in enumerate(zip(slots, initial_contexts, new_streams)):
            if not is_new or ctx is None or ctx.numel() == 0:
                continue
            key = context_keys[i] if context_keys is not None and self.context_cache_size > 0 else None
            if key is not None and (key in self._ctx_rows or key in primed):
                hits.append((slot, key))
                continue
            if key is not None:
                primed.add(key)
            groups.setdefault(ctx.size(-1), []).append((slot, ctx, key))
        for items in groups.values():
            ctx = torch.cat([c for _, c, _ in items], dim=0).to(self.dtype)
            idx = self._index([s for s, _, _ in items])
            step = self.max_graph_batch
            for i in range(0, len(items), step):
                self._forward(ctx[i : i + step], idx[i : i + step])
            keyed = [(s, k) for s, _, k in items if k is not None]
            if keyed:
                self._store_contexts(keyed)
        if hits:
            src = self._index([self._ctx_rows[k] for _, k in hits])
            dst = self._index([s for s, _ in hits])
            for state, cache in zip(self.states, self._ctx_cache):
                state.index_copy_(0, dst, cache.index_select(0, src))
            for _, k in hits:
                self._ctx_rows.move_to_end(k)

    def _store_contexts(self, keyed: list[tuple[int, Hashable]]) -> None:
        if self._ctx_cache is None:
            self._ctx_cache = [torch.empty(self.context_cache_size, *s.shape[1:], device=s.device, dtype=s.dtype)
                               for s in self.states]
        rows = []
        for _, key in keyed:
            if len(self._ctx_rows) < self.context_cache_size:
                row = len(self._ctx_rows)
            else:
                _, row = self._ctx_rows.popitem(last=False)  # least recently used
            self._ctx_rows[key] = row
            rows.append(row)
        src, dst = self._index([s for s, _ in keyed]), self._index(rows)
        for state, cache in zip(self.states, self._ctx_cache):
            cache.index_copy_(0, dst, state.index_select(0, src))

    def _index(self, values: list[int]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.long, device="cpu").pin_memory().to(self.device, non_blocking=True)

    # -- public API ------------------------------------------------------------
    @torch.inference_mode()
    def decode_chunks(
        self,
        z_chunks: torch.Tensor,
        stream_ids: Sequence[Hashable],
        initial_contexts: Sequence[torch.Tensor | None] | None = None,
        context_keys: Sequence[Hashable | None] | None = None,
    ) -> torch.Tensor:
        """Decode one latent chunk [B, D, T] per stream; returns audio [B, 1, T * hop].

        ``initial_contexts[i]`` primes a new stream's state; ``context_keys[i]`` (optional) identifies that
        context so its primed state can be reused (see ``_prime``).
        """
        if len(stream_ids) != z_chunks.size(0):
            raise ValueError("stream_ids length must match z_chunks batch size")
        new_streams = [sid not in self._slot_of for sid in stream_ids]
        slots = self._slots_for(stream_ids)
        if initial_contexts is not None:
            self._prime(slots, initial_contexts, new_streams, context_keys)
        z = z_chunks.to(self.dtype)
        bsz, t = z.size(0), z.size(-1)
        if self._graphs and t in self._zs and bsz <= self.max_graph_batch:
            bucket = next(b for b in self.graph_batch_sizes if b >= bsz)
            buf = self._zs[t]
            buf[:bsz].copy_(z)
            self._idx[:bsz].copy_(torch.tensor(slots, dtype=torch.long, device="cpu").pin_memory(), non_blocking=True)
            if bucket > bsz:
                buf[bsz:bucket].zero_()
                self._idx[bsz:bucket].fill_(self.dummy_slot)
            self._graphs[(bucket, t)].replay()
            return self._graph_out[(bucket, t)][:bsz]
        idx = self._index(slots)
        step = self.max_graph_batch
        return torch.cat([self._forward(z[i : i + step], idx[i : i + step]) for i in range(0, bsz, step)], dim=0)
