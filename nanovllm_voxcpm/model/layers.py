"""VoxCPM2's layers: activation, RMSNorm, the (parallel) linear layers, embedding and head, attention."""

import torch
import torch.distributed as dist
import torch.nn.functional as F
import triton
import triton.language as tl
from flash_attn import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache
from torch import nn

from nanovllm_voxcpm.engine.utils import get_context, get_tp_rank, get_tp_world_size, set_weight_loader, ShardId


# --- activation

class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y


# --- RMSNorm

class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return F.rms_norm(x, (self.hidden_size,), self.weight, self.eps)

    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float() + residual.float()
        residual = x.to(orig_dtype)
        return F.rms_norm(x, (self.hidden_size,), self.weight.float(), self.eps).to(orig_dtype), residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)


# --- linear layers

def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = get_tp_rank()
        self.tp_size = get_tp_world_size()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        set_weight_loader(self.weight, self.weight_loader)
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            set_weight_loader(self.bias, self.weight_loader)
        else:
            self.register_parameter("bias", None)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: ShardId | None = None,
    ) -> None:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ColumnParallelLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = get_tp_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: ShardId | None = None):
        param_data = param.data
        assert self.tp_dim is not None
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: ShardId | None = None):
        param_data = param.data
        assert isinstance(loaded_shard_id, int)
        assert self.tp_dim is not None
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = get_tp_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: ShardId | None = None):
        param_data = param.data
        assert self.tp_dim is not None
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = get_tp_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: ShardId | None = None):
        param_data = param.data
        assert self.tp_dim is not None
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y


# --- embedding and LM head

class VocabParallelEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        extra_rows: int = 0,
    ):
        """extra_rows: rows after the vocabulary (ids num_embeddings..), zero until ``set_rows`` writes them (e.g. a
        tagged adapter's dialect tags); the checkpoint fills the vocabulary rows only."""
        super().__init__()
        self.tp_rank = get_tp_rank()
        self.tp_size = get_tp_world_size()
        assert num_embeddings % self.tp_size == 0
        assert extra_rows == 0 or self.tp_size == 1, "extra embedding rows need tensor parallel size 1"
        self.num_embeddings = num_embeddings
        self.extra_rows = extra_rows
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition + extra_rows, embedding_dim))
        set_weight_loader(self.weight, self.weight_loader)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = self.num_embeddings_per_partition
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        if self.extra_rows:
            param_data[shard_size:].zero_()
            param_data = param_data[:shard_size]
        param_data.copy_(loaded_weight)

    @torch.no_grad()
    def set_rows(self, start: int, rows: torch.Tensor) -> None:
        """Write ``rows`` [n, dim] at ids start..start+n-1, which must be extra rows (not the vocabulary)."""
        if start < self.num_embeddings or start + rows.shape[0] > self.num_embeddings + self.extra_rows:
            raise ValueError(f"rows {start}..{start + rows.shape[0] - 1} are outside the extra embedding rows "
                             f"{self.num_embeddings}..{self.num_embeddings + self.extra_rows - 1}")
        self.weight.data[start:start + rows.shape[0]].copy_(rows.to(self.weight.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = torch.where(mask, x - self.vocab_start_idx, 0)
            y = F.embedding(x, self.weight)
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
            return y

        return F.embedding(x, self.weight)


# --- attention

@triton.jit
def store_kvcache_kernel(  # pragma: no cover – GPU Triton kernel, not executable on CPU
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](
        key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D
    )  # pragma: no cover – dispatches GPU Triton kernel


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        is_causal: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.is_causal = is_causal

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ):  # pragma: no cover – all paths call GPU flash-attn kernels
        if self.is_causal:
            context = get_context()
            k_cache, v_cache = self.k_cache, self.v_cache
            if k_cache.numel() and v_cache.numel():
                store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
            if context.is_prefill:
                if context.block_tables is not None:  # prefix cache
                    k, v = k_cache, v_cache
                o = flash_attn_varlen_func(
                    q,
                    k,
                    v,
                    max_seqlen_q=context.max_seqlen_q,
                    cu_seqlens_q=context.cu_seqlens_q,
                    max_seqlen_k=context.max_seqlen_k,
                    cu_seqlens_k=context.cu_seqlens_k,
                    softmax_scale=self.scale,
                    causal=True,
                    block_table=context.block_tables,
                )
            else:  # decode
                o = flash_attn_with_kvcache(
                    q.unsqueeze(1),
                    k_cache,
                    v_cache,
                    cache_seqlens=context.context_lens,
                    block_table=context.block_tables,
                    softmax_scale=self.scale,
                    causal=True,
                )
        else:
            # non causal attention, no kvcache required
            o = flash_attn_func(q, k, v, softmax_scale=self.scale, causal=False)
        return o
