"""The LoRA Triton kernels (shrink, expand) and their metadata, from vLLM (Apache-2.0)."""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TypeAlias

import torch
import triton
import triton.language as tl


# --- LoRA weight pointer tables

_LORA_A_PTR_DICT: dict[tuple[int, ...], tuple[torch.Tensor, int, int, int]] = {}
TensorOrInt: TypeAlias = torch.Tensor | int
_LORA_B_PTR_DICT: dict[
    tuple[int, ...], tuple[TensorOrInt, TensorOrInt, TensorOrInt, TensorOrInt, TensorOrInt, TensorOrInt, bool, int]
] = {}


def _get_lora_a_ptr(lora_a_weights: list[torch.Tensor], device: torch.device):
    key = tuple(weight.data_ptr() for weight in lora_a_weights)
    if values := _LORA_A_PTR_DICT.get(key):
        return values

    lora_strides_d0 = []
    lora_strides_d1 = []
    lora_strides_d2 = []
    tensor_ptrs = []
    for lora_a_weight in lora_a_weights:
        if lora_a_weight.ndim == 4:
            assert lora_a_weight.size(1) == 1
            lora_a_weight = lora_a_weight.squeeze(dim=1)
        else:
            assert lora_a_weight.ndim == 3
        assert lora_a_weight.is_contiguous()
        tensor_ptrs.append(lora_a_weight.data_ptr())
        lora_strides_d0.append(lora_a_weight.stride(0))
        lora_strides_d1.append(lora_a_weight.stride(1))
        lora_strides_d2.append(lora_a_weight.stride(2))

    lora_ptr_tensor = (
        torch.tensor(tensor_ptrs, device=device, dtype=torch.uint64) if len(lora_a_weights) > 1 else lora_a_weights[0]
    )
    if len(set(lora_strides_d0)) > 1 or len(set(lora_strides_d1)) > 1 or len(set(lora_strides_d2)) > 1:
        raise ValueError("All LoRA A weights must have the same stride")

    _LORA_A_PTR_DICT[key] = (lora_ptr_tensor, lora_strides_d0[0], lora_strides_d1[0], lora_strides_d2[0])
    return _LORA_A_PTR_DICT[key]


def _get_lora_b_ptr(lora_weights: list[torch.Tensor], offset_start: int, device: torch.device):
    # NOTE: slice_start_tensor below is derived from offset_start (it is the
    # running cumulative column offset for each slice in the packed output).
    # The same lora_weights list can legitimately be called with different
    # offset_start values (e.g. when the caller groups slices by (rank,
    # hidden_out) and each group starts at its own column offset inside the
    # packed output), so offset_start MUST be part of the cache key.
    key = (offset_start, *(weight.data_ptr() for weight in lora_weights))
    if values := _LORA_B_PTR_DICT.get(key):
        return values

    slice_offset_lst = []
    tensor_ptrs = []
    lora_strides_d0 = []
    lora_strides_d1 = []
    lora_strides_d2 = []
    hidden_sizes = []
    slice_offset = offset_start
    for lora_b_weight in lora_weights:
        if lora_b_weight.ndim == 4:
            assert lora_b_weight.size(1) == 1
            lora_b_weight = lora_b_weight.squeeze(dim=1)
        else:
            assert lora_b_weight.ndim == 3
        assert lora_b_weight.is_contiguous()
        tensor_ptrs.append(lora_b_weight.data_ptr())
        lora_strides_d0.append(lora_b_weight.stride(0))
        lora_strides_d1.append(lora_b_weight.stride(1))
        lora_strides_d2.append(lora_b_weight.stride(2))
        slice_offset_lst.append(slice_offset)
        slice_offset += lora_b_weight.size(1)
        hidden_sizes.append(lora_b_weight.size(1))

    slice_start_tensor: TensorOrInt
    if len(lora_weights) > 1:
        lora_ptr_tensor = torch.tensor(tensor_ptrs, device=device, dtype=torch.uint64)
        slice_start_tensor = torch.tensor(slice_offset_lst, device=device, dtype=torch.uint64)
    else:
        slice_start_tensor = slice_offset_lst[0]
        lora_ptr_tensor = lora_weights[0]

    if (
        len(set(lora_strides_d0)) == 1
        and len(set(lora_strides_d1)) == 1
        and len(set(lora_strides_d2)) == 1
        and len(set(hidden_sizes)) == 1
    ):
        lora_strides_d0_tensor: TensorOrInt = lora_strides_d0[0]
        lora_strides_d1_tensor: TensorOrInt = lora_strides_d1[0]
        lora_strides_d2_tensor: TensorOrInt = lora_strides_d2[0]
        hidden_sizes_tensor: TensorOrInt = hidden_sizes[0]
        same_stride = True
    else:
        lora_strides_d0_tensor = torch.tensor(lora_strides_d0, device=device)
        lora_strides_d1_tensor = torch.tensor(lora_strides_d1, device=device)
        lora_strides_d2_tensor = torch.tensor(lora_strides_d2, device=device)
        hidden_sizes_tensor = torch.tensor(hidden_sizes, device=device)
        same_stride = False

    max_n = max(hidden_sizes)
    _LORA_B_PTR_DICT[key] = (
        slice_start_tensor,
        lora_ptr_tensor,
        lora_strides_d0_tensor,
        lora_strides_d1_tensor,
        lora_strides_d2_tensor,
        hidden_sizes_tensor,
        same_stride,
        max_n,
    )
    return _LORA_B_PTR_DICT[key]


@lru_cache
def get_lora_op_configs(
    op_type: str,
    max_loras: int,
    batch: int,
    hidden_size: int,
    rank: int,
    num_slices: int,
    add_inputs: bool | None = None,
) -> dict[str, int | None]:
    if op_type == "shrink":
        split_k = 64 if batch < 128 else 8
        return {
            "block_m": 32,
            "block_n": 16,
            "block_k": 256 if batch < 128 else 32,
            "split_k": split_k,
            "num_warps": 4,
            "num_ctas": 1,
            "group_size_m": 8,
            "num_stages": 2,
            "max_nreg": None,
        }
    return {
        "block_m": 64,
        "block_n": 64 if num_slices > 1 else 128,
        "block_k": 32,
        "num_warps": 4,
        "num_ctas": 1,
        "num_stages": 2,
        "max_nreg": None,
    }


@lru_cache
def supports_pdl(device: torch.device | None = None) -> bool:
    return False


# --- the kernels' shared bodies

@triton.jit
def mm_k(
    a_ptr,
    b_ptr,
    ak_stride,
    bk_stride,
    offset_k,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    CAST_TYPE: tl.constexpr,
    b_dtype: tl.constexpr,
    USE_GDC: tl.constexpr,
    base_k,
):
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    step_k = BLOCK_K * SPLIT_K
    num_iters = tl.cdiv(K, step_k)

    for k in range(num_iters):
        iter_k = k * step_k + base_k
        block_end = iter_k + BLOCK_K

        if EVEN_K:
            tiled_b = tl.load(b_ptr)
            if USE_GDC:
                tl.extra.cuda.gdc_wait()
            tiled_a = tl.load(a_ptr)
            if CAST_TYPE:
                tiled_a = tiled_a.to(b_dtype)
            accumulator += tl.dot(tiled_a, tiled_b)
        else:
            if iter_k >= K:
                pass
            elif block_end <= K:
                tiled_b = tl.load(b_ptr)
                if USE_GDC:
                    tl.extra.cuda.gdc_wait()
                tiled_a = tl.load(a_ptr)
                if CAST_TYPE:
                    tiled_a = tiled_a.to(b_dtype)
                accumulator += tl.dot(tiled_a, tiled_b)
            else:
                k_offsets = tl.arange(0, BLOCK_K)
                mask = iter_k + k_offsets < K
                tiled_b = tl.load(b_ptr, mask=mask[:, None], other=0.0)
                if USE_GDC:
                    tl.extra.cuda.gdc_wait()
                tiled_a = tl.load(a_ptr, mask=mask[None, :], other=0.0)
                if CAST_TYPE:
                    tiled_a = tiled_a.to(b_dtype)
                accumulator += tl.dot(tiled_a, tiled_b)

        a_ptr += step_k * ak_stride
        b_ptr += step_k * bk_stride

    return accumulator


@triton.jit
def do_shrink_kernel(
    pid_n,
    pid_sk,
    slice_id,
    lora_index,
    input_ptr,
    lora_ptr,
    out_ptr,
    N,
    K,
    M_LEN,
    ram,
    input_d0_stride,
    input_d1_stride,
    lora_d0_stride,
    lora_d1_stride,
    lora_d2_stride,
    output_d0_stride,
    output_d1_stride,
    output_d2_stride,
    scaling,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    SLICE_NUM: tl.constexpr,
    USE_GDC: tl.constexpr,
):
    if SLICE_NUM == 1:
        cur_lora_ptr = lora_ptr
    else:
        cur_lora_ptr = tl.load(lora_ptr + slice_id).to(tl.pointer_type(input_ptr.dtype.element_ty))

    offset_n = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    rbn = tl.max_contiguous(tl.multiple_of(offset_n % N, BLOCK_N), BLOCK_N)
    offset_k = pid_sk * BLOCK_K + tl.arange(0, BLOCK_K)
    a_ptr = input_ptr + ram[:, None] * input_d0_stride + offset_k[None, :] * input_d1_stride
    b_ptr = (
        cur_lora_ptr + lora_d0_stride * lora_index + rbn[None, :] * lora_d1_stride + offset_k[:, None] * lora_d2_stride
    )

    accumulator = mm_k(
        a_ptr,
        b_ptr,
        input_d1_stride,
        lora_d2_stride,
        offset_k,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        EVEN_K,
        SPLIT_K,
        False,
        cur_lora_ptr.dtype.element_ty,
        False,
        base_k=pid_sk * BLOCK_K,
    )
    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()
    offset_cn = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    offset_cm = tl.arange(0, BLOCK_M)
    cur_out_ptr = out_ptr if SLICE_NUM == 1 else out_ptr + slice_id * output_d0_stride
    c_ptr = cur_out_ptr + ram[:, None] * output_d1_stride + offset_cn[None, :] * output_d2_stride
    c_mask = (offset_cm[:, None] < M_LEN) & (offset_cn[None, :] < N)
    accumulator *= scaling
    if SPLIT_K == 1:
        tl.store(c_ptr, accumulator, mask=c_mask)
    else:
        tl.atomic_add(c_ptr, accumulator, mask=c_mask, sem="relaxed")


@triton.jit
def do_expand_kernel(
    pid_n,
    lora_index,
    slice_id,
    input_ptr,
    lora_ptr,
    out_ptr,
    N,
    K,
    M_LEN,
    ram,
    slice_start_loc,
    input_d0_stride,
    input_d1_stride,
    input_d2_stride,
    ls_d0_ptr,
    ls_d1_ptr,
    ls_d2_ptr,
    output_d0_stride,
    output_d1_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SAME_STRIDE: tl.constexpr,
    SLICE_NUM: tl.constexpr,
    EVEN_K: tl.constexpr,
    CAST_TYPE: tl.constexpr,
    ADD_INPUTS: tl.constexpr,
    USE_GDC: tl.constexpr,
):
    if SAME_STRIDE:
        cur_lora_d0_stride = ls_d0_ptr
        cur_lora_d1_stride = ls_d1_ptr
        cur_lora_d2_stride = ls_d2_ptr
    else:
        cur_lora_d0_stride = tl.load(ls_d0_ptr + slice_id)
        cur_lora_d1_stride = tl.load(ls_d1_ptr + slice_id)
        cur_lora_d2_stride = tl.load(ls_d2_ptr + slice_id)

    if SLICE_NUM == 1:
        cur_input_ptr = input_ptr
        cur_lora_ptr = lora_ptr
    else:
        cur_input_ptr = input_ptr + slice_id * input_d0_stride
        cur_lora_ptr = tl.load(lora_ptr + slice_id).to(tl.pointer_type(out_ptr.dtype.element_ty))

    offset_n = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    rbn = tl.max_contiguous(tl.multiple_of(offset_n % N, BLOCK_N), BLOCK_N)
    offset_k = tl.arange(0, BLOCK_K)
    a_ptr = cur_input_ptr + ram[:, None] * input_d1_stride + offset_k[None, :] * input_d2_stride
    b_ptr = (
        cur_lora_ptr
        + cur_lora_d0_stride * lora_index
        + offset_k[:, None] * cur_lora_d2_stride
        + rbn[None, :] * cur_lora_d1_stride
    )

    accumulator = mm_k(
        a_ptr,
        b_ptr,
        input_d2_stride,
        cur_lora_d2_stride,
        offset_k,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        EVEN_K,
        1,
        CAST_TYPE,
        cur_lora_ptr.dtype.element_ty,
        USE_GDC,
        base_k=0,
    )

    tiled_c = accumulator.to(cur_lora_ptr.dtype.element_ty)
    cur_slice_start = slice_start_loc if SLICE_NUM == 1 else tl.load(slice_start_loc + slice_id)
    offset_cn = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N + cur_slice_start
    offset_cm = tl.arange(0, BLOCK_M)
    c_ptr = out_ptr + ram[:, None] * output_d0_stride + offset_cn[None, :] * output_d1_stride
    c_mask = (offset_cm[:, None] < M_LEN) & (offset_cn[None, :] < (cur_slice_start + N))
    if ADD_INPUTS:
        tiled_out = tl.load(c_ptr, mask=c_mask)
        tiled_c += tiled_out
    tl.store(c_ptr, tiled_c, mask=c_mask)


# --- kernel metadata

@dataclass
class LoRAKernelMeta:
    token_lora_mapping: torch.Tensor
    token_indices_sorted_by_lora_ids: torch.Tensor
    active_lora_ids: torch.Tensor
    num_tokens_per_lora: torch.Tensor
    lora_token_start_loc: torch.Tensor
    no_lora_flag: bool = False
    num_active_loras: int = 0
    default_num_active_loras: int = 0
    captured_lora_counts: list[int] = field(default_factory=list)

    @staticmethod
    def make(
        max_loras: int,
        max_num_tokens: int,
        device: torch.device | str,
        captured_lora_counts: list[int] | None = None,
    ) -> "LoRAKernelMeta":
        return LoRAKernelMeta(
            token_lora_mapping=torch.empty(max_num_tokens, dtype=torch.int32, device=device),
            token_indices_sorted_by_lora_ids=torch.empty(max_num_tokens, dtype=torch.int32, device=device),
            active_lora_ids=torch.empty(max_loras + 1, dtype=torch.int32, device=device),
            num_tokens_per_lora=torch.zeros(max_loras + 1, dtype=torch.int32, device=device),
            lora_token_start_loc=torch.zeros(max_loras + 2, dtype=torch.int32, device=device),
            no_lora_flag=False,
            num_active_loras=0,
            default_num_active_loras=max_loras + 1,
            captured_lora_counts=sorted(captured_lora_counts) if captured_lora_counts else [],
        )

    def _reset(self):
        self.active_lora_ids.fill_(-1)
        self.num_tokens_per_lora.fill_(0)
        self.lora_token_start_loc.fill_(0)
        self.no_lora_flag = False
        self.num_active_loras = 0

    def prepare_tensors(self, token_lora_mapping: torch.Tensor) -> None:
        self._reset()
        no_lora = torch.all(token_lora_mapping == -1)
        self.no_lora_flag = bool(no_lora.item())
        if no_lora:
            return
        num_tokens = token_lora_mapping.size(0)
        self.token_lora_mapping[:num_tokens].copy_(token_lora_mapping, non_blocking=True)
        _, token_indices_sorted = torch.sort(token_lora_mapping, stable=True)
        self.token_indices_sorted_by_lora_ids[:num_tokens].copy_(token_indices_sorted, non_blocking=True)
        lora_ids, num_tokens_per_lora = torch.unique(token_lora_mapping, sorted=True, return_counts=True)
        self.active_lora_ids[: lora_ids.size(0)].copy_(lora_ids, non_blocking=True)
        self.num_tokens_per_lora[: num_tokens_per_lora.size(0)].copy_(num_tokens_per_lora, non_blocking=True)
        num_active_loras = lora_ids.size(0)
        if self.captured_lora_counts and num_active_loras > 0:
            idx = bisect.bisect_left(self.captured_lora_counts, num_active_loras)
            if idx < len(self.captured_lora_counts):
                num_active_loras = self.captured_lora_counts[idx]
        self.num_active_loras = num_active_loras
        lora_token_start_loc = torch.cumsum(num_tokens_per_lora, dim=0)
        self.lora_token_start_loc[1 : 1 + lora_token_start_loc.size(0)].copy_(lora_token_start_loc, non_blocking=True)

    def meta_args(self, token_nums: int, specialize_active_lora: bool):
        num_active_loras = self.num_active_loras if specialize_active_lora else self.default_num_active_loras
        return (
            self.token_lora_mapping[:token_nums],
            self.token_indices_sorted_by_lora_ids[:token_nums],
            self.num_tokens_per_lora,
            self.lora_token_start_loc,
            self.active_lora_ids,
            self.no_lora_flag,
            num_active_loras,
        )


# --- shrink

_SMALL_M_THRESHOLD = 32
_SMALL_M_MAX_ACTIVE_LORAS = 32
_SMALL_M_MAX_BLOCK_RANK = 8
_SMALL_M_MAX_BLOCK_K = 1024


@triton.jit
def _lora_shrink_kernel(
    input_ptr,
    lora_ptr,
    out_ptr,
    M,
    N,
    K,
    token_indices_sorted_by_lora_ids,
    num_tokens_per_lora,
    lora_token_start_loc,
    lora_ids,
    scaling,
    input_d0_stride,
    input_d1_stride,
    lora_d0_stride,
    lora_d1_stride,
    lora_d2_stride,
    output_d0_stride,
    output_d1_stride,
    output_d2_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SLICE_NUM: tl.constexpr,
    USE_GDC: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    cta_n_num = tl.cdiv(N, BLOCK_N)
    cta_m_num = tl.cdiv(M, BLOCK_M)
    pid_sk_m_n = tl.program_id(axis=0)
    pid_sk = pid_sk_m_n % SPLIT_K
    pid_m_n = pid_sk_m_n // SPLIT_K
    num_pid_in_group = GROUP_SIZE_M * cta_n_num
    group_id = pid_m_n // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(cta_m_num - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid_m_n % num_pid_in_group) % group_size_m)
    pid_n = (pid_m_n % num_pid_in_group) // group_size_m
    slice_id = tl.program_id(axis=1)
    lora_idx = tl.program_id(axis=2)

    lora_id = tl.load(lora_ids + lora_idx)
    if lora_id == -1:
        return
    lora_m_size = tl.load(num_tokens_per_lora + lora_idx)
    cta_m_offset = pid_m * BLOCK_M
    if cta_m_offset >= lora_m_size:
        return
    cta_m_len = min(BLOCK_M, lora_m_size - cta_m_offset)
    lora_m_indices_start = tl.load(lora_token_start_loc + lora_idx)
    cta_lora_seq_indices = token_indices_sorted_by_lora_ids + lora_m_indices_start + cta_m_offset
    offset_m = tl.arange(0, BLOCK_M) % cta_m_len
    ram = tl.load(cta_lora_seq_indices + offset_m)
    do_shrink_kernel(
        pid_n,
        pid_sk,
        slice_id,
        lora_id,
        input_ptr,
        lora_ptr,
        out_ptr,
        N,
        K,
        cta_m_len,
        ram,
        input_d0_stride,
        input_d1_stride,
        lora_d0_stride,
        lora_d1_stride,
        lora_d2_stride,
        output_d0_stride,
        output_d1_stride,
        output_d2_stride,
        scaling,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        EVEN_K,
        SPLIT_K,
        SLICE_NUM,
        USE_GDC,
    )


@triton.jit
def _lora_shrink_small_m_kernel(
    input_ptr,
    lora_ptr,
    out_ptr,
    M,
    N,
    K,
    token_lora_mapping,
    scaling,
    input_d0_stride,
    input_d1_stride,
    lora_d0_stride,
    lora_d1_stride,
    lora_d2_stride,
    output_d0_stride,
    output_d1_stride,
    output_d2_stride,
    BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    SLICE_NUM: tl.constexpr,
):
    pid_sk_mr = tl.program_id(axis=0)
    slice_id = tl.program_id(axis=1)

    pid_sk = pid_sk_mr % SPLIT_K
    pid_mr = pid_sk_mr // SPLIT_K
    pid_m = pid_mr % M
    pid_r = pid_mr // M

    lora_id = tl.load(token_lora_mapping + pid_m)
    if lora_id == -1:
        return

    if SLICE_NUM == 1:
        cur_lora_ptr = lora_ptr
    else:
        cur_lora_ptr = tl.load(lora_ptr + slice_id).to(tl.pointer_type(input_ptr.dtype.element_ty))

    offset_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    mask_r = offset_r < N
    load_r = tl.max_contiguous(tl.multiple_of(offset_r % N, BLOCK_R), BLOCK_R)

    accumulator = tl.zeros((BLOCK_R,), dtype=tl.float32)
    split_k_start = pid_sk * BLOCK_K
    step_k = BLOCK_K * SPLIT_K
    a_ptr = input_ptr + pid_m * input_d0_stride + (split_k_start + tl.arange(0, BLOCK_K)) * input_d1_stride
    b_ptr = (
        cur_lora_ptr
        + lora_id * lora_d0_stride
        + load_r[None, :] * lora_d1_stride
        + (split_k_start + tl.arange(0, BLOCK_K))[:, None] * lora_d2_stride
    )
    for k_start in range(0, K, step_k):
        offset_k = k_start + split_k_start + tl.arange(0, BLOCK_K)
        mask_k = offset_k < K
        tiled_a = tl.load(a_ptr, mask=mask_k, other=0.0)
        tiled_b = tl.load(b_ptr, mask=mask_k[:, None] & mask_r[None, :], other=0.0)
        # one row times BLOCK_R (<= 8) columns: a multiply and a sum, since tl.dot needs M, N, K >= 16 (triton >= 3.4)
        accumulator += tl.sum(tiled_a[:, None].to(tl.float32) * tiled_b.to(tl.float32), axis=0)
        a_ptr += step_k * input_d1_stride
        b_ptr += step_k * lora_d2_stride

    accumulator *= scaling
    out_ptr = out_ptr + slice_id * output_d0_stride + pid_m * output_d1_stride + offset_r * output_d2_stride
    if SPLIT_K == 1:
        tl.store(out_ptr, accumulator, mask=mask_r)
    else:
        tl.atomic_add(out_ptr, accumulator, mask=mask_r, sem="relaxed")


def _use_small_m_shrink_path(M: int, N: int, num_active_loras: int, num_slices: int) -> bool:
    return M < _SMALL_M_THRESHOLD and N > 0 and 0 < num_active_loras <= _SMALL_M_MAX_ACTIVE_LORAS and num_slices > 0


def _select_small_m_kernel_config(K: int, N: int) -> tuple[int, int, int]:
    capped_k = min(K, _SMALL_M_MAX_BLOCK_K)
    if K >= 4096 and N <= 2:
        block_k = 512
        block_r = 1
        split_k = min(8, triton.cdiv(K, block_k))
    elif K >= 4096:
        block_k = 512
        block_r = min(4, N)
        split_k = min(8, triton.cdiv(K, block_k))
    elif capped_k >= 1024:
        block_k = 1024
        block_r = 1
        split_k = 1
    elif capped_k >= 512:
        block_k = 512
        block_r = 2
        split_k = 1
    elif capped_k >= 256:
        block_k = 256
        block_r = 4
        split_k = 1
    elif capped_k >= 128:
        block_k = 128
        block_r = 8
        split_k = 1
    else:
        block_k = 64
        block_r = 8
        split_k = 1
    return block_k, min(block_r, N, _SMALL_M_MAX_BLOCK_RANK), split_k


def _run_small_m_shrink(
    inputs: torch.Tensor,
    lora_ptr_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    M: int,
    N: int,
    K: int,
    token_lora_mapping: torch.Tensor,
    scaling: float,
    lora_strides_d0: int,
    lora_strides_d1: int,
    lora_strides_d2: int,
    num_slices: int,
) -> None:
    block_k, block_r, split_k = _select_small_m_kernel_config(K=K, N=N)
    num_warps = 8 if block_k >= 1024 else 4 if block_k >= 256 else 2
    output_tensor.zero_()
    grid = (split_k * M * triton.cdiv(N, block_r), num_slices)
    _lora_shrink_small_m_kernel[grid](
        inputs,
        lora_ptr_tensor,
        output_tensor,
        M,
        N,
        K,
        token_lora_mapping,
        scaling,
        inputs.stride(0),
        inputs.stride(1),
        lora_strides_d0,
        lora_strides_d1,
        lora_strides_d2,
        output_tensor.stride(0),
        output_tensor.stride(1),
        output_tensor.stride(2),
        BLOCK_R=block_r,
        BLOCK_K=block_k,
        SPLIT_K=split_k,
        SLICE_NUM=num_slices,
        num_warps=num_warps,
        num_stages=2,
    )


@torch.inference_mode()
def lora_shrink(
    inputs: torch.Tensor,
    lora_a_weights: list[torch.Tensor],
    output_tensor: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    token_indices_sorted_by_lora_ids: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    no_lora_flag: bool,
    num_active_loras: int,
    scaling: float,
) -> None:
    if no_lora_flag:
        return
    lora_ptr_tensor, lora_strides_d0, lora_strides_d1, lora_strides_d2 = _get_lora_a_ptr(lora_a_weights, inputs.device)
    M = inputs.size(0)
    N, K = lora_a_weights[0].shape[-2:]
    num_slices = len(lora_a_weights)
    max_loras = lora_ids.size(0)
    if _use_small_m_shrink_path(M=M, N=N, num_active_loras=num_active_loras, num_slices=num_slices):
        _run_small_m_shrink(
            inputs=inputs,
            lora_ptr_tensor=lora_ptr_tensor,
            output_tensor=output_tensor,
            M=M,
            N=N,
            K=K,
            token_lora_mapping=token_lora_mapping,
            scaling=scaling,
            lora_strides_d0=lora_strides_d0,
            lora_strides_d1=lora_strides_d1,
            lora_strides_d2=lora_strides_d2,
            num_slices=num_slices,
        )
        return

    output_tensor.zero_()
    kernel_config = get_lora_op_configs(
        "shrink", max_loras=max_loras, batch=M, hidden_size=K, rank=N, num_slices=num_slices
    )
    block_m = kernel_config["block_m"]
    block_n = kernel_config["block_n"]
    block_k = kernel_config["block_k"]
    split_k = kernel_config["split_k"]
    group_size_m = kernel_config["group_size_m"]
    if block_k is None or split_k is None or group_size_m is None:
        raise RuntimeError("Invalid shrink kernel config")
    even_k = K % (block_k * split_k) == 0
    grid = (split_k * triton.cdiv(M, block_m) * triton.cdiv(N, block_n), num_slices, num_active_loras)
    use_gdc = supports_pdl(inputs.device)
    _lora_shrink_kernel[grid](
        inputs,
        lora_ptr_tensor,
        output_tensor,
        M,
        N,
        K,
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
        scaling,
        inputs.stride(0),
        inputs.stride(1),
        lora_strides_d0,
        lora_strides_d1,
        lora_strides_d2,
        output_tensor.stride(0),
        output_tensor.stride(1),
        output_tensor.stride(2),
        block_m,
        block_n,
        block_k,
        even_k,
        split_k,
        group_size_m,
        num_slices,
        use_gdc,
        num_warps=kernel_config["num_warps"],
        num_ctas=kernel_config["num_ctas"],
        num_stages=kernel_config["num_stages"],
        launch_pdl=use_gdc,
    )


# --- expand

@triton.jit
def _lora_expand_kernel(
    input_ptr,
    lora_ptr,
    out_ptr,
    M,
    N,
    K,
    token_indices_sorted_by_lora_ids,
    num_tokens_per_lora,
    lora_token_start_loc,
    lora_ids,
    slice_start_loc,
    input_d0_stride,
    input_d1_stride,
    input_d2_stride,
    ls_d0_ptr,
    ls_d1_ptr,
    ls_d2_ptr,
    output_d0_stride,
    output_d1_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    ADD_INPUTS: tl.constexpr,
    CAST_TYPE: tl.constexpr,
    SLICE_NUM: tl.constexpr,
    SAME_STRIDE: tl.constexpr,
    USE_GDC: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    cta_n_num = tl.cdiv(N, BLOCK_N)
    cta_m_num = tl.cdiv(M, BLOCK_M)
    pid_mn = tl.program_id(axis=0)
    pid_m = pid_mn % cta_m_num
    pid_n = (pid_mn // cta_m_num) % cta_n_num
    slice_id = tl.program_id(axis=1)
    lora_idx = tl.program_id(axis=2)
    lora_id = tl.load(lora_ids + lora_idx)
    if lora_id == -1:
        return
    lora_m_size = tl.load(num_tokens_per_lora + lora_idx)
    cta_m_offset = pid_m * BLOCK_M
    if cta_m_offset >= lora_m_size:
        return
    cta_m_len = min(BLOCK_M, lora_m_size - cta_m_offset)
    lora_m_indices_start = tl.load(lora_token_start_loc + lora_idx)
    cta_lora_seq_indices = token_indices_sorted_by_lora_ids + lora_m_indices_start + cta_m_offset
    offset_m = tl.arange(0, BLOCK_M) % cta_m_len
    ram = tl.load(cta_lora_seq_indices + offset_m)
    do_expand_kernel(
        pid_n,
        lora_id,
        slice_id,
        input_ptr,
        lora_ptr,
        out_ptr,
        N,
        K,
        cta_m_len,
        ram,
        slice_start_loc,
        input_d0_stride,
        input_d1_stride,
        input_d2_stride,
        ls_d0_ptr,
        ls_d1_ptr,
        ls_d2_ptr,
        output_d0_stride,
        output_d1_stride,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        SAME_STRIDE,
        SLICE_NUM,
        EVEN_K,
        CAST_TYPE,
        ADD_INPUTS,
        USE_GDC,
    )


@torch.inference_mode()
def lora_expand(
    inputs: torch.Tensor,
    lora_b_weights: list[torch.Tensor],
    output_tensor: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    token_indices_sorted_by_lora_ids: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    no_lora_flag: bool,
    num_active_loras: int,
    offset_start: int = 0,
    add_inputs: bool = False,
) -> None:
    if no_lora_flag:
        return
    (
        slice_start_tensor,
        lora_ptr_tensor,
        lora_strides_d0_tensor,
        lora_strides_d1_tensor,
        lora_strides_d2_tensor,
        hidden_sizes_tensor,
        same_stride,
        max_n,
    ) = _get_lora_b_ptr(lora_b_weights, offset_start, inputs.device)
    M = inputs.size(1)
    K = lora_b_weights[0].shape[-1]
    max_loras = lora_ids.size(0)
    num_slices = len(lora_b_weights)
    cast_type = inputs.dtype == torch.float32 and lora_b_weights[0].dtype in [torch.float16, torch.bfloat16]
    kernel_config = get_lora_op_configs(
        "expand", max_loras=max_loras, batch=M, hidden_size=max_n, rank=K, num_slices=num_slices, add_inputs=add_inputs
    )
    block_m = kernel_config["block_m"]
    block_n = kernel_config["block_n"]
    block_k = kernel_config["block_k"]
    if block_k is None:
        raise RuntimeError("Invalid expand kernel config")
    even_k = K % block_k == 0
    grid = (triton.cdiv(M, block_m) * triton.cdiv(max_n, block_n), num_slices, num_active_loras)
    use_gdc = supports_pdl(inputs.device)
    _lora_expand_kernel[grid](
        inputs,
        lora_ptr_tensor,
        output_tensor,
        M,
        max_n,
        K,
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
        slice_start_tensor,
        inputs.stride(0),
        inputs.stride(1),
        inputs.stride(2),
        lora_strides_d0_tensor,
        lora_strides_d1_tensor,
        lora_strides_d2_tensor,
        output_tensor.stride(0),
        output_tensor.stride(1),
        block_m,
        block_n,
        block_k,
        even_k,
        add_inputs,
        cast_type,
        num_slices,
        same_stride,
        use_gdc,
        num_warps=kernel_config["num_warps"],
        num_ctas=kernel_config["num_ctas"],
        num_stages=kernel_config["num_stages"],
        launch_pdl=use_gdc,
    )
