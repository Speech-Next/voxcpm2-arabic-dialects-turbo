"""CPU tests of the burst host path (issue #9): every new path against the code it replaces (copied here from main as
reference implementations), on the same inputs. Nothing runs on a GPU: the one-copy staging (HostPack) uploads to the
CPU, CUDA calls are faked. The end-to-end checks on the real model (bit-identical audio, timings) are
examples/profile_burst.py --hashes / --compare."""

import os
import queue
import random
import sys
import threading
import time
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import nanovllm_voxcpm.engine.lora_manager as lm
from nanovllm_voxcpm.engine.block_manager import BlockManager
from nanovllm_voxcpm.engine.llm_engine import LLMEngineBase
from nanovllm_voxcpm.engine.lora_manager import (
    LoRAManager,
    LoRAModelPayload,
    LoRARuntime,
    LoRAResidentState,
    build_lora_context_from_slot_list,
    materialize_lora_context,
    stage_lora_context,
)
from nanovllm_voxcpm.engine.model_runner import HostPack, RunnerTask, expand_dit_lora_slots
from nanovllm_voxcpm.engine.scheduler import Scheduler
from nanovllm_voxcpm.engine.sequence import Sequence
from nanovllm_voxcpm.utils.context import (
    DIT_LORA_DOMAIN,
    LM_LORA_DOMAIN,
    PROJ_LORA_DOMAIN,
    get_context,
    get_lora_context,
    reset_all_contexts,
)
from nanovllm_voxcpm.utils.seed import derive_step_seed, derive_step_seeds

DOMAINS = (LM_LORA_DOMAIN, PROJ_LORA_DOMAIN, DIT_LORA_DOMAIN)


@pytest.fixture(autouse=True)
def _cpu_lora(monkeypatch):
    """LoRA registration without CUDA; device metadata of the reference path built on the CPU."""
    monkeypatch.setattr(lm, "assert_lora_available", lambda: None)
    monkeypatch.setattr(lm, "_cuda_int_tensor", lambda values: torch.tensor(values, dtype=torch.int32))

    def pack(*lists):
        sizes = [len(x) for x in lists]
        flat = torch.tensor([v for x in lists for v in x], dtype=torch.int32)
        return list(torch.split(flat, sizes)) if flat.numel() else [torch.empty(0, dtype=torch.int32) for _ in lists]

    monkeypatch.setattr(lm, "_pack_int_tensors_to_cuda", pack)
    yield
    reset_all_contexts()


# --- reference implementations (main, 7beebc7) ----------------------------------------------------------------------


def _old_metadata(token_to_slot):
    slot_to_indices: dict[int, list[int]] = {}
    for token_idx, slot_id in enumerate(token_to_slot):
        if slot_id >= 0:
            slot_to_indices.setdefault(slot_id, []).append(token_idx)
    active = sorted(slot_to_indices)
    counts = [len(slot_to_indices[s]) for s in active]
    indices = [t for s in active for t in slot_to_indices[s]]
    offsets = [0]
    for c in counts:
        offsets.append(offsets[-1] + c)
    return indices, active, counts, offsets


def _old_expand(sample_to_slot, sequence_length, cfg_branches, padded_batch_size=None):
    batch_size = len(sample_to_slot)
    padded_batch_size = batch_size if padded_batch_size is None else padded_batch_size
    padded = sample_to_slot + [-1] * (padded_batch_size - batch_size)
    return [slot for _ in range(cfg_branches) for slot in padded for _ in range(sequence_length)]


def _old_prefill_context(block_size, seqs, graphs=False):
    positions, cu_q, cu_k, max_q, max_k, slots = [], [0], [0], 0, 0, []
    for seq in seqs:
        seq_len = seq.seq_length
        positions.extend(range(seq.num_cached_tokens, seq_len))
        q = seq_len - seq.num_cached_tokens
        cu_q.append(cu_q[-1] + q)
        cu_k.append(cu_k[-1] + seq_len)
        max_q, max_k = max(q, max_q), max(seq_len, max_k)
        if not seq.block_table:
            continue
        pos = seq.num_cached_tokens
        while pos < seq_len:
            i, off = divmod(pos, block_size)
            end = min(seq_len, (i + 1) * block_size)
            start = seq.block_table[i] * block_size + off
            slots.extend(range(start, start + end - pos))
            pos = end
    tables = None
    if cu_k[-1] > cu_q[-1] or graphs:
        m = max(len(s.block_table) for s in seqs)
        tables = [s.block_table + [-1] * (m - len(s.block_table)) for s in seqs]
    return positions, cu_q, cu_k, max_q, max_k, slots, tables


def _old_can_schedule(manager, running, cand):
    if not manager.enabled or cand is None:
        return True
    required = set(running) | {cand}
    resident = {s.adapter_id for s in manager._slots if s.adapter_id is not None}
    reclaimable = sum(1 for s in manager._slots if s.adapter_id is None)
    reclaimable += sum(1 for s in manager._slots if s.adapter_id is not None and s.adapter_id not in required
                       and s.resident_state == LoRAResidentState.IDLE)
    return len(required - resident) <= reclaimable


def _old_allocate(bm, seq):
    h = bm.chain_seed(seq)
    cache_miss = False
    last = seq.num_blocks - 1
    for i in range(seq.num_blocks):
        token_ids = seq.block(i)
        h = bm.compute_hash(token_ids, h) if len(token_ids) == bm.block_size else -1
        block_id = bm.hash_to_block_id.get(h, -1)
        if block_id == -1 or i == last or bm.blocks[block_id].token_ids != token_ids:
            cache_miss = True
        if cache_miss:
            block_id = bm.free_block_ids[0]
            block = bm._allocate_block(block_id)
        else:
            seq.num_cached_tokens += bm.block_size
            if block_id in bm.used_block_ids:
                block = bm.blocks[block_id]
                block.ref_count += 1
            else:
                block = bm._allocate_block(block_id)
        if h != -1:
            bm._publish(block, h, token_ids)
        seq.block_table.append(block_id)


# --- 1. one host-to-device copy ------------------------------------------------------------------------------------


def test_host_pack_views_have_the_arrays_dtypes_shapes_and_values():
    rng = np.random.default_rng(0)
    arrays = [rng.integers(-5, 1 << 40, 13).astype(np.int64), rng.integers(0, 9, (3, 4)).astype(np.int32),
              rng.random(7) > 0.5, rng.random((2, 3, 5)).astype(np.float32), np.zeros(0, np.int64),
              np.array(3, dtype=np.int32), np.zeros((0, 4, 2), np.float32)]
    pack = HostPack("cpu")
    ids = [pack.add(a) for a in arrays]
    parts = [rng.random((n, 4, 2)).astype(np.float32) for n in (2, 0, 5)]
    i_cat = pack.add_concat(parts)
    views = pack.upload()
    for i, a in zip(ids, arrays):
        v = views[i]
        assert v.dtype == torch.from_numpy(np.asarray(a)).dtype and tuple(v.shape) == a.shape
        assert v.is_contiguous()
        np.testing.assert_array_equal(v.numpy(), a)
    np.testing.assert_array_equal(views[i_cat].numpy(), np.concatenate(parts))
    base = views[0].data_ptr()
    assert all((v.data_ptr() - base) % HostPack.ALIGN == 0 for v in views if v.numel())  # aligned like allocations


# --- 2. vectorized context and LoRA metadata -----------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_lora_metadata_matches_the_per_token_loop(seed):
    rng = np.random.default_rng(seed)
    for n in (0, 1, 7, 300, lm._LOOP_MAX_ROWS + 1, 1300):  # both the loop (small) and numpy (large) paths
        slots = rng.integers(-1, 6, n).tolist()
        assert lm._build_lora_metadata_from_token_to_slot(slots) == _old_metadata(slots)
    assert lm._build_lora_metadata_from_token_to_slot([-1, -1]) == _old_metadata([-1, -1])


def test_expand_dit_lora_slots_matches_the_comprehension():
    big = [i % 4 - 1 for i in range(40)]  # 48 x 11 rows: the numpy path; the others take the comprehension
    for slots, seq_len, cfg, padded in [([0, 2, -1], 11, 2, 8), ([1], 11, 1, None), ([], 3, 2, 4), ([3, 3], 0, 2, 2),
                                        (big, 11, 2, 48), (big, 11, 1, None)]:
        assert expand_dit_lora_slots(slots, seq_len, cfg, padded) == _old_expand(slots, seq_len, cfg, padded)


def test_staged_lora_context_equals_materialize():
    for slots in ([-1, -1, -1], [2, 2, 0, -1, 0], []):
        host = build_lora_context_from_slot_list(slots, materialize_device=False)
        want = materialize_lora_context(host)
        pack = HostPack("cpu")
        make = stage_lora_context(host, pack.add)
        got = make(pack.upload())
        for f in ("token_to_slot", "token_indices_sorted_by_slot", "active_slot_ids", "num_tokens_per_slot",
                  "slot_start_offsets"):
            a, b = getattr(want, f), getattr(got, f)
            assert (a is None) == (b is None), f
            if a is not None:
                assert a.dtype == b.dtype and torch.equal(a, b), f
        for f in ("no_lora_flag", "num_active_loras", "host_token_to_slot", "host_token_indices_sorted_by_slot",
                  "host_active_slot_ids", "host_num_tokens_per_slot"):
            assert getattr(want, f) == getattr(got, f), f


BS = 4  # small KV blocks: prefixes end inside blocks, sequences span several


def _runner(n_adapters=3, max_loras=3, graphs=False):
    from nanovllm_voxcpm.models.voxcpm2.runner import VoxCPM2Runner

    r = object.__new__(VoxCPM2Runner)
    r.block_size, r.world_size, r.rank = BS, 1, 0
    r.enforce_eager = False
    r.graph_bs = [1, 2, 4, 8, 16]
    r.prefill_diffusion_graphs = {"base": {}, "lora": {}}
    r.prefill_backbone_graphs = {64: None} if graphs else {}
    r.model = SimpleNamespace(forward_backbone=None)
    r.max_loras, r.max_lora_rank = max_loras, 8
    r.lora_config = SimpleNamespace(enable_dit=True)
    r.patch_size, r.feat_dim = 2, 3
    r._fast_noise = True
    r.lora_runtime = LoRARuntime(max_loras=max_loras, max_lora_rank=8)
    r._load_lora_slot = lambda slot, payload: None
    r._enc_cache_size = 16
    r._enc_rows = OrderedDict()
    r._enc_cache = torch.zeros(1)
    r.pack_device = "cpu"
    for a in range(n_adapters):
        r.lora_runtime.register_lora(f"a{a}", LoRAModelPayload(modules={}, rank=8, alpha=8.0), adapter_id=a)
    return r


def _tasks(rng, n, warmup=False, adapters=(None, 0, 1, 2)):
    tasks, next_block = [], 0
    for i in range(n):
        length = int(rng.integers(1, 15))
        cached = 0 if warmup else int(rng.integers(0, length))  # token-granular, inside a block
        blocks = [] if warmup else list(range(next_block, next_block + (length + BS - 1) // BS))
        next_block += len(blocks) + int(rng.integers(0, 3))
        tasks.append(RunnerTask(blocks, length, cached, BS, None, adapter_id=adapters[i % len(adapters)],
                                seq_id=f"s{i}"))
    return tasks


def _context_snapshot():
    c = get_context()
    out = {"max": (c.is_prefill, c.max_seqlen_q, c.max_seqlen_k)}
    for f in ("cu_seqlens_q", "cu_seqlens_k", "slot_mapping", "block_tables"):
        v = getattr(c, f)
        out[f] = None if v is None else (v.dtype, v.tolist())
    for d in DOMAINS:
        lc = get_lora_context(d)
        for f in ("token_to_slot", "token_indices_sorted_by_slot", "active_slot_ids", "num_tokens_per_slot",
                  "slot_start_offsets"):
            v = getattr(lc, f)
            out[d + f] = None if v is None else (v.dtype, v.tolist())
        out[d] = (lc.no_lora_flag, lc.num_active_loras, lc.host_token_to_slot, lc.host_token_indices_sorted_by_slot,
                  lc.host_active_slot_ids, lc.host_num_tokens_per_slot)
    return out


@pytest.mark.parametrize("seed,warmup,graphs,adapters", [(0, False, False, (None, 0, 1, 2)), (1, False, True, (None,)),
                                                         (2, True, False, (None,)), (3, False, False, (2, 2, 0))])
def test_prefill_context_matches_the_per_token_loop(seed, warmup, graphs, adapters):
    """Positions, cu_seqlens, max lengths, slot mapping, block tables and the three LoRA contexts of the staged,
    vectorized prefill context equal the per-token loop's (main) on the same tasks."""
    from nanovllm_voxcpm.engine.model_runner import LORA_DOMAINS, BaseModelRunner

    rng = np.random.default_rng(seed)
    tasks = _tasks(rng, 9, warmup=warmup, adapters=adapters)
    r = _runner()
    if graphs:
        r.prefill_backbone_graphs = {64: None}
    positions = r.prepare_prefill_context(tasks)
    got = _context_snapshot()
    got_pos = (positions.dtype, positions.tolist())
    reset_all_contexts()

    ref = _runner()
    if graphs:
        ref.prefill_backbone_graphs = {64: None}
    pos, cu_q, cu_k, max_q, max_k, slots, tables = _old_prefill_context(BS, tasks, graphs=graphs)
    assert got_pos == (torch.int64, pos)
    assert got["cu_seqlens_q"] == (torch.int32, cu_q) and got["cu_seqlens_k"] == (torch.int32, cu_k)
    assert got["max"] == (True, max_q, max_k)
    assert got["slot_mapping"] == (torch.int32, slots)
    assert got["block_tables"] == (None if tables is None else (torch.int32, tables))
    token_counts = [t.seq_length - t.num_cached_tokens for t in tasks]
    want = BaseModelRunner._build_lora_contexts(ref, tasks, token_counts,
                                                materialize_domains={LM_LORA_DOMAIN, PROJ_LORA_DOMAIN})
    for d, lc in want.items():
        for f in ("token_to_slot", "token_indices_sorted_by_slot", "active_slot_ids", "num_tokens_per_slot",
                  "slot_start_offsets"):
            v = getattr(lc, f)
            assert got[d + f] == (None if v is None else (v.dtype, v.tolist())), (d, f)
        assert got[d] == (lc.no_lora_flag, lc.num_active_loras, lc.host_token_to_slot,
                          lc.host_token_indices_sorted_by_slot, lc.host_active_slot_ids, lc.host_num_tokens_per_slot)
    assert LORA_DOMAINS == DOMAINS


# --- 3. the prefill step's inputs (one copy) -----------------------------------------------------------------------


def _payload_tasks(rng, n, feat_dim=3, patch=2):
    from nanovllm_voxcpm.models.voxcpm2.runner import VoxCPM2Payload

    tasks = _tasks(rng, n)
    for t in tasks:
        q = t.seq_length - t.num_cached_tokens
        masks = (rng.random(q) > 0.6).tolist()
        t.custom_payload = VoxCPM2Payload(
            text_tokens=rng.integers(0, 1000, q).tolist(),
            feats=np.where(np.asarray(masks)[:, None, None], rng.random((q, patch, feat_dim)), 0).astype(np.float32),
            feat_masks=masks, temperature=float(rng.random()), cfg_value=float(1 + rng.random()),
            seed=int(rng.integers(0, 1 << 62)) if rng.random() > 0.3 else None, seed_step=int(rng.integers(0, 5)),
            feat_keys=[int(k) if m else 0 for k, m in zip(rng.integers(1, 6, q), masks)] if rng.random() > 0.2 else None,
        )
    return tasks


def _old_inputs(r, tasks):
    """main's VoxCPM2Runner._run inputs (per-task arrays, concatenated) and its _locenc_inputs."""
    p = [t.custom_payload for t in tasks]
    tok = np.concatenate([np.array(x.text_tokens, dtype=np.int64) for x in p])
    mask = np.concatenate([np.array(x.feat_masks, dtype=np.bool_) for x in p])
    feat = np.concatenate([x.feats for x in p])
    pos, keys, off = [], [], 0
    for x in p:
        m = np.array(x.feat_masks, dtype=np.bool_)
        idx = np.flatnonzero(m)
        pos.append(idx + off)
        off += len(m)
        keys.extend(np.array(x.feat_keys, dtype=np.uint64)[idx].tolist() if x.feat_keys is not None else [0] * len(idx))
    cache = r._enc_rows
    for k in keys:
        if k in cache:
            cache.move_to_end(k)
    rows, miss, fresh, scratch = [], [], {}, r._enc_cache_size
    for i, k in enumerate(keys):
        row = cache.get(k) if k else None
        if row is None and k in fresh:
            row = fresh[k]
        if row is None:
            if not k:
                row, scratch = scratch, scratch + 1
            elif len(cache) < r._enc_cache_size:
                row = len(cache)
            else:
                _, row = cache.popitem(last=False)
            if k:
                cache[k] = fresh[k] = row
            miss.append(i)
        rows.append(row)
    return {"text_tokens": tok, "feat_mask": mask, "feat": feat,
            "temperature": np.array([x.temperature for x in p], dtype=np.float32),
            "cfg_value": np.array([x.cfg_value for x in p], dtype=np.float32),
            "enc_pos": np.concatenate(pos).astype(np.int64), "enc_rows": np.array(rows, dtype=np.int64),
            "enc_miss": np.array(miss, dtype=np.int64)}


def _old_noise_seeds(tasks):
    return [derive_step_seed(int(t.custom_payload.seed), t.custom_payload.seed_step)
            if t.custom_payload.seed is not None and t.custom_payload.seed >= 0 else random.getrandbits(63)
            for t in tasks]


@pytest.mark.parametrize("seed", range(3))
def test_prefill_inputs_and_noise_seeds_match_the_per_tensor_path(seed, monkeypatch):
    import voxcpm_kernels

    rng = np.random.default_rng(seed)
    tasks = _payload_tasks(rng, 8)
    r, ref = _runner(), _runner()
    seen = {}

    def fake_randn(seeds, row_shape, dtype=torch.float32, device="cuda", seed_t=None):
        seen["seed_t"] = seed_t.clone()
        return torch.zeros(len(seeds), *row_shape, dtype=dtype)

    monkeypatch.setattr(voxcpm_kernels, "seeded_randn", fake_randn)
    r._run_with_inputs = lambda seqs, inputs, is_prefill, prof: seen.update(inputs=inputs)
    random.seed(seed)
    r._run(tasks, True)
    random.seed(seed)
    want = _old_inputs(ref, tasks)
    want["seed_t"] = np.array([s & 0x7FFFFFFFFFFFFFFF for s in _old_noise_seeds(tasks)], dtype=np.int64)
    got = seen["inputs"]
    assert r._prefill_has_audio == bool(want["feat_mask"].any())
    for k in ("text_tokens", "feat_mask", "enc_pos", "enc_rows", "enc_miss"):
        assert got[k].dtype == torch.from_numpy(want[k]).dtype, k
        np.testing.assert_array_equal(got[k].numpy(), want[k], err_msg=k)
    for k in ("feat", "temperature", "cfg_value"):  # cast to the model dtype (bf16) as before
        assert torch.equal(got[k], torch.from_numpy(want[k]).to(r.dtype)), k
    np.testing.assert_array_equal(seen["seed_t"].numpy(), want["seed_t"])
    assert r._enc_rows == ref._enc_rows  # the LocEnc cache's rows and LRU order


@pytest.mark.parametrize("n", [1, 16, 17, 60])
def test_noise_seeds_match_the_per_row_derivation(n):
    """Per row up to 16 rows, numpy above: the same seeds, and unseeded rows draw random bits in row order."""
    rng = np.random.default_rng(n)
    tasks = _payload_tasks(rng, n)
    r = _runner()
    random.seed(n)
    got = r._noise_seeds(tasks)
    random.seed(n)
    want = np.array([s & 0x7FFFFFFFFFFFFFFF for s in _old_noise_seeds(tasks)], dtype=np.int64)
    assert got.dtype == np.int64
    np.testing.assert_array_equal(got, want)


def test_vectorized_step_seeds_match_splitmix():
    rng = random.Random(0)
    seeds = [rng.getrandbits(64) for _ in range(200)] + [0, 1, (1 << 64) - 1]
    steps = [rng.randrange(0, 5000) for _ in seeds]
    got = derive_step_seeds(seeds, steps).tolist()
    assert got == [derive_step_seed(s, k) for s, k in zip(seeds, steps)]


# --- 4. scheduling: admission ahead of the step, LoRA bookkeeping, allocation -----------------------------------------


class _Runner:
    def __init__(self):
        self.calls = []

    def call(self, name, *args):
        self.calls.append((name, args))


class _Engine(LLMEngineBase):
    def __init__(self, max_loras=2, n_adapters=3, **cfg):
        self.model_runner = _Runner()
        self.lora_manager = LoRAManager(max_loras=max_loras, max_lora_rank=8)
        for a in range(n_adapters):
            self.lora_manager.register_lora(f"a{a}", LoRAModelPayload(modules={}, rank=8, alpha=8.0))
        self.scheduler = Scheduler(_config(**cfg), callbacks=self)
        self.ps = []


def _config(max_num_seqs=6, blocks=40, block_size=BS, max_tokens=40, prefill_max_seqs=0):
    return SimpleNamespace(max_num_seqs=max_num_seqs, max_num_batched_tokens=max_tokens, num_kvcache_blocks=blocks,
                           kvcache_block_size=block_size,
                           model_config=SimpleNamespace(turbo=SimpleNamespace(prefill_max_seqs=prefill_max_seqs)))


def _requests(rng, n):
    return [(f"r{i}", rng.integers(1, 50, int(rng.integers(1, 12))).tolist(), [None, 0, 1, 2][int(rng.integers(0, 4))])
            for i in range(n)]


def _state(eng, seqs):
    sch = eng.scheduler
    return ([s.seq_id for s in seqs], [s.num_cached_tokens for s in seqs], [s.seq_id for s in sch.waiting],
            [s.seq_id for s in sch.running], sch.num_streams,
            [(e.adapter_id, e.cpu_ref_count, e.gpu_running_ref_count, e.state) for e in eng.lora_manager.list_loras()])


@pytest.mark.parametrize("seed", range(8))
def test_admitting_ahead_of_the_step_schedules_what_the_step_would(seed):
    """Requests registered in chunks with admit() between them (the admission window's idle moments) end up in the
    same prefill step, in the same order, with the same blocks, budgets, refusals (token budget, KV blocks, stream
    slots, LoRA slots) and bookkeeping as when the step admits them all at once."""
    rng = np.random.default_rng(seed)
    reqs = _requests(rng, 14)
    cuts = sorted(rng.choice(np.arange(1, 14), 3, replace=False).tolist())
    a, b = _Engine(), _Engine()
    for k, (sid, toks, adapter) in enumerate(reqs):
        a.add_sequence(Sequence(sid, toks, BS, adapter_id=adapter))
        b.add_sequence(Sequence(sid, toks, BS, adapter_id=adapter))
        if k + 1 in cuts:
            b.scheduler.admit()
    seqs_a, pre_a = a.scheduler.schedule()
    seqs_b, pre_b = b.scheduler.schedule()
    assert pre_a and pre_b
    assert _state(a, seqs_a) == _state(b, seqs_b)
    assert [s.block_table for s in seqs_a] == [s.block_table for s in seqs_b]
    assert b.scheduler.num_queued == len(b.scheduler.waiting)
    # the next steps too (the refused ones in the next prefill, then decodes)
    for _ in range(3):
        for s in seqs_a + seqs_b:
            s.append_token(7)
        seqs_a, pre_a = a.scheduler.schedule()
        seqs_b, pre_b = b.scheduler.schedule()
        assert pre_a == pre_b and _state(a, seqs_a) == _state(b, seqs_b)


def test_cancel_of_an_admitted_request_returns_its_budget():
    eng = _Engine(max_tokens=12)
    for sid, n in (("x", 6), ("y", 6), ("z", 6)):
        eng.add_sequence(Sequence(sid, list(range(1, n + 1)), BS))
    assert [s.seq_id for s in eng.scheduler.admit()] == ["x", "y"]  # z does not fit the token budget
    assert eng.scheduler.num_queued == 3
    eng.cancel_sequence("x")
    seqs, is_prefill = eng.scheduler.schedule()
    assert is_prefill and [s.seq_id for s in seqs] == ["y", "z"]  # x's budget went to z
    assert eng.scheduler.num_streams == 2 and not eng.scheduler.waiting


def test_lora_starts_are_one_runner_call_per_pass_with_the_same_counts():
    eng = _Engine(max_loras=3)
    for i, adapter in enumerate([0, None, 1, 0, 2]):
        eng.add_sequence(Sequence(f"s{i}", [i + 1], BS, adapter_id=adapter))
    eng.scheduler.schedule()
    started = [args for name, args in eng.model_runner.calls if name.startswith("lora_on_sequence") and "started" in name]
    assert started == [([0, 1, 0, 2],)]
    counts = {e.adapter_id: e.gpu_running_ref_count for e in eng.lora_manager.list_loras()}
    assert counts == {0: 2, 1: 1, 2: 1}


def test_single_pass_can_schedule_matches_the_set_version():
    rng = random.Random(0)
    rt = LoRARuntime(max_loras=4, max_lora_rank=8)
    for a in range(7):
        rt.register_lora(f"a{a}", LoRAModelPayload(modules={}, rank=8, alpha=8.0), adapter_id=a)
    for trial in range(300):
        for slot in rt._slots:  # random residency: free, idle or active slots, distinct adapters
            slot.adapter_id = None
        for slot, a in zip(rt._slots, rng.sample(range(7), rng.randint(0, 4))):
            slot.adapter_id = a
            slot.resident_state = rng.choice([LoRAResidentState.IDLE, LoRAResidentState.ACTIVE])
        running = set(rng.sample(range(7), rng.randint(0, 5)))
        cand = rng.choice([None, *range(7)])
        assert rt.can_schedule(running, cand) == _old_can_schedule(rt, running, cand), trial


def test_targeted_slot_refresh_matches_the_full_refresh():
    """A runtime with resident adapters through starts, preemptions and batch plans: the slot states after each
    event equal those of a twin that refreshes every slot."""
    payload = LoRAModelPayload(modules={}, rank=8, alpha=8.0)
    a, b = LoRARuntime(max_loras=2, max_lora_rank=8), LoRARuntime(max_loras=2, max_lora_rank=8)
    for rt in (a, b):
        for i in range(3):
            rt.register_lora(f"a{i}", payload, adapter_id=i)
    full = lambda rt: (lambda entry: rt._refresh_slot_states())  # noqa: E731
    b._refresh_entry_slot = full(b)
    rng = random.Random(1)
    running = []
    for _ in range(200):
        op = rng.random()
        if op < 0.4:
            ids = [rng.choice([None, 0, 1, 2]) for _ in range(rng.randint(1, 3))]
            ids = [i for i in ids if i is None or len({x for x in running + ids if x is not None}) <= 2]
            try:
                plans = [rt.build_batch_plan(ids, [1] * len(ids), lambda s, p: None) for rt in (a, b)]
            except RuntimeError:
                continue
            assert plans[0].adapter_to_slot == plans[1].adapter_to_slot
        elif op < 0.7 and len(running) < 4:
            aid = rng.choice([None, 0, 1, 2])
            for rt in (a, b):
                rt.on_sequence_enqueued(aid)
                rt.on_sequence_started(aid)
            running.append(aid)
        elif running:
            aid = running.pop(rng.randrange(len(running)))
            for rt in (a, b):
                rt.on_sequence_preempted(aid)
                rt.on_sequence_finished(aid, was_running=False)
        assert [(s.adapter_id, s.resident_state) for s in a._slots] == [(s.adapter_id, s.resident_state)
                                                                        for s in b._slots]


def test_allocate_fast_path_matches_the_loop():
    rng = np.random.default_rng(0)
    a, b = BlockManager(64, BS), BlockManager(64, BS)
    held = []
    for i in range(300):
        n = int(rng.integers(1, 3 * BS))
        toks = rng.integers(0, 3, n).tolist()  # repeats: full blocks hit the prefix cache
        adapter = [None, 3][i % 2]
        sa, sb = Sequence(f"a{i}", toks, BS, adapter_id=adapter), Sequence(f"b{i}", toks, BS, adapter_id=adapter)
        while not a.can_allocate(sa):
            x, y = held.pop(0)
            a.deallocate(x)
            b.deallocate(y)
        a.allocate(sa)
        _old_allocate(b, sb)
        assert (sa.block_table, sa.num_cached_tokens) == (sb.block_table, sb.num_cached_tokens)
        held.append((sa, sb))
        if rng.random() < 0.5:
            x, y = held.pop(int(rng.integers(0, len(held))))
            a.deallocate(x)
            b.deallocate(y)
    assert a.hash_to_block_id == b.hash_to_block_id and list(a.free_block_ids) == list(b.free_block_ids)


# --- 5. VoxCPM2 engine: reference-prefix match and copies, prepared steps ------------------------------------------

FEAT, PATCH = 4, 2


class _FakeRunner:
    """model_runner.call: run() records its tasks and returns one patch per sequence."""

    def __init__(self):
        self.runs, self.copies, self.slots, self.overlap_hook = [], [], set(), None

    def call(self, name, *args):
        if name == "run":
            tasks, is_prefill = args[:2]
            self.runs.append((tasks, is_prefill, list(self.copies)))  # (the copies made before this run)
            if self.overlap_hook is not None:
                self.overlap_hook()
            self.slots |= {t.seq_id for t in tasks}
            return [{"latents": np.full((PATCH, FEAT), 0.5, np.float32), "stop_flag": 0,
                     "waveforms": np.ones(8, np.float32), "patch_silent": False} for _ in tasks]
        if name == "copy_kv_blocks":
            self.copies.append((list(args[0]), list(args[1])))
        if name == "release_sequence_state":
            self.slots.discard(args[0])
        return None


def _voxcpm2_engine(max_num_seqs=8, blocks=64):
    from nanovllm_voxcpm.models.voxcpm2.engine import VoxCPM2Engine

    eng = object.__new__(VoxCPM2Engine)
    eng.n_decode_pad_frames, eng.feat_dim, eng.patch_size = 12, FEAT, PATCH
    eng.audio_start_token, eng.ref_audio_start_token, eng.ref_audio_end_token = 101, 103, 104
    eng.block_size, eng.max_model_len = 256, 512
    eng.ref_prefix_blocks, eng.voices = OrderedDict(), {}
    eng.ref_prefix_cache_enabled, eng.ref_prefix_cache_size = True, 8
    eng.locenc_cache_enabled = True
    eng.head_tokens, eng.head_patches, eng.head_min_rest = 0, 3, 8
    eng.ps = []
    eng.lora_manager = LoRAManager(max_loras=2, max_lora_rank=8)
    for name in ("d0", "d1"):
        eng.lora_manager.register_lora(name, LoRAModelPayload(modules={}, rank=8, alpha=8.0))
    eng.model_runner = _FakeRunner()
    eng.scheduler = Scheduler(_config(max_num_seqs=max_num_seqs, blocks=blocks, block_size=256, max_tokens=4096),
                              callbacks=eng)
    eng._text_ids = lambda prompt_text, target_text: [7 + (hash(w) % 50) for w in (prompt_text + target_text).split()]
    return eng


def _plain_match(eng, seq):
    entry = eng.ref_prefix_blocks.get(seq.custom_payload.ref_key)
    if entry is None:
        return -1, 0
    tokens = entry[1]
    n, k = min(len(tokens), len(seq) - 1), 0
    while k < n and seq.token_ids[k] == tokens[k]:
        k += 1
    return entry[0], k


def _cache_voice(eng, voice, lora, text, seed=1):
    """A first request of (voice, lora) runs its prefill: its first block becomes the cached prefix."""
    eng.add_request(f"first-{voice}-{lora}-{text}", text, voice=voice, lora_name=lora, seed=seed, max_generate_length=1)
    eng.step()


def test_prefix_match_with_memo_and_reference_provenance_equals_the_plain_loop():
    rng = np.random.default_rng(0)
    eng = _voxcpm2_engine()
    lat = rng.standard_normal((PATCH * 20, FEAT)).astype(np.float32)
    eng.register_voice("v", lat)
    eng.register_voice("twin", lat.copy())  # equal latents, another PreparedAudio: the same cache bucket
    _cache_voice(eng, "v", "d0", "a b c d")
    cases = [eng.add_request(f"q{i}", text, voice=voice, lora_name="d0", max_generate_length=5)
             for i, (voice, text) in enumerate([("v", "a b x"), ("v", "a b c d e"), ("twin", "a b c"), ("v", "z"),
                                                ("twin", "a"), ("v", "a b c d")])]
    for seq in cases:
        want = _plain_match(eng, seq)
        assert eng._prefix_match(seq) == want and eng._prefix_match(seq) == want  # computed, then the memo
        assert want[1] >= 22  # [ref start] 20 patches [ref end]
    # a replaced cache entry (same key) is not answered from the memo
    key = cases[0].custom_payload.ref_key
    block, tokens, ref = eng.ref_prefix_blocks[key]
    eng.ref_prefix_blocks[key] = (block, tokens[:10], ref)
    assert eng._prefix_match(cases[0]) == _plain_match(eng, cases[0]) == (block, 10)


def test_prefix_copies_start_at_admission_with_the_runners_pairs():
    """The engine copies the claimed prefixes at admission (one call); the runner's tasks no longer ask for a copy,
    and the (source, destination, tokens) are the ones the runner used to copy."""
    rng = np.random.default_rng(1)
    eng = _voxcpm2_engine()
    eng.register_voice("v", rng.standard_normal((PATCH * 6, FEAT)).astype(np.float32))
    _cache_voice(eng, "v", "d1", "p q")
    eng.model_runner.copies.clear()
    seqs = [eng.add_request(f"r{i}", f"w{i} x", voice="v", lora_name="d1", max_generate_length=4) for i in range(3)]
    want = [(eng.ref_prefix_blocks[s.custom_payload.ref_key][0], _plain_match(eng, s)[1]) for s in seqs]
    eng.step()
    tasks, is_prefill, copies = eng.model_runner.runs[-1]
    assert is_prefill and all(t.custom_payload.prefix_src_block is None for t in tasks)
    (pairs, counts), = copies  # one call, before the step's run
    assert [(src, n) for (src, _), n in zip(pairs, counts)] == want
    assert [dst for _, dst in pairs] == [s.block_table[0] for s in seqs]
    assert [t.num_cached_tokens for t in tasks] == [n for _, n in want]


def test_prepared_prefill_runs_the_same_tasks_as_the_step():
    """prepare_prefill between registrations (idle moments) gives the step the tasks it would have built itself."""
    rng = np.random.default_rng(2)
    lat = rng.standard_normal((PATCH * 5, FEAT)).astype(np.float32)
    runs = []
    for prepare in (False, True):
        eng = _voxcpm2_engine()
        eng.register_voice("v", lat)
        _cache_voice(eng, "v", "d0", "k l m")
        eng.model_runner.copies.clear()
        for i in range(6):
            eng.add_request(f"r{i}", f"t{i} u v", voice="v", lora_name=["d0", None, "d1"][i % 3], seed=10 + i,
                            max_generate_length=3)
            if prepare and i in (1, 4):
                assert eng.prepare_prefill() > 0
        eng.cancel_sequence("r1")  # (with preparation: cancelled after it was admitted and prepared)
        eng.step()
        tasks, _, copies = eng.model_runner.runs[-1]
        runs.append(([(t.seq_id, t.block_table, t.seq_length, t.num_cached_tokens, t.adapter_id,
                       list(t.custom_payload.text_tokens), list(t.custom_payload.feat_masks),
                       t.custom_payload.feats.tobytes(), t.custom_payload.seed) for t in tasks],
                     sorted((s, n) for p, c in copies for (s, _), n in zip(p, c))))
    assert "r1" not in [row[0] for row in runs[0][0] + runs[1][0]] and len(runs[0][0]) == 5
    # blocks may differ (the cancelled request held one), everything else is the same
    strip = lambda rows: [r[:1] + r[2:] for r in rows]  # noqa: E731
    assert strip(runs[0][0]) == strip(runs[1][0]) and runs[0][1] == runs[1][1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_steps_upload_through_one_page_locked_buffer():
    """PinnedStaging: every upload through the one buffer reaches the GPU with its own values (the next one waits
    for the copy before writing), the buffer grows when a step needs more, and the views keep their alignment."""
    from nanovllm_voxcpm.engine.model_runner import PinnedStaging

    staging, rng = PinnedStaging(), np.random.default_rng(0)
    for n in (100, 3000, 50, 20000):
        a = rng.integers(0, 1 << 30, n, dtype=np.int64)
        b = rng.standard_normal((n, 3)).astype(np.float32)
        pack = HostPack("cuda", staging)
        ia, ib = pack.add(a), pack.add_concat([b[: n // 2], b[n // 2:]])
        views = pack.upload()
        assert staging.buf.is_pinned() and staging.buf.numel() >= n * 8
        assert np.array_equal(views[ia].cpu().numpy(), a) and np.array_equal(views[ib].cpu().numpy(), b)
        assert all(v.data_ptr() % HostPack.ALIGN == views[0].data_ptr() % HostPack.ALIGN for v in views)


def test_a_text_twice_in_a_registration_batch_is_tokenized_once():
    from nanovllm_voxcpm.models.voxcpm2.engine import VoxCPM2Engine

    batches, singles = [], []

    class Enc:
        def __init__(self, t):
            self.ids, self.offsets = [ord(c) for c in t], [(i, i + 1) for i in range(len(t))]

    class Tok:  # the fast tokenizer behind the wrapper: batched calls and the per-text fallback
        _tokenizer = SimpleNamespace(encode_batch=lambda texts, add_special_tokens: batches.append(list(texts))
                                     or [Enc(t) for t in texts])

        def __call__(self, text, **kw):
            singles.append(text)
            return {"input_ids": [ord(c) for c in text], "offset_mapping": [(i, i + 1) for i in range(len(text))]}

    eng = object.__new__(VoxCPM2Engine)
    eng.tokenizer = SimpleNamespace(tokenizer=Tok())
    eng.pretokenize(["ab", "ab", "cd"])
    assert batches == [["ab", "cd"]]
    assert eng._text_ids("", "ab") == eng._text_ids("", "ab") == [97, 98]  # the second one from the cache too
    assert eng._text_ids("", "cd") == [99, 100] and singles == []
    assert eng._pretok_map == {} and eng._pretok_uses == {}  # nothing kept once every registration came
    assert eng._text_ids("", "xy") == [120, 121] and singles == ["xy"] and eng._pretok_map == {}


def _tiny_tokenizer():
    """A BPE tokenizer shaped like the model's (prepend + replace normalizer, no pre-tokenizer, byte fallback)."""
    from tokenizers import Tokenizer, models, normalizers

    chars = list("abcdefgh ") + ["▁", "<unk>"] + [f"<0x{i:02X}>" for i in range(256)]
    vocab = {c: i for i, c in enumerate(chars)}
    merges = [("▁", "a"), ("b", "c"), ("▁a", "bc")]
    for x, y in merges:
        vocab.setdefault(x + y, len(vocab))
    tok = Tokenizer(models.BPE(vocab, merges, unk_token="<unk>", byte_fallback=True))
    tok.normalizer = normalizers.Sequence([normalizers.Prepend("▁"), normalizers.Replace(" ", "▁")])
    return tok


def _add_cmd(seq_id, target, prompt=""):
    return {"id": seq_id, "type": "add_request", "args": (seq_id, target, None, prompt, 400, 1.0, 2.0, None, None, 1),
            "kwargs": {}}


def test_the_client_tokenizes_a_batch_with_the_engines_tokenizer():
    """NANOVLLM_CLIENT_TOKENIZE: the client's ids and offsets are the engine's (its tokenizer, sent with init_ok);
    CJK texts and other commands are left to the engine, and the engine registers with the client's ids."""
    from tokenizers import Tokenizer

    from nanovllm_voxcpm.models.voxcpm2.engine import VoxCPM2Engine
    from nanovllm_voxcpm.models.voxcpm2.server import AsyncVoxCPM2Server, _request_text

    backend = _tiny_tokenizer()
    client = object.__new__(AsyncVoxCPM2Server)
    client._tokenizer = Tokenizer.from_str(backend.to_str())
    cmds = [_add_cmd("r0", "abc ab"), _add_cmd("r1", "h é", prompt="bc "), _add_cmd("r2", "ab 中文"),
            {"id": "x", "type": "health", "args": (), "kwargs": {}}]
    client._tokenize(cmds)
    for c in cmds[:2]:
        e = backend.encode(_request_text(c), add_special_tokens=False)
        assert c["tok"] == (0, e.ids, e.offsets)  # (the tokenizer's version: init_ok's)
    assert "tok" not in cmds[2] and "tok" not in cmds[3] and _request_text(cmds[3]) is None

    batches = []
    eng = object.__new__(VoxCPM2Engine)
    eng.tokenizer = SimpleNamespace(tokenizer=SimpleNamespace(_tokenizer=SimpleNamespace(
        encode_batch=lambda texts, add_special_tokens: batches.append(list(texts)) or backend.encode_batch(texts))))
    texts = [_request_text(c) for c in cmds[:3]]
    eng.pretokenize(texts, {_request_text(c): c["tok"][1:] for c in cmds[:2]})
    assert batches == []  # the client's two; the CJK one is the per-text path's (not cached)
    assert eng._text_ids("", "abc ab") == cmds[0]["tok"][1] and eng._text_ids("bc ", "h é") == cmds[1]["tok"][1]


def test_the_client_follows_the_engines_tokenizer_when_it_changes():
    """A tagged adapter registered after start replaces the engine's tokenizer (its dialect tags as single tokens):
    the client gets the new one with a new version, and ids from the old one are not used (the engine tokenizes)."""
    from tokenizers import AddedToken

    from nanovllm_voxcpm.models.voxcpm2.server import AsyncVoxCPM2Server, _TokenizerSync

    base, tagged = _tiny_tokenizer(), _tiny_tokenizer()
    tagged.add_special_tokens([AddedToken("<eg>", normalized=False)])
    llm = SimpleNamespace(pretokenize=None, tokenizer=SimpleNamespace(tokenizer=SimpleNamespace(_tokenizer=base)))
    sync = _TokenizerSync(llm)
    client = object.__new__(AsyncVoxCPM2Server)
    client._init_fut = SimpleNamespace(done=lambda: True)
    client._handle({"type": "init_ok", **sync.init_fields()})
    old = [_add_cmd("r0", "<eg>abc")]
    client._tokenize(old)
    assert sync.changed() is None and sync.accept(old[0]["tok"]) == old[0]["tok"][1:]

    llm.tokenizer = SimpleNamespace(tokenizer=SimpleNamespace(_tokenizer=tagged))  # register_lora's swap
    msg = sync.changed()
    assert msg["type"] == "tokenizer" and msg["version"] == 1 and sync.changed() is None
    assert sync.accept(old[0]["tok"]) is None  # tokenized with the old one: not used
    client._handle(msg)
    new = [_add_cmd("r1", "<eg>abc")]
    client._tokenize(new)
    e = tagged.encode("<eg>abc", add_special_tokens=False)
    assert new[0]["tok"] == (1, e.ids, e.offsets) and sync.accept(new[0]["tok"]) == (e.ids, e.offsets)
    assert new[0]["tok"][1] != old[0]["tok"][1]  # the tag is one token now


def test_the_models_tokenizer_round_trips_to_the_client():
    """The engine's fast tokenizer (LlamaTokenizerFast's backend) and the client's copy of it (to_str / from_str)
    give the same ids and offsets on the example texts (skipped without the model's files)."""
    import glob

    from tokenizers import Tokenizer

    snaps = glob.glob(os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
                                   "hub/models--VoxCPM-Kernel--VoxCPM2-Arabic-Dialects/snapshots/*/tokenizer.json"))
    if not snaps:
        pytest.skip("the model's tokenizer is not downloaded")
    from transformers import LlamaTokenizerFast

    backend = LlamaTokenizerFast.from_pretrained(os.path.dirname(snaps[0]))._tokenizer
    client = Tokenizer.from_str(backend.to_str())
    here = os.path.join(os.path.dirname(__file__), "..", "examples")
    texts = [s.strip() for f in ("ar_dialects.txt", "hi_sentences.txt")
             for s in open(os.path.join(here, f), encoding="utf-8") if s.strip()]
    texts += ["<|im_start|>" + texts[0], texts[1] + "  ", " " + texts[2]]
    a = backend.encode_batch(texts, add_special_tokens=False)
    b = client.encode_batch(texts, add_special_tokens=False)
    assert [(e.ids, e.offsets) for e in a] == [(e.ids, e.offsets) for e in b]


def test_a_bursts_first_flush_is_small():
    """NANOVLLM_SUBMIT_FLUSH_FIRST: a burst's commands reach the engine in batches of 16, then 48 (the engine starts
    registering while the client builds the rest), the last one not marked "more", and the next burst starts with 16
    again."""
    import asyncio

    from nanovllm_voxcpm.models.voxcpm2.server import AsyncVoxCPM2Server

    async def run():
        sent = []
        client = object.__new__(AsyncVoxCPM2Server)
        client._engine_error, client._spin_task, client.op_table, client._pending_cmds = None, None, {}, []
        client.queue_in = SimpleNamespace(put_nowait=lambda m: sent.append(
            (len(m["cmds"]), bool(m.get("more"))) if "cmds" in m else (1, False)))
        for burst in (70, 20, 64, 1):
            tasks = [asyncio.ensure_future(client.submit("health")) for _ in range(burst)]
            for _ in range(3):
                await asyncio.sleep(0)
            for t in tasks:
                t.cancel()
        return sent

    # a turn whose commands all left in early flushes (64 = 16 + 48) still ends with an (empty) batch: the engine's
    # admission window ends with the client's turn instead of waiting for more
    assert asyncio.run(run()) == [(16, True), (48, True), (6, False), (16, True), (4, False),
                                  (16, True), (48, True), (0, False), (1, False)]


def test_a_prefill_steps_first_audio_reaches_the_client_as_its_chunks():
    """NANOVLLM_EMIT_BLOCK: equal chunks travel through the ring as one block; the client hands each stream its own
    chunk (the same bytes, in order); chunks that differ, or a single one, keep the per-chunk items."""
    from nanovllm_voxcpm.models.voxcpm2.server import AsyncVoxCPM2Server, _AudioRing, _block_item

    ring = _AudioRing(size=1 << 16)
    try:
        client = object.__new__(AsyncVoxCPM2Server)
        client._ring = _AudioRing(name=ring.shm.name)
        waves = [np.arange(k, k + 6, dtype=np.int16) for k in range(0, 30, 10)]
        item = _block_item(ring, ["a", "b", "c"], waves)
        assert item["type"] == "stream_block"
        res = client._convert({"type": "batch", "items": [{"type": "response", "id": "x", "data": None}, item]})
        assert [i["id"] for i in res["items"]] == ["x", "a", "b", "c"]
        for i, w in zip(res["items"][1:], waves):
            assert i["type"] == "stream" and i["data"].dtype == np.int16 and np.array_equal(i["data"], w)
        assert _block_item(ring, ["a", "b"], [waves[0], waves[1][:5]]) is None
        assert _block_item(ring, ["a"], waves[:1]) is None and _block_item(None, ["a", "b"], waves[:2]) is None
        del client._ring.total, client._ring.data
        client._ring.shm.close()
    finally:
        ring.close()


# --- 6. the server's admission window ------------------------------------------------------------------------------


def _run_main_loop(monkeypatch, feed, env):
    """server.main_loop on a VoxCPM2 engine with a fake runner; ``feed(put)`` sends commands from a thread."""
    import nanovllm_voxcpm.models.voxcpm2.server as server

    eng = _voxcpm2_engine(max_num_seqs=16)
    eng.register_voice("v", np.random.default_rng(3).standard_normal((PATCH * 4, FEAT)).astype(np.float32))
    prepared = []
    orig = eng.prepare_prefill

    def counting_prepare():
        n = orig()
        prepared.append(n)
        return n

    eng.prepare_prefill = counting_prepare

    class Impl:
        def __init__(self, *a, **k):
            self.llm = eng

        def add_request(self, seq_id, target_text, *a, **k):
            eng.add_request(seq_id, target_text, voice="v", lora_name="d0", seed=1, max_generate_length=2)

        def step(self):
            return eng.step()

        def is_finished(self):
            return eng.is_finished()

    monkeypatch.setattr(server, "VoxCPM2ServerImpl", Impl)
    for k, v in {"NANOVLLM_SHM_AUDIO": "0", "NANOVLLM_GC_FREEZE": "0", "RAYON_NUM_THREADS": "1", **env}.items():
        monkeypatch.setenv(k, v)
    q_in, q_out = queue.Queue(), queue.Queue()
    t = threading.Thread(target=feed, args=(q_in.put,), daemon=True)
    t.start()
    server.main_loop(q_in, q_out, (), {})
    t.join(1)
    out = []
    while not q_out.empty():
        out.append(q_out.get())
    return eng, prepared, out


def _add(i):
    return {"id": f"op{i}", "type": "add_request", "args": (f"s{i}", f"w{i} x"), "kwargs": {}}


def _ended(out):
    return {i["id"] for m in out if m.get("type") == "batch" for i in m["items"]
            if i.get("type") == "stream" and i.get("data") is None}


def test_trickling_requests_are_prepared_in_the_windows_idle_moments(monkeypatch):
    def feed(put):
        for i in range(3):
            put(_add(i))
            time.sleep(0.005)
        time.sleep(0.3)
        put({"id": "stop", "type": "stop", "args": (), "kwargs": {}})

    eng, prepared, out = _run_main_loop(monkeypatch, feed, {"NANOVLLM_QUEUE_GAP_MS": "50",
                                                             "NANOVLLM_QUEUE_COALESCE_MS": "50"})
    assert sum(prepared) == 3  # each request admitted in an idle moment of the window
    first, is_prefill, _ = eng.model_runner.runs[0]
    assert is_prefill and [t.seq_id for t in first] == ["s0", "s1", "s2"]  # one head step, as without preparation
    assert _ended(out) == {"s0", "s1", "s2"}


def test_a_burst_handed_over_at_once_runs_without_preparation(monkeypatch):
    def feed(put):
        put({"type": "batch", "cmds": [_add(i) for i in range(4)]})
        time.sleep(0.3)
        put({"id": "stop", "type": "stop", "args": (), "kwargs": {}})

    eng, prepared, out = _run_main_loop(monkeypatch, feed, {"NANOVLLM_QUEUE_BATCH_GO": "4"})
    assert prepared == []  # the window ends with the client's turn: nothing to overlap
    first, is_prefill, _ = eng.model_runner.runs[0]
    assert is_prefill and [t.seq_id for t in first] == ["s0", "s1", "s2", "s3"]
    assert _ended(out) == {"s0", "s1", "s2", "s3"}


# --- 7. the profile helper -----------------------------------------------------------------------------------------


def test_get_by_waits_until_its_deadline_not_to_the_next_millisecond():
    """mp.Queue.get(timeout) polls whole milliseconds: after prepare(), a gap's ~0.7 ms rest waited ~1.07 ms, which
    put every lone request's step ~0.3 ms later than main's."""
    import multiprocessing as mp
    import statistics

    from nanovllm_voxcpm.models.voxcpm2.server import _get_by

    q = mp.Queue()
    for wait in (0.0003, 0.0025):
        took = []
        for _ in range(15):
            t0 = time.perf_counter()
            assert _get_by(q, t0 + wait) is None
            took.append(time.perf_counter() - t0)
        assert wait <= statistics.median(took) < wait + 0.0005
    q.put("cmd")
    time.sleep(0.05)  # the queue's feeder thread has sent it
    assert _get_by(q, time.perf_counter() - 1.0) == "cmd"  # one already there is taken past the deadline


def test_profile_timeline_from_records():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
    import profile_burst

    t0 = 100.0
    ms = lambda x: t0 + x / 1000  # noqa: E731
    recs = [{"kind": "client_flush", "t": ms(0.4), "n": 48, "put_ms": 0.1},
            {"kind": "client_flush", "t": ms(1.0), "n": 2, "put_ms": 0.1},
            {"kind": "batch_recv", "t": ms(0.6), "n": 48, "register_ms": 1.4},
            {"kind": "batch_recv", "t": ms(2.0), "n": 2, "register_ms": 0.1},
            {"kind": "server", "t": ms(2.2), "n": 50},
            {"kind": "prefill", "t": ms(3.0), "t_begin": ms(2.25), "bs": 50, "tokens": 300, "ctx": 0.4, "h2d": 0.3,
             "prep": 0.05, "gpu_prep": 1.6},
            {"kind": "emit", "t": ms(60.0), "n": 50}]
    tl = profile_burst.timeline(recs, t0, [70.0, 71.0, 75.0])
    assert tl["client_sends"] == [0.5, 1.1] and tl["registered"] == [2.0, 2.1]
    assert tl["step_start"] == 2.2 and tl["gap_last_registration_to_step"] == pytest.approx(0.1)
    assert tl["sched_preprocess_ms"] == pytest.approx(0.75) and tl["first_kernel_launch"] == pytest.approx(3.75)
    assert tl["first_kernel_gpu"] == pytest.approx(3.85) and tl["first_audio_out"] == 60.0
    assert tl["ttfa_p50"] == 71.0 and tl["head_seqs"] == 50
