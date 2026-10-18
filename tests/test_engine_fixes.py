"""CPU tests of the engine robustness fixes: segmented-GEMM tables, prefix cache, scheduler admission, mid-step
cancels, head start, streamed-audio errors. Nothing runs on a GPU (CUDA calls are faked); the end-to-end checks on
the real model are tests/gpu/verify_engine_fixes.py."""

import asyncio
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nanovllm_voxcpm.engine.block_manager import BlockManager
from nanovllm_voxcpm.engine.llm_engine import LLMEngineBase
from nanovllm_voxcpm.engine.lora_manager import LoRAManager
from nanovllm_voxcpm.engine.model_runner import BaseModelRunner, RunnerTask, expand_dit_lora_slots
from nanovllm_voxcpm.engine.scheduler import Scheduler, _by_adapter
from nanovllm_voxcpm.engine.sequence import Sequence, SequenceStatus
from nanovllm_voxcpm.utils.context import DIT_LORA_DOMAIN, LM_LORA_DOMAIN, LoRAContext, reset_all_contexts


def _config(max_num_seqs=4, blocks=64, block_size=4, max_tokens=4096, prefill_max_seqs=0):
    return SimpleNamespace(
        max_num_seqs=max_num_seqs, max_num_batched_tokens=max_tokens, num_kvcache_blocks=blocks,
        kvcache_block_size=block_size, model_config=SimpleNamespace(turbo=SimpleNamespace(prefill_max_seqs=prefill_max_seqs)),
    )


def _seq(seq_id, tokens, block_size=4, adapter_id=None) -> Sequence:
    return Sequence(seq_id, list(tokens), block_size, adapter_id=adapter_id)


def _segments(slots) -> int:
    s = np.asarray(slots)
    return int((s[1:] != s[:-1]).sum()) + 1 if s.size else 0


# --- 1. segmented-GEMM tile tables ----------------------------------------------------------------------------------

MAX_LORAS = 11  # VoxCPM2-Arabic-Dialects: every adapter resident
DIT_SEQ_LEN = 3 + 2 * 4  # VoxCPM2Runner: dit_lora_seq_len_offset + 2 * patch_size


def _mixed_step(base_rows=2, graph_bs=16):
    """A decode step's sample slots: base rows and all MAX_LORAS adapters (slot = adapter id), as scheduled."""
    seqs = [_seq(f"b{i}", [1], adapter_id=None) for i in range(base_rows)]
    seqs += [_seq(f"a{a}", [1], adapter_id=a) for a in range(MAX_LORAS)]
    order = _by_adapter(seqs)
    return [-1 if s.adapter_id is None else s.adapter_id for s in order], graph_bs


def test_by_adapter_groups_adapters_and_puts_base_rows_last():
    seqs = [_seq(str(i), [1], adapter_id=a) for i, a in enumerate([None, 2, None, 0, 2, 1])]
    assert [s.seq_id for s in _by_adapter(seqs)] == ["3", "5", "1", "4", "0", "2"]  # stable within a group
    base = [_seq(str(i), [1]) for i in range(3)]
    assert _by_adapter(base) is base


def _padded(slots, rows):
    return list(slots) + [-1] * (rows - len(slots))  # as BaseModelRunner._set_graph_lora_context


def test_mixed_batch_with_base_and_all_adapters_fits_the_captured_segments():
    from voxcpm_kernels.seg_gemm import SegTables

    sample_slots, graph_bs = _mixed_step()
    max_bs = 32  # the domain buffers' rows: the largest captured batch
    n_weights = MAX_LORAS + 1
    lm = _padded(sample_slots, max_bs)
    dit = _padded(expand_dit_lora_slots(sample_slots, DIT_SEQ_LEN, 2, padded_batch_size=graph_bs), 2 * DIT_SEQ_LEN * max_bs)
    assert _segments(lm) <= n_weights
    assert _segments(dit) <= 2 * n_weights
    tables = SegTables()  # no buffers: update() only checks the layout
    tables.segments[DIT_LORA_DOMAIN] = 2 * n_weights  # reserve(..., segments=max_lora_buckets * cfg_branches)
    tables.update(LM_LORA_DOMAIN, lm, n_weights)
    tables.update(DIT_LORA_DOMAIN, dit, n_weights)

    # base rows first (the old order) + graph padding: one segment too many in each domain -> ValueError
    base_first = sorted(sample_slots, key=lambda s: s)
    with pytest.raises(ValueError):
        tables.update(LM_LORA_DOMAIN, _padded(base_first, max_bs), n_weights)
    with pytest.raises(ValueError):
        tables.update(DIT_LORA_DOMAIN, _padded(expand_dit_lora_slots(base_first, DIT_SEQ_LEN, 2, padded_batch_size=graph_bs),
                                               2 * DIT_SEQ_LEN * max_bs), n_weights)


@pytest.mark.parametrize("bm", [64, 128])
def test_captured_grid_covers_every_row_with_its_weight(bm):
    """The graph's grid (max_tiles over the replayed rows) reaches every row of the base-last layout."""
    from voxcpm_kernels.seg_gemm import build_tiles, max_tiles

    sample_slots, graph_bs = _mixed_step()
    max_bs = 32
    cases = [
        (_padded(sample_slots, max_bs), graph_bs, MAX_LORAS + 1),
        (_padded(expand_dit_lora_slots(sample_slots, DIT_SEQ_LEN, 2, padded_batch_size=graph_bs), 2 * DIT_SEQ_LEN * max_bs),
         2 * DIT_SEQ_LEN * graph_bs, 2 * (MAX_LORAS + 1)),
    ]
    for slots, m, segments in cases:
        tiles = build_tiles(slots, bm)
        weight = np.full(m, -100)
        for r0, r1, w in tiles[: max_tiles(m, bm, segments)]:
            weight[r0:min(r1, m)] = w
        np.testing.assert_array_equal(weight, np.asarray(slots[:m]) + 1)


class _FakeEvent:
    made: list = []

    def __init__(self):
        self.recorded = self.waited = 0
        _FakeEvent.made.append(self)

    def record(self):
        self.recorded += 1

    def synchronize(self):
        self.waited += 1


def test_seg_tables_wait_for_the_staging_copy_and_reset_to_base(monkeypatch):
    import voxcpm_kernels.seg_gemm as sg

    _FakeEvent.made = []
    monkeypatch.setattr(sg.torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(sg.torch.cuda, "is_current_stream_capturing", lambda: False)
    t = sg.SegTables()
    key = (LM_LORA_DOMAIN, 64)
    t.bufs[key] = torch.zeros(8, 3, dtype=torch.int32)  # CPU stand-ins for the device table and its pinned staging
    t.staging[key] = torch.zeros(8, 3, dtype=torch.int32)
    t.update(LM_LORA_DOMAIN, [0, 0, 1, -1], 3)
    ev = _FakeEvent.made[0]
    assert (ev.recorded, ev.waited) == (1, 0)
    assert t.bufs[key][:3].tolist() == [[0, 2, 1], [2, 3, 2], [3, 4, 0]]
    t.update(LM_LORA_DOMAIN, [1, 1, 1, 1], 3)
    assert ev.waited == 1  # the previous H2D copy is waited for before its staging buffer is rewritten
    t.update_base(LM_LORA_DOMAIN, 4, 3)
    assert t.bufs[key][:2].tolist() == [[0, 4, 0], [0, 0, 0]]
    copies = ev.recorded
    t.update_base(LM_LORA_DOMAIN, 4, 3)  # the tables hold the base layout already: no copy
    assert ev.recorded == copies


def test_graph_lora_context_resets_the_tables_of_a_domain_without_adapters(monkeypatch):
    import voxcpm_kernels.seg_gemm as sg

    calls = []

    class Recorder:
        def update(self, domain, slots, n_weights):
            calls.append(("update", domain, list(slots)))

        def update_base(self, domain, rows, n_weights):
            calls.append(("base", domain, rows))

    monkeypatch.setattr(sg, "SEG_TABLES", Recorder())
    runner = SimpleNamespace(max_loras=2, _fused_lora_layers=lambda: [object()],
                             _copy_lora_domain_to_graph_vars=lambda *a: None)
    dv = {k: torch.zeros(4, dtype=torch.int32)
          for k in ("token_to_slot", "token_indices_sorted_by_slot", "num_tokens_per_slot", "slot_start_offsets")}
    dv["active_slot_ids"] = torch.arange(-1, 2, dtype=torch.int32)
    gv = {"lora_domains": {DIT_LORA_DOMAIN: dv}}
    try:
        BaseModelRunner._set_graph_lora_context(runner, gv, DIT_LORA_DOMAIN, LoRAContext(no_lora_flag=True))
        BaseModelRunner._set_graph_lora_context(runner, gv, DIT_LORA_DOMAIN,
                                                LoRAContext(no_lora_flag=False, host_token_to_slot=[1, 0]))
    finally:
        reset_all_contexts()
    assert calls == [("base", DIT_LORA_DOMAIN, 4), ("update", DIT_LORA_DOMAIN, [1, 0, -1, -1])]


# --- 2. prefix cache ------------------------------------------------------------------------------------------------

TOKENS = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # two full blocks of 4 + one token


def test_prefix_cache_never_shares_kv_across_adapters():
    bm = BlockManager(num_blocks=32, block_size=4)
    a0 = _seq("a0", TOKENS, adapter_id=0)
    bm.allocate(a0)
    a1 = _seq("a1", TOKENS, adapter_id=1)
    bm.allocate(a1)
    assert a1.num_cached_tokens == 0 and not set(a1.block_table) & set(a0.block_table)
    base = _seq("base", TOKENS)
    bm.allocate(base)
    assert base.num_cached_tokens == 0 and not set(base.block_table) & set(a0.block_table + a1.block_table)
    again = _seq("again", TOKENS, adapter_id=0)
    bm.allocate(again)
    assert again.num_cached_tokens == 8 and again.block_table[:2] == a0.block_table[:2]
    base2 = _seq("base2", TOKENS)
    bm.allocate(base2)
    assert base2.num_cached_tokens == 8 and base2.block_table[:2] == base.block_table[:2]
    # base sequences hash as before (no seed)
    assert bm.blocks[base.block_table[0]].hash == BlockManager.compute_hash(TOKENS[:4])


def test_decode_filled_block_is_hashed_with_the_adapter_seed():
    bm = BlockManager(num_blocks=32, block_size=4)
    s = _seq("s", TOKENS[:3], adapter_id=5)
    bm.allocate(s)
    s.append_token(TOKENS[3])
    bm.may_append(s)  # the block is full now: hashed
    assert bm.blocks[s.block_table[0]].hash == BlockManager.compute_hash(TOKENS[:4], BlockManager.chain_seed(s))
    same, other = _seq("same", TOKENS[:5], adapter_id=5), _seq("other", TOKENS[:5], adapter_id=6)
    bm.allocate(same)
    bm.allocate(other)
    assert same.num_cached_tokens == 4 and other.num_cached_tokens == 0


def _assert_hash_consistency(bm):
    assert len(bm.hash_to_block_id) <= len(bm.blocks)
    for h, block_id in bm.hash_to_block_id.items():
        assert bm.blocks[block_id].hash == h


def test_freed_blocks_stay_cache_candidates_until_recycled():  # upstream: hash-entry lifecycle
    bm = BlockManager(num_blocks=8, block_size=4)
    s1 = _seq("s1", TOKENS)
    bm.allocate(s1)
    before = list(s1.block_table)
    bm.deallocate(s1)
    s2 = _seq("s2", TOKENS)
    bm.allocate(s2)
    assert s2.num_cached_tokens == 8 and s2.block_table[:2] == before[:2]
    assert all(bm.blocks[b].ref_count == 1 for b in s2.block_table)


def test_recycling_drops_stale_hash_entries():
    bm = BlockManager(num_blocks=4, block_size=4)
    s1 = _seq("s1", TOKENS[:8])
    bm.allocate(s1)
    stale = [bm.blocks[b].hash for b in s1.block_table]
    bm.deallocate(s1)
    for i in range(10):
        s = _seq(f"s{i}", [100 * (i + 1) + j for j in range(8)])
        bm.allocate(s)
        bm.deallocate(s)
    assert not set(stale) & set(bm.hash_to_block_id)
    _assert_hash_consistency(bm)


def test_recycled_block_with_equal_tokens_but_another_prefix_is_not_a_hit():
    """A stale entry of a freed block must not match after the block was recycled for equal tokens under another
    prefix (its KV belongs to that prefix)."""
    bm = BlockManager(num_blocks=5, block_size=4)
    p, t, z = [1, 1, 1, 1], [9, 9, 9, 9], [5, 5, 5, 5]
    a = _seq("a", p + t + z)
    bm.allocate(a)  # blocks 0 (p), 1 (t), 2 (z)
    a_blocks = list(a.block_table)
    bm.deallocate(a)  # free: 3 4 2 1 0
    f1, f2 = _seq("f1", [7]), _seq("f2", [8])
    bm.allocate(f1)
    bm.allocate(f2)  # take 3, 4
    b = _seq("b", [2, 2, 2, 2] + t)
    bm.allocate(b)  # its t block lands in a's old t block (1), under prefix [2, 2, 2, 2]
    assert b.block_table[1] == a_blocks[1]
    bm.deallocate(f1)
    bm.deallocate(f2)
    again = _seq("again", p + t + z)
    bm.allocate(again)
    assert again.num_cached_tokens == 4  # p only (resurrected); t is not b's block
    assert again.block_table[1] != b.block_table[1]


def test_fully_cached_prompt_recomputes_its_last_block():
    bm = BlockManager(num_blocks=8, block_size=4)
    a = _seq("a", TOKENS[:8])
    bm.allocate(a)
    b = _seq("b", TOKENS[:8])
    bm.allocate(b)
    assert b.num_cached_tokens == 4  # at least the last token runs: its prefill makes the first patch
    assert b.block_table[0] == a.block_table[0] and b.block_table[1] != a.block_table[1]
    b.append_token(9)
    bm.may_append(b)  # the next decode starts a block: the recomputed last block was hashed
    assert len(b.block_table) == 3


# --- 3. scheduler ---------------------------------------------------------------------------------------------------


def test_fully_cached_prompt_is_prefilled_and_counted():
    sch = Scheduler(_config(max_num_seqs=4, prefill_max_seqs=1))
    sch.add(_seq("a", TOKENS[:8]))
    assert [s.seq_id for s in sch.schedule()[0]] == ["a"]
    b, c = _seq("b", TOKENS[:8]), _seq("c", [7, 7])
    sch.add(b)
    sch.add(c)
    seqs, is_prefill = sch.schedule()
    assert is_prefill and seqs == [b] and b.num_cached_tokens == 4
    assert list(sch.waiting) == [c]  # b took the step's one prefill seat


def _step(sch):
    """schedule() plus the engine's postprocess (every scheduled sequence gets its next token)."""
    seqs, is_prefill = sch.schedule()
    for s in seqs:
        s.append_token(0)
    return seqs, is_prefill


def test_running_streams_are_capped_at_max_num_seqs():
    sch = Scheduler(_config(max_num_seqs=2))
    seqs = [_seq(f"s{i}", [i + 1]) for i in range(3)]
    for s in seqs:
        sch.add(s)
    assert _step(sch) == (seqs[:2], True)
    assert _step(sch) == (seqs[:2], False)  # the third waits instead of a third concurrent stream
    assert list(sch.waiting) == [seqs[2]] and sch.num_streams == 2
    sch.finish(seqs[0])
    assert _step(sch) == ([seqs[2]], True)
    assert sch.num_streams == 2 and len(sch.running) == 2


def test_continuation_resumes_at_the_front_despite_the_cap():
    sch = Scheduler(_config(max_num_seqs=2))
    a, b, c = _seq("a", [1]), _seq("b", [2]), _seq("c", [3])
    for s in (a, b, c):
        sch.add(s)
    sch.schedule()
    a.stoped = True
    sch.finish(a)  # a head start's head: its continuation keeps the stream
    cont = _seq("a", [1, 4, 5])
    cont.has_stream = True
    sch.add(cont, front=True)
    assert list(sch.waiting) == [cont, c] and sch.num_streams == 2
    assert sch.schedule() == ([cont], True)
    assert list(sch.waiting) == [c]


class _Buffers:
    """Scheduler callbacks: every sequence schedulable, a stream's queued audio from ``buf`` (seconds; absent: None)."""

    def __init__(self, buf):
        self.buf = buf

    def on_seq_added(self, seq): pass
    def on_seq_running(self, seq): pass
    def on_seq_waiting(self, seq): pass
    def on_seq_removed(self, seq, was_running): pass
    def can_schedule(self, running_ids, seq): return True
    def stream_buffer(self, seq): return self.buf.get(seq.seq_id)


def _guarded(guard_ms, buf, max_num_seqs=4):
    cfg = _config(max_num_seqs=max_num_seqs)
    cfg.model_config.turbo.stream_guard_ms = guard_ms
    return Scheduler(cfg, callbacks=_Buffers(buf))


def test_a_new_request_waits_while_a_playing_stream_is_short_of_audio():
    buf = {"a": 0.1}
    sch = _guarded(300, buf)
    a, b = _seq("a", [1]), _seq("b", [2])
    sch.add(a)
    assert sch.schedule() == ([a], True)  # nothing playing yet: admitted
    a.append_token(9)  # (its patch, as the engine's postprocess appends it)
    sch.add(b)
    assert sch.schedule() == ([a], False)  # a has 100 ms queued (< 300): decode it, b waits
    assert list(sch.waiting) == [b]
    a.append_token(9)
    buf["a"] = 0.5
    assert sch.schedule() == ([b], True)  # enough slack: b's prefill
    off = _guarded(0, {"a": 0.1})  # guard off: the original prefill-first order
    off.add(a2 := _seq("a", [1]))
    off.schedule()
    off.add(b2 := _seq("b", [2]))
    assert off.schedule() == ([b2], True)


def test_a_waiting_stream_with_less_audio_than_the_playing_ones_goes_first():
    buf = {"a": 0.2}
    sch = _guarded(300, buf)
    a, c = _seq("a", [1]), _seq("c", [3])
    sch.add(a)
    sch.schedule()
    a.append_token(9)
    cont = _seq("h", [4, 5])  # a head start's continuation: its stream is playing the head's audio
    cont.has_stream = True
    sch.add(c)
    sch.add(cont, front=True)
    buf["h"] = 0.25  # more than a's 200 ms: a decodes first
    assert sch.schedule() == ([a], False)
    a.append_token(9)
    buf["h"] = 0.05  # less: its prefill, and only it (c, a new request, still waits)
    assert sch.schedule() == ([cont], True)
    assert list(sch.waiting) == [c]


def test_new_request_at_the_front_does_not_block_a_resuming_one():
    sch = Scheduler(_config(max_num_seqs=2))
    r = _seq("r", [1])
    sch.add(r)
    sch.schedule()
    fresh, resuming = _seq("f", [2]), _seq("s", [3])
    resuming.has_stream = True  # e.g. preempted
    sch.add(fresh)
    sch.add(resuming)
    assert sch.schedule() == ([resuming], True)
    assert list(sch.waiting) == [fresh]


def test_stream_bookkeeping_through_preempt_cancel_and_finish():
    sch = Scheduler(_config(max_num_seqs=3))
    a, b, c = _seq("a", [1]), _seq("b", [2]), _seq("c", [3])
    for s in (a, b, c):
        sch.add(s)
    sch.schedule()
    sch.running.remove(b)
    sch.preempt(b)  # keeps its stream
    assert b.status == SequenceStatus.WAITING and b.has_stream and sch.num_streams == 3
    sch.cancel("b")
    assert sch.num_streams == 2
    sch.cancel("a")
    assert sch.num_streams == 1
    sch.finish(a)  # cancelled already: no-op, no double count
    sch.finish(c)
    assert sch.num_streams == 0 and sch.is_finished()
    w = _seq("w", [4])
    sch.add(w)
    sch.cancel("w")  # never ran: held no stream
    assert sch.num_streams == 0


class _Timed(_Buffers):
    """_Buffers with a time model: a step costs ``per_row`` per row and ``per_token`` per token (prefill), or
    ``decode_row`` per row (decode)."""

    patch_seconds = 0.16

    def __init__(self, buf, per_row=0.0, per_token=0.01, decode_row=0.05):
        super().__init__(buf)
        self.per_row, self.per_token, self.decode_row = per_row, per_token, decode_row

    def step_seconds(self, rows, tokens, prefill):
        return self.per_row * rows + self.per_token * tokens if prefill else self.decode_row * rows


def _timed(buf, max_num_seqs=4, **turbo):
    cfg = _config(max_num_seqs=max_num_seqs)
    for k, v in turbo.items():
        setattr(cfg.model_config.turbo, k, v)
    return Scheduler(cfg, callbacks=_Timed(buf))


def _playing(sch, *seqs):
    """seqs admitted, prefilled and holding their first patch (as the engine's postprocess appends it)."""
    for s in seqs:
        sch.add(s)
    assert sch.schedule() == (list(seqs), True)
    for s in seqs:
        s.append_token(9)


def _resuming(seq_id, tokens):
    """A stream waiting to resume (a head start's continuation): it keeps its stream."""
    seq = _seq(seq_id, tokens)
    seq.has_stream = True
    return seq


def test_a_resuming_step_decodes_the_streams_short_of_audio_too():
    buf = {"a": 0.05, "c": 0.5}
    sch = _timed(buf, mix_decode=True)  # 10 ms a token
    _playing(sch, a := _seq("a", [1]), c := _seq("c", [3]))
    sch.add(b := _seq("b", [2]))
    assert sch.schedule() == ([b], True) and sch.mixed == []  # new requests only (first audio): a plain prefill
    b.append_token(9)
    buf["b"], buf["h"] = 0.5, 0.0
    sch.add(h := _resuming("h", [4, 5]), front=True)
    assert sch.schedule() == ([h, a], True)  # predicted 50 ms (+ 50): a rides along, b and c sit it out
    assert sch.mixed == [a] and sch.last_step == (True, 2, 3)
    off = _timed({"a": 0.05, "h": 0.0})  # mix_decode off: the prefill step pauses a
    _playing(off, _seq("a", [1]))
    off.add(h2 := _resuming("h", [4, 5]), front=True)
    assert off.schedule() == ([h2], True) and off.mixed == []


def test_waiting_continuations_go_by_their_queued_audio():
    buf = {"x": 0.3, "y": 0.1, "z": 0.2}
    sch = _timed(buf)
    sch.add(n := _seq("n", [1]))
    for k in "xyz":
        sch.add(_resuming(k, [2]), front=True)
    assert [s.seq_id for s in sch.waiting] == ["y", "z", "x", "n"]


def test_the_step_cap_bounds_a_prefill_step_while_streams_play():
    sch = _timed({"a": 0.5}, step_cap_ms=50)  # 10 ms a token
    _playing(sch, a := _seq("a", [1]))
    b, c, d = _seq("b", [2, 2, 2]), _seq("c", [3, 3, 3]), _seq("d", [4] * 20)
    for s in (b, c):
        sch.add(s)
    assert sch.schedule() == ([b], True) and list(sch.waiting) == [c]  # b: 30 ms, b + c: 60 > 50
    b.append_token(9)
    assert sch.schedule() == ([a, b], False)  # c alone would fit, but a decode step comes between two pauses
    a.append_token(9), b.append_token(9)
    assert sch.schedule() == ([c], True)
    one = _timed({"a": 0.5}, step_cap_ms=50)
    _playing(one, _seq("a", [1]))
    one.add(d)
    assert one.schedule() == ([d], True)  # 200 ms alone: the first one goes when the least buffer covers it
    sch2 = _timed({"a": 0.15}, step_cap_ms=50)
    _playing(sch2, _seq("a", [1]))
    sch2.add(d2 := _seq("d", [4] * 20))
    assert sch2.schedule()[1] is False and list(sch2.waiting) == [d2]  # 200 ms > 150 ms of buffer: decode first
    idle = _timed({}, step_cap_ms=50)  # nothing plays: no cap
    for s in (e := _seq("e", [5] * 3), f := _seq("f", [6] * 3)):
        idle.add(s)
    assert idle.schedule() == ([e, f], True)


def test_the_step_cap_counts_the_mixed_rows():
    sch = _timed({"a": 0.05, "b": 0.05, "c": 0.0, "d": 0.0}, step_cap_ms=50, mix_decode=True)
    _playing(sch, a := _seq("a", [1]), b := _seq("b", [2]))
    c, d = _resuming("c", [3, 3]), _resuming("d", [4, 4])
    sch.add(c, front=True), sch.add(d, front=True)
    seqs, prefill = sch.schedule()  # a and b ride along: 20 ms, c 20 ms, d would make 60
    assert prefill and seqs == [c, a, b] and list(sch.waiting) == [d]


def test_a_new_request_waits_while_the_streams_would_fall_behind_real_time():
    sch = _timed({}, realtime_margin_ms=9)  # a decode step: 50 ms a row: 3 rows = 150 ms < 160 - 9
    _playing(sch, *(seqs := [_seq(f"s{i}", [i + 1]) for i in range(3)]))
    late = _seq("late", [9])
    sch.add(late)
    assert sch.schedule() == (seqs, False) and list(sch.waiting) == [late]  # 4 rows: 200 ms > 151
    cont = _seq("h", [4, 5])  # a stream already playing (a continuation) is not held back
    cont.has_stream = True
    sch.add(cont, front=True)
    for s in seqs:
        s.append_token(9)
    assert sch.schedule() == ([cont], True) and list(sch.waiting) == [late]


# --- 4. cancels during a step, VAE stream slots ---------------------------------------------------------------------


class _FakeRunner:
    """model_runner.call: run() calls the overlap hook (commands mid-step) and then takes a VAE slot per sequence,
    as VoxCPM2Runner._run_with_inputs and SlotStreamingVAEDecoder do."""

    def __init__(self, max_streams, make_output):
        self.max_streams = max_streams
        self.make_output = make_output
        self.slots: set[str] = set()
        self.overlap_hook = None

    def call(self, name, *args):
        if name == "run":
            tasks, is_prefill = args[:2]
            if self.overlap_hook is not None:
                self.overlap_hook()
            for t in tasks:
                if t.seq_id not in self.slots:
                    if len(self.slots) >= self.max_streams:
                        raise RuntimeError(f"more than {self.max_streams} concurrent VAE streams")
                    self.slots.add(t.seq_id)
            return [self.make_output(t, is_prefill) for t in tasks]
        if name == "release_sequence_state":
            self.slots.discard(args[0])
        return None


class _Engine(LLMEngineBase):
    def __init__(self, max_num_seqs=4, stop_at=3):
        self.stop_at = stop_at
        self.model_runner = _FakeRunner(max_num_seqs, lambda t, p: {})
        self.lora_manager = LoRAManager(max_loras=0)
        self.scheduler = Scheduler(_config(max_num_seqs=max_num_seqs), callbacks=self)
        self.ps = []

    def preprocess_seq(self, seq, is_prefill):
        return RunnerTask(seq.block_table, len(seq), seq.num_cached_tokens if is_prefill else len(seq) - 1,
                          seq.block_size, None, seq_id=seq.seq_id)

    def postprocess_seq(self, seq, outputs, is_prefill):
        seq.append_token(100 + len(seq))
        seq.stoped = seq.num_completion_tokens >= self.stop_at


def _run(engine, max_steps=50):
    for _ in range(max_steps):
        if engine.is_finished():
            return
        engine.step()
    raise AssertionError("engine did not finish")


def test_only_a_prefill_step_of_new_streams_is_a_first_patch_step():
    """The runner's first_patch (the first patch's LocDiT settings, TurboConfig.dit_first_t_span): a prefill step of
    new streams only; a step that resumes a stream (a preempted one, a continuation) or decodes running ones is not."""
    eng = _Engine(stop_at=4)
    firsts = []
    call = eng.model_runner.call

    def record(name, *args):
        if name == "run":
            firsts.append((args[1], args[2]))
        return call(name, *args)

    eng.model_runner.call = record
    for i in range(2):
        eng.add_sequence(_seq(f"s{i}", [i + 1]))
    eng.step()  # the two new streams' prefill
    eng.step()  # a decode step
    resumed = eng.scheduler.running.pop()
    eng.scheduler.preempt(resumed)
    eng.step()  # the preempted stream resumes: a prefill step, not a first patch
    assert firsts == [(True, True), (False, False), (True, False)]


def test_cancel_during_a_step_runs_after_it():
    eng = _Engine(stop_at=2)
    for i in range(2):
        eng.add_sequence(_seq(f"s{i}", [i + 1]))
    eng.step()  # prefill: both stream
    eng.model_runner.overlap_hook = lambda: eng.cancel_sequence("s0")  # arrives while s0 decodes its last patch
    seqs = eng.step()  # s0 stops in this very step
    eng.model_runner.overlap_hook = None
    assert {s.seq_id for s in seqs} == {"s0", "s1"}
    assert eng.model_runner.slots == set() and eng.scheduler.num_streams == 0 and eng.is_finished()


def test_cancel_during_a_step_releases_the_stream_slot():
    eng = _Engine(stop_at=10)
    eng.add_sequence(_seq("s0", [1]))
    eng.add_sequence(_seq("s1", [2]))
    eng.step()
    eng.model_runner.overlap_hook = lambda: eng.cancel_sequence("s0")
    eng.step()  # decodes s0 once more, then cancels it
    eng.model_runner.overlap_hook = None
    assert eng.model_runner.slots == {"s1"} and eng.scheduler.num_streams == 1
    _run(eng)
    assert eng.model_runner.slots == set()


def test_more_requests_than_vae_streams_wait_instead_of_failing():
    eng = _Engine(max_num_seqs=2, stop_at=3)
    for i in range(5):
        eng.add_sequence(_seq(f"s{i}", [i + 1]))
    _run(eng)
    assert eng.model_runner.slots == set() and eng.scheduler.num_streams == 0


class _MixedEngine(_Engine):
    """_Engine with mixed steps: every stream that has a patch is playing; records how each step's sequences went."""

    def __init__(self):
        super().__init__(max_num_seqs=4, stop_at=3)
        self.scheduler.callbacks = None
        cfg = _config(max_num_seqs=4)
        cfg.model_config.turbo.mix_decode = True
        self.scheduler = Scheduler(cfg, callbacks=self)
        self.log = []

    def stream_buffer(self, seq):
        return 0.5 if seq.num_completion_tokens else None

    def preprocess_seq(self, seq, is_prefill):
        self.log.append(("pre", seq.seq_id, is_prefill))
        return super().preprocess_seq(seq, is_prefill)

    def postprocess_seq(self, seq, outputs, is_prefill):
        self.log.append(("post", seq.seq_id, is_prefill))
        super().postprocess_seq(seq, outputs, is_prefill)

    def after_prefill(self, seqs):
        self.log.append(("after", tuple(s.seq_id for s in seqs)))


def test_a_mixed_step_gives_its_running_sequences_their_decode_inputs():
    eng = _MixedEngine()
    eng.add_sequence(_seq("a", [1]))
    eng.step()  # a's prefill
    eng.add_sequence(_resuming("b", [2]), front=True)  # (a continuation)
    eng.log.clear()
    seqs = eng.step()  # b's prefill, a's next patch
    assert {s.seq_id for s in seqs} == {"a", "b"}
    assert set(eng.log) == {("pre", "b", True), ("pre", "a", False), ("post", "b", True), ("post", "a", False),
                            ("after", ("b",))}
    a = next(s for s in seqs if s.seq_id == "a")
    assert a.num_completion_tokens == 2 and eng.model_runner.slots == {"a", "b"}
    _run(eng)
    assert eng.model_runner.slots == set() and eng.scheduler.num_streams == 0


# --- 5. head start --------------------------------------------------------------------------------------------------

FEAT, PATCH = 4, 2


def _voxcpm2_engine(max_model_len=64, max_num_seqs=4):
    from nanovllm_voxcpm.models.voxcpm2.engine import VoxCPM2Engine

    eng = object.__new__(VoxCPM2Engine)
    eng.n_decode_pad_frames, eng.feat_dim, eng.patch_size = 12, FEAT, PATCH
    eng.audio_start_token, eng.ref_audio_start_token, eng.ref_audio_end_token = 101, 103, 104
    eng.block_size, eng.max_model_len = 256, max_model_len
    eng.ref_prefix_blocks, eng.voices = OrderedDict(), {}
    eng.head_tokens, eng.head_patches, eng.head_min_rest = 4, 3, 2
    eng.ps = []
    eng.lora_manager = LoRAManager(max_loras=0)
    eng.model_runner = _FakeRunner(max_num_seqs, lambda t, p: {
        "latents": np.full((PATCH, FEAT), 0.5, np.float32), "stop_flag": 0,
        "waveforms": np.ones(8, np.float32), "patch_silent": False})
    eng.scheduler = Scheduler(_config(max_num_seqs=max_num_seqs, blocks=64, block_size=256), callbacks=eng)
    # tokenizer stand-ins: one id per word; the head is the first 3 words
    eng._text_ids = lambda prompt_text, target_text: [7] * len((prompt_text + target_text).split())
    eng._split_head = lambda prompt_text, target_text, voice, clone_mode: (
        [7] * 3, [7] * len((prompt_text + target_text).split()))
    return eng


def _words(n):
    return " ".join(["w"] * n)


def test_head_start_checks_the_whole_request_length():
    eng = _voxcpm2_engine(max_model_len=64)
    with pytest.raises(ValueError, match="max_model_len"):
        eng.add_request("r", _words(30), max_generate_length=40)  # 31 + 40 > 64; the head alone would fit
    assert not eng.scheduler.waiting
    eng.register_voice("v", np.zeros((PATCH * 20, FEAT), np.float32))
    with pytest.raises(ValueError, match="max_model_len"):  # a registered voice's reference counts too
        eng.add_request("r", _words(20), max_generate_length=25, voice="v")  # 21 + 22 + 25 > 64 (its head: 29)
    assert not eng.scheduler.waiting
    head = eng.add_request("r", _words(20), max_generate_length=20)
    assert head.custom_payload.chain is not None and list(eng.scheduler.waiting) == [head]


def test_head_start_for_at_most_head_start_limit_streams_at_once():
    eng = _voxcpm2_engine(max_model_len=128)
    eng.head_start_limit, eng._head_phase = 2, {}
    h1, h2, w = (eng.add_request(f"r{i}", _words(20), max_generate_length=20) for i in range(3))
    assert h1.custom_payload.chain is not None and h2.custom_payload.chain is not None
    assert w.custom_payload.chain is None and set(eng._head_phase) == {"r0", "r1"}  # r2: prefilled whole
    assert eng._head_phase["r0"] == 20 + 1 + 3  # its continuation's tokens: text, audio start, the head's patches
    for _ in range(3):  # the heads' prefill and patches; they end and queue their continuations
        eng.step()
    assert set(eng._head_phase) == {"r0", "r1"}  # continuations queued, not prefilled yet
    eng.step()  # their prefill
    assert not eng._head_phase
    assert eng.add_request("r3", _words(20), max_generate_length=20).custom_payload.chain is not None
    eng.cancel_sequence("r3")
    assert not eng._head_phase


def test_head_patches_cover_the_continuation_prefill_ahead():
    from nanovllm_voxcpm.engine.step_cost import head_patches_for

    plan = lambda backlog: head_patches_for(backlog, 0.003, 0.05, 0.16, 3, 5, 0.15, 0.02)  # noqa: E731
    assert plan(0.0) == (3, True)  # 480 ms - 150 >= 2 decode steps + its own continuation
    assert plan(0.3) == (4, True)
    assert plan(0.6) == (5, False)  # none covers it: the most


def test_heads_under_load_are_longer_and_continue_after_their_patches():
    eng = _voxcpm2_engine(max_model_len=256)
    eng.head_patches_max, eng.head_margin, eng.patch_seconds, eng._head_phase = 5, 0.15, 0.16, {}
    eng.step_cost = SimpleNamespace(prefill=SimpleNamespace(coef=(0.02, 0.0, 0.01)),
                                    seconds=lambda rows, tokens, prefill: 0.0,  # 10 ms a prefilled token
                                    observe=lambda *a: None)
    first = eng.add_request("r0", _words(20), max_generate_length=40)  # its continuation: 24 tokens, 0.24 s
    second = eng.add_request("r1", _words(20), max_generate_length=40)  # after r0's: 0.51 s
    assert first.custom_payload.max_generate_length == 3 and second.custom_payload.max_generate_length == 5
    assert eng._head_phase == {"r0": 24, "r1": 26}
    cont = _steps_until_chained(eng, "r1")
    assert len(cont.custom_payload.generated_waveforms) == 5
    assert eng.scheduler.waiting[0].custom_payload.max_generate_length <= 40 - 5  # the rest of the request


def _loaded(per_token):
    eng = _voxcpm2_engine(max_model_len=256, max_num_seqs=8)
    eng.head_patches_max, eng.head_margin, eng.patch_seconds, eng._head_phase = 5, 0.15, 0.16, {}
    eng.step_cost = SimpleNamespace(prefill=SimpleNamespace(coef=(0.0, 0.0, per_token)), observe=lambda *a: None,
                                    seconds=lambda rows, tokens, prefill: 0.0)
    return eng


def test_heads_are_lengthened_when_the_burst_grows():
    eng = _loaded(0.005)  # 5 ms a prefilled token: a continuation of 24 tokens, 0.12 s
    first = eng.add_request("r0", _words(20), max_generate_length=40)
    assert first.custom_payload.max_generate_length == 3
    eng.step()  # its prefill
    eng.step_cost.prefill.coef = (0.0, 0.0, 0.012)  # the steps turn out slower (a bigger burst, a slower GPU)
    eng.add_request("r1", _words(20), max_generate_length=40)
    eng.step()  # re-planned before it: r0's own continuation (0.29 s) no longer fits in 3 patches - 150 ms
    assert first.custom_payload.max_generate_length == 4 and eng._head_phase["r0"] == 25


def test_a_head_waits_while_even_the_longest_would_not_last():
    eng = _loaded(0.01)  # 10 ms a token: each continuation 0.24-0.26 s
    eng.head_hold = True
    heads = [eng.add_request(f"r{i}", _words(20), max_generate_length=40) for i in range(3)]
    assert [h.custom_payload.max_generate_length for h in heads] == [3, 5, 5]
    assert eng.head_fits(heads[0]) and eng.head_fits(heads[1]) and not eng.head_fits(heads[2])
    seqs = eng.step()
    assert {s.seq_id for s in seqs} == {"r0", "r1"} and list(eng.scheduler.waiting) == [heads[2]]
    for _ in range(4):  # r0 ends after 3 patches, its continuation is prefilled: r2's wait is shorter
        eng.step()
    assert "r0" not in eng._head_phase and heads[2].status != SequenceStatus.WAITING
    assert eng._head_sum == sum(eng._head_phase.values())
    free = _loaded(0.01)  # head_hold off: no waiting, the third head is as long as it gets
    free_heads = [free.add_request(f"r{i}", _words(20), max_generate_length=40) for i in range(3)]
    assert free.head_fits(free_heads[2]) and {s.seq_id for s in free.step()} == {"r0", "r1", "r2"}


def _steps_until_chained(eng, seq_id):
    for _ in range(10):
        for s in eng.step():
            if s.seq_id == seq_id and s.custom_payload.chained:
                return s
    raise AssertionError("the head was not continued")


def test_continuation_survives_unregister_voice_and_goes_first():
    eng = _voxcpm2_engine(max_model_len=128, max_num_seqs=1)
    eng.register_voice("v", np.zeros((PATCH * 5, FEAT), np.float32))
    head = eng.add_request("r1", _words(20), max_generate_length=20, voice="v")
    eng.add_request("r2", _words(2), max_generate_length=2, head_start=False)  # a new request behind it
    eng.step()  # the head's prefill (r2 waits: one stream slot)
    eng.unregister_voice("v")  # DELETE /v1/audio/voices while the head runs
    _steps_until_chained(eng, "r1")
    cont = eng.scheduler.waiting[0]
    assert cont.seq_id == "r1" and cont is not head and cont.has_stream
    assert [s.seq_id for s in eng.scheduler.waiting] == ["r1", "r2"]
    assert len(cont) == 1 + 5 + 1 + 20 + 1 + 3  # [ref] voice [/ref] + text + audio start + the head's 3 patches
    assert eng.model_runner.slots == {"r1"} and eng.scheduler.num_streams == 1  # the stream carries on
    assert cont.custom_payload.seed_step == head.custom_payload.seed_step
    for _ in range(100):  # everything ends; r2 runs after the continuation
        if eng.is_finished():
            break
        eng.step()
    assert eng.is_finished() and eng.model_runner.slots == set()


def _streamed(eng, seq_id, steps=40):
    """Per generated patch of ``seq_id``'s stream (head and continuation): True = streamed (the server skips muted)."""
    out = []
    for _ in range(steps):
        if eng.is_finished():
            break
        out += [not s.custom_payload.muted for s in eng.step() if s.seq_id == seq_id]
    return out


def _silent_first(n):
    """A runner output whose patch is silent for the first n patches of each stream, then with sound; a pause
    (silent patches 9 and 10) later in the sentence."""
    count = {}

    def make(t, is_prefill):
        k = count[t.seq_id] = count.get(t.seq_id, 0) + 1
        return {"latents": np.full((PATCH, FEAT), 0.5, np.float32), "stop_flag": 0,
                "waveforms": np.ones(8, np.float32), "patch_silent": k <= n or k in (9, 10)}
    return make


@pytest.mark.parametrize("cap, silent, muted", [
    (2, 5, [3, 4, 5]),  # a silent head (patches 1-3) and a continuation that pauses again: 3 of them not streamed
    (2, 2, []),  # a natural pause of 2 patches stays
    (-1, 5, []),  # cap off: as generated
])
def test_leading_silence_past_the_cap_is_not_streamed(cap, silent, muted):
    eng = _voxcpm2_engine(max_model_len=128)
    eng.lead_silence_patches = cap
    eng.model_runner.make_output = _silent_first(silent)
    eng.add_request("r", _words(20), max_generate_length=14)
    streamed = _streamed(eng, "r")
    assert len(streamed) == 14  # every patch was generated (the head's 3 and the continuation's 11)
    assert [k for k, on in enumerate(streamed, 1) if not on] == muted  # the later pause (9, 10) always streams


def test_leading_silence_without_head_start():
    eng = _voxcpm2_engine(max_model_len=128)
    eng.lead_silence_patches = 1
    eng.model_runner.make_output = _silent_first(3)
    eng.add_request("r", _words(20), max_generate_length=6, head_start=False)
    assert _streamed(eng, "r") == [True, False, False, True, True, True]


def test_cancel_while_the_head_ends_also_cancels_its_continuation():
    eng = _voxcpm2_engine(max_model_len=128)
    eng.add_request("r1", _words(20), max_generate_length=20)
    eng.step()
    eng.step()
    eng.model_runner.overlap_hook = lambda: eng.cancel_sequence("r1")
    seqs = eng.step()  # the head's third (last) patch: it stops and is continued in this step
    eng.model_runner.overlap_hook = None
    assert [s.seq_id for s in seqs] == ["r1"] and seqs[0].stoped
    assert eng.is_finished() and eng.model_runner.slots == set() and eng.scheduler.num_streams == 0


# --- 6. streamed audio overwritten in the shared ring ---------------------------------------------------------------


def test_overwritten_audio_fails_the_stream_and_cancels_it():
    from nanovllm_voxcpm.models.voxcpm2.server import AsyncVoxCPM2Server

    async def scenario():
        srv = object.__new__(AsyncVoxCPM2Server)
        srv.op_table, srv._pending_cmds, srv.stream_table = {}, [], {}
        sent = []
        srv.queue_in = SimpleNamespace(put_nowait=sent.append)

        class Ring:
            def read(self, *args):
                raise RuntimeError("streamed audio was overwritten before it was read (shared ring too small)")

        srv._ring = Ring()

        async def until(n):
            for _ in range(1000):
                if len(sent) >= n:
                    return True
                await asyncio.sleep(0)
            return False

        async def engine():  # answers commands; the request's first chunk arrives overwritten
            assert await until(1)
            add = sent[0]
            srv._handle({"type": "response", "id": add["id"], "data": None})
            seq_id = add["args"][0]
            srv._handle(srv._convert({"type": "batch", "items": [
                {"type": "stream_shm", "id": seq_id, "off": 0, "nb": 4, "end": 4, "dtype": "<i2", "n": 2},
                {"type": "stream", "id": seq_id, "data": None}]}))  # the step's end of the stream
            if await until(2):
                srv._handle({"type": "response", "id": sent[1]["id"], "data": None})

        task = asyncio.ensure_future(engine())
        chunks = []
        with pytest.raises(RuntimeError, match="overwritten"):
            async for chunk in srv.generate("text"):
                chunks.append(chunk)
        await task
        return chunks, sent

    chunks, sent = asyncio.run(scenario())
    assert chunks == []
    assert sent[1]["type"] == "cancel" and sent[1]["args"] == (sent[0]["args"][0],)


# --- 7. small ones --------------------------------------------------------------------------------------------------


def test_exit_is_idempotent():
    eng = object.__new__(LLMEngineBase)
    calls = []
    eng.model_runner = SimpleNamespace(call=lambda name, *a: calls.append(name))
    eng.ps = []
    eng.exit()
    eng.exit()  # atexit after an explicit exit
    assert calls == ["exit"]


def test_register_lora_rolls_back_the_runner_when_the_manager_refuses():
    eng = object.__new__(LLMEngineBase)
    calls = []
    eng.model_runner = SimpleNamespace(call=lambda name, *a: calls.append(name))
    eng.lora_manager = LoRAManager(max_loras=0)  # refuses every registration
    payload = SimpleNamespace(rank=8, alpha=16.0, modules={})
    with pytest.raises(RuntimeError):
        eng.register_lora("x", payload)
    assert calls == ["validate_lora_payload", "register_lora", "unregister_lora"]


def test_cuda_ext_failure_is_cached_and_reported_without_a_message(monkeypatch, capsys):
    from voxcpm_kernels import cuda_ext

    attempts = []

    def failing_load(name, sweep=False):
        attempts.append(name)
        raise RuntimeError("")

    monkeypatch.setattr(cuda_ext, "load", failing_load)
    cuda_ext.available.cache_clear()
    try:
        assert cuda_ext.available("nope") is False
        assert cuda_ext.available("nope") is False
    finally:
        cuda_ext.available.cache_clear()
    assert attempts == ["nope"]
    assert "RuntimeError" in capsys.readouterr().out


def test_step_cost_starts_from_its_prior_and_follows_the_measured_steps():
    from nanovllm_voxcpm.engine.step_cost import StepCost

    cost = StepCost()
    assert 0.08 < cost.seconds(200, 200, False) < 0.1  # the A10 prior: 15.6 + 0.365 * 200 ms
    for rows in (8, 64, 128, 200, 256) * 40:  # a GPU twice as slow
        cost.observe(rows, rows, False, 2 * (0.0156 + 0.000365 * rows))
    assert abs(cost.seconds(200, 200, False) - 2 * 0.0886) < 0.01
    for rows, tokens in ((10, 1000), (50, 4000), (200, 1600), (30, 300)) * 40:
        cost.observe(rows, tokens, True, 0.01 + 0.0001 * rows + 0.00005 * tokens)
    assert abs(cost.seconds(100, 2000, True) - 0.12) < 0.01
    for tokens in (8, 40, 200) * 100:  # small prefill steps (first audio: launch-bound) have a fit of their own
        cost.observe(4, tokens, True, 0.035)
    assert abs(cost.seconds(100, 2000, True) - 0.12) < 0.01 and abs(cost.seconds(4, 40, True) - 0.035) < 0.003
    cost.observe(0, 0, True, 1.0)  # an empty step, a host stall: ignored
    cost.observe(5, 5, False, 30.0)
    assert abs(cost.seconds(200, 200, False) - 2 * 0.0886) < 0.01
    for _ in range(5):  # a compile or a pause: at most twice the prediction counts
        cost.observe(50, 2000, True, 2.0)
    assert abs(cost.seconds(100, 2000, True) - 0.12) < 0.03 and min(cost.prefill.coef) >= 0.0
    fresh = StepCost()  # steps whose tokens are in proportion to their rows (one head each): the fit stays solvable
    for rows in (1, 2, 4, 8, 16, 32, 64) * 200:
        fresh.observe(rows, 8 * rows, True, 0.026 + 0.0003 * rows)
    assert abs(fresh.seconds(16, 128, True) - (0.026 + 0.0003 * 16)) < 0.005


def test_turbo_config_rejects_unknown_keys_and_accepts_the_repos_own():
    from pydantic import ValidationError

    from nanovllm_voxcpm.dialect_tts import turbo_config
    from nanovllm_voxcpm.models.voxcpm2.config import TurboConfig

    with pytest.raises(ValidationError):
        TurboConfig.model_validate_json('{"head_token": 8}')
    for cfg in (turbo_config(None, 8), turbo_config(None, 0, "light"), turbo_config(None, 8, None, w8a8=False),
                {**turbo_config(None, 8), "pcm16_output": True}):
        TurboConfig.model_validate(cfg)
