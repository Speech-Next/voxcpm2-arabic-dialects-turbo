"""CPU tests of #10 (first audio frame): the NANOVLLM_FIRST_AUDIO_MS -> phase mapping and the one-buffer VAE state."""

from types import SimpleNamespace

import pytest
import torch

from nanovllm_voxcpm.models.voxcpm2.runner import first_audio_phases
from voxcpm_kernels.vae_stream import flatten_states


@pytest.mark.parametrize("ms, phases", [(20, 4), (5, 1), (35, 7), (10, 2), (0, 0), (-5, 0), (40, 0), (80, 0)])
def test_first_audio_phases(ms, phases):
    assert first_audio_phases(ms, 40.0, 8) == phases


@pytest.mark.parametrize("ms", [12, 2.5, 39])
def test_first_audio_phases_rejects_ms_between_phases(ms):
    with pytest.raises(ValueError, match="multiple of 5 ms"):
        first_audio_phases(ms, 40.0, 8)


def _holders(slots=5, dtype=torch.float16):
    g = torch.Generator().manual_seed(0)
    shapes = [(3, 6), (7, 1), (2, 54), (1, 3)]
    return [SimpleNamespace(state=torch.randn(slots, *s, generator=g).to(dtype)) for s in shapes]


def test_flatten_states_keeps_values_and_shares_one_buffer():
    holders = _holders()
    before = [h.state.clone() for h in holders]
    flat = flatten_states(holders)
    assert flat is not None and flat.shape[0] == 5
    for h, b in zip(holders, before):
        assert torch.equal(h.state, b)  # same values and shape
        assert h.state.stride(0) == flat.shape[1]  # the slot stride the kernels read
        assert h.state[0].is_contiguous()
        assert h.state.data_ptr() % 16 == 0  # 16-byte aligned in every slot row
        assert h.state.untyped_storage().data_ptr() == flat.untyped_storage().data_ptr()
    rows = torch.tensor([1, 3])
    flat.index_fill_(0, rows, 0)  # zeroing a fresh slot: one fill for every layer
    for h, b in zip(holders, before):
        assert not h.state[rows].any()
        assert torch.equal(h.state[0], b[0]) and torch.equal(h.state[2], b[2]) and torch.equal(h.state[4], b[4])
    h0 = holders[2]
    h0.state.index_copy_(0, torch.tensor([4]), torch.ones(1, 2, 54, dtype=torch.float16))  # a layer's own write
    assert torch.equal(h0.state[4], torch.ones(2, 54, dtype=torch.float16))
    assert torch.equal(holders[1].state[4], before[1][4]) and torch.equal(holders[3].state[4], before[3][4])


def test_flatten_states_leaves_mixed_states_alone():
    holders = _holders()
    holders[1].state = holders[1].state.float()
    states = [h.state for h in holders]
    assert flatten_states(holders) is None
    assert all(h.state is s for h, s in zip(holders, states))
    assert flatten_states([]) is None
