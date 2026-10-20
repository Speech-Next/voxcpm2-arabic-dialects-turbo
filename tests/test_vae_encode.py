"""AudioVAEV2.encode in pieces (#47): the same latents as one pass, GPU memory of one piece."""
import pytest
import torch

from nanovllm_voxcpm.layers import audio_vae_v2
from nanovllm_voxcpm.layers.audio_vae_v2 import AudioVAEConfigV2, AudioVAEV2


@pytest.fixture
def tiny_cpu(monkeypatch):
    """The snake activation uncompiled (on the CPU torch.compile spends a minute building it) and 4 threads (all of a
    96-core machine's spend 30x longer on such small convs)."""
    monkeypatch.setattr(audio_vae_v2, "snake", audio_vae_v2.snake.__wrapped__)
    threads = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(threads)


def _tiny_vae() -> AudioVAEV2:
    """The model's encoder strides with few channels (the receptive field is the model's), a small decoder."""
    torch.manual_seed(0)
    cfg = AudioVAEConfigV2(encoder_dim=4, latent_dim=4, decoder_dim=16, decoder_rates=[2, 2], sr_bin_boundaries=None)
    return AudioVAEV2(config=cfg).double().eval()


def test_encoder_context_is_the_receptive_field(tiny_cpu):
    """A sample changes the frames from its own to the last one whose field (encoder_context samples before the
    frame's own) still holds it, and no other."""
    vae = _tiny_vae()
    hop, ctx = vae.encoder_chunk_size, vae.encoder_context
    assert (hop, ctx) == (640, 9272)
    x = torch.randn(1, hop * 40, dtype=torch.float64)
    base = vae.encode(x, vae.sample_rate)
    for p in (hop * 3 + 17, hop * 10):
        y = x.clone()
        y[0, p] += 1.0
        changed = (vae.encode(y, vae.sample_rate) != base).any(dim=1)[0].nonzero().flatten().tolist()
        assert changed == list(range(p // hop, (p + ctx) // hop + 1))


def test_pieces_give_the_one_pass_latents(tiny_cpu):
    vae = _tiny_vae()
    x = torch.randn(1, 640 * 97 + 123, dtype=torch.float64)  # (not whole pieces, nor whole frames)
    vae.encode_chunk_seconds = 1e9
    one = vae.encode(x, vae.sample_rate)
    for seconds in (0.2, 0.6, 1.0):  # pieces shorter and longer than the context (0.6 s)
        vae.encode_chunk_seconds = seconds
        # (fp64 rounding only: the CPU's conv sums in an order that follows the input's length; a missing context
        # would be off by ~1e-2)
        torch.testing.assert_close(vae.encode(x, vae.sample_rate), one, rtol=0, atol=1e-12, msg=str(seconds))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_long_clip_on_gpu_same_latents_bounded_memory():
    """The model's VAE size: a 60 s clip takes the memory of one 10 s piece, and with TF32 off (cuDNN's
    algorithms then agree bit for bit) its latents are those of one pass."""
    torch.manual_seed(0)
    with torch.device("cuda"):
        vae = AudioVAEV2(config=AudioVAEConfigV2()).eval()
    x = torch.randn(1, 16000 * 60, device="cuda") * 0.1
    tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    try:
        peaks, mus = [], []
        for seconds in (1e9, 10.0):
            vae.encode_chunk_seconds = seconds
            vae.encode(x, 16000)  # (warm: cuDNN, the compiled snake)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            mus.append(vae.encode(x, 16000))
            torch.cuda.synchronize()
            peaks.append(torch.cuda.max_memory_allocated() - base)
    finally:
        torch.backends.cudnn.allow_tf32 = tf32
    assert torch.equal(mus[0], mus[1])
    assert peaks[1] < peaks[0] / 4  # one pass: ~2.3 GiB; 10 s pieces: ~0.5 GiB
