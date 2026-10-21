"""The kernels against plain PyTorch references (#47: they had none): the int8 GEMM epilogues (bias, residual,
SwiGLU + row quantization, RoPE + addend, the adapters' segmented GEMMs), the attention kernels (paged prefill over a
cached prefix, short causal and non-causal, the decode path's KV store), and the streaming VAE decoder's kernels
(cuda/resunit_fused.cu, cuda/head_fused.cu, the fused snake + transposed conv, the slot-streaming decoder as a whole)
against the modules' own forward. Shapes are the engine's (2048-wide LM, head_dim 128, 2 KV heads, 256-token pages,
the VAE's channel widths and strides) at small batch sizes. All but the first test need CUDA."""

import copy

import pytest
import torch
import torch.nn.functional as F

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _tiny_layer():
    from nanovllm_voxcpm.model.config import MiniCPM4Config, RopeScalingConfig
    from nanovllm_voxcpm.model.model import Cpm4DecoderLayer

    rope = RopeScalingConfig(type="longrope", long_factor=[1.0] * 64, short_factor=[1.0] * 64,
                             original_max_position_embeddings=64)
    cfg = MiniCPM4Config(bos_token_id=1, eos_token_id=2, hidden_size=256, intermediate_size=512,
                         max_position_embeddings=64, num_attention_heads=4, num_hidden_layers=1, num_key_value_heads=2,
                         rms_norm_eps=1e-5, rope_scaling=rope, vocab_size=16, scale_emb=1.0, dim_model_base=256,
                         scale_depth=1.0, rope_theta=10000.0, kv_channels=128)
    return Cpm4DecoderLayer(cfg)


def test_int8_layer_refuses_an_attention_bias_its_kernels_would_drop():
    """o_proj's int8 GEMM fuses the residual and adds no bias (the adapters' segmented one dropped it): such a layer
    is refused when it is quantized, not run with a wrong output. A q/k/v bias without adapters is applied."""
    from nanovllm_voxcpm.kernels.fused_layer import FusedCpm4Layer

    layer = _tiny_layer()
    assert FusedCpm4Layer(layer, True, 512, "plain").q8 is not None
    layer.self_attn.qkv_proj.bias = torch.nn.Parameter(torch.randn(layer.self_attn.qkv_proj.weight.shape[0]))
    assert FusedCpm4Layer(layer, True, 512, "qkv bias").q8["qkv"].bias is not None  # qkv_rope_gemm adds it
    layer.self_attn.o_proj.bias = torch.nn.Parameter(torch.randn(256))
    with pytest.raises(NotImplementedError, match="attention bias"):
        FusedCpm4Layer(layer, True, 512, "o bias")
    assert FusedCpm4Layer(layer, False, 512, "bf16").q8 is None  # the bf16 path runs the modules (with the bias)


# --- int8 GEMM epilogues ------------------------------------------------------------------------------------------

def _deq(q, scale):
    return q.float() * scale.float()[:, None]


def _close(out, ref, rel, what=""):
    err = (out.float() - ref).abs().max().item()
    peak = ref.abs().max().item()
    assert err <= rel * peak, f"{what}: max error {err:.3g} > {rel} x {peak:.3g}"
    return err / peak


def _rope_tables(max_pos: int, head_dim: int = 128):
    inv = 1.0 / 10000 ** (torch.arange(0, head_dim, 2, device="cuda", dtype=torch.float32) / head_dim)
    ang = torch.arange(max_pos, device="cuda", dtype=torch.float32)[:, None] * inv[None, :]
    return torch.cat([ang.cos(), ang.cos()], -1).contiguous(), torch.cat([ang.sin(), ang.sin()], -1).contiguous()


def _rope_ref(y, pos, cos, sin, n_rope, head_dim=128):
    """Rotate-half RoPE on the first n_rope heads of y [M, heads * head_dim] (fp32)."""
    y = y.clone()
    half = head_dim // 2
    c, s = cos[pos][:, :half], sin[pos][:, :half]
    for h in range(n_rope):
        x1, x2 = y[:, h * head_dim:h * head_dim + half].clone(), y[:, h * head_dim + half:(h + 1) * head_dim].clone()
        y[:, h * head_dim:h * head_dim + half] = x1 * c - x2 * s
        y[:, h * head_dim + half:(h + 1) * head_dim] = x2 * c + x1 * s
    return y


@CUDA
@pytest.mark.parametrize("M", [1, 7, 64, 200, 513])
def test_int8_gemm_epilogues_match_the_dequantized_product(M):
    """w8a8_gemm (+ bias), the o / down GEMM with the residual in its epilogue, gate_up with SwiGLU and the down
    GEMM's row quantization, and qkv with bias, the LoRA addend and RoPE on the q/k heads: each against the fp32
    product of the dequantized int8 operands (the int32 accumulation is exact; what differs is the output's
    rounding)."""
    from nanovllm_voxcpm.kernels.fused_layer import _gemm, qkv_rope_gemm, rope_weights, swiglu_gemm_quant, swiglu_weights
    from nanovllm_voxcpm.kernels.int8_linear import W8A8Linear, quantize_rows, w8a8_gemm

    torch.manual_seed(M)
    H, I, heads, kv = 2048, 6144, 16, 2
    x = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)
    q, s = quantize_rows(x)
    assert _close(_deq(q, s), x.float(), 0.01, "quantize_rows") < 0.01
    xd = _deq(q, s)

    n_qkv = (heads + 2 * kv) * 128
    w_qkv, b_qkv = torch.randn(n_qkv, H, device="cuda") / H**0.5, torch.randn(n_qkv, device="cuda")
    qkv = W8A8Linear(w_qkv, b_qkv)
    ref = xd @ _deq(qkv.w_q, qkv.w_scale).T + qkv.bias
    _close(w8a8_gemm(q, s, qkv.w_q, qkv.w_scale, qkv.bias, torch.bfloat16), ref, 0.01, "w8a8_gemm + bias")

    w_o = torch.randn(H, heads * 128, device="cuda") / H**0.5
    o, res = W8A8Linear(w_o), torch.randn(M, H, device="cuda", dtype=torch.bfloat16)
    ref = xd @ _deq(o.w_q, o.w_scale).T + res.float()
    _close(_gemm(q, s, o, torch.bfloat16, residual=res), ref, 0.01, "residual epilogue")

    gu, dn = W8A8Linear(torch.randn(2 * I, H, device="cuda") / H**0.5), W8A8Linear(torch.randn(H, I, device="cuda"))
    hq, hs = swiglu_gemm_quant(q, s, swiglu_weights(gu, dn), torch.bfloat16)
    g_u = xd @ _deq(gu.w_q, gu.w_scale).T
    h_ref = F.silu(g_u[:, :I]) * g_u[:, I:]
    err = (_deq(hq, hs) - h_ref).abs()
    assert bool((err <= 0.51 * hs[:, None] + 0.01 * h_ref.abs()).all()), f"swiglu: max error {err.max().item():.3g}"

    cos, sin = _rope_tables(64)
    pos = torch.randint(0, 64, (M,), device="cuda")
    add = torch.randn(M, n_qkv, device="cuda", dtype=torch.bfloat16)
    out = qkv_rope_gemm(q, s, rope_weights(qkv, 128), torch.bfloat16, pos, cos, sin, heads + kv, addend=add)
    ref = _rope_ref(xd @ _deq(qkv.w_q, qkv.w_scale).T + qkv.bias + add.float(), pos, cos, sin, heads + kv)
    _close(out, ref, 0.01, "qkv + bias + addend + RoPE")


@CUDA
@pytest.mark.parametrize("M", [5, 64, 300])
def test_segmented_qkv_rope_and_residual_gemms_take_each_segment_s_weights(M):
    """The adapters' segmented GEMMs (rows grouped by adapter, one merged int8 weight per slot): qkv with RoPE and
    the o GEMM with the residual, each segment against the fp32 product with its own weights."""
    from nanovllm_voxcpm.kernels.fused_layer import rope_weights
    from nanovllm_voxcpm.kernels.int8_linear import W8A8Linear, quantize_rows
    from nanovllm_voxcpm.kernels.seg_gemm import build_tiles, seg_gemm, seg_qkv_rope_gemm

    torch.manual_seed(M)
    H, n_qkv, n_rope = 1024, 2560, 18
    lins = [W8A8Linear(torch.randn(n_qkv, H, device="cuda") / H**0.5) for _ in range(3)]  # base, slots 0 and 1
    rope = [rope_weights(lin, 128) for lin in lins]
    w, sc = torch.stack([r.w_q for r in rope]), torch.stack([r.w_scale for r in rope])
    olins = [W8A8Linear(torch.randn(H, H, device="cuda") / H**0.5) for _ in range(3)]
    wo, so = torch.stack([o.w_q for o in olins]), torch.stack([o.w_scale for o in olins])
    row_slots = sorted(torch.randint(-1, 2, (M,)).tolist(), key=lambda t: (t < 0, t))
    bm = 128 if M > 256 else 64
    tiles = build_tiles(row_slots, bm)
    seg = (torch.from_numpy(tiles).cuda(), tiles.shape[0], bm)
    x = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)
    q, s = quantize_rows(x)
    cos, sin = _rope_tables(64)
    pos = torch.randint(0, 64, (M,), device="cuda")
    out = seg_qkv_rope_gemm(q, s, w, sc, torch.bfloat16, pos, cos, sin, n_rope, *seg)
    res = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)
    xo = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)
    oq, os_ = quantize_rows(xo)
    out_o = seg_gemm(oq, os_, wo, so, None, torch.bfloat16, *seg, residual=res)
    for slot in (-1, 0, 1):
        rows = torch.tensor([i for i, t in enumerate(row_slots) if t == slot], device="cuda", dtype=torch.long)
        if not rows.numel():
            continue
        lin, olin = lins[slot + 1], olins[slot + 1]
        ref = _rope_ref(_deq(q[rows], s[rows]) @ _deq(lin.w_q, lin.w_scale).T, pos[rows], cos, sin, n_rope)
        _close(out[rows], ref, 0.01, f"seg qkv rope, slot {slot}")
        ref = _deq(oq[rows], os_[rows]) @ _deq(olin.w_q, olin.w_scale).T + res[rows].float()
        _close(out_o[rows], ref, 0.01, f"seg residual, slot {slot}")


# --- attention ----------------------------------------------------------------------------------------------------

def _attend(q, k, v, causal: bool):
    """q [n_q, H, D], k / v [n_k, HK, D], fp32; causal: query i sees keys up to n_k - n_q + i."""
    H, HK = q.shape[1], k.shape[1]
    k, v = k.repeat_interleave(H // HK, 1), v.repeat_interleave(H // HK, 1)
    att = torch.einsum("qhd,khd->hqk", q, k) / q.shape[-1] ** 0.5
    if causal:
        n_q, n_k = q.shape[0], k.shape[0]
        mask = torch.arange(n_k, device=q.device)[None, :] > (n_k - n_q + torch.arange(n_q, device=q.device))[:, None]
        att = att.masked_fill(mask[None], float("-inf"))
    return torch.einsum("hqk,khd->qhd", att.softmax(-1), v).reshape(q.shape[0], -1)


def _check_quant(out, ref, what):
    o, scale = out
    err = (_deq(o, scale) - ref).abs()
    assert bool((err <= 0.51 * scale[:, None] + 0.02 * ref.abs().max()).all()), f"{what}: {err.max().item():.3g}"


@CUDA
@pytest.mark.parametrize("quant", [False, True])
def test_paged_prefill_attention_over_a_cached_prefix(quant):
    """Prefill of new tokens after a cached prefix, K/V read through scattered 256-token pages: a sequence without
    prefix, one with 293 cached tokens (two pages), one single-token prefill after 260."""
    from nanovllm_voxcpm.kernels.prefill_attn import paged_prefill_attention

    torch.manual_seed(1)
    H, HK, D, page = 16, 2, 128, 256
    lens = [(5, 5), (7, 300), (1, 261)]  # (new, total)
    k_cache = torch.randn(8, page, HK, D, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.randn(8, page, HK, D, device="cuda", dtype=torch.bfloat16)
    tables = torch.tensor([[6, 0], [3, 5], [1, 7]], device="cuda", dtype=torch.int32)
    T = sum(n for n, _ in lens)
    q = torch.randn(T, H, D, device="cuda", dtype=torch.bfloat16)
    cu_q = torch.tensor([0, 5, 12, 13], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, 5, 305, 566], device="cuda", dtype=torch.int32)
    out = paged_prefill_attention(q, k_cache, v_cache, tables, cu_q, cu_k, 7, D**-0.5, quant=quant)
    refs = []
    for b, (n_q, n_k) in enumerate(lens):
        pages = tables[b].long()
        k = k_cache[pages].reshape(-1, HK, D)[:n_k].float()
        v = v_cache[pages].reshape(-1, HK, D)[:n_k].float()
        refs.append(_attend(q[cu_q[b]:cu_q[b + 1]].float(), k, v, causal=True))
    ref = torch.cat(refs)
    if quant:
        _check_quant(out, ref, "paged prefill (int8 o)")
    else:
        _close(out, ref, 0.02, "paged prefill")


@CUDA
@pytest.mark.parametrize("quant", [False, True])
def test_short_attention_kernels(quant):
    """A burst's short causal prefills (1-32 tokens per sequence, packed) and the non-causal attention of short rows
    (LocEnc / LocDiT, <= 16 tokens), against the plain softmax attention."""
    from nanovllm_voxcpm.kernels.short_attn import short_attention, short_causal_attention

    torch.manual_seed(2)
    H, HK, D = 16, 2, 128
    lens = [1, 7, 16, 32, 3]
    T = sum(lens)
    q = torch.randn(T, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(T, HK, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(T, HK, D, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), device="cuda", dtype=torch.int32)
    out = short_causal_attention(q, k, v, cu, max(lens), D**-0.5, quant=quant)
    ref = torch.cat([_attend(q[a:b].float(), k[a:b].float(), v[a:b].float(), causal=True)
                     for a, b in zip(cu[:-1].tolist(), cu[1:].tolist())])
    _check_quant(out, ref, "short causal") if quant else _close(out, ref, 0.02, "short causal")

    for L in (1, 5, 16):
        B = 6
        q4 = torch.randn(B, L, H, D, device="cuda", dtype=torch.bfloat16)
        k4 = torch.randn(B, L, HK, D, device="cuda", dtype=torch.bfloat16)
        v4 = torch.randn(B, L, HK, D, device="cuda", dtype=torch.bfloat16)
        out = short_attention(q4, k4, v4, D**-0.5, quant=quant)
        ref = torch.cat([_attend(q4[b].float(), k4[b].float(), v4[b].float(), causal=False) for b in range(B)])
        _check_quant(out, ref, f"short L={L}") if quant else _close(out, ref, 0.02, f"short L={L}")


@CUDA
def test_kv_store_writes_each_token_to_its_slot():
    """The decode path's KV store (flash-attn reads the pages): each row lands at its slot; slot -1 is skipped."""
    from nanovllm_voxcpm.model.layers import store_kvcache

    torch.manual_seed(3)
    HK, D, page = 2, 128, 256
    k_cache = torch.zeros(4, page, HK, D, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.zeros_like(k_cache)
    k = torch.randn(9, HK, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(9, HK, D, device="cuda", dtype=torch.bfloat16)
    slots = torch.tensor([5, 300, 511, 512, 1023, -1, 0, 700, 256], device="cuda", dtype=torch.int32)
    store_kvcache(k, v, k_cache, v_cache, slots)
    flat_k, flat_v = k_cache.view(-1, HK, D), v_cache.view(-1, HK, D)
    for i, slot in enumerate(slots.tolist()):
        if slot >= 0:
            assert torch.equal(flat_k[slot], k[i]) and torch.equal(flat_v[slot], v[i]), i
    assert int((flat_k.abs().sum((1, 2)) > 0).sum()) == 8  # nothing else written


# --- streaming VAE decoder ----------------------------------------------------------------------------------------

def _plain(module):
    """The module with weight norm folded (as the runner does), on the GPU in fp16."""
    from torch.nn.utils import remove_weight_norm

    for m in module.modules():
        if hasattr(m, "weight_g"):
            remove_weight_norm(m)
    return module.cuda().half().eval()


def _stream(op, x, chunks, idx):
    """op over x [B, C, T] in ``chunks`` time pieces (its causal state carried per slot), concatenated (fp32)."""
    T = x.shape[-1]
    cuts = [T * i // chunks for i in range(chunks + 1)]
    return torch.cat([op(x[..., a:b].contiguous(), idx).float() for a, b in zip(cuts[:-1], cuts[1:])], -1)


def _ext(name):
    from nanovllm_voxcpm.kernels.cuda_ext import available

    if not available(name):
        pytest.skip(f"CUDA extension {name} not built")


@CUDA
@pytest.mark.parametrize("C", [32, 64, 128, 256])
@pytest.mark.parametrize("dilation", [1, 3, 9])
@torch.no_grad()
def test_cuda_residual_unit_matches_the_module(C, dilation):
    """cuda/resunit_fused.cu (snake, dilated depthwise conv, snake, 1x1 conv on tensor cores, residual) streamed in
    3 chunks on slots 2 and 0, against the module's forward over the whole input in fp32."""
    _ext("resunit_fused")
    from nanovllm_voxcpm.model.vae import CausalResidualUnit
    from nanovllm_voxcpm.kernels.resunit import CudaResidualUnit

    torch.manual_seed(C + dilation)
    mod = _plain(CausalResidualUnit(C, dilation=dilation, groups=C))  # depthwise, as the decoder's
    x = torch.randn(2, C, 120, device="cuda", dtype=torch.float16)
    ref = copy.deepcopy(mod).float()(x.float())
    out = _stream(CudaResidualUnit(mod, slots=3), x, 3, torch.tensor([2, 0], device="cuda"))
    _close(out, ref, 0.01, f"resunit C={C} dilation={dilation}")


@CUDA
@torch.no_grad()
def test_cuda_output_head_matches_the_module():
    """cuda/head_fused.cu (snake -> 32 -> 1 channel causal conv, kernel 7 -> tanh), streamed in 4 chunks, against
    the modules over the whole input in fp32."""
    _ext("head_fused")
    from nanovllm_voxcpm.model.vae import CausalConv1d, Snake1d
    from nanovllm_voxcpm.kernels.resunit import SnakeConvTanhHead

    torch.manual_seed(4)
    snake, conv = Snake1d(32).cuda().half(), CausalConv1d(32, 1, kernel_size=7, padding=3).cuda().half()
    snake.alpha.uniform_(0.5, 2.0)
    head = SnakeConvTanhHead(snake, conv, slots=3)
    assert head.ext is not None
    x = torch.randn(2, 32, 400, device="cuda", dtype=torch.float16)
    ref = torch.tanh(copy.deepcopy(conv).float()(copy.deepcopy(snake).float()(x.float())))
    _close(_stream(head, x, 4, torch.tensor([1, 2], device="cuda")), ref, 0.01, "head")


@CUDA
@pytest.mark.parametrize("c_in,stride,t", [(2048, 8, 4), (2048, 8, 1), (1024, 6, 32), (512, 5, 192), (256, 2, 64),
                                           (64, 2, 96)])
@torch.no_grad()
def test_fused_snake_transposed_conv_matches_the_modules(c_in, stride, t):
    """The decoder blocks' snake + causal transposed conv (tconv.py; the VAE's widths and strides, a latent chunk's
    input length and a one-frame chunk at the first block), streamed in 3 chunks of ``t`` steps, against the modules
    over the whole input in fp32."""
    from nanovllm_voxcpm.model.vae import CausalTransposeConv1d, Snake1d
    from nanovllm_voxcpm.kernels.tconv import FusedSnakeTransposeConv

    torch.manual_seed(c_in + t)
    snake = Snake1d(c_in).cuda().half()
    snake.alpha.uniform_(0.5, 2.0)
    tconv = CausalTransposeConv1d(c_in, c_in // 2, kernel_size=2 * stride, stride=stride,
                                  padding=(stride + 1) // 2, output_padding=stride % 2).cuda().half()
    op = FusedSnakeTransposeConv(snake, tconv, slots=3)
    x = torch.randn(2, c_in, 3 * t, device="cuda", dtype=torch.float16)
    ref = copy.deepcopy(tconv).float()(copy.deepcopy(snake).float()(x.float()))
    _close(_stream(op, x, 3, torch.tensor([0, 2], device="cuda")), ref, 0.01, f"tconv {c_in}/{stride}")


@CUDA
@torch.no_grad()
def test_slot_streaming_decoder_matches_the_whole_decode():
    """The engine's slot-streaming VAE decoder (fused kernels, the model's config: 2048 -> 32 channels, strides 8 6 5
    2 2 2, sample-rate conditioning), two streams decoded in 4-frame chunks plus a 1-frame one, against the decoder
    module over the whole latent sequence in fp32."""
    from nanovllm_voxcpm.model.vae import AudioVAEConfigV2, AudioVAEV2
    from nanovllm_voxcpm.kernels import SlotStreamingVAEDecoder

    torch.manual_seed(5)
    cfg = AudioVAEConfigV2(encoder_dim=128, encoder_rates=[2, 5, 8, 8], latent_dim=64, decoder_dim=2048,
                           decoder_rates=[8, 6, 5, 2, 2, 2], sr_bin_boundaries=[20000, 30000, 40000],
                           sample_rate=16000, out_sample_rate=48000)
    decoder = _plain(AudioVAEV2(config=cfg).decoder)
    z = torch.randn(2, 64, 9, device="cuda", dtype=torch.float16)
    ref = copy.deepcopy(decoder).float()(z.float())
    assert ref.std() > 0.05  # the random decoder's output is not silent or saturated (a meaningful comparison)
    dec = SlotStreamingVAEDecoder(decoder, max_streams=4, latent_dim=64, chunk_frames=4, fused=True,
                                  use_cuda_graph=False)
    out = torch.cat([dec.decode_chunks(z[..., a:b], ["a", "b"]).float() for a, b in ((0, 4), (4, 8), (8, 9))], -1)
    assert out.shape == ref.shape
    _close(out, ref, 0.02, "slot-streaming decoder")
