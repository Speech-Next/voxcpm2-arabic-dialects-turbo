from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class RopeScalingConfig(BaseModel):
    type: str
    long_factor: List[float]
    short_factor: List[float]
    original_max_position_embeddings: int


class MiniCPM4Config(BaseModel):
    bos_token_id: int
    eos_token_id: int
    hidden_size: int
    intermediate_size: int
    max_position_embeddings: int
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_scaling: RopeScalingConfig
    vocab_size: int
    use_mup: bool = True
    scale_emb: float
    dim_model_base: int
    scale_depth: float
    rope_theta: float
    kv_channels: int | None = None


class CfmConfig(BaseModel):
    sigma_min: float = 1e-06
    solver: str = "euler"
    t_scheduler: str = "log-norm"


class VoxCPM2EncoderConfig(BaseModel):
    hidden_dim: int = 1024
    ffn_dim: int = 4096
    num_heads: int = 16
    num_layers: int = 4
    kv_channels: int | None = None


class VoxCPM2DitConfig(BaseModel):
    hidden_dim: int = 1024
    ffn_dim: int = 4096
    num_heads: int = 16
    num_layers: int = 4
    kv_channels: int | None = None
    concat_hidden: bool = True
    cfm_config: CfmConfig


class AudioVAEV2Config(BaseModel):
    encoder_dim: int = 128
    encoder_rates: List[int] = [2, 5, 8, 8]
    latent_dim: int = 64
    decoder_dim: int = 2048
    decoder_rates: List[int] = [8, 6, 5, 2, 2, 2]
    depthwise: bool = True
    sample_rate: int = 16000
    out_sample_rate: int = 48000
    use_noise_block: bool = False
    sr_bin_boundaries: Optional[List[int]] = [20000, 30000, 40000]
    cond_type: str = "scale_bias"
    cond_dim: int = 128
    cond_out_layer: bool = False


class LoRAConfig(BaseModel):
    """LoRA configuration for VoxCPM inference.

    Attributes:
        enable_lm: Apply LoRA to base_lm and residual_lm
        enable_dit: Apply LoRA to VoxCPMLocDiT (feat_decoder.estimator)
        enable_proj: Apply LoRA to projection Linear layers
        max_loras: Maximum concurrently resident LoRAs per layer
        max_lora_rank: Maximum supported LoRA rank per layer slot
        target_modules_lm: Target modules in LM layers (e.g., ["q_proj", "k_proj", "v_proj", "o_proj"])
        target_modules_dit: Target modules in DiT layers
        target_proj_modules: Projection layer names to apply LoRA
    """

    model_config = ConfigDict(extra="forbid")

    enable_lm: bool = True
    enable_dit: bool = True
    enable_proj: bool = False
    max_loras: int = 1
    max_lora_rank: int = 32
    target_modules_lm: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj"]
    target_modules_dit: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj"]
    target_proj_modules: List[str] = [
        "enc_to_lm_proj",
        "lm_to_dit_proj",
        "res_to_dit_proj",
    ]


class TurboConfig(BaseModel):
    """Latency/throughput trade-offs (``NANOVLLM_VOXCPM2_TURBO`` JSON env var).

    Defaults reproduce the original algorithm except ``skip_text_locenc``,
    which only drops work whose result was masked out anyway.
    """

    first_steps: Optional[int] = None  # CFM steps for the first patch (None = inference_timesteps)
    cfg_steps: int = -1  # evaluated decode steps that use CFG (-1 = all, 0 = none)
    first_cfg_steps: int = -1  # same for the first patch
    skip_text_locenc: bool = True  # prefill: run LocEnc only on audio positions
    vae_kernel: bool = True  # voxcpm_kernels slot-based CUDA-graph streaming VAE decoder
    vae_fused: bool = True  # fused Triton residual units inside it (same output, ~1.4x faster VAE)
    # W8A8 (voxcpm_kernels): per-token int8 activations x per-channel int8 weights, for calls with
    # at least w8a8_min_rows rows (prefill, a full decode batch, DiT, LocEnc); smaller batches stay bf16. At 128+
    # rows the int8 path beats bf16 at the A10's power cap (256-stream decode step 116 -> 110 ms, same quality).
    fast_noise: bool = True  # one Philox launch for all rows' seeded noise (voxcpm_kernels.seeded_randn)
    prefill_graph: bool = True  # CUDA-graph the prefill backbone per token bucket (paged KV, clone audio too)
    # largest prefill step replayed from a graph: small steps are launch-bound (a single cloned request), large
    # ones GPU-bound, where padding up to the next bucket would only add work
    prefill_graph_max_tokens: int = 512
    # Reference-prefix KV cache: the first KV block of each voice's (reference latents') first request is kept;
    # later requests of that voice copy the KV of the longest token prefix they share with it (the reference
    # audio, plus the prompt text in prompt+reference cloning) and prefill only the tokens after it.
    ref_prefix_cache: bool = True
    ref_prefix_cache_size: int = 64
    # LocEnc embedding cache (rows; 0 = off): prefill embeds of audio patches seen before (a cloned voice's
    # prompt audio) are reused instead of re-running LocEnc; LocEnc encodes each patch independently.
    locenc_cache_size: int = 8192
    # at most this many sequences per prefill step (0 = only max_num_batched_tokens limits): with a large token
    # budget, short (text-only / cached-prefix) prompts still start streaming in groups instead of all at once
    prefill_max_seqs: int = 0
    # stream int16 PCM instead of float32 waveforms: converted on the GPU before the copy to the host, half the
    # bytes through the result pipe (the OpenAI-style server turns this on)
    pcm16_output: bool = False
    w8a8: bool = False
    w8a8_min_rows: int = 128
    w8a8_targets: List[str] = ["base_lm", "residual_lm", "feat_encoder", "feat_decoder"]
    fused_layers: bool = True  # voxcpm_kernels fused decoder layers (RoPE, norm+quant, epilogue residual)
    # Distilled LocDiT: estimator weights + explicit Euler time grid (no zero-init step, no CFG).
    dit_checkpoint: Optional[str] = None
    dit_t_span: Optional[List[float]] = None  # e.g. [1.0, 0.7071, 0.0]; read from the checkpoint if None
    # the first patch's grid (prefill), e.g. [1.0, 0.0]: fewer LocDiT solves before the first audio
    dit_first_t_span: Optional[List[float]] = None
    # Head start (0 = off): a request whose text has more than head_tokens + head_min_rest tokens first generates
    # head_patches patches from a head of its text (a word-boundary prefix of at most head_tokens tokens; a clause
    # boundary when there is one), then continues with the whole text: the head's patches become prompt audio of a
    # continuation that keeps the request's stream, VAE state, seed and voice. Time to first audio then waits only
    # for the head's prefill; only the first head_patches patches (160 ms each) were sampled seeing the head alone.
    head_tokens: int = 0
    head_patches: int = 3
    head_min_rest: int = 8


class VoxCPM2Config(BaseModel):
    lm_config: MiniCPM4Config
    patch_size: int = 4
    feat_dim: int = 64
    residual_lm_num_layers: int = 8
    residual_lm_no_rope: bool = False
    scalar_quantization_latent_dim: int = 512
    scalar_quantization_scale: int = 9

    encoder_config: VoxCPM2EncoderConfig
    dit_config: VoxCPM2DitConfig
    audio_vae_config: Optional[AudioVAEV2Config] = None
    max_length: int = 8192
    device: str = "cuda"
    dtype: str = "bfloat16"
    dit_mean_mode: bool = Field(default=False)

    inference_timesteps: int = 10
    turbo: TurboConfig = Field(default_factory=TurboConfig)
