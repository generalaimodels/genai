"""
RSS-MoD Configuration
Recursive State-Space Mixture-of-Depths Architecture Configuration
"""

from dataclasses import dataclass, field
from typing import Optional, Literal, Tuple
from enum import Enum


class PrecisionMode(Enum):
    """Supported precision modes for kernel execution."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8_E4M3 = "fp8_e4m3"  # Hopper+ forward pass
    FP8_E5M2 = "fp8_e5m2"  # Hopper+ gradients


class SSMVariant(Enum):
    """SSM backbone variants."""
    MAMBA_1 = "mamba_1"
    MAMBA_2 = "mamba_2"
    S4D = "s4d"
    S6 = "s6"


@dataclass
class SSMConfig:
    """Selective State Space Model configuration."""
    d_state: int = 128                    # SSM state dimension N
    d_conv: int = 4                       # Local convolution width
    expand_factor: int = 2                # Inner dimension expansion E
    dt_rank: str = "auto"                 # Delta rank: "auto" = ceil(d_model/16)
    dt_min: float = 0.001                 # Minimum delta
    dt_max: float = 0.1                   # Maximum delta
    dt_init: str = "random"               # Delta initialization: "random" | "constant"
    dt_scale: float = 1.0                 # Delta scaling factor
    dt_init_floor: float = 1e-4           # Delta init floor
    conv_bias: bool = True                # Use bias in conv1d
    bias: bool = False                    # Use bias in linear projections
    use_fast_path: bool = True            # Use fused CUDA kernels
    variant: SSMVariant = SSMVariant.MAMBA_2
    chunk_size: int = 256                 # Parallel scan chunk size


@dataclass
class AttentionConfig:
    """Sliding Window Attention configuration."""
    num_heads: int = 32                   # Number of attention heads
    num_kv_heads: int = 8                 # GQA: KV heads (None = MHA)
    window_size: int = 4096               # Sliding window size
    max_position_embeddings: int = 131072 # Maximum sequence length
    rope_theta: float = 500000.0          # RoPE base frequency
    rope_scaling: Optional[dict] = None   # RoPE scaling config (YaRN/NTK)
    attention_dropout: float = 0.0        # Attention dropout
    use_flash_attn: bool = True           # Use FlashAttention-2/3
    use_sliding_window: bool = True       # Enable sliding window
    softcap: Optional[float] = None       # Attention logit soft-capping


@dataclass
class MoDConfig:
    """Mixture-of-Depths router configuration."""
    capacity_factor: float = 1.25         # Token capacity multiplier
    top_k: int = 1                        # Tokens selected per position
    aux_loss_weight: float = 0.01         # Load balancing auxiliary loss
    router_jitter: float = 0.0            # Router noise (training)
    use_aux_loss: bool = True             # Enable auxiliary load balance loss
    router_dtype: str = "float32"         # Router computation dtype


@dataclass
class MoEConfig:
    """Mixture-of-Experts configuration."""
    num_experts: int = 8                  # Total number of experts
    num_experts_per_tok: int = 2          # Experts activated per token
    expert_capacity: Optional[int] = None # Expert capacity (None = auto)
    shared_expert: bool = True            # Include shared expert
    capacity_factor: float = 1.25         # Capacity factor
    router_aux_loss_coef: float = 0.01    # Router auxiliary loss coefficient
    router_z_loss_coef: float = 0.001     # Router z-loss coefficient
    use_sinkhorn: bool = True             # Sinkhorn-Knopp balancing
    sinkhorn_iters: int = 3               # Sinkhorn iterations
    expert_parallel: bool = False         # Expert parallelism across devices


@dataclass
class JEPAConfig:
    """Joint-Embedding Predictive Architecture configuration."""
    predictor_depth: int = 6              # Predictor network depth
    predictor_width: int = 2048           # Predictor hidden dim
    ema_decay: float = 0.999              # Target encoder EMA decay
    loss_weight: float = 1.0              # JEPA loss weight
    mask_ratio: float = 0.75              # Context masking ratio
    target_aspect_ratio: Tuple[float, float] = (0.75, 1.5)
    target_scale: Tuple[float, float] = (0.15, 0.4)


@dataclass
class PauseHeadConfig:
    """System-2 Pause Token Head configuration."""
    max_pause_steps: int = 8              # Maximum vertical recurrence
    pause_threshold: float = 0.5          # Confidence threshold to emit
    hidden_dim: int = 1024                # Pause head hidden dim
    use_adaptive: bool = True             # Adaptive pause based on entropy


@dataclass
class HTCConfig:
    """Hierarchical Timescale Controller configuration."""
    num_timescales: int = 4               # Number of hierarchical levels
    fast_timescale: int = 1               # Fast processing interval
    slow_timescale: int = 8               # Slow processing interval
    gate_type: str = "learned"            # "learned" | "fixed" | "adaptive"


@dataclass
class SpeculativeConfig:
    """Self-Speculative Decoding configuration."""
    draft_layers: int = 4                 # Number of draft layers
    speculation_length: int = 5           # Tokens to speculate
    temperature: float = 1.0              # Sampling temperature
    top_k: int = 50                       # Top-k sampling
    top_p: float = 0.9                    # Nucleus sampling


@dataclass
class RSSMoDConfig:
    """
    Complete RSS-MoD Architecture Configuration.
    
    Recursive State-Space Mixture-of-Depths with:
    - Selective SSM (Mamba-2) backbone
    - Mixture-of-Depths dynamic compute
    - Sliding Window Attention interleaving
    - Mixture-of-Experts FFN
    - JEPA world model objective
    - System-2 pause tokens
    """
    # Core dimensions
    vocab_size: int = 128256              # Vocabulary size
    d_model: int = 4096                   # Model dimension
    n_layers: int = 32                    # Total number of layers
    
    # Layer composition
    ssm_layer_indices: Tuple[int, ...] = field(
        default_factory=lambda: tuple(range(0, 32))  # All layers have SSM
    )
    attn_layer_indices: Tuple[int, ...] = field(
        default_factory=lambda: tuple(range(0, 32, 4))  # Every 4th layer has attention
    )
    moe_layer_indices: Tuple[int, ...] = field(
        default_factory=lambda: tuple(range(1, 32, 2))  # Every 2nd layer has MoE
    )
    
    # Component configs
    ssm: SSMConfig = field(default_factory=SSMConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    mod: MoDConfig = field(default_factory=MoDConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    jepa: JEPAConfig = field(default_factory=JEPAConfig)
    pause_head: PauseHeadConfig = field(default_factory=PauseHeadConfig)
    htc: HTCConfig = field(default_factory=HTCConfig)
    speculative: SpeculativeConfig = field(default_factory=SpeculativeConfig)
    
    # FFN configuration
    intermediate_size: int = 14336        # FFN intermediate dim (for non-MoE)
    hidden_act: str = "silu"              # Activation function
    
    # Normalization
    rms_norm_eps: float = 1e-5            # RMSNorm epsilon
    
    # Regularization
    hidden_dropout: float = 0.0           # Hidden state dropout
    attention_dropout: float = 0.0        # Attention dropout
    
    # Precision
    precision: PrecisionMode = PrecisionMode.BF16
    use_fp8_linear: bool = False          # FP8 linear layers (Hopper+)
    
    # Initialization
    initializer_range: float = 0.02       # Weight init std
    
    # Optimization
    tie_word_embeddings: bool = False     # Tie input/output embeddings
    use_cache: bool = True                # Enable KV/state caching
    use_gradient_checkpointing: bool = True
    
    # Hardware targeting
    target_hardware: Literal["ampere", "hopper", "auto"] = "auto"
    
    def __post_init__(self):
        """Validate and auto-configure parameters."""
        # Auto-compute dt_rank
        if self.ssm.dt_rank == "auto":
            self.ssm.dt_rank = max(1, self.d_model // 16)
        
        # Validate layer indices
        for idx in self.attn_layer_indices:
            assert 0 <= idx < self.n_layers, f"Invalid attention layer index: {idx}"
        for idx in self.moe_layer_indices:
            assert 0 <= idx < self.n_layers, f"Invalid MoE layer index: {idx}"
    
    @classmethod
    def tiny(cls) -> "RSSMoDConfig":
        """Tiny config for testing (125M params)."""
        return cls(
            d_model=768,
            n_layers=12,
            ssm=SSMConfig(d_state=64),
            attention=AttentionConfig(num_heads=12, num_kv_heads=4),
            moe=MoEConfig(num_experts=4, num_experts_per_tok=1),
            intermediate_size=3072,
            attn_layer_indices=(0, 4, 8),
            moe_layer_indices=(1, 3, 5, 7, 9, 11),
        )
    
    @classmethod
    def small(cls) -> "RSSMoDConfig":
        """Small config (1.3B params)."""
        return cls(
            d_model=2048,
            n_layers=24,
            ssm=SSMConfig(d_state=128),
            attention=AttentionConfig(num_heads=16, num_kv_heads=4),
            moe=MoEConfig(num_experts=8, num_experts_per_tok=2),
            intermediate_size=8192,
            attn_layer_indices=tuple(range(0, 24, 4)),
            moe_layer_indices=tuple(range(1, 24, 2)),
        )
    
    @classmethod
    def base(cls) -> "RSSMoDConfig":
        """Base config (7B params)."""
        return cls(
            d_model=4096,
            n_layers=32,
            ssm=SSMConfig(d_state=128),
            attention=AttentionConfig(num_heads=32, num_kv_heads=8),
            moe=MoEConfig(num_experts=8, num_experts_per_tok=2),
            intermediate_size=14336,
        )
    
    @classmethod
    def large(cls) -> "RSSMoDConfig":
        """Large config (70B params)."""
        return cls(
            d_model=8192,
            n_layers=80,
            ssm=SSMConfig(d_state=256),
            attention=AttentionConfig(num_heads=64, num_kv_heads=8),
            moe=MoEConfig(num_experts=16, num_experts_per_tok=2),
            intermediate_size=28672,
            attn_layer_indices=tuple(range(0, 80, 8)),
            moe_layer_indices=tuple(range(1, 80, 2)),
        )
