"""HNS-SS-JEPA-MoE Model Package"""

from .config import ModelConfig, get_config_1B, get_config_7B, get_config_70B
from .model import HNS_SS_JEPA_MoE, create_model
from .kernels import (
    ssm_scan_fwd,
    rms_norm_linear,
    conv1d_silu,
    rope_embedding,
    precompute_freqs_cis,
    flash_attention_gqa,
    topk_gating,
    swiglu_expert,
)
from .modules import (
    SelectiveSSM,
    HybridAttention,
    HierarchicalMoE,
    DualHeadPredictor,
    HybridResidualBlock,
)

__version__ = '0.1.0'

__all__ = [
    # Config
    'ModelConfig',
    'get_config_1B',
    'get_config_7B',
    'get_config_70B',
    # Model
    'HNS_SS_JEPA_MoE',
    'create_model',
    # Kernels
    'ssm_scan_fwd',
    'rms_norm_linear',
    'conv1d_silu',
    'rope_embedding',
    'precompute_freqs_cis',
    'flash_attention_gqa',
    'topk_gating',
    'swiglu_expert',
    # Modules
    'SelectiveSSM',
    'HybridAttention',
    'HierarchicalMoE',
    'DualHeadPredictor',
    'HybridResidualBlock',
]
