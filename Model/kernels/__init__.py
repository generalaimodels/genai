"""
HNS-SS-JEPA-MoE Kernel Layer
Triton-optimized compute kernels for maximum hardware utilization
"""

from .triton import (
    ssm_scan_fwd,
    rms_norm_linear,
    conv1d_silu,
    rope_embedding,
    precompute_freqs_cis,
    flash_attention_gqa,
    topk_gating,
    swiglu_expert,
)

__all__ = [
    'ssm_scan_fwd',
    'rms_norm_linear',
    'conv1d_silu',
    'rope_embedding',
    'precompute_freqs_cis',
    'flash_attention_gqa',
    'topk_gating',
    'swiglu_expert',
]
