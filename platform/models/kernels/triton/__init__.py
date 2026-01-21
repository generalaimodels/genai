# Triton Kernel Exports
# SOTA High-Performance GPU Kernels

from .rmsnorm import triton_rmsnorm, triton_rmsnorm_backward, RMSNorm
from .rotary_embedding import triton_rotary_embedding, precompute_freqs_cis, apply_rotary_pos_emb
from .ssm_scan import triton_parallel_scan, parallel_scan_forward, parallel_scan_backward
from .selective_scan import (
    triton_selective_scan,
    selective_scan_forward,
    selective_scan_backward,
    SelectiveScanFn,
)
from .sliding_window_attn import (
    triton_sliding_window_attention,
    flash_attn_sliding_window,
    FlashAttentionSlidingWindow,
)
from .mod_router import (
    triton_mod_router,
    mod_top_k_routing,
    MoDRouter,
)
from .moe_dispatch import (
    triton_moe_dispatch,
    triton_moe_gather,
    moe_dispatch_forward,
    moe_gather_forward,
    MoEDispatch,
)
from .fused_cross_entropy import (
    triton_fused_cross_entropy,
    fused_cross_entropy_forward,
    FusedCrossEntropyLoss,
)

__all__ = [
    # RMSNorm
    "triton_rmsnorm",
    "triton_rmsnorm_backward",
    "RMSNorm",
    # Rotary Embedding
    "triton_rotary_embedding",
    "precompute_freqs_cis",
    "apply_rotary_pos_emb",
    # Parallel Scan
    "triton_parallel_scan",
    "parallel_scan_forward",
    "parallel_scan_backward",
    # Selective Scan
    "triton_selective_scan",
    "selective_scan_forward",
    "selective_scan_backward",
    "SelectiveScanFn",
    # Sliding Window Attention
    "triton_sliding_window_attention",
    "flash_attn_sliding_window",
    "FlashAttentionSlidingWindow",
    # MoD Router
    "triton_mod_router",
    "mod_top_k_routing",
    "MoDRouter",
    # MoE Dispatch
    "triton_moe_dispatch",
    "triton_moe_gather",
    "moe_dispatch_forward",
    "moe_gather_forward",
    "MoEDispatch",
    # Cross Entropy
    "triton_fused_cross_entropy",
    "fused_cross_entropy_forward",
    "FusedCrossEntropyLoss",
]
