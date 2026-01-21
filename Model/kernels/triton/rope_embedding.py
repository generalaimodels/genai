"""
Rotary Position Embedding (RoPE) Kernel
Applies rotation matrices to Q/K for relative position encoding
"""

import torch
import triton
import triton.language as tl
import math


@triton.jit
def _rope_embedding_kernel(
    q_ptr,        # [B, H, L, D] query tensor
    k_ptr,        # [B, H, L, D] key tensor
    cos_ptr,      # [L, D/2] cosine cache
    sin_ptr,      # [L, D/2] sine cache
    q_out_ptr,    # [B, H, L, D] rotated query
    k_out_ptr,    # [B, H, L, D] rotated key
    B, H, L, D,
    stride_qb, stride_qh, stride_ql, stride_qd,
    stride_kb, stride_kh, stride_kl, stride_kd,
    stride_qob, stride_qoh, stride_qol, stride_qod,
    stride_kob, stride_koh, stride_kol, stride_kod,
    BLOCK_SIZE_D: tl.constexpr,
):
    """
    RoPE: Rotate pairs of dimensions using precomputed sin/cos
    [x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]
    """
    
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_l = tl.program_id(2)
    
    # Process dimension pairs
    offs_d = tl.arange(0, BLOCK_SIZE_D)
    mask_d = offs_d < D // 2
    
    # Load Q pair
    q_offset_0 = (pid_b * stride_qb + pid_h * stride_qh + 
                  pid_l * stride_ql + (offs_d * 2) * stride_qd)
    q_offset_1 = q_offset_0 + stride_qd
    q0 = tl.load(q_ptr + q_offset_0, mask=mask_d, other=0.0)
    q1 = tl.load(q_ptr + q_offset_1, mask=mask_d, other=0.0)
    
    # Load K pair
    k_offset_0 = (pid_b * stride_kb + pid_h * stride_kh + 
                  pid_l * stride_kl + (offs_d * 2) * stride_kd)
    k_offset_1 = k_offset_0 + stride_kd
    k0 = tl.load(k_ptr + k_offset_0, mask=mask_d, other=0.0)
    k1 = tl.load(k_ptr + k_offset_1, mask=mask_d, other=0.0)
    
    # Load sin/cos for position pid_l
    cos_offset = pid_l * (D // 2) + offs_d
    sin_offset = pid_l * (D // 2) + offs_d
    cos = tl.load(cos_ptr + cos_offset, mask=mask_d, other=1.0)
    sin = tl.load(sin_ptr + sin_offset, mask=mask_d, other=0.0)
    
    # Apply rotation
    q0_rot = q0 * cos - q1 * sin
    q1_rot = q0 * sin + q1 * cos
    k0_rot = k0 * cos - k1 * sin
    k1_rot = k0 * sin + k1 * cos
    
    # Store rotated Q
    qo_offset_0 = (pid_b * stride_qob + pid_h * stride_qoh + 
                   pid_l * stride_qol + (offs_d * 2) * stride_qod)
    qo_offset_1 = qo_offset_0 + stride_qod
    tl.store(q_out_ptr + qo_offset_0, q0_rot, mask=mask_d)
    tl.store(q_out_ptr + qo_offset_1, q1_rot, mask=mask_d)
    
    # Store rotated K
    ko_offset_0 = (pid_b * stride_kob + pid_h * stride_koh + 
                   pid_l * stride_kol + (offs_d * 2) * stride_kod)
    ko_offset_1 = ko_offset_0 + stride_kod
    tl.store(k_out_ptr + ko_offset_0, k0_rot, mask=mask_d)
    tl.store(k_out_ptr + ko_offset_1, k1_rot, mask=mask_d)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_D': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE_D': 64}, num_warps=4, num_stages=2),
    ],
    key=['D'],
)
@triton.jit
def _rope_embedding_kernel_optimized(
    q_ptr, k_ptr, cos_ptr, sin_ptr, q_out_ptr, k_out_ptr,
    B, H, L, D,
    stride_qb, stride_qh, stride_ql, stride_qd,
    stride_kb, stride_kh, stride_kl, stride_kd,
    stride_qob, stride_qoh, stride_qol, stride_qod,
    stride_kob, stride_koh, stride_kol, stride_kod,
    BLOCK_SIZE_D: tl.constexpr,
):
    """Autotuned version"""
    _rope_embedding_kernel(
        q_ptr, k_ptr, cos_ptr, sin_ptr, q_out_ptr, k_out_ptr,
        B, H, L, D,
        stride_qb, stride_qh, stride_ql, stride_qd,
        stride_kb, stride_kh, stride_kl, stride_kd,
        stride_qob, stride_qoh, stride_qol, stride_qod,
        stride_kob, stride_koh, stride_kol, stride_kod,
        BLOCK_SIZE_D,
    )


def precompute_freqs_cis(dim, max_seq_len=8192, theta=10000.0, device='cuda'):
    """Precompute cos/sin frequencies for RoPE"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_seq_len, device=device)
    freqs = torch.outer(t, freqs)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin


def rope_embedding(q, k, cos_cache, sin_cache):
    """
    Apply Rotary Position Embedding to Q and K
    
    Args:
        q: [B, H, L, D] query tensor
        k: [B, H, L, D] key tensor
        cos_cache: [max_L, D/2] precomputed cosines
        sin_cache: [max_L, D/2] precomputed sines
        
    Returns:
        q_rot: [B, H, L, D] rotated query
        k_rot: [B, H, L, D] rotated key
    """
    B, H, L, D = q.shape
    
    # Allocate outputs
    q_rot = torch.empty_like(q)
    k_rot = torch.empty_like(k)
    
    # Launch grid
    grid = (B, H, L)
    
    _rope_embedding_kernel_optimized[grid](
        q, k, cos_cache, sin_cache, q_rot, k_rot,
        B, H, L, D,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        q_rot.stride(0), q_rot.stride(1), q_rot.stride(2), q_rot.stride(3),
        k_rot.stride(0), k_rot.stride(1), k_rot.stride(2), k_rot.stride(3),
    )
    
    return q_rot, k_rot
