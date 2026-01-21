"""
Grouped Query Flash Attention with Sliding Window
FlashAttention-2 tiling strategy with GQA and causal masking
IO Complexity: O(N²d²/M) where M = SRAM size
"""

import torch
import triton
import triton.language as tl
import math


@triton.jit
def _flash_attention_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    L_ptr,  # LSE (log-sum-exp) for backward
    B, H_q, H_kv, L, D,
    window_size: tl.constexpr,
    stride_qb, stride_qh, stride_ql, stride_qd,
    stride_kb, stride_kh, stride_kl, stride_kd,
    stride_vb, stride_vh, stride_vl, stride_vd,
    stride_ob, stride_oh, stride_ol, stride_od,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Flash Attention Forward with GQA and Sliding Window
    - GQA: H_q query heads share H_kv key/value heads (H_q % H_kv == 0)
    - Sliding window: attention only within window_size tokens
    - Causal masking: can't attend to future positions
    """
    
    # Program IDs
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)
    
    # GQA: map query head to key/value head
    h_kv = pid_h // (H_q // H_kv)
    
    # Query block offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    
    # Load query block [BLOCK_M, BLOCK_D]
    q_offset = (pid_b * stride_qb + pid_h * stride_qh + 
                offs_m[:, None] * stride_ql + offs_d[None, :] * stride_qd)
    mask_m = offs_m < L
    Q = tl.load(Q_ptr + q_offset, mask=mask_m[:, None], other=0.0)
    
    # Initialize output accumulator and normalization
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    
    # Scaling factor
    scale = 1.0 / math.sqrt(D)
    
    # Iterate over key/value blocks
    for start_n in range(0, L, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < L
        
        # Sliding window + causal mask
        # Can attend if: offs_n <= offs_m AND offs_m - offs_n < window_size
        mask_window = (offs_n[None, :] <= offs_m[:, None]) & \
                      (offs_m[:, None] - offs_n[None, :] < window_size)
        
        # Load key block [BLOCK_N, BLOCK_D]
        k_offset = (pid_b * stride_kb + h_kv * stride_kh + 
                    offs_n[:, None] * stride_kl + offs_d[None, :] * stride_kd)
        K = tl.load(K_ptr + k_offset, mask=mask_n[:, None], other=0.0)
        
        # Compute QK^T [BLOCK_M, BLOCK_N]
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(Q, tl.trans(K))
        qk *= scale
        
        # Apply mask (set invalid positions to -inf)
        qk = tl.where(mask_window, qk, float('-inf'))
        
        # Online softmax: update running max and sum
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_ij - m_new)
        
        # Softmax numerator
        p = tl.exp(qk - m_new[:, None])
        
        # Update running sum
        l_new = alpha * l_i + beta * tl.sum(p, axis=1)
        
        # Load value block [BLOCK_N, BLOCK_D]
        v_offset = (pid_b * stride_vb + h_kv * stride_vh + 
                    offs_n[:, None] * stride_vl + offs_d[None, :] * stride_vd)
        V = tl.load(V_ptr + v_offset, mask=mask_n[:, None], other=0.0)
        
        # Update accumulator: rescale old values, add new
        acc = alpha[:, None] * acc + tl.dot(p.to(V.dtype), V)
        
        # Update statistics
        m_i = m_new
        l_i = l_new
    
    # Final normalization
    acc = acc / l_i[:, None]
    
    # Store output
    o_offset = (pid_b * stride_ob + pid_h * stride_oh + 
                offs_m[:, None] * stride_ol + offs_d[None, :] * stride_od)
    tl.store(Out_ptr + o_offset, acc, mask=mask_m[:, None])
    
    # Store LSE for backward pass
    lse = m_i + tl.log(l_i)
    l_offset = pid_b * H_q * L + pid_h * L + offs_m
    tl.store(L_ptr + l_offset, lse, mask=mask_m)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_D': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_D': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_D': 64}, num_warps=4, num_stages=3),
    ],
    key=['L', 'D'],
)
@triton.jit
def _flash_attention_fwd_kernel_optimized(
    Q_ptr, K_ptr, V_ptr, Out_ptr, L_ptr,
    B, H_q, H_kv, L, D, window_size: tl.constexpr,
    stride_qb, stride_qh, stride_ql, stride_qd,
    stride_kb, stride_kh, stride_kl, stride_kd,
    stride_vb, stride_vh, stride_vl, stride_vd,
    stride_ob, stride_oh, stride_ol, stride_od,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Autotuned version"""
    _flash_attention_fwd_kernel(
        Q_ptr, K_ptr, V_ptr, Out_ptr, L_ptr,
        B, H_q, H_kv, L, D, window_size,
        stride_qb, stride_qh, stride_ql, stride_qd,
        stride_kb, stride_kh, stride_kl, stride_kd,
        stride_vb, stride_vh, stride_vl, stride_vd,
        stride_ob, stride_oh, stride_ol, stride_od,
        BLOCK_M, BLOCK_N, BLOCK_D,
    )


def flash_attention_gqa(Q, K, V, window_size=4096):
    """
    Grouped Query Flash Attention with Sliding Window
    
    Args:
        Q: [B, H_q, L, D] query (multiple heads)
        K: [B, H_kv, L, D] key (fewer heads, GQA)
        V: [B, H_kv, L, D] value
        window_size: sliding window size (causal constraint)
        
    Returns:
        Out: [B, H_q, L, D] attention output
        LSE: [B, H_q, L] log-sum-exp for backward
    """
    B, H_q, L, D = Q.shape
    H_kv = K.shape[1]
    
    assert H_q % H_kv == 0, "Query heads must be divisible by KV heads (GQA)"
    
    # Allocate outputs
    Out = torch.empty_like(Q)
    LSE = torch.empty(B, H_q, L, device=Q.device, dtype=torch.float32)
    
    # Launch grid
    grid = lambda META: (
        B,
        H_q,
        triton.cdiv(L, META['BLOCK_M'])
    )
    
    _flash_attention_fwd_kernel_optimized[grid](
        Q, K, V, Out, LSE,
        B, H_q, H_kv, L, D, window_size,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
    )
    
    return Out, LSE
