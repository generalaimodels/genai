"""
Sliding Window Attention (SWA) Triton Kernel
=============================================
FlashAttention-2 style IO-aware implementation with sliding window masking.

Features:
- Tiled softmax with online normalization
- Sliding window + causal masking fusion
- GQA (Grouped Query Attention) support with KV broadcast
- Softcapping for logit stability
- BF16/FP16 with FP32 accumulators

Performance:
- O(L * W) instead of O(L^2) for window size W
- Memory: O(L) vs O(L^2) for full attention
- SRAM tiling eliminates HBM softmax materialization
"""

import torch
import triton
import triton.language as tl
from typing import Optional, Tuple
import math


@triton.jit
def _swa_fwd_kernel(
    # Inputs
    Q,              # Query: (batch, seq_len, num_heads, head_dim)
    K,              # Key: (batch, seq_len, num_kv_heads, head_dim)
    V,              # Value: (batch, seq_len, num_kv_heads, head_dim)
    # Output
    O,              # Output: (batch, seq_len, num_heads, head_dim)
    LSE,            # Log-sum-exp for backward: (batch, num_heads, seq_len)
    # Dimensions
    batch_size,
    seq_len,
    num_heads,
    num_kv_heads,
    head_dim,
    # Window parameters
    window_size,    # Size of sliding window
    # Scaling
    sm_scale,       # Softmax scale (1/sqrt(head_dim))
    softcap,        # Softcap value (0 = disabled)
    # Strides Q
    stride_qb,
    stride_qs,
    stride_qh,
    stride_qd,
    # Strides K
    stride_kb,
    stride_ks,
    stride_kh,
    stride_kd,
    # Strides V
    stride_vb,
    stride_vs,
    stride_vh,
    stride_vd,
    # Strides O
    stride_ob,
    stride_os,
    stride_oh,
    stride_od,
    # Stride LSE
    stride_lseb,
    stride_lseh,
    stride_lses,
    # Config
    IS_CAUSAL: tl.constexpr,
    USE_SOFTCAP: tl.constexpr,
    BLOCK_M: tl.constexpr,      # Block size for Q sequence
    BLOCK_N: tl.constexpr,      # Block size for K/V sequence
    BLOCK_D: tl.constexpr,      # Block size for head_dim
):
    """
    FlashAttention-2 style sliding window attention forward pass.
    
    Grid: (cdiv(seq_len, BLOCK_M), batch * num_heads)
    Each program computes BLOCK_M output positions for one head.
    
    Online softmax normalization avoids materializing full attention matrix.
    """
    seq_block_idx = tl.program_id(0)
    batch_head_idx = tl.program_id(1)
    
    batch_idx = batch_head_idx // num_heads
    head_idx = batch_head_idx % num_heads
    
    # GQA: map query head to KV head
    kv_head_idx = head_idx // (num_heads // num_kv_heads)
    
    # Sequence offsets for this block
    m_offs = seq_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    
    # Initialize output accumulator and normalization stats
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)  # Max logit
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # Sum of exp
    
    # Load Q for this block: (BLOCK_M, BLOCK_D)
    q_ptrs = Q + batch_idx * stride_qb + m_offs[:, None] * stride_qs + \
             head_idx * stride_qh + tl.arange(0, BLOCK_D)[None, :] * stride_qd
    q_mask = (m_offs[:, None] < seq_len) & (tl.arange(0, BLOCK_D)[None, :] < head_dim)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)
    
    # Compute valid K/V range for sliding window
    # For position m, valid K positions are [max(0, m - window_size + 1), m + 1) for causal
    # or [max(0, m - window_size // 2), min(seq_len, m + window_size // 2 + 1)) for non-causal
    
    q_start = seq_block_idx * BLOCK_M
    
    if IS_CAUSAL:
        # Causal: only attend to previous positions within window
        kv_start = tl.maximum(0, q_start - window_size + 1)
        kv_end = tl.minimum(seq_len, q_start + BLOCK_M)
    else:
        # Symmetric window
        half_window = window_size // 2
        kv_start = tl.maximum(0, q_start - half_window)
        kv_end = tl.minimum(seq_len, q_start + BLOCK_M + half_window)
    
    # Iterate over K/V blocks in window
    for kv_block_start in range(kv_start, kv_end, BLOCK_N):
        n_offs = kv_block_start + tl.arange(0, BLOCK_N)
        
        # Load K: (BLOCK_N, BLOCK_D)
        k_ptrs = K + batch_idx * stride_kb + n_offs[:, None] * stride_ks + \
                 kv_head_idx * stride_kh + tl.arange(0, BLOCK_D)[None, :] * stride_kd
        k_mask = (n_offs[:, None] < seq_len) & (tl.arange(0, BLOCK_D)[None, :] < head_dim)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)
        
        # Compute attention scores: Q @ K^T
        # scores shape: (BLOCK_M, BLOCK_N)
        scores = tl.dot(q, tl.trans(k)) * sm_scale
        
        # Apply softcap if enabled
        if USE_SOFTCAP:
            scores = softcap * tl.tanh(scores / softcap)
        
        # Create attention mask
        # Sliding window mask: |m - n| <= window_size // 2
        # Plus causal mask if enabled: n <= m
        attn_mask = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int1)
        
        for i in range(BLOCK_M):
            m_pos = q_start + i
            for j in range(BLOCK_N):
                n_pos = kv_block_start + j
                
                # Window bounds
                in_window = True
                if IS_CAUSAL:
                    in_window = (n_pos <= m_pos) & (n_pos >= m_pos - window_size + 1)
                else:
                    half_w = window_size // 2
                    in_window = (n_pos >= m_pos - half_w) & (n_pos <= m_pos + half_w)
                
                # Sequence bounds
                valid = (m_pos < seq_len) & (n_pos < seq_len) & in_window
                
                attn_mask = tl.where(
                    (tl.arange(0, BLOCK_M)[:, None] == i) & 
                    (tl.arange(0, BLOCK_N)[None, :] == j),
                    valid, attn_mask
                )
        
        # Apply mask
        scores = tl.where(attn_mask, scores, float("-inf"))
        
        # Online softmax update
        # Compute row-wise max for numerical stability
        m_ij = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        
        # Correction factor for previously accumulated values
        alpha = tl.exp(m_i - m_new)
        
        # Compute softmax numerators
        p = tl.exp(scores - m_new[:, None])
        
        # Update running sum
        l_new = alpha * l_i + tl.sum(p, axis=1)
        
        # Load V: (BLOCK_N, BLOCK_D)
        v_ptrs = V + batch_idx * stride_vb + n_offs[:, None] * stride_vs + \
                 kv_head_idx * stride_vh + tl.arange(0, BLOCK_D)[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=k_mask, other=0.0).to(tl.float32)
        
        # Update accumulator: rescale old values and add new contribution
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        
        # Update stats
        m_i = m_new
        l_i = l_new
    
    # Final normalization
    acc = acc / l_i[:, None]
    
    # Compute log-sum-exp for backward
    lse = m_i + tl.log(l_i)
    
    # Store output
    o_ptrs = O + batch_idx * stride_ob + m_offs[:, None] * stride_os + \
             head_idx * stride_oh + tl.arange(0, BLOCK_D)[None, :] * stride_od
    o_mask = (m_offs[:, None] < seq_len) & (tl.arange(0, BLOCK_D)[None, :] < head_dim)
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=o_mask)
    
    # Store LSE
    lse_ptrs = LSE + batch_idx * stride_lseb + head_idx * stride_lseh + \
               m_offs * stride_lses
    lse_mask = m_offs < seq_len
    tl.store(lse_ptrs, lse, mask=lse_mask)


@triton.jit
def _swa_bwd_kernel(
    # Inputs
    Q, K, V, O,
    DO,             # Gradient of output
    LSE,            # Log-sum-exp from forward
    # Outputs
    DQ, DK, DV,
    # Dimensions
    batch_size,
    seq_len,
    num_heads,
    num_kv_heads,
    head_dim,
    window_size,
    sm_scale,
    softcap,
    # Strides (same as forward)
    stride_qb, stride_qs, stride_qh, stride_qd,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_vb, stride_vs, stride_vh, stride_vd,
    stride_ob, stride_os, stride_oh, stride_od,
    stride_dqb, stride_dqs, stride_dqh, stride_dqd,
    stride_dkb, stride_dks, stride_dkh, stride_dkd,
    stride_dvb, stride_dvs, stride_dvh, stride_dvd,
    stride_lseb, stride_lseh, stride_lses,
    # Config
    IS_CAUSAL: tl.constexpr,
    USE_SOFTCAP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Backward pass for sliding window attention.
    Uses recomputation to avoid storing full attention matrix.
    """
    seq_block_idx = tl.program_id(0)
    batch_head_idx = tl.program_id(1)
    
    batch_idx = batch_head_idx // num_heads
    head_idx = batch_head_idx % num_heads
    kv_head_idx = head_idx // (num_heads // num_kv_heads)
    
    m_offs = seq_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offs = tl.arange(0, BLOCK_D)
    
    # Load Q, dO, O for this block
    q_ptrs = Q + batch_idx * stride_qb + m_offs[:, None] * stride_qs + \
             head_idx * stride_qh + d_offs[None, :] * stride_qd
    q_mask = (m_offs[:, None] < seq_len) & (d_offs[None, :] < head_dim)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)
    
    do_ptrs = DO + batch_idx * stride_ob + m_offs[:, None] * stride_os + \
              head_idx * stride_oh + d_offs[None, :] * stride_od
    do = tl.load(do_ptrs, mask=q_mask, other=0.0).to(tl.float32)
    
    o_ptrs = O + batch_idx * stride_ob + m_offs[:, None] * stride_os + \
             head_idx * stride_oh + d_offs[None, :] * stride_od
    o = tl.load(o_ptrs, mask=q_mask, other=0.0).to(tl.float32)
    
    lse_ptrs = LSE + batch_idx * stride_lseb + head_idx * stride_lseh + \
               m_offs * stride_lses
    lse = tl.load(lse_ptrs, mask=m_offs < seq_len, other=0.0).to(tl.float32)
    
    # D = rowsum(dO * O)
    D = tl.sum(do * o, axis=1)
    
    # Initialize dQ accumulator
    dq = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    q_start = seq_block_idx * BLOCK_M
    
    if IS_CAUSAL:
        kv_start = tl.maximum(0, q_start - window_size + 1)
        kv_end = tl.minimum(seq_len, q_start + BLOCK_M)
    else:
        half_window = window_size // 2
        kv_start = tl.maximum(0, q_start - half_window)
        kv_end = tl.minimum(seq_len, q_start + BLOCK_M + half_window)
    
    for kv_block_start in range(kv_start, kv_end, BLOCK_N):
        n_offs = kv_block_start + tl.arange(0, BLOCK_N)
        
        # Load K, V
        k_ptrs = K + batch_idx * stride_kb + n_offs[:, None] * stride_ks + \
                 kv_head_idx * stride_kh + d_offs[None, :] * stride_kd
        k_mask = (n_offs[:, None] < seq_len) & (d_offs[None, :] < head_dim)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)
        
        v_ptrs = V + batch_idx * stride_vb + n_offs[:, None] * stride_vs + \
                 kv_head_idx * stride_vh + d_offs[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=k_mask, other=0.0).to(tl.float32)
        
        # Recompute attention scores
        scores = tl.dot(q, tl.trans(k)) * sm_scale
        
        if USE_SOFTCAP:
            scores = softcap * tl.tanh(scores / softcap)
        
        # Apply mask (same as forward)
        attn_mask = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int1)
        for i in range(BLOCK_M):
            m_pos = q_start + i
            for j in range(BLOCK_N):
                n_pos = kv_block_start + j
                if IS_CAUSAL:
                    in_window = (n_pos <= m_pos) & (n_pos >= m_pos - window_size + 1)
                else:
                    half_w = window_size // 2
                    in_window = (n_pos >= m_pos - half_w) & (n_pos <= m_pos + half_w)
                valid = (m_pos < seq_len) & (n_pos < seq_len) & in_window
                attn_mask = tl.where(
                    (tl.arange(0, BLOCK_M)[:, None] == i) & 
                    (tl.arange(0, BLOCK_N)[None, :] == j),
                    valid, attn_mask
                )
        
        scores = tl.where(attn_mask, scores, float("-inf"))
        
        # Recompute softmax
        p = tl.exp(scores - lse[:, None])
        
        # dP = dO @ V^T
        dp = tl.dot(do, tl.trans(v))
        
        # dS = P * (dP - D)
        ds = p * (dp - D[:, None])
        
        if USE_SOFTCAP:
            # Chain rule for softcap: d/dx[c * tanh(x/c)] = 1 - tanh^2(x/c)
            tanh_scores = tl.tanh(scores / softcap)
            ds = ds * (1 - tanh_scores * tanh_scores)
        
        ds = ds * sm_scale
        
        # Accumulate dQ
        dq += tl.dot(ds.to(k.dtype), k)
        
        # Compute dK, dV and atomic add
        dk = tl.dot(tl.trans(ds.to(q.dtype)), q)
        dv = tl.dot(tl.trans(p.to(do.dtype)), do)
        
        # Atomic add for dK, dV (accumulate across Q blocks)
        dk_ptrs = DK + batch_idx * stride_dkb + n_offs[:, None] * stride_dks + \
                  kv_head_idx * stride_dkh + d_offs[None, :] * stride_dkd
        dv_ptrs = DV + batch_idx * stride_dvb + n_offs[:, None] * stride_dvs + \
                  kv_head_idx * stride_dvh + d_offs[None, :] * stride_dvd
        
        tl.atomic_add(dk_ptrs, dk.to(DK.dtype.element_ty), mask=k_mask)
        tl.atomic_add(dv_ptrs, dv.to(DV.dtype.element_ty), mask=k_mask)
    
    # Store dQ
    dq_ptrs = DQ + batch_idx * stride_dqb + m_offs[:, None] * stride_dqs + \
              head_idx * stride_dqh + d_offs[None, :] * stride_dqd
    tl.store(dq_ptrs, dq.to(DQ.dtype.element_ty), mask=q_mask)


def triton_sliding_window_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int = 4096,
    causal: bool = True,
    softcap: float = 0.0,
    return_lse: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Triton sliding window attention with FlashAttention-2 style tiling.
    
    Args:
        q: Query tensor (batch, seq_len, num_heads, head_dim)
        k: Key tensor (batch, seq_len, num_kv_heads, head_dim)
        v: Value tensor (batch, seq_len, num_kv_heads, head_dim)
        window_size: Sliding window size
        causal: Use causal masking
        softcap: Softcap value (0 = disabled)
        return_lse: Return log-sum-exp for backward
        
    Returns:
        Tuple of (output, lse) where lse is None if not requested
    """
    assert q.is_cuda, "Input must be on CUDA"
    
    batch_size, seq_len, num_heads, head_dim = q.shape
    num_kv_heads = k.shape[2]
    
    assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
    
    # Ensure contiguous
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    
    # Allocate outputs
    o = torch.empty_like(q)
    lse = torch.empty(
        batch_size, num_heads, seq_len,
        device=q.device, dtype=torch.float32
    )
    
    # Scaling factor
    sm_scale = 1.0 / math.sqrt(head_dim)
    
    # Block sizes
    BLOCK_M = min(64, seq_len)
    BLOCK_N = min(64, seq_len)
    BLOCK_D = triton.next_power_of_2(head_dim)
    
    # Grid
    grid = (triton.cdiv(seq_len, BLOCK_M), batch_size * num_heads)
    
    _swa_fwd_kernel[grid](
        q, k, v, o, lse,
        batch_size, seq_len, num_heads, num_kv_heads, head_dim,
        window_size, sm_scale, softcap,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        IS_CAUSAL=causal,
        USE_SOFTCAP=softcap > 0,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )
    
    return o, lse if return_lse else None


def flash_attn_sliding_window(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: Tuple[int, int] = (-1, -1),
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    """
    Flash Attention compatible interface for sliding window attention.
    
    Args:
        q, k, v: Query, Key, Value tensors
        window_size: (left, right) window sizes. -1 = infinite
        softmax_scale: Optional custom scale
        causal: Causal masking
        
    Returns:
        Attention output
    """
    left, right = window_size
    
    if left == -1 and right == -1:
        # Full attention - use large window
        ws = q.shape[1] * 2
    elif left == -1:
        ws = right * 2
    elif right == -1:
        ws = left * 2
    else:
        ws = left + right + 1
    
    output, _ = triton_sliding_window_attention(q, k, v, ws, causal)
    return output


class FlashAttentionSlidingWindow(torch.nn.Module):
    """
    Sliding Window Attention module with FlashAttention-2 kernel.
    
    Supports:
    - Grouped Query Attention (GQA)
    - Causal masking
    - Softcapping
    - Mixed precision
    """
    
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        num_kv_heads: Optional[int] = None,
        window_size: int = 4096,
        causal: bool = True,
        softcap: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads or num_heads
        self.window_size = window_size
        self.causal = causal
        self.softcap = softcap
        self.dropout = dropout
        
        assert num_heads % self.num_kv_heads == 0
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            q: Query (batch, seq_len, num_heads, head_dim)
            k: Key (batch, seq_len, num_kv_heads, head_dim)
            v: Value (batch, seq_len, num_kv_heads, head_dim)
            attention_mask: Currently ignored (mask built into kernel)
            
        Returns:
            Attention output (batch, seq_len, num_heads, head_dim)
        """
        output, _ = triton_sliding_window_attention(
            q, k, v,
            window_size=self.window_size,
            causal=self.causal,
            softcap=self.softcap,
        )
        
        if self.training and self.dropout > 0:
            output = torch.nn.functional.dropout(output, p=self.dropout, training=True)
        
        return output
