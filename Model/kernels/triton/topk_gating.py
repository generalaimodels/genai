"""
Top-K Expert Gating with Load Balancing
Selects top-k experts per token with auxiliary loss for balanced routing
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _topk_gating_kernel(
    x_ptr,           # [B*L, D] input
    W_g_ptr,         # [D, N_experts] gating weights
    indices_ptr,     # [B*L, K] selected expert indices
    scores_ptr,      # [B*L, K] gating scores (normalized)
    load_ptr,        # [N_experts] load distribution (for balancing)
    B_L, D, N_experts, K: tl.constexpr,
    stride_xb, stride_xd,
    stride_wb, stride_wn,
    stride_ib, stride_ik,
    stride_sb, stride_sk,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Top-K gating: 
    1. Compute logits = x @ W_g
    2. Select top-K expert indices
    3. Normalize scores with softmax
    4. Track load distribution
    """
    
    pid = tl.program_id(0)
    
    # Load input vector
    offs_d = tl.arange(0, BLOCK_SIZE_D)
    mask_d = offs_d < D
    x_offset = pid * stride_xb + offs_d * stride_xd
    x = tl.load(x_ptr + x_offset, mask=mask_d, other=0.0)
    
    # Compute gating logits for all experts
    logits = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
    
    for d_chunk in range(0, tl.cdiv(D, BLOCK_SIZE_D)):
        offs_d_chunk = d_chunk * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
        mask_d_chunk = offs_d_chunk < D
        
        # Load input chunk
        if d_chunk > 0:
            x_offset_chunk = pid * stride_xb + offs_d_chunk * stride_xd
            x_chunk = tl.load(x_ptr + x_offset_chunk, mask=mask_d_chunk, other=0.0)
        else:
            x_chunk = x
        
        # Load weight matrix [D, N_experts]
        offs_n = tl.arange(0, BLOCK_SIZE_N)
        mask_n = offs_n < N_experts
        W_offset = offs_d_chunk[:, None] * stride_wb + offs_n[None, :] * stride_wn
        W = tl.load(W_g_ptr + W_offset, mask=mask_d_chunk[:, None] & mask_n[None, :], other=0.0)
        
        # Accumulate: logits = x @ W_g
        logits += tl.sum(x_chunk[:, None] * W, axis=0)
    
    # Top-K selection using bubble sort (small K)
    # For K=2: find max, then second max
    indices = tl.zeros([K], dtype=tl.int32)
    scores = tl.zeros([K], dtype=tl.float32)
    
    logits_copy = logits  # Working copy
    mask_n_full = tl.arange(0, BLOCK_SIZE_N) < N_experts
    
    for k in range(K):
        # Find max
        max_val = tl.max(tl.where(mask_n_full, logits_copy, float('-inf')))
        max_idx = tl.argmax(tl.where(mask_n_full, logits_copy, float('-inf')), axis=0)
        
        indices[k] = max_idx
        scores[k] = max_val
        
        # Mask out selected expert
        logits_copy = tl.where(tl.arange(0, BLOCK_SIZE_N) == max_idx, 
                               float('-inf'), logits_copy)
    
    # Softmax normalization over selected K experts
    scores_exp = tl.exp(scores - tl.max(scores))
    scores_norm = scores_exp / tl.sum(scores_exp)
    
    # Store indices and normalized scores
    for k in range(K):
        idx_offset = pid * stride_ib + k * stride_ik
        score_offset = pid * stride_sb + k * stride_sk
        tl.store(indices_ptr + idx_offset, indices[k])
        tl.store(scores_ptr + score_offset, scores_norm[k])
        
        # Update load counter (atomic add)
        tl.atomic_add(load_ptr + indices[k], 1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_D': 128, 'BLOCK_SIZE_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_D': 256, 'BLOCK_SIZE_N': 128}, num_warps=8, num_stages=2),
    ],
    key=['D', 'N_experts'],
)
@triton.jit
def _topk_gating_kernel_optimized(
    x_ptr, W_g_ptr, indices_ptr, scores_ptr, load_ptr,
    B_L, D, N_experts, K: tl.constexpr,
    stride_xb, stride_xd,
    stride_wb, stride_wn,
    stride_ib, stride_ik,
    stride_sb, stride_sk,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Autotuned version"""
    _topk_gating_kernel(
        x_ptr, W_g_ptr, indices_ptr, scores_ptr, load_ptr,
        B_L, D, N_experts, K,
        stride_xb, stride_xd,
        stride_wb, stride_wn,
        stride_ib, stride_ik,
        stride_sb, stride_sk,
        BLOCK_SIZE_D, BLOCK_SIZE_N,
    )


def topk_gating(x, W_g, k=2):
    """
    Top-K Expert Gating
    
    Args:
        x: [B, L, D] or [B*L, D] input tokens
        W_g: [D, N_experts] gating weight matrix
        k: number of experts to activate per token
        
    Returns:
        indices: [B*L, K] selected expert indices
        scores: [B*L, K] normalized gating scores
        load: [N_experts] token count per expert (for load balancing)
    """
    # Flatten batch and sequence
    if x.dim() == 3:
        B, L, D = x.shape
        x = x.reshape(B * L, D)
    else:
        B_L, D = x.shape
    
    N_experts = W_g.shape[1]
    B_L = x.shape[0]
    
    # Allocate outputs
    indices = torch.empty(B_L, k, device=x.device, dtype=torch.int32)
    scores = torch.empty(B_L, k, device=x.device, dtype=torch.float32)
    load = torch.zeros(N_experts, device=x.device, dtype=torch.int32)
    
    # Launch kernel (one program per token)
    grid = (B_L,)
    
    _topk_gating_kernel_optimized[grid](
        x, W_g, indices, scores, load,
        B_L, D, N_experts, k,
        x.stride(0), x.stride(1),
        W_g.stride(0), W_g.stride(1),
        indices.stride(0), indices.stride(1),
        scores.stride(0), scores.stride(1),
    )
    
    return indices, scores, load
