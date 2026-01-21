"""
Fused RMSNorm + Linear Projection Kernel
Eliminates intermediate materialization, reduces HBM round-trips from 3 to 1
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_linear_kernel(
    x_ptr,        # [B, L, D_in] input
    weight_ptr,   # [D_in] RMSNorm weight
    W_ptr,        # [D_out, D_in] linear weight
    y_ptr,        # [B, L, D_out] output
    B, L, D_in, D_out,
    eps: tl.constexpr,
    stride_xb, stride_xl, stride_xd,
    stride_yb, stride_yl, stride_yd,
    BLOCK_SIZE_D_IN: tl.constexpr,
    BLOCK_SIZE_D_OUT: tl.constexpr,
):
    """
    Fused RMSNorm + Linear:
    1. RMS = sqrt(mean(x^2) + eps)
    2. x_norm = x / RMS * weight
    3. y = x_norm @ W^T
    """
    
    # Block indices
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_d_out = tl.program_id(2)
    
    # Output dimension offsets
    offs_d_out = pid_d_out * BLOCK_SIZE_D_OUT + tl.arange(0, BLOCK_SIZE_D_OUT)
    mask_d_out = offs_d_out < D_out
    
    # Input dimension offsets
    offs_d_in = tl.arange(0, BLOCK_SIZE_D_IN)
    mask_d_in = offs_d_in < D_in
    
    # Load input vector
    x_offset = pid_b * stride_xb + pid_l * stride_xl + offs_d_in * stride_xd
    x = tl.load(x_ptr + x_offset, mask=mask_d_in, other=0.0)
    
    # Compute RMS
    x_sq = x * x
    rms = tl.sqrt(tl.sum(x_sq) / D_in + eps)
    
    # Load RMSNorm weight and normalize
    weight = tl.load(weight_ptr + offs_d_in, mask=mask_d_in, other=1.0)
    x_norm = (x / rms) * weight
    
    # Linear projection: accumulator for matrix multiply
    acc = tl.zeros([BLOCK_SIZE_D_OUT], dtype=tl.float32)
    
    # Iterate over input dimension in chunks
    for d_chunk in range(0, tl.cdiv(D_in, BLOCK_SIZE_D_IN)):
        offs_d = d_chunk * BLOCK_SIZE_D_IN + tl.arange(0, BLOCK_SIZE_D_IN)
        mask_d = offs_d < D_in
        
        # Load weight matrix chunk [D_out, D_in]
        W_offset = offs_d_out[:, None] * D_in + offs_d[None, :]
        W = tl.load(W_ptr + W_offset, mask=mask_d_out[:, None] & mask_d[None, :], other=0.0)
        
        # Load normalized input chunk
        if d_chunk == 0:
            x_chunk = x_norm
        else:
            x_offset_chunk = pid_b * stride_xb + pid_l * stride_xl + offs_d * stride_xd
            x_chunk = tl.load(x_ptr + x_offset_chunk, mask=mask_d, other=0.0)
            weight_chunk = tl.load(weight_ptr + offs_d, mask=mask_d, other=1.0)
            x_chunk = (x_chunk / rms) * weight_chunk
        
        # Accumulate: y = W @ x_norm
        acc += tl.sum(W * x_chunk[None, :], axis=1)
    
    # Store output
    y_offset = pid_b * stride_yb + pid_l * stride_yl + offs_d_out * stride_yd
    tl.store(y_ptr + y_offset, acc, mask=mask_d_out)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_D_IN': 128, 'BLOCK_SIZE_D_OUT': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_D_IN': 256, 'BLOCK_SIZE_D_OUT': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE_D_IN': 64, 'BLOCK_SIZE_D_OUT': 32}, num_warps=2, num_stages=4),
    ],
    key=['D_in', 'D_out'],
)
@triton.jit
def _rms_norm_linear_kernel_optimized(
    x_ptr, weight_ptr, W_ptr, y_ptr,
    B, L, D_in, D_out, eps: tl.constexpr,
    stride_xb, stride_xl, stride_xd,
    stride_yb, stride_yl, stride_yd,
    BLOCK_SIZE_D_IN: tl.constexpr,
    BLOCK_SIZE_D_OUT: tl.constexpr,
):
    """Autotuned version"""
    _rms_norm_linear_kernel(
        x_ptr, weight_ptr, W_ptr, y_ptr,
        B, L, D_in, D_out, eps,
        stride_xb, stride_xl, stride_xd,
        stride_yb, stride_yl, stride_yd,
        BLOCK_SIZE_D_IN, BLOCK_SIZE_D_OUT,
    )


def rms_norm_linear(x, weight, W, eps=1e-6):
    """
    Fused RMSNorm + Linear Projection
    
    Args:
        x: [B, L, D_in] input tensor
        weight: [D_in] RMSNorm scaling weight
        W: [D_out, D_in] linear projection matrix
        eps: numerical stability constant
        
    Returns:
        y: [B, L, D_out] normalized and projected output
    """
    B, L, D_in = x.shape
    D_out = W.shape[0]
    
    # Allocate output
    y = torch.empty(B, L, D_out, device=x.device, dtype=x.dtype)
    
    # Launch grid: (batch, sequence, output_features)
    grid = lambda META: (
        B,
        L,
        triton.cdiv(D_out, META['BLOCK_SIZE_D_OUT'])
    )
    
    _rms_norm_linear_kernel_optimized[grid](
        x, weight, W, y,
        B, L, D_in, D_out, eps,
        x.stride(0), x.stride(1), x.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
    )
    
    return y
