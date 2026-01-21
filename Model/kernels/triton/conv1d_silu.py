"""
Depthwise Conv1D + SiLU Activation Kernel
Fused convolution and activation for SSM input processing
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _conv1d_silu_kernel(
    x_ptr,        # [B, L, D] input
    weight_ptr,   # [D, K] depthwise conv weights
    bias_ptr,     # [D] bias
    y_ptr,        # [B, L, D] output
    B, L, D, K,
    stride_xb, stride_xl, stride_xd,
    stride_yb, stride_yl, stride_yd,
    BLOCK_SIZE_D: tl.constexpr,
):
    """
    Depthwise Conv1D: each channel has independent kernel
    SiLU activation: x * sigmoid(x)
    """
    
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_d = tl.program_id(2)
    
    offs_d = pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    mask_d = offs_d < D
    
    # Compute convolution output
    acc = tl.zeros([BLOCK_SIZE_D], dtype=tl.float32)
    
    # Apply kernel (causal: only look at current and past)
    for k in range(K):
        l_offset = pid_l - k
        
        # Causal masking: only valid positions
        if l_offset >= 0:
            x_offset = pid_b * stride_xb + l_offset * stride_xl + offs_d * stride_xd
            x = tl.load(x_ptr + x_offset, mask=mask_d, other=0.0)
            
            # Load depthwise weight [D, K]
            w_offset = offs_d * K + k
            w = tl.load(weight_ptr + w_offset, mask=mask_d, other=0.0)
            
            acc += x * w
    
    # Add bias
    bias = tl.load(bias_ptr + offs_d, mask=mask_d, other=0.0)
    conv_out = acc + bias
    
    # SiLU activation: x * sigmoid(x)
    sig = tl.sigmoid(conv_out)
    y = conv_out * sig
    
    # Store output
    y_offset = pid_b * stride_yb + pid_l * stride_yl + offs_d * stride_yd
    tl.store(y_ptr + y_offset, y, mask=mask_d)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_D': 64}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_SIZE_D': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_D': 256}, num_warps=8, num_stages=2),
    ],
    key=['D'],
)
@triton.jit
def _conv1d_silu_kernel_optimized(
    x_ptr, weight_ptr, bias_ptr, y_ptr,
    B, L, D, K,
    stride_xb, stride_xl, stride_xd,
    stride_yb, stride_yl, stride_yd,
    BLOCK_SIZE_D: tl.constexpr,
):
    """Autotuned version"""
    _conv1d_silu_kernel(
        x_ptr, weight_ptr, bias_ptr, y_ptr,
        B, L, D, K,
        stride_xb, stride_xl, stride_xd,
        stride_yb, stride_yl, stride_yd,
        BLOCK_SIZE_D,
    )


def conv1d_silu(x, weight, bias, kernel_size=4):
    """
    Depthwise Conv1D with fused SiLU activation
    
    Args:
        x: [B, L, D] input sequence
        weight: [D, K] depthwise convolution weights
        bias: [D] bias term
        kernel_size: convolution kernel size
        
    Returns:
        y: [B, L, D] convolved and activated output
    """
    B, L, D = x.shape
    K = kernel_size
    
    # Allocate output
    y = torch.empty_like(x)
    
    # Launch grid
    grid = lambda META: (
        B,
        L,
        triton.cdiv(D, META['BLOCK_SIZE_D'])
    )
    
    _conv1d_silu_kernel_optimized[grid](
        x, weight, bias, y,
        B, L, D, K,
        x.stride(0), x.stride(1), x.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
    )
    
    return y
