"""
Fused RMSNorm Triton Kernel
============================
IO-Aware implementation with forward/backward fusion.

Performance targets:
- Zero global memory round-trips for normalization
- Fused weight multiplication
- BF16/FP16/FP32 precision support
- Ampere+ optimized tiling

Mathematical formulation:
    y = (x / sqrt(mean(x^2) + eps)) * weight
"""

import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


@triton.jit
def _rmsnorm_fwd_kernel(
    X,              # Input tensor pointer
    Y,              # Output tensor pointer
    W,              # Weight tensor pointer
    RMS,            # RMS values output (for backward)
    stride_x_row,   # Row stride for X
    stride_y_row,   # Row stride for Y
    N,              # Number of columns (hidden dim)
    eps,            # Epsilon for numerical stability
    BLOCK_N: tl.constexpr,  # Block size for columns
):
    """
    Forward pass RMSNorm kernel.
    
    Grid: (num_rows,)
    Each program instance processes one row.
    """
    row_idx = tl.program_id(0)
    
    # Pointers to current row
    x_ptr = X + row_idx * stride_x_row
    y_ptr = Y + row_idx * stride_y_row
    
    # Compute mean of squares using online algorithm
    # Two-pass for numerical stability with large hidden dims
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    
    # Load input row
    x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Compute sum of squares
    x_sq = x * x
    sum_sq = tl.sum(x_sq, axis=0)
    
    # Compute RMS
    mean_sq = sum_sq / N
    rms = tl.sqrt(mean_sq + eps)
    
    # Store RMS for backward pass
    tl.store(RMS + row_idx, rms)
    
    # Normalize
    x_norm = x / rms
    
    # Load weight and apply
    w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)
    y = x_norm * w
    
    # Store output
    tl.store(y_ptr + cols, y.to(tl.float16) if Y.dtype.element_ty == tl.float16 
             else y.to(tl.bfloat16) if Y.dtype.element_ty == tl.bfloat16 
             else y, mask=mask)


@triton.jit
def _rmsnorm_fwd_kernel_large(
    X,              # Input tensor pointer
    Y,              # Output tensor pointer
    W,              # Weight tensor pointer
    RMS,            # RMS values output (for backward)
    stride_x_row,   # Row stride for X
    stride_y_row,   # Row stride for Y
    N,              # Number of columns (hidden dim)
    eps,            # Epsilon for numerical stability
    BLOCK_N: tl.constexpr,  # Block size for columns
):
    """
    Forward pass RMSNorm kernel for large hidden dimensions.
    Uses block-wise accumulation for hidden dims > BLOCK_N.
    
    Grid: (num_rows,)
    """
    row_idx = tl.program_id(0)
    
    # Pointers to current row
    x_ptr = X + row_idx * stride_x_row
    y_ptr = Y + row_idx * stride_y_row
    
    # First pass: compute sum of squares
    sum_sq = tl.zeros([1], dtype=tl.float32)
    
    for block_start in range(0, N, BLOCK_N):
        cols = block_start + tl.arange(0, BLOCK_N)
        mask = cols < N
        x_block = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(x_block * x_block, axis=0)
    
    # Compute RMS
    mean_sq = sum_sq / N
    rms = tl.sqrt(mean_sq + eps)
    
    # Store RMS for backward
    tl.store(RMS + row_idx, rms)
    
    # Second pass: normalize and apply weight
    for block_start in range(0, N, BLOCK_N):
        cols = block_start + tl.arange(0, BLOCK_N)
        mask = cols < N
        
        x_block = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w_block = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)
        
        y_block = (x_block / rms) * w_block
        
        tl.store(y_ptr + cols, y_block.to(tl.float16) if Y.dtype.element_ty == tl.float16
                 else y_block.to(tl.bfloat16) if Y.dtype.element_ty == tl.bfloat16
                 else y_block, mask=mask)


@triton.jit
def _rmsnorm_bwd_kernel(
    DY,             # Gradient of output
    X,              # Input tensor
    W,              # Weight tensor
    RMS,            # Stored RMS values from forward
    DX,             # Gradient of input (output)
    DW,             # Gradient of weight (output, atomic add)
    stride_dy_row,
    stride_x_row,
    stride_dx_row,
    N,              # Hidden dimension
    num_rows,       # Number of rows
    eps,
    BLOCK_N: tl.constexpr,
    HAS_DW: tl.constexpr,  # Whether to compute weight gradient
):
    """
    Backward pass RMSNorm kernel.
    
    Computes:
        dx = (1/rms) * (dy * w - x * mean(dy * w * x / rms^2))
        dw = sum(dy * x / rms) across batch
    """
    row_idx = tl.program_id(0)
    
    dy_ptr = DY + row_idx * stride_dy_row
    x_ptr = X + row_idx * stride_x_row
    dx_ptr = DX + row_idx * stride_dx_row
    
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    
    # Load values
    dy = tl.load(dy_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)
    rms = tl.load(RMS + row_idx)
    
    # Compute x_norm
    x_norm = x / rms
    
    # Gradient of weight (if needed)
    if HAS_DW:
        dw_local = dy * x_norm
        # Atomic add to accumulate across rows
        tl.atomic_add(DW + cols, dw_local, mask=mask)
    
    # Gradient of input
    # d(x/rms) = (1/rms) * dx - x/(rms^3) * d(rms)
    # d(rms) = (1/(2*rms)) * d(mean(x^2)) = (1/(2*rms)) * (2*x/N) * dx = x/(N*rms) * dx
    
    dy_w = dy * w
    grad_x_norm = dy_w / rms
    
    # Correction term: -x * mean(dy * w * x) / rms^3
    c = tl.sum(dy_w * x_norm, axis=0) / N
    dx = grad_x_norm - x_norm * c
    
    tl.store(dx_ptr + cols, dx.to(tl.float16) if DX.dtype.element_ty == tl.float16
             else dx.to(tl.bfloat16) if DX.dtype.element_ty == tl.bfloat16
             else dx, mask=mask)


@triton.jit
def _rmsnorm_bwd_kernel_large(
    DY,
    X,
    W,
    RMS,
    DX,
    DW,
    stride_dy_row,
    stride_x_row,
    stride_dx_row,
    N,
    num_rows,
    eps,
    BLOCK_N: tl.constexpr,
    HAS_DW: tl.constexpr,
):
    """Backward pass for large hidden dimensions."""
    row_idx = tl.program_id(0)
    
    dy_ptr = DY + row_idx * stride_dy_row
    x_ptr = X + row_idx * stride_x_row
    dx_ptr = DX + row_idx * stride_dx_row
    
    rms = tl.load(RMS + row_idx)
    
    # First pass: compute correction coefficient
    c = tl.zeros([1], dtype=tl.float32)
    
    for block_start in range(0, N, BLOCK_N):
        cols = block_start + tl.arange(0, BLOCK_N)
        mask = cols < N
        
        dy = tl.load(dy_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)
        
        x_norm = x / rms
        c += tl.sum(dy * w * x_norm, axis=0)
    
    c = c / N
    
    # Second pass: compute gradients
    for block_start in range(0, N, BLOCK_N):
        cols = block_start + tl.arange(0, BLOCK_N)
        mask = cols < N
        
        dy = tl.load(dy_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)
        
        x_norm = x / rms
        
        if HAS_DW:
            dw_local = dy * x_norm
            tl.atomic_add(DW + cols, dw_local, mask=mask)
        
        dx = (dy * w / rms) - (x_norm * c)
        
        tl.store(dx_ptr + cols, dx.to(tl.float16) if DX.dtype.element_ty == tl.float16
                 else dx.to(tl.bfloat16) if DX.dtype.element_ty == tl.bfloat16
                 else dx, mask=mask)


def triton_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
    out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Triton RMSNorm forward pass.
    
    Args:
        x: Input tensor of shape (*, hidden_size)
        weight: Weight tensor of shape (hidden_size,)
        eps: Epsilon for numerical stability
        out: Optional pre-allocated output tensor
        
    Returns:
        Tuple of (output tensor, rms values for backward)
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA"
    
    # Flatten to 2D
    orig_shape = x.shape
    x = x.view(-1, orig_shape[-1])
    num_rows, N = x.shape
    
    # Allocate outputs
    if out is None:
        y = torch.empty_like(x)
    else:
        y = out.view(-1, N)
    
    rms = torch.empty(num_rows, dtype=torch.float32, device=x.device)
    
    # Determine block size
    BLOCK_N = triton.next_power_of_2(N)
    
    # Launch kernel
    if BLOCK_N <= 8192:
        # Single-pass kernel for smaller hidden dims
        BLOCK_N = min(BLOCK_N, 8192)
        _rmsnorm_fwd_kernel[(num_rows,)](
            x, y, weight, rms,
            x.stride(0), y.stride(0),
            N, eps,
            BLOCK_N=BLOCK_N,
        )
    else:
        # Multi-pass kernel for large hidden dims
        BLOCK_N = 8192
        _rmsnorm_fwd_kernel_large[(num_rows,)](
            x, y, weight, rms,
            x.stride(0), y.stride(0),
            N, eps,
            BLOCK_N=BLOCK_N,
        )
    
    return y.view(orig_shape), rms


def triton_rmsnorm_backward(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    rms: torch.Tensor,
    eps: float = 1e-5,
    compute_weight_grad: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Triton RMSNorm backward pass.
    
    Args:
        dy: Gradient of output
        x: Input from forward pass
        weight: Weight tensor
        rms: RMS values from forward pass
        eps: Epsilon
        compute_weight_grad: Whether to compute weight gradient
        
    Returns:
        Tuple of (dx, dw) where dw is None if compute_weight_grad=False
    """
    assert dy.is_cuda and x.is_cuda and weight.is_cuda
    
    orig_shape = dy.shape
    dy = dy.view(-1, orig_shape[-1])
    x = x.view(-1, orig_shape[-1])
    num_rows, N = dy.shape
    
    dx = torch.empty_like(x)
    dw = torch.zeros_like(weight, dtype=torch.float32) if compute_weight_grad else None
    
    BLOCK_N = triton.next_power_of_2(N)
    
    if BLOCK_N <= 8192:
        BLOCK_N = min(BLOCK_N, 8192)
        _rmsnorm_bwd_kernel[(num_rows,)](
            dy, x, weight, rms, dx, dw,
            dy.stride(0), x.stride(0), dx.stride(0),
            N, num_rows, eps,
            BLOCK_N=BLOCK_N,
            HAS_DW=compute_weight_grad,
        )
    else:
        BLOCK_N = 8192
        _rmsnorm_bwd_kernel_large[(num_rows,)](
            dy, x, weight, rms, dx, dw,
            dy.stride(0), x.stride(0), dx.stride(0),
            N, num_rows, eps,
            BLOCK_N=BLOCK_N,
            HAS_DW=compute_weight_grad,
        )
    
    if dw is not None:
        dw = dw.to(weight.dtype)
    
    return dx.view(orig_shape), dw


class RMSNormFunction(torch.autograd.Function):
    """Autograd function wrapping Triton RMSNorm."""
    
    @staticmethod
    def forward(ctx, x, weight, eps):
        y, rms = triton_rmsnorm(x, weight, eps)
        ctx.save_for_backward(x, weight, rms)
        ctx.eps = eps
        return y
    
    @staticmethod
    def backward(ctx, dy):
        x, weight, rms = ctx.saved_tensors
        dx, dw = triton_rmsnorm_backward(dy, x, weight, rms, ctx.eps)
        return dx, dw, None


class RMSNorm(torch.nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Uses Triton kernels for efficient GPU execution.
    
    Args:
        hidden_size: Size of the last dimension
        eps: Epsilon for numerical stability
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return RMSNormFunction.apply(x, self.weight, self.eps)
    
    def extra_repr(self) -> str:
        return f"{self.hidden_size}, eps={self.eps}"
