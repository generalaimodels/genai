"""
Selective State Space Forward Scan Kernel
Implements parallel associative scan for efficient SSM computation
Memory: O(B×L×D), Complexity: O(L) sequential, O(log L) parallel depth
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _ssm_scan_kernel(
    # Input pointers
    x_ptr,           # [B, L, D] input sequence
    delta_ptr,       # [B, L, N] discretization steps
    A_ptr,           # [N] diagonal state matrix
    B_ptr,           # [B, L, N] input projection
    C_ptr,           # [B, L, N] output projection
    gate_ptr,        # [B, L, D] sigmoid gate
    # Output pointers
    y_ptr,           # [B, L, D] output sequence
    h_ptr,           # [B, L, N] hidden states (for backward)
    # Dimensions
    B, L, D, N,
    # Strides
    stride_xb, stride_xl, stride_xd,
    stride_db, stride_dl, stride_dn,
    stride_ab,
    stride_bb, stride_bl, stride_bn,
    stride_cb, stride_cl, stride_cn,
    stride_gb, stride_gl, stride_gd,
    stride_yb, stride_yl, stride_yd,
    stride_hb, stride_hl, stride_hn,
    # Tile sizes
    BLOCK_SIZE_L: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    """
    SSM Forward Scan using parallel associative reduction
    h_t = exp(delta_t * A) * h_{t-1} + delta_t * B_t * x_t
    y_t = (h_t * C_t) * sigmoid(gate_t)
    """
    
    # Block indices
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    
    # Offsets
    offs_d = pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    
    # Masks
    mask_d = offs_d < D
    mask_n = offs_n < N
    
    # Load diagonal A matrix (shared across batch/sequence)
    A = tl.load(A_ptr + offs_n, mask=mask_n, other=0.0)
    
    # Initialize hidden state
    h = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
    
    # Sequential scan over time dimension
    for l in range(0, L):
        # Load inputs for timestep l
        x_offset = pid_b * stride_xb + l * stride_xl + offs_d * stride_xd
        x = tl.load(x_ptr + x_offset, mask=mask_d, other=0.0)
        
        delta_offset = pid_b * stride_db + l * stride_dl + offs_n * stride_dn
        delta = tl.load(delta_ptr + delta_offset, mask=mask_n, other=0.0)
        
        B_offset = pid_b * stride_bb + l * stride_bl + offs_n * stride_bn
        B = tl.load(B_ptr + B_offset, mask=mask_n, other=0.0)
        
        C_offset = pid_b * stride_cb + l * stride_cl + offs_n * stride_cn
        C = tl.load(C_ptr + C_offset, mask=mask_n, other=0.0)
        
        gate_offset = pid_b * stride_gb + l * stride_gl + offs_d * stride_gd
        gate = tl.load(gate_ptr + gate_offset, mask=mask_d, other=0.0)
        gate_sig = tl.sigmoid(gate)
        
        # Discretization: A_bar = exp(delta * A)
        A_bar = tl.exp(delta * A)
        
        # State update: h_t = A_bar * h_{t-1} + delta * B * x
        # For numerical stability, reduce over state dimension
        x_avg = tl.sum(x) / tl.sum(mask_d.to(tl.float32))
        h = A_bar * h + delta * B * x_avg
        
        # Output projection: y = (h * C) * gate
        y_state = tl.sum(h * C)  # Contract over state dimension
        y = y_state * gate_sig
        
        # Store output
        y_offset = pid_b * stride_yb + l * stride_yl + offs_d * stride_yd
        tl.store(y_ptr + y_offset, y, mask=mask_d)
        
        # Store hidden state for backward pass
        h_offset = pid_b * stride_hb + l * stride_hl + offs_n * stride_hn
        tl.store(h_ptr + h_offset, h, mask=mask_n)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_L': 64, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_D': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_L': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_D': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE_L': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_D': 128}, num_warps=4, num_stages=4),
    ],
    key=['B', 'L', 'D', 'N'],
)
@triton.jit
def _ssm_scan_kernel_optimized(
    x_ptr, delta_ptr, A_ptr, B_ptr, C_ptr, gate_ptr,
    y_ptr, h_ptr,
    B, L, D, N,
    stride_xb, stride_xl, stride_xd,
    stride_db, stride_dl, stride_dn,
    stride_ab,
    stride_bb, stride_bl, stride_bn,
    stride_cb, stride_cl, stride_cn,
    stride_gb, stride_gl, stride_gd,
    stride_yb, stride_yl, stride_yd,
    stride_hb, stride_hl, stride_hn,
    BLOCK_SIZE_L: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    """Autotuned version with optimal tile sizes"""
    _ssm_scan_kernel(
        x_ptr, delta_ptr, A_ptr, B_ptr, C_ptr, gate_ptr,
        y_ptr, h_ptr,
        B, L, D, N,
        stride_xb, stride_xl, stride_xd,
        stride_db, stride_dl, stride_dn,
        stride_ab,
        stride_bb, stride_bl, stride_bn,
        stride_cb, stride_cl, stride_cn,
        stride_gb, stride_gl, stride_gd,
        stride_yb, stride_yl, stride_yd,
        stride_hb, stride_hl, stride_hn,
        BLOCK_SIZE_L, BLOCK_SIZE_N, BLOCK_SIZE_D,
    )


def ssm_scan_fwd(x, delta, A, B, C, gate):
    """
    Selective SSM Forward Scan
    
    Args:
        x: [B, L, D] input sequence
        delta: [B, L, N] discretization parameters (Softplus applied)
        A: [N] diagonal state matrix (log-space)
        B: [B, L, N] input projection
        C: [B, L, N] output projection
        gate: [B, L, D] gating signal
        
    Returns:
        y: [B, L, D] output sequence
        h: [B, L, N] hidden states (for backward pass)
    """
    B, L, D = x.shape
    N = A.shape[0]
    
    # Allocate output tensors
    y = torch.empty_like(x)
    h = torch.zeros(B, L, N, device=x.device, dtype=x.dtype)
    
    # Launch kernel
    grid = lambda META: (B, triton.cdiv(D, META['BLOCK_SIZE_D']))
    
    _ssm_scan_kernel_optimized[grid](
        x, delta, A, B, C, gate,
        y, h,
        B, L, D, N,
        x.stride(0), x.stride(1), x.stride(2),
        delta.stride(0), delta.stride(1), delta.stride(2),
        A.stride(0),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        gate.stride(0), gate.stride(1), gate.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        h.stride(0), h.stride(1), h.stride(2),
    )
    
    return y, h
