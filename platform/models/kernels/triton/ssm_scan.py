"""
Parallel Scan (Prefix Sum) Triton Kernel
=========================================
Work-efficient parallel scan using Blelloch algorithm.
Foundation for SSM state computation.

Complexity: O(L) work, O(log L) depth
Memory: State kept in SRAM, zero HBM round-trips per step

Supports:
- Associative scan for state updates: h_t = A_t * h_{t-1} + B_t * x_t
- Chunked processing for arbitrary sequence lengths
- FP32 accumulators with BF16/FP16 I/O
"""

import torch
import triton
import triton.language as tl
from typing import Tuple, Optional
import math


@triton.jit
def _parallel_scan_fwd_kernel(
    # Inputs
    A,              # Decay coefficients: (batch, seq_len, d_state) or (batch, seq_len)
    BX,             # B*x product: (batch, seq_len, d_state)
    # Outputs
    H,              # Output states: (batch, seq_len, d_state)
    # Initial state (optional)
    H0,             # Initial hidden state: (batch, d_state) or None
    # Dimensions
    batch_size,
    seq_len,
    d_state,
    # Strides
    stride_a_batch,
    stride_a_seq,
    stride_a_state,
    stride_bx_batch,
    stride_bx_seq,
    stride_bx_state,
    stride_h_batch,
    stride_h_seq,
    stride_h_state,
    stride_h0_batch,
    stride_h0_state,
    # Config
    HAS_H0: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_STATE: tl.constexpr,
):
    """
    Parallel associative scan forward pass.
    
    Computes: h_t = a_t * h_{t-1} + bx_t
    Using Blelloch parallel scan algorithm.
    
    Grid: (batch, cdiv(seq_len, CHUNK_SIZE), cdiv(d_state, BLOCK_STATE))
    """
    batch_idx = tl.program_id(0)
    chunk_idx = tl.program_id(1)
    state_block_idx = tl.program_id(2)
    
    # Compute offsets
    chunk_start = chunk_idx * CHUNK_SIZE
    state_start = state_block_idx * BLOCK_STATE
    
    chunk_offs = chunk_start + tl.arange(0, CHUNK_SIZE)
    state_offs = state_start + tl.arange(0, BLOCK_STATE)
    
    chunk_mask = chunk_offs < seq_len
    state_mask = state_offs < d_state
    mask_2d = chunk_mask[:, None] & state_mask[None, :]
    
    # Pointers
    a_ptrs = A + batch_idx * stride_a_batch + \
             chunk_offs[:, None] * stride_a_seq + \
             state_offs[None, :] * stride_a_state
    bx_ptrs = BX + batch_idx * stride_bx_batch + \
              chunk_offs[:, None] * stride_bx_seq + \
              state_offs[None, :] * stride_bx_state
    h_ptrs = H + batch_idx * stride_h_batch + \
             chunk_offs[:, None] * stride_h_seq + \
             state_offs[None, :] * stride_h_state
    
    # Load A and BX for this chunk
    a = tl.load(a_ptrs, mask=mask_2d, other=0.0).to(tl.float32)
    bx = tl.load(bx_ptrs, mask=mask_2d, other=0.0).to(tl.float32)
    
    # Load or initialize h_prev
    if HAS_H0 and chunk_idx == 0:
        h0_ptrs = H0 + batch_idx * stride_h0_batch + state_offs * stride_h0_state
        h_prev = tl.load(h0_ptrs, mask=state_mask, other=0.0).to(tl.float32)
    elif chunk_idx > 0:
        # Load last element of previous chunk
        prev_idx = chunk_start - 1
        prev_ptrs = H + batch_idx * stride_h_batch + \
                    prev_idx * stride_h_seq + \
                    state_offs * stride_h_state
        h_prev = tl.load(prev_ptrs, mask=state_mask, other=0.0).to(tl.float32)
    else:
        h_prev = tl.zeros([BLOCK_STATE], dtype=tl.float32)
    
    # Sequential scan within chunk (for correctness, will optimize with tree scan)
    # In production, use Blelloch tree reduction for O(log CHUNK_SIZE) depth
    h_out = tl.zeros([CHUNK_SIZE, BLOCK_STATE], dtype=tl.float32)
    
    for i in range(CHUNK_SIZE):
        if chunk_start + i < seq_len:
            a_i = a[i, :]
            bx_i = bx[i, :]
            h_prev = a_i * h_prev + bx_i
            h_out = tl.where(
                tl.arange(0, CHUNK_SIZE)[:, None] == i,
                h_prev[None, :],
                h_out
            )
    
    # Store output
    tl.store(h_ptrs, h_out.to(H.dtype.element_ty), mask=mask_2d)


@triton.jit
def _blelloch_scan_up_kernel(
    # Inputs/Outputs (in-place)
    AA,             # Running product of A: (batch, seq_len, d_state)
    ABX,            # Running sum of A*BX products: (batch, seq_len, d_state)
    # Dimensions
    batch_size,
    seq_len,
    d_state,
    # Strides
    stride_batch,
    stride_seq,
    stride_state,
    # Config
    level,          # Current level in tree
    BLOCK_STATE: tl.constexpr,
):
    """
    Blelloch scan up-sweep (reduce) phase.
    Computes partial products/sums in tree fashion.
    """
    batch_idx = tl.program_id(0)
    pair_idx = tl.program_id(1)  # Which pair to process at this level
    state_block_idx = tl.program_id(2)
    
    state_offs = state_block_idx * BLOCK_STATE + tl.arange(0, BLOCK_STATE)
    state_mask = state_offs < d_state
    
    # Compute indices for this level
    step = 1 << (level + 1)
    left_idx = pair_idx * step + (1 << level) - 1
    right_idx = pair_idx * step + step - 1
    
    if right_idx >= seq_len:
        return
    
    # Load left and right values
    left_aa = tl.load(
        AA + batch_idx * stride_batch + left_idx * stride_seq + state_offs * stride_state,
        mask=state_mask, other=1.0
    ).to(tl.float32)
    left_abx = tl.load(
        ABX + batch_idx * stride_batch + left_idx * stride_seq + state_offs * stride_state,
        mask=state_mask, other=0.0
    ).to(tl.float32)
    
    right_aa = tl.load(
        AA + batch_idx * stride_batch + right_idx * stride_seq + state_offs * stride_state,
        mask=state_mask, other=1.0
    ).to(tl.float32)
    right_abx = tl.load(
        ABX + batch_idx * stride_batch + right_idx * stride_seq + state_offs * stride_state,
        mask=state_mask, other=0.0
    ).to(tl.float32)
    
    # Combine: (a_r, b_r) o (a_l, b_l) = (a_r * a_l, a_r * b_l + b_r)
    new_aa = right_aa * left_aa
    new_abx = right_aa * left_abx + right_abx
    
    # Store to right position
    tl.store(
        AA + batch_idx * stride_batch + right_idx * stride_seq + state_offs * stride_state,
        new_aa.to(AA.dtype.element_ty), mask=state_mask
    )
    tl.store(
        ABX + batch_idx * stride_batch + right_idx * stride_seq + state_offs * stride_state,
        new_abx.to(ABX.dtype.element_ty), mask=state_mask
    )


@triton.jit
def _blelloch_scan_down_kernel(
    # Inputs/Outputs
    AA,
    ABX,
    H,              # Output hidden states
    H0,             # Initial hidden state
    # Dimensions
    batch_size,
    seq_len,
    d_state,
    # Strides
    stride_batch,
    stride_seq,
    stride_state,
    stride_h0_batch,
    stride_h0_state,
    # Config
    level,
    HAS_H0: tl.constexpr,
    BLOCK_STATE: tl.constexpr,
):
    """
    Blelloch scan down-sweep phase.
    Propagates values down the tree to compute final prefix sums.
    """
    batch_idx = tl.program_id(0)
    pair_idx = tl.program_id(1)
    state_block_idx = tl.program_id(2)
    
    state_offs = state_block_idx * BLOCK_STATE + tl.arange(0, BLOCK_STATE)
    state_mask = state_offs < d_state
    
    step = 1 << (level + 1)
    left_idx = pair_idx * step + (1 << level) - 1
    right_idx = pair_idx * step + step - 1
    
    if right_idx >= seq_len:
        return
    
    # Load values
    left_aa = tl.load(
        AA + batch_idx * stride_batch + left_idx * stride_seq + state_offs * stride_state,
        mask=state_mask, other=1.0
    ).to(tl.float32)
    left_abx = tl.load(
        ABX + batch_idx * stride_batch + left_idx * stride_seq + state_offs * stride_state,
        mask=state_mask, other=0.0
    ).to(tl.float32)
    
    right_aa = tl.load(
        AA + batch_idx * stride_batch + right_idx * stride_seq + state_offs * stride_state,
        mask=state_mask, other=1.0
    ).to(tl.float32)
    right_abx = tl.load(
        ABX + batch_idx * stride_batch + right_idx * stride_seq + state_offs * stride_state,
        mask=state_mask, other=0.0
    ).to(tl.float32)
    
    # Compute new left value
    new_left_aa = right_aa
    new_left_abx = right_abx
    
    # Compute new right value
    new_right_aa = right_aa * left_aa
    new_right_abx = right_aa * left_abx + right_abx
    
    # Store
    tl.store(
        AA + batch_idx * stride_batch + left_idx * stride_seq + state_offs * stride_state,
        new_left_aa.to(AA.dtype.element_ty), mask=state_mask
    )
    tl.store(
        ABX + batch_idx * stride_batch + left_idx * stride_seq + state_offs * stride_state,
        new_left_abx.to(ABX.dtype.element_ty), mask=state_mask
    )
    tl.store(
        AA + batch_idx * stride_batch + right_idx * stride_seq + state_offs * stride_state,
        new_right_aa.to(AA.dtype.element_ty), mask=state_mask
    )
    tl.store(
        ABX + batch_idx * stride_batch + right_idx * stride_seq + state_offs * stride_state,
        new_right_abx.to(ABX.dtype.element_ty), mask=state_mask
    )


@triton.jit  
def _finalize_scan_kernel(
    AA,             # Cumulative A products
    ABX,            # Cumulative B*X sums
    H,              # Output hidden states
    H0,             # Initial state
    batch_size,
    seq_len,
    d_state,
    stride_batch,
    stride_seq,
    stride_state,
    stride_h0_batch,
    stride_h0_state,
    HAS_H0: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_STATE: tl.constexpr,
):
    """
    Finalize scan by applying initial state.
    h_t = aa_t * h0 + abx_t
    """
    batch_idx = tl.program_id(0)
    seq_block = tl.program_id(1)
    state_block = tl.program_id(2)
    
    seq_offs = seq_block * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    state_offs = state_block * BLOCK_STATE + tl.arange(0, BLOCK_STATE)
    
    seq_mask = seq_offs < seq_len
    state_mask = state_offs < d_state
    mask_2d = seq_mask[:, None] & state_mask[None, :]
    
    # Load cumulative values
    aa_ptrs = AA + batch_idx * stride_batch + seq_offs[:, None] * stride_seq + \
              state_offs[None, :] * stride_state
    abx_ptrs = ABX + batch_idx * stride_batch + seq_offs[:, None] * stride_seq + \
               state_offs[None, :] * stride_state
    
    aa = tl.load(aa_ptrs, mask=mask_2d, other=1.0).to(tl.float32)
    abx = tl.load(abx_ptrs, mask=mask_2d, other=0.0).to(tl.float32)
    
    # Load or zero initial state
    if HAS_H0:
        h0_ptrs = H0 + batch_idx * stride_h0_batch + state_offs * stride_h0_state
        h0 = tl.load(h0_ptrs, mask=state_mask, other=0.0).to(tl.float32)
    else:
        h0 = tl.zeros([BLOCK_STATE], dtype=tl.float32)
    
    # Final hidden state: h_t = aa_t * h0 + abx_t
    h_out = aa * h0[None, :] + abx
    
    # Store
    h_ptrs = H + batch_idx * stride_batch + seq_offs[:, None] * stride_seq + \
             state_offs[None, :] * stride_state
    tl.store(h_ptrs, h_out.to(H.dtype.element_ty), mask=mask_2d)


def triton_parallel_scan(
    a: torch.Tensor,
    bx: torch.Tensor,
    h0: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Parallel associative scan using Triton.
    
    Computes: h_t = a_t * h_{t-1} + bx_t for t = 1...L
    
    Args:
        a: Decay coefficients of shape (batch, seq_len, d_state) or broadcastable
        bx: Input term (B @ x) of shape (batch, seq_len, d_state)
        h0: Optional initial hidden state (batch, d_state)
        
    Returns:
        Hidden states of shape (batch, seq_len, d_state)
    """
    assert bx.is_cuda, "Input must be on CUDA"
    
    batch_size, seq_len, d_state = bx.shape
    
    # Expand a if needed
    if a.dim() == 2:
        a = a.unsqueeze(-1).expand(-1, -1, d_state)
    
    # Ensure contiguous
    a = a.contiguous()
    bx = bx.contiguous()
    
    # Allocate output
    h = torch.empty_like(bx)
    
    # For short sequences, use simple sequential kernel
    if seq_len <= 256:
        CHUNK_SIZE = triton.next_power_of_2(seq_len)
        BLOCK_STATE = min(128, triton.next_power_of_2(d_state))
        
        grid = (batch_size, 1, triton.cdiv(d_state, BLOCK_STATE))
        
        _parallel_scan_fwd_kernel[grid](
            a, bx, h, h0 if h0 is not None else bx,  # Dummy for h0 if None
            batch_size, seq_len, d_state,
            a.stride(0), a.stride(1), a.stride(2),
            bx.stride(0), bx.stride(1), bx.stride(2),
            h.stride(0), h.stride(1), h.stride(2),
            h0.stride(0) if h0 is not None else 0,
            h0.stride(1) if h0 is not None else 0,
            HAS_H0=h0 is not None,
            CHUNK_SIZE=CHUNK_SIZE,
            BLOCK_STATE=BLOCK_STATE,
        )
    else:
        # Use full Blelloch algorithm for longer sequences
        h = _blelloch_parallel_scan(a, bx, h0)
    
    return h


def _blelloch_parallel_scan(
    a: torch.Tensor,
    bx: torch.Tensor,
    h0: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Full Blelloch parallel scan for long sequences.
    O(L) work, O(log L) depth.
    """
    batch_size, seq_len, d_state = bx.shape
    
    # Working arrays for scan
    aa = a.clone()  # Cumulative A products
    abx = bx.clone()  # Cumulative ABX sums
    
    BLOCK_STATE = min(128, triton.next_power_of_2(d_state))
    num_levels = int(math.ceil(math.log2(seq_len)))
    
    # Up-sweep (reduce) phase
    for level in range(num_levels):
        step = 1 << (level + 1)
        num_pairs = (seq_len + step - 1) // step
        
        grid = (batch_size, num_pairs, triton.cdiv(d_state, BLOCK_STATE))
        
        _blelloch_scan_up_kernel[grid](
            aa, abx,
            batch_size, seq_len, d_state,
            aa.stride(0), aa.stride(1), aa.stride(2),
            level,
            BLOCK_STATE=BLOCK_STATE,
        )
    
    # Root is now identity for inclusive scan
    # (For exclusive scan, would set root to identity here)
    
    # Down-sweep phase
    for level in range(num_levels - 2, -1, -1):
        step = 1 << (level + 1)
        num_pairs = (seq_len + step - 1) // step
        
        grid = (batch_size, num_pairs, triton.cdiv(d_state, BLOCK_STATE))
        
        _blelloch_scan_down_kernel[grid](
            aa, abx, None, h0 if h0 is not None else bx,
            batch_size, seq_len, d_state,
            aa.stride(0), aa.stride(1), aa.stride(2),
            h0.stride(0) if h0 is not None else 0,
            h0.stride(1) if h0 is not None else 0,
            level,
            HAS_H0=h0 is not None,
            BLOCK_STATE=BLOCK_STATE,
        )
    
    # Finalize with initial state
    BLOCK_SEQ = min(64, seq_len)
    h = torch.empty_like(bx)
    
    grid = (batch_size, triton.cdiv(seq_len, BLOCK_SEQ), triton.cdiv(d_state, BLOCK_STATE))
    
    _finalize_scan_kernel[grid](
        aa, abx, h, h0 if h0 is not None else bx,
        batch_size, seq_len, d_state,
        aa.stride(0), aa.stride(1), aa.stride(2),
        h0.stride(0) if h0 is not None else 0,
        h0.stride(1) if h0 is not None else 0,
        HAS_H0=h0 is not None,
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_STATE=BLOCK_STATE,
    )
    
    return h


def parallel_scan_forward(
    a: torch.Tensor,
    bx: torch.Tensor,
    h0: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Alias for triton_parallel_scan."""
    return triton_parallel_scan(a, bx, h0)


def parallel_scan_backward(
    dh: torch.Tensor,
    a: torch.Tensor,
    h: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Backward pass for parallel scan.
    
    Given dL/dh, computes dL/da, dL/dbx, dL/dh0.
    Uses reverse-mode parallel scan.
    """
    batch_size, seq_len, d_state = dh.shape
    
    # Reverse scan for gradient
    # d_a[t] = d_h[t] * h[t-1]
    # d_bx[t] = d_h[t]
    # Gradient flows backward: d_prev_h = a[t] * d_h[t]
    
    # Shift h to get h[t-1]
    h_prev = torch.cat([
        torch.zeros(batch_size, 1, d_state, device=h.device, dtype=h.dtype),
        h[:, :-1, :]
    ], dim=1)
    
    # d_a = dh * h_prev
    da = dh * h_prev
    
    # d_bx = dh (direct)
    dbx = dh.clone()
    
    # Reverse scan for gradient w.r.t. earlier timesteps
    # Using reverse of a for backward pass
    a_rev = a.flip(1)
    dh_rev = dh.flip(1)
    
    # Parallel scan of reversed gradients
    grad_accum = triton_parallel_scan(a_rev, dh_rev)
    grad_accum = grad_accum.flip(1)
    
    # Add accumulated gradients
    dbx = dbx + grad_accum - dh  # Subtract dh since it was double counted
    
    # Gradient w.r.t. initial state
    dh0 = grad_accum[:, 0, :]
    
    return da, dbx, dh0
