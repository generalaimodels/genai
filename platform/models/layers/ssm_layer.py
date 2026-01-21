"""
Selective State Space Model (SSM) Layer
=======================================
Mamba-2 style SSM with input-dependent dynamics.

Features:
- Selective gating mechanism (Δ, B, C from input)
- Efficient parallel scan for training
- O(1) recurrent mode for inference
- Conv1D pre-processing for local context
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from dataclasses import dataclass
import math

from ..config import SSMConfig, RSSMoDConfig
from ..kernels.triton import (
    triton_selective_scan,
    triton_rmsnorm,
    SelectiveScanFn,
    RMSNorm,
)


class Mamba2Block(nn.Module):
    """
    Mamba-2 Selective State Space Block.
    
    Architecture:
        x -> norm -> [in_proj] -> [conv1d] -> [silu] -> [ssm] -> [out_proj] -> residual
                     ↓                                  ↓
                   (Δ, B, C)                         (gate)
    
    Key features:
    - Input-dependent state transition (selective mechanism)
    - Discretized continuous SSM (ZOH discretization)
    - Parallel scan for efficient training
    - Recurrent mode for O(1) inference per token
    """
    
    def __init__(
        self,
        d_model: int,
        config: Optional[SSMConfig] = None,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.config = config or SSMConfig()
        self.layer_idx = layer_idx
        
        # Derived dimensions
        self.d_state = self.config.d_state
        self.d_conv = self.config.d_conv
        self.expand = self.config.expand_factor
        self.d_inner = int(self.expand * d_model)
        
        # Determine dt_rank
        if isinstance(self.config.dt_rank, str) and self.config.dt_rank == "auto":
            self.dt_rank = max(1, d_model // 16)
        else:
            self.dt_rank = int(self.config.dt_rank)
        
        # Input projection: x -> (z, x, Δ, B, C)
        # z: gate, x: input to SSM, Δ: timestep, B: input matrix, C: output matrix
        self.in_proj = nn.Linear(
            d_model,
            self.d_inner * 2 + self.dt_rank + self.d_state * 2,
            bias=self.config.bias,
        )
        
        # Local convolution for short-range context
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=self.d_conv,
            padding=self.d_conv - 1,
            groups=self.d_inner,
            bias=self.config.conv_bias,
        )
        
        # SSM parameters
        # A is learnable log-space diagonal (for stability)
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).expand(self.d_inner, -1)  # (d_inner, d_state)
        self.A_log = nn.Parameter(torch.log(A))
        
        # D skip connection
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Δ projection
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # Initialize dt_proj bias
        dt_init_std = self.dt_rank ** -0.5 * self.config.dt_scale
        if self.config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        else:
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        
        # Constrain dt_proj bias to [dt_min, dt_max]
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(self.config.dt_max) - math.log(self.config.dt_min))
            + math.log(self.config.dt_min)
        ).clamp(min=self.config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=self.config.bias)
        
        # Pre-norm
        self.norm = RMSNorm(d_model)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass.
        
        Args:
            hidden_states: (batch, seq_len, d_model)
            cache: Optional tuple of (conv_state, ssm_state) for inference
            use_cache: Whether to return updated cache
            
        Returns:
            Tuple of (output, new_cache)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Pre-norm
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        
        # Input projection
        xz = self.in_proj(hidden_states)
        
        # Split projections
        z, x, dt_raw, B, C = xz.split(
            [self.d_inner, self.d_inner, self.dt_rank, self.d_state, self.d_state],
            dim=-1
        )
        
        # Conv1D
        if cache is not None:
            # Inference mode: use cached conv state
            conv_state, ssm_state = cache
            
            # Update conv state
            conv_state = torch.cat([conv_state[:, :, 1:], x.transpose(1, 2)], dim=-1)
            x = self.conv1d(conv_state)[..., :seq_len].transpose(1, 2)
        else:
            # Training mode: full conv
            x = x.transpose(1, 2)  # (batch, d_inner, seq_len)
            x = self.conv1d(x)[..., :seq_len].transpose(1, 2)
        
        # Activation
        x = F.silu(x)
        
        # Project Δ
        dt = self.dt_proj(dt_raw)
        dt = F.softplus(dt)  # Ensure positive
        
        # Get A from log-space
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        
        # Selective scan
        if cache is not None and seq_len == 1:
            # Recurrent mode for single-token inference
            y, ssm_state = self._recurrent_step(x, dt, A, B, C, ssm_state)
        else:
            # Parallel scan for training/prefill
            y, final_state = self._parallel_scan(x, dt, A, B, C)
            ssm_state = final_state if use_cache else None
        
        # Gate and project
        y = y * F.silu(z)
        output = self.out_proj(y)
        
        # Residual
        output = output + residual
        
        # Build cache
        if use_cache:
            if cache is None:
                # Initialize conv state
                conv_state = x.transpose(1, 2)[:, :, -self.d_conv + 1:]
                if conv_state.shape[-1] < self.d_conv - 1:
                    conv_state = F.pad(conv_state, (self.d_conv - 1 - conv_state.shape[-1], 0))
            new_cache = (conv_state, ssm_state)
        else:
            new_cache = None
        
        return output, new_cache
    
    def _parallel_scan(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parallel scan implementation for training.
        
        Uses fused Triton kernel for efficiency.
        """
        batch_size, seq_len, d_inner = x.shape
        
        # Use Triton kernel
        if self.config.use_fast_path and x.is_cuda:
            y, final_state = triton_selective_scan(
                x, dt, A, B, C, self.D, None, return_last_state=True
            )
        else:
            # Fallback PyTorch implementation
            y, final_state = self._pytorch_scan(x, dt, A, B, C)
        
        return y, final_state
    
    def _recurrent_step(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single recurrent step for O(1) inference.
        
        Args:
            x: (batch, 1, d_inner)
            dt: (batch, 1, d_inner)
            A: (d_inner, d_state)
            B: (batch, 1, d_state)
            C: (batch, 1, d_state)
            ssm_state: (batch, d_inner, d_state)
        """
        batch_size = x.shape[0]
        
        x = x.squeeze(1)  # (batch, d_inner)
        dt = dt.squeeze(1)  # (batch, d_inner)
        B = B.squeeze(1)  # (batch, d_state)
        C = C.squeeze(1)  # (batch, d_state)
        
        # Discretize
        # A_bar = exp(dt * A)
        dA = torch.einsum("bd,dn->bdn", dt, A)  # (batch, d_inner, d_state)
        A_bar = torch.exp(dA)
        
        # B_bar = dt * B
        dB = torch.einsum("bd,bn->bdn", dt, B)  # (batch, d_inner, d_state)
        
        # State update: h = A_bar * h + B_bar * x
        ssm_state = A_bar * ssm_state + dB * x.unsqueeze(-1)
        
        # Output: y = C @ h + D * x
        y = torch.einsum("bdn,bn->bd", ssm_state, C) + self.D * x
        
        return y.unsqueeze(1), ssm_state
    
    def _pytorch_scan(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pure PyTorch fallback implementation.
        """
        batch_size, seq_len, d_inner = x.shape
        d_state = A.shape[1]
        
        # Initialize hidden state
        h = torch.zeros(batch_size, d_inner, d_state, device=x.device, dtype=x.dtype)
        
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, d_inner)
            dt_t = dt[:, t, :]  # (batch, d_inner)
            B_t = B[:, t, :]  # (batch, d_state)
            C_t = C[:, t, :]  # (batch, d_state)
            
            # Discretize
            dA = torch.einsum("bd,dn->bdn", dt_t, A)
            A_bar = torch.exp(dA)
            dB = torch.einsum("bd,bn->bdn", dt_t, B_t)
            
            # Update state
            h = A_bar * h + dB * x_t.unsqueeze(-1)
            
            # Output
            y_t = torch.einsum("bdn,bn->bd", h, C_t) + self.D * x_t
            outputs.append(y_t)
        
        y = torch.stack(outputs, dim=1)
        
        return y, h


class SSMLayer(nn.Module):
    """
    Complete SSM layer with optional normalization and dropout.
    
    Wraps Mamba2Block with additional functionality.
    """
    
    def __init__(
        self,
        d_model: int,
        config: Optional[SSMConfig] = None,
        dropout: float = 0.0,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.ssm = Mamba2Block(d_model, config, layer_idx)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass with optional dropout."""
        output, new_cache = self.ssm(hidden_states, cache, use_cache)
        output = self.dropout(output)
        return output, new_cache
    
    def allocate_cache(
        self,
        batch_size: int,
        max_seq_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Allocate inference cache.
        
        Returns:
            Tuple of (conv_state, ssm_state)
        """
        conv_state = torch.zeros(
            batch_size,
            self.ssm.d_inner,
            self.ssm.d_conv - 1,
            device=device,
            dtype=dtype,
        )
        
        ssm_state = torch.zeros(
            batch_size,
            self.ssm.d_inner,
            self.ssm.d_state,
            device=device,
            dtype=dtype,
        )
        
        return conv_state, ssm_state
