"""
Hierarchical Timescale Controller (HTC)
=======================================
Multi-timescale processing for hierarchical reasoning.

Implements:
- Slow controller for long-term planning
- Fast actor for immediate execution
- Cross-timescale modulation

Features:
- Configurable timescale ratios
- Learned vs fixed gating
- Bidirectional information flow
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math

from ..config import HTCConfig, RSSMoDConfig
from ..kernels.triton import RMSNorm


class HTCGate(nn.Module):
    """
    Timescale gating mechanism.
    
    Controls information flow between hierarchical levels.
    """
    
    def __init__(
        self,
        d_model: int,
        gate_type: str = "learned",
    ):
        super().__init__()
        self.d_model = d_model
        self.gate_type = gate_type
        
        if gate_type == "learned":
            self.gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid(),
            )
        elif gate_type == "adaptive":
            self.query = nn.Linear(d_model, d_model)
            self.key = nn.Linear(d_model, d_model)
            self.scale = d_model ** -0.5
        else:
            # Fixed gate (alpha blending)
            self.alpha = nn.Parameter(torch.tensor(0.5))
    
    def forward(
        self,
        fast_state: torch.Tensor,
        slow_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gated combination of timescales.
        
        Args:
            fast_state: Fast timescale hidden state
            slow_state: Slow timescale hidden state
            
        Returns:
            Tuple of (gated_fast, gated_slow)
        """
        if self.gate_type == "learned":
            combined = torch.cat([fast_state, slow_state], dim=-1)
            gate = self.gate(combined)
            gated_fast = gate * fast_state + (1 - gate) * slow_state
            gated_slow = (1 - gate) * fast_state + gate * slow_state
            
        elif self.gate_type == "adaptive":
            # Cross-attention style gating
            q = self.query(fast_state)
            k = self.key(slow_state)
            attn = torch.sigmoid((q * k).sum(dim=-1, keepdim=True) * self.scale)
            gated_fast = attn * fast_state + (1 - attn) * slow_state
            gated_slow = slow_state  # Slow state unchanged
            
        else:
            # Fixed blending
            alpha = torch.sigmoid(self.alpha)
            gated_fast = alpha * fast_state + (1 - alpha) * slow_state
            gated_slow = slow_state
        
        return gated_fast, gated_slow


class TimescaleSSM(nn.Module):
    """
    SSM operating at a specific timescale.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        timescale: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.timescale = timescale
        
        # Input projection
        self.in_proj = nn.Linear(d_model, d_model * 2)
        
        # SSM parameters
        self.A = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.D = nn.Parameter(torch.ones(d_model))
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Norm
        self.norm = RMSNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass at this timescale.
        
        Only processes every `timescale` positions.
        """
        batch_size, seq_len, d_model = x.shape
        
        # Subsample input at this timescale
        if self.timescale > 1:
            # Process every timescale-th position
            indices = torch.arange(0, seq_len, self.timescale, device=x.device)
            x_sub = x[:, indices]
        else:
            x_sub = x
        
        sub_len = x_sub.shape[1]
        
        # Pre-norm
        x_sub = self.norm(x_sub)
        
        # Project
        xz = self.in_proj(x_sub)
        x_proj, gate = xz.chunk(2, dim=-1)
        
        # Initialize state
        if state is None:
            state = torch.zeros(
                batch_size, d_model, self.d_state,
                device=x.device, dtype=x.dtype
            )
        
        # Simple SSM scan
        outputs = []
        for t in range(sub_len):
            x_t = x_proj[:, t]
            
            # State update
            state = torch.tanh(state @ self.A.T + x_t.unsqueeze(-1) @ self.B.unsqueeze(0))
            
            # Output
            y_t = (state @ self.C.T).sum(dim=-1) + self.D * x_t
            outputs.append(y_t)
        
        output = torch.stack(outputs, dim=1)
        
        # Gate
        output = output * F.silu(gate)
        output = self.out_proj(output)
        
        # Interpolate back to full length if subsampled
        if self.timescale > 1:
            # Repeat each output for timescale positions
            output = output.unsqueeze(2).expand(-1, -1, self.timescale, -1)
            output = output.reshape(batch_size, -1, d_model)[:, :seq_len]
        
        return output, state


class HierarchicalTimescaleController(nn.Module):
    """
    Hierarchical Timescale Controller.
    
    Combines multiple SSMs operating at different timescales:
    - Fast: Every position (syntax, immediate context)
    - Medium: Every N positions (clause/sentence level)
    - Slow: Every M positions (paragraph/document level)
    
    Cross-timescale modulation allows high-level planning
    to influence low-level execution.
    """
    
    def __init__(
        self,
        d_model: int,
        config: Optional[HTCConfig] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.config = config or HTCConfig()
        
        self.num_timescales = self.config.num_timescales
        self.fast_timescale = self.config.fast_timescale
        self.slow_timescale = self.config.slow_timescale
        
        # Compute timescale sequence
        ratio = (self.slow_timescale / self.fast_timescale) ** (1 / (self.num_timescales - 1))
        self.timescales = [
            int(self.fast_timescale * (ratio ** i))
            for i in range(self.num_timescales)
        ]
        
        # SSMs at each timescale
        self.ssms = nn.ModuleList([
            TimescaleSSM(d_model, d_state=64, timescale=ts)
            for ts in self.timescales
        ])
        
        # Cross-timescale gates
        self.gates = nn.ModuleList([
            HTCGate(d_model, self.config.gate_type)
            for _ in range(self.num_timescales - 1)
        ])
        
        # Final combination
        self.combine = nn.Linear(d_model * self.num_timescales, d_model)
        
        # Normalization
        self.input_norm = RMSNorm(d_model)
        self.output_norm = RMSNorm(d_model)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        states: Optional[List[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass through hierarchical controller.
        
        Args:
            hidden_states: (batch, seq_len, d_model)
            states: Optional list of states for each timescale
            use_cache: Whether to return updated states
            
        Returns:
            Tuple of (output, new_states)
        """
        batch_size, seq_len, d_model = hidden_states.shape
        
        residual = hidden_states
        hidden_states = self.input_norm(hidden_states)
        
        # Initialize states if not provided
        if states is None:
            states = [None] * self.num_timescales
        
        # Process each timescale
        timescale_outputs = []
        new_states = []
        
        for i, (ssm, state) in enumerate(zip(self.ssms, states)):
            output, new_state = ssm(hidden_states, state)
            timescale_outputs.append(output)
            new_states.append(new_state)
        
        # Cross-timescale modulation (bottom-up: slow influences fast)
        modulated = timescale_outputs.copy()
        
        for i in range(self.num_timescales - 2, -1, -1):
            # Gate between timescale i and i+1
            fast = modulated[i]
            slow = modulated[i + 1]
            
            gated_fast, gated_slow = self.gates[i](fast, slow)
            modulated[i] = gated_fast
            modulated[i + 1] = gated_slow
        
        # Combine timescales
        combined = torch.cat(modulated, dim=-1)
        output = self.combine(combined)
        output = self.output_norm(output)
        
        # Residual
        output = output + residual
        
        return output, new_states if use_cache else None
    
    def allocate_cache(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> List[torch.Tensor]:
        """Allocate cache for all timescales."""
        return [
            torch.zeros(
                batch_size, self.d_model, 64,  # d_state
                device=device, dtype=dtype
            )
            for _ in range(self.num_timescales)
        ]


class MultiScaleFusion(nn.Module):
    """
    Fuses features from multiple timescales using attention.
    
    Alternative to HTCGate for more flexible fusion.
    """
    
    def __init__(
        self,
        d_model: int,
        num_scales: int,
        num_heads: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_scales = num_scales
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Query from fastest scale
        self.q_proj = nn.Linear(d_model, d_model)
        
        # Keys and values from all scales
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # Output
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self,
        scale_features: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Fuse features from multiple scales.
        
        Args:
            scale_features: List of (batch, seq_len, d_model) tensors
            
        Returns:
            Fused features (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = scale_features[0].shape
        
        # Query from fastest scale (most detailed)
        q = self.q_proj(scale_features[0])
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Stack all scales for keys and values
        all_scales = torch.stack(scale_features, dim=2)  # (B, S, num_scales, D)
        
        k = self.k_proj(all_scales)
        v = self.v_proj(all_scales)
        
        k = k.view(batch_size, seq_len, self.num_scales, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_scales, self.num_heads, self.head_dim)
        
        # Attention: query each position against all scales at that position
        # q: (B, S, H, D), k: (B, S, N, H, D) -> attn: (B, S, H, N)
        attn = torch.einsum("bshd,bsnhd->bshn", q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        # attn: (B, S, H, N), v: (B, S, N, H, D) -> out: (B, S, H, D)
        out = torch.einsum("bshn,bsnhd->bshd", attn, v)
        
        # Reshape and project
        out = out.reshape(batch_size, seq_len, d_model)
        out = self.out_proj(out)
        
        return out
