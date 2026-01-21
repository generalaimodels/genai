"""
Selective State Space Module
PyTorch wrapper for SSM Triton kernels with autograd support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..kernels.triton import ssm_scan_fwd, conv1d_silu


class SelectiveSSMFunction(torch.autograd.Function):
    """Custom autograd function for SSM forward/backward"""
    
    @staticmethod
    def forward(ctx, x, delta, A, B, C, gate):
        """
        Args:
            x: [B, L, D] input
            delta: [B, L, N] discretization
            A: [N] diagonal state matrix
            B: [B, L, N] input projection
            C: [B, L, N] output projection
            gate: [B, L, D] sigmoid gate
        """
        y, h = ssm_scan_fwd(x, delta, A, B, C, gate)
        ctx.save_for_backward(x, delta, A, B, C, gate, h)
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass (simplified: use PyTorch autograd for now)"""
        # In production: implement efficient backward SSM scan kernel
        # For now: checkpoint forward and use PyTorch autograd
        raise NotImplementedError("Backward pass requires custom gradient kernel")


class SelectiveSSM(nn.Module):
    """
    Selective State Space Layer (Mamba-style)
    
    Architecture:
    1. Linear projection to expand dim by factor E
    2. Conv1D with SiLU activation
    3. Content-aware discretization (delta, B, C)
    4. SSM scan with gating
    5. Output projection
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand
        
        # Input projections
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Conv1D parameters
        self.conv_weight = nn.Parameter(torch.randn(self.d_inner, d_conv))
        self.conv_bias = nn.Parameter(torch.zeros(self.d_inner))
        
        # SSM parameters
        self.A_log = nn.Parameter(torch.randn(d_state))  # Log-space for stability
        self.D = nn.Parameter(torch.ones(self.d_inner))  # Skip connection
        
        # Content-aware projections
        self.delta_proj = nn.Linear(self.d_inner, d_state, bias=True)
        self.B_proj = nn.Linear(self.d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(self.d_inner, d_state, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights following Mamba initialization"""
        # A matrix: initialized to be stable (negative log-space)
        nn.init.uniform_(self.A_log, -4.0, -1.0)
        
        # Delta projection: small init for stability
        nn.init.normal_(self.delta_proj.weight, std=0.02)
        nn.init.constant_(self.delta_proj.bias, 1.0)
    
    def forward(self, x):
        """
        Args:
            x: [B, L, D] input sequence
            
        Returns:
            y: [B, L, D] output sequence
        """
        B, L, D = x.shape
        
        # Input projection
        x_inner = self.in_proj(x)  # [B, L, D_inner * 2]
        x_ssm, x_gate = x_inner.chunk(2, dim=-1)  # Each [B, L, D_inner]
        
        # Conv1D with SiLU
        x_conv = conv1d_silu(
            x_ssm,
            self.conv_weight,
            self.conv_bias,
            kernel_size=self.d_conv
        )  # [B, L, D_inner]
        
        # Content-aware parameters
        delta = F.softplus(self.delta_proj(x_conv))  # [B, L, N]
        B_ssm = self.B_proj(x_conv)  # [B, L, N]
        C_ssm = self.C_proj(x_conv)  # [B, L, N]
        
        # A matrix (convert from log-space)
        A = -torch.exp(self.A_log)  # [N] negative for stability
        
        # SSM forward scan using Triton kernel
        # Note: In production, wrap with custom autograd
        # For now: use PyTorch native ops as fallback
        y_ssm = self._ssm_scan_native(x_conv, delta, A, B_ssm, C_ssm)
        
        # Gating
        y_gated = y_ssm * F.silu(x_gate)
        
        # Skip connection
        y_skip = y_gated + x_conv * self.D
        
        # Output projection
        y = self.out_proj(y_skip)
        
        return y
    
    def _ssm_scan_native(self, x, delta, A, B, C):
        """
        Native PyTorch SSM scan (fallback for autograd)
        In production: use SelectiveSSMFunction.apply()
        """
        B_batch, L, D_inner = x.shape
        N = A.shape[0]
        
        # Initialize hidden state
        h = torch.zeros(B_batch, N, device=x.device, dtype=x.dtype)
        
        outputs = []
        for t in range(L):
            # Discretization
            delta_t = delta[:, t, :]  # [B, N]
            A_bar = torch.exp(delta_t * A[None, :])  # [B, N]
            B_t = B[:, t, :]  # [B, N]
            C_t = C[:, t, :]  # [B, N]
            x_t = x[:, t, :]  # [B, D_inner]
            
            # State update: h_t = A_bar * h_{t-1} + delta * B * x
            # Average x over D_inner to match state dimension
            x_avg = x_t.mean(dim=-1, keepdim=True)  # [B, 1]
            h = A_bar * h + delta_t * B_t * x_avg
            
            # Output: y = h * C
            y_t = (h * C_t).sum(dim=-1, keepdim=True)  # [B, 1]
            y_t = y_t.expand(-1, D_inner)  # [B, D_inner]
            
            outputs.append(y_t)
        
        y = torch.stack(outputs, dim=1)  # [B, L, D_inner]
        return y
