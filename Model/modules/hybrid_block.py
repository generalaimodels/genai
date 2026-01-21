"""
Hybrid Residual Block
Orchestrates SSM/Attention interleaving with MoE and residual connections
"""

import torch
import torch.nn as nn
from .selective_ssm import SelectiveSSM
from .hybrid_attention import HybridAttention
from .hierarchical_moe import HierarchicalMoE


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # RMS = sqrt(mean(x^2) + eps)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return x_norm * self.weight


class HybridResidualBlock(nn.Module):
    """
    Single residual block with:
    - Mixer (SSM or Attention)
    - MoE FFN
    - RMSNorm
    - Residual connections
    
    Architecture:
    x -> RMSNorm -> Mixer -> + -> RMSNorm -> MoE -> +
    |                        ^                     ^
    +------------------------+---------------------+
    """
    
    def __init__(
        self,
        d_model: int,
        mixer_type: str = 'ssm',  # 'ssm' or 'attention'
        # SSM params
        d_state: int = 16,
        d_conv: int = 4,
        ssm_expand: int = 2,
        # Attention params
        n_heads: int = 32,
        n_kv_heads: int = 8,
        window_size: int = 4096,
        # MoE params
        n_experts: int = 8,
        topk_experts: int = 2,
        moe_ff_ratio: int = 4,
        # General
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.mixer_type = mixer_type
        
        # Pre-mixer normalization
        self.norm1 = RMSNorm(d_model)
        
        # Mixer layer
        if mixer_type == 'ssm':
            self.mixer = SelectiveSSM(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=ssm_expand,
            )
        elif mixer_type == 'attention':
            self.mixer = HybridAttention(
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                window_size=window_size,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown mixer type: {mixer_type}")
        
        # Pre-MoE normalization
        self.norm2 = RMSNorm(d_model)
        
        # MoE layer
        self.moe = HierarchicalMoE(
            d_model=d_model,
            n_experts=n_experts,
            topk=topk_experts,
            d_ff_ratio=moe_ff_ratio,
            dropout=dropout,
        )
    
    def forward(self, x, use_cache=False):
        """
        Args:
            x: [B, L, D] input sequence
            use_cache: whether to use KV-cache (attention only)
            
        Returns:
            x: [B, L, D] output sequence
            aux_loss: MoE load balancing loss
        """
        # Mixer branch
        x_norm1 = self.norm1(x)
        
        if self.mixer_type == 'attention':
            mixer_out = self.mixer(x_norm1, use_cache=use_cache)
        else:
            mixer_out = self.mixer(x_norm1)
        
        x = x + mixer_out
        
        # MoE branch
        x_norm2 = self.norm2(x)
        moe_out, aux_loss = self.moe(x_norm2)
        x = x + moe_out
        
        return x, aux_loss
