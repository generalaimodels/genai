"""
Hybrid Attention Module
Sliding window GQA Flash Attention with RoPE
"""

import torch
import torch.nn as nn
import math
from ..kernels.triton import rope_embedding, flash_attention_gqa, precompute_freqs_cis


class HybridAttention(nn.Module):
    """
    Grouped Query Attention with:
    - RoPE positional encoding
    - Sliding window causal mask
    - Flash Attention optimization
    - KV-cache support for inference
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int = None,
        window_size: int = 4096,
        max_seq_len: int = 8192,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.window_size = window_size
        self.head_dim = d_model // n_heads
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads (GQA)"
        
        # QKV projections
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        
        # Output projection
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # Precompute RoPE frequencies
        cos, sin = precompute_freqs_cis(self.head_dim, max_seq_len)
        self.register_buffer('cos_cache', cos)
        self.register_buffer('sin_cache', sin)
        
        # KV-cache for inference
        self.kv_cache = None
    
    def forward(self, x, use_cache=False, cache_position=None):
        """
        Args:
            x: [B, L, D] input sequence
            use_cache: whether to use KV-cache (inference)
            cache_position: current position in cache
            
        Returns:
            out: [B, L, D] attention output
        """
        B, L, D = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim)
        K = self.k_proj(x).view(B, L, self.n_kv_heads, self.head_dim)
        V = self.v_proj(x).view(B, L, self.n_kv_heads, self.head_dim)
        
        # Transpose to [B, H, L, D]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Apply RoPE
        cos = self.cos_cache[:L]
        sin = self.sin_cache[:L]
        Q_rot, K_rot = rope_embedding(Q, K, cos, sin)
        
        # KV-cache management (autoregressive inference)
        if use_cache:
            if self.kv_cache is None:
                self.kv_cache = {'K': K_rot, 'V': V}
            else:
                K_rot = torch.cat([self.kv_cache['K'], K_rot], dim=2)
                V = torch.cat([self.kv_cache['V'], V], dim=2)
                self.kv_cache = {'K': K_rot, 'V': V}
        
        # Flash Attention with GQA and sliding window
        attn_out, _ = flash_attention_gqa(
            Q_rot,
            K_rot,
            V,
            window_size=self.window_size
        )  # [B, H, L, D]
        
        # Transpose back and merge heads
        attn_out = attn_out.transpose(1, 2).contiguous()  # [B, L, H, D]
        attn_out = attn_out.view(B, L, self.n_heads * self.head_dim)
        
        # Output projection
        out = self.o_proj(attn_out)
        out = self.dropout(out)
        
        return out
    
    def reset_cache(self):
        """Clear KV-cache"""
        self.kv_cache = None
