"""
Hybrid Attention Layer
======================
Sliding Window Attention with GQA for RSS-MoD architecture.

Interleaved with SSM layers to provide:
- Precise in-context retrieval (induction heads)
- Local pattern matching
- Global context when needed

Features:
- Sliding window with configurable size
- Grouped Query Attention (GQA) for efficiency
- RoPE position embeddings
- KV-cache for inference
- Softcapping for stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from ..config import AttentionConfig, RSSMoDConfig
from ..kernels.triton import (
    triton_sliding_window_attention,
    triton_rmsnorm,
    precompute_freqs_cis,
    apply_rotary_pos_emb,
    RMSNorm,
)


class SlidingWindowAttentionLayer(nn.Module):
    """
    Sliding Window Attention with Grouped Query Attention.
    
    Uses FlashAttention-2 style Triton kernel for efficiency.
    """
    
    def __init__(
        self,
        d_model: int,
        config: Optional[AttentionConfig] = None,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.config = config or AttentionConfig()
        self.layer_idx = layer_idx
        
        self.num_heads = self.config.num_heads
        self.num_kv_heads = self.config.num_kv_heads or self.num_heads
        self.head_dim = d_model // self.num_heads
        self.window_size = self.config.window_size
        
        assert d_model % self.num_heads == 0
        assert self.num_heads % self.num_kv_heads == 0
        
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        
        # QKV projections
        self.q_proj = nn.Linear(d_model, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, d_model, bias=False)
        
        # RoPE frequencies
        self.rope_theta = self.config.rope_theta
        self._freqs_cis = None
        
        # Scaling
        self.scale = self.head_dim ** -0.5
        self.softcap = self.config.softcap
        
        # Dropout
        self.attn_dropout = nn.Dropout(self.config.attention_dropout)
    
    def _init_rope(self, device: torch.device):
        """Initialize RoPE frequencies."""
        if self._freqs_cis is None or self._freqs_cis.device != device:
            self._freqs_cis = precompute_freqs_cis(
                self.head_dim,
                self.config.max_position_embeddings,
                theta=self.rope_theta,
                device=device,
                scaling_type=self.config.rope_scaling.get("type") if self.config.rope_scaling else None,
                scaling_factor=self.config.rope_scaling.get("factor", 1.0) if self.config.rope_scaling else 1.0,
            )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass.
        
        Args:
            hidden_states: (batch, seq_len, d_model)
            attention_mask: Optional attention mask
            position_ids: Optional position indices
            past_key_value: Optional KV cache
            use_cache: Whether to return updated cache
            
        Returns:
            Tuple of (output, new_kv_cache)
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        
        # Initialize RoPE
        self._init_rope(device)
        
        # Compute Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape: (batch, seq, num_heads, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Apply RoPE
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        
        cos = self._freqs_cis[:seq_len].real
        sin = self._freqs_cis[:seq_len].imag
        
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=2)
        
        # Handle KV cache
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)
        
        # Update cache
        if use_cache:
            # For sliding window, only cache up to window_size
            cache_len = min(k.shape[1], self.window_size)
            new_kv_cache = (k[:, -cache_len:], v[:, -cache_len:])
        else:
            new_kv_cache = None
        
        # Expand K,V for GQA
        if self.num_kv_groups > 1:
            k = k.unsqueeze(3).expand(-1, -1, -1, self.num_kv_groups, -1)
            k = k.reshape(batch_size, k.shape[1], self.num_heads, self.head_dim)
            v = v.unsqueeze(3).expand(-1, -1, -1, self.num_kv_groups, -1)
            v = v.reshape(batch_size, v.shape[1], self.num_heads, self.head_dim)
        
        # Use Triton sliding window attention
        if q.is_cuda and self.config.use_flash_attn:
            # For query, reshape to match kernel expectations
            attn_output, _ = triton_sliding_window_attention(
                q, k, v,
                window_size=self.window_size,
                causal=True,
                softcap=self.softcap or 0.0,
            )
        else:
            # Fallback to PyTorch
            attn_output = self._pytorch_attention(q, k, v, attention_mask)
        
        # Reshape and project
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        output = self.o_proj(attn_output)
        
        return output, new_kv_cache
    
    def _pytorch_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fallback PyTorch attention implementation."""
        batch_size, q_len, num_heads, head_dim = q.shape
        kv_len = k.shape[1]
        
        # Transpose for matmul: (batch, num_heads, seq, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply softcap
        if self.softcap is not None:
            scores = self.softcap * torch.tanh(scores / self.softcap)
        
        # Create causal + sliding window mask
        causal_mask = torch.triu(
            torch.ones(q_len, kv_len, device=q.device, dtype=torch.bool),
            diagonal=kv_len - q_len + 1
        )
        
        # Sliding window mask
        window_mask = torch.ones(q_len, kv_len, device=q.device, dtype=torch.bool)
        for i in range(q_len):
            start = max(0, kv_len - q_len + i - self.window_size + 1)
            end = kv_len - q_len + i + 1
            window_mask[i, start:end] = False
        
        # Combined mask
        mask = causal_mask | window_mask
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        
        # Apply additional attention mask if provided
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Compute output
        output = torch.matmul(attn_weights, v)
        
        # Transpose back: (batch, seq, num_heads, head_dim)
        output = output.transpose(1, 2)
        
        return output


class HybridAttention(nn.Module):
    """
    Hybrid Attention block combining sliding window attention
    with pre/post normalization and residual connections.
    
    Used at stride intervals in RSS-MoD architecture.
    """
    
    def __init__(
        self,
        d_model: int,
        config: Optional[AttentionConfig] = None,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.0,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.layer_idx = layer_idx
        
        # Pre-norm
        self.attn_norm = RMSNorm(d_model)
        
        # Attention
        self.attention = SlidingWindowAttentionLayer(d_model, config, layer_idx)
        
        # Optional FFN
        if ffn_dim is not None:
            self.ffn_norm = RMSNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, ffn_dim, bias=False),
                nn.SiLU(),
                nn.Linear(ffn_dim, d_model, bias=False),
            )
        else:
            self.ffn = None
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass.
        
        Args:
            hidden_states: (batch, seq_len, d_model)
            
        Returns:
            Tuple of (output, kv_cache)
        """
        # Attention block
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, new_cache = self.attention(
            hidden_states, attention_mask, position_ids, past_key_value, use_cache
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        # FFN block (if present)
        if self.ffn is not None:
            residual = hidden_states
            hidden_states = self.ffn_norm(hidden_states)
            hidden_states = self.ffn(hidden_states)
            hidden_states = self.dropout(hidden_states)
            hidden_states = residual + hidden_states
        
        return hidden_states, new_cache
    
    def allocate_cache(
        self,
        batch_size: int,
        max_seq_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Allocate KV cache for inference.
        
        Returns:
            Tuple of (key_cache, value_cache)
        """
        cache_len = min(max_seq_len, self.attention.window_size)
        
        k_cache = torch.zeros(
            batch_size,
            cache_len,
            self.attention.num_kv_heads,
            self.attention.head_dim,
            device=device,
            dtype=dtype,
        )
        
        v_cache = torch.zeros(
            batch_size,
            cache_len,
            self.attention.num_kv_heads,
            self.attention.head_dim,
            device=device,
            dtype=dtype,
        )
        
        return k_cache, v_cache
