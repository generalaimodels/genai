"""
Hierarchical Mixture of Experts Module
Top-K routing with SwiGLU experts and load balancing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..kernels.triton import topk_gating, swiglu_expert


class HierarchicalMoE(nn.Module):
    """
    Sparse MoE Layer:
    - Top-K expert selection per token
    - SwiGLU expert architecture
    - Load balancing auxiliary loss
    - Expert parallelism ready
    """
    
    def __init__(
        self,
        d_model: int,
        n_experts: int,
        topk: int = 2,
        d_ff_ratio: int = 4,
        dropout: float = 0.0,
        load_balance_weight: float = 0.01,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_experts = n_experts
        self.topk = topk
        self.d_ff = d_model * d_ff_ratio
        self.load_balance_weight = load_balance_weight
        
        # Gating network
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        
        # Expert parameters
        # Each expert: SwiGLU(x) = (Swish(x*W1) ⊙ (x*V1)) * W2
        self.experts_W1 = nn.Parameter(torch.randn(n_experts, self.d_ff, d_model))
        self.experts_V1 = nn.Parameter(torch.randn(n_experts, self.d_ff, d_model))
        self.experts_W2 = nn.Parameter(torch.randn(n_experts, d_model, self.d_ff))
        
        self.dropout = nn.Dropout(dropout)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Kaiming initialization for experts"""
        nn.init.kaiming_normal_(self.experts_W1, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.experts_V1, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.experts_W2, mode='fan_in', nonlinearity='relu')
        
        # Gate: small random init
        nn.init.normal_(self.gate.weight, std=0.01)
    
    def forward(self, x):
        """
        Args:
            x: [B, L, D] input sequence
            
        Returns:
            out: [B, L, D] MoE output
            aux_loss: load balancing auxiliary loss
        """
        B, L, D = x.shape
        
        # Flatten batch and sequence
        x_flat = x.view(-1, D)  # [B*L, D]
        
        # Top-K gating
        indices, scores, load = topk_gating(
            x_flat,
            self.gate.weight.T,  # [D, N_experts]
            k=self.topk
        )  # indices: [B*L, K], scores: [B*L, K], load: [N_experts]
        
        # Compute load balancing loss
        # Encourages uniform distribution: L_aux = N * sum(f_i * P_i)
        # f_i = fraction of tokens routed to expert i
        # P_i = average gate probability for expert i
        
        total_tokens = B * L
        f = load.float() / total_tokens  # [N_experts]
        
        # Average gate probability (softmax of all gate values)
        all_scores = F.softmax(x_flat @ self.gate.weight.T, dim=-1)  # [B*L, N_experts]
        P = all_scores.mean(dim=0)  # [N_experts]
        
        aux_loss = self.n_experts * (f * P).sum() * self.load_balance_weight
        
        # Route tokens to experts
        output = torch.zeros_like(x_flat)
        
        # Expert computation (batched by expert)
        for expert_idx in range(self.n_experts):
            # Find tokens routed to this expert
            mask = (indices == expert_idx)  # [B*L, K]
            token_indices = mask.any(dim=1).nonzero(as_tuple=True)[0]
            
            if len(token_indices) == 0:
                continue
            
            # Get expert weights
            W1 = self.experts_W1[expert_idx]  # [D_ff, D]
            V1 = self.experts_V1[expert_idx]
            W2 = self.experts_W2[expert_idx]  # [D, D_ff]
            
            # Get tokens for this expert
            x_expert = x_flat[token_indices]  # [N_tokens, D]
            
            # Compute expert output using Triton kernel
            y_expert = swiglu_expert(x_expert, W1, V1, W2)  # [N_tokens, D]
            
            # Weight by gating scores
            for k_idx in range(self.topk):
                k_mask = mask[token_indices, k_idx]
                if k_mask.any():
                    k_tokens = token_indices[k_mask]
                    k_scores = scores[k_tokens, k_idx]
                    
                    # Accumulate weighted expert output
                    output[k_tokens] += y_expert[k_mask] * k_scores.unsqueeze(-1)
        
        # Reshape back
        output = output.view(B, L, D)
        output = self.dropout(output)
        
        return output, aux_loss
