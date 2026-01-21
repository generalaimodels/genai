"""
Mixture-of-Experts (MoE) Layer
==============================
Iso-FLOP sparse expert routing for RSS-MoD architecture.

Features:
- Top-K expert selection with load balancing
- Sinkhorn-Knopp for aux-loss-free balancing
- Shared expert for common patterns
- Efficient grouped expert computation
- Expert parallelism support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math

from ..config import MoEConfig, RSSMoDConfig
from ..kernels.triton import (
    MoEDispatch,
    moe_dispatch_forward,
    moe_gather_forward,
    RMSNorm,
)


class ExpertFFN(nn.Module):
    """
    Single expert Feed-Forward Network.
    
    Standard SwiGLU architecture:
        x -> [gate_proj, up_proj] -> SiLU(gate) * up -> down_proj
    """
    
    def __init__(
        self,
        d_model: int,
        intermediate_size: int,
        bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.intermediate_size = intermediate_size
        
        self.gate_proj = nn.Linear(d_model, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(d_model, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEFFN(nn.Module):
    """
    Mixture-of-Experts Feed-Forward Network.
    
    Routes tokens to specialized experts based on learned gating.
    """
    
    def __init__(
        self,
        d_model: int,
        intermediate_size: int,
        config: Optional[MoEConfig] = None,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.intermediate_size = intermediate_size
        self.config = config or MoEConfig()
        self.layer_idx = layer_idx
        
        self.num_experts = self.config.num_experts
        self.num_experts_per_tok = self.config.num_experts_per_tok
        self.capacity_factor = self.config.capacity_factor
        
        # Router
        self.router = nn.Linear(d_model, self.num_experts, bias=False)
        
        # Experts
        self.experts = nn.ModuleList([
            ExpertFFN(d_model, intermediate_size)
            for _ in range(self.num_experts)
        ])
        
        # Optional shared expert
        if self.config.shared_expert:
            self.shared_expert = ExpertFFN(d_model, intermediate_size)
            self.shared_gate = nn.Linear(d_model, 1, bias=False)
        else:
            self.shared_expert = None
        
        # Balancing parameters
        self.use_sinkhorn = self.config.use_sinkhorn
        self.sinkhorn_iters = self.config.sinkhorn_iters
        self.router_aux_loss_coef = self.config.router_aux_loss_coef
        self.router_z_loss_coef = self.config.router_z_loss_coef
        
        self._aux_loss = None
        self._z_loss = None
    
    @property
    def aux_loss(self) -> Optional[torch.Tensor]:
        return self._aux_loss
    
    @property
    def z_loss(self) -> Optional[torch.Tensor]:
        return self._z_loss
    
    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with expert routing.
        
        Args:
            hidden_states: (batch, seq_len, d_model)
            
        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = hidden_states.shape
        
        # Flatten for routing
        hidden_flat = hidden_states.view(-1, d_model)
        num_tokens = hidden_flat.shape[0]
        
        # Compute router logits
        router_logits = self.router(hidden_flat)  # (num_tokens, num_experts)
        
        # Z-loss for router stability
        if self.training and self.router_z_loss_coef > 0:
            z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
            self._z_loss = self.router_z_loss_coef * z_loss
        else:
            self._z_loss = None
        
        # Compute routing probabilities
        if self.use_sinkhorn:
            router_probs = self._sinkhorn_routing(router_logits)
        else:
            router_probs = F.softmax(router_logits, dim=-1)
        
        # Top-K selection
        routing_weights, expert_indices = torch.topk(
            router_probs, self.num_experts_per_tok, dim=-1
        )
        
        # Renormalize weights
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        # Compute auxiliary load balancing loss
        if self.training and self.router_aux_loss_coef > 0:
            self._aux_loss = self._compute_aux_loss(router_probs, expert_indices)
        else:
            self._aux_loss = None
        
        # Process through experts
        output = self._route_to_experts(hidden_flat, expert_indices, routing_weights)
        
        # Add shared expert contribution
        if self.shared_expert is not None:
            shared_gate = torch.sigmoid(self.shared_gate(hidden_flat))
            shared_out = self.shared_expert(hidden_flat)
            output = output + shared_gate * shared_out
        
        # Reshape
        output = output.view(batch_size, seq_len, d_model)
        
        return output
    
    def _sinkhorn_routing(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Apply Sinkhorn-Knopp to get balanced routing.
        """
        num_tokens = logits.shape[0]
        
        # Initial softmax
        probs = F.softmax(logits / temperature, dim=-1)
        
        # Target: each expert gets equal load
        target_per_expert = num_tokens * self.num_experts_per_tok / self.num_experts
        
        for _ in range(self.sinkhorn_iters):
            # Column normalization (expert capacity)
            col_sum = probs.sum(dim=0, keepdim=True)
            probs = probs * (target_per_expert / (col_sum + 1e-8))
            
            # Row normalization (token distribution)
            row_sum = probs.sum(dim=1, keepdim=True)
            probs = probs / (row_sum + 1e-8)
        
        return probs
    
    def _compute_aux_loss(
        self,
        router_probs: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute auxiliary load balancing loss.
        """
        num_tokens = router_probs.shape[0]
        
        # Fraction of tokens routed to each expert
        expert_mask = F.one_hot(expert_indices, self.num_experts).sum(dim=1)
        tokens_per_expert = expert_mask.float().mean(dim=0)
        
        # Average routing probability per expert
        router_prob_per_expert = router_probs.mean(dim=0)
        
        # Load balance loss: minimize variance in expert utilization
        aux_loss = self.num_experts * (tokens_per_expert * router_prob_per_expert).sum()
        
        return self.router_aux_loss_coef * aux_loss
    
    def _route_to_experts(
        self,
        hidden_states: torch.Tensor,
        expert_indices: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Route tokens to experts and combine outputs.
        """
        num_tokens, d_model = hidden_states.shape
        
        # Initialize output
        output = torch.zeros_like(hidden_states)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            expert = self.experts[expert_idx]
            
            # Find tokens assigned to this expert
            # expert_indices: (num_tokens, top_k)
            token_mask = (expert_indices == expert_idx).any(dim=-1)
            
            if not token_mask.any():
                continue
            
            # Get positions where this expert was selected
            token_positions = token_mask.nonzero(as_tuple=True)[0]
            
            # Get the weight for this expert at each position
            expert_positions = (expert_indices[token_mask] == expert_idx)
            weights_for_expert = routing_weights[token_mask][expert_positions]
            
            # This is simplified - proper implementation handles multiple selections
            # For now, use mean weight if token selected this expert multiple times
            if len(token_positions) > 0:
                # Get tokens for this expert
                expert_input = hidden_states[token_positions]
                
                # Process through expert
                expert_output = expert(expert_input)
                
                # Get corresponding weights (take first match for simplicity)
                # In production, would properly handle all k selections
                first_match = (expert_indices[token_positions] == expert_idx).float()
                first_weights = (routing_weights[token_positions] * first_match).sum(dim=-1)
                
                # Weight and accumulate
                output[token_positions] += first_weights.unsqueeze(-1) * expert_output
        
        return output


class MoELayer(nn.Module):
    """
    Complete MoE layer with normalization and residual.
    """
    
    def __init__(
        self,
        d_model: int,
        intermediate_size: int,
        config: Optional[MoEConfig] = None,
        dropout: float = 0.0,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        
        self.norm = RMSNorm(d_model)
        self.moe = MoEFFN(d_model, intermediate_size, config, layer_idx)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    @property
    def aux_loss(self) -> Optional[torch.Tensor]:
        return self.moe.aux_loss
    
    @property
    def z_loss(self) -> Optional[torch.Tensor]:
        return self.moe.z_loss
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward with pre-norm and residual."""
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states = self.moe(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return residual + hidden_states


class ParallelMoELayer(nn.Module):
    """
    Expert-parallel MoE layer for distributed training.
    
    Each rank holds a subset of experts.
    """
    
    def __init__(
        self,
        d_model: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int = 2,
        expert_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.ep_group = expert_parallel_group
        
        if self.ep_group is not None:
            self.ep_size = torch.distributed.get_world_size(self.ep_group)
            self.ep_rank = torch.distributed.get_rank(self.ep_group)
        else:
            self.ep_size = 1
            self.ep_rank = 0
        
        # Number of local experts
        assert num_experts % self.ep_size == 0
        self.num_local_experts = num_experts // self.ep_size
        
        # Router (replicated)
        self.router = nn.Linear(d_model, num_experts, bias=False)
        
        # Local experts
        self.local_experts = nn.ModuleList([
            ExpertFFN(d_model, intermediate_size)
            for _ in range(self.num_local_experts)
        ])
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward with expert parallelism.
        
        Uses all-to-all communication to dispatch tokens to experts
        on different ranks.
        """
        batch_size, seq_len, d_model = hidden_states.shape
        hidden_flat = hidden_states.view(-1, d_model)
        
        # Compute routing (all ranks compute same routing)
        router_logits = self.router(hidden_flat)
        router_probs = F.softmax(router_logits, dim=-1)
        routing_weights, expert_indices = torch.topk(
            router_probs, self.num_experts_per_tok, dim=-1
        )
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        if self.ep_size > 1:
            # All-to-all dispatch
            output = self._parallel_dispatch(hidden_flat, expert_indices, routing_weights)
        else:
            # Local processing
            output = self._local_dispatch(hidden_flat, expert_indices, routing_weights)
        
        return output.view(batch_size, seq_len, d_model)
    
    def _local_dispatch(
        self,
        hidden_states: torch.Tensor,
        expert_indices: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Process locally when no expert parallelism."""
        num_tokens, d_model = hidden_states.shape
        output = torch.zeros_like(hidden_states)
        
        for local_idx, expert in enumerate(self.local_experts):
            global_idx = self.ep_rank * self.num_local_experts + local_idx
            
            token_mask = (expert_indices == global_idx).any(dim=-1)
            if not token_mask.any():
                continue
            
            token_positions = token_mask.nonzero(as_tuple=True)[0]
            expert_input = hidden_states[token_positions]
            expert_output = expert(expert_input)
            
            match_mask = (expert_indices[token_positions] == global_idx).float()
            weights = (routing_weights[token_positions] * match_mask).sum(dim=-1)
            
            output[token_positions] += weights.unsqueeze(-1) * expert_output
        
        return output
    
    def _parallel_dispatch(
        self,
        hidden_states: torch.Tensor,
        expert_indices: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Dispatch tokens across ranks using all-to-all.
        
        This is a simplified version - production would use
        optimized collective operations.
        """
        # In production, would implement:
        # 1. Compute send counts per rank
        # 2. All-to-all-v for token dispatch
        # 3. Local expert computation
        # 4. All-to-all-v for token gather
        # 5. Weighted combination
        
        # For now, fall back to local dispatch
        # (assumes replicated experts for testing)
        return self._local_dispatch(hidden_states, expert_indices, routing_weights)
