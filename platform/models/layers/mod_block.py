"""
Mixture-of-Depths (MoD) Block
=============================
Dynamic compute allocation via learned token routing.

Selectively processes tokens through expensive computations,
bypassing "easy" tokens with residual connections.

Features:
- Top-K token selection based on learned router
- Capacity-controlled routing
- Auxiliary load balancing
- Efficient scatter-gather operations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable
import math

from ..config import MoDConfig, RSSMoDConfig
from ..kernels.triton import MoDRouter, mod_top_k_routing, RMSNorm


class MoDBlock(nn.Module):
    """
    Mixture-of-Depths Block.
    
    Routes tokens dynamically: easy tokens bypass, hard tokens process.
    
    Architecture:
        x -> router -> [selected: block(x)] + [bypassed: x]
    """
    
    def __init__(
        self,
        d_model: int,
        inner_block: nn.Module,
        config: Optional[MoDConfig] = None,
        layer_idx: Optional[int] = None,
    ):
        """
        Args:
            d_model: Model dimension
            inner_block: The block to apply to selected tokens
            config: MoD configuration
            layer_idx: Layer index
        """
        super().__init__()
        self.d_model = d_model
        self.config = config or MoDConfig()
        self.layer_idx = layer_idx
        
        self.capacity_factor = self.config.capacity_factor
        self.aux_loss_weight = self.config.aux_loss_weight
        
        # Router
        self.router = MoDRouter(
            d_model=d_model,
            capacity_factor=self.capacity_factor,
            aux_loss_weight=self.aux_loss_weight,
            router_jitter=self.config.router_jitter if hasattr(self.config, "router_jitter") else 0.0,
        )
        
        # Inner computation block
        self.inner_block = inner_block
        
        # Store aux loss for training
        self._aux_loss = None
    
    @property
    def aux_loss(self) -> Optional[torch.Tensor]:
        """Get the auxiliary loss from last forward pass."""
        return self._aux_loss
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with dynamic routing.
        
        Args:
            hidden_states: (batch, seq_len, d_model)
            **kwargs: Additional arguments for inner block
            
        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = hidden_states.shape
        
        # Compute routing
        selected_hidden, indices, weights, aux_loss = self.router(hidden_states)
        
        # Store auxiliary loss
        self._aux_loss = aux_loss
        
        num_selected = selected_hidden.shape[1]
        
        if num_selected == 0:
            # No tokens selected - return input unchanged
            return hidden_states
        
        # Process selected tokens through inner block
        # Note: inner_block should handle the variable length gracefully
        processed = self.inner_block(selected_hidden, **kwargs)
        
        # Handle case where inner_block returns tuple
        if isinstance(processed, tuple):
            processed = processed[0]
        
        # Scatter processed tokens back with residual
        output = self.router.scatter_back(
            hidden_states, processed, indices, weights
        )
        
        return output
    
    def get_routing_stats(
        self,
        hidden_states: torch.Tensor,
    ) -> dict:
        """
        Get routing statistics for analysis.
        
        Returns:
            Dictionary with routing statistics
        """
        with torch.no_grad():
            scores, indices, weights, mask = self.router.router(hidden_states)
            
            return {
                "mean_score": scores.mean().item(),
                "std_score": scores.std().item(),
                "selected_ratio": mask.float().mean().item(),
                "mean_weight": weights.mean().item(),
            }


class MoDWrapper(nn.Module):
    """
    Wrapper to add MoD routing to any block.
    
    Provides a simple interface to wrap existing layers with
    Mixture-of-Depths dynamic routing.
    """
    
    def __init__(
        self,
        wrapped_module: nn.Module,
        d_model: int,
        capacity_factor: float = 1.0,
        aux_loss_weight: float = 0.01,
        bypass_residual: bool = True,
    ):
        """
        Args:
            wrapped_module: Module to wrap
            d_model: Model dimension
            capacity_factor: Fraction of tokens to process
            aux_loss_weight: Weight for load balancing loss
            bypass_residual: Whether bypassed tokens use residual
        """
        super().__init__()
        
        self.wrapped_module = wrapped_module
        self.d_model = d_model
        self.bypass_residual = bypass_residual
        
        # Router weights
        self.router_weight = nn.Parameter(torch.randn(d_model) * 0.02)
        self.router_bias = nn.Parameter(torch.zeros(1))
        
        self.capacity_factor = capacity_factor
        self.aux_loss_weight = aux_loss_weight
        
        self._aux_loss = None
    
    @property
    def aux_loss(self) -> Optional[torch.Tensor]:
        return self._aux_loss
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Forward with routing."""
        batch_size, seq_len, d_model = hidden_states.shape
        
        # Compute routing
        selected, indices, weights, aux_loss = mod_top_k_routing(
            hidden_states,
            self.router_weight,
            self.router_bias,
            self.capacity_factor,
            self.aux_loss_weight if self.training else 0.0,
            self.training,
        )
        
        self._aux_loss = aux_loss
        
        if selected.shape[1] == 0:
            return hidden_states
        
        # Process selected
        processed = self.wrapped_module(selected, **kwargs)
        if isinstance(processed, tuple):
            processed = processed[0]
        
        # Scatter back
        output = hidden_states.clone()
        
        # Weight processed output
        weighted = processed * weights.unsqueeze(-1)
        
        # Add to original positions
        if self.bypass_residual:
            # Residual: output = input + processed
            output.scatter_add_(
                1,
                indices.unsqueeze(-1).expand(-1, -1, d_model),
                weighted
            )
        else:
            # Replace: output[selected] = processed
            output.scatter_(
                1,
                indices.unsqueeze(-1).expand(-1, -1, d_model),
                selected + weighted  # Add back residual from gathered
            )
        
        return output


class AdaptiveMoDBlock(nn.Module):
    """
    Adaptive Mixture-of-Depths with dynamic capacity.
    
    Adjusts routing capacity based on input complexity.
    """
    
    def __init__(
        self,
        d_model: int,
        inner_block: nn.Module,
        min_capacity: float = 0.25,
        max_capacity: float = 1.0,
        complexity_head: Optional[nn.Module] = None,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.inner_block = inner_block
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        
        # Complexity estimator
        if complexity_head is None:
            self.complexity_head = nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, 1),
                nn.Sigmoid(),
            )
        else:
            self.complexity_head = complexity_head
        
        # Router
        self.router_weight = nn.Parameter(torch.randn(d_model) * 0.02)
        self.router_bias = nn.Parameter(torch.zeros(1))
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Forward with adaptive capacity."""
        batch_size, seq_len, d_model = hidden_states.shape
        
        # Estimate complexity (using sequence mean)
        complexity = self.complexity_head(hidden_states.mean(dim=1))  # (batch, 1)
        
        # Scale capacity based on complexity
        capacity = self.min_capacity + complexity * (self.max_capacity - self.min_capacity)
        capacity = capacity.mean().item()  # Use batch mean for now
        
        # Route with adaptive capacity
        selected, indices, weights, aux_loss = mod_top_k_routing(
            hidden_states,
            self.router_weight,
            self.router_bias,
            capacity,
            0.01,
            self.training,
        )
        
        if selected.shape[1] == 0:
            return hidden_states
        
        # Process
        processed = self.inner_block(selected, **kwargs)
        if isinstance(processed, tuple):
            processed = processed[0]
        
        # Scatter back with residual
        output = hidden_states.clone()
        weighted = processed * weights.unsqueeze(-1)
        output.scatter_add_(
            1,
            indices.unsqueeze(-1).expand(-1, -1, d_model),
            weighted
        )
        
        return output
