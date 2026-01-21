"""
Pause Token Head (System-2 Reasoning)
=====================================
Vertical recurrence mechanism for complex reasoning.

Implements adaptive computation time by:
- Generating internal "pause" tokens during hard queries
- Performing recurrent computation on same position
- Allowing O(N) compute for hard problems, O(1) for easy

Features:
- Learned pause decision
- Configurable max depth
- Entropy-based adaptive halting
- Hidden state refinement loop
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from ..config import PauseHeadConfig, RSSMoDConfig
from ..kernels.triton import RMSNorm


class PauseTokenHead(nn.Module):
    """
    Pause Token Head for System-2 reasoning.
    
    During inference, can iteratively refine hidden states
    before emitting output token. Enables deeper "thinking"
    for complex queries.
    
    Architecture:
        hidden -> [halt_gate] -> if halt < threshold:
                                    hidden = refine(hidden)
                                    repeat
                                 else:
                                    emit output
    """
    
    def __init__(
        self,
        d_model: int,
        config: Optional[PauseHeadConfig] = None,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.config = config or PauseHeadConfig()
        self.layer_idx = layer_idx
        
        self.max_pause_steps = self.config.max_pause_steps
        self.pause_threshold = self.config.pause_threshold
        self.hidden_dim = self.config.hidden_dim
        self.use_adaptive = self.config.use_adaptive
        
        # Halting decision network
        self.halt_gate = nn.Sequential(
            nn.Linear(d_model, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid(),
        )
        
        # State refinement network
        self.refine_norm = RMSNorm(d_model)
        self.refine_net = nn.Sequential(
            nn.Linear(d_model, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, d_model),
        )
        
        # Residual gate for refinement
        self.residual_gate = nn.Linear(d_model, d_model)
        
        # Track ponder cost for training
        self._ponder_cost = None
    
    @property
    def ponder_cost(self) -> Optional[torch.Tensor]:
        """Get ponder time cost from last forward pass."""
        return self._ponder_cost
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        force_pause_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with adaptive computation.
        
        Args:
            hidden_states: (batch, seq_len, d_model)
            force_pause_steps: Optional fixed number of pause steps
            
        Returns:
            Tuple of (refined_hidden, halt_probs)
        """
        batch_size, seq_len, d_model = hidden_states.shape
        device = hidden_states.device
        
        if force_pause_steps is not None:
            # Fixed computation time
            return self._fixed_pondering(hidden_states, force_pause_steps)
        
        if self.use_adaptive:
            return self._adaptive_pondering(hidden_states)
        else:
            return self._threshold_pondering(hidden_states)
    
    def _fixed_pondering(
        self,
        hidden_states: torch.Tensor,
        num_steps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fixed number of pondering steps."""
        batch_size, seq_len, d_model = hidden_states.shape
        
        current = hidden_states
        halt_probs = torch.zeros(batch_size, seq_len, device=hidden_states.device)
        
        for step in range(num_steps):
            # Refine
            residual = current
            current = self.refine_norm(current)
            refined = self.refine_net(current)
            
            # Gated residual
            gate = torch.sigmoid(self.residual_gate(current))
            current = residual + gate * refined
            
            # Compute halt probability (for monitoring)
            halt_prob = self.halt_gate(current).squeeze(-1)
            halt_probs = halt_probs + halt_prob / num_steps
        
        return current, halt_probs
    
    def _threshold_pondering(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ponder until halt threshold exceeded."""
        batch_size, seq_len, d_model = hidden_states.shape
        device = hidden_states.device
        
        current = hidden_states
        halted = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        halt_probs = torch.zeros(batch_size, seq_len, device=device)
        ponder_times = torch.zeros(batch_size, seq_len, device=device)
        
        for step in range(self.max_pause_steps):
            # Compute halt probability
            halt_prob = self.halt_gate(current).squeeze(-1)  # (batch, seq_len)
            
            # Update halted mask
            should_halt = (halt_prob > self.pause_threshold) | halted
            just_halted = should_halt & ~halted
            
            # Record halt probability for newly halted
            halt_probs = torch.where(just_halted, halt_prob, halt_probs)
            
            # Record ponder time for newly halted
            ponder_times = torch.where(
                just_halted,
                torch.full_like(ponder_times, step + 1),
                ponder_times
            )
            
            halted = should_halt
            
            # Check if all halted
            if halted.all():
                break
            
            # Refine non-halted states
            residual = current
            normed = self.refine_norm(current)
            refined = self.refine_net(normed)
            gate = torch.sigmoid(self.residual_gate(normed))
            
            # Only update non-halted positions
            update = gate * refined
            current = torch.where(
                halted.unsqueeze(-1).expand_as(current),
                current,
                residual + update
            )
        
        # Store ponder cost for training
        self._ponder_cost = ponder_times.mean()
        
        return current, halt_probs
    
    def _adaptive_pondering(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adaptive Computation Time (ACT) style pondering.
        
        Accumulates halting probabilities and outputs weighted
        combination of states.
        """
        batch_size, seq_len, d_model = hidden_states.shape
        device = hidden_states.device
        
        # Accumulators
        output = torch.zeros_like(hidden_states)
        halt_cumsum = torch.zeros(batch_size, seq_len, device=device)
        remainders = torch.zeros(batch_size, seq_len, device=device)
        n_updates = torch.zeros(batch_size, seq_len, device=device)
        
        current = hidden_states
        active = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        
        for step in range(self.max_pause_steps):
            # Compute halt probability
            halt_prob = self.halt_gate(current).squeeze(-1)
            
            # Check if this is the last step
            new_cumsum = halt_cumsum + halt_prob
            halting = (new_cumsum >= 1.0) | (step == self.max_pause_steps - 1)
            
            # Compute contribution weight
            weight = torch.where(
                halting,
                1.0 - halt_cumsum,  # Remainder for final step
                halt_prob
            )
            
            # Accumulate output
            output = output + weight.unsqueeze(-1) * current
            
            # Update tracking
            halt_cumsum = torch.where(halting, halt_cumsum, new_cumsum)
            n_updates = n_updates + active.float()
            
            # Update active mask
            active = active & ~halting
            
            if not active.any():
                break
            
            # Refine still-active states
            residual = current
            normed = self.refine_norm(current)
            refined = self.refine_net(normed)
            gate = torch.sigmoid(self.residual_gate(normed))
            
            current = residual + gate * refined
        
        # Ponder cost: mean number of steps
        self._ponder_cost = n_updates.mean()
        
        return output, halt_cumsum


class SystemTwoHead(nn.Module):
    """
    System-2 reasoning module combining:
    - Pause token generation
    - Internal deliberation
    - Confidence-based output selection
    """
    
    def __init__(
        self,
        d_model: int,
        config: Optional[PauseHeadConfig] = None,
        num_deliberation_heads: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_deliberation_heads
        
        # Multiple deliberation heads
        self.pause_heads = nn.ModuleList([
            PauseTokenHead(d_model, config, layer_idx=i)
            for i in range(num_deliberation_heads)
        ])
        
        # Confidence scoring
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        
        # Aggregation
        self.aggregate = nn.Linear(d_model * num_deliberation_heads, d_model)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward with multi-head deliberation.
        
        Each head ponders independently, results aggregated by confidence.
        """
        batch_size, seq_len, d_model = hidden_states.shape
        
        # Collect outputs from each head
        head_outputs = []
        head_confidences = []
        total_ponder_cost = 0
        
        for head in self.pause_heads:
            output, halt_probs = head(hidden_states)
            head_outputs.append(output)
            
            # Compute confidence
            confidence = self.confidence_head(output).squeeze(-1)
            head_confidences.append(confidence)
            
            if head.ponder_cost is not None:
                total_ponder_cost = total_ponder_cost + head.ponder_cost
        
        # Stack and weight by confidence
        outputs = torch.stack(head_outputs, dim=-2)  # (batch, seq, num_heads, d_model)
        confidences = torch.stack(head_confidences, dim=-1)  # (batch, seq, num_heads)
        weights = F.softmax(confidences, dim=-1)
        
        # Weighted combination
        weighted = outputs * weights.unsqueeze(-1)
        combined = weighted.sum(dim=-2)
        
        # Alternative: concatenate and project
        # concat = outputs.view(batch_size, seq_len, -1)
        # combined = self.aggregate(concat)
        
        avg_confidence = weights.max(dim=-1).values.mean()
        
        return combined, avg_confidence
