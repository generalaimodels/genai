"""
Prefill Engine
==============
Optimized prefill phase for prompt processing.
Uses chunked prefill to avoid memory spikes.
"""

from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .config import InferenceConfig


@dataclass
class PrefillOutput:
    """Output from prefill phase."""
    hidden_states: "torch.Tensor"
    kv_cache: List[Tuple["torch.Tensor", "torch.Tensor"]]
    logits: Optional["torch.Tensor"] = None
    positions: Optional["torch.Tensor"] = None


class PrefillEngine:
    """
    Prefill engine for prompt processing.
    
    Optimizations:
    - Chunked prefill for memory efficiency
    - Flash Attention for O(n) memory
    - Parallel prefix computation
    
    Compute pattern: Matrix-Matrix (GEMM) - compute bound
    """
    
    def __init__(
        self,
        model: "nn.Module",
        config: InferenceConfig,
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch required")
        
        self.model = model
        self.config = config
        self.device = config.device
        
        self.chunk_size = config.prefill_chunk_size
        self.chunked = config.chunked_prefill
    
    @torch.inference_mode()
    def prefill(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> PrefillOutput:
        """
        Process prompt tokens.
        
        Args:
            input_ids: (batch, seq_len) prompt tokens
            attention_mask: Optional attention mask
            position_ids: Optional position IDs
            
        Returns:
            PrefillOutput with hidden states and KV cache
        """
        batch_size, seq_len = input_ids.shape
        
        if self.chunked and seq_len > self.chunk_size:
            return self._chunked_prefill(input_ids, attention_mask, position_ids)
        
        return self._full_prefill(input_ids, attention_mask, position_ids)
    
    def _full_prefill(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
    ) -> PrefillOutput:
        """Full prefill in one pass."""
        batch_size, seq_len = input_ids.shape
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Forward through model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            output_hidden_states=True,
        )
        
        hidden_states = outputs.hidden_states[-1] if hasattr(outputs, "hidden_states") else None
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        kv_cache = outputs.past_key_values if hasattr(outputs, "past_key_values") else []
        
        return PrefillOutput(
            hidden_states=hidden_states,
            kv_cache=list(kv_cache) if kv_cache else [],
            logits=logits,
            positions=position_ids,
        )
    
    def _chunked_prefill(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
    ) -> PrefillOutput:
        """
        Chunked prefill for long sequences.
        
        Processes prompt in chunks to limit memory usage.
        Each chunk builds upon previous KV cache.
        """
        batch_size, seq_len = input_ids.shape
        chunk_size = self.chunk_size
        
        kv_cache = None
        all_hidden_states = []
        
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            
            chunk_ids = input_ids[:, start:end]
            chunk_mask = attention_mask[:, :end] if attention_mask is not None else None
            
            # Position IDs for this chunk
            chunk_positions = torch.arange(start, end, device=input_ids.device)
            chunk_positions = chunk_positions.unsqueeze(0).expand(batch_size, -1)
            
            # Forward with accumulated KV cache
            outputs = self.model(
                input_ids=chunk_ids,
                attention_mask=chunk_mask,
                position_ids=chunk_positions,
                past_key_values=kv_cache,
                use_cache=True,
                output_hidden_states=True,
            )
            
            kv_cache = outputs.past_key_values
            
            if hasattr(outputs, "hidden_states"):
                all_hidden_states.append(outputs.hidden_states[-1])
        
        # Concatenate hidden states
        hidden_states = torch.cat(all_hidden_states, dim=1) if all_hidden_states else None
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        
        final_positions = torch.arange(seq_len, device=input_ids.device)
        final_positions = final_positions.unsqueeze(0).expand(batch_size, -1)
        
        return PrefillOutput(
            hidden_states=hidden_states,
            kv_cache=list(kv_cache) if kv_cache else [],
            logits=logits,
            positions=final_positions,
        )
    
    def estimate_memory(self, seq_len: int, batch_size: int = 1) -> Dict[str, int]:
        """
        Estimate memory usage for prefill.
        
        Returns memory in bytes for:
        - Activations
        - KV cache
        - Model parameters
        """
        # Rough estimates
        hidden_size = getattr(self.model.config, "hidden_size", 4096)
        num_layers = getattr(self.model.config, "num_hidden_layers", 32)
        num_heads = getattr(self.model.config, "num_attention_heads", 32)
        head_dim = hidden_size // num_heads
        
        dtype_size = 2  # BF16
        
        # KV cache per layer
        kv_per_layer = 2 * batch_size * seq_len * num_heads * head_dim * dtype_size
        total_kv = kv_per_layer * num_layers
        
        # Activations (rough estimate)
        activations = batch_size * seq_len * hidden_size * dtype_size * 4  # Factor for intermediate
        
        if self.chunked:
            activations = batch_size * self.chunk_size * hidden_size * dtype_size * 4
        
        return {
            "kv_cache_bytes": total_kv,
            "activation_bytes": activations,
            "total_bytes": total_kv + activations,
        }
