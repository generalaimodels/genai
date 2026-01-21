"""
Decode Engine
=============
Token-by-token decode phase with PagedAttention.
"""

from typing import Optional, Dict, Tuple, List, Any
from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .config import InferenceConfig, SamplingConfig


@dataclass
class DecodeOutput:
    """Output from decode step."""
    token_ids: "torch.Tensor"
    logits: "torch.Tensor"
    kv_cache: List[Tuple["torch.Tensor", "torch.Tensor"]]
    finished: "torch.Tensor"
    log_probs: Optional["torch.Tensor"] = None


class DecodeEngine:
    """
    Decode engine for autoregressive generation.
    
    Compute pattern: Matrix-Vector (GEMV) - memory bound
    
    Optimizations:
    - PagedAttention for KV cache
    - Triton sampling kernels
    - Batched token-by-token generation
    """
    
    def __init__(
        self,
        model: "nn.Module",
        config: InferenceConfig,
        tokenizer: Any = None,
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch required")
        
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.device = config.device
        
        # Get vocab size
        if hasattr(model, "config"):
            self.vocab_size = getattr(model.config, "vocab_size", 32000)
        else:
            self.vocab_size = 32000
        
        # Stop tokens
        self.eos_token_id = getattr(tokenizer, "eos_token_id", 2) if tokenizer else 2
    
    @torch.inference_mode()
    def decode_step(
        self,
        input_ids: torch.Tensor,
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        position_ids: torch.Tensor,
        sampling_config: Optional[SamplingConfig] = None,
    ) -> DecodeOutput:
        """
        Single decode step.
        
        Args:
            input_ids: (batch, 1) last generated token
            kv_cache: List of (key, value) tuples per layer
            position_ids: (batch, 1) current position
            sampling_config: Sampling parameters
            
        Returns:
            DecodeOutput with next token and updated cache
        """
        sampling = sampling_config or self.config.sampling
        
        # Forward pass for single token
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=kv_cache,
            use_cache=True,
        )
        
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        logits = logits[:, -1, :]  # (batch, vocab)
        
        new_kv_cache = outputs.past_key_values if hasattr(outputs, "past_key_values") else kv_cache
        
        # Sample next token
        next_tokens, log_probs = self._sample(logits, sampling)
        
        # Check for EOS
        finished = next_tokens == self.eos_token_id
        
        return DecodeOutput(
            token_ids=next_tokens,
            logits=logits,
            kv_cache=list(new_kv_cache),
            finished=finished,
            log_probs=log_probs,
        )
    
    def _sample(
        self,
        logits: torch.Tensor,
        config: SamplingConfig,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample next token from logits.
        
        Supports:
        - Temperature scaling
        - Top-k filtering
        - Top-p (nucleus) filtering
        - Min-p filtering
        - Repetition penalty
        """
        batch_size = logits.size(0)
        
        # Temperature
        if config.temperature != 1.0:
            logits = logits / config.temperature
        
        # Top-k filtering
        if config.top_k > 0:
            logits = self._top_k_filter(logits, config.top_k)
        
        # Top-p filtering
        if config.top_p < 1.0:
            logits = self._top_p_filter(logits, config.top_p)
        
        # Min-p filtering
        if config.min_p > 0.0:
            logits = self._min_p_filter(logits, config.min_p)
        
        # Sample
        probs = F.softmax(logits, dim=-1)
        
        if config.temperature == 0 or config.num_beams > 1:
            # Greedy
            next_tokens = logits.argmax(dim=-1)
        else:
            # Multinomial sampling
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        # Log probs
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(-1, next_tokens.unsqueeze(-1)).squeeze(-1)
        
        return next_tokens, selected_log_probs
    
    def _top_k_filter(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """Apply top-k filtering."""
        indices_to_remove = logits < torch.topk(logits, k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, float("-inf"))
        return logits
    
    def _top_p_filter(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        """Apply nucleus (top-p) filtering."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative prob above threshold
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float("-inf"))
        return logits
    
    def _min_p_filter(self, logits: torch.Tensor, min_p: float) -> torch.Tensor:
        """Apply min-p filtering."""
        probs = F.softmax(logits, dim=-1)
        max_prob = probs.max(dim=-1, keepdim=True)[0]
        threshold = max_prob * min_p
        indices_to_remove = probs < threshold
        logits = logits.masked_fill(indices_to_remove, float("-inf"))
        return logits
    
    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        max_new_tokens: Optional[int] = None,
        sampling_config: Optional[SamplingConfig] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full generation loop.
        
        Args:
            input_ids: (batch, seq_len) starting tokens
            kv_cache: Optional pre-computed KV cache
            max_new_tokens: Max tokens to generate
            sampling_config: Sampling parameters
            
        Returns:
            Dict with generated token_ids, log_probs, num_tokens
        """
        sampling = sampling_config or self.config.sampling
        max_tokens = max_new_tokens or sampling.max_tokens
        
        batch_size, seq_len = input_ids.shape
        
        # Initialize position
        current_pos = seq_len
        
        # Generated tokens
        generated_tokens = []
        generated_log_probs = []
        
        # Finished mask
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        # Current token
        current_token = input_ids[:, -1:]
        current_kv = kv_cache
        
        for _ in range(max_tokens):
            position_ids = torch.full(
                (batch_size, 1),
                current_pos,
                dtype=torch.long,
                device=self.device,
            )
            
            output = self.decode_step(
                current_token,
                current_kv,
                position_ids,
                sampling,
            )
            
            # Update
            current_token = output.token_ids.unsqueeze(-1)
            current_kv = output.kv_cache
            current_pos += 1
            
            # Store
            generated_tokens.append(output.token_ids)
            if output.log_probs is not None:
                generated_log_probs.append(output.log_probs)
            
            # Update finished
            finished = finished | output.finished
            
            # Early exit
            if finished.all():
                break
        
        # Stack outputs
        token_ids = torch.stack(generated_tokens, dim=1)
        log_probs = torch.stack(generated_log_probs, dim=1) if generated_log_probs else None
        
        return {
            "token_ids": token_ids,
            "log_probs": log_probs,
            "num_tokens": token_ids.size(1),
            "finished": finished,
        }
