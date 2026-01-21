"""
HNS-SS-JEPA-MoE: End-to-End Model Architecture
Compound flow: Embedding -> Layer Stack -> Dual-Head Prediction
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from typing import Optional, Tuple, Dict

from .config import ModelConfig
from .modules import (
    SelectiveSSM,
    HybridAttention,
    HierarchicalMoE,
    DualHeadPredictor,
    HybridResidualBlock,
    RMSNorm,
)


class HNS_SS_JEPA_MoE(nn.Module):
    """
    Hybrid Architecture combining:
    - Selective State Spaces (Mamba-style)
    - Sliding Window Attention with RoPE
    - Hierarchical Mixture of Experts
    - Dual-Head Prediction (JEPA World Model + Medusa Language Model)
    
    Forward Pass:
    ŷ = MedusaHeads(RMSNorm(Σ Block_l(Embed(T))))
    
    Block Pattern: (SSM × N) -> Attention -> ... (repeated)
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        config.validate()
        self.config = config
        
        # Token embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        
        # Layer stack
        self.layers = nn.ModuleList()
        blocks_per_cycle = config.ssm_blocks_per_attn + 1  # N SSM + 1 Attention
        n_cycles = config.n_layers // blocks_per_cycle
        
        for cycle in range(n_cycles):
            # SSM blocks
            for _ in range(config.ssm_blocks_per_attn):
                self.layers.append(
                    HybridResidualBlock(
                        d_model=config.d_model,
                        mixer_type='ssm',
                        d_state=config.d_state,
                        d_conv=config.d_conv,
                        ssm_expand=config.ssm_expand,
                        n_experts=config.n_experts,
                        topk_experts=config.topk_experts,
                        moe_ff_ratio=config.moe_ff_ratio,
                        dropout=config.dropout,
                    )
                )
            
            # Attention block
            self.layers.append(
                HybridResidualBlock(
                    d_model=config.d_model,
                    mixer_type='attention',
                    n_heads=config.n_heads,
                    n_kv_heads=config.n_kv_heads,
                    window_size=config.window_size,
                    n_experts=config.n_experts,
                    topk_experts=config.topk_experts,
                    moe_ff_ratio=config.moe_ff_ratio,
                    dropout=config.dropout,
                )
            )
        
        # Final normalization
        self.norm = RMSNorm(config.d_model)
        
        # Dual prediction heads
        self.predictor = DualHeadPredictor(
            d_model=config.d_model,
            vocab_size=config.vocab_size,
            d_latent=config.d_latent,
            n_future_jepa=config.n_future_jepa,
            n_heads_medusa=config.n_heads_medusa,
        )
        
        # Weight tying (share embedding with first Medusa head)
        self.predictor.medusa.heads[0][-1].weight = self.embed_tokens.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Kaiming initialization"""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        x_future: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        mode: str = 'medusa',  # 'jepa', 'medusa', or 'both'
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            input_ids: [B, L] token indices
            x_future: [B, n_future, D] future hidden states (training JEPA)
            use_cache: whether to use KV-cache for attention (inference)
            mode: prediction head mode
            
        Returns:
            outputs: dict containing:
                - 'hidden_states': [B, L, D] final hidden states
                - 'medusa': {'logits': list of [B, L, V]}
                - 'jepa': {'z_pred': [B, L, n_future, D_latent], 'loss': scalar}
                - 'aux_loss': MoE load balancing loss
        """
        B, L = input_ids.shape
        
        # Embedding
        x = self.embed_tokens(input_ids)  # [B, L, D]
        
        # Layer stack
        aux_loss_total = 0.0
        
        for layer_idx, layer in enumerate(self.layers):
            if self.config.gradient_checkpointing and self.training:
                # Gradient checkpointing for memory efficiency
                x, aux_loss = checkpoint.checkpoint(
                    layer, x, use_cache, 
                    use_reentrant=False
                )
            else:
                x, aux_loss = layer(x, use_cache=use_cache)
            
            aux_loss_total += aux_loss
        
        # Final normalization
        x = self.norm(x)  # [B, L, D]
        
        # Dual-head prediction
        pred_outputs = self.predictor(x, x_future, mode=mode)
        
        # Assemble outputs
        outputs = {
            'hidden_states': x,
            'aux_loss': aux_loss_total,
        }
        
        if 'medusa' in pred_outputs:
            outputs['medusa'] = pred_outputs['medusa']
        
        if 'jepa' in pred_outputs:
            outputs['jepa'] = pred_outputs['jepa']
        
        return outputs
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        use_medusa: bool = True,
    ) -> torch.Tensor:
        """
        Autoregressive generation with optional Medusa speculative decoding
        
        Args:
            input_ids: [B, L] prompt tokens
            max_new_tokens: maximum tokens to generate
            temperature: sampling temperature
            top_k: top-k sampling
            use_medusa: whether to use speculative decoding
            
        Returns:
            generated_ids: [B, L + max_new_tokens] generated sequence
        """
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                outputs = self.forward(input_ids, use_cache=True, mode='medusa')
                
                # Get logits from first Medusa head (t+1)
                logits = outputs['medusa']['logits'][0][:, -1, :]  # [B, V]
                logits = logits / temperature
                
                # Top-K sampling
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                    probs = torch.softmax(top_k_logits, dim=-1)
                    next_token_idx = torch.multinomial(probs, num_samples=1)
                    next_token = top_k_indices.gather(-1, next_token_idx)
                else:
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Speculative decoding with Medusa (accept/reject)
                if use_medusa and len(outputs['medusa']['logits']) > 1:
                    # Simplified: accept next Medusa predictions probabilistically
                    # Production: implement tree attention verification
                    pass
        
        return input_ids
    
    def reset_cache(self):
        """Reset KV-cache for all attention layers"""
        for layer in self.layers:
            if hasattr(layer.mixer, 'reset_cache'):
                layer.mixer.reset_cache()


# Convenience function
def create_model(size: str = '7B') -> HNS_SS_JEPA_MoE:
    """
    Create model from preset configuration
    
    Args:
        size: '1B', '7B', or '70B'
        
    Returns:
        model: HNS_SS_JEPA_MoE instance
    """
    from .config import get_config_1B, get_config_7B, get_config_70B
    
    config_map = {
        '1B': get_config_1B,
        '7B': get_config_7B,
        '70B': get_config_70B,
    }
    
    if size not in config_map:
        raise ValueError(f"Unknown size: {size}. Choose from {list(config_map.keys())}")
    
    config = config_map[size]()
    model = HNS_SS_JEPA_MoE(config)
    
    return model
