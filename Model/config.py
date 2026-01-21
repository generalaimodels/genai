"""
HNS-SS-JEPA-MoE Model Configuration
Hyperparameters for model architecture
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Model hyperparameters"""
    
    # Model dimensions
    vocab_size: int = 128000  # Tokenizer vocabulary size
    d_model: int = 4096       # Hidden dimension
    n_layers: int = 48        # Total layer depth
    
    # SSM parameters
    d_state: int = 16         # SSM state dimension
    d_conv: int = 4           # Conv1D kernel size
    ssm_expand: int = 2       # SSM expansion factor
    
    # Attention parameters
    n_heads: int = 32         # Query heads
    n_kv_heads: int = 8       # KV heads (GQA ratio: 4:1)
    window_size: int = 4096   # Sliding window size
    max_seq_len: int = 8192   # Maximum sequence length
    
    # MoE parameters
    n_experts: int = 16       # Total experts
    topk_experts: int = 2     # Active experts per token
    moe_ff_ratio: int = 4     # FFN expansion ratio
    load_balance_weight: float = 0.01  # Auxiliary loss weight
    
    # World Model parameters
    d_latent: int = 4096      # JEPA latent dimension
    n_future_jepa: int = 1    # Future steps to predict
    n_heads_medusa: int = 3   # Medusa decoding heads
    
    # Architecture pattern
    ssm_blocks_per_attn: int = 3  # SSM blocks before each Attention block
    
    # Regularization
    dropout: float = 0.1
    
    # Training
    gradient_checkpointing: bool = True
    
    @property
    def head_dim(self):
        """Attention head dimension"""
        return self.d_model // self.n_heads
    
    @property
    def total_blocks(self):
        """Total number of residual blocks"""
        return self.n_layers
    
    def validate(self):
        """Validate configuration constraints"""
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        
        assert self.n_heads % self.n_kv_heads == 0, \
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads}) for GQA"
        
        assert self.n_layers % (self.ssm_blocks_per_attn + 1) == 0, \
            f"n_layers ({self.n_layers}) must be divisible by (ssm_blocks_per_attn + 1)"


# Preset configurations
def get_config_1B():
    """1B parameter model"""
    return ModelConfig(
        vocab_size=128000,
        d_model=2048,
        n_layers=24,
        n_heads=16,
        n_kv_heads=4,
        n_experts=8,
    )


def get_config_7B():
    """7B parameter model"""
    return ModelConfig(
        vocab_size=128000,
        d_model=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        n_experts=16,
    )


def get_config_70B():
    """70B parameter model"""
    return ModelConfig(
        vocab_size=128000,
        d_model=8192,
        n_layers=80,
        n_heads=64,
        n_kv_heads=16,
        n_experts=64,
        topk_experts=4,
    )
