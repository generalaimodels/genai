"""
Example Usage: HNS-SS-JEPA-MoE Model
Demonstrates training and inference workflows
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from Model import create_model, ModelConfig


def example_training():
    """Example training loop"""
    
    # Create model
    model = create_model(size='1B').cuda()
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    
    # Dummy data
    batch_size = 4
    seq_len = 512
    
    for step in range(100):
        # Generate random batch
        input_ids = torch.randint(0, 128000, (batch_size, seq_len)).cuda()
        target_ids = torch.randint(0, 128000, (batch_size, seq_len)).cuda()
        
        # Forward pass (language modeling)
        outputs = model(input_ids, mode='medusa')
        
        # Compute loss (first Medusa head)
        logits = outputs['medusa']['logits'][0]  # [B, L, V]
        lm_loss = nn.functional.cross_entropy(
            logits.view(-1, 128000),
            target_ids.view(-1),
            ignore_index=-100
        )
        
        # MoE auxiliary loss
        aux_loss = outputs['aux_loss']
        
        # Total loss
        loss = lm_loss + aux_loss
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 10 == 0:
            print(f"Step {step}: LM Loss = {lm_loss.item():.4f}, "
                  f"Aux Loss = {aux_loss.item():.6f}")


def example_world_model_training():
    """Example world model (JEPA) training"""
    
    model = create_model(size='1B').cuda()
    optimizer = AdamW(model.parameters(), lr=1e-4)
    
    batch_size = 4
    seq_len = 512
    n_future = 1
    
    for step in range(100):
        # Current sequence
        input_ids = torch.randint(0, 128000, (batch_size, seq_len)).cuda()
        
        # Forward to get hidden states
        outputs = model(input_ids, mode='both')
        
        # Future hidden states (for JEPA target)
        # In practice: encode next segment with target encoder
        x_future = torch.randn(batch_size, n_future, 2048).cuda()
        
        # Get JEPA loss
        jepa_loss = outputs['jepa']['loss']
        
        # Language modeling loss
        logits = outputs['medusa']['logits'][0]
        target_ids = torch.randint(0, 128000, (batch_size, seq_len)).cuda()
        lm_loss = nn.functional.cross_entropy(
            logits.view(-1, 128000),
            target_ids.view(-1)
        )
        
        # Combined loss
        loss = lm_loss + 0.1 * jepa_loss + outputs['aux_loss']
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update target encoder (EMA)
        # In practice: update predictor.jepa.target_encoder
        
        if step % 10 == 0:
            print(f"Step {step}: LM = {lm_loss.item():.4f}, "
                  f"JEPA = {jepa_loss.item():.4f}")


def example_inference():
    """Example inference with generation"""
    
    model = create_model(size='1B').cuda()
    model.eval()
    
    # Prompt
    prompt = "The future of AI is"
    # In practice: use tokenizer
    prompt_ids = torch.randint(0, 128000, (1, 10)).cuda()
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            prompt_ids,
            max_new_tokens=100,
            temperature=0.8,
            top_k=50,
            use_medusa=True,
        )
    
    print(f"Generated {generated_ids.shape[1]} tokens")
    # In practice: decode with tokenizer


def example_custom_config():
    """Example with custom configuration"""
    
    # Custom config
    config = ModelConfig(
        vocab_size=50000,
        d_model=1024,
        n_layers=16,
        n_heads=16,
        n_kv_heads=4,
        n_experts=8,
        topk_experts=2,
        window_size=2048,
    )
    
    config.validate()
    
    from Model import HNS_SS_JEPA_MoE
    model = HNS_SS_JEPA_MoE(config).cuda()
    
    print(f"Custom model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")


def example_memory_efficient_training():
    """Example with gradient checkpointing for large models"""
    
    # 7B model with checkpointing
    config = ModelConfig(
        d_model=4096,
        n_layers=32,
        gradient_checkpointing=True  # Enable checkpointing
    )
    
    from Model import HNS_SS_JEPA_MoE
    model = HNS_SS_JEPA_MoE(config).cuda()
    
    # Training loop (same as above)
    print("Training with gradient checkpointing enabled")


if __name__ == '__main__':
    print("=== Example 1: Standard Training ===")
    example_training()
    
    print("\n=== Example 2: World Model Training ===")
    example_world_model_training()
    
    print("\n=== Example 3: Inference ===")
    example_inference()
    
    print("\n=== Example 4: Custom Config ===")
    example_custom_config()
