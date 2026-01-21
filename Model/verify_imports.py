"""
Quick Import Verification Script
Tests all critical imports to ensure no missing exports
"""

def test_imports():
    """Test all package imports"""
    
    print("Testing imports...")
    
    # Test kernel imports
    print("\n1. Testing kernel imports...")
    try:
        from Model.kernels.triton import (
            ssm_scan_fwd,
            rms_norm_linear,
            conv1d_silu,
            rope_embedding,
            precompute_freqs_cis,
            flash_attention_gqa,
            topk_gating,
            swiglu_expert,
        )
        print("   ✓ All kernel imports successful")
    except ImportError as e:
        print(f"   ✗ Kernel import error: {e}")
        return False
    
    # Test module imports
    print("\n2. Testing module imports...")
    try:
        from Model.modules import (
            SelectiveSSM,
            HybridAttention,
            HierarchicalMoE,
            DualHeadPredictor,
            HybridResidualBlock,
            RMSNorm,
        )
        print("   ✓ All module imports successful")
    except ImportError as e:
        print(f"   ✗ Module import error: {e}")
        return False
    
    # Test config imports
    print("\n3. Testing config imports...")
    try:
        from Model import ModelConfig, get_config_1B, get_config_7B, get_config_70B
        print("   ✓ All config imports successful")
    except ImportError as e:
        print(f"   ✗ Config import error: {e}")
        return False
    
    # Test model import
    print("\n4. Testing model imports...")
    try:
        from Model import HNS_SS_JEPA_MoE, create_model
        print("   ✓ Model imports successful")
    except ImportError as e:
        print(f"   ✗ Model import error: {e}")
        return False
    
    # Test model creation (without GPU)
    print("\n5. Testing model instantiation...")
    try:
        import torch
        config = ModelConfig(
            vocab_size=1000,
            d_model=256,
            n_layers=4,
            n_heads=4,
            n_kv_heads=2,
            n_experts=4,
            ssm_blocks_per_attn=1,
        )
        model = HNS_SS_JEPA_MoE(config)
        print(f"   ✓ Model created: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
    except Exception as e:
        print(f"   ✗ Model creation error: {e}")
        return False
    
    # Test forward pass (CPU)
    print("\n6. Testing forward pass (CPU)...")
    try:
        input_ids = torch.randint(0, 1000, (1, 32))
        with torch.no_grad():
            outputs = model(input_ids, mode='medusa')
        print(f"   ✓ Forward pass successful")
        print(f"   - Hidden states shape: {outputs['hidden_states'].shape}")
        print(f"   - Medusa heads: {len(outputs['medusa']['logits'])}")
    except Exception as e:
        print(f"   ✗ Forward pass error: {e}")
        return False
    
    print("\n" + "="*50)
    print("✓ All tests passed! Package is functional.")
    print("="*50)
    return True


if __name__ == '__main__':
    import sys
    success = test_imports()
    sys.exit(0 if success else 1)
