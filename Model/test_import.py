"""
Quick Test: Verify Model Import
Run this from the parent directory of Model folder
"""

# Test basic imports
print("Testing imports from Model package...")

try:
    from Model import create_model
    print("✓ create_model imported successfully")
    
    # Create small test model
    model = create_model(size='1B')
    print(f"✓ 1B model created: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    
    # Test forward pass
    import torch
    input_ids = torch.randint(0, 128000, (1, 16))
    
    with torch.no_grad():
        outputs = model(input_ids, mode='medusa')
    
    print(f"✓ Forward pass successful")
    print(f"  - Output shape: {outputs['hidden_states'].shape}")
    print(f"  - Medusa heads: {len(outputs['medusa']['logits'])}")
    print(f"  - Aux loss: {outputs['aux_loss'].item():.6f}")
    
    print("\n✅ All imports working correctly!")
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("\nMake sure you're running from the parent directory of 'Model' folder")
    print("Example: cd c:/Users/heman/Desktop/code/workspace && python Model/test_import.py")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
