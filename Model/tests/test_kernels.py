"""
Test Suite for Kernel Validation
Numerical correctness and performance benchmarks
"""

import torch
import pytest
import time
from Model import create_model
from Model.kernels.triton import (
    ssm_scan_fwd,
    rms_norm_linear,
    flash_attention_gqa,
    topk_gating,
)


@pytest.fixture
def device():
    """Use CUDA if available, else CPU"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TestSSMKernel:
    """Test Selective State Space kernels"""
    
    def test_ssm_scan_shape(self, device):
        """Verify output shapes"""
        B, L, D, N = 2, 128, 512, 16
        
        x = torch.randn(B, L, D, device=device)
        delta = torch.randn(B, L, N, device=device)
        A = torch.randn(N, device=device)
        B_ssm = torch.randn(B, L, N, device=device)
        C = torch.randn(B, L, N, device=device)
        gate = torch.randn(B, L, D, device=device)
        
        y, h = ssm_scan_fwd(x, delta, A, B_ssm, C, gate)
        
        assert y.shape == (B, L, D)
        assert h.shape == (B, L, N)
    
    def test_ssm_scan_causality(self, device):
        """Verify causal property: y_t doesn't depend on x_{t'>t}"""
        B, L, D, N = 1, 64, 256, 16
        
        # Create input
        x = torch.randn(B, L, D, device=device)
        delta = torch.randn(B, L, N, device=device)
        A = torch.randn(N, device=device)
        B_ssm = torch.randn(B, L, N, device=device)
        C = torch.randn(B, L, N, device=device)
        gate = torch.randn(B, L, D, device=device)
        
        # Forward pass
        y1, _ = ssm_scan_fwd(x, delta, A, B_ssm, C, gate)
        
        # Modify future tokens
        x_modified = x.clone()
        x_modified[:, 32:, :] += 10.0
        
        y2, _ = ssm_scan_fwd(x_modified, delta, A, B_ssm, C, gate)
        
        # First 32 tokens should be identical
        assert torch.allclose(y1[:, :32, :], y2[:, :32, :], atol=1e-5)


class TestAttentionKernel:
    """Test Flash Attention kernels"""
    
    def test_gqa_flash_attention_shape(self, device):
        """Verify output shapes for GQA"""
        B, H_q, H_kv, L, D = 2, 32, 8, 256, 64
        
        Q = torch.randn(B, H_q, L, D, device=device)
        K = torch.randn(B, H_kv, L, D, device=device)
        V = torch.randn(B, H_kv, L, D, device=device)
        
        Out, LSE = flash_attention_gqa(Q, K, V, window_size=128)
        
        assert Out.shape == (B, H_q, L, D)
        assert LSE.shape == (B, H_q, L)
    
    def test_attention_causality(self, device):
        """Verify causal masking"""
        B, H_q, H_kv, L, D = 1, 8, 4, 64, 64
        
        Q = torch.randn(B, H_q, L, D, device=device)
        K = torch.randn(B, H_kv, L, D, device=device)
        V = torch.randn(B, H_kv, L, D, device=device)
        
        Out, _ = flash_attention_gqa(Q, K, V, window_size=64)
        
        # Modify future values
        V_modified = V.clone()
        V_modified[:, :, 32:, :] += 10.0
        
        Out_modified, _ = flash_attention_gqa(Q, K, V_modified, window_size=64)
        
        # First 32 tokens should be identical
        assert torch.allclose(Out[:, :, :32, :], Out_modified[:, :, :32, :], atol=1e-4)


class TestMoEKernel:
    """Test MoE routing kernels"""
    
    def test_topk_gating_routing(self, device):
        """Verify top-k selection"""
        B_L, D, N_experts, K = 128, 512, 16, 2
        
        x = torch.randn(B_L, D, device=device)
        W_g = torch.randn(D, N_experts, device=device)
        
        indices, scores, load = topk_gating(x, W_g, k=K)
        
        assert indices.shape == (B_L, K)
        assert scores.shape == (B_L, K)
        assert load.shape == (N_experts,)
        
        # Scores should sum to 1 (softmax)
        assert torch.allclose(scores.sum(dim=1), torch.ones(B_L, device=device), atol=1e-5)
        
        # Load should sum to B_L * K
        assert load.sum() == B_L * K


class TestEndToEnd:
    """End-to-end model tests"""
    
    def test_forward_pass_1B(self, device):
        """Test 1B model forward pass"""
        if device.type == 'cpu':
            pytest.skip("Skipping on CPU (slow)")
        
        model = create_model(size='1B').to(device)
        
        B, L = 2, 128
        input_ids = torch.randint(0, 128000, (B, L), device=device)
        
        outputs = model(input_ids, mode='medusa')
        
        # Check shapes
        assert outputs['hidden_states'].shape == (B, L, 2048)
        assert len(outputs['medusa']['logits']) == 3  # 3 Medusa heads
        assert outputs['medusa']['logits'][0].shape == (B, L, 128000)
    
    def test_generation(self, device):
        """Test autoregressive generation"""
        if device.type == 'cpu':
            pytest.skip("Skipping on CPU (slow)")
        
        model = create_model(size='1B').to(device)
        
        prompt = torch.randint(0, 128000, (1, 32), device=device)
        
        generated = model.generate(
            prompt,
            max_new_tokens=10,
            temperature=1.0,
            top_k=50,
        )
        
        assert generated.shape == (1, 42)  # 32 + 10
    
    def test_gradient_flow(self, device):
        """Test backward pass and gradient flow"""
        if device.type == 'cpu':
            pytest.skip("Skipping on CPU (slow)")
        
        model = create_model(size='1B').to(device)
        
        input_ids = torch.randint(0, 128000, (1, 64), device=device)
        target_ids = torch.randint(0, 128000, (1, 64), device=device)
        
        # Forward
        outputs = model(input_ids, mode='medusa')
        logits = outputs['medusa']['logits'][0]
        
        # Loss
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, 128000),
            target_ids.view(-1)
        )
        
        # Backward
        loss.backward()
        
        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestNumericalStability:
    """Numerical stability tests"""
    
    def test_rms_norm_stability(self, device):
        """Test RMSNorm numerical stability"""
        from Model.modules.hybrid_block import RMSNorm
        
        norm = RMSNorm(512).to(device)
        
        # Large values
        x_large = torch.randn(2, 128, 512, device=device) * 1000
        y_large = norm(x_large)
        assert not torch.isnan(y_large).any()
        assert not torch.isinf(y_large).any()
        
        # Small values
        x_small = torch.randn(2, 128, 512, device=device) * 1e-6
        y_small = norm(x_small)
        assert not torch.isnan(y_small).any()
        assert not torch.isinf(y_small).any()


@pytest.mark.benchmark
class TestPerformance:
    """Performance benchmarks"""
    
    def test_ssm_throughput(self, device):
        """Benchmark SSM scan throughput"""
        if device.type == 'cpu':
            pytest.skip("Skipping on CPU")
        
        B, L, D, N = 4, 2048, 2048, 16
        
        x = torch.randn(B, L, D, device=device)
        delta = torch.randn(B, L, N, device=device)
        A = torch.randn(N, device=device)
        B_ssm = torch.randn(B, L, N, device=device)
        C = torch.randn(B, L, N, device=device)
        gate = torch.randn(B, L, D, device=device)
        
        # Warmup
        for _ in range(10):
            ssm_scan_fwd(x, delta, A, B_ssm, C, gate)
        
        torch.cuda.synchronize()
        
        # Benchmark
        n_iters = 100
        start = time.time()
        for _ in range(n_iters):
            ssm_scan_fwd(x, delta, A, B_ssm, C, gate)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        tokens_per_sec = (B * L * n_iters) / elapsed
        print(f"\nSSM Throughput: {tokens_per_sec/1000:.2f}k tokens/sec")
        
        assert tokens_per_sec > 50000  # Minimum 50k tokens/sec


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
