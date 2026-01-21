"""
PagedAttention
==============
Block-based KV cache allocation for efficient memory management.
"""

from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


@dataclass
class KVBlock:
    """Single KV cache block."""
    block_id: int
    key: "torch.Tensor"      # (block_size, num_heads, head_dim)
    value: "torch.Tensor"    # (block_size, num_heads, head_dim)
    num_tokens: int = 0
    ref_count: int = 0
    
    @property
    def is_full(self) -> bool:
        return self.num_tokens >= self.key.size(0)
    
    @property
    def available_space(self) -> int:
        return self.key.size(0) - self.num_tokens


class PagedKVCache:
    """
    PagedAttention KV cache implementation.
    
    Key features:
    - Block-based allocation (like OS virtual memory)
    - Copy-on-write for beam search
    - Memory fragmentation prevention
    - Dynamic memory allocation
    """
    
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        block_size: int = 16,
        num_gpu_blocks: int = 1024,
        num_cpu_blocks: int = 256,
        dtype: "torch.dtype" = None,
        device: str = "cuda",
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch required")
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.dtype = dtype or torch.bfloat16
        self.device = device
        
        # Pre-allocate GPU block pool per layer
        self.gpu_key_blocks = torch.zeros(
            (num_layers, num_gpu_blocks, block_size, num_heads, head_dim),
            dtype=self.dtype,
            device=device,
        )
        self.gpu_value_blocks = torch.zeros(
            (num_layers, num_gpu_blocks, block_size, num_heads, head_dim),
            dtype=self.dtype,
            device=device,
        )
        
        # CPU swap space
        self.cpu_key_blocks = torch.zeros(
            (num_layers, num_cpu_blocks, block_size, num_heads, head_dim),
            dtype=self.dtype,
            device="cpu",
            pin_memory=True,
        )
        self.cpu_value_blocks = torch.zeros(
            (num_layers, num_cpu_blocks, block_size, num_heads, head_dim),
            dtype=self.dtype,
            device="cpu",
            pin_memory=True,
        )
        
        # Free block indices
        self.free_gpu_blocks = list(range(num_gpu_blocks))
        self.free_cpu_blocks = list(range(num_cpu_blocks))
        
        # Block tables per sequence: seq_id -> List[block_id per layer]
        self.block_tables: Dict[int, List[List[int]]] = {}
        
        # Reference counts for copy-on-write
        self.block_ref_counts = torch.zeros(num_gpu_blocks, dtype=torch.int32, device=device)
    
    def allocate_block(self, layer_idx: int) -> Optional[int]:
        """Allocate a free GPU block."""
        if not self.free_gpu_blocks:
            return None
        
        block_id = self.free_gpu_blocks.pop()
        self.block_ref_counts[block_id] = 1
        
        return block_id
    
    def free_block(self, block_id: int):
        """Free a block, handling reference counting."""
        self.block_ref_counts[block_id] -= 1
        
        if self.block_ref_counts[block_id] <= 0:
            self.block_ref_counts[block_id] = 0
            self.free_gpu_blocks.append(block_id)
    
    def allocate_sequence(self, seq_id: int, num_tokens: int) -> bool:
        """
        Allocate blocks for a new sequence.
        
        Returns True if allocation succeeded.
        """
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        
        if len(self.free_gpu_blocks) < num_blocks_needed * self.num_layers:
            return False
        
        block_table = []
        for layer_idx in range(self.num_layers):
            layer_blocks = []
            for _ in range(num_blocks_needed):
                block_id = self.allocate_block(layer_idx)
                if block_id is None:
                    # Rollback
                    self._rollback_allocation(block_table)
                    return False
                layer_blocks.append(block_id)
            block_table.append(layer_blocks)
        
        self.block_tables[seq_id] = block_table
        return True
    
    def _rollback_allocation(self, block_table: List[List[int]]):
        """Rollback partial allocation."""
        for layer_blocks in block_table:
            for block_id in layer_blocks:
                self.free_block(block_id)
    
    def free_sequence(self, seq_id: int):
        """Free all blocks for a sequence."""
        if seq_id not in self.block_tables:
            return
        
        for layer_blocks in self.block_tables[seq_id]:
            for block_id in layer_blocks:
                self.free_block(block_id)
        
        del self.block_tables[seq_id]
    
    def append_token(
        self,
        seq_id: int,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> bool:
        """
        Append KV for new token.
        
        Args:
            seq_id: Sequence ID
            layer_idx: Layer index
            key: (1, num_heads, head_dim)
            value: (1, num_heads, head_dim)
            
        Returns True if successful.
        """
        if seq_id not in self.block_tables:
            return False
        
        block_table = self.block_tables[seq_id][layer_idx]
        
        # Find current block
        # Calculate slot within last block
        current_block_id = block_table[-1]
        
        # Copy-on-write if shared
        if self.block_ref_counts[current_block_id] > 1:
            new_block_id = self.allocate_block(layer_idx)
            if new_block_id is None:
                return False
            
            # Copy data
            self.gpu_key_blocks[layer_idx, new_block_id] = self.gpu_key_blocks[layer_idx, current_block_id]
            self.gpu_value_blocks[layer_idx, new_block_id] = self.gpu_value_blocks[layer_idx, current_block_id]
            
            # Update ref counts
            self.block_ref_counts[current_block_id] -= 1
            block_table[-1] = new_block_id
            current_block_id = new_block_id
        
        # Append to block
        slot = 0  # Would track actual position
        self.gpu_key_blocks[layer_idx, current_block_id, slot] = key
        self.gpu_value_blocks[layer_idx, current_block_id, slot] = value
        
        return True
    
    def get_kv_cache(
        self,
        seq_id: int,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get KV cache for sequence and layer.
        
        Returns concatenated key/value tensors.
        """
        if seq_id not in self.block_tables:
            raise ValueError(f"Sequence {seq_id} not found")
        
        block_ids = self.block_tables[seq_id][layer_idx]
        
        keys = [self.gpu_key_blocks[layer_idx, bid] for bid in block_ids]
        values = [self.gpu_value_blocks[layer_idx, bid] for bid in block_ids]
        
        key = torch.cat(keys, dim=0)
        value = torch.cat(values, dim=0)
        
        return key, value
    
    def swap_out(self, seq_id: int):
        """Swap sequence KV cache to CPU."""
        if seq_id not in self.block_tables:
            return
        
        if not self.free_cpu_blocks:
            return
        
        for layer_idx, layer_blocks in enumerate(self.block_tables[seq_id]):
            for i, gpu_block_id in enumerate(layer_blocks):
                if not self.free_cpu_blocks:
                    return
                
                cpu_block_id = self.free_cpu_blocks.pop()
                
                # Async copy
                self.cpu_key_blocks[layer_idx, cpu_block_id].copy_(
                    self.gpu_key_blocks[layer_idx, gpu_block_id],
                    non_blocking=True,
                )
                self.cpu_value_blocks[layer_idx, cpu_block_id].copy_(
                    self.gpu_value_blocks[layer_idx, gpu_block_id],
                    non_blocking=True,
                )
                
                # Free GPU block
                self.free_block(gpu_block_id)
                layer_blocks[i] = -cpu_block_id  # Negative = CPU
    
    def swap_in(self, seq_id: int):
        """Swap sequence KV cache back to GPU."""
        if seq_id not in self.block_tables:
            return
        
        for layer_idx, layer_blocks in enumerate(self.block_tables[seq_id]):
            for i, block_id in enumerate(layer_blocks):
                if block_id >= 0:
                    continue  # Already on GPU
                
                cpu_block_id = -block_id
                gpu_block_id = self.allocate_block(layer_idx)
                
                if gpu_block_id is None:
                    return
                
                # Async copy
                self.gpu_key_blocks[layer_idx, gpu_block_id].copy_(
                    self.cpu_key_blocks[layer_idx, cpu_block_id],
                    non_blocking=True,
                )
                self.gpu_value_blocks[layer_idx, gpu_block_id].copy_(
                    self.cpu_value_blocks[layer_idx, cpu_block_id],
                    non_blocking=True,
                )
                
                # Free CPU block
                self.free_cpu_blocks.append(cpu_block_id)
                layer_blocks[i] = gpu_block_id


if HAS_TRITON:
    @triton.jit
    def paged_attention_kernel(
        out_ptr,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        block_tables_ptr,
        seq_lens_ptr,
        scale,
        BLOCK_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        NUM_BLOCKS: tl.constexpr,
    ):
        """
        Triton kernel for PagedAttention.
        
        Reads KV from paged blocks and computes attention.
        """
        batch_id = tl.program_id(0)
        head_id = tl.program_id(1)
        
        # Load query
        q_offset = batch_id * HEAD_DIM + tl.arange(0, HEAD_DIM)
        query = tl.load(query_ptr + q_offset)
        
        # Initialize accumulator
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
        m = tl.zeros([1], dtype=tl.float32) - float("inf")
        l = tl.zeros([1], dtype=tl.float32)
        
        seq_len = tl.load(seq_lens_ptr + batch_id)
        num_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        # Iterate over blocks
        for block_idx in range(num_blocks):
            block_id = tl.load(block_tables_ptr + batch_id * NUM_BLOCKS + block_idx)
            
            # Load key block
            k_offset = block_id * BLOCK_SIZE * HEAD_DIM
            keys = tl.load(key_cache_ptr + k_offset + tl.arange(0, BLOCK_SIZE * HEAD_DIM))
            keys = tl.reshape(keys, [BLOCK_SIZE, HEAD_DIM])
            
            # Compute attention scores
            scores = tl.sum(query[None, :] * keys, axis=1) * scale
            
            # Online softmax
            m_new = tl.maximum(m, tl.max(scores))
            p = tl.exp(scores - m_new)
            l_new = tl.exp(m - m_new) * l + tl.sum(p)
            
            # Load value block
            values = tl.load(value_cache_ptr + k_offset + tl.arange(0, BLOCK_SIZE * HEAD_DIM))
            values = tl.reshape(values, [BLOCK_SIZE, HEAD_DIM])
            
            # Accumulate
            acc = acc * tl.exp(m - m_new) + tl.sum(p[:, None] * values, axis=0)
            
            m = m_new
            l = l_new
        
        # Final normalization
        out = acc / l
        
        out_offset = batch_id * HEAD_DIM + tl.arange(0, HEAD_DIM)
        tl.store(out_ptr + out_offset, out)


class PagedAttentionKernel:
    """Wrapper for PagedAttention computation."""
    
    @staticmethod
    def forward(
        query: torch.Tensor,
        kv_cache: PagedKVCache,
        seq_ids: List[int],
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Compute attention with paged KV cache.
        
        Args:
            query: (batch, num_heads, head_dim)
            kv_cache: PagedKVCache instance
            seq_ids: List of sequence IDs
            layer_idx: Current layer index
            
        Returns:
            Attention output (batch, num_heads, head_dim)
        """
        batch_size = query.size(0)
        num_heads = query.size(1)
        head_dim = query.size(2)
        
        outputs = []
        
        for i, seq_id in enumerate(seq_ids):
            key, value = kv_cache.get_kv_cache(seq_id, layer_idx)
            
            # Standard attention
            q = query[i]  # (num_heads, head_dim)
            k = key  # (seq_len, num_heads, head_dim)
            v = value  # (seq_len, num_heads, head_dim)
            
            # Reshape for matmul
            q = q.unsqueeze(1)  # (num_heads, 1, head_dim)
            k = k.permute(1, 2, 0)  # (num_heads, head_dim, seq_len)
            v = v.permute(1, 0, 2)  # (num_heads, seq_len, head_dim)
            
            # Attention
            scale = head_dim ** -0.5
            scores = torch.matmul(q, k) * scale  # (num_heads, 1, seq_len)
            attn = F.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)  # (num_heads, 1, head_dim)
            
            outputs.append(out.squeeze(1))
        
        return torch.stack(outputs, dim=0)
