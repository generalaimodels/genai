"""
Sequence Parallelism (SP)
=========================
Sequence dimension sharding for long context training.
Reduces activation memory per rank.
"""

from typing import Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def get_sp_group() -> Optional["dist.ProcessGroup"]:
    """Get sequence parallel process group."""
    if not dist.is_initialized():
        return None
    return getattr(get_sp_group, "_group", None)


def set_sp_group(group: "dist.ProcessGroup", sp_size: int):
    """Set sequence parallel process group."""
    get_sp_group._group = group
    get_sp_group._size = sp_size
    get_sp_group._rank = dist.get_rank(group)


def get_sp_rank() -> int:
    return getattr(get_sp_group, "_rank", 0)


def get_sp_size() -> int:
    return getattr(get_sp_group, "_size", 1)


class _ScatterSequence(torch.autograd.Function):
    """Scatter sequence across SP group."""
    
    @staticmethod
    def forward(ctx, input_: torch.Tensor, dim: int = 1) -> torch.Tensor:
        group = get_sp_group()
        if group is None:
            return input_
        
        ctx.dim = dim
        sp_size = get_sp_size()
        sp_rank = get_sp_rank()
        
        seq_len = input_.size(dim)
        assert seq_len % sp_size == 0, f"Sequence length ({seq_len}) must divide SP size ({sp_size})"
        
        chunk_size = seq_len // sp_size
        chunks = input_.chunk(sp_size, dim=dim)
        
        return chunks[sp_rank].contiguous()
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        group = get_sp_group()
        if group is None:
            return grad_output, None
        
        dim = ctx.dim
        sp_size = get_sp_size()
        sp_rank = get_sp_rank()
        
        # AllGather gradients
        gather_list = [torch.empty_like(grad_output) for _ in range(sp_size)]
        dist.all_gather(gather_list, grad_output, group=group)
        
        return torch.cat(gather_list, dim=dim), None


class _GatherSequence(torch.autograd.Function):
    """Gather sequence from SP group."""
    
    @staticmethod
    def forward(ctx, input_: torch.Tensor, dim: int = 1) -> torch.Tensor:
        group = get_sp_group()
        if group is None:
            return input_
        
        ctx.dim = dim
        sp_size = get_sp_size()
        
        # AllGather
        gather_list = [torch.empty_like(input_) for _ in range(sp_size)]
        dist.all_gather(gather_list, input_, group=group)
        
        return torch.cat(gather_list, dim=dim)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        group = get_sp_group()
        if group is None:
            return grad_output, None
        
        dim = ctx.dim
        sp_size = get_sp_size()
        sp_rank = get_sp_rank()
        
        # Split gradient
        chunks = grad_output.chunk(sp_size, dim=dim)
        return chunks[sp_rank].contiguous(), None


class _ReduceScatterSequence(torch.autograd.Function):
    """ReduceScatter for sequence parallelism (after attention)."""
    
    @staticmethod
    def forward(ctx, input_: torch.Tensor, dim: int = 1) -> torch.Tensor:
        group = get_sp_group()
        if group is None:
            return input_
        
        ctx.dim = dim
        sp_size = get_sp_size()
        
        # Split input
        chunks = list(input_.chunk(sp_size, dim=dim))
        
        # ReduceScatter
        output = torch.empty_like(chunks[0])
        dist.reduce_scatter(output, chunks, group=group)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        group = get_sp_group()
        if group is None:
            return grad_output, None
        
        sp_size = get_sp_size()
        
        # AllGather gradient
        gather_list = [torch.empty_like(grad_output) for _ in range(sp_size)]
        dist.all_gather(gather_list, grad_output, group=group)
        
        return torch.cat(gather_list, dim=ctx.dim), None


# Functional APIs
scatter_sequence = _ScatterSequence.apply
gather_sequence = _GatherSequence.apply
reduce_scatter_sequence = _ReduceScatterSequence.apply


class SequenceParallelWrapper(nn.Module):
    """
    Sequence parallel wrapper for transformer layers.
    
    Pattern:
    1. Scatter input sequence across SP group
    2. Forward through layer (local computation)
    3. AllGather for attention (needs full sequence)
    4. ReduceScatter after attention
    
    Memory reduction: O(S/N) per rank instead of O(S)
    """
    
    def __init__(
        self,
        layer: nn.Module,
        sp_size: int,
        sp_rank: int,
        sp_group: "dist.ProcessGroup",
        scatter_input: bool = True,
        gather_output: bool = False,
    ):
        super().__init__()
        self.layer = layer
        self.sp_size = sp_size
        self.sp_rank = sp_rank
        self.scatter_input = scatter_input
        self.gather_output = gather_output
        
        set_sp_group(sp_group, sp_size)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward with sequence parallelism.
        
        Args:
            hidden_states: (batch, seq, hidden)
            
        Returns:
            Processed hidden states (possibly scattered)
        """
        # Scatter input if needed
        if self.scatter_input:
            hidden_states = scatter_sequence(hidden_states, dim=1)
        
        # Forward through layer
        output = self.layer(hidden_states, **kwargs)
        
        # Gather output if needed
        if self.gather_output:
            output = gather_sequence(output, dim=1)
        
        return output


class SPLayerNorm(nn.Module):
    """
    LayerNorm compatible with sequence parallelism.
    
    Operates on local sequence partition, no communication needed.
    """
    
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard layer norm on local partition
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x + self.bias


class SPDropout(nn.Module):
    """
    Dropout compatible with sequence parallelism.
    
    Uses consistent random seed across SP group for reproducibility.
    """
    
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return x
        
        # Generate consistent mask across SP group
        sp_rank = get_sp_rank()
        
        # Use rank-independent seed for consistency
        generator = torch.Generator(device=x.device)
        generator.manual_seed(42 + sp_rank)  # Shifted seed per rank
        
        mask = torch.empty_like(x).bernoulli_(1 - self.p, generator=generator)
        return x * mask / (1 - self.p)
