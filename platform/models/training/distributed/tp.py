"""
Tensor Parallelism (TP)
=======================
Column and row parallel linear layers for tensor parallelism.
Shards weights across TP group for large model training.
"""

from typing import Optional, Tuple, List
import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.distributed as dist
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def get_tp_group() -> Optional["dist.ProcessGroup"]:
    """Get tensor parallel process group."""
    if not dist.is_initialized():
        return None
    # Assume TP group is stored as module attribute
    return getattr(get_tp_group, "_group", None)


def set_tp_group(group: "dist.ProcessGroup", tp_size: int):
    """Set tensor parallel process group."""
    get_tp_group._group = group
    get_tp_group._size = tp_size
    get_tp_group._rank = dist.get_rank(group)


def get_tp_rank() -> int:
    """Get rank within TP group."""
    return getattr(get_tp_group, "_rank", 0)


def get_tp_size() -> int:
    """Get TP world size."""
    return getattr(get_tp_group, "_size", 1)


class _CopyToTensorParallelRegion(torch.autograd.Function):
    """Copy input to TP region (identity forward, all-reduce backward)."""
    
    @staticmethod
    def forward(ctx, input_: torch.Tensor) -> torch.Tensor:
        return input_
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        group = get_tp_group()
        if group is not None:
            dist.all_reduce(grad_output, group=group)
        return grad_output


class _ReduceFromTensorParallelRegion(torch.autograd.Function):
    """All-reduce output from TP region (all-reduce forward, identity backward)."""
    
    @staticmethod
    def forward(ctx, input_: torch.Tensor) -> torch.Tensor:
        group = get_tp_group()
        if group is not None:
            dist.all_reduce(input_, group=group)
        return input_
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


class _GatherFromTensorParallelRegion(torch.autograd.Function):
    """Gather output along last dim from TP region."""
    
    @staticmethod
    def forward(ctx, input_: torch.Tensor) -> torch.Tensor:
        group = get_tp_group()
        if group is None:
            return input_
        
        world_size = get_tp_size()
        rank = get_tp_rank()
        
        tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
        tensor_list[rank] = input_
        dist.all_gather(tensor_list, input_, group=group)
        
        return torch.cat(tensor_list, dim=-1)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        group = get_tp_group()
        if group is None:
            return grad_output
        
        world_size = get_tp_size()
        rank = get_tp_rank()
        
        dim_size = grad_output.size(-1)
        chunk_size = dim_size // world_size
        
        return grad_output[..., rank * chunk_size:(rank + 1) * chunk_size].contiguous()


class _ScatterToTensorParallelRegion(torch.autograd.Function):
    """Scatter input along last dim to TP region."""
    
    @staticmethod
    def forward(ctx, input_: torch.Tensor) -> torch.Tensor:
        group = get_tp_group()
        if group is None:
            return input_
        
        world_size = get_tp_size()
        rank = get_tp_rank()
        
        dim_size = input_.size(-1)
        chunk_size = dim_size // world_size
        
        return input_[..., rank * chunk_size:(rank + 1) * chunk_size].contiguous()
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        group = get_tp_group()
        if group is None:
            return grad_output
        
        world_size = get_tp_size()
        rank = get_tp_rank()
        
        tensor_list = [torch.empty_like(grad_output) for _ in range(world_size)]
        tensor_list[rank] = grad_output
        dist.all_gather(tensor_list, grad_output, group=group)
        
        return torch.cat(tensor_list, dim=-1)


# Functional APIs
copy_to_tp_region = _CopyToTensorParallelRegion.apply
reduce_from_tp_region = _ReduceFromTensorParallelRegion.apply
gather_from_tp_region = _GatherFromTensorParallelRegion.apply
scatter_to_tp_region = _ScatterToTensorParallelRegion.apply


class ColumnParallelLinear(nn.Module):
    """
    Column parallel linear layer.
    
    Shards output dimension across TP group.
    Y = XA, where A is sharded column-wise: A = [A1, A2, ..., An]
    Each rank computes Y_i = X * A_i
    
    Use with gather_output=True for standalone layer,
    or gather_output=False when followed by RowParallelLinear.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = True,
        init_method: str = "xavier",
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        
        tp_size = get_tp_size()
        assert out_features % tp_size == 0, f"out_features ({out_features}) must be divisible by tp_size ({tp_size})"
        
        self.out_features_per_partition = out_features // tp_size
        
        self.weight = nn.Parameter(torch.empty(self.out_features_per_partition, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_partition))
        else:
            self.register_parameter("bias", None)
        
        self._init_weights(init_method)
    
    def _init_weights(self, method: str):
        if method == "xavier":
            nn.init.xavier_uniform_(self.weight)
        elif method == "kaiming":
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        else:
            nn.init.normal_(self.weight, std=0.02)
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Copy input to TP region (all-reduce on backward)
        x = copy_to_tp_region(x)
        
        # Local matmul
        output = F.linear(x, self.weight, self.bias)
        
        # Gather outputs if needed
        if self.gather_output:
            output = gather_from_tp_region(output)
        
        return output


class RowParallelLinear(nn.Module):
    """
    Row parallel linear layer.
    
    Shards input dimension across TP group.
    Y = XA, where X is sharded: X = [X1, X2, ..., Xn]
    All-reduce to combine: Y = sum(X_i * A_i)
    
    Use with input_is_parallel=True when preceded by ColumnParallelLinear,
    or input_is_parallel=False for standalone layer.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        init_method: str = "xavier",
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        
        tp_size = get_tp_size()
        assert in_features % tp_size == 0, f"in_features ({in_features}) must be divisible by tp_size ({tp_size})"
        
        self.in_features_per_partition = in_features // tp_size
        
        self.weight = nn.Parameter(torch.empty(out_features, self.in_features_per_partition))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        
        self._init_weights(init_method)
    
    def _init_weights(self, method: str):
        if method == "xavier":
            nn.init.xavier_uniform_(self.weight)
        elif method == "kaiming":
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        else:
            nn.init.normal_(self.weight, std=0.02)
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scatter input if not already parallel
        if not self.input_is_parallel:
            x = scatter_to_tp_region(x)
        
        # Local matmul (no bias yet)
        output = F.linear(x, self.weight, None)
        
        # All-reduce to combine partial results
        output = reduce_from_tp_region(output)
        
        # Add bias after reduce
        if self.bias is not None:
            output = output + self.bias
        
        return output


class TensorParallelWrapper:
    """
    Utility for applying tensor parallelism to model.
    
    Replaces Linear layers with ColumnParallel/RowParallel variants.
    """
    
    def __init__(
        self,
        tp_size: int,
        tp_rank: int,
        tp_group: "dist.ProcessGroup",
    ):
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        set_tp_group(tp_group, tp_size)
    
    def parallelize_attention(
        self,
        module: nn.Module,
        names: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj"),
    ) -> nn.Module:
        """
        Parallelize attention projections.
        
        Q, K, V: Column parallel (shard heads)
        O: Row parallel (reduce outputs)
        """
        for name in names[:3]:  # Q, K, V
            if hasattr(module, name):
                linear = getattr(module, name)
                setattr(module, name, ColumnParallelLinear(
                    linear.in_features,
                    linear.out_features,
                    bias=linear.bias is not None,
                    gather_output=False,
                ))
        
        # O projection
        if hasattr(module, names[3]):
            linear = getattr(module, names[3])
            setattr(module, names[3], RowParallelLinear(
                linear.in_features,
                linear.out_features,
                bias=linear.bias is not None,
                input_is_parallel=True,
            ))
        
        return module
    
    def parallelize_mlp(
        self,
        module: nn.Module,
        names: Tuple[str, ...] = ("gate_proj", "up_proj", "down_proj"),
    ) -> nn.Module:
        """
        Parallelize MLP projections.
        
        Gate, Up: Column parallel
        Down: Row parallel
        """
        for name in names[:2]:  # Gate, Up
            if hasattr(module, name):
                linear = getattr(module, name)
                setattr(module, name, ColumnParallelLinear(
                    linear.in_features,
                    linear.out_features,
                    bias=linear.bias is not None,
                    gather_output=False,
                ))
        
        # Down projection
        if hasattr(module, names[2]):
            linear = getattr(module, names[2])
            setattr(module, names[2], RowParallelLinear(
                linear.in_features,
                linear.out_features,
                bias=linear.bias is not None,
                input_is_parallel=True,
            ))
        
        return module
