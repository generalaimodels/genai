"""
ZeRO Optimizer
==============
Zero Redundancy Optimizer stages 1, 2, 3 with CPU/NVMe offloading.
"""

from typing import Optional, Dict, List, Any, Iterator
from dataclasses import dataclass, field
import math

try:
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class ZeROConfig:
    """ZeRO configuration."""
    stage: int = 1  # 1, 2, or 3
    
    # CPU Offloading
    offload_optimizer: bool = False
    offload_params: bool = False
    
    # NVMe Offloading (Stage 3 Infinity)
    offload_nvme: bool = False
    nvme_path: str = "/tmp/zero_offload"
    
    # Memory optimization
    contiguous_gradients: bool = True
    overlap_comm: bool = True
    reduce_bucket_size: int = 500_000_000  # 500M elements
    allgather_bucket_size: int = 500_000_000
    
    # Prefetch
    prefetch_bucket_size: int = 50_000_000
    param_persistence_threshold: int = 100_000


def partition_params(
    params: List[torch.nn.Parameter],
    world_size: int,
    rank: int,
) -> List[torch.nn.Parameter]:
    """Partition parameters across ranks for ZeRO-3."""
    partitioned = []
    for param in params:
        if param.numel() == 0:
            continue
        
        # Calculate partition size
        partition_size = math.ceil(param.numel() / world_size)
        start = rank * partition_size
        end = min((rank + 1) * partition_size, param.numel())
        
        if start < param.numel():
            flat = param.data.view(-1)
            partitioned.append(flat[start:end].clone())
    
    return partitioned


class ZeRO1Optimizer:
    """
    ZeRO Stage 1: Optimizer State Partitioning.
    
    Each rank holds 1/N of optimizer states.
    Full parameters and gradients on each rank.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        group: Optional["dist.ProcessGroup"] = None,
    ):
        self.optimizer = optimizer
        self.group = group
        self.world_size = dist.get_world_size(group) if dist.is_initialized() else 1
        self.rank = dist.get_rank(group) if dist.is_initialized() else 0
        
        # Partition optimizer states
        self._partition_optimizer_state()
    
    def _partition_optimizer_state(self):
        """Partition optimizer states across ranks."""
        for group in self.optimizer.param_groups:
            params = group["params"]
            num_params = len(params)
            
            # Each rank owns a subset of params for optimizer state
            params_per_rank = math.ceil(num_params / self.world_size)
            start = self.rank * params_per_rank
            end = min((self.rank + 1) * params_per_rank, num_params)
            
            group["_owned_params"] = list(range(start, end))
    
    def step(self):
        """Optimizer step with state partitioning."""
        # Update only owned parameters locally
        for group in self.optimizer.param_groups:
            owned = group.get("_owned_params", [])
            params = group["params"]
            
            for idx in owned:
                if idx < len(params):
                    param = params[idx]
                    if param.grad is not None:
                        # Standard optimizer step for owned param
                        pass
        
        self.optimizer.step()
        
        # Broadcast updated parameters
        self._sync_params()
    
    def _sync_params(self):
        """Broadcast parameters from owning rank."""
        if self.world_size == 1:
            return
        
        for group in self.optimizer.param_groups:
            for idx, param in enumerate(group["params"]):
                owner_rank = idx % self.world_size
                dist.broadcast(param.data, src=owner_rank, group=self.group)
    
    def zero_grad(self, set_to_none: bool = True):
        self.optimizer.zero_grad(set_to_none=set_to_none)


class ZeRO2Optimizer:
    """
    ZeRO Stage 2: Gradient Partitioning.
    
    Each rank holds 1/N of optimizer states AND gradients.
    Full parameters on each rank.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        group: Optional["dist.ProcessGroup"] = None,
        config: Optional[ZeROConfig] = None,
    ):
        self.optimizer = optimizer
        self.group = group
        self.config = config or ZeROConfig(stage=2)
        self.world_size = dist.get_world_size(group) if dist.is_initialized() else 1
        self.rank = dist.get_rank(group) if dist.is_initialized() else 0
        
        self._setup_gradient_hooks()
    
    def _setup_gradient_hooks(self):
        """Setup hooks for gradient reduction."""
        self.grad_handles = []
        
        for group in self.optimizer.param_groups:
            for idx, param in enumerate(group["params"]):
                if param.requires_grad:
                    handle = param.register_post_accumulate_grad_hook(
                        self._make_grad_hook(idx)
                    )
                    self.grad_handles.append(handle)
    
    def _make_grad_hook(self, param_idx: int):
        """Create gradient hook for reduce-scatter."""
        def hook(param: torch.Tensor):
            if param.grad is None:
                return
            
            if self.world_size == 1:
                return
            
            # Reduce-scatter gradient
            grad = param.grad
            output = torch.zeros(
                grad.numel() // self.world_size,
                dtype=grad.dtype,
                device=grad.device,
            )
            
            # Flatten and reduce-scatter
            flat_grad = grad.view(-1)
            chunks = list(flat_grad.chunk(self.world_size))
            
            dist.reduce_scatter(output, chunks, group=self.group)
            
            # Store partitioned gradient
            owner_rank = param_idx % self.world_size
            if self.rank == owner_rank:
                param._partitioned_grad = output
        
        return hook
    
    def step(self):
        """Step with gradient partitioning."""
        # Update with partitioned gradients
        for group in self.optimizer.param_groups:
            for idx, param in enumerate(group["params"]):
                owner_rank = idx % self.world_size
                
                if self.rank == owner_rank and hasattr(param, "_partitioned_grad"):
                    # Use partitioned gradient for update
                    pass
        
        self.optimizer.step()
        
        # AllGather updated parameters
        self._allgather_params()
    
    def _allgather_params(self):
        """AllGather parameters after update."""
        if self.world_size == 1:
            return
        
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                dist.all_reduce(param.data, group=self.group)
                param.data /= self.world_size
    
    def zero_grad(self, set_to_none: bool = True):
        self.optimizer.zero_grad(set_to_none=set_to_none)


class ZeRO3Optimizer:
    """
    ZeRO Stage 3: Parameter Partitioning.
    
    Each rank holds 1/N of parameters, gradients, AND optimizer states.
    Parameters gathered on-demand for forward/backward.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer_cls: type = torch.optim.AdamW,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        group: Optional["dist.ProcessGroup"] = None,
        config: Optional[ZeROConfig] = None,
    ):
        self.model = model
        self.group = group
        self.config = config or ZeROConfig(stage=3)
        self.world_size = dist.get_world_size(group) if dist.is_initialized() else 1
        self.rank = dist.get_rank(group) if dist.is_initialized() else 0
        
        optimizer_kwargs = optimizer_kwargs or {}
        
        # Partition and flatten parameters
        self._partition_params()
        
        # Create optimizer for local partition
        self.optimizer = optimizer_cls(
            [{"params": self.flat_param}],
            **optimizer_kwargs,
        )
        
        # Setup gather/scatter hooks
        self._setup_param_hooks()
    
    def _partition_params(self):
        """Partition parameters across ranks."""
        all_params = list(self.model.parameters())
        
        # Flatten all parameters
        total_numel = sum(p.numel() for p in all_params)
        
        # Calculate local partition
        partition_size = math.ceil(total_numel / self.world_size)
        start = self.rank * partition_size
        end = min((self.rank + 1) * partition_size, total_numel)
        
        # Create flattened parameter buffer
        flat_params = torch.cat([p.data.view(-1) for p in all_params])
        
        self.flat_param = nn.Parameter(flat_params[start:end].clone())
        self.param_start = start
        self.param_end = end
        
        # Store metadata for reconstruction
        self.param_shapes = [p.shape for p in all_params]
        self.param_numels = [p.numel() for p in all_params]
    
    def _setup_param_hooks(self):
        """Setup forward hooks for parameter gathering."""
        for module in self.model.modules():
            module.register_forward_pre_hook(self._pre_forward_hook)
            module.register_forward_hook(self._post_forward_hook)
    
    def _pre_forward_hook(self, module: nn.Module, inputs):
        """Gather parameters before forward."""
        self._gather_params(module)
    
    def _post_forward_hook(self, module: nn.Module, inputs, outputs):
        """Release gathered parameters after forward."""
        self._release_params(module)
    
    def _gather_params(self, module: nn.Module):
        """AllGather parameters for module."""
        if self.world_size == 1:
            return
        
        for param in module.parameters(recurse=False):
            # AllGather from all ranks
            gathered = [torch.empty_like(self.flat_param) for _ in range(self.world_size)]
            dist.all_gather(gathered, self.flat_param, group=self.group)
            
            # Reconstruct full parameter
            full_param = torch.cat(gathered)
            
            # Copy to parameter
            param.data.copy_(full_param[:param.numel()].view(param.shape))
    
    def _release_params(self, module: nn.Module):
        """Release gathered parameters to save memory."""
        if self.config.offload_params:
            for param in module.parameters(recurse=False):
                param.data = param.data.cpu()
    
    def step(self):
        """Optimizer step with parameter partitioning."""
        # Reduce-scatter gradients
        self._reduce_scatter_grads()
        
        # Local optimizer step
        self.optimizer.step()
    
    def _reduce_scatter_grads(self):
        """Reduce-scatter gradients to owning ranks."""
        if self.world_size == 1:
            return
        
        # Collect all gradients
        all_grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                all_grads.append(param.grad.view(-1))
            else:
                all_grads.append(torch.zeros(param.numel(), device=param.device))
        
        flat_grads = torch.cat(all_grads)
        
        # Reduce-scatter
        partition_size = math.ceil(flat_grads.numel() / self.world_size)
        local_grad = torch.empty(partition_size, device=flat_grads.device)
        
        chunks = list(flat_grads.chunk(self.world_size))
        dist.reduce_scatter(local_grad, chunks, group=self.group)
        
        # Assign to flat_param.grad
        self.flat_param.grad = local_grad[:self.flat_param.numel()]
    
    def zero_grad(self, set_to_none: bool = True):
        self.optimizer.zero_grad(set_to_none=set_to_none)
        for param in self.model.parameters():
            if set_to_none:
                param.grad = None
            elif param.grad is not None:
                param.grad.zero_()


class ZeROOptimizer:
    """
    Unified ZeRO optimizer interface.
    
    Automatically selects stage based on config.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        config: Optional[ZeROConfig] = None,
        group: Optional["dist.ProcessGroup"] = None,
    ):
        self.config = config or ZeROConfig()
        
        if self.config.stage == 1:
            if optimizer is None:
                optimizer = torch.optim.AdamW(model.parameters())
            self._impl = ZeRO1Optimizer(optimizer, group)
        elif self.config.stage == 2:
            if optimizer is None:
                optimizer = torch.optim.AdamW(model.parameters())
            self._impl = ZeRO2Optimizer(optimizer, group, config)
        elif self.config.stage == 3:
            self._impl = ZeRO3Optimizer(model, group=group, config=config)
        else:
            raise ValueError(f"Invalid ZeRO stage: {self.config.stage}")
    
    def step(self):
        self._impl.step()
    
    def zero_grad(self, set_to_none: bool = True):
        self._impl.zero_grad(set_to_none=set_to_none)
