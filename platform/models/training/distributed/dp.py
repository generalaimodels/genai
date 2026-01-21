"""
Data Parallelism (DP)
=====================
DDP and FSDP wrappers for distributed data parallelism.
"""

from typing import Optional, List, Any
from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        ShardingStrategy,
        MixedPrecision,
        BackwardPrefetch,
        CPUOffload,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    HAS_FSDP = True
except ImportError:
    HAS_FSDP = False


@dataclass
class DDPConfig:
    """DDP configuration."""
    find_unused_parameters: bool = False
    broadcast_buffers: bool = True
    bucket_cap_mb: int = 25
    gradient_as_bucket_view: bool = True
    static_graph: bool = False


@dataclass
class FSDPConfig:
    """FSDP configuration."""
    sharding_strategy: str = "FULL_SHARD"  # FULL_SHARD, SHARD_GRAD_OP, NO_SHARD
    cpu_offload: bool = False
    backward_prefetch: str = "BACKWARD_PRE"  # BACKWARD_PRE, BACKWARD_POST
    mixed_precision: bool = True
    use_orig_params: bool = True
    auto_wrap_policy: Optional[Any] = None


def setup_ddp(
    model: "nn.Module",
    device_id: int,
    config: Optional[DDPConfig] = None,
) -> "nn.Module":
    """
    Wrap model with DDP.
    
    Args:
        model: Model to wrap
        device_id: Local GPU device ID
        config: DDP configuration
        
    Returns:
        DDP-wrapped model
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required")
    
    if config is None:
        config = DDPConfig()
    
    model = model.to(device_id)
    
    return DDP(
        model,
        device_ids=[device_id],
        output_device=device_id,
        find_unused_parameters=config.find_unused_parameters,
        broadcast_buffers=config.broadcast_buffers,
        bucket_cap_mb=config.bucket_cap_mb,
        gradient_as_bucket_view=config.gradient_as_bucket_view,
        static_graph=config.static_graph,
    )


def setup_fsdp(
    model: "nn.Module",
    device_id: int,
    config: Optional[FSDPConfig] = None,
    mixed_precision_dtype: Optional["torch.dtype"] = None,
    wrap_classes: Optional[List[type]] = None,
) -> "nn.Module":
    """
    Wrap model with FSDP.
    
    Args:
        model: Model to wrap
        device_id: Local GPU device ID
        config: FSDP configuration
        mixed_precision_dtype: Precision for mixed-precision training
        wrap_classes: Module classes to wrap individually
        
    Returns:
        FSDP-wrapped model
    """
    if not HAS_FSDP:
        raise ImportError("FSDP not available")
    
    if config is None:
        config = FSDPConfig()
    
    # Sharding strategy
    strategy_map = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
    }
    sharding_strategy = strategy_map.get(config.sharding_strategy, ShardingStrategy.FULL_SHARD)
    
    # Mixed precision
    mp_policy = None
    if config.mixed_precision and mixed_precision_dtype is not None:
        mp_policy = MixedPrecision(
            param_dtype=mixed_precision_dtype,
            reduce_dtype=mixed_precision_dtype,
            buffer_dtype=mixed_precision_dtype,
        )
    
    # CPU offload
    cpu_offload = CPUOffload(offload_params=True) if config.cpu_offload else None
    
    # Backward prefetch
    prefetch_map = {
        "BACKWARD_PRE": BackwardPrefetch.BACKWARD_PRE,
        "BACKWARD_POST": BackwardPrefetch.BACKWARD_POST,
    }
    backward_prefetch = prefetch_map.get(config.backward_prefetch, BackwardPrefetch.BACKWARD_PRE)
    
    # Auto wrap policy
    auto_wrap_policy = config.auto_wrap_policy
    if auto_wrap_policy is None and wrap_classes:
        auto_wrap_policy = transformer_auto_wrap_policy(wrap_classes)
    
    return FSDP(
        model,
        sharding_strategy=sharding_strategy,
        cpu_offload=cpu_offload,
        mixed_precision=mp_policy,
        backward_prefetch=backward_prefetch,
        device_id=device_id,
        auto_wrap_policy=auto_wrap_policy,
        use_orig_params=config.use_orig_params,
    )


class DataParallelWrapper:
    """
    Unified data parallelism wrapper.
    
    Supports DDP and FSDP with automatic selection based on config.
    """
    
    def __init__(
        self,
        model: "nn.Module",
        device_id: int,
        strategy: str = "ddp",  # "ddp" or "fsdp"
        ddp_config: Optional[DDPConfig] = None,
        fsdp_config: Optional[FSDPConfig] = None,
        mixed_precision_dtype: Optional["torch.dtype"] = None,
    ):
        self.original_model = model
        self.device_id = device_id
        self.strategy = strategy
        
        if strategy == "fsdp":
            self.model = setup_fsdp(
                model,
                device_id,
                fsdp_config,
                mixed_precision_dtype,
            )
        else:
            self.model = setup_ddp(model, device_id, ddp_config)
    
    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def __getattr__(self, name: str):
        if name in ("model", "original_model", "device_id", "strategy"):
            return object.__getattribute__(self, name)
        return getattr(self.model, name)
    
    @property
    def unwrapped_model(self) -> "nn.Module":
        """Get underlying model without wrapper."""
        if hasattr(self.model, "module"):
            return self.model.module
        return self.model
