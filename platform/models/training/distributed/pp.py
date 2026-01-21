"""
Pipeline Parallelism (PP)
=========================
1F1B schedule for efficient pipeline parallelism.
Partitions model layers across pipeline stages.
"""

from typing import Optional, List, Tuple, Any, Dict
from dataclasses import dataclass
import threading
import queue

try:
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class PipelineConfig:
    """Pipeline parallelism configuration."""
    num_stages: int = 4
    num_microbatches: int = 8
    schedule: str = "1f1b"  # "1f1b", "gpipe", "interleaved"


class PipelineSchedule:
    """
    Pipeline schedule implementations.
    
    Supports:
    - GPipe: All forward then all backward
    - 1F1B: Interleaved forward-backward for memory efficiency
    - Interleaved: Multiple virtual stages per rank
    """
    
    def __init__(
        self,
        num_stages: int,
        num_microbatches: int,
        stage_id: int,
    ):
        self.num_stages = num_stages
        self.num_microbatches = num_microbatches
        self.stage_id = stage_id
        
        self.is_first_stage = stage_id == 0
        self.is_last_stage = stage_id == num_stages - 1
    
    def gpipe_schedule(self) -> List[Tuple[str, int]]:
        """
        GPipe schedule: all forwards then all backwards.
        
        Pros: Simple
        Cons: High memory (stores all activations)
        """
        schedule = []
        
        # All forwards
        for mb in range(self.num_microbatches):
            schedule.append(("forward", mb))
        
        # All backwards (reverse order)
        for mb in reversed(range(self.num_microbatches)):
            schedule.append(("backward", mb))
        
        return schedule
    
    def one_f_one_b_schedule(self) -> List[Tuple[str, int]]:
        """
        1F1B schedule: steady state of 1 forward, 1 backward.
        
        Pros: Lower peak memory
        Cons: Slightly more complex
        """
        schedule = []
        
        # Warmup: forward only
        warmup_steps = self.num_stages - self.stage_id - 1
        for mb in range(min(warmup_steps, self.num_microbatches)):
            schedule.append(("forward", mb))
        
        # Steady state: 1F1B
        num_microbatches_remaining = self.num_microbatches - warmup_steps
        for i in range(num_microbatches_remaining):
            schedule.append(("forward", warmup_steps + i))
            schedule.append(("backward", i))
        
        # Cooldown: backward only
        for mb in range(num_microbatches_remaining, self.num_microbatches):
            schedule.append(("backward", mb))
        
        return schedule
    
    def get_schedule(self, schedule_type: str = "1f1b") -> List[Tuple[str, int]]:
        """Get schedule by type."""
        if schedule_type == "gpipe":
            return self.gpipe_schedule()
        elif schedule_type == "1f1b":
            return self.one_f_one_b_schedule()
        else:
            raise ValueError(f"Unknown schedule: {schedule_type}")


class PipelineStage(nn.Module):
    """
    Single pipeline stage wrapping a subset of model layers.
    """
    
    def __init__(
        self,
        layers: nn.ModuleList,
        stage_id: int,
        num_stages: int,
    ):
        super().__init__()
        self.layers = layers
        self.stage_id = stage_id
        self.num_stages = num_stages
        
        self.is_first_stage = stage_id == 0
        self.is_last_stage = stage_id == num_stages - 1
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, **kwargs)
        return x


def send_tensor(
    tensor: torch.Tensor,
    dst_rank: int,
    group: Optional["dist.ProcessGroup"] = None,
):
    """Send tensor to destination rank."""
    dist.send(tensor, dst=dst_rank, group=group)


def recv_tensor(
    tensor: torch.Tensor,
    src_rank: int,
    group: Optional["dist.ProcessGroup"] = None,
):
    """Receive tensor from source rank."""
    dist.recv(tensor, src=src_rank, group=group)


class PipelineParallelWrapper:
    """
    Pipeline parallel model wrapper.
    
    Handles:
    - Model partitioning across stages
    - Communication between stages
    - Schedule execution
    """
    
    def __init__(
        self,
        model: "nn.Module",
        num_stages: int,
        stage_id: int,
        num_microbatches: int = 8,
        schedule: str = "1f1b",
        group: Optional["dist.ProcessGroup"] = None,
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch required")
        
        self.num_stages = num_stages
        self.stage_id = stage_id
        self.num_microbatches = num_microbatches
        self.group = group
        
        # Partition model
        self.stage = self._partition_model(model)
        
        # Create schedule
        self.scheduler = PipelineSchedule(
            num_stages=num_stages,
            num_microbatches=num_microbatches,
            stage_id=stage_id,
        )
        self.schedule = self.scheduler.get_schedule(schedule)
        
        # Buffers for activations
        self.input_tensors: List[Optional[torch.Tensor]] = [None] * num_microbatches
        self.output_tensors: List[Optional[torch.Tensor]] = [None] * num_microbatches
    
    def _partition_model(self, model: "nn.Module") -> PipelineStage:
        """
        Partition model layers across stages.
        
        Assumes model has a 'layers' attribute (nn.ModuleList).
        """
        if hasattr(model, "layers"):
            all_layers = list(model.layers)
        elif hasattr(model, "blocks"):
            all_layers = list(model.blocks)
        else:
            all_layers = list(model.children())
        
        num_layers = len(all_layers)
        layers_per_stage = num_layers // self.num_stages
        remainder = num_layers % self.num_stages
        
        # Calculate layer range for this stage
        start_layer = self.stage_id * layers_per_stage + min(self.stage_id, remainder)
        end_layer = start_layer + layers_per_stage + (1 if self.stage_id < remainder else 0)
        
        stage_layers = nn.ModuleList(all_layers[start_layer:end_layer])
        
        return PipelineStage(
            layers=stage_layers,
            stage_id=self.stage_id,
            num_stages=self.num_stages,
        )
    
    def forward_step(
        self,
        microbatch_id: int,
        input_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Execute forward pass for one microbatch."""
        # Receive from previous stage
        if not self.stage.is_first_stage and input_tensor is None:
            input_tensor = self.input_tensors[microbatch_id]
            if input_tensor is None:
                raise ValueError(f"Input tensor not set for microbatch {microbatch_id}")
        
        # Forward through stage
        output_tensor = self.stage(input_tensor)
        self.output_tensors[microbatch_id] = output_tensor
        
        return output_tensor
    
    def backward_step(
        self,
        microbatch_id: int,
        output_grad: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Execute backward pass for one microbatch."""
        output_tensor = self.output_tensors[microbatch_id]
        
        if output_tensor is None:
            raise ValueError(f"Output tensor not set for microbatch {microbatch_id}")
        
        # Compute gradients
        if output_grad is not None or self.stage.is_last_stage:
            output_tensor.backward(output_grad)
        
        # Get input gradient for previous stage
        input_tensor = self.input_tensors[microbatch_id]
        if input_tensor is not None and input_tensor.requires_grad:
            return input_tensor.grad
        
        return None
    
    def run_schedule(
        self,
        input_tensors: List[torch.Tensor],
        loss_fn: Optional[Any] = None,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """
        Run the pipeline schedule.
        
        Args:
            input_tensors: List of microbatch inputs (first stage only)
            loss_fn: Loss function (last stage only)
            
        Returns:
            Total loss, metrics dict
        """
        if self.stage.is_first_stage:
            for i, tensor in enumerate(input_tensors):
                self.input_tensors[i] = tensor
        
        total_loss = None
        
        for action, microbatch_id in self.schedule:
            if action == "forward":
                output = self.forward_step(microbatch_id)
                
                # Send to next stage
                if not self.stage.is_last_stage:
                    next_rank = self.stage_id + 1
                    send_tensor(output, next_rank, self.group)
                elif loss_fn is not None:
                    # Compute loss at last stage
                    loss = loss_fn(output)
                    if total_loss is None:
                        total_loss = loss
                    else:
                        total_loss = total_loss + loss
            
            elif action == "backward":
                # Receive gradient from next stage
                output_grad = None
                if not self.stage.is_last_stage:
                    output_grad = torch.zeros_like(self.output_tensors[microbatch_id])
                    next_rank = self.stage_id + 1
                    recv_tensor(output_grad, next_rank, self.group)
                
                input_grad = self.backward_step(microbatch_id, output_grad)
                
                # Send gradient to previous stage
                if not self.stage.is_first_stage and input_grad is not None:
                    prev_rank = self.stage_id - 1
                    send_tensor(input_grad, prev_rank, self.group)
        
        return total_loss, {"num_microbatches": self.num_microbatches}
