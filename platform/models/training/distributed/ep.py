"""
Expert Parallelism (EP)
=======================
Distributed MoE expert placement with AllToAll dispatch.
"""

from typing import Optional, Tuple, List

try:
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def get_ep_group() -> Optional["dist.ProcessGroup"]:
    """Get expert parallel process group."""
    if not dist.is_initialized():
        return None
    return getattr(get_ep_group, "_group", None)


def set_ep_group(group: "dist.ProcessGroup", ep_size: int):
    """Set expert parallel process group."""
    get_ep_group._group = group
    get_ep_group._size = ep_size
    get_ep_group._rank = dist.get_rank(group)


def get_ep_rank() -> int:
    """Get rank within EP group."""
    return getattr(get_ep_group, "_rank", 0)


def get_ep_size() -> int:
    """Get EP world size."""
    return getattr(get_ep_group, "_size", 1)


class AllToAllDispatch(torch.autograd.Function):
    """
    AllToAll communication for expert dispatch.
    
    Forward: Dispatch tokens to experts across ranks
    Backward: Gather gradients back to original ranks
    """
    
    @staticmethod
    def forward(
        ctx,
        input_: torch.Tensor,
        expert_indices: torch.Tensor,
        ep_size: int,
    ) -> torch.Tensor:
        """
        Dispatch tokens to experts.
        
        Args:
            input_: (batch, seq, hidden) tokens
            expert_indices: (batch * seq,) target expert for each token
            ep_size: Expert parallel size
            
        Returns:
            Tokens dispatched to local experts
        """
        ctx.ep_size = ep_size
        ctx.expert_indices = expert_indices
        ctx.input_shape = input_.shape
        
        group = get_ep_group()
        if group is None or ep_size == 1:
            return input_
        
        batch_seq = input_.size(0) * input_.size(1)
        hidden = input_.size(2)
        
        # Flatten input
        flat_input = input_.view(batch_seq, hidden)
        
        # Count tokens per expert per rank
        local_expert_count = input_.new_zeros(ep_size, dtype=torch.long)
        for i in range(ep_size):
            local_expert_count[i] = (expert_indices == i).sum()
        
        # AllToAll token counts
        global_expert_count = torch.zeros_like(local_expert_count)
        dist.all_to_all_single(global_expert_count, local_expert_count, group=group)
        
        ctx.local_expert_count = local_expert_count
        ctx.global_expert_count = global_expert_count
        
        # Sort tokens by target expert
        sorted_indices = expert_indices.argsort()
        sorted_input = flat_input[sorted_indices]
        
        ctx.sorted_indices = sorted_indices
        
        # Prepare AllToAll buffers
        send_counts = local_expert_count.tolist()
        recv_counts = global_expert_count.tolist()
        
        # AllToAll dispatch
        output_size = sum(recv_counts)
        output = input_.new_empty(output_size, hidden)
        
        # Split tensors for all_to_all
        send_splits = [sorted_input[sum(send_counts[:i]):sum(send_counts[:i+1])] 
                       for i in range(ep_size)]
        recv_splits = [input_.new_empty(recv_counts[i], hidden) for i in range(ep_size)]
        
        dist.all_to_all(recv_splits, send_splits, group=group)
        
        output = torch.cat(recv_splits, dim=0)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        """Gather gradients back via AllToAll."""
        ep_size = ctx.ep_size
        group = get_ep_group()
        
        if group is None or ep_size == 1:
            return grad_output, None, None
        
        # Reverse AllToAll
        send_counts = ctx.global_expert_count.tolist()
        recv_counts = ctx.local_expert_count.tolist()
        
        hidden = grad_output.size(-1)
        
        send_splits = []
        offset = 0
        for count in send_counts:
            send_splits.append(grad_output[offset:offset + count])
            offset += count
        
        recv_splits = [grad_output.new_empty(count, hidden) for count in recv_counts]
        
        dist.all_to_all(recv_splits, send_splits, group=group)
        
        grad_input = torch.cat(recv_splits, dim=0)
        
        # Unsort gradients
        unsorted_grad = grad_input.new_empty_like(grad_input)
        unsorted_grad[ctx.sorted_indices] = grad_input
        
        # Reshape back
        grad_input = unsorted_grad.view(ctx.input_shape)
        
        return grad_input, None, None


class AllToAllGather(torch.autograd.Function):
    """
    AllToAll communication for expert gather.
    
    Forward: Gather processed tokens back to original ranks
    Backward: Dispatch gradients to expert ranks
    """
    
    @staticmethod
    def forward(
        ctx,
        input_: torch.Tensor,
        local_expert_count: torch.Tensor,
        global_expert_count: torch.Tensor,
        sorted_indices: torch.Tensor,
        original_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """Gather tokens back to original ranks."""
        ctx.local_expert_count = local_expert_count
        ctx.global_expert_count = global_expert_count
        ctx.sorted_indices = sorted_indices
        
        group = get_ep_group()
        ep_size = get_ep_size()
        
        if group is None or ep_size == 1:
            return input_.view(original_shape)
        
        # Reverse of dispatch
        send_counts = global_expert_count.tolist()
        recv_counts = local_expert_count.tolist()
        
        hidden = input_.size(-1)
        
        send_splits = []
        offset = 0
        for count in send_counts:
            send_splits.append(input_[offset:offset + count])
            offset += count
        
        recv_splits = [input_.new_empty(count, hidden) for count in recv_counts]
        
        dist.all_to_all(recv_splits, send_splits, group=group)
        
        gathered = torch.cat(recv_splits, dim=0)
        
        # Unsort
        output = gathered.new_empty_like(gathered)
        output[sorted_indices] = gathered
        
        return output.view(original_shape)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Dispatch gradients back to expert ranks."""
        # Similar to dispatch forward, but for gradients
        return grad_output, None, None, None, None


def expert_parallel_dispatch(
    input_: torch.Tensor,
    expert_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Dispatch tokens to experts across EP group.
    
    Args:
        input_: Token embeddings (batch, seq, hidden)
        expert_indices: Target expert index for each token
        
    Returns:
        Tokens routed to local experts
    """
    ep_size = get_ep_size()
    return AllToAllDispatch.apply(input_, expert_indices, ep_size)


class ExpertParallelWrapper:
    """
    Expert parallel MoE wrapper.
    
    Distributes experts across EP group with AllToAll dispatch.
    """
    
    def __init__(
        self,
        experts: nn.ModuleList,
        ep_size: int,
        ep_rank: int,
        ep_group: "dist.ProcessGroup",
    ):
        self.experts = experts
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        
        set_ep_group(ep_group, ep_size)
        
        # Calculate local experts
        num_experts = len(experts)
        assert num_experts % ep_size == 0, f"Experts ({num_experts}) must divide EP size ({ep_size})"
        
        experts_per_rank = num_experts // ep_size
        self.local_expert_start = ep_rank * experts_per_rank
        self.local_expert_end = (ep_rank + 1) * experts_per_rank
        self.local_experts = experts[self.local_expert_start:self.local_expert_end]
    
    def forward(
        self,
        input_: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with expert parallel dispatch.
        
        Args:
            input_: (batch, seq, hidden)
            expert_indices: (batch * seq,) target expert per token
            
        Returns:
            Expert processed output
        """
        # Dispatch to experts
        dispatched = expert_parallel_dispatch(input_, expert_indices)
        
        # Process with local experts
        outputs = []
        local_indices = (expert_indices >= self.local_expert_start) & \
                        (expert_indices < self.local_expert_end)
        
        for i, expert in enumerate(self.local_experts):
            global_idx = self.local_expert_start + i
            mask = expert_indices == global_idx
            if mask.any():
                expert_input = dispatched[mask]
                expert_output = expert(expert_input)
                outputs.append((mask, expert_output))
        
        # Combine outputs
        combined = dispatched.clone()
        for mask, output in outputs:
            combined[mask] = output
        
        return combined
