"""
Base Trainer
============
SOTA distributed trainer with mixed-precision, gradient checkpointing,
and kernel-level optimizations.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable, List, Union, Iterator
from pathlib import Path
from enum import Enum
import math
import time
import json
import os

try:
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    from torch.cuda.amp import GradScaler, autocast
    from torch.utils.data import DataLoader, DistributedSampler
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .config import TrainingConfig, PrecisionMode, ParallelismStrategy


@dataclass
class TrainerState:
    """Training state for checkpointing."""
    global_step: int = 0
    epoch: int = 0
    best_loss: float = float("inf")
    total_tokens: int = 0
    training_loss: float = 0.0
    learning_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "total_tokens": self.total_tokens,
            "training_loss": self.training_loss,
            "learning_rate": self.learning_rate,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainerState":
        return cls(**data)


class Trainer:
    """
    Base distributed trainer.
    
    Features:
    - Mixed-precision training (BF16/FP16/FP8)
    - Gradient checkpointing
    - FSDP/DDP/DeepSpeed ZeRO
    - Gradient accumulation and clipping
    - LR scheduling with warmup
    - Checkpoint save/resume
    """
    
    def __init__(
        self,
        model: "nn.Module",
        config: TrainingConfig,
        train_dataloader: Optional[DataLoader] = None,
        eval_dataloader: Optional[DataLoader] = None,
        optimizer: Optional["torch.optim.Optimizer"] = None,
        scheduler: Optional[Any] = None,
        compute_loss_fn: Optional[Callable] = None,
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch required")
        
        self.config = config
        self.state = TrainerState()
        self._model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self._optimizer = optimizer
        self._scheduler = scheduler
        self.compute_loss_fn = compute_loss_fn or self._default_loss_fn
        
        # Setup
        self._setup_distributed()
        self._setup_precision()
        self._setup_model()
        self._setup_optimizer()
        self._setup_scheduler()
        
        # Metrics
        self._step_times: List[float] = []
        self._losses: List[float] = []
    
    # =========================================================================
    # Setup Methods
    # =========================================================================
    
    def _setup_distributed(self):
        """Initialize distributed training."""
        self.is_distributed = self.config.distributed.world_size > 1
        self.rank = self.config.distributed.rank
        self.local_rank = self.config.distributed.local_rank
        self.world_size = self.config.distributed.world_size
        
        if self.is_distributed and not dist.is_initialized():
            dist.init_process_group(
                backend=self.config.distributed.backend,
                init_method=self.config.distributed.init_method,
                world_size=self.world_size,
                rank=self.rank,
            )
        
        self.is_main_process = self.rank == 0
        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
    
    def _setup_precision(self):
        """Configure mixed-precision training."""
        self.precision = self.config.precision
        self.use_amp = self.precision in (PrecisionMode.FP16, PrecisionMode.BF16)
        
        if self.use_amp:
            self.amp_dtype = torch.bfloat16 if self.precision == PrecisionMode.BF16 else torch.float16
            self.scaler = GradScaler(enabled=self.precision == PrecisionMode.FP16)
        else:
            self.amp_dtype = torch.float32
            self.scaler = None
    
    def _setup_model(self):
        """Setup model with distributed wrapper and gradient checkpointing."""
        self._model = self._model.to(self.device)
        
        # Gradient checkpointing
        if self.config.gradient_checkpointing:
            if hasattr(self._model, "gradient_checkpointing_enable"):
                self._model.gradient_checkpointing_enable()
        
        # Distributed wrapper
        strategy = self.config.distributed.strategy
        
        if strategy == ParallelismStrategy.DDP and self.is_distributed:
            from torch.nn.parallel import DistributedDataParallel as DDP
            self._model = DDP(
                self._model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
            )
        elif strategy == ParallelismStrategy.FSDP and self.is_distributed:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
            
            mp_policy = MixedPrecision(
                param_dtype=self.amp_dtype,
                reduce_dtype=self.amp_dtype,
                buffer_dtype=self.amp_dtype,
            )
            
            self._model = FSDP(
                self._model,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=mp_policy,
                device_id=self.local_rank,
            )
    
    @property
    def model(self) -> "nn.Module":
        """Get underlying model (unwrap DDP/FSDP if needed)."""
        if hasattr(self._model, "module"):
            return self._model.module
        return self._model
    
    def _setup_optimizer(self):
        """Create optimizer with weight decay groups."""
        if self._optimizer is not None:
            return
        
        # Separate parameters with/without weight decay
        decay_params = []
        no_decay_params = []
        
        no_decay_names = ["bias", "LayerNorm", "layer_norm", "rmsnorm"]
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in no_decay_names):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        
        # Create optimizer
        opt_cls = torch.optim.AdamW
        
        try:
            from apex.optimizers import FusedAdam
            if self.config.optimizer.value == "fused_adamw":
                opt_cls = FusedAdam
        except ImportError:
            pass
        
        self._optimizer = opt_cls(
            param_groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps,
        )
    
    def _setup_scheduler(self):
        """Create learning rate scheduler."""
        if self._scheduler is not None:
            return
        
        from torch.optim.lr_scheduler import LambdaLR
        
        warmup_steps = self.config.warmup_steps
        max_steps = self.config.max_steps
        min_lr_ratio = self.config.min_lr_ratio
        
        def cosine_warmup_schedule(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        
        self._scheduler = LambdaLR(self._optimizer, cosine_warmup_schedule)
    
    # =========================================================================
    # Training Loop
    # =========================================================================
    
    def train(self) -> TrainerState:
        """Main training loop."""
        if self.train_dataloader is None:
            raise ValueError("train_dataloader required")
        
        self._model.train()
        
        # Resume from checkpoint
        if self.config.resume_from_checkpoint:
            self.load_checkpoint(self.config.resume_from_checkpoint)
        
        # Training loop
        accumulation_steps = self.config.gradient_accumulation_steps
        
        epoch = self.state.epoch
        while self.state.global_step < self.config.max_steps:
            epoch_loss = 0.0
            num_batches = 0
            
            if self.is_distributed and hasattr(self.train_dataloader.sampler, "set_epoch"):
                self.train_dataloader.sampler.set_epoch(epoch)
            
            for step, batch in enumerate(self.train_dataloader):
                step_start = time.perf_counter()
                
                # Move batch to device
                batch = self._prepare_batch(batch)
                
                # Forward pass with AMP
                with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                    loss = self.compute_loss_fn(self._model, batch)
                    loss = loss / accumulation_steps
                
                # Backward pass
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                epoch_loss += loss.item() * accumulation_steps
                num_batches += 1
                
                # Gradient accumulation
                if (step + 1) % accumulation_steps == 0:
                    # Gradient clipping
                    if self.config.gradient_clipping > 0:
                        if self.scaler:
                            self.scaler.unscale_(self._optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self._model.parameters(),
                            self.config.gradient_clipping,
                        )
                    
                    # Optimizer step
                    if self.scaler:
                        self.scaler.step(self._optimizer)
                        self.scaler.update()
                    else:
                        self._optimizer.step()
                    
                    self._scheduler.step()
                    self._optimizer.zero_grad(set_to_none=True)
                    
                    self.state.global_step += 1
                    self.state.learning_rate = self._scheduler.get_last_lr()[0]
                    
                    step_time = time.perf_counter() - step_start
                    self._step_times.append(step_time)
                    self._losses.append(loss.item() * accumulation_steps)
                    
                    # Logging
                    if self.state.global_step % self.config.logging_steps == 0:
                        self._log_metrics()
                    
                    # Checkpointing
                    if self.state.global_step % self.config.save_steps == 0:
                        self.save_checkpoint()
                    
                    if self.state.global_step >= self.config.max_steps:
                        break
            
            self.state.epoch = epoch
            self.state.training_loss = epoch_loss / max(1, num_batches)
            epoch += 1
        
        # Final checkpoint
        self.save_checkpoint()
        
        return self.state
    
    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Move batch tensors to device."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
    
    def _default_loss_fn(
        self,
        model: "nn.Module",
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Default causal LM loss computation."""
        outputs = model(**batch)
        if hasattr(outputs, "loss") and outputs.loss is not None:
            return outputs.loss
        raise ValueError("Model must return loss or provide custom compute_loss_fn")
    
    def _log_metrics(self):
        """Log training metrics."""
        if not self.is_main_process:
            return
        
        avg_loss = sum(self._losses[-self.config.logging_steps:]) / self.config.logging_steps
        avg_time = sum(self._step_times[-self.config.logging_steps:]) / self.config.logging_steps
        tokens_per_sec = (
            self.config.per_device_batch_size * 
            self.config.max_seq_length * 
            self.world_size / avg_time
        )
        
        print(
            f"Step {self.state.global_step} | "
            f"Loss: {avg_loss:.4f} | "
            f"LR: {self.state.learning_rate:.2e} | "
            f"Tokens/s: {tokens_per_sec:.0f}"
        )
    
    # =========================================================================
    # Checkpointing
    # =========================================================================
    
    def save_checkpoint(self, path: Optional[str] = None):
        """Save training checkpoint."""
        if not self.is_main_process:
            return
        
        if path is None:
            path = Path(self.config.output_dir) / f"checkpoint-{self.state.global_step}"
        else:
            path = Path(path)
        
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = path / "model.pt"
        torch.save(self.model.state_dict(), model_path)
        
        # Save optimizer
        opt_path = path / "optimizer.pt"
        torch.save(self._optimizer.state_dict(), opt_path)
        
        # Save scheduler
        sched_path = path / "scheduler.pt"
        torch.save(self._scheduler.state_dict(), sched_path)
        
        # Save state
        state_path = path / "trainer_state.json"
        with open(state_path, "w") as f:
            json.dump(self.state.to_dict(), f, indent=2)
        
        # Save config
        config_path = path / "training_config.json"
        with open(config_path, "w") as f:
            json.dump(vars(self.config), f, indent=2, default=str)
        
        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        path = Path(path)
        
        # Load model
        model_path = path / "model.pt"
        if model_path.exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        # Load optimizer
        opt_path = path / "optimizer.pt"
        if opt_path.exists():
            self._optimizer.load_state_dict(torch.load(opt_path, map_location=self.device))
        
        # Load scheduler
        sched_path = path / "scheduler.pt"
        if sched_path.exists():
            self._scheduler.load_state_dict(torch.load(sched_path))
        
        # Load state
        state_path = path / "trainer_state.json"
        if state_path.exists():
            with open(state_path) as f:
                self.state = TrainerState.from_dict(json.load(f))
        
        print(f"Checkpoint loaded: {path}")
    
    # =========================================================================
    # Evaluation
    # =========================================================================
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation."""
        if self.eval_dataloader is None:
            return {}
        
        self._model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.eval_dataloader:
            batch = self._prepare_batch(batch)
            
            with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                loss = self.compute_loss_fn(self._model, batch)
            
            total_loss += loss.item()
            num_batches += 1
        
        self._model.train()
        
        avg_loss = total_loss / max(1, num_batches)
        perplexity = math.exp(min(avg_loss, 20))
        
        return {"eval_loss": avg_loss, "perplexity": perplexity}
