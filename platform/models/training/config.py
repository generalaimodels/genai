"""
Training Configuration
======================
Dataclass configs for all training modes with distributed parallelism support.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Literal, Dict, Any
from enum import Enum


class PrecisionMode(str, Enum):
    """Training precision modes."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"


class ParallelismStrategy(str, Enum):
    """Distributed parallelism strategies."""
    NONE = "none"
    DDP = "ddp"              # Data Parallel
    FSDP = "fsdp"            # Fully Sharded Data Parallel
    TP = "tp"                # Tensor Parallel
    PP = "pp"                # Pipeline Parallel
    EP = "ep"                # Expert Parallel
    SP = "sp"                # Sequence Parallel
    ZERO_1 = "zero_1"
    ZERO_2 = "zero_2"
    ZERO_3 = "zero_3"


class OptimizerType(str, Enum):
    """Optimizer types."""
    ADAMW = "adamw"
    ADAM = "adam"
    SGD = "sgd"
    FUSED_ADAMW = "fused_adamw"
    LAMB = "lamb"


class SchedulerType(str, Enum):
    """LR scheduler types."""
    COSINE = "cosine"
    LINEAR = "linear"
    CONSTANT = "constant"
    COSINE_WARMUP = "cosine_warmup"
    INVERSE_SQRT = "inverse_sqrt"


@dataclass
class DistributedConfig:
    """Distributed training configuration."""
    strategy: ParallelismStrategy = ParallelismStrategy.DDP
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    
    # Tensor Parallelism
    tp_size: int = 1
    tp_group: Optional[List[int]] = None
    
    # Pipeline Parallelism
    pp_size: int = 1
    pp_chunks: int = 1
    pp_schedule: Literal["1f1b", "interleaved", "gpipe"] = "1f1b"
    
    # Expert Parallelism
    ep_size: int = 1
    
    # Sequence Parallelism
    sp_enabled: bool = False
    
    # ZeRO Optimizer
    zero_stage: int = 0
    zero_offload_cpu: bool = False
    zero_offload_nvme: bool = False
    
    # Communication
    backend: Literal["nccl", "gloo", "mpi"] = "nccl"
    init_method: str = "env://"
    timeout_seconds: int = 1800


@dataclass
class TrainingConfig:
    """Base training configuration."""
    # Model
    model_name: str = "rssmod"
    
    # Precision
    precision: PrecisionMode = PrecisionMode.BF16
    grad_scaler_enabled: bool = True
    
    # Optimization
    optimizer: OptimizerType = OptimizerType.FUSED_ADAMW
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    
    # Scheduler
    scheduler: SchedulerType = SchedulerType.COSINE_WARMUP
    warmup_steps: int = 2000
    warmup_ratio: float = 0.0
    min_lr_ratio: float = 0.1
    
    # Training loop
    max_steps: int = 100000
    max_epochs: Optional[int] = None
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    gradient_clipping: float = 1.0
    
    # Batch
    per_device_batch_size: int = 4
    max_seq_length: int = 2048
    
    # Distributed
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    
    # Checkpointing
    save_steps: int = 1000
    save_total_limit: int = 3
    output_dir: str = "./checkpoints"
    resume_from_checkpoint: Optional[str] = None
    
    # Logging
    logging_steps: int = 10
    log_level: str = "info"
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    
    # Seed
    seed: int = 42
    
    # Data
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    dataloader_prefetch_factor: int = 2
    
    def __post_init__(self):
        if self.warmup_ratio > 0:
            self.warmup_steps = int(self.max_steps * self.warmup_ratio)


@dataclass
class PretrainingConfig(TrainingConfig):
    """Pretraining-specific configuration."""
    # Dataset
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-103-v1"
    text_column: str = "text"
    streaming: bool = True
    
    # Sequence packing
    packing_enabled: bool = True
    
    # Loss
    use_fused_cross_entropy: bool = True


@dataclass
class FinetuningConfig(TrainingConfig):
    """Fine-tuning configuration."""
    # Dataset
    dataset_name: str = "tatsu-lab/alpaca"
    instruction_column: str = "instruction"
    input_column: str = "input"
    output_column: str = "output"
    
    # LoRA
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    # QLoRA
    use_qlora: bool = False
    bnb_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    
    # Chat template
    chat_template: Optional[str] = None


@dataclass
class PostTrainingConfig(TrainingConfig):
    """Post-training configuration (RLHF/DPO/etc)."""
    # Algorithm
    algorithm: Literal["sft", "dpo", "ppo", "grpo"] = "dpo"
    
    # DPO
    dpo_beta: float = 0.1
    dpo_label_smoothing: float = 0.0
    
    # PPO
    ppo_epochs: int = 4
    ppo_clip_range: float = 0.2
    ppo_value_loss_coef: float = 0.5
    ppo_entropy_coef: float = 0.01
    ppo_kl_penalty: float = 0.1
    
    # GRPO
    grpo_group_size: int = 8
    grpo_num_generations: int = 4
    
    # Reference model
    ref_model_path: Optional[str] = None
    ref_model_offload: bool = False
    
    # Reward model
    reward_model_path: Optional[str] = None
