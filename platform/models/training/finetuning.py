"""
Fine-tuning Trainer
===================
Instruction tuning with LoRA/QLoRA support.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path
import json

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

from .config import FinetuningConfig
from .trainer_base import Trainer


class InstructionDataset(Dataset):
    """
    Dataset for instruction fine-tuning.
    
    Supports Alpaca-style format:
    - instruction: Task description
    - input: Optional context
    - output: Expected response
    """
    
    def __init__(
        self,
        dataset: Any,
        tokenizer: Any,
        max_seq_length: int = 2048,
        instruction_column: str = "instruction",
        input_column: str = "input",
        output_column: str = "output",
        chat_template: Optional[str] = None,
    ):
        self.data = list(dataset)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.instruction_column = instruction_column
        self.input_column = input_column
        self.output_column = output_column
        self.chat_template = chat_template
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.data[idx]
        
        instruction = example.get(self.instruction_column, "")
        input_text = example.get(self.input_column, "")
        output = example.get(self.output_column, "")
        
        # Format prompt
        if self.chat_template:
            prompt = self._apply_chat_template(instruction, input_text)
        else:
            prompt = self._format_alpaca(instruction, input_text)
        
        full_text = prompt + output
        
        # Tokenize
        encoded = self.tokenizer.encode(
            full_text,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            truncation=True,
            return_tensors="pt",
        )
        
        if isinstance(encoded, dict):
            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).squeeze(0)
        else:
            input_ids = encoded.squeeze(0) if isinstance(encoded, torch.Tensor) else torch.tensor(encoded)
            attention_mask = torch.ones_like(input_ids)
        
        # Create labels (mask prompt tokens)
        labels = input_ids.clone()
        
        prompt_encoded = self.tokenizer.encode(
            prompt,
            add_special_tokens=True,
            return_tensors="pt",
        )
        if isinstance(prompt_encoded, dict):
            prompt_len = prompt_encoded["input_ids"].size(-1)
        else:
            prompt_len = len(prompt_encoded) if not isinstance(prompt_encoded, torch.Tensor) else prompt_encoded.size(-1)
        
        labels[:prompt_len] = -100  # Mask prompt tokens
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    
    def _format_alpaca(self, instruction: str, input_text: str) -> str:
        """Format as Alpaca-style prompt."""
        if input_text:
            return (
                f"Below is an instruction that describes a task, paired with an input that provides further context. "
                f"Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{input_text}\n\n"
                f"### Response:\n"
            )
        return (
            f"Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n"
        )
    
    def _apply_chat_template(self, instruction: str, input_text: str) -> str:
        """Apply chat template."""
        content = instruction
        if input_text:
            content += f"\n\n{input_text}"
        
        messages = [{"role": "user", "content": content}]
        
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        
        return self._format_alpaca(instruction, input_text)


class LoRALayer(nn.Module):
    """
    LoRA adapter layer.
    
    Low-Rank Adaptation for efficient fine-tuning.
    W_new = W + BA where B: (d, r), A: (r, k)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 64,
        alpha: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


def apply_lora(
    model: nn.Module,
    config: FinetuningConfig,
) -> nn.Module:
    """
    Apply LoRA adapters to model.
    
    Freezes base model and adds trainable LoRA layers.
    """
    # Freeze base model
    for param in model.parameters():
        param.requires_grad = False
    
    # Find target modules and add LoRA
    target_modules = config.lora_target_modules
    
    for name, module in model.named_modules():
        if not any(target in name for target in target_modules):
            continue
        
        if isinstance(module, nn.Linear):
            # Add LoRA adapter
            lora = LoRALayer(
                in_features=module.in_features,
                out_features=module.out_features,
                r=config.lora_r,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
            )
            
            # Store original forward and create new one
            original_forward = module.forward
            
            def make_lora_forward(orig, lora_layer):
                def lora_forward(x):
                    return orig(x) + lora_layer(x)
                return lora_forward
            
            module.forward = make_lora_forward(original_forward, lora)
            module.lora = lora  # Store reference
    
    return model


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Get only LoRA parameters for optimization."""
    params = []
    for module in model.modules():
        if hasattr(module, "lora"):
            params.extend(module.lora.parameters())
    return params


class FinetuningTrainer(Trainer):
    """
    Fine-tuning trainer with LoRA/QLoRA support.
    
    Features:
    - Instruction tuning format
    - LoRA parameter-efficient fine-tuning
    - QLoRA 4-bit quantization
    """
    
    def __init__(
        self,
        model: "nn.Module",
        config: FinetuningConfig,
        tokenizer: Any,
        train_dataset: Optional[Any] = None,
        eval_dataset: Optional[Any] = None,
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch required")
        
        self.tokenizer = tokenizer
        self.finetuning_config = config
        
        # Apply LoRA if enabled
        if config.use_lora:
            model = apply_lora(model, config)
        
        # Build datasets
        train_ds = None
        eval_ds = None
        
        if train_dataset is not None:
            train_ds = InstructionDataset(
                dataset=train_dataset,
                tokenizer=tokenizer,
                max_seq_length=config.max_seq_length,
                instruction_column=config.instruction_column,
                input_column=config.input_column,
                output_column=config.output_column,
                chat_template=config.chat_template,
            )
        
        if eval_dataset is not None:
            eval_ds = InstructionDataset(
                dataset=eval_dataset,
                tokenizer=tokenizer,
                max_seq_length=config.max_seq_length,
                instruction_column=config.instruction_column,
                input_column=config.input_column,
                output_column=config.output_column,
                chat_template=config.chat_template,
            )
        
        # Create dataloaders
        train_dataloader = DataLoader(
            train_ds,
            batch_size=config.per_device_batch_size,
            shuffle=True,
            num_workers=config.dataloader_num_workers,
            pin_memory=config.dataloader_pin_memory,
        ) if train_ds else None
        
        eval_dataloader = DataLoader(
            eval_ds,
            batch_size=config.per_device_batch_size,
            shuffle=False,
            num_workers=config.dataloader_num_workers,
        ) if eval_ds else None
        
        super().__init__(
            model=model,
            config=config,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
        )
    
    def _setup_optimizer(self):
        """Setup optimizer for LoRA parameters only."""
        if self._optimizer is not None:
            return
        
        if self.finetuning_config.use_lora:
            params = get_lora_parameters(self.model)
        else:
            params = [p for p in self.model.parameters() if p.requires_grad]
        
        if not params:
            raise ValueError("No trainable parameters found")
        
        self._optimizer = torch.optim.AdamW(
            params,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps,
            weight_decay=self.config.weight_decay,
        )
    
    @classmethod
    def from_huggingface(
        cls,
        model: "nn.Module",
        config: FinetuningConfig,
        tokenizer: Any,
    ) -> "FinetuningTrainer":
        """Create trainer with HuggingFace dataset."""
        if not HAS_DATASETS:
            raise ImportError("datasets library required")
        
        dataset = load_dataset(config.dataset_name, split="train")
        
        # Split for eval
        split_dataset = dataset.train_test_split(test_size=0.1, seed=config.seed)
        
        return cls(
            model=model,
            config=config,
            tokenizer=tokenizer,
            train_dataset=split_dataset["train"],
            eval_dataset=split_dataset["test"],
        )
