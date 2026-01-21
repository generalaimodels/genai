"""
Pretraining Trainer
===================
Causal LM pretraining with sequence packing and HuggingFace Datasets.
"""

from typing import Optional, Iterator, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path
import json

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, IterableDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from datasets import load_dataset, IterableDataset as HFIterableDataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

from .config import PretrainingConfig
from .trainer_base import Trainer, TrainerState


class PackedDataset(IterableDataset):
    """
    Sequence packing dataset for efficient pretraining.
    
    Concatenates sequences and splits into fixed-length chunks
    to maximize GPU utilization.
    """
    
    def __init__(
        self,
        dataset: "HFIterableDataset",
        tokenizer: Any,
        max_seq_length: int = 2048,
        text_column: str = "text",
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.text_column = text_column
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        buffer_input_ids = []
        buffer_attention_mask = []
        
        for example in self.dataset:
            text = example.get(self.text_column, "")
            if not text:
                continue
            
            # Tokenize
            encoded = self.tokenizer.encode(
                text,
                add_special_tokens=True,
                return_tensors=None,
            )
            
            if isinstance(encoded, dict):
                input_ids = encoded["input_ids"]
            else:
                input_ids = encoded
            
            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.tolist()
            
            buffer_input_ids.extend(input_ids)
            buffer_attention_mask.extend([1] * len(input_ids))
            
            # Yield chunks
            while len(buffer_input_ids) >= self.max_seq_length:
                chunk_ids = buffer_input_ids[:self.max_seq_length]
                chunk_mask = buffer_attention_mask[:self.max_seq_length]
                
                buffer_input_ids = buffer_input_ids[self.max_seq_length:]
                buffer_attention_mask = buffer_attention_mask[self.max_seq_length:]
                
                input_ids_tensor = torch.tensor(chunk_ids, dtype=torch.long)
                attention_mask_tensor = torch.tensor(chunk_mask, dtype=torch.long)
                
                # Labels are input_ids shifted by 1 (handled in loss)
                yield {
                    "input_ids": input_ids_tensor,
                    "attention_mask": attention_mask_tensor,
                    "labels": input_ids_tensor.clone(),
                }


class PretrainingTrainer(Trainer):
    """
    Pretraining trainer for causal language modeling.
    
    Features:
    - HuggingFace Datasets streaming
    - Sequence packing for efficiency
    - Fused cross-entropy loss
    """
    
    def __init__(
        self,
        model: "nn.Module",
        config: PretrainingConfig,
        tokenizer: Any,
        train_dataset: Optional[Any] = None,
        eval_dataset: Optional[Any] = None,
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch required")
        
        self.tokenizer = tokenizer
        self._train_dataset = train_dataset
        self._eval_dataset = eval_dataset
        
        # Build dataloaders
        train_dataloader = self._create_dataloader(train_dataset, is_train=True) if train_dataset else None
        eval_dataloader = self._create_dataloader(eval_dataset, is_train=False) if eval_dataset else None
        
        super().__init__(
            model=model,
            config=config,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            compute_loss_fn=self._compute_loss,
        )
        
        self.pretraining_config = config
    
    def _create_dataloader(
        self,
        dataset: Any,
        is_train: bool = True,
    ) -> DataLoader:
        """Create dataloader with optional sequence packing."""
        if self.pretraining_config.packing_enabled:
            dataset = PackedDataset(
                dataset=dataset,
                tokenizer=self.tokenizer,
                max_seq_length=self.config.max_seq_length,
                text_column=self.pretraining_config.text_column,
            )
        
        return DataLoader(
            dataset,
            batch_size=self.config.per_device_batch_size,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=self.config.dataloader_pin_memory,
            prefetch_factor=self.config.dataloader_prefetch_factor if self.config.dataloader_num_workers > 0 else None,
        )
    
    def _compute_loss(
        self,
        model: "nn.Module",
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute causal LM loss."""
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        labels = batch.get("labels", input_ids)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        if hasattr(outputs, "loss") and outputs.loss is not None:
            return outputs.loss
        
        # Manual loss computation with fused cross-entropy
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        
        if self.pretraining_config.use_fused_cross_entropy:
            try:
                from ..kernels.triton import FusedCrossEntropyLoss
                loss_fn = FusedCrossEntropyLoss()
                
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss = loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                return loss
            except ImportError:
                pass
        
        # Fallback to PyTorch cross-entropy
        loss_fn = nn.CrossEntropyLoss()
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        
        return loss
    
    @classmethod
    def from_huggingface(
        cls,
        model: "nn.Module",
        config: PretrainingConfig,
        tokenizer: Any,
    ) -> "PretrainingTrainer":
        """
        Create trainer with HuggingFace dataset.
        
        Args:
            model: Model to train
            config: Pretraining configuration
            tokenizer: Tokenizer for encoding
            
        Returns:
            Configured PretrainingTrainer
        """
        if not HAS_DATASETS:
            raise ImportError("datasets library required")
        
        train_dataset = load_dataset(
            config.dataset_name,
            config.dataset_config,
            split="train",
            streaming=config.streaming,
        )
        
        eval_dataset = None
        try:
            eval_dataset = load_dataset(
                config.dataset_name,
                config.dataset_config,
                split="validation",
                streaming=config.streaming,
            )
        except Exception:
            pass
        
        return cls(
            model=model,
            config=config,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
