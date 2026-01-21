"""
Supervised Fine-Tuning (SFT)
============================
Standard supervised fine-tuning on instruction-response pairs.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from ..trainer_base import Trainer
from ..config import PostTrainingConfig


class SFTTrainer(Trainer):
    """
    Supervised Fine-Tuning trainer.
    
    Standard LM loss on instruction-response pairs.
    Masks instruction tokens from loss computation.
    """
    
    def __init__(
        self,
        model: "nn.Module",
        config: PostTrainingConfig,
        train_dataloader: Optional[DataLoader] = None,
        eval_dataloader: Optional[DataLoader] = None,
    ):
        super().__init__(
            model=model,
            config=config,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            compute_loss_fn=self._compute_sft_loss,
        )
    
    def _compute_sft_loss(
        self,
        model: "nn.Module",
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute SFT loss.
        
        Labels should have -100 for prompt tokens (masked from loss).
        """
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        labels = batch["labels"]
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        if hasattr(outputs, "loss") and outputs.loss is not None:
            return outputs.loss
        
        # Manual loss computation
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        
        return loss
