"""
Direct Preference Optimization (DPO)
====================================
Offline preference learning without reward model.
"""

from typing import Optional, Dict, Any, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from ..trainer_base import Trainer
from ..config import PostTrainingConfig


class DPOTrainer(Trainer):
    """
    Direct Preference Optimization trainer.
    
    Loss: -log(sigmoid(beta * (log(pi/ref)_chosen - log(pi/ref)_rejected)))
    
    No reward model needed - directly optimizes from preference pairs.
    """
    
    def __init__(
        self,
        model: "nn.Module",
        ref_model: "nn.Module",
        config: PostTrainingConfig,
        train_dataloader: Optional[DataLoader] = None,
        eval_dataloader: Optional[DataLoader] = None,
    ):
        super().__init__(
            model=model,
            config=config,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            compute_loss_fn=self._compute_dpo_loss,
        )
        
        self.ref_model = ref_model
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        self.beta = config.dpo_beta
        self.label_smoothing = config.dpo_label_smoothing
    
    def _compute_logps(
        self,
        model: "nn.Module",
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probabilities for sequences.
        
        Returns per-token log probs summed over sequence.
        """
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        
        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous()
        
        # Per-token log probs
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Gather log probs for actual tokens
        per_token_logps = torch.gather(
            log_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1),
        ).squeeze(-1)
        
        # Mask padding and sum
        per_token_logps = per_token_logps * shift_mask
        sequence_logps = per_token_logps.sum(dim=-1)
        
        return sequence_logps
    
    def _compute_dpo_loss(
        self,
        model: "nn.Module",
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute DPO loss.
        
        Batch should contain:
        - chosen_input_ids, chosen_attention_mask, chosen_labels
        - rejected_input_ids, rejected_attention_mask, rejected_labels
        """
        # Policy log probs
        policy_chosen_logps = self._compute_logps(
            model,
            batch["chosen_input_ids"],
            batch["chosen_attention_mask"],
            batch["chosen_labels"],
        )
        
        policy_rejected_logps = self._compute_logps(
            model,
            batch["rejected_input_ids"],
            batch["rejected_attention_mask"],
            batch["rejected_labels"],
        )
        
        # Reference log probs (no gradient)
        with torch.no_grad():
            ref_chosen_logps = self._compute_logps(
                self.ref_model,
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                batch["chosen_labels"],
            )
            
            ref_rejected_logps = self._compute_logps(
                self.ref_model,
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                batch["rejected_labels"],
            )
        
        # DPO loss
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        
        logits = pi_logratios - ref_logratios
        
        # Label smoothing
        if self.label_smoothing > 0:
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        else:
            losses = -F.logsigmoid(self.beta * logits)
        
        return losses.mean()
    
    @torch.no_grad()
    def compute_metrics(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Compute DPO evaluation metrics."""
        policy_chosen_logps = self._compute_logps(
            self.model,
            batch["chosen_input_ids"],
            batch["chosen_attention_mask"],
            batch["chosen_labels"],
        )
        
        policy_rejected_logps = self._compute_logps(
            self.model,
            batch["rejected_input_ids"],
            batch["rejected_attention_mask"],
            batch["rejected_labels"],
        )
        
        # Accuracy: chosen > rejected
        accuracy = (policy_chosen_logps > policy_rejected_logps).float().mean()
        
        # Reward margin
        margin = (policy_chosen_logps - policy_rejected_logps).mean()
        
        return {
            "accuracy": accuracy.item(),
            "reward_margin": margin.item(),
        }
