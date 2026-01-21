"""
Group Relative Policy Optimization (GRPO)
==========================================
DeepSeek-R1 style group-based policy optimization.
"""

from typing import Optional, Dict, Any, List, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from ..trainer_base import Trainer, TrainerState
from ..config import PostTrainingConfig


class GRPOTrainer(Trainer):
    """
    Group Relative Policy Optimization trainer.
    
    Key insight: Use group-relative rewards instead of absolute rewards.
    For each prompt, generate multiple responses and rank them.
    Advantage = reward - mean(group_rewards)
    
    No value model needed (uses group statistics).
    """
    
    def __init__(
        self,
        model: "nn.Module",
        ref_model: "nn.Module",
        reward_model: "nn.Module",
        config: PostTrainingConfig,
        train_dataloader: Optional[DataLoader] = None,
        tokenizer: Any = None,
    ):
        super().__init__(
            model=model,
            config=config,
            train_dataloader=train_dataloader,
            compute_loss_fn=self._compute_grpo_loss,
        )
        
        self.ref_model = ref_model
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        self.reward_model = reward_model
        self.reward_model.eval()
        for param in self.reward_model.parameters():
            param.requires_grad = False
        
        self.tokenizer = tokenizer
        
        # GRPO hyperparameters
        self.group_size = config.grpo_group_size
        self.num_generations = config.grpo_num_generations
        self.kl_penalty = config.ppo_kl_penalty
        self.clip_range = config.ppo_clip_range
    
    @torch.no_grad()
    def generate_group(
        self,
        prompt: str,
        max_new_tokens: int = 256,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate a group of responses for one prompt.
        
        Returns:
            responses: (group_size, seq_len)
            rewards: (group_size,)
            log_probs: (group_size, seq_len)
        """
        prompt_ids = self.tokenizer.encode(
            prompt,
            return_tensors="pt",
        ).to(self.device)
        
        # Expand prompt for group
        prompt_ids = prompt_ids.expand(self.group_size, -1)
        
        responses_list = []
        log_probs_list = []
        
        self._model.eval()
        
        for _ in range(self.group_size):
            outputs = self.model.generate(
                prompt_ids[0:1],
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                return_dict_in_generate=True,
                output_scores=True,
            )
            
            response_ids = outputs.sequences
            responses_list.append(response_ids)
            
            # Compute log probs
            logits = self._model(response_ids).logits
            log_probs = self._compute_log_probs(logits, response_ids)
            log_probs_list.append(log_probs)
        
        # Pad to same length
        max_len = max(r.size(1) for r in responses_list)
        
        padded_responses = []
        padded_log_probs = []
        
        for resp, lp in zip(responses_list, log_probs_list):
            pad_len = max_len - resp.size(1)
            if pad_len > 0:
                resp = F.pad(resp, (0, pad_len), value=self.tokenizer.pad_token_id or 0)
                lp = F.pad(lp, (0, pad_len), value=0)
            padded_responses.append(resp)
            padded_log_probs.append(lp)
        
        responses = torch.cat(padded_responses, dim=0)
        log_probs = torch.stack([lp.squeeze(0) for lp in padded_log_probs], dim=0)
        
        # Compute rewards for each response
        rewards = self._compute_group_rewards(responses)
        
        self._model.train()
        
        return {
            "responses": responses,
            "log_probs": log_probs,
            "rewards": rewards,
            "prompt_length": prompt_ids.size(1),
        }
    
    def _compute_log_probs(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-token log probabilities."""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        per_token = torch.gather(
            log_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1),
        ).squeeze(-1)
        
        return per_token
    
    @torch.no_grad()
    def _compute_group_rewards(
        self,
        responses: torch.Tensor,
    ) -> torch.Tensor:
        """Compute rewards for group of responses."""
        outputs = self.reward_model(responses)
        
        if hasattr(outputs, "logits"):
            rewards = outputs.logits[:, -1]
        else:
            rewards = outputs[:, -1]
        
        return rewards
    
    def _compute_group_advantages(
        self,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute group-relative advantages.
        
        Advantage_i = reward_i - mean(rewards)
        Normalized by std for stability.
        """
        mean = rewards.mean()
        std = rewards.std() + 1e-8
        
        advantages = (rewards - mean) / std
        
        return advantages
    
    def _compute_grpo_loss(
        self,
        model: "nn.Module",
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute GRPO loss.
        
        Uses group-relative advantages for policy gradient.
        """
        responses = batch["responses"]
        old_log_probs = batch["log_probs"]
        advantages = batch["advantages"]
        
        # Forward pass for new log probs
        outputs = model(responses)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        
        new_log_probs = self._compute_log_probs(logits, responses)
        
        # Sequence log probs
        old_seq_lp = old_log_probs.sum(dim=-1)
        new_seq_lp = new_log_probs.sum(dim=-1)
        
        # Policy gradient with clipping
        ratio = torch.exp(new_seq_lp - old_seq_lp)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
        
        pg_loss_1 = -advantages * ratio
        pg_loss_2 = -advantages * clipped_ratio
        policy_loss = torch.max(pg_loss_1, pg_loss_2).mean()
        
        # KL penalty
        with torch.no_grad():
            ref_logits = self.ref_model(responses).logits
            ref_log_probs = self._compute_log_probs(ref_logits, responses)
            ref_seq_lp = ref_log_probs.sum(dim=-1)
        
        kl = (new_seq_lp - ref_seq_lp).mean()
        
        total_loss = policy_loss + self.kl_penalty * kl
        
        return total_loss
    
    def train(self) -> TrainerState:
        """GRPO training loop."""
        for step in range(self.config.max_steps):
            # Sample prompts
            prompts = self._sample_prompts()
            
            all_batches = []
            
            for prompt in prompts:
                # Generate group of responses
                group_data = self.generate_group(prompt)
                
                # Compute group-relative advantages
                advantages = self._compute_group_advantages(group_data["rewards"])
                group_data["advantages"] = advantages
                
                all_batches.append(group_data)
            
            # Training step
            total_loss = 0
            
            for batch in all_batches:
                loss = self._compute_grpo_loss(self._model, batch)
                
                loss.backward()
                total_loss += loss.item()
            
            # Gradient step
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.config.gradient_clipping)
            self._optimizer.step()
            self._scheduler.step()
            self._optimizer.zero_grad()
            
            self.state.global_step = step + 1
            self.state.training_loss = total_loss / len(all_batches)
            
            if step % self.config.logging_steps == 0:
                self._log_metrics()
            
            if step % self.config.save_steps == 0:
                self.save_checkpoint()
        
        return self.state
    
    def _sample_prompts(self) -> List[str]:
        """Sample prompts from dataloader."""
        try:
            batch = next(iter(self.train_dataloader))
            if "prompt" in batch:
                return batch["prompt"][:self.num_generations]
        except Exception:
            pass
        
        return ["Generate a helpful response:"] * self.num_generations
