"""
Proximal Policy Optimization (PPO)
==================================
Online RLHF with reward model feedback.
"""

from typing import Optional, Dict, Any, Tuple, List

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


class ValueHead(nn.Module):
    """Value head for PPO critic."""
    
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.value = nn.Linear(hidden_size, 1)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.dropout(F.relu(self.dense(hidden_states)))
        return self.value(x).squeeze(-1)


class PPOTrainer(Trainer):
    """
    Proximal Policy Optimization trainer.
    
    Components:
    - Policy model (actor)
    - Value model (critic)
    - Reference model (KL constraint)
    - Reward model (feedback)
    
    PPO objective: clip(ratio, 1-eps, 1+eps) * advantage
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
            compute_loss_fn=self._compute_ppo_loss,
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
        
        # Value head
        hidden_size = getattr(model.config, "hidden_size", 4096)
        self.value_head = ValueHead(hidden_size).to(self.device)
        
        # PPO hyperparameters
        self.clip_range = config.ppo_clip_range
        self.value_loss_coef = config.ppo_value_loss_coef
        self.entropy_coef = config.ppo_entropy_coef
        self.kl_penalty = config.ppo_kl_penalty
        self.ppo_epochs = config.ppo_epochs
        
        # Experience buffer
        self.experience_buffer: List[Dict[str, torch.Tensor]] = []
    
    @torch.no_grad()
    def generate_experience(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Generate responses and compute rewards.
        
        Returns list of experiences with:
        - input_ids, attention_mask
        - log_probs (old policy)
        - values (old value estimates)
        - rewards (from reward model)
        """
        experiences = []
        
        for prompt in prompts:
            # Tokenize prompt
            prompt_ids = self.tokenizer.encode(
                prompt,
                return_tensors="pt",
            ).to(self.device)
            
            # Generate response
            self._model.eval()
            with torch.no_grad():
                outputs = self.model.generate(
                    prompt_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            
            response_ids = outputs.sequences
            
            # Compute log probs for generated tokens
            logits = self._model(response_ids).logits
            log_probs = self._compute_sequence_log_probs(logits, response_ids)
            
            # Compute values
            hidden = self._get_last_hidden(response_ids)
            values = self.value_head(hidden)
            
            # Compute reward
            reward = self._compute_reward(response_ids)
            
            # Compute KL penalty
            ref_logits = self.ref_model(response_ids).logits
            ref_log_probs = self._compute_sequence_log_probs(ref_logits, response_ids)
            kl = (log_probs - ref_log_probs).sum(dim=-1)
            
            # Adjusted reward with KL penalty
            adjusted_reward = reward - self.kl_penalty * kl
            
            experiences.append({
                "input_ids": response_ids,
                "attention_mask": torch.ones_like(response_ids),
                "old_log_probs": log_probs,
                "old_values": values,
                "rewards": adjusted_reward,
                "prompt_length": prompt_ids.size(1),
            })
        
        self._model.train()
        return experiences
    
    def _compute_sequence_log_probs(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-token log probs."""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        per_token = torch.gather(
            log_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1),
        ).squeeze(-1)
        
        return per_token
    
    def _get_last_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get last hidden state for value prediction."""
        outputs = self._model(input_ids, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        return hidden[:, -1, :]
    
    @torch.no_grad()
    def _compute_reward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute reward from reward model."""
        outputs = self.reward_model(input_ids)
        if hasattr(outputs, "logits"):
            return outputs.logits[:, -1]
        return outputs[:, -1]
    
    def _compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).
        """
        batch_size, seq_len = values.shape
        
        advantages = torch.zeros_like(values)
        last_gae = 0
        
        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                next_value = 0
            else:
                next_value = values[:, t + 1]
            
            delta = rewards[:, t] + gamma * next_value - values[:, t]
            advantages[:, t] = last_gae = delta + gamma * lam * last_gae
        
        returns = advantages + values
        
        return advantages, returns
    
    def _compute_ppo_loss(
        self,
        model: "nn.Module",
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute PPO loss.
        
        L = L_policy + c1 * L_value - c2 * entropy
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        old_log_probs = batch["old_log_probs"]
        old_values = batch["old_values"]
        advantages = batch["advantages"]
        returns = batch["returns"]
        
        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        
        # New log probs
        new_log_probs = self._compute_sequence_log_probs(logits, input_ids)
        
        # Policy loss (clipped)
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
        
        policy_loss_1 = -advantages * ratio
        policy_loss_2 = -advantages * clipped_ratio
        policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()
        
        # Value loss
        hidden = self._get_last_hidden(input_ids)
        new_values = self.value_head(hidden)
        
        value_loss = F.mse_loss(new_values, returns)
        
        # Entropy bonus
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        
        # Total loss
        total_loss = (
            policy_loss +
            self.value_loss_coef * value_loss -
            self.entropy_coef * entropy
        )
        
        return total_loss
    
    def train(self) -> TrainerState:
        """PPO training loop with experience generation."""
        for step in range(self.config.max_steps):
            # Generate experience
            prompts = self._sample_prompts()
            experiences = self.generate_experience(prompts)
            
            # Compute advantages
            for exp in experiences:
                adv, ret = self._compute_advantages(
                    exp["rewards"].unsqueeze(0),
                    exp["old_values"].unsqueeze(0),
                )
                exp["advantages"] = adv.squeeze(0)
                exp["returns"] = ret.squeeze(0)
            
            # PPO epochs
            for _ in range(self.ppo_epochs):
                for exp in experiences:
                    loss = self._compute_ppo_loss(self._model, exp)
                    
                    loss.backward()
                    self._optimizer.step()
                    self._optimizer.zero_grad()
            
            self.state.global_step = step + 1
            
            if step % self.config.logging_steps == 0:
                self._log_metrics()
        
        return self.state
    
    def _sample_prompts(self) -> List[str]:
        """Sample prompts from dataloader."""
        batch = next(iter(self.train_dataloader))
        if "prompt" in batch:
            return batch["prompt"]
        return [""] * self.config.per_device_batch_size
