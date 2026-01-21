"""
Dual-Head Predictor Module
World Model (JEPA) + Language Modeling (Medusa) prediction heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class JEPAPredictor(nn.Module):
    """
    Joint-Embedding Predictive Architecture
    Predicts future latent representations for world modeling
    """
    
    def __init__(self, d_model: int, d_latent: int = None, n_future: int = 1):
        super().__init__()
        
        self.d_model = d_model
        self.d_latent = d_latent or d_model
        self.n_future = n_future
        
        # Predictor MLP
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, self.d_latent * n_future),
        )
        
        # Target encoder (stop gradient)
        self.target_encoder = nn.Linear(d_model, self.d_latent, bias=False)
        
        # Freeze target encoder (updated via EMA in training)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
    
    def forward(self, x, x_future=None):
        """
        Args:
            x: [B, L, D] current hidden state
            x_future: [B, n_future, D] future hidden states (ground truth, training only)
            
        Returns:
            z_pred: [B, L, n_future, D_latent] predicted future latents
            loss: JEPA loss (if x_future provided)
        """
        B, L, D = x.shape
        
        # Predict future latents
        z_pred = self.predictor(x)  # [B, L, n_future * D_latent]
        z_pred = z_pred.view(B, L, self.n_future, self.d_latent)
        
        # Compute loss if ground truth provided
        loss = None
        if x_future is not None and self.training:
            # Encode future with stop-gradient
            with torch.no_grad():
                z_target = self.target_encoder(x_future)  # [B, n_future, D_latent]
            
            # L2 loss between prediction and target
            # Broadcast: z_pred[:, -n_future:] vs z_target
            z_pred_last = z_pred[:, -self.n_future:, :, :]  # [B, n_future, n_future, D_latent]
            
            # Only compare diagonal (t predicts t+1)
            diagonal_pred = torch.diagonal(z_pred_last, dim1=1, dim2=2)  # [B, D_latent, n_future]
            diagonal_pred = diagonal_pred.transpose(1, 2)  # [B, n_future, D_latent]
            
            loss = F.mse_loss(diagonal_pred, z_target)
        
        return z_pred, loss


class MedusaHeads(nn.Module):
    """
    Speculative Multi-Token Decoding Heads
    Generates K future tokens in parallel
    """
    
    def __init__(self, d_model: int, vocab_size: int, n_heads: int = 3):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_heads = n_heads
        
        # Medusa heads: each predicts tokens at t+k
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, vocab_size, bias=False),
            )
            for _ in range(n_heads)
        ])
    
    def forward(self, x):
        """
        Args:
            x: [B, L, D] final hidden states
            
        Returns:
            logits: list of [B, L, V] for each head
        """
        logits = [head(x) for head in self.heads]
        return logits


class DualHeadPredictor(nn.Module):
    """
    Combines JEPA (world model) and Medusa (language) prediction
    """
    
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        d_latent: int = None,
        n_future_jepa: int = 1,
        n_heads_medusa: int = 3,
    ):
        super().__init__()
        
        # World model head
        self.jepa = JEPAPredictor(d_model, d_latent, n_future_jepa)
        
        # Language model heads
        self.medusa = MedusaHeads(d_model, vocab_size, n_heads_medusa)
    
    def forward(self, x, x_future=None, mode='both'):
        """
        Args:
            x: [B, L, D] final hidden states
            x_future: [B, n_future, D] future states (training)
            mode: 'jepa', 'medusa', or 'both'
            
        Returns:
            outputs: dict with 'jepa' and/or 'medusa' predictions
        """
        outputs = {}
        
        if mode in ['jepa', 'both']:
            z_pred, jepa_loss = self.jepa(x, x_future)
            outputs['jepa'] = {'z_pred': z_pred, 'loss': jepa_loss}
        
        if mode in ['medusa', 'both']:
            logits = self.medusa(x)
            outputs['medusa'] = {'logits': logits}
        
        return outputs
