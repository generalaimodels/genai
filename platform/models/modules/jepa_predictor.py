"""
JEPA Predictor (Joint-Embedding Predictive Architecture)
=========================================================
Latent world model for predictive representation learning.

Features:
- Predicts latent state transitions instead of raw outputs
- EMA target encoder for stable targets
- Masking strategy for context/target separation
- Energy-based training objective
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
import copy


class JEPAPredictor(nn.Module):
    """
    JEPA Predictor network.
    
    Predicts target representations from context representations
    in latent space. Core component of world model.
    
    Architecture:
        context_z -> [predictor] -> predicted_z
        target_z <- [target_encoder] <- target_x
        
        Loss: ||predicted_z - target_z||^2
    """
    
    def __init__(
        self,
        d_model: int,
        predictor_depth: int = 6,
        predictor_width: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.predictor_depth = predictor_depth
        self.predictor_width = predictor_width
        
        # Context encoding
        self.context_norm = nn.LayerNorm(d_model)
        
        # Target position embedding (for predicting specific positions)
        self.target_pos_embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Predictor transformer
        predictor_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=predictor_width,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.predictor = nn.TransformerEncoder(
            predictor_layer,
            num_layers=predictor_depth,
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )
    
    def forward(
        self,
        context_repr: torch.Tensor,
        target_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict target representations from context.
        
        Args:
            context_repr: Context representations (batch, ctx_len, d_model)
            target_positions: Positions to predict (batch, num_targets)
            
        Returns:
            Predicted representations (batch, num_targets, d_model)
        """
        batch_size, ctx_len, d_model = context_repr.shape
        num_targets = target_positions.shape[1]
        
        # Normalize context
        context = self.context_norm(context_repr)
        
        # Create target queries
        target_queries = self.target_pos_embed.expand(batch_size, num_targets, -1)
        
        # Add positional information to queries
        # (In production, would use proper positional encoding)
        pos_offset = target_positions.unsqueeze(-1).float() / 1000.0
        target_queries = target_queries + pos_offset
        
        # Concatenate context and target queries
        combined = torch.cat([context, target_queries], dim=1)
        
        # Predict
        output = self.predictor(combined)
        
        # Extract target predictions
        predictions = output[:, ctx_len:]
        predictions = self.output_proj(predictions)
        
        return predictions


class EMAEncoder(nn.Module):
    """
    Exponential Moving Average encoder for stable targets.
    
    Maintains an EMA copy of the online encoder for
    computing regression targets.
    """
    
    def __init__(
        self,
        online_encoder: nn.Module,
        ema_decay: float = 0.999,
    ):
        super().__init__()
        self.online_encoder = online_encoder
        self.ema_decay = ema_decay
        
        # Create EMA copy
        self.target_encoder = copy.deepcopy(online_encoder)
        
        # Freeze target encoder
        for param in self.target_encoder.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def update_ema(self):
        """Update target encoder with EMA of online encoder."""
        for online_param, target_param in zip(
            self.online_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            target_param.data = (
                self.ema_decay * target_param.data +
                (1.0 - self.ema_decay) * online_param.data
            )
    
    def encode_online(self, x: torch.Tensor) -> torch.Tensor:
        """Encode with online encoder (gradients flow)."""
        return self.online_encoder(x)
    
    @torch.no_grad()
    def encode_target(self, x: torch.Tensor) -> torch.Tensor:
        """Encode with target encoder (no gradients)."""
        return self.target_encoder(x)


class LatentWorldModel(nn.Module):
    """
    Complete Latent World Model with JEPA objective.
    
    Learns to predict future states in latent space,
    enabling physical/causal reasoning without
    explicit supervision.
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        d_model: int,
        predictor_depth: int = 6,
        predictor_width: int = 2048,
        ema_decay: float = 0.999,
        mask_ratio: float = 0.75,
        loss_weight: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.mask_ratio = mask_ratio
        self.loss_weight = loss_weight
        
        # EMA encoder pair
        self.ema_encoder = EMAEncoder(encoder, ema_decay)
        
        # Predictor
        self.predictor = JEPAPredictor(
            d_model=d_model,
            predictor_depth=predictor_depth,
            predictor_width=predictor_width,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass computing JEPA loss.
        
        Args:
            x: Input (batch, seq_len, d_model or input_dim)
            context_mask: Which positions to use as context
            target_mask: Which positions to predict
            
        Returns:
            Tuple of (predicted_repr, jepa_loss)
        """
        batch_size, seq_len = x.shape[:2]
        device = x.device
        
        # Generate masks if not provided
        if context_mask is None or target_mask is None:
            context_mask, target_mask = self._generate_masks(batch_size, seq_len, device)
        
        # Encode full sequence with target encoder
        with torch.no_grad():
            target_repr = self.ema_encoder.encode_target(x)
        
        # Encode context with online encoder
        context_repr = self.ema_encoder.encode_online(x)
        
        # Mask to get context only
        context = context_repr * context_mask.unsqueeze(-1)
        
        # Get target positions
        target_positions = target_mask.nonzero(as_tuple=True)[1].view(batch_size, -1)
        
        # Predict target representations
        predictions = self.predictor(context, target_positions)
        
        # Get target representations
        targets = target_repr[target_mask].view(batch_size, -1, self.d_model)
        
        # Compute loss (L2 in latent space)
        loss = F.mse_loss(predictions, targets.detach())
        loss = self.loss_weight * loss
        
        # Update EMA encoder
        if self.training:
            self.ema_encoder.update_ema()
        
        return predictions, loss
    
    def _generate_masks(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate random context/target masks."""
        num_mask = int(seq_len * self.mask_ratio)
        num_context = seq_len - num_mask
        
        # Random permutation for each batch
        rand_indices = torch.rand(batch_size, seq_len, device=device).argsort(dim=1)
        
        # Context: first (1 - mask_ratio) positions
        context_indices = rand_indices[:, :num_context]
        target_indices = rand_indices[:, num_context:]
        
        # Create masks
        context_mask = torch.zeros(batch_size, seq_len, device=device, dtype=torch.bool)
        target_mask = torch.zeros(batch_size, seq_len, device=device, dtype=torch.bool)
        
        context_mask.scatter_(1, context_indices, True)
        target_mask.scatter_(1, target_indices, True)
        
        return context_mask.float(), target_mask
    
    def predict_future(
        self,
        context: torch.Tensor,
        num_steps: int = 1,
    ) -> torch.Tensor:
        """
        Autoregressively predict future states.
        
        Args:
            context: Initial context (batch, ctx_len, d_model)
            num_steps: Number of future steps to predict
            
        Returns:
            Predicted future states (batch, num_steps, d_model)
        """
        batch_size = context.shape[0]
        device = context.device
        
        current_context = context
        predictions = []
        
        for step in range(num_steps):
            # Predict next position
            next_pos = torch.full(
                (batch_size, 1),
                current_context.shape[1],
                device=device,
                dtype=torch.long
            )
            
            pred = self.predictor(current_context, next_pos)
            predictions.append(pred)
            
            # Extend context with prediction
            current_context = torch.cat([current_context, pred], dim=1)
        
        return torch.cat(predictions, dim=1)


class VICRegObjective(nn.Module):
    """
    VICReg (Variance-Invariance-Covariance Regularization) objective.
    
    Alternative to JEPA's MSE loss with explicit regularization
    for representation quality.
    """
    
    def __init__(
        self,
        d_model: int,
        invariance_weight: float = 25.0,
        variance_weight: float = 25.0,
        covariance_weight: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.invariance_weight = invariance_weight
        self.variance_weight = variance_weight
        self.covariance_weight = covariance_weight
    
    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute VICReg loss between two views.
        
        Args:
            z1, z2: Two views of same examples (batch, d_model)
            
        Returns:
            Tuple of (total_loss, loss_components)
        """
        # Invariance: MSE between views
        invariance_loss = F.mse_loss(z1, z2)
        
        # Variance: std should be > 1
        std_z1 = z1.std(dim=0)
        std_z2 = z2.std(dim=0)
        variance_loss = (
            F.relu(1.0 - std_z1).mean() +
            F.relu(1.0 - std_z2).mean()
        )
        
        # Covariance: off-diagonal should be 0
        z1_centered = z1 - z1.mean(dim=0)
        z2_centered = z2 - z2.mean(dim=0)
        
        cov_z1 = (z1_centered.T @ z1_centered) / (z1.shape[0] - 1)
        cov_z2 = (z2_centered.T @ z2_centered) / (z2.shape[0] - 1)
        
        # Off-diagonal elements
        off_diag_z1 = cov_z1.pow(2).sum() - cov_z1.diag().pow(2).sum()
        off_diag_z2 = cov_z2.pow(2).sum() - cov_z2.diag().pow(2).sum()
        covariance_loss = (off_diag_z1 + off_diag_z2) / self.d_model
        
        # Total loss
        total_loss = (
            self.invariance_weight * invariance_loss +
            self.variance_weight * variance_loss +
            self.covariance_weight * covariance_loss
        )
        
        return total_loss, {
            "invariance": invariance_loss,
            "variance": variance_loss,
            "covariance": covariance_loss,
        }
