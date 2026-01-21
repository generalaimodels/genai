"""
Continuous Latent Encoder
=========================
Tokenizer-free encoding for continuous input modalities.

Features:
- Patch-based encoding for images
- Spectral encoding for audio
- Contrastive alignment to text space
- VQ-VAE style discrete bottleneck (optional)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class PatchEncoder(nn.Module):
    """
    Patch-based encoder for vision inputs.
    
    Converts images into a sequence of latent patch embeddings
    aligned with the model's text embedding space.
    """
    
    def __init__(
        self,
        d_model: int,
        image_size: int = 224,
        patch_size: int = 14,
        in_channels: int = 3,
        num_layers: int = 4,
        num_heads: int = 8,
        use_cls_token: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        self.image_size = image_size
        self.use_cls_token = use_cls_token
        
        self.num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size
        
        # Patch projection
        self.patch_proj = nn.Conv2d(
            in_channels, d_model,
            kernel_size=patch_size, stride=patch_size
        )
        
        # CLS token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Positional embedding
        num_positions = self.num_patches + (1 if use_cls_token else 0)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_positions, d_model) * 0.02)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output norm
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        images: torch.Tensor,
        return_all_patches: bool = True,
    ) -> torch.Tensor:
        """
        Encode images to latent patches.
        
        Args:
            images: (batch, channels, height, width)
            return_all_patches: If False, return only CLS token
            
        Returns:
            Encoded patches (batch, num_patches, d_model)
        """
        batch_size = images.shape[0]
        
        # Patchify and project
        x = self.patch_proj(images)  # (batch, d_model, h', w')
        x = x.flatten(2).transpose(1, 2)  # (batch, num_patches, d_model)
        
        # Add CLS token
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embedding
        
        # Transform
        x = self.transformer(x)
        x = self.norm(x)
        
        if return_all_patches:
            return x
        else:
            return x[:, 0]  # CLS token only


class SpectralEncoder(nn.Module):
    """
    Spectral encoder for audio inputs.
    
    Converts mel-spectrograms to latent representations.
    """
    
    def __init__(
        self,
        d_model: int,
        mel_bins: int = 80,
        num_layers: int = 4,
        num_heads: int = 8,
        conv_layers: int = 2,
        conv_kernel: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.mel_bins = mel_bins
        
        # Convolutional front-end
        conv_blocks = []
        in_channels = 1
        out_channels = d_model // 4
        
        for i in range(conv_layers):
            conv_blocks.extend([
                nn.Conv1d(
                    in_channels if i == 0 else out_channels,
                    out_channels * 2 if i < conv_layers - 1 else d_model,
                    kernel_size=conv_kernel,
                    padding=conv_kernel // 2,
                    stride=2 if i < conv_layers - 1 else 1,
                ),
                nn.GELU(),
            ])
            in_channels = out_channels
            out_channels *= 2
        
        self.conv_frontend = nn.Sequential(*conv_blocks)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Norm
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Encode mel-spectrogram.
        
        Args:
            mel_spec: (batch, time, mel_bins)
            
        Returns:
            Encoded sequence (batch, seq_len, d_model)
        """
        # Reshape for conv1d: (batch, mel_bins, time) -> treat as channels
        x = mel_spec.transpose(1, 2)
        
        # Apply conv frontend
        x = self.conv_frontend(x)  # (batch, d_model, time')
        x = x.transpose(1, 2)  # (batch, time', d_model)
        
        # Transform
        x = self.transformer(x)
        x = self.norm(x)
        
        return x


class ContinuousLatentEncoder(nn.Module):
    """
    Unified continuous latent encoder supporting multiple modalities.
    
    Creates aligned representations in a shared latent space.
    """
    
    def __init__(
        self,
        d_model: int,
        image_size: int = 224,
        patch_size: int = 14,
        mel_bins: int = 80,
        use_contrastive_align: bool = True,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_contrastive_align = use_contrastive_align
        self.temperature = temperature
        
        # Modality-specific encoders
        self.vision_encoder = PatchEncoder(
            d_model=d_model,
            image_size=image_size,
            patch_size=patch_size,
        )
        
        self.audio_encoder = SpectralEncoder(
            d_model=d_model,
            mel_bins=mel_bins,
        )
        
        # Alignment projections (for contrastive learning)
        if use_contrastive_align:
            self.vision_proj = nn.Linear(d_model, d_model)
            self.audio_proj = nn.Linear(d_model, d_model)
            self.text_proj = nn.Linear(d_model, d_model)
    
    def encode_vision(
        self,
        images: torch.Tensor,
        return_for_alignment: bool = False,
    ) -> torch.Tensor:
        """Encode vision input."""
        features = self.vision_encoder(images)
        
        if return_for_alignment and self.use_contrastive_align:
            # Return CLS token projected for alignment
            cls_feat = features[:, 0]
            return self.vision_proj(cls_feat)
        
        return features
    
    def encode_audio(
        self,
        mel_spec: torch.Tensor,
        return_for_alignment: bool = False,
    ) -> torch.Tensor:
        """Encode audio input."""
        features = self.audio_encoder(mel_spec)
        
        if return_for_alignment and self.use_contrastive_align:
            # Mean pool and project
            pooled = features.mean(dim=1)
            return self.audio_proj(pooled)
        
        return features
    
    def encode_text(
        self,
        text_features: torch.Tensor,
        return_for_alignment: bool = False,
    ) -> torch.Tensor:
        """Project text features for alignment."""
        if return_for_alignment and self.use_contrastive_align:
            # Assume text_features is [CLS] representation
            return self.text_proj(text_features)
        return text_features
    
    def compute_alignment_loss(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        audio_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute contrastive alignment loss between modalities.
        
        Uses InfoNCE loss for cross-modal alignment.
        """
        # Normalize features
        vision_norm = F.normalize(vision_features, dim=-1)
        text_norm = F.normalize(text_features, dim=-1)
        
        # Vision-text alignment
        logits_v2t = torch.matmul(vision_norm, text_norm.T) / self.temperature
        logits_t2v = logits_v2t.T
        
        batch_size = vision_features.shape[0]
        labels = torch.arange(batch_size, device=vision_features.device)
        
        loss_v2t = F.cross_entropy(logits_v2t, labels)
        loss_t2v = F.cross_entropy(logits_t2v, labels)
        
        total_loss = (loss_v2t + loss_t2v) / 2
        
        # Add audio alignment if present
        if audio_features is not None:
            audio_norm = F.normalize(audio_features, dim=-1)
            
            logits_a2t = torch.matmul(audio_norm, text_norm.T) / self.temperature
            logits_t2a = logits_a2t.T
            
            loss_a2t = F.cross_entropy(logits_a2t, labels)
            loss_t2a = F.cross_entropy(logits_t2a, labels)
            
            total_loss = total_loss + (loss_a2t + loss_t2a) / 2
        
        return total_loss


class VQEncoder(nn.Module):
    """
    Vector Quantized encoder for discrete bottleneck.
    
    Optional component for models that benefit from
    discrete latent representations.
    """
    
    def __init__(
        self,
        d_model: int,
        num_codes: int = 8192,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_codes = num_codes
        self.commitment_cost = commitment_cost
        
        # Codebook
        self.codebook = nn.Embedding(num_codes, d_model)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_codes, 1.0 / num_codes)
    
    def forward(
        self,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize continuous latents.
        
        Args:
            z: Continuous latents (batch, seq_len, d_model)
            
        Returns:
            Tuple of (quantized, commitment_loss, indices)
        """
        # Flatten
        flat_z = z.reshape(-1, self.d_model)
        
        # Compute distances to codebook
        distances = (
            (flat_z ** 2).sum(dim=-1, keepdim=True)
            + (self.codebook.weight ** 2).sum(dim=-1)
            - 2 * torch.matmul(flat_z, self.codebook.weight.T)
        )
        
        # Find nearest codes
        indices = distances.argmin(dim=-1)
        quantized_flat = self.codebook(indices)
        
        # Reshape
        quantized = quantized_flat.view_as(z)
        indices = indices.view(z.shape[:-1])
        
        # Compute losses
        commitment_loss = self.commitment_cost * F.mse_loss(z, quantized.detach())
        codebook_loss = F.mse_loss(z.detach(), quantized)
        
        # Straight-through estimator
        quantized = z + (quantized - z).detach()
        
        return quantized, commitment_loss + codebook_loss, indices
