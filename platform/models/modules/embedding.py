"""
Continuous Embedding Layer
==========================
Tokenizer-free continuous latent space encoder.

Features:
- Direct mapping from input to continuous latent space
- Patch-based encoding for vision/audio
- Learned positional embeddings
- Modality-agnostic interface
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class ContinuousEmbedding(nn.Module):
    """
    Continuous embedding layer for tokenizer-free operation.
    
    Maps inputs directly to continuous latent representations
    without discrete tokenization.
    """
    
    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 131072,
        learnable_position: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Positional embeddings
        if learnable_position:
            self.position_embedding = nn.Embedding(max_seq_len, d_model)
        else:
            # Sinusoidal
            position = torch.arange(max_seq_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(max_seq_len, d_model)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("position_embedding", pe)
        
        self.learnable_position = learnable_position
        self.dropout = nn.Dropout(dropout)
        
        # Scaling
        self.scale = d_model ** 0.5
    
    def forward(
        self,
        latent_vectors: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Add positional embeddings to latent vectors.
        
        Args:
            latent_vectors: (batch, seq_len, d_model)
            position_ids: Optional position indices
            
        Returns:
            Embedded vectors (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = latent_vectors.shape
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=latent_vectors.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        if self.learnable_position:
            pos_embed = self.position_embedding(position_ids)
        else:
            pos_embed = self.position_embedding[position_ids]
        
        output = latent_vectors * self.scale + pos_embed
        output = self.dropout(output)
        
        return output


class TokenFreeLateralEncoder(nn.Module):
    """
    Token-free lateral encoder for continuous-to-continuous mapping.
    
    Used for encoding continuous inputs (audio, vision patches)
    directly into the model's latent space without tokenization.
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        num_layers: int = 2,
        activation: str = "gelu",
        dropout: float = 0.0,
        normalize_input: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input normalization
        if normalize_input:
            self.input_norm = nn.LayerNorm(input_dim)
        else:
            self.input_norm = nn.Identity()
        
        # Build encoder layers
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            out_dim = d_model if i == num_layers - 1 else (input_dim + d_model) // 2
            
            layers.append(nn.Linear(current_dim, out_dim))
            
            if i < num_layers - 1:
                if activation == "gelu":
                    layers.append(nn.GELU())
                elif activation == "silu":
                    layers.append(nn.SiLU())
                else:
                    layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            
            current_dim = out_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Output normalization
        self.output_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode continuous input to model latent space.
        
        Args:
            x: (batch, seq_len, input_dim) or (batch, input_dim)
            
        Returns:
            Encoded vectors (batch, seq_len, d_model) or (batch, d_model)
        """
        x = self.input_norm(x)
        x = self.encoder(x)
        x = self.output_norm(x)
        return x


class AdaptiveInputEmbedding(nn.Module):
    """
    Adaptive input embedding with vocabulary clustering.
    
    Uses fewer parameters for rare tokens, more for common ones.
    Alternative to standard embedding for vocabulary efficiency.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        cutoffs: Tuple[int, ...] = (20000, 60000),
        div_value: float = 4.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.cutoffs = [0] + list(cutoffs) + [vocab_size]
        self.div_value = div_value
        
        # Build embedding layers for each cluster
        self.embeddings = nn.ModuleList()
        self.projections = nn.ModuleList()
        
        for i in range(len(self.cutoffs) - 1):
            cluster_size = self.cutoffs[i + 1] - self.cutoffs[i]
            cluster_dim = int(d_model / (div_value ** i))
            
            self.embeddings.append(nn.Embedding(cluster_size, cluster_dim))
            
            if cluster_dim != d_model:
                self.projections.append(nn.Linear(cluster_dim, d_model, bias=False))
            else:
                self.projections.append(nn.Identity())
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Embed tokens with cluster-based dimensionality.
        """
        batch_size, seq_len = input_ids.shape
        output = torch.zeros(
            batch_size, seq_len, self.d_model,
            device=input_ids.device, dtype=self.embeddings[0].weight.dtype
        )
        
        for i in range(len(self.cutoffs) - 1):
            low = self.cutoffs[i]
            high = self.cutoffs[i + 1]
            
            mask = (input_ids >= low) & (input_ids < high)
            if mask.any():
                local_ids = input_ids[mask] - low
                embedded = self.embeddings[i](local_ids)
                projected = self.projections[i](embedded)
                output[mask] = projected
        
        return output


class MultiModalEmbedding(nn.Module):
    """
    Multi-modal embedding combining text, vision, and audio.
    
    Each modality has its own encoder, outputs are aligned
    in shared latent space.
    """
    
    def __init__(
        self,
        d_model: int,
        vocab_size: int = 128256,
        image_size: int = 224,
        patch_size: int = 14,
        audio_dim: int = 80,
        audio_stride: int = 160,
    ):
        super().__init__()
        self.d_model = d_model
        
        # Text embedding
        self.text_embedding = nn.Embedding(vocab_size, d_model)
        
        # Vision patch embedding
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size
        
        self.vision_embedding = nn.Sequential(
            nn.Linear(patch_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.vision_position = nn.Embedding(num_patches + 1, d_model)  # +1 for CLS
        self.vision_cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Audio embedding
        self.audio_embedding = TokenFreeLateralEncoder(
            input_dim=audio_dim,
            d_model=d_model,
            num_layers=2,
        )
        
        # Modality type embedding
        self.modality_embedding = nn.Embedding(3, d_model)  # text, vision, audio
    
    def embed_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed text tokens."""
        embed = self.text_embedding(input_ids)
        modality = self.modality_embedding(
            torch.zeros_like(input_ids[:, :1]).expand(-1, input_ids.shape[1])
        )
        return embed + modality
    
    def embed_vision(self, images: torch.Tensor) -> torch.Tensor:
        """
        Embed vision patches.
        
        Args:
            images: (batch, channels, height, width)
        """
        batch_size = images.shape[0]
        
        # Patchify
        patches = images.unfold(2, self.patch_size, self.patch_size)
        patches = patches.unfold(3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(batch_size, -1, -1)  # (batch, num_patches, patch_dim)
        
        # Embed
        embed = self.vision_embedding(patches)
        
        # Add CLS token
        cls_tokens = self.vision_cls.expand(batch_size, -1, -1)
        embed = torch.cat([cls_tokens, embed], dim=1)
        
        # Add position
        positions = torch.arange(embed.shape[1], device=embed.device)
        pos_embed = self.vision_position(positions)
        embed = embed + pos_embed
        
        # Add modality
        modality = self.modality_embedding(
            torch.ones(batch_size, 1, dtype=torch.long, device=embed.device)
        )
        embed = embed + modality
        
        return embed
    
    def embed_audio(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Embed audio features.
        
        Args:
            mel_spec: (batch, time, mel_bins)
        """
        embed = self.audio_embedding(mel_spec)
        
        # Add modality
        batch_size, seq_len, _ = embed.shape
        modality = self.modality_embedding(
            torch.full((batch_size, seq_len), 2, dtype=torch.long, device=embed.device)
        )
        embed = embed + modality
        
        return embed
