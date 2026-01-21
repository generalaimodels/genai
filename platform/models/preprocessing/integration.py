"""
Model Integration Module

Utilities for integrating preprocessing with RSSMoDModel.
Provides seamless input preparation and output processing.
"""

from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .config import PreprocessingConfig
from .processor import MultimodalProcessor, MultimodalInput, MultimodalOutput


@dataclass
class ModelInputs:
    """
    Prepared inputs for RSSMoDModel.
    
    Compatible with model.forward() signature.
    """
    input_ids: "torch.Tensor"
    attention_mask: "torch.Tensor"
    inputs_embeds: Optional["torch.Tensor"] = None
    position_ids: Optional["torch.Tensor"] = None
    labels: Optional["torch.Tensor"] = None
    
    # Multimodal embeddings (if applicable)
    image_embeds: Optional["torch.Tensor"] = None
    video_embeds: Optional["torch.Tensor"] = None
    audio_embeds: Optional["torch.Tensor"] = None
    
    def to_dict(self) -> Dict[str, "torch.Tensor"]:
        """Convert to dict for model forward."""
        result = {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
        }
        if self.inputs_embeds is not None:
            result["inputs_embeds"] = self.inputs_embeds
        if self.position_ids is not None:
            result["position_ids"] = self.position_ids
        if self.labels is not None:
            result["labels"] = self.labels
        return result


class PreprocessingPipeline:
    """
    End-to-end preprocessing pipeline for RSSMoDModel.
    
    Integrates:
    - Text tokenization
    - Image/video/audio processing
    - Multimodal embedding fusion
    - Model input preparation
    """
    
    def __init__(
        self,
        processor: Optional[MultimodalProcessor] = None,
        config: Optional[PreprocessingConfig] = None,
        device: str = "cuda",
        max_seq_length: int = 2048,
    ):
        """
        Initialize pipeline.
        
        Args:
            processor: Pre-configured processor
            config: Preprocessing configuration
            device: Target device
            max_seq_length: Maximum sequence length
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required")
        
        self.processor = processor or MultimodalProcessor(
            config=config,
            device=device,
        )
        self.device = device
        self.max_seq_length = max_seq_length
    
    def prepare_inputs(
        self,
        inputs: Union[Dict[str, Any], MultimodalInput, List],
        return_labels: bool = False,
    ) -> ModelInputs:
        """
        Prepare inputs for model forward pass.
        
        Args:
            inputs: Raw input(s)
            return_labels: Include labels for training
            
        Returns:
            ModelInputs ready for model.forward()
        """
        # Process through multimodal processor
        output = self.processor(inputs)
        
        # Ensure batch dimension
        if output.input_ids is not None and output.input_ids.dim() == 1:
            input_ids = output.input_ids.unsqueeze(0)
            attention_mask = output.attention_mask.unsqueeze(0) if output.attention_mask is not None else None
        else:
            input_ids = output.input_ids
            attention_mask = output.attention_mask
        
        # Handle case where no text
        if input_ids is None:
            # Create dummy input for non-text modalities
            input_ids = torch.zeros((1, 1), dtype=torch.long, device=self.device)
            attention_mask = torch.ones_like(input_ids)
        
        # Create labels (shifted input_ids for LM training)
        labels = None
        if return_labels and input_ids is not None:
            labels = input_ids.clone()
            # Shift happens in model, just pass same IDs
        
        return ModelInputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            image_embeds=output.pixel_values,
            video_embeds=output.video_pixel_values,
            audio_embeds=output.audio_features,
        )
    
    def __call__(
        self,
        inputs: Union[Dict[str, Any], MultimodalInput, List],
        **kwargs,
    ) -> ModelInputs:
        """Prepare inputs."""
        return self.prepare_inputs(inputs, **kwargs)


def create_preprocessing_pipeline(
    vocab_size: int = 32000,
    max_seq_length: int = 2048,
    device: str = "cuda",
    use_triton: bool = True,
) -> PreprocessingPipeline:
    """
    Create preprocessing pipeline with default settings.
    
    Args:
        vocab_size: Vocabulary size (should match model config)
        max_seq_length: Maximum sequence length
        device: Target device
        use_triton: Use Triton kernels
        
    Returns:
        Configured PreprocessingPipeline
    """
    from .config import TextConfig
    
    config = PreprocessingConfig(
        text=TextConfig(
            vocab_size=vocab_size,
            max_seq_length=max_seq_length,
        ),
        device=device,
        use_triton=use_triton,
    )
    
    return PreprocessingPipeline(config=config, device=device)
