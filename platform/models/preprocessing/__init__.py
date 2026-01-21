"""
SOTA Multimodal Preprocessing Module

High-performance preprocessing infrastructure for text, image, video, audio
modalities with Triton kernel acceleration.

Architecture:
- Text Pipeline: BPE tokenization with Unicode/emoji handling
- Image Pipeline: Resolution normalization with color space conversion
- Video Pipeline: Temporal sampling with frame extraction
- Audio Pipeline: Spectrogram computation with MFCC extraction

All pipelines leverage Triton kernels for maximum throughput.
"""

from .config import (
    PreprocessingConfig,
    TextConfig,
    ImageConfig,
    VideoConfig,
    AudioConfig,
    ModalityType,
)
from .processor import MultimodalProcessor, MultimodalInput, MultimodalOutput
from .integration import PreprocessingPipeline, ModelInputs, create_preprocessing_pipeline
from .text import BPETokenizer, Vocabulary, train_tokenizer

__all__ = [
    # Config
    "PreprocessingConfig",
    "TextConfig",
    "ImageConfig",
    "VideoConfig",
    "AudioConfig",
    "ModalityType",
    # Processor
    "MultimodalProcessor",
    "MultimodalInput",
    "MultimodalOutput",
    # Integration
    "PreprocessingPipeline",
    "ModelInputs",
    "create_preprocessing_pipeline",
    # Text
    "BPETokenizer",
    "Vocabulary",
    "train_tokenizer",
]

__version__ = "1.0.0"

