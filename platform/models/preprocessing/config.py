"""
Preprocessing Configuration Module

Dataclass configurations for multimodal preprocessing pipelines.
Supports text, image, video, audio modalities with precision control.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Literal, Union
from enum import Enum
from pathlib import Path


class ModalityType(Enum):
    """Supported input modality types."""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class PrecisionMode(Enum):
    """Compute precision modes."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"


class NormalizationType(Enum):
    """Unicode normalization forms."""
    NFC = "NFC"
    NFKC = "NFKC"
    NFD = "NFD"
    NFKD = "NFKD"


class ResizeMode(Enum):
    """Image resize interpolation modes."""
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    NEAREST = "nearest"
    LANCZOS = "lanczos"


class ColorSpace(Enum):
    """Supported color spaces."""
    RGB = "rgb"
    BGR = "bgr"
    GRAY = "gray"
    LAB = "lab"
    HSV = "hsv"


class AudioWindowType(Enum):
    """Audio window functions for STFT."""
    HANN = "hann"
    HAMMING = "hamming"
    BLACKMAN = "blackman"
    BARTLETT = "bartlett"


@dataclass
class TextConfig:
    """
    Text preprocessing configuration.
    
    Attributes:
        vocab_size: Maximum vocabulary size for BPE tokenizer
        max_seq_length: Maximum sequence length after tokenization
        pad_token: Padding token string
        unk_token: Unknown token string
        bos_token: Beginning of sequence token
        eos_token: End of sequence token
        normalize: Unicode normalization form
        lowercase: Convert to lowercase before tokenization
        handle_emoji: Enable emoji tokenization
        byte_fallback: Use byte-level fallback for unknown chars
        min_frequency: Minimum token frequency for BPE
    """
    vocab_size: int = 32000
    max_seq_length: int = 8192
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"
    bos_token: str = "<s>"
    eos_token: str = "</s>"
    normalize: NormalizationType = NormalizationType.NFC
    lowercase: bool = False
    handle_emoji: bool = True
    byte_fallback: bool = True
    min_frequency: int = 2
    special_tokens: List[str] = field(default_factory=lambda: [
        "<pad>", "<unk>", "<s>", "</s>", "<mask>"
    ])


@dataclass
class ImageConfig:
    """
    Image preprocessing configuration.
    
    Attributes:
        resolution: Target (height, width) after resize
        channels: Number of color channels
        color_space: Target color space
        resize_mode: Interpolation method
        normalize_mean: Per-channel mean for normalization
        normalize_std: Per-channel std for normalization
        preserve_aspect: Preserve aspect ratio during resize
        max_size: Maximum dimension size (for aspect-preserved resize)
    """
    resolution: Tuple[int, int] = (224, 224)
    channels: int = 3
    color_space: ColorSpace = ColorSpace.RGB
    resize_mode: ResizeMode = ResizeMode.BILINEAR
    normalize_mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    preserve_aspect: bool = False
    max_size: Optional[int] = None
    precision: PrecisionMode = PrecisionMode.FP32


@dataclass
class VideoConfig:
    """
    Video preprocessing configuration.
    
    Attributes:
        fps: Target frames per second
        max_frames: Maximum frames to extract
        frame_size: Target frame (height, width)
        sampling_strategy: Frame sampling method
        temporal_stride: Stride for uniform sampling
        extract_audio: Extract audio track
        keyframe_only: Extract only keyframes
    """
    fps: float = 30.0
    max_frames: int = 32
    frame_size: Tuple[int, int] = (224, 224)
    sampling_strategy: Literal["uniform", "random", "keyframe"] = "uniform"
    temporal_stride: int = 1
    extract_audio: bool = False
    keyframe_only: bool = False
    precision: PrecisionMode = PrecisionMode.FP32


@dataclass
class AudioConfig:
    """
    Audio preprocessing configuration.
    
    Attributes:
        sample_rate: Target sample rate
        n_fft: FFT window size
        hop_length: STFT hop length
        n_mels: Number of mel filterbanks
        n_mfcc: Number of MFCCs (if extracting)
        window_type: Window function type
        fmin: Minimum frequency for mel scale
        fmax: Maximum frequency for mel scale
        max_duration: Maximum audio duration (seconds)
        normalize: Normalize audio amplitude
    """
    sample_rate: int = 16000
    n_fft: int = 400
    hop_length: int = 160
    n_mels: int = 80
    n_mfcc: int = 13
    window_type: AudioWindowType = AudioWindowType.HANN
    fmin: float = 0.0
    fmax: Optional[float] = 8000.0
    max_duration: Optional[float] = 30.0
    normalize: bool = True
    precision: PrecisionMode = PrecisionMode.FP32


@dataclass
class PreprocessingConfig:
    """
    Master preprocessing configuration.
    
    Unified configuration for all modality preprocessing pipelines.
    Supports dynamic modality selection and batch processing.
    """
    text: TextConfig = field(default_factory=TextConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    
    # Global settings
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    use_triton: bool = True
    device: str = "cuda"
    
    # Memory optimization
    gradient_checkpointing: bool = False
    mixed_precision: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.num_workers < 0:
            raise ValueError("num_workers must be >= 0")
        if self.text.vocab_size < 256:
            raise ValueError("vocab_size must be >= 256")
        if self.text.max_seq_length < 1:
            raise ValueError("max_seq_length must be >= 1")
    
    @classmethod
    def for_llm(cls) -> "PreprocessingConfig":
        """Optimized config for LLM text processing."""
        return cls(
            text=TextConfig(
                vocab_size=128000,
                max_seq_length=131072,
            ),
            batch_size=8,
            mixed_precision=True,
        )
    
    @classmethod
    def for_vision(cls) -> "PreprocessingConfig":
        """Optimized config for vision models."""
        return cls(
            image=ImageConfig(
                resolution=(384, 384),
                preserve_aspect=True,
            ),
            batch_size=64,
            mixed_precision=True,
        )
    
    @classmethod
    def for_multimodal(cls) -> "PreprocessingConfig":
        """Balanced config for multimodal processing."""
        return cls(
            text=TextConfig(vocab_size=32000, max_seq_length=2048),
            image=ImageConfig(resolution=(224, 224)),
            video=VideoConfig(max_frames=16),
            audio=AudioConfig(max_duration=10.0),
            batch_size=16,
        )
