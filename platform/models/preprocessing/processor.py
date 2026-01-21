"""
Unified Multimodal Processor

Single entry point for processing text, image, video, audio inputs.
Handles the specified input schema with dynamic modality routing.
"""

from typing import Optional, Union, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .config import PreprocessingConfig, ModalityType
from .text import BPETokenizer
from .image import ImageProcessor
from .video import VideoProcessor
from .audio import AudioProcessor


@dataclass
class MultimodalInput:
    """
    Structured multimodal input.
    
    Matches the specified schema:
    {
        'role': str,
        'content': {
            'input_text': str,
            'input_image': path/url,
            'input_video': path/url,
            'input_audio': path/url,
        }
    }
    """
    role: str
    input_text: Optional[str] = None
    input_image: Optional[Union[str, Path, bytes]] = None
    input_video: Optional[Union[str, Path]] = None
    input_audio: Optional[Union[str, Path, bytes]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultimodalInput":
        """Create from dictionary."""
        content = data.get("content", {})
        return cls(
            role=data.get("role", "user"),
            input_text=content.get("input_text"),
            input_image=content.get("input_image"),
            input_video=content.get("input_video"),
            input_audio=content.get("input_audio"),
        )
    
    def get_modalities(self) -> List[ModalityType]:
        """Get list of present modalities."""
        modalities = []
        if self.input_text is not None:
            modalities.append(ModalityType.TEXT)
        if self.input_image is not None:
            modalities.append(ModalityType.IMAGE)
        if self.input_video is not None:
            modalities.append(ModalityType.VIDEO)
        if self.input_audio is not None:
            modalities.append(ModalityType.AUDIO)
        return modalities


@dataclass
class MultimodalOutput:
    """
    Unified multimodal output.
    
    Contains processed outputs for all modalities present in input.
    """
    # Text outputs
    input_ids: Optional["torch.Tensor"] = None
    attention_mask: Optional["torch.Tensor"] = None
    
    # Image outputs
    pixel_values: Optional["torch.Tensor"] = None
    
    # Video outputs
    video_pixel_values: Optional["torch.Tensor"] = None
    frame_indices: Optional[List[List[int]]] = None
    
    # Audio outputs
    audio_features: Optional["torch.Tensor"] = None
    audio_attention_mask: Optional["torch.Tensor"] = None
    
    # Metadata
    role: str = "user"
    modalities: List[ModalityType] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {"role": self.role, "modalities": [m.value for m in self.modalities]}
        
        if self.input_ids is not None:
            result["input_ids"] = self.input_ids
            result["attention_mask"] = self.attention_mask
        
        if self.pixel_values is not None:
            result["pixel_values"] = self.pixel_values
        
        if self.video_pixel_values is not None:
            result["video_pixel_values"] = self.video_pixel_values
            result["frame_indices"] = self.frame_indices
        
        if self.audio_features is not None:
            result["audio_features"] = self.audio_features
            result["audio_attention_mask"] = self.audio_attention_mask
        
        return result


class MultimodalProcessor:
    """
    Unified multimodal preprocessing processor.
    
    Features:
    - Single entry point for all modalities
    - Dynamic modality routing
    - Configurable per-modality processing
    - Batch collation
    - Memory-efficient processing
    
    Supports processing any combination:
    - Text only
    - Image only
    - Text + Image
    - Text + Image + Audio
    - All modalities
    - etc.
    """
    
    def __init__(
        self,
        config: Optional[PreprocessingConfig] = None,
        tokenizer: Optional[BPETokenizer] = None,
        device: str = "cuda",
        use_triton: bool = True,
    ):
        """
        Initialize processor.
        
        Args:
            config: Preprocessing configuration
            tokenizer: Pre-trained tokenizer (optional)
            device: Target device
            use_triton: Use Triton kernels
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required for MultimodalProcessor")
        
        self.config = config or PreprocessingConfig()
        self.device = device
        self.use_triton = use_triton
        
        # Initialize modality processors lazily
        self._tokenizer = tokenizer
        self._image_processor: Optional[ImageProcessor] = None
        self._video_processor: Optional[VideoProcessor] = None
        self._audio_processor: Optional[AudioProcessor] = None
    
    @property
    def tokenizer(self) -> BPETokenizer:
        """Get or create tokenizer."""
        if self._tokenizer is None:
            self._tokenizer = BPETokenizer()
        return self._tokenizer
    
    @property
    def image_processor(self) -> ImageProcessor:
        """Get or create image processor."""
        if self._image_processor is None:
            cfg = self.config.image
            self._image_processor = ImageProcessor(
                size=cfg.resolution,
                mean=cfg.normalize_mean,
                std=cfg.normalize_std,
                device=self.device,
                use_triton=self.use_triton,
            )
        return self._image_processor
    
    @property
    def video_processor(self) -> VideoProcessor:
        """Get or create video processor."""
        if self._video_processor is None:
            cfg = self.config.video
            from .video.sampler import SamplingStrategy
            
            strategy_map = {
                "uniform": SamplingStrategy.UNIFORM,
                "random": SamplingStrategy.RANDOM,
                "keyframe": SamplingStrategy.KEYFRAME,
            }
            
            self._video_processor = VideoProcessor(
                n_frames=cfg.max_frames,
                frame_size=cfg.frame_size,
                sampling_strategy=strategy_map.get(
                    cfg.sampling_strategy, SamplingStrategy.UNIFORM
                ),
                device=self.device,
                use_triton=self.use_triton,
            )
        return self._video_processor
    
    @property
    def audio_processor(self) -> AudioProcessor:
        """Get or create audio processor."""
        if self._audio_processor is None:
            cfg = self.config.audio
            self._audio_processor = AudioProcessor(
                sample_rate=cfg.sample_rate,
                n_fft=cfg.n_fft,
                hop_length=cfg.hop_length,
                n_mels=cfg.n_mels,
                n_mfcc=cfg.n_mfcc,
                max_duration=cfg.max_duration,
                device=self.device,
                use_triton=self.use_triton,
            )
        return self._audio_processor
    
    def process(
        self,
        inputs: Union[Dict[str, Any], MultimodalInput],
        modalities: Optional[List[ModalityType]] = None,
    ) -> MultimodalOutput:
        """
        Process multimodal input.
        
        Args:
            inputs: Input data (dict or MultimodalInput)
            modalities: Override which modalities to process
            
        Returns:
            MultimodalOutput with processed data
        """
        # Parse input
        if isinstance(inputs, dict):
            mm_input = MultimodalInput.from_dict(inputs)
        else:
            mm_input = inputs
        
        # Determine modalities to process
        if modalities is None:
            modalities = mm_input.get_modalities()
        
        output = MultimodalOutput(
            role=mm_input.role,
            modalities=modalities,
        )
        
        # Process each modality
        if ModalityType.TEXT in modalities and mm_input.input_text is not None:
            text_output = self.tokenizer.encode(
                mm_input.input_text,
                max_length=self.config.text.max_seq_length,
                padding=True,
            )
            output.input_ids = torch.tensor(
                text_output.input_ids, device=self.device
            )
            output.attention_mask = torch.tensor(
                text_output.attention_mask, device=self.device
            )
        
        if ModalityType.IMAGE in modalities and mm_input.input_image is not None:
            image_output = self.image_processor(mm_input.input_image)
            output.pixel_values = image_output.pixel_values
        
        if ModalityType.VIDEO in modalities and mm_input.input_video is not None:
            video_output = self.video_processor(mm_input.input_video)
            output.video_pixel_values = video_output.pixel_values
            output.frame_indices = video_output.frame_indices
        
        if ModalityType.AUDIO in modalities and mm_input.input_audio is not None:
            audio_output = self.audio_processor(mm_input.input_audio)
            output.audio_features = audio_output.input_features
            output.audio_attention_mask = audio_output.attention_mask
        
        return output
    
    def process_batch(
        self,
        inputs: List[Union[Dict[str, Any], MultimodalInput]],
        modalities: Optional[List[ModalityType]] = None,
    ) -> MultimodalOutput:
        """
        Process batch of multimodal inputs.
        
        Args:
            inputs: List of input data
            modalities: Override which modalities to process
            
        Returns:
            MultimodalOutput with batched tensors
        """
        outputs = [self.process(inp, modalities) for inp in inputs]
        
        # Determine batch modalities
        batch_modalities = set()
        for out in outputs:
            batch_modalities.update(out.modalities)
        
        result = MultimodalOutput(
            role="batch",
            modalities=list(batch_modalities),
        )
        
        # Collate text
        if ModalityType.TEXT in batch_modalities:
            ids = [o.input_ids for o in outputs if o.input_ids is not None]
            masks = [o.attention_mask for o in outputs if o.attention_mask is not None]
            
            if ids:
                # Pad to same length
                max_len = max(t.shape[0] for t in ids)
                padded_ids = []
                padded_masks = []
                
                for id_tensor, mask_tensor in zip(ids, masks):
                    pad_len = max_len - id_tensor.shape[0]
                    if pad_len > 0:
                        id_tensor = torch.cat([
                            id_tensor,
                            torch.zeros(pad_len, dtype=id_tensor.dtype, device=id_tensor.device)
                        ])
                        mask_tensor = torch.cat([
                            mask_tensor,
                            torch.zeros(pad_len, dtype=mask_tensor.dtype, device=mask_tensor.device)
                        ])
                    padded_ids.append(id_tensor)
                    padded_masks.append(mask_tensor)
                
                result.input_ids = torch.stack(padded_ids)
                result.attention_mask = torch.stack(padded_masks)
        
        # Collate images
        if ModalityType.IMAGE in batch_modalities:
            pixels = [o.pixel_values for o in outputs if o.pixel_values is not None]
            if pixels:
                result.pixel_values = torch.stack(pixels)
        
        # Collate videos
        if ModalityType.VIDEO in batch_modalities:
            video_pixels = [o.video_pixel_values for o in outputs if o.video_pixel_values is not None]
            if video_pixels:
                result.video_pixel_values = torch.stack(video_pixels)
                result.frame_indices = [o.frame_indices[0] for o in outputs if o.frame_indices]
        
        # Collate audio
        if ModalityType.AUDIO in batch_modalities:
            audio_feats = [o.audio_features for o in outputs if o.audio_features is not None]
            if audio_feats:
                # Pad to same length
                max_len = max(f.shape[-1] for f in audio_feats)
                padded = []
                masks = []
                
                for feat in audio_feats:
                    pad_len = max_len - feat.shape[-1]
                    if pad_len > 0:
                        feat = torch.nn.functional.pad(feat, (0, pad_len))
                        mask = torch.cat([
                            torch.ones(feat.shape[-1] - pad_len, device=feat.device),
                            torch.zeros(pad_len, device=feat.device)
                        ])
                    else:
                        mask = torch.ones(feat.shape[-1], device=feat.device)
                    padded.append(feat)
                    masks.append(mask)
                
                result.audio_features = torch.stack(padded)
                result.audio_attention_mask = torch.stack(masks)
        
        return result
    
    def __call__(
        self,
        inputs: Union[Dict[str, Any], MultimodalInput, List],
        modalities: Optional[List[ModalityType]] = None,
    ) -> MultimodalOutput:
        """
        Process input(s).
        
        Args:
            inputs: Single input or list of inputs
            modalities: Override which modalities to process
            
        Returns:
            MultimodalOutput
        """
        if isinstance(inputs, list):
            return self.process_batch(inputs, modalities)
        return self.process(inputs, modalities)
    
    def save_pretrained(self, path: Path) -> None:
        """
        Save processor to directory.
        
        Args:
            path: Output directory
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save tokenizer
        if self._tokenizer is not None:
            self._tokenizer.save(path / "tokenizer")
        
        # Save config
        import json
        config_dict = {
            "text": {
                "vocab_size": self.config.text.vocab_size,
                "max_seq_length": self.config.text.max_seq_length,
            },
            "image": {
                "resolution": self.config.image.resolution,
                "normalize_mean": self.config.image.normalize_mean,
                "normalize_std": self.config.image.normalize_std,
            },
            "video": {
                "max_frames": self.config.video.max_frames,
                "frame_size": self.config.video.frame_size,
                "sampling_strategy": self.config.video.sampling_strategy,
            },
            "audio": {
                "sample_rate": self.config.audio.sample_rate,
                "n_fft": self.config.audio.n_fft,
                "n_mels": self.config.audio.n_mels,
            },
        }
        with open(path / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, path: Path, device: str = "cuda") -> "MultimodalProcessor":
        """
        Load processor from directory.
        
        Args:
            path: Input directory
            device: Target device
            
        Returns:
            Loaded MultimodalProcessor
        """
        path = Path(path)
        
        # Load tokenizer
        tokenizer = None
        if (path / "tokenizer").exists():
            tokenizer = BPETokenizer.load(path / "tokenizer")
        
        # Load config
        import json
        with open(path / "config.json", "r") as f:
            config_dict = json.load(f)
        
        # Create config
        from .config import TextConfig, ImageConfig, VideoConfig, AudioConfig
        
        config = PreprocessingConfig(
            text=TextConfig(
                vocab_size=config_dict["text"]["vocab_size"],
                max_seq_length=config_dict["text"]["max_seq_length"],
            ),
            image=ImageConfig(
                resolution=tuple(config_dict["image"]["resolution"]),
                normalize_mean=tuple(config_dict["image"]["normalize_mean"]),
                normalize_std=tuple(config_dict["image"]["normalize_std"]),
            ),
            video=VideoConfig(
                max_frames=config_dict["video"]["max_frames"],
                frame_size=tuple(config_dict["video"]["frame_size"]),
                sampling_strategy=config_dict["video"]["sampling_strategy"],
            ),
            audio=AudioConfig(
                sample_rate=config_dict["audio"]["sample_rate"],
                n_fft=config_dict["audio"]["n_fft"],
                n_mels=config_dict["audio"]["n_mels"],
            ),
        )
        
        return cls(config=config, tokenizer=tokenizer, device=device)
