"""
Video Processor Module

Unified video preprocessing pipeline with:
- Frame extraction
- Temporal sampling
- Spatial processing
- Audio extraction
"""

from typing import Optional, Union, List, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from .extractor import FrameExtractor, VideoMetadata
from .sampler import TemporalSampler, SamplingStrategy

# Import image processor for frame processing
import sys
from pathlib import Path as PathLib
sys.path.insert(0, str(PathLib(__file__).parent.parent))
from image.transforms import ImageTransforms, InterpolationMode


@dataclass
class VideoProcessorOutput:
    """Output from video processor."""
    pixel_values: "torch.Tensor"  # (n_frames, C, H, W) or (batch, n_frames, C, H, W)
    frame_indices: List[List[int]]
    metadata: List[VideoMetadata]


class VideoProcessor:
    """
    Unified video preprocessing pipeline.
    
    Features:
    - Any format support
    - Temporal sampling strategies
    - Spatial transforms
    - Audio extraction
    - Batch processing
    """
    
    def __init__(
        self,
        n_frames: int = 16,
        frame_size: Tuple[int, int] = (224, 224),
        sampling_strategy: SamplingStrategy = SamplingStrategy.UNIFORM,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        do_normalize: bool = True,
        do_rescale: bool = True,
        rescale_factor: float = 1.0 / 255.0,
        extract_audio: bool = False,
        device: str = "cuda",
        use_triton: bool = True,
    ):
        """
        Initialize processor.
        
        Args:
            n_frames: Number of frames to extract
            frame_size: Target (height, width)
            sampling_strategy: Frame sampling strategy
            mean: Normalization mean
            std: Normalization std
            do_normalize: Apply normalization
            do_rescale: Rescale pixel values
            rescale_factor: Rescale multiplier
            extract_audio: Also extract audio track
            device: Target device
            use_triton: Use Triton kernels
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required for VideoProcessor")
        
        self.n_frames = n_frames
        self.frame_size = frame_size
        self.mean = mean
        self.std = std
        self.do_normalize = do_normalize
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.extract_audio = extract_audio
        self.device = device
        
        self.extractor = FrameExtractor()
        self.sampler = TemporalSampler(
            n_frames=n_frames,
            strategy=sampling_strategy,
        )
        self.transforms = ImageTransforms(device=device, use_triton=use_triton)
    
    def _frames_to_tensor(self, frames: "np.ndarray") -> "torch.Tensor":
        """Convert numpy frames to tensor."""
        # frames: (n_frames, H, W, C) -> (n_frames, C, H, W)
        tensor = torch.from_numpy(frames).float()
        tensor = tensor.permute(0, 3, 1, 2)
        return tensor.to(self.device)
    
    def _process_frames(self, frames: "torch.Tensor") -> "torch.Tensor":
        """Apply spatial transforms to frames."""
        n_frames = frames.shape[0]
        processed = []
        
        for i in range(n_frames):
            frame = frames[i]
            
            # Rescale
            if self.do_rescale:
                frame = frame * self.rescale_factor
            
            # Resize
            frame = self.transforms.resize(
                frame,
                self.frame_size,
                mode=InterpolationMode.BILINEAR,
            )
            
            # Normalize
            if self.do_normalize:
                frame = self.transforms.normalize(frame, self.mean, self.std)
            
            processed.append(frame)
        
        return torch.stack(processed, dim=0)
    
    def process(
        self,
        video: Union[str, Path],
        return_tensors: bool = True,
        keyframe_indices: Optional[List[int]] = None,
    ) -> Union[VideoProcessorOutput, Dict[str, Any]]:
        """
        Process single video.
        
        Args:
            video: Video file path or URL
            return_tensors: Return tensor output
            keyframe_indices: Pre-computed keyframe indices
            
        Returns:
            VideoProcessorOutput or dict
        """
        # Get metadata
        metadata = self.extractor.get_metadata(video)
        
        # Detect keyframes if not provided and using keyframe strategy
        if keyframe_indices is None and self.sampler.strategy == SamplingStrategy.KEYFRAME:
            keyframe_indices = self.extractor.detect_keyframes(video)
        
        # Sample frame indices
        frame_indices = self.sampler.sample(
            metadata.total_frames,
            keyframe_indices=keyframe_indices,
        )
        
        # Extract frames
        extracted = self.extractor.extract_frames(video, indices=frame_indices)
        
        # Convert to tensor and process
        frames = self._frames_to_tensor(extracted.frames)
        frames = self._process_frames(frames)
        
        if return_tensors:
            return VideoProcessorOutput(
                pixel_values=frames,
                frame_indices=[extracted.indices],
                metadata=[metadata],
            )
        
        return {
            "pixel_values": frames,
            "frame_indices": [extracted.indices],
            "metadata": [metadata],
        }
    
    def process_batch(
        self,
        videos: List[Union[str, Path]],
        return_tensors: bool = True,
    ) -> Union[VideoProcessorOutput, Dict[str, Any]]:
        """
        Process batch of videos.
        
        Args:
            videos: List of video paths/URLs
            return_tensors: Return tensor output
            
        Returns:
            VideoProcessorOutput or dict with batched tensors
        """
        outputs = [self.process(v, return_tensors=True) for v in videos]
        
        # Stack tensors
        pixel_values = torch.stack([o.pixel_values for o in outputs], dim=0)
        frame_indices = [o.frame_indices[0] for o in outputs]
        metadata = [o.metadata[0] for o in outputs]
        
        if return_tensors:
            return VideoProcessorOutput(
                pixel_values=pixel_values,
                frame_indices=frame_indices,
                metadata=metadata,
            )
        
        return {
            "pixel_values": pixel_values,
            "frame_indices": frame_indices,
            "metadata": metadata,
        }
    
    def __call__(
        self,
        videos: Union[Any, List[Any]],
        return_tensors: bool = True,
    ) -> Union[VideoProcessorOutput, Dict[str, Any]]:
        """
        Process video(s).
        
        Args:
            videos: Single video or list of videos
            return_tensors: Return tensor output
            
        Returns:
            VideoProcessorOutput or dict
        """
        if isinstance(videos, list):
            return self.process_batch(videos, return_tensors)
        return self.process(videos, return_tensors)
