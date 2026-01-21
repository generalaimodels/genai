"""
Temporal Sampler Module

Temporal sampling strategies for video frame selection.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import math

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class SamplingStrategy(Enum):
    """Frame sampling strategies."""
    UNIFORM = "uniform"       # Evenly spaced frames
    RANDOM = "random"         # Random frame selection
    KEYFRAME = "keyframe"     # Only keyframes
    DENSE = "dense"           # Dense sampling around events
    STRIDE = "stride"         # Fixed stride sampling


@dataclass
class SamplingConfig:
    """Sampling configuration."""
    strategy: SamplingStrategy
    n_frames: int
    stride: int = 1
    jitter: float = 0.0  # Random offset for uniform sampling


class TemporalSampler:
    """
    Temporal sampling for video frames.
    
    Features:
    - Multiple sampling strategies
    - Configurable frame count
    - Temporal jitter for augmentation
    - Efficient index computation
    """
    
    def __init__(
        self,
        n_frames: int = 16,
        strategy: SamplingStrategy = SamplingStrategy.UNIFORM,
        stride: int = 1,
        jitter: float = 0.0,
    ):
        """
        Initialize sampler.
        
        Args:
            n_frames: Number of frames to sample
            strategy: Sampling strategy
            stride: Frame stride for stride sampling
            jitter: Random jitter for uniform sampling
        """
        self.n_frames = n_frames
        self.strategy = strategy
        self.stride = stride
        self.jitter = jitter
    
    def sample(
        self,
        total_frames: int,
        keyframe_indices: Optional[List[int]] = None,
    ) -> List[int]:
        """
        Sample frame indices.
        
        Args:
            total_frames: Total number of frames in video
            keyframe_indices: Optional list of keyframe indices
            
        Returns:
            List of sampled frame indices
        """
        if total_frames <= 0:
            return []
        
        if total_frames <= self.n_frames:
            # Return all frames if video is shorter
            return list(range(total_frames))
        
        if self.strategy == SamplingStrategy.UNIFORM:
            return self._uniform_sample(total_frames)
        elif self.strategy == SamplingStrategy.RANDOM:
            return self._random_sample(total_frames)
        elif self.strategy == SamplingStrategy.KEYFRAME:
            return self._keyframe_sample(total_frames, keyframe_indices)
        elif self.strategy == SamplingStrategy.STRIDE:
            return self._stride_sample(total_frames)
        elif self.strategy == SamplingStrategy.DENSE:
            return self._dense_sample(total_frames)
        else:
            return self._uniform_sample(total_frames)
    
    def _uniform_sample(self, total_frames: int) -> List[int]:
        """Uniform temporal sampling."""
        # Calculate evenly spaced indices
        indices = []
        step = total_frames / self.n_frames
        
        for i in range(self.n_frames):
            idx = int(i * step)
            
            # Apply jitter if configured
            if self.jitter > 0:
                import random
                jitter_range = int(step * self.jitter)
                jitter = random.randint(-jitter_range, jitter_range)
                idx = max(0, min(total_frames - 1, idx + jitter))
            
            indices.append(idx)
        
        return indices
    
    def _random_sample(self, total_frames: int) -> List[int]:
        """Random frame sampling."""
        import random
        indices = sorted(random.sample(range(total_frames), self.n_frames))
        return indices
    
    def _keyframe_sample(
        self,
        total_frames: int,
        keyframe_indices: Optional[List[int]],
    ) -> List[int]:
        """Keyframe-based sampling."""
        if not keyframe_indices:
            # Fallback to uniform if no keyframes
            return self._uniform_sample(total_frames)
        
        # Sample from keyframes
        if len(keyframe_indices) <= self.n_frames:
            indices = keyframe_indices.copy()
            # Fill remaining with uniform sampling
            remaining = self.n_frames - len(indices)
            if remaining > 0:
                extra = self._uniform_sample(total_frames)
                for idx in extra:
                    if idx not in indices and len(indices) < self.n_frames:
                        indices.append(idx)
            return sorted(indices)
        
        # Uniformly sample from keyframes
        step = len(keyframe_indices) / self.n_frames
        indices = [keyframe_indices[int(i * step)] for i in range(self.n_frames)]
        return indices
    
    def _stride_sample(self, total_frames: int) -> List[int]:
        """Fixed stride sampling."""
        indices = []
        idx = 0
        while len(indices) < self.n_frames and idx < total_frames:
            indices.append(idx)
            idx += self.stride
        
        # If not enough frames, add remaining from end
        while len(indices) < self.n_frames:
            indices.append(total_frames - 1)
        
        return indices
    
    def _dense_sample(self, total_frames: int) -> List[int]:
        """Dense sampling (consecutive frames)."""
        # Start from middle of video
        start = max(0, (total_frames - self.n_frames) // 2)
        indices = list(range(start, min(start + self.n_frames, total_frames)))
        
        # Pad if needed
        while len(indices) < self.n_frames:
            indices.append(indices[-1])
        
        return indices
    
    def sample_with_context(
        self,
        total_frames: int,
        context_before: int = 1,
        context_after: int = 1,
    ) -> List[Tuple[int, List[int]]]:
        """
        Sample frames with temporal context.
        
        Args:
            total_frames: Total frame count
            context_before: Context frames before each sample
            context_after: Context frames after each sample
            
        Returns:
            List of (center_idx, [context_indices]) tuples
        """
        center_indices = self.sample(total_frames)
        results = []
        
        for center in center_indices:
            context = []
            
            # Before context
            for i in range(context_before, 0, -1):
                ctx_idx = max(0, center - i)
                context.append(ctx_idx)
            
            # Center
            context.append(center)
            
            # After context
            for i in range(1, context_after + 1):
                ctx_idx = min(total_frames - 1, center + i)
                context.append(ctx_idx)
            
            results.append((center, context))
        
        return results
