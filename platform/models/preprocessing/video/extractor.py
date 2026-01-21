"""
Frame Extractor Module

High-performance video frame extraction with:
- Multi-format support
- Efficient decoding
- Keyframe detection
- Memory-efficient streaming
"""

import io
from typing import Optional, Union, List, Tuple, Iterator, BinaryIO
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class VideoFormat(Enum):
    """Detected video format."""
    MP4 = "mp4"
    AVI = "avi"
    MKV = "mkv"
    MOV = "mov"
    WEBM = "webm"
    FLV = "flv"
    WMV = "wmv"
    UNKNOWN = "unknown"


@dataclass
class VideoMetadata:
    """Video metadata container."""
    width: int
    height: int
    fps: float
    total_frames: int
    duration: float  # seconds
    codec: str
    format: VideoFormat


@dataclass
class ExtractedFrames:
    """Extracted frames container."""
    frames: "np.ndarray"  # (n_frames, height, width, channels)
    indices: List[int]
    metadata: VideoMetadata


class FrameExtractor:
    """
    High-performance video frame extractor.
    
    Features:
    - Any format support via OpenCV
    - Efficient frame seeking
    - Keyframe detection
    - Memory-efficient streaming
    - URL support
    """
    
    __slots__ = (
        '_max_size', '_timeout', '_chunk_size'
    )
    
    def __init__(
        self,
        max_size: int = 2 * 1024 * 1024 * 1024,  # 2GB
        timeout: float = 60.0,
        chunk_size: int = 32,  # Frames to process at once
    ):
        """
        Initialize extractor.
        
        Args:
            max_size: Maximum file size
            timeout: URL fetch timeout
            chunk_size: Frames to decode in each batch
        """
        if not HAS_CV2:
            raise ImportError("OpenCV required for FrameExtractor")
        
        self._max_size = max_size
        self._timeout = timeout
        self._chunk_size = chunk_size
    
    def get_metadata(self, source: Union[str, Path]) -> VideoMetadata:
        """
        Get video metadata without decoding frames.
        
        Args:
            source: Video file path or URL
            
        Returns:
            VideoMetadata
        """
        source_str = str(source)
        cap = cv2.VideoCapture(source_str)
        
        if not cap.isOpened():
            raise IOError(f"Failed to open video: {source_str}")
        
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Get codec (fourcc)
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            
            # Calculate duration
            duration = total_frames / fps if fps > 0 else 0.0
            
            # Detect format from extension
            path = Path(source_str)
            ext = path.suffix.lower().lstrip('.')
            try:
                fmt = VideoFormat(ext)
            except ValueError:
                fmt = VideoFormat.UNKNOWN
            
            return VideoMetadata(
                width=width,
                height=height,
                fps=fps,
                total_frames=total_frames,
                duration=duration,
                codec=codec,
                format=fmt,
            )
        finally:
            cap.release()
    
    def extract_frames(
        self,
        source: Union[str, Path],
        indices: Optional[List[int]] = None,
        n_frames: Optional[int] = None,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
    ) -> ExtractedFrames:
        """
        Extract frames from video.
        
        Args:
            source: Video file path or URL
            indices: Specific frame indices to extract
            n_frames: Number of frames (uniform sampling if indices not provided)
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            ExtractedFrames with frames array
        """
        source_str = str(source)
        metadata = self.get_metadata(source_str)
        
        # Calculate frame range
        start_frame = int(start_time * metadata.fps)
        end_frame = int(end_time * metadata.fps) if end_time else metadata.total_frames
        end_frame = min(end_frame, metadata.total_frames)
        
        # Determine indices to extract
        if indices is not None:
            frame_indices = [i for i in indices if start_frame <= i < end_frame]
        elif n_frames is not None:
            frame_range = end_frame - start_frame
            step = frame_range / n_frames
            frame_indices = [start_frame + int(i * step) for i in range(n_frames)]
        else:
            frame_indices = list(range(start_frame, end_frame))
        
        # Extract frames
        cap = cv2.VideoCapture(source_str)
        if not cap.isOpened():
            raise IOError(f"Failed to open video: {source_str}")
        
        try:
            frames = []
            extracted_indices = []
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if ret:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                    extracted_indices.append(idx)
            
            if not frames:
                # Return empty array with correct shape
                frames_array = np.zeros(
                    (0, metadata.height, metadata.width, 3),
                    dtype=np.uint8
                )
            else:
                frames_array = np.stack(frames, axis=0)
            
            return ExtractedFrames(
                frames=frames_array,
                indices=extracted_indices,
                metadata=metadata,
            )
        finally:
            cap.release()
    
    def extract_frames_streaming(
        self,
        source: Union[str, Path],
        indices: Optional[List[int]] = None,
        chunk_size: Optional[int] = None,
    ) -> Iterator[Tuple[List[int], "np.ndarray"]]:
        """
        Stream frames in chunks for memory efficiency.
        
        Args:
            source: Video file path or URL
            indices: Frame indices to extract (all if None)
            chunk_size: Override default chunk size
            
        Yields:
            (indices, frames) tuples for each chunk
        """
        source_str = str(source)
        metadata = self.get_metadata(source_str)
        chunk_size = chunk_size or self._chunk_size
        
        if indices is None:
            indices = list(range(metadata.total_frames))
        
        cap = cv2.VideoCapture(source_str)
        if not cap.isOpened():
            raise IOError(f"Failed to open video: {source_str}")
        
        try:
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i:i + chunk_size]
                frames = []
                extracted_indices = []
                
                for idx in chunk_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    
                    if ret:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(frame)
                        extracted_indices.append(idx)
                
                if frames:
                    yield extracted_indices, np.stack(frames, axis=0)
        finally:
            cap.release()
    
    def detect_keyframes(
        self,
        source: Union[str, Path],
        threshold: float = 30.0,
    ) -> List[int]:
        """
        Detect keyframes using frame difference.
        
        Args:
            source: Video file path or URL
            threshold: Difference threshold for keyframe detection
            
        Returns:
            List of keyframe indices
        """
        source_str = str(source)
        cap = cv2.VideoCapture(source_str)
        
        if not cap.isOpened():
            raise IOError(f"Failed to open video: {source_str}")
        
        try:
            keyframes = [0]  # First frame is always a keyframe
            prev_frame = None
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale for comparison
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_frame is not None:
                    # Compute frame difference
                    diff = cv2.absdiff(prev_frame, gray)
                    mean_diff = np.mean(diff)
                    
                    if mean_diff > threshold:
                        keyframes.append(frame_idx)
                
                prev_frame = gray
                frame_idx += 1
            
            return keyframes
        finally:
            cap.release()
    
    def extract_audio(
        self,
        source: Union[str, Path],
        output_path: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Extract audio track from video.
        
        Args:
            source: Video file path
            output_path: Output audio file path
            
        Returns:
            Path to extracted audio file, or None if no audio
        """
        # This requires ffmpeg, so we check if it's available
        try:
            import subprocess
            
            source_str = str(source)
            if output_path is None:
                output_path = Path(source_str).with_suffix('.wav')
            
            result = subprocess.run([
                'ffmpeg', '-i', source_str,
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', '16000', '-ac', '1',
                str(output_path), '-y'
            ], capture_output=True)
            
            if result.returncode == 0 and output_path.exists():
                return output_path
            return None
        except Exception:
            return None
