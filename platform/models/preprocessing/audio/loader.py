"""
Audio Loader Module

Multi-codec audio loading with fallback strategies.
Supports WAV, MP3, FLAC, OGG, and more.
"""

import io
import struct
from typing import Optional, Union, Tuple, BinaryIO
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


class AudioFormat(Enum):
    """Detected audio format."""
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"
    AAC = "aac"
    M4A = "m4a"
    WMA = "wma"
    UNKNOWN = "unknown"


@dataclass
class AudioData:
    """Loaded audio data container."""
    waveform: "np.ndarray"  # Shape: (channels, samples) or (samples,)
    sample_rate: int
    duration: float  # seconds
    channels: int
    format: AudioFormat
    source: str


# Magic bytes for format detection
FORMAT_SIGNATURES = {
    b'RIFF': AudioFormat.WAV,  # RIFF....WAVE
    b'ID3': AudioFormat.MP3,
    b'\xff\xfb': AudioFormat.MP3,
    b'\xff\xfa': AudioFormat.MP3,
    b'\xff\xf3': AudioFormat.MP3,
    b'\xff\xf2': AudioFormat.MP3,
    b'fLaC': AudioFormat.FLAC,
    b'OggS': AudioFormat.OGG,
}


def detect_format(data: bytes) -> AudioFormat:
    """
    Detect audio format from magic bytes.
    
    Args:
        data: Audio bytes (at least 12 bytes)
        
    Returns:
        Detected AudioFormat
    """
    if len(data) < 12:
        return AudioFormat.UNKNOWN
    
    # Check signatures
    for sig, fmt in FORMAT_SIGNATURES.items():
        if data[:len(sig)] == sig:
            # Special case for WAV
            if sig == b'RIFF' and data[8:12] != b'WAVE':
                continue
            return fmt
    
    return AudioFormat.UNKNOWN


def parse_wav_header(data: bytes) -> Tuple[int, int, int]:
    """
    Parse WAV header for sample rate, channels, bit depth.
    
    Args:
        data: WAV file bytes
        
    Returns:
        (sample_rate, channels, bits_per_sample)
    """
    if len(data) < 44:
        return 0, 0, 0
    
    # Find fmt chunk
    i = 12
    while i < len(data) - 8:
        chunk_id = data[i:i+4]
        chunk_size = struct.unpack('<I', data[i+4:i+8])[0]
        
        if chunk_id == b'fmt ':
            fmt_data = data[i+8:i+8+chunk_size]
            if len(fmt_data) >= 16:
                channels = struct.unpack('<H', fmt_data[2:4])[0]
                sample_rate = struct.unpack('<I', fmt_data[4:8])[0]
                bits_per_sample = struct.unpack('<H', fmt_data[14:16])[0]
                return sample_rate, channels, bits_per_sample
        
        i += 8 + chunk_size
    
    return 0, 0, 0


class AudioLoader:
    """
    High-performance audio loader.
    
    Features:
    - Multi-codec support
    - URL fetching
    - Resampling
    - Mono/stereo handling
    - Duration limiting
    """
    
    __slots__ = (
        '_target_sr', '_mono', '_max_duration',
        '_timeout', '_max_size'
    )
    
    def __init__(
        self,
        target_sample_rate: Optional[int] = None,
        mono: bool = True,
        max_duration: Optional[float] = None,
        timeout: float = 30.0,
        max_size: int = 500 * 1024 * 1024,  # 500MB
    ):
        """
        Initialize loader.
        
        Args:
            target_sample_rate: Resample to this rate (None=keep original)
            mono: Convert to mono
            max_duration: Maximum duration in seconds
            timeout: URL fetch timeout
            max_size: Maximum file size in bytes
        """
        self._target_sr = target_sample_rate
        self._mono = mono
        self._max_duration = max_duration
        self._timeout = timeout
        self._max_size = max_size
    
    def load(self, source: Union[str, Path, bytes, BinaryIO]) -> AudioData:
        """
        Load audio from any source.
        
        Args:
            source: File path, URL, bytes, or file-like object
            
        Returns:
            AudioData with loaded audio
        """
        if isinstance(source, bytes):
            return self._load_bytes(source, "bytes")
        
        if hasattr(source, 'read'):
            data = source.read()
            return self._load_bytes(data, "stream")
        
        source_str = str(source)
        
        # URL
        if source_str.startswith(('http://', 'https://')):
            return self._load_url(source_str)
        
        # File path
        return self._load_file(Path(source_str))
    
    def _load_bytes(self, data: bytes, source: str) -> AudioData:
        """Load from raw bytes."""
        if len(data) > self._max_size:
            raise ValueError(f"Audio exceeds max size: {len(data)} > {self._max_size}")
        
        fmt = detect_format(data)
        
        # Try soundfile first (faster)
        if HAS_SOUNDFILE:
            try:
                waveform, sr = sf.read(io.BytesIO(data), dtype='float32')
                return self._process_waveform(waveform, sr, fmt, source)
            except Exception:
                pass
        
        # Fallback to librosa
        if HAS_LIBROSA:
            try:
                waveform, sr = librosa.load(
                    io.BytesIO(data),
                    sr=self._target_sr,
                    mono=self._mono,
                    duration=self._max_duration,
                )
                
                channels = 1 if self._mono or waveform.ndim == 1 else waveform.shape[0]
                duration = len(waveform) / sr if waveform.ndim == 1 else waveform.shape[1] / sr
                
                return AudioData(
                    waveform=waveform,
                    sample_rate=sr,
                    duration=duration,
                    channels=channels,
                    format=fmt,
                    source=source,
                )
            except Exception as e:
                raise IOError(f"Failed to decode audio: {e}")
        
        raise ImportError("soundfile or librosa required for audio loading")
    
    def _load_file(self, path: Path) -> AudioData:
        """Load from file path."""
        if not path.exists():
            raise FileNotFoundError(f"Audio not found: {path}")
        
        if path.stat().st_size > self._max_size:
            raise ValueError(f"Audio exceeds max size: {path}")
        
        # Direct file loading (more efficient)
        if HAS_SOUNDFILE:
            try:
                waveform, sr = sf.read(str(path), dtype='float32')
                fmt = AudioFormat(path.suffix.lower().lstrip('.')) if path.suffix else AudioFormat.UNKNOWN
                return self._process_waveform(waveform, sr, fmt, str(path))
            except Exception:
                pass
        
        if HAS_LIBROSA:
            try:
                waveform, sr = librosa.load(
                    str(path),
                    sr=self._target_sr,
                    mono=self._mono,
                    duration=self._max_duration,
                )
                
                fmt = AudioFormat(path.suffix.lower().lstrip('.')) if path.suffix else AudioFormat.UNKNOWN
                channels = 1 if self._mono or waveform.ndim == 1 else waveform.shape[0]
                duration = len(waveform) / sr if waveform.ndim == 1 else waveform.shape[1] / sr
                
                return AudioData(
                    waveform=waveform,
                    sample_rate=sr,
                    duration=duration,
                    channels=channels,
                    format=fmt,
                    source=str(path),
                )
            except Exception as e:
                raise IOError(f"Failed to load audio: {e}")
        
        # Fallback to bytes loading
        with open(path, 'rb') as f:
            data = f.read()
        return self._load_bytes(data, str(path))
    
    def _load_url(self, url: str) -> AudioData:
        """Load from HTTP/HTTPS URL."""
        try:
            import urllib.request
            import ssl
            
            context = ssl.create_default_context()
            req = urllib.request.Request(
                url,
                headers={'User-Agent': 'AudioLoader/1.0'}
            )
            
            with urllib.request.urlopen(
                req, timeout=self._timeout, context=context
            ) as response:
                data = response.read()
            
            return self._load_bytes(data, url)
        
        except Exception as e:
            raise IOError(f"Failed to load URL: {url}, error: {e}")
    
    def _process_waveform(
        self,
        waveform: "np.ndarray",
        sr: int,
        fmt: AudioFormat,
        source: str,
    ) -> AudioData:
        """Process loaded waveform with resampling and mono conversion."""
        # Convert to (channels, samples) or (samples,)
        if waveform.ndim == 1:
            channels = 1
            n_samples = len(waveform)
        else:
            # soundfile returns (samples, channels)
            if waveform.shape[1] <= 8:  # Assume channel dim is smaller
                waveform = waveform.T
            channels = waveform.shape[0]
            n_samples = waveform.shape[1]
        
        duration = n_samples / sr
        
        # Mono conversion
        if self._mono and channels > 1:
            waveform = waveform.mean(axis=0)
            channels = 1
        
        # Duration limiting
        if self._max_duration and duration > self._max_duration:
            max_samples = int(self._max_duration * sr)
            if waveform.ndim == 1:
                waveform = waveform[:max_samples]
            else:
                waveform = waveform[:, :max_samples]
            duration = self._max_duration
        
        # Resampling
        if self._target_sr and sr != self._target_sr:
            if HAS_LIBROSA:
                waveform = librosa.resample(
                    waveform, 
                    orig_sr=sr, 
                    target_sr=self._target_sr
                )
                sr = self._target_sr
                duration = (waveform.shape[-1] if waveform.ndim > 1 else len(waveform)) / sr
        
        return AudioData(
            waveform=waveform,
            sample_rate=sr,
            duration=duration,
            channels=channels,
            format=fmt,
            source=source,
        )
