"""
Image Loader Module

Extension-agnostic image loading with fallback strategies.
Supports local paths, HTTP/HTTPS URLs, and base64 data.
"""

import io
import base64
from typing import Optional, Union, Tuple, BinaryIO
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import struct

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class ImageFormat(Enum):
    """Detected image format."""
    JPEG = "jpeg"
    PNG = "png"
    WEBP = "webp"
    GIF = "gif"
    BMP = "bmp"
    TIFF = "tiff"
    ICO = "ico"
    UNKNOWN = "unknown"


@dataclass
class ImageData:
    """Loaded image data container."""
    data: bytes
    width: int
    height: int
    channels: int
    format: ImageFormat
    source: str  # Original source path/URL


# Magic bytes for format detection
FORMAT_SIGNATURES = {
    b'\x89PNG\r\n\x1a\n': ImageFormat.PNG,
    b'\xff\xd8\xff': ImageFormat.JPEG,
    b'RIFF': ImageFormat.WEBP,  # RIFF....WEBP
    b'GIF87a': ImageFormat.GIF,
    b'GIF89a': ImageFormat.GIF,
    b'BM': ImageFormat.BMP,
    b'II*\x00': ImageFormat.TIFF,  # Little-endian
    b'MM\x00*': ImageFormat.TIFF,  # Big-endian
    b'\x00\x00\x01\x00': ImageFormat.ICO,
}


def detect_format(data: bytes) -> ImageFormat:
    """
    Detect image format from magic bytes.
    
    Args:
        data: Image bytes (at least 12 bytes)
        
    Returns:
        Detected ImageFormat
    """
    if len(data) < 12:
        return ImageFormat.UNKNOWN
    
    # Check signatures
    for sig, fmt in FORMAT_SIGNATURES.items():
        if data[:len(sig)] == sig:
            # Special case for WEBP
            if sig == b'RIFF' and data[8:12] != b'WEBP':
                continue
            return fmt
    
    return ImageFormat.UNKNOWN


def get_image_dimensions(data: bytes, fmt: ImageFormat) -> Tuple[int, int]:
    """
    Extract image dimensions without full decode.
    
    Args:
        data: Image bytes
        fmt: Image format
        
    Returns:
        (width, height) tuple
    """
    if fmt == ImageFormat.PNG:
        # PNG: width at bytes 16-20, height at 20-24
        if len(data) >= 24:
            width = struct.unpack('>I', data[16:20])[0]
            height = struct.unpack('>I', data[20:24])[0]
            return width, height
    
    elif fmt == ImageFormat.JPEG:
        # JPEG: scan for SOF markers
        i = 2
        while i < len(data) - 9:
            if data[i] != 0xFF:
                i += 1
                continue
            
            marker = data[i + 1]
            
            # SOF markers (0xC0-0xCF except 0xC4, 0xC8, 0xCC)
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                          0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                height = struct.unpack('>H', data[i + 5:i + 7])[0]
                width = struct.unpack('>H', data[i + 7:i + 9])[0]
                return width, height
            
            # Skip to next marker
            if marker == 0xD8 or marker == 0xD9:  # SOI, EOI
                i += 2
            else:
                length = struct.unpack('>H', data[i + 2:i + 4])[0]
                i += 2 + length
    
    elif fmt == ImageFormat.GIF:
        # GIF: width at bytes 6-8, height at 8-10
        if len(data) >= 10:
            width = struct.unpack('<H', data[6:8])[0]
            height = struct.unpack('<H', data[8:10])[0]
            return width, height
    
    elif fmt == ImageFormat.BMP:
        # BMP: width at bytes 18-22, height at 22-26
        if len(data) >= 26:
            width = struct.unpack('<I', data[18:22])[0]
            height = abs(struct.unpack('<i', data[22:26])[0])
            return width, height
    
    elif fmt == ImageFormat.WEBP:
        # WEBP: more complex, skip for now
        pass
    
    return 0, 0


class ImageLoader:
    """
    High-performance image loader.
    
    Features:
    - Extension-agnostic loading
    - URL fetching (HTTP/HTTPS)
    - Base64 decoding
    - Format auto-detection
    - Fast dimension extraction
    - PIL/OpenCV fallback
    """
    
    __slots__ = ('_use_pil', '_use_cv2', '_timeout', '_max_size')
    
    def __init__(
        self,
        prefer_pil: bool = True,
        timeout: float = 30.0,
        max_size: int = 100 * 1024 * 1024,  # 100MB
    ):
        """
        Initialize loader.
        
        Args:
            prefer_pil: Prefer PIL over OpenCV
            timeout: URL fetch timeout
            max_size: Maximum file size in bytes
        """
        self._use_pil = HAS_PIL and prefer_pil
        self._use_cv2 = HAS_CV2 and not prefer_pil
        self._timeout = timeout
        self._max_size = max_size
    
    def load(self, source: Union[str, Path, bytes, BinaryIO]) -> ImageData:
        """
        Load image from any source.
        
        Args:
            source: File path, URL, bytes, or file-like object
            
        Returns:
            ImageData with loaded image
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
        
        # Base64
        if source_str.startswith('data:image/'):
            return self._load_base64(source_str)
        
        # File path
        return self._load_file(Path(source_str))
    
    def _load_bytes(self, data: bytes, source: str) -> ImageData:
        """Load from raw bytes."""
        if len(data) > self._max_size:
            raise ValueError(f"Image exceeds max size: {len(data)} > {self._max_size}")
        
        fmt = detect_format(data)
        width, height = get_image_dimensions(data, fmt)
        
        # If dimensions not extracted, decode fully
        if width == 0 or height == 0:
            width, height, channels = self._decode_dimensions(data)
        else:
            channels = 3  # Assume RGB
        
        return ImageData(
            data=data,
            width=width,
            height=height,
            channels=channels,
            format=fmt,
            source=source,
        )
    
    def _load_file(self, path: Path) -> ImageData:
        """Load from file path."""
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        
        if path.stat().st_size > self._max_size:
            raise ValueError(f"Image exceeds max size: {path}")
        
        with open(path, 'rb') as f:
            data = f.read()
        
        return self._load_bytes(data, str(path))
    
    def _load_url(self, url: str) -> ImageData:
        """Load from HTTP/HTTPS URL."""
        try:
            import urllib.request
            import ssl
            
            context = ssl.create_default_context()
            req = urllib.request.Request(
                url,
                headers={'User-Agent': 'ImageLoader/1.0'}
            )
            
            with urllib.request.urlopen(
                req, timeout=self._timeout, context=context
            ) as response:
                data = response.read()
            
            return self._load_bytes(data, url)
        
        except Exception as e:
            raise IOError(f"Failed to load URL: {url}, error: {e}")
    
    def _load_base64(self, data_uri: str) -> ImageData:
        """Load from base64 data URI."""
        # Format: data:image/png;base64,<data>
        try:
            header, encoded = data_uri.split(',', 1)
            data = base64.b64decode(encoded)
            return self._load_bytes(data, "base64")
        except Exception as e:
            raise ValueError(f"Invalid base64 data URI: {e}")
    
    def _decode_dimensions(self, data: bytes) -> Tuple[int, int, int]:
        """Fully decode image to get dimensions."""
        if self._use_pil and HAS_PIL:
            try:
                img = Image.open(io.BytesIO(data))
                width, height = img.size
                channels = len(img.getbands())
                return width, height, channels
            except Exception:
                pass
        
        if HAS_CV2:
            try:
                arr = np.frombuffer(data, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    height, width = img.shape[:2]
                    channels = img.shape[2] if len(img.shape) > 2 else 1
                    return width, height, channels
            except Exception:
                pass
        
        return 0, 0, 3
    
    def to_numpy(self, image_data: ImageData) -> "np.ndarray":
        """
        Convert ImageData to numpy array.
        
        Args:
            image_data: Loaded image data
            
        Returns:
            Numpy array (H, W, C)
        """
        if not HAS_CV2:
            raise ImportError("OpenCV required for numpy conversion")
        
        arr = np.frombuffer(image_data.data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            raise ValueError("Failed to decode image")
        
        return img
    
    def to_pil(self, image_data: ImageData) -> "Image.Image":
        """
        Convert ImageData to PIL Image.
        
        Args:
            image_data: Loaded image data
            
        Returns:
            PIL Image
        """
        if not HAS_PIL:
            raise ImportError("PIL required for PIL conversion")
        
        return Image.open(io.BytesIO(image_data.data))
