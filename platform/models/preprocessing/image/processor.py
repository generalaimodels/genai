"""
Image Processor Module

Unified image preprocessing pipeline with:
- Extension-agnostic loading
- Resolution normalization
- Color space conversion
- Augmentation transforms
"""

from typing import Tuple, Optional, Union, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .loader import ImageLoader, ImageData
from .transforms import ImageTransforms, InterpolationMode, ColorSpace


@dataclass
class ImageProcessorOutput:
    """Output from image processor."""
    pixel_values: "torch.Tensor"  # (C, H, W) or (B, C, H, W)
    original_sizes: List[Tuple[int, int]]
    transformed_size: Tuple[int, int]


class ImageProcessor:
    """
    Unified image preprocessing pipeline.
    
    Features:
    - Any extension support
    - URL/path/bytes input
    - Triton-accelerated transforms
    - Batch processing
    - Configurable augmentation
    """
    
    def __init__(
        self,
        size: Tuple[int, int] = (224, 224),
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        color_space: ColorSpace = ColorSpace.RGB,
        do_normalize: bool = True,
        do_rescale: bool = True,
        rescale_factor: float = 1.0 / 255.0,
        device: str = "cuda",
        use_triton: bool = True,
    ):
        """
        Initialize processor.
        
        Args:
            size: Target (height, width)
            mean: Normalization mean
            std: Normalization std
            interpolation: Resize interpolation
            color_space: Target color space
            do_normalize: Apply mean/std normalization
            do_rescale: Rescale pixel values
            rescale_factor: Rescale multiplier
            device: Target device
            use_triton: Use Triton kernels
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required for ImageProcessor")
        
        self.size = size
        self.mean = mean
        self.std = std
        self.interpolation = interpolation
        self.color_space = color_space
        self.do_normalize = do_normalize
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.device = device
        
        self.loader = ImageLoader()
        self.transforms = ImageTransforms(device=device, use_triton=use_triton)
    
    def _to_tensor(self, image_data: ImageData) -> "torch.Tensor":
        """Convert ImageData to tensor."""
        try:
            # Try PIL
            pil_image = self.loader.to_pil(image_data)
            
            # Convert to RGB if needed
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert to tensor (C, H, W)
            import numpy as np
            arr = np.array(pil_image, dtype=np.float32)
            tensor = torch.from_numpy(arr).permute(2, 0, 1)
            
        except Exception:
            # Fallback to OpenCV
            arr = self.loader.to_numpy(image_data)
            
            # BGR to RGB
            if len(arr.shape) == 3 and arr.shape[2] == 3:
                arr = arr[:, :, ::-1].copy()
            elif len(arr.shape) == 2:
                arr = np.stack([arr] * 3, axis=-1)
            
            tensor = torch.from_numpy(arr.astype(np.float32)).permute(2, 0, 1)
        
        return tensor
    
    def process(
        self,
        image: Union[str, Path, bytes, ImageData],
        return_tensors: bool = True,
    ) -> Union[ImageProcessorOutput, Dict[str, Any]]:
        """
        Process single image.
        
        Args:
            image: Input image (path, URL, bytes, ImageData)
            return_tensors: Return tensor output
            
        Returns:
            ImageProcessorOutput or dict
        """
        # Load if needed
        if isinstance(image, ImageData):
            image_data = image
        else:
            image_data = self.loader.load(image)
        
        original_size = (image_data.height, image_data.width)
        
        # Convert to tensor
        tensor = self._to_tensor(image_data)
        
        # Move to device
        tensor = tensor.to(self.device)
        
        # Rescale
        if self.do_rescale:
            tensor = tensor * self.rescale_factor
        
        # Resize
        tensor = self.transforms.resize(
            tensor, 
            self.size, 
            mode=self.interpolation
        )
        
        # Color space conversion
        if self.color_space == ColorSpace.GRAY:
            tensor = self.transforms.to_grayscale(tensor)
        elif self.color_space == ColorSpace.HSV:
            tensor = self.transforms.to_hsv(tensor)
        
        # Normalize
        if self.do_normalize:
            # Adjust mean/std for grayscale
            if self.color_space == ColorSpace.GRAY:
                mean = (sum(self.mean) / len(self.mean),)
                std = (sum(self.std) / len(self.std),)
            else:
                mean = self.mean
                std = self.std
            
            tensor = self.transforms.normalize(tensor, mean, std)
        
        if return_tensors:
            return ImageProcessorOutput(
                pixel_values=tensor,
                original_sizes=[original_size],
                transformed_size=self.size,
            )
        
        return {
            "pixel_values": tensor,
            "original_sizes": [original_size],
            "transformed_size": self.size,
        }
    
    def process_batch(
        self,
        images: List[Union[str, Path, bytes, ImageData]],
        return_tensors: bool = True,
    ) -> Union[ImageProcessorOutput, Dict[str, Any]]:
        """
        Process batch of images.
        
        Args:
            images: List of input images
            return_tensors: Return tensor output
            
        Returns:
            ImageProcessorOutput or dict with batched tensors
        """
        outputs = [self.process(img, return_tensors=True) for img in images]
        
        # Stack tensors
        pixel_values = torch.stack([o.pixel_values for o in outputs], dim=0)
        original_sizes = [o.original_sizes[0] for o in outputs]
        
        if return_tensors:
            return ImageProcessorOutput(
                pixel_values=pixel_values,
                original_sizes=original_sizes,
                transformed_size=self.size,
            )
        
        return {
            "pixel_values": pixel_values,
            "original_sizes": original_sizes,
            "transformed_size": self.size,
        }
    
    def __call__(
        self,
        images: Union[Any, List[Any]],
        return_tensors: bool = True,
    ) -> Union[ImageProcessorOutput, Dict[str, Any]]:
        """
        Process image(s).
        
        Args:
            images: Single image or list of images
            return_tensors: Return tensor output
            
        Returns:
            ImageProcessorOutput or dict
        """
        if isinstance(images, list):
            return self.process_batch(images, return_tensors)
        return self.process(images, return_tensors)
