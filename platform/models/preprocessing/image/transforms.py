"""
Image Transforms Module

GPU-accelerated image transformations with Triton kernels.
Supports resize, normalize, color conversion, augmentation.
"""

from typing import Tuple, Optional, Union, List
from dataclasses import dataclass
from enum import Enum
import math

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


class InterpolationMode(Enum):
    """Interpolation modes for resize."""
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"


class ColorSpace(Enum):
    """Color space formats."""
    RGB = "rgb"
    BGR = "bgr"
    GRAY = "gray"
    LAB = "lab"
    HSV = "hsv"


@dataclass
class TransformOutput:
    """Output from transform operations."""
    data: "torch.Tensor"
    original_size: Tuple[int, int]
    transformed_size: Tuple[int, int]


# ============================================================================
# Triton Kernels
# ============================================================================

if HAS_TRITON:
    
    @triton.jit
    def normalize_kernel(
        x_ptr,
        out_ptr,
        mean_ptr,
        std_ptr,
        n_elements,
        n_channels,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Fused normalization kernel: out = (x - mean) / std
        
        Memory-coalesced access pattern for HBM throughput.
        """
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load input
        x = tl.load(x_ptr + offsets, mask=mask)
        
        # Determine channel index
        channel_idx = (offsets // (n_elements // n_channels)) % n_channels
        
        # Load per-channel mean/std
        mean = tl.load(mean_ptr + channel_idx, mask=mask)
        std = tl.load(std_ptr + channel_idx, mask=mask)
        
        # Normalize
        out = (x - mean) / (std + 1e-8)
        
        # Store
        tl.store(out_ptr + offsets, out, mask=mask)
    
    
    @triton.jit
    def bilinear_resize_kernel(
        x_ptr,
        out_ptr,
        in_h, in_w,
        out_h, out_w,
        n_channels,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Bilinear interpolation resize kernel.
        
        Computes interpolated values with fused memory access.
        """
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        
        total_elements = out_h * out_w * n_channels
        mask = offsets < total_elements
        
        # Compute output coordinates
        c = offsets % n_channels
        temp = offsets // n_channels
        out_x = temp % out_w
        out_y = temp // out_w
        
        # Map to input coordinates (scale)
        scale_y = tl.fdiv(in_h.to(tl.float32), out_h.to(tl.float32))
        scale_x = tl.fdiv(in_w.to(tl.float32), out_w.to(tl.float32))
        
        src_y = (out_y.to(tl.float32) + 0.5) * scale_y - 0.5
        src_x = (out_x.to(tl.float32) + 0.5) * scale_x - 0.5
        
        # Clamp to valid range
        src_y = tl.maximum(src_y, 0.0)
        src_x = tl.maximum(src_x, 0.0)
        src_y = tl.minimum(src_y, (in_h - 1).to(tl.float32))
        src_x = tl.minimum(src_x, (in_w - 1).to(tl.float32))
        
        # Integer coordinates
        y0 = src_y.to(tl.int32)
        x0 = src_x.to(tl.int32)
        y1 = tl.minimum(y0 + 1, in_h - 1)
        x1 = tl.minimum(x0 + 1, in_w - 1)
        
        # Fractional parts
        fy = src_y - y0.to(tl.float32)
        fx = src_x - x0.to(tl.float32)
        
        # Compute input offsets
        stride_c = 1
        stride_w = n_channels
        stride_h = in_w * n_channels
        
        idx_00 = y0 * stride_h + x0 * stride_w + c * stride_c
        idx_01 = y0 * stride_h + x1 * stride_w + c * stride_c
        idx_10 = y1 * stride_h + x0 * stride_w + c * stride_c
        idx_11 = y1 * stride_h + x1 * stride_w + c * stride_c
        
        # Load corner values
        v00 = tl.load(x_ptr + idx_00, mask=mask)
        v01 = tl.load(x_ptr + idx_01, mask=mask)
        v10 = tl.load(x_ptr + idx_10, mask=mask)
        v11 = tl.load(x_ptr + idx_11, mask=mask)
        
        # Bilinear interpolation
        v0 = v00 * (1.0 - fx) + v01 * fx
        v1 = v10 * (1.0 - fx) + v11 * fx
        result = v0 * (1.0 - fy) + v1 * fy
        
        # Store
        tl.store(out_ptr + offsets, result, mask=mask)
    
    
    @triton.jit
    def rgb_to_gray_kernel(
        x_ptr,
        out_ptr,
        n_pixels,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        RGB to grayscale conversion.
        
        Uses ITU-R BT.601 coefficients: 0.299R + 0.587G + 0.114B
        """
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_pixels
        
        # Load RGB values (interleaved: RGBRGBRGB...)
        r_idx = offsets * 3
        g_idx = offsets * 3 + 1
        b_idx = offsets * 3 + 2
        
        r = tl.load(x_ptr + r_idx, mask=mask)
        g = tl.load(x_ptr + g_idx, mask=mask)
        b = tl.load(x_ptr + b_idx, mask=mask)
        
        # Grayscale
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        
        # Store
        tl.store(out_ptr + offsets, gray, mask=mask)
    
    
    @triton.jit
    def rgb_to_hsv_kernel(
        x_ptr,
        out_ptr,
        n_pixels,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        RGB to HSV color space conversion.
        
        H: [0, 360), S: [0, 1], V: [0, 1]
        """
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_pixels
        
        # Load RGB (normalized to [0,1])
        r_idx = offsets * 3
        g_idx = offsets * 3 + 1
        b_idx = offsets * 3 + 2
        
        r = tl.load(x_ptr + r_idx, mask=mask)
        g = tl.load(x_ptr + g_idx, mask=mask)
        b = tl.load(x_ptr + b_idx, mask=mask)
        
        # Max and min
        max_val = tl.maximum(tl.maximum(r, g), b)
        min_val = tl.minimum(tl.minimum(r, g), b)
        delta = max_val - min_val
        
        # Value
        v = max_val
        
        # Saturation
        s = tl.where(max_val > 0, delta / (max_val + 1e-8), 0.0)
        
        # Hue (simplified)
        h = tl.where(delta > 0, 
                     tl.where(max_val == r, 60.0 * ((g - b) / (delta + 1e-8)),
                     tl.where(max_val == g, 60.0 * (2.0 + (b - r) / (delta + 1e-8)),
                              60.0 * (4.0 + (r - g) / (delta + 1e-8)))), 
                     0.0)
        h = tl.where(h < 0, h + 360.0, h)
        
        # Store HSV
        tl.store(out_ptr + r_idx, h / 360.0, mask=mask)  # Normalize H to [0,1]
        tl.store(out_ptr + g_idx, s, mask=mask)
        tl.store(out_ptr + b_idx, v, mask=mask)


class ImageTransforms:
    """
    GPU-accelerated image transformations.
    
    Features:
    - Triton kernel acceleration
    - Fused operations
    - Memory-efficient processing
    - Batch support
    """
    
    def __init__(self, device: str = "cuda", use_triton: bool = True):
        """
        Initialize transforms.
        
        Args:
            device: Target device ('cuda' or 'cpu')
            use_triton: Use Triton kernels when available
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required for ImageTransforms")
        
        self.device = device
        self.use_triton = use_triton and HAS_TRITON and device == "cuda"
    
    def resize(
        self,
        x: "torch.Tensor",
        size: Tuple[int, int],
        mode: InterpolationMode = InterpolationMode.BILINEAR,
    ) -> "torch.Tensor":
        """
        Resize image tensor.
        
        Args:
            x: Input tensor (B, C, H, W) or (C, H, W)
            size: Target (height, width)
            mode: Interpolation mode
            
        Returns:
            Resized tensor
        """
        if self.use_triton and mode == InterpolationMode.BILINEAR:
            return self._triton_resize(x, size)
        
        # Fallback to PyTorch
        is_batched = x.dim() == 4
        if not is_batched:
            x = x.unsqueeze(0)
        
        mode_str = mode.value
        if mode == InterpolationMode.NEAREST:
            mode_str = "nearest"
        elif mode == InterpolationMode.BILINEAR:
            mode_str = "bilinear"
        elif mode == InterpolationMode.BICUBIC:
            mode_str = "bicubic"
        
        out = F.interpolate(x, size=size, mode=mode_str, align_corners=False)
        
        if not is_batched:
            out = out.squeeze(0)
        
        return out
    
    def _triton_resize(
        self,
        x: "torch.Tensor",
        size: Tuple[int, int],
    ) -> "torch.Tensor":
        """Triton-accelerated bilinear resize."""
        is_batched = x.dim() == 4
        if not is_batched:
            x = x.unsqueeze(0)
        
        B, C, H, W = x.shape
        out_h, out_w = size
        
        # Reshape to (B, H, W, C) for kernel
        x = x.permute(0, 2, 3, 1).contiguous()
        out = torch.empty((B, out_h, out_w, C), device=x.device, dtype=x.dtype)
        
        for b in range(B):
            n_elements = out_h * out_w * C
            grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
            
            bilinear_resize_kernel[grid](
                x[b].data_ptr(),
                out[b].data_ptr(),
                H, W,
                out_h, out_w,
                C,
                BLOCK_SIZE=1024,
            )
        
        # Reshape back to (B, C, H, W)
        out = out.permute(0, 3, 1, 2)
        
        if not is_batched:
            out = out.squeeze(0)
        
        return out
    
    def normalize(
        self,
        x: "torch.Tensor",
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    ) -> "torch.Tensor":
        """
        Normalize image tensor.
        
        Args:
            x: Input tensor (B, C, H, W) or (C, H, W)
            mean: Per-channel mean
            std: Per-channel std
            
        Returns:
            Normalized tensor
        """
        if self.use_triton:
            return self._triton_normalize(x, mean, std)
        
        # Fallback to PyTorch
        is_batched = x.dim() == 4
        if not is_batched:
            x = x.unsqueeze(0)
        
        mean_t = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
        std_t = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
        
        out = (x - mean_t) / (std_t + 1e-8)
        
        if not is_batched:
            out = out.squeeze(0)
        
        return out
    
    def _triton_normalize(
        self,
        x: "torch.Tensor",
        mean: Tuple[float, ...],
        std: Tuple[float, ...],
    ) -> "torch.Tensor":
        """Triton-accelerated normalization."""
        is_batched = x.dim() == 4
        if not is_batched:
            x = x.unsqueeze(0)
        
        B, C, H, W = x.shape
        x = x.contiguous()
        out = torch.empty_like(x)
        
        mean_t = torch.tensor(mean, device=x.device, dtype=x.dtype)
        std_t = torch.tensor(std, device=x.device, dtype=x.dtype)
        
        n_elements = B * C * H * W
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        
        normalize_kernel[grid](
            x.data_ptr(),
            out.data_ptr(),
            mean_t.data_ptr(),
            std_t.data_ptr(),
            n_elements,
            C,
            BLOCK_SIZE=1024,
        )
        
        if not is_batched:
            out = out.squeeze(0)
        
        return out
    
    def to_grayscale(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Convert RGB to grayscale.
        
        Args:
            x: Input tensor (B, 3, H, W) or (3, H, W)
            
        Returns:
            Grayscale tensor (B, 1, H, W) or (1, H, W)
        """
        if self.use_triton:
            return self._triton_to_grayscale(x)
        
        # Fallback: weighted sum
        is_batched = x.dim() == 4
        if not is_batched:
            x = x.unsqueeze(0)
        
        weights = torch.tensor([0.299, 0.587, 0.114], device=x.device, dtype=x.dtype)
        weights = weights.view(1, 3, 1, 1)
        
        out = (x * weights).sum(dim=1, keepdim=True)
        
        if not is_batched:
            out = out.squeeze(0)
        
        return out
    
    def _triton_to_grayscale(self, x: "torch.Tensor") -> "torch.Tensor":
        """Triton-accelerated grayscale conversion."""
        is_batched = x.dim() == 4
        if not is_batched:
            x = x.unsqueeze(0)
        
        B, C, H, W = x.shape
        assert C == 3, "Input must have 3 channels"
        
        # Reshape to (B, H, W, C)
        x = x.permute(0, 2, 3, 1).contiguous()
        out = torch.empty((B, H, W, 1), device=x.device, dtype=x.dtype)
        
        for b in range(B):
            n_pixels = H * W
            grid = lambda meta: (triton.cdiv(n_pixels, meta['BLOCK_SIZE']),)
            
            rgb_to_gray_kernel[grid](
                x[b].data_ptr(),
                out[b].data_ptr(),
                n_pixels,
                BLOCK_SIZE=1024,
            )
        
        # Reshape to (B, 1, H, W)
        out = out.permute(0, 3, 1, 2)
        
        if not is_batched:
            out = out.squeeze(0)
        
        return out
    
    def to_hsv(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Convert RGB to HSV.
        
        Args:
            x: Input tensor (B, 3, H, W) in [0, 1]
            
        Returns:
            HSV tensor (B, 3, H, W)
        """
        is_batched = x.dim() == 4
        if not is_batched:
            x = x.unsqueeze(0)
        
        if self.use_triton:
            return self._triton_to_hsv(x, is_batched)
        
        # PyTorch fallback
        r, g, b = x[:, 0], x[:, 1], x[:, 2]
        
        max_val, _ = x.max(dim=1)
        min_val, _ = x.min(dim=1)
        delta = max_val - min_val
        
        # Value
        v = max_val
        
        # Saturation
        s = torch.where(max_val > 0, delta / (max_val + 1e-8), torch.zeros_like(max_val))
        
        # Hue
        h = torch.zeros_like(max_val)
        
        mask_r = (max_val == r) & (delta > 0)
        mask_g = (max_val == g) & (delta > 0)
        mask_b = (max_val == b) & (delta > 0)
        
        h = torch.where(mask_r, 60 * ((g - b) / (delta + 1e-8)) % 360, h)
        h = torch.where(mask_g, 60 * (2 + (b - r) / (delta + 1e-8)), h)
        h = torch.where(mask_b, 60 * (4 + (r - g) / (delta + 1e-8)), h)
        h = h / 360  # Normalize to [0, 1]
        
        out = torch.stack([h, s, v], dim=1)
        
        if not is_batched:
            out = out.squeeze(0)
        
        return out
    
    def _triton_to_hsv(self, x: "torch.Tensor", is_batched: bool) -> "torch.Tensor":
        """Triton-accelerated HSV conversion."""
        B, C, H, W = x.shape
        
        # Reshape to (B, H, W, C)
        x = x.permute(0, 2, 3, 1).contiguous()
        out = torch.empty_like(x)
        
        for b in range(B):
            n_pixels = H * W
            grid = lambda meta: (triton.cdiv(n_pixels, meta['BLOCK_SIZE']),)
            
            rgb_to_hsv_kernel[grid](
                x[b].data_ptr(),
                out[b].data_ptr(),
                n_pixels,
                BLOCK_SIZE=1024,
            )
        
        # Reshape to (B, C, H, W)
        out = out.permute(0, 3, 1, 2)
        
        if not is_batched:
            out = out.squeeze(0)
        
        return out
    
    def center_crop(
        self,
        x: "torch.Tensor",
        size: Tuple[int, int],
    ) -> "torch.Tensor":
        """
        Center crop image.
        
        Args:
            x: Input tensor (B, C, H, W) or (C, H, W)
            size: Crop (height, width)
            
        Returns:
            Cropped tensor
        """
        is_batched = x.dim() == 4
        if not is_batched:
            x = x.unsqueeze(0)
        
        _, _, H, W = x.shape
        crop_h, crop_w = size
        
        start_h = (H - crop_h) // 2
        start_w = (W - crop_w) // 2
        
        out = x[:, :, start_h:start_h + crop_h, start_w:start_w + crop_w]
        
        if not is_batched:
            out = out.squeeze(0)
        
        return out.contiguous()
    
    def random_crop(
        self,
        x: "torch.Tensor",
        size: Tuple[int, int],
    ) -> "torch.Tensor":
        """
        Random crop image.
        
        Args:
            x: Input tensor (B, C, H, W) or (C, H, W)
            size: Crop (height, width)
            
        Returns:
            Cropped tensor
        """
        is_batched = x.dim() == 4
        if not is_batched:
            x = x.unsqueeze(0)
        
        _, _, H, W = x.shape
        crop_h, crop_w = size
        
        max_h = H - crop_h
        max_w = W - crop_w
        
        start_h = torch.randint(0, max(max_h, 1), (1,)).item()
        start_w = torch.randint(0, max(max_w, 1), (1,)).item()
        
        out = x[:, :, start_h:start_h + crop_h, start_w:start_w + crop_w]
        
        if not is_batched:
            out = out.squeeze(0)
        
        return out.contiguous()
    
    def horizontal_flip(self, x: "torch.Tensor") -> "torch.Tensor":
        """Horizontal flip."""
        return torch.flip(x, dims=[-1])
    
    def vertical_flip(self, x: "torch.Tensor") -> "torch.Tensor":
        """Vertical flip."""
        return torch.flip(x, dims=[-2])
