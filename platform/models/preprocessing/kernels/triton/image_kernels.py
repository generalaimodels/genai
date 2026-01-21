"""
Image Preprocessing Triton Kernels

Optimized kernels for image resizing, normalization, and color conversion.
"""

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    
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
        
        Memory-coalesced resizing for optimal HBM bandwidth utilization.
        
        Args:
            x_ptr: Input image (H, W, C)
            out_ptr: Output image (out_H, out_W, C)
            in_h, in_w: Input dimensions
            out_h, out_w: Output dimensions
            n_channels: Number of channels
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
        
        # Compute input offsets (HWC layout)
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
    def normalize_kernel(
        x_ptr,
        out_ptr,
        mean_ptr,
        std_ptr,
        n_elements,
        n_channels,
        height,
        width,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Fused normalization kernel: out = (x - mean) / std
        
        Per-channel normalization with memory coalescing.
        
        Args:
            x_ptr: Input tensor (C, H, W) or flattened
            out_ptr: Output tensor
            mean_ptr: Per-channel mean (C,)
            std_ptr: Per-channel std (C,)
            n_elements: Total elements
            n_channels: Number of channels
            height, width: Spatial dimensions
        """
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load input
        x = tl.load(x_ptr + offsets, mask=mask)
        
        # Compute channel index (for CHW layout)
        spatial_size = height * width
        channel_idx = offsets // spatial_size
        channel_idx = channel_idx % n_channels
        
        # Load per-channel mean/std
        mean = tl.load(mean_ptr + channel_idx, mask=mask)
        std = tl.load(std_ptr + channel_idx, mask=mask)
        
        # Normalize
        out = (x - mean) / (std + 1e-8)
        
        # Store
        tl.store(out_ptr + offsets, out, mask=mask)
    
    
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
        
        Args:
            x_ptr: Input RGB image (H*W, 3) or (H, W, 3) flattened
            out_ptr: Output grayscale (H*W,)
            n_pixels: Number of pixels
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
        
        # ITU-R BT.601 grayscale conversion
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
        
        H: [0, 1], S: [0, 1], V: [0, 1]
        
        Args:
            x_ptr: Input RGB (H*W*3,) normalized to [0,1]
            out_ptr: Output HSV (H*W*3,)
            n_pixels: Number of pixels
        """
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_pixels
        
        # Load RGB (interleaved)
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
        
        # Hue computation
        h = tl.zeros_like(r)
        
        # Case: max == r
        h = tl.where(
            (delta > 0) & (max_val == r),
            (g - b) / (delta + 1e-8) / 6.0,
            h
        )
        # Case: max == g
        h = tl.where(
            (delta > 0) & (max_val == g),
            (2.0 + (b - r) / (delta + 1e-8)) / 6.0,
            h
        )
        # Case: max == b
        h = tl.where(
            (delta > 0) & (max_val == b),
            (4.0 + (r - g) / (delta + 1e-8)) / 6.0,
            h
        )
        
        # Wrap negative hue
        h = tl.where(h < 0, h + 1.0, h)
        
        # Store HSV
        tl.store(out_ptr + r_idx, h, mask=mask)
        tl.store(out_ptr + g_idx, s, mask=mask)
        tl.store(out_ptr + b_idx, v, mask=mask)
    
    
    @triton.jit
    def random_crop_kernel(
        x_ptr,
        out_ptr,
        in_h, in_w,
        crop_h, crop_w,
        start_y, start_x,
        n_channels,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Random/center crop kernel.
        
        Extracts a crop from input image.
        
        Args:
            x_ptr: Input image (H, W, C)
            out_ptr: Output crop (crop_H, crop_W, C)
            in_h, in_w: Input dimensions
            crop_h, crop_w: Crop dimensions
            start_y, start_x: Crop start position
            n_channels: Number of channels
        """
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        
        total_elements = crop_h * crop_w * n_channels
        mask = offsets < total_elements
        
        # Compute crop coordinates
        c = offsets % n_channels
        temp = offsets // n_channels
        crop_x = temp % crop_w
        crop_y = temp // crop_w
        
        # Map to input coordinates
        in_x = start_x + crop_x
        in_y = start_y + crop_y
        
        # Compute input index
        in_idx = in_y * in_w * n_channels + in_x * n_channels + c
        
        # Load and store
        val = tl.load(x_ptr + in_idx, mask=mask)
        tl.store(out_ptr + offsets, val, mask=mask)
    
    
    @triton.jit
    def horizontal_flip_kernel(
        x_ptr,
        out_ptr,
        height, width,
        n_channels,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Horizontal flip kernel.
        
        Args:
            x_ptr: Input image (H, W, C)
            out_ptr: Output flipped image (H, W, C)
            height, width: Dimensions
            n_channels: Number of channels
        """
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        
        total_elements = height * width * n_channels
        mask = offsets < total_elements
        
        # Compute coordinates
        c = offsets % n_channels
        temp = offsets // n_channels
        x = temp % width
        y = temp // width
        
        # Flip x coordinate
        flipped_x = width - 1 - x
        
        # Compute flipped index
        flipped_idx = y * width * n_channels + flipped_x * n_channels + c
        
        # Load and store
        val = tl.load(x_ptr + flipped_idx, mask=mask)
        tl.store(out_ptr + offsets, val, mask=mask)
