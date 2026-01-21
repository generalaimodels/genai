"""
Triton Kernel Autotuning Module

Autotuning configurations for preprocessing kernels.
Dynamically selects optimal tile sizes and block configurations.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import os
import json
from pathlib import Path

try:
    import triton
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class AutotuneConfig:
    """Autotuning configuration for a kernel."""
    kernel_name: str
    param_grid: Dict[str, List[int]]
    num_warps_options: List[int] = field(default_factory=lambda: [4, 8])
    num_stages_options: List[int] = field(default_factory=lambda: [2, 3, 4])
    best_config: Optional[Dict[str, int]] = None


# Default configurations for preprocessing kernels
DEFAULT_AUTOTUNE_CONFIGS = {
    "bilinear_resize": AutotuneConfig(
        kernel_name="bilinear_resize_kernel",
        param_grid={
            "BLOCK_SIZE": [256, 512, 1024, 2048],
        },
        num_warps_options=[4, 8],
    ),
    "normalize": AutotuneConfig(
        kernel_name="normalize_kernel",
        param_grid={
            "BLOCK_SIZE": [256, 512, 1024],
        },
        num_warps_options=[4, 8],
    ),
    "rgb_to_gray": AutotuneConfig(
        kernel_name="rgb_to_gray_kernel",
        param_grid={
            "BLOCK_SIZE": [256, 512, 1024],
        },
        num_warps_options=[4, 8],
    ),
    "mel_filterbank": AutotuneConfig(
        kernel_name="mel_filterbank_kernel",
        param_grid={
            "BLOCK_SIZE_MEL": [16, 32, 64],
            "BLOCK_SIZE_FREQ": [32, 64, 128],
        },
        num_warps_options=[4, 8],
    ),
    "log_mel": AutotuneConfig(
        kernel_name="log_mel_kernel",
        param_grid={
            "BLOCK_SIZE": [512, 1024, 2048],
        },
        num_warps_options=[4, 8],
    ),
    "embedding_lookup": AutotuneConfig(
        kernel_name="fused_tokenize_lookup_kernel",
        param_grid={
            "BLOCK_SIZE_SEQ": [16, 32, 64],
            "BLOCK_SIZE_EMB": [64, 128, 256],
        },
        num_warps_options=[4, 8],
    ),
}


class AutotuneCache:
    """
    Cache for autotuning results.
    
    Stores optimal configurations per kernel and input shape.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize cache.
        
        Args:
            cache_dir: Directory to store cache files
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "preprocessing_autotune"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._load_cache()
    
    def _load_cache(self):
        """Load cache from disk."""
        cache_file = self.cache_dir / "autotune_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    self._cache = json.load(f)
            except Exception:
                self._cache = {}
    
    def _save_cache(self):
        """Save cache to disk."""
        cache_file = self.cache_dir / "autotune_cache.json"
        try:
            with open(cache_file, "w") as f:
                json.dump(self._cache, f, indent=2)
        except Exception:
            pass
    
    def get(
        self,
        kernel_name: str,
        input_shape: Tuple[int, ...],
    ) -> Optional[Dict[str, int]]:
        """
        Get cached configuration for kernel and shape.
        
        Args:
            kernel_name: Kernel name
            input_shape: Input tensor shape
            
        Returns:
            Cached configuration or None
        """
        key = f"{kernel_name}_{input_shape}"
        return self._cache.get(key)
    
    def set(
        self,
        kernel_name: str,
        input_shape: Tuple[int, ...],
        config: Dict[str, int],
    ):
        """
        Store configuration in cache.
        
        Args:
            kernel_name: Kernel name
            input_shape: Input tensor shape
            config: Optimal configuration
        """
        key = f"{kernel_name}_{input_shape}"
        self._cache[key] = config
        self._save_cache()


if HAS_TRITON and HAS_TORCH:
    
    def create_autotune_decorator(
        config: AutotuneConfig,
        key: List[str],
    ):
        """
        Create Triton autotune decorator for a kernel.
        
        Args:
            config: Autotuning configuration
            key: Keys for caching (e.g., ['n_elements'])
            
        Returns:
            Autotune decorator
        """
        # Build configurations
        configs = []
        
        for block_values in _param_grid_product(config.param_grid):
            for num_warps in config.num_warps_options:
                for num_stages in config.num_stages_options:
                    cfg = triton.Config(
                        kwargs=block_values,
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                    configs.append(cfg)
        
        return triton.autotune(
            configs=configs,
            key=key,
        )
    
    
    def _param_grid_product(
        param_grid: Dict[str, List[int]]
    ) -> List[Dict[str, int]]:
        """Generate all combinations of parameter values."""
        if not param_grid:
            return [{}]
        
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        results = []
        
        def _product(idx: int, current: Dict[str, int]):
            if idx == len(keys):
                results.append(current.copy())
                return
            
            for val in values[idx]:
                current[keys[idx]] = val
                _product(idx + 1, current)
        
        _product(0, {})
        return results
    
    
    # Autotuned kernel wrappers
    
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=3),
            triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=4),
        ],
        key=['n_elements'],
    )
    @triton.jit
    def autotuned_normalize_kernel(
        x_ptr,
        out_ptr,
        mean_ptr,
        std_ptr,
        n_elements,
        n_channels,
        height,
        width,
        BLOCK_SIZE: triton.language.constexpr,
    ):
        """Autotuned normalization kernel."""
        from .image_kernels import normalize_kernel
        # Forward to base kernel implementation
        pid = triton.language.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + triton.language.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        x = triton.language.load(x_ptr + offsets, mask=mask)
        
        spatial_size = height * width
        channel_idx = offsets // spatial_size
        channel_idx = channel_idx % n_channels
        
        mean = triton.language.load(mean_ptr + channel_idx, mask=mask)
        std = triton.language.load(std_ptr + channel_idx, mask=mask)
        
        out = (x - mean) / (std + 1e-8)
        
        triton.language.store(out_ptr + offsets, out, mask=mask)
    
    
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=3),
            triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=4),
        ],
        key=['n_elements'],
    )
    @triton.jit
    def autotuned_log_mel_kernel(
        x_ptr,
        out_ptr,
        n_elements,
        log_offset,
        BLOCK_SIZE: triton.language.constexpr,
    ):
        """Autotuned log mel kernel."""
        pid = triton.language.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + triton.language.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        x = triton.language.load(x_ptr + offsets, mask=mask)
        x = triton.language.maximum(x, log_offset)
        out = triton.language.log(x)
        
        triton.language.store(out_ptr + offsets, out, mask=mask)


def get_optimal_config(
    kernel_name: str,
    input_shape: Tuple[int, ...],
    device: str = "cuda",
) -> Dict[str, int]:
    """
    Get optimal kernel configuration for input shape.
    
    Args:
        kernel_name: Kernel name
        input_shape: Input tensor shape
        device: Target device
        
    Returns:
        Optimal configuration dict
    """
    # Check cache
    cache = AutotuneCache()
    cached = cache.get(kernel_name, input_shape)
    if cached:
        return cached
    
    # Use default configurations
    if kernel_name in DEFAULT_AUTOTUNE_CONFIGS:
        config = DEFAULT_AUTOTUNE_CONFIGS[kernel_name]
        # Return first (default) configuration
        defaults = {}
        for param, values in config.param_grid.items():
            defaults[param] = values[1] if len(values) > 1 else values[0]
        return defaults
    
    # Fallback
    return {"BLOCK_SIZE": 1024}


def benchmark_kernel(
    kernel_fn,
    input_tensors: List["torch.Tensor"],
    configs: List[Dict[str, int]],
    warmup: int = 10,
    rep: int = 100,
) -> Tuple[Dict[str, int], float]:
    """
    Benchmark kernel with different configurations.
    
    Args:
        kernel_fn: Kernel function
        input_tensors: Input tensors
        configs: Configurations to test
        warmup: Warmup iterations
        rep: Repetitions for timing
        
    Returns:
        (best_config, best_time_ms)
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required for benchmarking")
    
    best_config = configs[0]
    best_time = float('inf')
    
    for config in configs:
        # Warmup
        for _ in range(warmup):
            kernel_fn(*input_tensors, **config)
        
        torch.cuda.synchronize()
        
        # Time
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(rep):
            kernel_fn(*input_tensors, **config)
        end.record()
        
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end) / rep
        
        if elapsed < best_time:
            best_time = elapsed
            best_config = config
    
    return best_config, best_time
