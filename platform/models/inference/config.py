"""
Inference Configuration
=======================
Configuration for high-performance inference engine.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal, List
from enum import Enum


class QuantizationType(str, Enum):
    """Quantization methods."""
    NONE = "none"
    INT8 = "int8"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    AWQ = "awq"
    GPTQ = "gptq"


class DecodingStrategy(str, Enum):
    """Decoding strategies."""
    GREEDY = "greedy"
    SAMPLING = "sampling"
    BEAM_SEARCH = "beam_search"
    SPECULATIVE = "speculative"


@dataclass
class KVCacheConfig:
    """KV cache configuration."""
    block_size: int = 16
    num_gpu_blocks: Optional[int] = None
    num_cpu_blocks: Optional[int] = None
    max_num_seqs: int = 256
    max_seq_len: int = 8192
    
    # PagedAttention
    enable_paging: bool = True
    swap_space_gb: float = 4.0
    
    # Prefix caching
    enable_prefix_caching: bool = True
    prefix_cache_size_mb: int = 1024


@dataclass
class SamplingConfig:
    """Sampling parameters."""
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    min_p: float = 0.0
    
    # Penalties
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    # Length
    max_tokens: int = 256
    min_tokens: int = 0
    stop_strings: List[str] = field(default_factory=list)
    
    # Beam search
    num_beams: int = 1
    length_penalty: float = 1.0
    early_stopping: bool = False


@dataclass
class SpeculativeConfig:
    """Speculative decoding configuration."""
    enabled: bool = True
    num_draft_tokens: int = 5
    draft_model_path: Optional[str] = None
    use_self_draft: bool = True
    tree_attention: bool = False


@dataclass
class InferenceConfig:
    """Complete inference configuration."""
    # Model
    model_path: str = ""
    device: str = "cuda"
    dtype: str = "bfloat16"
    
    # Quantization
    quantization: QuantizationType = QuantizationType.NONE
    
    # KV Cache
    kv_cache: KVCacheConfig = field(default_factory=KVCacheConfig)
    
    # Batching
    max_batch_size: int = 64
    max_concurrent_requests: int = 256
    
    # Prefill
    chunked_prefill: bool = True
    prefill_chunk_size: int = 512
    
    # Sampling
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    
    # Speculative
    speculative: SpeculativeConfig = field(default_factory=SpeculativeConfig)
    
    # Performance
    use_flash_attn: bool = True
    use_triton_kernels: bool = True
    cuda_graphs: bool = True
    max_cuda_graph_batch_size: int = 32
    
    # Tensor parallelism
    tensor_parallel_size: int = 1
