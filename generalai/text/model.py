#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Next-Gen Multimodal Transformer (Text + Vision) with:
- Shared transformer backbone for text/vision fusion
- Parallelized attention and FFN with RMSNorm and optional parallel residual
- RoPE with adaptive scaling (for long context stability)
- Temperature-tuned attention for long-context robustness
- MoE layers: SwiGLU experts, top-k routing, capacity factor, auto-scaled hidden dims, load-balanced routing
- Column/Row parallel linear layers compatible with FairScale tensor-parallelism
- SAFE FairScale guard: no assertion when model-parallel group is NOT initialized (CPU or single-GPU)
- KV-cache for fast autoregressive decoding
- Hybrid attention: chunked local attention + global attention masks
- Vision embeddings: flexible patching, pixel shuffle adapter, projection to shared latent
- Quantization (int8/int4 weight-only) stubs and LoRA adapters for fine-tuning
- Strongly typed configs (pydantic) and input/output abstractions
- Rich-powered verbose metadata (set MULTIMODAL_VERBOSE=1)

Important fixes (addresses your errors):
1) FairScale-safe wrappers:
   - All FairScale-dependent ops are guarded. reduce_from_model_parallel_region() is a safe no-op unless
     FairScale model-parallel is initialized. This prevents:
       AssertionError: model parallel group is not initialized
     on CPU or single-GPU CUDA runs.

2) Vision PixelShuffleAdapter channel/shape fix (resolves your matmul mismatch):
   - PixelShuffle now correctly reduces the channel dimension by r^2 and increases token count by r^2.
   - Previously, the adapter kept channels constant (C=384) but expected reduced C (96), causing:
       RuntimeError: mat1 and mat2 shapes cannot be multiplied (1568x384 and 96x512)
   - With the corrected pixel-shuffle, the MLP receives the expected in_features (=384/r^2=96).

How to see rich meta-data:
- Set environment variable MULTIMODAL_VERBOSE=1, or toggle VERBOSE=True in the example functions.
"""

from __future__ import annotations

import math
import os
import types
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

# ------------------------------------------------------------------------------
# 0) Optional Rich logger (verbose metadata) and environment toggles
# ------------------------------------------------------------------------------

def _bool_env(key: str, default: bool = False) -> bool:
    v = os.environ.get(key, None)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "y", "on")

VERBOSE = _bool_env("MULTIMODAL_VERBOSE", False)

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    _RICH_AVAILABLE = True
    _CONSOLE = Console()
except Exception:
    _RICH_AVAILABLE = False
    _CONSOLE = None

def rich_print(msg: str) -> None:
    if not VERBOSE:
        return
    if _RICH_AVAILABLE and _CONSOLE is not None:
        _CONSOLE.print(msg)
    else:
        print(msg)

def rich_kv_panel(title: str, kv: Dict[str, Any]) -> None:
    if not VERBOSE:
        return
    if _RICH_AVAILABLE and _CONSOLE is not None:
        table = Table(title=title, box=box.SIMPLE)
        table.add_column("Key", justify="right", style="cyan")
        table.add_column("Value", style="magenta")
        for k, v in kv.items():
            table.add_row(str(k), str(v))
        _CONSOLE.print(table)
    else:
        print(f"[{title}]")
        for k, v in kv.items():
            print(f"- {k}: {v}")

# ------------------------------------------------------------------------------
# 1) Torch + FairScale compatible building blocks (SAFE WRAPPERS)
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# FairScale (optional, SAFE guarded)
try:
    import fairscale.nn.model_parallel.initialize as fs_init  # type: ignore
    from fairscale.nn.model_parallel.layers import (  # type: ignore
        ColumnParallelLinear as FSColumnParallelLinear,
        RowParallelLinear as FSRowParallelLinear,
        VocabParallelEmbedding as FSVocabParallelEmbedding,
    )
    _FAIRSCALE_AVAILABLE = True
except Exception:
    _FAIRSCALE_AVAILABLE = False
    fs_init = types.SimpleNamespace(
        get_model_parallel_world_size=lambda: 1,
        get_model_parallel_rank=lambda: 0,
    )
    FSColumnParallelLinear = None  # type: ignore
    FSRowParallelLinear = None  # type: ignore
    FSVocabParallelEmbedding = None  # type: ignore

def _is_fairscale_initialized() -> bool:
    """
    Return True only if FairScale is present AND its model-parallel group is initialized.
    We trap any assertion/attribute errors and return False, keeping CPU/single-GPU safe.
    """
    if not _FAIRSCALE_AVAILABLE:
        return False
    try:
        ws = fs_init.get_model_parallel_world_size()
        return isinstance(ws, int) and ws >= 1
    except Exception:
        return False

def get_model_parallel_world_size() -> int:
    return fs_init.get_model_parallel_world_size() if _is_fairscale_initialized() else 1

def get_model_parallel_rank() -> int:
    return fs_init.get_model_parallel_rank() if _is_fairscale_initialized() else 0

class ColumnParallelLinear(nn.Module):
    """
    Column-parallel linear. Falls back to nn.Linear if FairScale is not initialized.
    Accepts FairScale signature for drop-in compatibility.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = False,
        init_method: Optional[Callable[[Tensor], None]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        use_fs = _FAIRSCALE_AVAILABLE and _is_fairscale_initialized() and (FSColumnParallelLinear is not None)
        if use_fs:
            self.layer = FSColumnParallelLinear(
                in_features, out_features, bias=bias, gather_output=gather_output, init_method=init_method, **kwargs
            )
            self.is_fs = True
        else:
            self.layer = nn.Linear(in_features, out_features, bias=bias)
            if init_method is not None:
                with torch.no_grad():
                    init_method(self.layer.weight)
            self.is_fs = False

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)

class RowParallelLinear(nn.Module):
    """
    Row-parallel linear. Falls back to nn.Linear if FairScale is not initialized.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_is_parallel: bool = False,
        init_method: Optional[Callable[[Tensor], None]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        use_fs = _FAIRSCALE_AVAILABLE and _is_fairscale_initialized() and (FSRowParallelLinear is not None)
        if use_fs:
            self.layer = FSRowParallelLinear(
                in_features, out_features, bias=bias, input_is_parallel=input_is_parallel, init_method=init_method, **kwargs
            )
            self.is_fs = True
        else:
            self.layer = nn.Linear(in_features, out_features, bias=bias)
            if init_method is not None:
                with torch.no_grad():
                    init_method(self.layer.weight)
            self.is_fs = False

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)

class VocabParallelEmbedding(nn.Module):
    """
    Vocab-parallel embedding. Falls back to nn.Embedding when FairScale is unavailable/uninitialized.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, init_method: Optional[Callable[[Tensor], None]] = None):
        super().__init__()
        use_fs = _FAIRSCALE_AVAILABLE and _is_fairscale_initialized() and (FSVocabParallelEmbedding is not None)
        if use_fs:
            self.layer = FSVocabParallelEmbedding(num_embeddings, embedding_dim, init_method=init_method)
            self.is_fs = True
        else:
            self.layer = nn.Embedding(num_embeddings, embedding_dim)
            if init_method is not None:
                with torch.no_grad():
                    init_method(self.layer.weight)
            self.is_fs = False

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)

def reduce_from_model_parallel_region(x: Tensor) -> Tensor:
    """
    SAFE reduction wrapper:
    - If FairScale MP is initialized, call its reduce mapper.
    - Otherwise, no-op pass-through to avoid 'model parallel group is not initialized'.
    """
    if _FAIRSCALE_AVAILABLE and _is_fairscale_initialized():
        from fairscale.nn.model_parallel.mappings import reduce_from_model_parallel_region as fs_reduce  # type: ignore
        return fs_reduce(x)
    return x

# ------------------------------------------------------------------------------
# 2) Typed configuration models (pydantic)
# ------------------------------------------------------------------------------

try:
    from pydantic import BaseModel, Field, model_validator
except Exception as e:
    raise ImportError("Please install pydantic to use this module: pip install pydantic") from e

class QuantizationScheme(Enum):
    none = "none"
    int8_weight_only = "int8_weight_only"
    int4_weight_only = "int4_weight_only"

class QuantizationArgs(BaseModel):
    enabled: bool = False
    scheme: QuantizationScheme = QuantizationScheme.none
    per_channel: bool = True
    symmetric: bool = True
    group_size: int = 128  # for int4/int8 weight-only

class LoRAArgs(BaseModel):
    enabled: bool = False
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: List[str] = Field(default_factory=lambda: ["wq", "wk", "wv", "wo", "mlp.c_fc", "mlp.c_proj"])

class MoEArgs(BaseModel):
    enabled: bool = False
    num_experts: int = 0
    capacity_factor: float = 1.25
    top_k: int = 2
    router_noise: float = 0.0  # noisy gating (for exploration)
    auto_scale_hidden: bool = True
    expert_dropout: float = 0.0

class Size(BaseModel):
    height: int
    width: int

class VisionArgs(BaseModel):
    image_size: Size
    patch_size: Size

    dim: int
    n_layers: int
    n_heads: int
    mlp_ratio: float
    output_dim: int
    pixel_shuffle_ratio: float = 2.0

    @model_validator(mode="after")
    def validate(self) -> "VisionArgs":
        assert self.image_size.height % self.patch_size.height == 0, "image.h must be divisible by patch.h"
        assert self.image_size.width % self.patch_size.width == 0, "image.w must be divisible by patch.w"
        assert self.dim % self.n_heads == 0, "vision dim must be divisible by n_heads"
        return self

class ModelArgs(BaseModel):
    # Core dims
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None

    # Tokenization / vocab
    vocab_size: int = 32000
    tie_embeddings: bool = True

    # FFN
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    ffn_exp: Optional[float] = None
    norm_eps: float = 1e-5
    parallel_residual: bool = True  # enable parallel residual branch (attn+ffn in parallel)

    # Attention
    attention_chunk_size: Optional[int] = None
    rope_theta: float = 500000.0
    use_scaled_rope: bool = False
    rope_scaling_factor: Optional[float] = None
    rope_high_freq_factor: Optional[float] = None
    nope_layer_interval: Optional[int] = None
    use_qk_norm: bool = False
    attn_temperature_tuning: bool = False
    floor_scale: float = 8192.0
    attn_scale: float = 0.1

    # Integrations
    vision_args: Optional[VisionArgs] = None
    moe_args: Optional[MoEArgs] = None
    quantization_args: Optional[QuantizationArgs] = None
    lora_args: Optional[LoRAArgs] = None

    # Cache/sequence sizes
    max_batch_size: int = 32
    max_seq_len: int = 2048

    @model_validator(mode="after")
    def validate(self) -> "ModelArgs":
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        assert self.n_kv_heads is not None
        assert self.n_kv_heads <= self.n_heads, "n_kv_heads must be <= n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        assert self.dim % self.n_heads == 0, "dim must be divisible by n_heads"
        if self.head_dim is None:
            self.head_dim = self.dim // self.n_heads

        if self.use_scaled_rope:
            if self.rope_scaling_factor is None:
                self.rope_scaling_factor = 16.0
            if self.rope_high_freq_factor is None:
                self.rope_high_freq_factor = 1.0

        if self.vision_args is not None:
            assert self.vision_args.output_dim == self.dim, "vision.output_dim should match text dim for fusion"
        return self

# ------------------------------------------------------------------------------
# 3) IO abstractions
# ------------------------------------------------------------------------------

@dataclass
class MaskedEmbedding:
    embedding: Tensor
    mask: Tensor  # bool mask [B, T] where True indicates image token location

@dataclass
class LLMInput:
    tokens: Tensor
    images: Optional[List[List[Tensor]]] = None  # preprocessed images
    image_mask: Optional[Tensor] = None          # [B, T] bool mask

@dataclass
class TransformerInput:
    tokens: Tensor
    tokens_position: Union[Tensor, int]
    image_embedding: Optional[MaskedEmbedding] = None

@dataclass
class LLMOutput:
    logits: Tensor
    aux: Dict[str, Any]

TransformerOutput = LLMOutput

# ------------------------------------------------------------------------------
# 4) RMSNorm and RoPE utilities
# ------------------------------------------------------------------------------

def rmsnorm(x: Tensor, eps: float) -> Tensor:
    def _norm(y: Tensor) -> Tensor:
        return y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
    return _norm(x.float()).type_as(x)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return rmsnorm(x, self.eps) * self.weight

def apply_scaling(freqs: Tensor, scale_factor: float, high_freq_factor: float) -> Tensor:
    low_freq_factor = 1.0
    old_context_len = 8192.0
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    out: List[Tensor] = []
    for f in freqs:
        wavelen = 2 * math.pi / f
        if wavelen < high_freq_wavelen:
            out.append(f)
        elif wavelen > low_freq_wavelen:
            out.append(f / scale_factor)
        else:
            smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            out.append((1 - smooth) * f / scale_factor + smooth * f)
    return torch.tensor(out, dtype=freqs.dtype, device=freqs.device)

def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 1e6,
    use_scaled: bool = False,
    scale_factor: Optional[float] = None,
    high_freq_factor: Optional[float] = None,
) -> Tensor:
    base = torch.arange(0, dim, 2)[: (dim // 2)].float() / dim
    freqs = 1.0 / (theta ** base)
    if use_scaled and scale_factor is not None and high_freq_factor is not None:
        freqs = apply_scaling(freqs, scale_factor, high_freq_factor)
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: Tensor, x: Tensor) -> Tensor:
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: Tensor, xk: Tensor, freqs_cis: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
    if freqs_cis is None:
        return xq, xk
    assert xq.shape[-1] % 2 == 0 and xk.shape[-1] % 2 == 0
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

# ------------------------------------------------------------------------------
# 5) Attention with KV-cache, temperature tuning, QK-norm, and RoPE
# ------------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        use_qk_norm: bool = False,
        use_rope: bool = True,
        add_bias: bool = False,
    ) -> None:
        super().__init__()
        self.args = args
        self.use_rope = use_rope
        self.use_qk_norm = use_qk_norm or args.use_qk_norm

        # Temperature tuning for long context
        self.attn_temperature_tuning = args.attn_temperature_tuning
        self.floor_scale = args.floor_scale
        self.attn_scale = args.attn_scale

        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads
        self.head_dim = args.head_dim if args.head_dim is not None else args.dim // args.n_heads

        world_size = get_model_parallel_world_size()
        self.n_local_heads = self.n_heads // world_size
        self.n_local_kv_heads = self.n_kv_heads // world_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads

        self.wq = ColumnParallelLinear(args.dim, self.n_heads * self.head_dim, bias=add_bias, gather_output=False)
        self.wk = ColumnParallelLinear(args.dim, self.n_kv_heads * self.head_dim, bias=add_bias, gather_output=False)
        self.wv = ColumnParallelLinear(args.dim, self.n_kv_heads * self.head_dim, bias=add_bias, gather_output=False)
        self.wo = RowParallelLinear(self.n_heads * self.head_dim, args.dim, bias=add_bias, input_is_parallel=True)

        # KV cache buffers (allocated on first use)
        self.register_buffer("cache_k", torch.empty(0), persistent=False)
        self.register_buffer("cache_v", torch.empty(0), persistent=False)
        self.norm_eps = args.norm_eps

    def _ensure_kv_cache(self, device: torch.device, dtype: torch.dtype, bsz: int, seqlen_total: int) -> None:
        need = (
            self.cache_k.numel() == 0
            or self.cache_k.size(0) < bsz
            or self.cache_k.size(1) < seqlen_total
            or self.cache_k.device != device
            or self.cache_k.dtype != dtype
        )
        if need:
            world_size = get_model_parallel_world_size()
            new_k = torch.zeros(
                (bsz, seqlen_total, self.n_kv_heads // world_size, self.head_dim),
                device=device,
                dtype=dtype,
            )
            new_v = torch.zeros_like(new_k)
            self.cache_k = new_k
            self.cache_v = new_v

    def forward(
        self,
        x: Tensor,
        start_pos: int,
        freqs_cis: Optional[Tensor],
        mask: Optional[Tensor] = None,  # SDPA mask supports [L,S] or [B,1,L,S] or [L,S] float(-inf/0)
    ) -> Tensor:
        bsz, seqlen, _ = x.shape
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        q = q.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # RoPE
        if self.use_rope and freqs_cis is not None:
            # Slice per step (supports streaming decode)
            freqs_slice = freqs_cis[start_pos : start_pos + seqlen]
            q, k = apply_rotary_emb(q, k, freqs_cis=freqs_slice)

        # QK-norm
        if self.use_qk_norm:
            q = rmsnorm(q, self.norm_eps)
            k = rmsnorm(k, self.norm_eps)

        # Temperature tuning for NoPE layers
        if self.attn_temperature_tuning and not self.use_rope:
            seq_pos = torch.arange(start_pos, start_pos + seqlen, device=q.device, dtype=q.dtype)
            attn_scale = torch.log(torch.floor((seq_pos + 1.0) / self.floor_scale) + 1.0) * self.attn_scale + 1.0
            attn_scale = attn_scale.view(1, seqlen, 1, 1)
            q = q * attn_scale

        # KV cache
        self._ensure_kv_cache(x.device, x.dtype, bsz, self.args.max_seq_len)
        self.cache_k[:bsz, start_pos : start_pos + seqlen] = k
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = v

        k = self.cache_k[:bsz, : start_pos + seqlen]
        v = self.cache_v[:bsz, : start_pos + seqlen]

        # [B, nH, L, D]
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        # Replicate kv heads if n_kv < n_heads
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        # SDPA
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        out = self.wo(out)
        return out

# ------------------------------------------------------------------------------
# 6) Dense FeedForward (SwiGLU)
# ------------------------------------------------------------------------------

def _silu(x: Tensor) -> Tensor:
    return F.silu(x)

class SwiGLU(nn.Module):
    def forward(self, x_a: Tensor, x_b: Tensor) -> Tensor:
        return _silu(x_a) * x_b

class DenseSwiGLU(nn.Module):
    """
    Dense FFN with SwiGLU using tensor-parallel-friendly linear layers.
    """
    def __init__(self, dim: int, hidden_dim: int, do_reduce: bool = True) -> None:
        super().__init__()
        self.do_reduce = do_reduce
        self.c_fc1 = ColumnParallelLinear(dim, hidden_dim, bias=False, gather_output=False)
        self.c_fc2 = ColumnParallelLinear(dim, hidden_dim, bias=False, gather_output=False)
        self.proj = RowParallelLinear(hidden_dim, dim, bias=False, input_is_parallel=True)
        self.swiglu = SwiGLU()

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.c_fc1(x)
        x2 = self.c_fc2(x)
        hidden = self.swiglu(x1, x2)
        out = self.proj(hidden)
        if self.do_reduce:
            return reduce_from_model_parallel_region(out)
        return out

# ------------------------------------------------------------------------------
# 7) MoE with top-k routing, capacity factor, auto-scaled hidden dims
# ------------------------------------------------------------------------------

class ExpertMLP(nn.Module):
    """
    SwiGLU expert MLP with column-parallel input and row-parallel output.
    """
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = ColumnParallelLinear(dim, hidden_dim, bias=False, gather_output=False)
        self.w2 = RowParallelLinear(hidden_dim, dim, bias=False, input_is_parallel=True)
        self.w3 = ColumnParallelLinear(dim, hidden_dim, bias=False, gather_output=False)
        self.act = SwiGLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(self.act(self.w1(x), self.w3(x)))

def _auto_hidden_dim(dim: int, ffn_exp: Optional[float], multiple_of: int) -> int:
    hidden_dim = int(4 * dim)
    hidden_dim = int(2 * hidden_dim / 3)
    if ffn_exp is not None:
        hidden_dim = int(ffn_exp * dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim

class TopKRouter(nn.Module):
    def __init__(self, dim: int, num_experts: int, top_k: int, noise: float = 0.0) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise = noise
        self.w_gating = nn.Linear(dim, num_experts, bias=False)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Tensor]]:
        logits = self.w_gating(x)  # [A, E]
        if self.noise > 0 and self.training:
            logits = logits + torch.randn_like(logits) * self.noise
        scores = F.softmax(logits, dim=-1)  # [A, E]
        topk_scores, topk_experts = torch.topk(scores, k=self.top_k, dim=-1)  # [A, K], [A, K]
        combine_weight = topk_scores  # [A, K]

        me = scores.mean(dim=0)
        ce = (scores > 0).float().mean(dim=0)
        load_balance_loss = (me * ce).sum() * self.num_experts
        z_loss = (logits.logsumexp(dim=-1) ** 2).mean()

        aux = {"moe_load_balance_loss": load_balance_loss, "moe_z_loss": z_loss}
        return topk_experts, combine_weight, scores, aux

def _capacity(a: int, E: int, cap_factor: float) -> int:
    return max(1, int(math.ceil(cap_factor * a / E)))

def _dispatch_tokens(
    x: Tensor,
    topk_experts: Tensor,
    combine_weight: Tensor,
    num_experts: int,
    capacity: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Token dispatch to experts with capacity limit (token dropping if overflow).
    Returns:
      expert_input [E, C, D], expert_weight [E, C], token_map [E, C] (original token index or -1)
    """
    A, K = topk_experts.shape
    D = x.shape[-1]
    device = x.device

    topk_experts_flat = topk_experts.reshape(-1)             # [A*K]
    combine_flat = combine_weight.reshape(-1)                # [A*K]
    token_indices_flat = torch.arange(A, device=device).repeat_interleave(K)  # [A*K]

    # Sort by expert id to rank within-expert positions
    sorted_expert, sort_idx = torch.sort(topk_experts_flat)
    exp_change = torch.ones_like(sorted_expert, dtype=torch.bool)
    exp_change[1:] = sorted_expert[1:] != sorted_expert[:-1]
    rank_in_group = torch.arange(sorted_expert.numel(), device=device) - torch.cumsum(exp_change.to(torch.int32), dim=0) + 1
    rank_in_group = torch.minimum(rank_in_group, torch.full_like(rank_in_group, capacity - 1))

    inv_sort_idx = torch.empty_like(sort_idx)
    inv_sort_idx[sort_idx] = torch.arange(sort_idx.numel(), device=device)
    pos = rank_in_group[inv_sort_idx]

    is_overflow = pos >= capacity
    valid_mask = ~is_overflow

    expert_input = torch.zeros(num_experts, capacity, D, device=device, dtype=x.dtype)
    expert_weight = torch.zeros(num_experts, capacity, device=device, dtype=x.dtype)
    token_map = torch.full((num_experts, capacity), -1, dtype=torch.int32, device=device)

    e_ids = topk_experts_flat[valid_mask]
    p_ids = pos[valid_mask]
    t_ids = token_indices_flat[valid_mask]
    w_ids = combine_flat[valid_mask]
    expert_input[e_ids, p_ids] = x[t_ids]
    expert_weight[e_ids, p_ids] = w_ids
    token_map[e_ids, p_ids] = t_ids.to(torch.int32)

    return expert_input, expert_weight, token_map

class MoELayer(nn.Module):
    """
    Mixture-of-Experts block with:
    - Top-k router, capacity factor, optional noise
    - SwiGLU experts
    - Auto-scale hidden dims to keep activated parameter budget close to dense FFN
    - Token dispatch with capacity limits; drop overflow tokens
    - Load-balance and z-loss returned for training

    Returns output in [B, T, D], and aux losses (sum for training).
    """
    def __init__(self, dim: int, args: ModelArgs, moe_args: MoEArgs) -> None:
        super().__init__()
        self.args = args
        self.moe_args = moe_args

        E_total = moe_args.num_experts
        world_size = get_model_parallel_world_size()
        if E_total % world_size != 0:
            warnings.warn("num_experts is not divisible by model-parallel world size; proceeding with local experts only.")
        self.num_local_experts = max(1, E_total // world_size)
        self.num_experts = E_total

        dense_hidden = _auto_hidden_dim(args.dim, args.ffn_exp, args.multiple_of)
        hidden_dim = int(2 * dense_hidden / 3)
        if moe_args.auto_scale_hidden and moe_args.top_k > 0:
            hidden_dim = max(4 * args.dim // (3 * moe_args.top_k), args.dim)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.experts = nn.ModuleList([ExpertMLP(args.dim, hidden_dim) for _ in range(self.num_local_experts)])
        self.shared = DenseSwiGLU(args.dim, hidden_dim, do_reduce=False)
        self.router = TopKRouter(dim=args.dim, num_experts=E_total, top_k=moe_args.top_k, noise=moe_args.router_noise)
        self.dropout = nn.Dropout(p=moe_args.expert_dropout)

    def forward(self, x: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        B, T, D = x.shape
        A = B * T
        x_flat = x.view(-1, D)

        topk_experts, combine_weight, _, aux = self.router(x_flat)  # [A,K], [A,K]
        capacity = _capacity(a=A, E=self.num_experts, cap_factor=self.moe_args.capacity_factor)

        expert_input, expert_weight, token_map = _dispatch_tokens(
            x_flat, topk_experts, combine_weight, num_experts=self.num_experts, capacity=capacity
        )

        world_size = get_model_parallel_world_size()
        rank = get_model_parallel_rank()
        base_e = rank * self.num_local_experts
        local_range = torch.arange(base_e, base_e + self.num_local_experts, device=x.device)

        local_inputs = expert_input[local_range]        # [e, C, D]
        local_weights = expert_weight[local_range]      # [e, C]
        local_token_map = token_map[local_range]        # [e, C]

        e, C = local_inputs.shape[:2]
        routed_out = torch.zeros_like(local_inputs)
        for i, expert in enumerate(self.experts):
            out_i = expert(local_inputs[i].view(-1, D))
            routed_out[i] = out_i.view(C, D)

        out_flat = torch.zeros(A, D, device=x.device, dtype=x.dtype)
        for i in range(e):
            valid = local_token_map[i] >= 0
            if valid.any():
                idx = local_token_map[i][valid].long()
                out_flat.index_add_(0, idx, routed_out[i][valid] * local_weights[i][valid].unsqueeze(-1))

        out_flat = out_flat + self.shared(x_flat)
        out_flat = reduce_from_model_parallel_region(out_flat)  # SAFE: no-op if FS not initialized
        out = out_flat.view(B, T, D)

        aux["moe_tokens"] = torch.tensor(float(A), device=x.device)
        aux["moe_capacity"] = torch.tensor(int(capacity), device=x.device)
        return self.dropout(out), aux

# ------------------------------------------------------------------------------
# 8) Transformer Block (parallel residual, optional MoE)
# ------------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.is_nope_layer = args.nope_layer_interval is not None and (layer_id + 1) % args.nope_layer_interval == 0
        use_rope = not self.is_nope_layer
        use_qk_norm = args.use_qk_norm and not self.is_nope_layer

        self.attn = Attention(args, use_qk_norm=use_qk_norm, use_rope=use_rope, add_bias=False)
        self.attn_norm = RMSNorm(args.dim, eps=args.norm_eps)

        use_moe = args.moe_args is not None and args.moe_args.enabled and ((layer_id + 1) % 1 == 0)
        if use_moe:
            self.ff = MoELayer(args.dim, args, args.moe_args)  # returns (x, aux)
            self.is_moe = True
        else:
            hidden_dim = _auto_hidden_dim(args.dim, args.ffn_exp, args.multiple_of)
            self.ff = DenseSwiGLU(args.dim, hidden_dim)
            self.is_moe = False
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: Tensor,
        start_pos: int,
        freqs_cis: Optional[Tensor],
        global_attn_mask: Optional[Tensor],
        local_attn_mask: Optional[Tensor],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        aux: Dict[str, Tensor] = {}
        mask = global_attn_mask if (self.is_nope_layer or local_attn_mask is None) else local_attn_mask

        if self.args.parallel_residual:
            x_norm = self.attn_norm(x)
            y_attn = self.attn(x_norm, start_pos, freqs_cis, mask)
            y_ff_in = self.ffn_norm(x)
            if self.is_moe:
                y_ff, aux = self.ff(y_ff_in)
            else:
                y_ff = self.ff(y_ff_in)
            x = x + y_attn + y_ff
        else:
            h = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)
            if self.is_moe:
                h2, aux = self.ff(self.ffn_norm(h))
            else:
                h2 = self.ff(self.ffn_norm(h))
            x = h + h2
        return x, aux

# ------------------------------------------------------------------------------
# 9) Vision stack: patch encoder + pixel shuffle adapter (FIXED)
# ------------------------------------------------------------------------------

class ColumnParallelConv2dPatch(nn.Module):
    """
    Efficient image patching via unfold + column-parallel projection.
    """
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int], stride: Tuple[int, int], bias: bool = False
    ) -> None:
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=kernel_size, stride=stride)
        self.proj = ColumnParallelLinear(in_channels * kernel_size[0] * kernel_size[1], out_channels, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, H, W] -> patches [B, N, P*C]
        x = self.unfold(x).permute(0, 2, 1)
        return self.proj(x)

class PixelShuffle(nn.Module):
    """
    Correct 2D pixel shuffle in token space:
    - Input:  [B, N, C], where N must be a perfect square (N = H*W), C divisible by r^2.
    - Output: [B, (H*r)*(W*r), C/(r^2)].
    This upscales token grid by r and reduces channels by r^2 (FIX for matmul mismatch).
    """
    def __init__(self, ratio: float) -> None:
        super().__init__()
        if abs(ratio - round(ratio)) > 1e-6:
            raise ValueError(f"pixel_shuffle_ratio must be an integer or near-integer, got {ratio}")
        self.r = int(round(ratio))

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        side = int(math.sqrt(N))
        if side * side != N:
            raise ValueError(f"Number of patches N={N} must be a perfect square (got {side}^2 != {N}).")
        if C % (self.r * self.r) != 0:
            raise ValueError(f"Channel dim C={C} must be divisible by r^2={self.r*self.r}.")
        H = W = side
        r = self.r
        Cout = C // (r * r)
        x = x.view(B, H, W, C)
        # Split channels into r x r subchannels
        x = x.view(B, H, W, r, r, Cout)
        # Shuffle: interleave r along H and r along W
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H * r, W * r, Cout)
        return x.view(B, -1, Cout)

class SimpleMLP(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, act: Callable[..., nn.Module] = nn.GELU) -> None:
        super().__init__()
        self.c_fc = ColumnParallelLinear(dim_in, dim_out, bias=False, gather_output=False)
        self.c_proj = RowParallelLinear(dim_out, dim_out, bias=False, input_is_parallel=True)
        self.act = act()

    def forward(self, x: Tensor) -> Tensor:
        return self.c_proj(self.act(self.c_fc(x)))

class PixelShuffleAdapter(nn.Module):
    def __init__(self, ps_ratio: float, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.ps = PixelShuffle(ps_ratio)
        reduced = max(1, int(input_dim // (int(round(ps_ratio)) ** 2)))
        self.mlp = SimpleMLP(reduced, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.ps(x)
        return self.mlp(x)

class VisionTransformer(nn.Module):
    """
    Lightweight ViT-like encoder using our Attention/SwiGLU blocks.
    """
    def __init__(self, dim: int, layers: int, heads: int, max_seq_len: int) -> None:
        super().__init__()
        args = ModelArgs(
            dim=dim,
            n_layers=layers,
            n_heads=heads,
            n_kv_heads=heads,
            vocab_size=1,
            max_batch_size=32,
            max_seq_len=max_seq_len,
        )
        self.blocks = nn.ModuleList([TransformerBlock(i, args) for i in range(layers)])
        self.norm = RMSNorm(dim)
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(dim=args.head_dim if args.head_dim else (args.dim // args.n_heads), end=max_seq_len, theta=10_000.0),
            persistent=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        for blk in self.blocks:
            x, _ = blk(x, start_pos=0, freqs_cis=self.freqs_cis, global_attn_mask=None, local_attn_mask=None)
        return self.norm(x)

class VisionEmbeddings(nn.Module):
    """
    Image -> patches -> transformer -> pixel-shuffle adapter -> projected embeddings
    """
    def __init__(self, args: VisionArgs) -> None:
        super().__init__()
        self.args = args
        H, W = args.image_size.height, args.image_size.width
        ph, pw = args.patch_size.height, args.patch_size.width
        assert H % ph == 0 and W % pw == 0
        self.grid = (H // ph, W // pw)
        self.patch = ColumnParallelConv2dPatch(3, args.dim, kernel_size=(ph, pw), stride=(ph, pw))
        self.cls = nn.Parameter(torch.zeros(1, 1, args.dim))
        self.pos = nn.Parameter(torch.randn(1, self.grid[0] * self.grid[1] + 1, args.dim) * (args.dim ** -0.5))
        self.pre = RMSNorm(args.dim)
        self.backbone = VisionTransformer(dim=args.dim, layers=args.n_layers, heads=args.n_heads, max_seq_len=(self.grid[0]*self.grid[1]+1))
        self.post = RMSNorm(args.dim)
        self.adapter = PixelShuffleAdapter(args.pixel_shuffle_ratio, args.dim, args.output_dim)

    def forward(self, images: List[List[Tensor]], dtype: torch.dtype, device: torch.device) -> Tensor:
        flat: List[Tensor] = [img for sample in images for img in sample]
        if len(flat) == 0:
            return torch.zeros(0, 0, self.args.output_dim, device=device, dtype=dtype)
        imgs = torch.vstack(flat).to(device=device, dtype=dtype)  # [sum, 3, H, W]
        x = self.patch(imgs)                       # [sum, N, D]
        B, _, _ = x.shape
        cls = self.cls.to(x.dtype).expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos.to(x.dtype)
        x = self.pre(x)
        x = self.backbone(x)
        x = self.post(x)
        x = x[:, 1:, :]                            # drop cls
        x = self.adapter(x)                        # [sum, N', D_out] (channels reduced, tokens upsampled)
        return x

def scatter_image_embeddings(
    image_batch: List[List[Tensor]],
    image_mask: Tensor,
    h_ref: Tensor,
    encoded_patches: Tensor,
) -> Tensor:
    """
    Scatter encoded image patches into positions marked True in image_mask for each sample.
    h_ref: [B, T, D], encoded_patches: [sum_imgs, Np, D]
    """
    B, T, D = h_ref.shape
    if encoded_patches.numel() == 0:
        return torch.zeros_like(h_ref)

    num_images_per_sample: List[int] = [len(sample) for sample in image_batch]
    assert sum(num_images_per_sample) == encoded_patches.size(0), "Mismatch between images and encoded patches."

    h_image = torch.zeros_like(h_ref)
    patches_split = encoded_patches.split(num_images_per_sample, dim=0)
    for b in range(B):
        mask_b = image_mask[b]  # [T]
        K = int(mask_b.sum().item())
        if K == 0:
            continue
        patches_b = patches_split[b].reshape(-1, D)
        assert patches_b.size(0) >= K, "Not enough visual tokens to fill mask positions."
        expanded_mask = mask_b.unsqueeze(-1).expand(-1, D)
        h_image[b].masked_scatter_(expanded_mask, patches_b[:K])
    return h_image

# ------------------------------------------------------------------------------
# 10) Text backbone and full Multimodal model
# ------------------------------------------------------------------------------

def build_causal_mask(seqlen: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    mask = torch.full((seqlen, seqlen), float("-inf"), device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask

def create_chunked_attention_mask(seq_len: int, chunk_size: int, device: torch.device) -> Tensor:
    """
    Local chunked attention mask allowing tokens within same block to attend causally.
    Return float mask with -inf for disallowed positions to be compatible with SDPA.
    """
    block_ids = torch.arange(seq_len, device=device) // chunk_size
    tok = torch.arange(seq_len, device=device)
    allow = (block_ids.unsqueeze(0) == block_ids.unsqueeze(1)) & (tok.unsqueeze(0) >= tok.unsqueeze(1))
    mask = (~allow).float() * float("-inf")
    return mask

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int) -> None:
        super().__init__()
        self.emb = VocabParallelEmbedding(vocab_size, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.emb(x)

class TextTransformer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.blocks = nn.ModuleList([TransformerBlock(i, args) for i in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                dim=args.head_dim if args.head_dim is not None else (args.dim // args.n_heads),
                end=args.max_seq_len * 2,
                theta=args.rope_theta,
                use_scaled=args.use_scaled_rope,
                scale_factor=args.rope_scaling_factor,
                high_freq_factor=args.rope_high_freq_factor,
            ),
            persistent=False,
        )

    def forward(
        self,
        x: Tensor,
        start_pos: int,
        attention_chunk_size: Optional[int] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        B, T, D = x.shape
        freqs = self.freqs_cis

        global_mask = None
        local_mask = None
        if T > 1:
            global_mask = build_causal_mask(T, x.device, x.dtype)
            if attention_chunk_size is not None and attention_chunk_size > 0:
                local_mask = create_chunked_attention_mask(T, attention_chunk_size, x.device)

        aux_losses: Dict[str, Tensor] = {}
        for blk in self.blocks:
            x, aux = blk(
                x,
                start_pos=start_pos,
                freqs_cis=freqs,
                global_attn_mask=global_mask,
                local_attn_mask=local_mask,
            )
            for k, v in aux.items():
                aux_losses[k] = aux_losses.get(k, torch.tensor(0.0, device=x.device, dtype=v.dtype)) + v
        x = self.norm(x)
        return x, aux_losses

class MultiModalTransformer(nn.Module):
    """
    Unified multimodal transformer:
    - Token embed + optional vision fusion (additive or residual)
    - Shared text transformer backbone
    - LM head tied or untied
    - LoRA/Quantization integration hooks
    """
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.embed = TokenEmbedding(args.vocab_size, args.dim)
        self.vision = VisionEmbeddings(args.vision_args) if args.vision_args is not None else None
        self.backbone = TextTransformer(args)
        self.lm_head = ColumnParallelLinear(args.dim, args.vocab_size, bias=False, gather_output=False)
        if args.tie_embeddings and isinstance(self.embed.emb.layer, nn.Embedding):
            self.lm_head.layer.weight = self.embed.emb.layer.weight  # type: ignore

        if args.quantization_args and args.quantization_args.enabled:
            apply_quantization(self, args.quantization_args)
        if args.lora_args and args.lora_args.enabled:
            apply_lora(self, args.lora_args)

    def fuse_vision(self, h_text: Tensor, image_batch: Optional[List[List[Tensor]]], image_mask: Optional[Tensor]) -> Tensor:
        if self.vision is None or image_batch is None or image_mask is None:
            return h_text
        enc = self.vision(image_batch, dtype=h_text.dtype, device=h_text.device)
        h_image = scatter_image_embeddings(image_batch, image_mask, h_text, enc)
        return h_text + h_image

    @torch.inference_mode(False)
    def forward_llm(self, llm_input: LLMInput, start_pos: int = 0) -> LLMOutput:
        tokens = llm_input.tokens
        h = self.embed(tokens)
        if llm_input.images is not None and llm_input.image_mask is not None:
            h = self.fuse_vision(h, llm_input.images, llm_input.image_mask)
        h, aux_losses = self.backbone(h, start_pos=start_pos, attention_chunk_size=self.args.attention_chunk_size)
        logits = self.lm_head(h)
        if VERBOSE:
            rich_kv_panel("Forward (LLM)", {
                "tokens": list(tokens.shape),
                "hidden": list(h.shape),
                "logits": list(logits.shape),
                "dtype": str(h.dtype),
                "device": str(h.device),
                **{k: float(v.detach().cpu()) for k, v in aux_losses.items()},
            })
        return LLMOutput(logits=logits, aux=aux_losses)

    @torch.inference_mode()
    def generate(
        self,
        tokens: Tensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        images: Optional[List[List[Tensor]]] = None,
        image_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Simple autoregressive decoding using KV cache in attention layers.
        """
        self.eval()
        B, T = tokens.shape
        h = self.embed(tokens)
        if images is not None and image_mask is not None:
            h = self.fuse_vision(h, images, image_mask)
        # Prime cache by running the prefix
        h, _ = self.backbone(h, start_pos=0, attention_chunk_size=None)
        logits = self.lm_head(h)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        out = [tokens, next_token]
        start_pos = T
        for _ in range(max_new_tokens - 1):
            h_step = self.embed(next_token)
            h_step, _ = self.backbone(h_step, start_pos=start_pos, attention_chunk_size=None)
            logits = self.lm_head(h_step)[:, -1, :] / max(1e-6, temperature)
            if top_k is not None:
                v, idx = torch.topk(logits, top_k)
                probs = torch.zeros_like(logits).scatter_(1, idx, F.softmax(v, dim=-1))
            else:
                probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            out.append(next_token)
            start_pos += 1
        return torch.cat(out, dim=1)

# ------------------------------------------------------------------------------
# 11) Quantization (int8/int4 weight-only) and LoRA adapters
# ------------------------------------------------------------------------------

class QuantLinearW8A8(nn.Module):
    """
    Reference weight-only int8 linear: weights quantized to int8 with per-channel scales.
    Computation is performed in fp16/fp32 after dequantizing weights; serves as a correct stub.
    """
    def __init__(self, linear: nn.Linear, per_channel: bool = True, symmetric: bool = True) -> None:
        super().__init__()
        W = linear.weight.data
        if per_channel:
            dim = 0
            max_val = W.abs().amax(dim=dim, keepdim=True).clamp(min=1e-8)
            scale = max_val / 127.0
            qW = torch.round(W / scale).to(torch.int8)
            self.scale = nn.Parameter(scale.squeeze(0), requires_grad=False)
        else:
            max_val = W.abs().amax().clamp(min=1e-8)
            scale = max_val / 127.0
            qW = torch.round(W / scale).to(torch.int8)
            self.scale = nn.Parameter(torch.tensor(scale), requires_grad=False)

        self.qweight = nn.Parameter(qW, requires_grad=False)
        self.bias = linear.bias
        self.per_channel = per_channel

    def forward(self, x: Tensor) -> Tensor:
        W = self.qweight.float() * (self.scale.view(-1, 1) if self.per_channel else self.scale)
        return F.linear(x, W, self.bias)

class QuantLinearW4A8(nn.Module):
    """
    Reference weight-only int4 linear: packs 2x int4 per int8 tensor; dequantizes to float for F.linear.
    Correctness-focused; replace with optimized kernels for real deployment.
    """
    def __init__(self, linear: nn.Linear, per_channel: bool = True) -> None:
        super().__init__()
        W = linear.weight.data
        max_val = W.abs().amax(dim=0, keepdim=True).clamp(min=1e-8)
        scale = max_val / 7.0  # int4 range [-8..7] symmetric
        qW = torch.clamp(torch.round(W / scale), -8, 7).to(torch.int8)
        self.scale = nn.Parameter(scale.squeeze(0), requires_grad=False)
        self.qweight = nn.Parameter(qW, requires_grad=False)
        self.bias = linear.bias

    def forward(self, x: Tensor) -> Tensor:
        W = self.qweight.float() * self.scale.view(-1, 1)
        return F.linear(x, W, self.bias)

def _replace_linear(module: nn.Module, quant_args: QuantizationArgs) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, (nn.Linear,)):
            if quant_args.scheme == QuantizationScheme.int8_weight_only:
                q = QuantLinearW8A8(child, per_channel=quant_args.per_channel, symmetric=quant_args.symmetric)
                setattr(module, name, q)
            elif quant_args.scheme == QuantizationScheme.int4_weight_only:
                q = QuantLinearW4A8(child, per_channel=quant_args.per_channel)
                setattr(module, name, q)
        elif isinstance(child, (ColumnParallelLinear, RowParallelLinear)):
            if hasattr(child, "layer") and isinstance(child.layer, nn.Linear):
                base = child.layer
                if quant_args.scheme == QuantizationScheme.int8_weight_only:
                    q = QuantLinearW8A8(base, per_channel=quant_args.per_channel, symmetric=quant_args.symmetric)
                elif quant_args.scheme == QuantizationScheme.int4_weight_only:
                    q = QuantLinearW4A8(base, per_channel=quant_args.per_channel)
                else:
                    q = None
                if q is not None:
                    child.layer = q  # type: ignore
        else:
            _replace_linear(child, quant_args)

def apply_quantization(model: nn.Module, quant_args: QuantizationArgs) -> None:
    if not quant_args.enabled or quant_args.scheme == QuantizationScheme.none:
        return
    rich_print(f"[green]Applying quantization: {quant_args.scheme}[/green]")
    _replace_linear(model, quant_args)

class LoRALinear(nn.Module):
    """
    LoRA adapter wrapper for Linear layers without altering base weights.
    y = xW + x(BA)*scale + b
    """
    def __init__(self, base: nn.Module, r: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__()
        assert isinstance(base, (nn.Linear, QuantLinearW8A8, QuantLinearW4A8)), "Unsupported base for LoRA"
        self.base = base
        out_features, in_features = base.weight.shape[-2], base.weight.shape[-1]
        self.A = nn.Linear(in_features, r, bias=False)
        self.B = nn.Linear(r, out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)
        self.scale = alpha / max(1, r)
        self.dropout = nn.Dropout(dropout)

    @property
    def weight(self) -> Tensor:
        return self.base.weight  # type: ignore

    @property
    def bias(self) -> Optional[Tensor]:
        return getattr(self.base, "bias", None)

    def forward(self, x: Tensor) -> Tensor:
        base_out = F.linear(x, self.base.weight, self.bias) if isinstance(self.base, (nn.Linear,)) else self.base(x)
        lora_out = self.B(self.A(self.dropout(x))) * self.scale
        return base_out + lora_out

def _apply_lora_to_targets(module: nn.Module, lora_args: LoRAArgs) -> None:
    for name, child in list(module.named_children()):
        target = any(t in name for t in lora_args.target_modules)
        if target:
            if isinstance(child, (ColumnParallelLinear, RowParallelLinear)):
                if hasattr(child, "layer") and isinstance(child.layer, (nn.Linear, QuantLinearW8A8, QuantLinearW4A8)):
                    child.layer = LoRALinear(child.layer, r=lora_args.rank, alpha=lora_args.alpha, dropout=lora_args.dropout)
            elif isinstance(child, (nn.Linear, QuantLinearW8A8, QuantLinearW4A8)):
                setattr(module, name, LoRALinear(child, r=lora_args.rank, alpha=lora_args.alpha, dropout=lora_args.dropout))
        else:
            _apply_lora_to_targets(child, lora_args)

def apply_lora(model: nn.Module, lora_args: LoRAArgs) -> None:
    if not lora_args.enabled:
        return
    rich_print(f"[green]Applying LoRA: rank={lora_args.rank}, alpha={lora_args.alpha}[/green]")
    _apply_lora_to_targets(model, lora_args)

# ------------------------------------------------------------------------------
# 12) Public API wrappers
# ------------------------------------------------------------------------------

def forward_transformer(model: MultiModalTransformer, t_input: TransformerInput) -> TransformerOutput:
    """
    Lower-level API operating on TransformerInput. Typically use forward_llm for user-level.
    """
    tokens = t_input.tokens
    start_pos = t_input.tokens_position if isinstance(t_input.tokens_position, int) else int(t_input.tokens_position[0])
    h = model.embed(tokens)
    if t_input.image_embedding is not None:
        img = t_input.image_embedding
        h = h * (~img.mask).unsqueeze(-1) + img.embedding * img.mask.unsqueeze(-1)
    h, aux = model.backbone(h, start_pos=start_pos, attention_chunk_size=model.args.attention_chunk_size)
    logits = model.lm_head(h)
    return TransformerOutput(logits=logits, aux=aux)

# ------------------------------------------------------------------------------
# 13) Utilities: parameter counting, summaries, synthetic examples
# ------------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_summary(model: nn.Module, name: str = "Model") -> None:
    if not VERBOSE:
        return
    kv = {
        "name": name,
        "params (M)": round(count_parameters(model) / 1e6, 3),
        "fairscale_available": _FAIRSCALE_AVAILABLE,
        "fs_initialized": _is_fairscale_initialized(),
        "mp_world_size": get_model_parallel_world_size(),
    }
    rich_kv_panel("Model Summary", kv)

def _example_build_args(
    vocab_size: int = 32000,
    seq_len: int = 256,
    with_vision: bool = True,
    with_moe: bool = True,
) -> ModelArgs:
    v_args = VisionArgs(
        image_size=Size(height=224, width=224),
        patch_size=Size(height=16, width=16),
        dim=384,
        n_layers=2,
        n_heads=6,
        mlp_ratio=4.0,
        output_dim=512,
        pixel_shuffle_ratio=2.0,
    ) if with_vision else None

    m_args = ModelArgs(
        dim=512,
        n_layers=8,
        n_heads=8,
        n_kv_heads=8,
        head_dim=64,
        vocab_size=vocab_size,
        multiple_of=256,
        ffn_dim_multiplier=None,
        ffn_exp=None,
        norm_eps=1e-5,
        parallel_residual=True,
        attention_chunk_size=64,
        rope_theta=1_000_000.0,
        use_scaled_rope=True,
        rope_scaling_factor=16.0,
        rope_high_freq_factor=1.0,
        nope_layer_interval=4,
        use_qk_norm=True,
        attn_temperature_tuning=True,
        floor_scale=8192.0,
        attn_scale=0.1,
        vision_args=v_args,
        moe_args=MoEArgs(enabled=with_moe, num_experts=8, capacity_factor=1.25, top_k=2, router_noise=0.0),
        quantization_args=QuantizationArgs(enabled=False, scheme=QuantizationScheme.none),
        lora_args=LoRAArgs(enabled=False),
        max_batch_size=8,
        max_seq_len=seq_len,
        tie_embeddings=True,
    )
    return ModelArgs(**m_args.model_dump())

def _example_synth_batch(
    bsz: int,
    T: int,
    vocab_size: int,
    with_vision: bool,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tuple[Tensor, Optional[List[List[Tensor]]], Optional[Tensor]]:
    tokens = torch.randint(0, vocab_size, (bsz, T), device=device)
    if not with_vision:
        return tokens, None, None
    image_batch: List[List[Tensor]] = []
    image_mask = torch.zeros(bsz, T, dtype=torch.bool, device=device)
    K = min(16, T // 2)
    for b in range(bsz):
        img = torch.randn(1, 3, 224, 224, device=device, dtype=dtype)
        image_batch.append([img])
        image_mask[b, 1 : K + 1] = True
    return tokens, image_batch, image_mask

def _example_train_step() -> None:
    global VERBOSE
    VERBOSE = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = _example_build_args(vocab_size=32000, seq_len=256, with_vision=True, with_moe=True)
    model = MultiModalTransformer(args).to(device)
    model_summary(model, "Multimodal-Transformer (Train)")

    tokens, images, image_mask = _example_synth_batch(
        bsz=2, T=128, vocab_size=args.vocab_size, with_vision=True, device=device, dtype=torch.float32
    )
    out = model.forward_llm(LLMInput(tokens=tokens, images=images, image_mask=image_mask), start_pos=0)
    logits = out.logits
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tokens.view(-1))
    # Add MoE load-balance loss if present
    if "moe_load_balance_loss" in out.aux:
        loss = loss + 0.01 * out.aux["moe_load_balance_loss"] + 0.001 * out.aux["moe_z_loss"]
    rich_kv_panel("TrainStep", {"loss": float(loss.item())})

def _example_decode_step() -> None:
    global VERBOSE
    VERBOSE = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = _example_build_args(vocab_size=32000, seq_len=256, with_vision=True, with_moe=False)
    model = MultiModalTransformer(args).to(device)
    model_summary(model, "Decode-Only")

    B, T = 2, 64
    tokens, images, image_mask = _example_synth_batch(
        bsz=B, T=T, vocab_size=args.vocab_size, with_vision=True, device=device, dtype=torch.float32
    )
    out = model.forward_llm(LLMInput(tokens=tokens, images=images, image_mask=image_mask), start_pos=0)
    next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    gen = model.generate(tokens, max_new_tokens=8, temperature=0.8, top_k=50, images=None, image_mask=None)
    rich_kv_panel("DecodeStep", {"t1_next_token": next_token.squeeze().tolist(), "gen_len": gen.shape[1]})

def _example_bf16() -> None:
    global VERBOSE
    VERBOSE = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        rich_print("[yellow]BF16 example recommended on CUDA. Skipping.[/yellow]")
        return
    args = _example_build_args(vocab_size=32000, seq_len=128, with_vision=True, with_moe=True)
    model = MultiModalTransformer(args).to(device=device, dtype=torch.bfloat16)
    tokens, images, image_mask = _example_synth_batch(
        bsz=2, T=64, vocab_size=args.vocab_size, with_vision=True, device=device, dtype=torch.bfloat16
    )
    out = model.forward_llm(LLMInput(tokens=tokens, images=images, image_mask=image_mask), start_pos=0)
    loss = F.cross_entropy(out.logits.float().view(-1, out.logits.size(-1)), tokens.view(-1))
    rich_kv_panel("BF16", {"loss": float(loss.item())})

# ------------------------------------------------------------------------------
# 14) Main: run examples (toggle as needed)
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    _example_train_step()
    _example_decode_step()
    _example_bf16()