#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modular Multimodal Transformer (Vision + Text) with:
- Vision encoder using parallelized Conv2D patching and pixel-shuffle adapters
- Rotary Position Embeddings (RoPE) with optional scaling and robust complex-cis handling
- Cache-based attention with temperature tuning for long contexts
- RMSNorm, MLP projections, and FairScale model parallel-compatible layers (safe fallback)
- Dynamic positional embeddings for vision streams
- Scalable transformer blocks optimized for large-batch training and efficient inference
- Extensible hooks for normalization, quantization, and LoRA/MoE integration
- Verbose meta logging via rich (opt-in)

Robustness fixes implemented:
- FairScale safety: if model-parallel group is not initialized, we fall back to nn.Linear and world_size=1
- Dtype consistency: attention temperature tuning now preserves dtype; RoPE complex cis re-validated per forward
- Vision scatter: mask expansion fixed to [T, D] to avoid shape mismatches in masked_scatter_
"""

from __future__ import annotations

import math
import os
import types
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# ------------------------------------------------------------------------------
# 1) Compatibility: FairScale + Rich
# ------------------------------------------------------------------------------

try:
    import fairscale.nn.model_parallel.initialize as fs_init  # type: ignore
    from fairscale.nn.model_parallel.layers import ColumnParallelLinear as FSColumnParallelLinear  # type: ignore
    from fairscale.nn.model_parallel.layers import RowParallelLinear as FSRowParallelLinear  # type: ignore
    _FAIRSCALE_IMPORTED = True
except Exception:
    _FAIRSCALE_IMPORTED = False
    fs_init = types.SimpleNamespace(  # type: ignore
        get_model_parallel_world_size=lambda: 1,
        get_model_parallel_rank=lambda: 0,
    )
    FSColumnParallelLinear = None  # type: ignore
    FSRowParallelLinear = None  # type: ignore

def _fs_model_parallel_ready() -> bool:
    if not _FAIRSCALE_IMPORTED:
        return False
    try:
        _ = fs_init.get_model_parallel_world_size()
        return True
    except Exception:
        return False

def mp_world_size() -> int:
    if not _FAIRSCALE_IMPORTED:
        return 1
    try:
        return fs_init.get_model_parallel_world_size()
    except Exception:
        return 1

def mp_rank() -> int:
    if not _FAIRSCALE_IMPORTED:
        return 0
    try:
        return fs_init.get_model_parallel_rank()
    except Exception:
        return 0

_FS_READY = _fs_model_parallel_ready()

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    _RICH_AVAILABLE = True
    _CONSOLE = Console()
except Exception:
    _RICH_AVAILABLE = False
    _CONSOLE = None

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

# ------------------------------------------------------------------------------
# 1a) Rich logger (opt-in via MULTIMODAL_VERBOSE=1)
# ------------------------------------------------------------------------------

def _bool_env(key: str, default: bool = False) -> bool:
    v = os.environ.get(key, None)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "y", "on")

VERBOSE = _bool_env("MULTIMODAL_VERBOSE", False)

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
# 1b) FairScale-compatible shims
# ------------------------------------------------------------------------------

class ColumnParallelLinear(nn.Module):
    """
    FairScale-compatible shim. Uses FairScale when imported AND MP group initialized, otherwise nn.Linear.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = False,
        init_method: Optional[Callable] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if _FAIRSCALE_IMPORTED and _FS_READY:
            self.layer = FSColumnParallelLinear(
                in_features, out_features, bias=bias, gather_output=gather_output, init_method=init_method, **kwargs
            )
            self.is_fs = True
        else:
            self.layer = nn.Linear(in_features, out_features, bias=bias)
            if init_method:
                with torch.no_grad():
                    init_method(self.layer.weight)
            self.is_fs = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)

class RowParallelLinear(nn.Module):
    """
    FairScale-compatible shim. Uses FairScale when imported AND MP group initialized, otherwise nn.Linear.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_is_parallel: bool = False,
        init_method: Optional[Callable] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if _FAIRSCALE_IMPORTED and _FS_READY:
            self.layer = FSRowParallelLinear(
                in_features, out_features, bias=bias, input_is_parallel=input_is_parallel, init_method=init_method, **kwargs
            )
            self.is_fs = True
        else:
            self.layer = nn.Linear(in_features, out_features, bias=bias)
            if init_method:
                with torch.no_grad():
                    init_method(self.layer.weight)
            self.is_fs = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)

# ------------------------------------------------------------------------------
# 2) Pydantic Args
# ------------------------------------------------------------------------------

try:
    from pydantic import BaseModel, model_validator, Field
except Exception as e:
    raise ImportError("pydantic is required for args validation. Please `pip install pydantic`.") from e

class Size(BaseModel):
    height: int = Field(..., ge=1)
    width: int = Field(..., ge=1)

    @model_validator(mode="after")
    def check_square(self) -> "Size":
        # Non-square allowed; kept for future constraints
        return self

class LoRAArgs(BaseModel):
    enabled: bool = False
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: List[str] = Field(default_factory=lambda: ["wq", "wk", "wv", "wo", "c_fc", "c_proj"])

class QuantizationArgs(BaseModel):
    enabled: bool = False
    dtype: Optional[str] = None
    per_channel: bool = True
    symmetric: bool = True

class MoEArgs(BaseModel):
    enabled: bool = False
    num_experts: int = 0
    capacity_factor: float = 1.0
    top_k: int = 1

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
        assert self.pixel_shuffle_ratio >= 1.0, "pixel_shuffle_ratio must be >= 1.0"
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

    # Attention + RoPE
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

        if self.attention_chunk_size is not None:
            assert self.attention_chunk_size > 0, "attention_chunk_size must be positive"

        if self.vision_args is not None:
            assert self.vision_args.output_dim == self.dim, "vision.output_dim should match text dim for fusion"
        return self

# ------------------------------------------------------------------------------
# 3) Core building blocks: RMSNorm + RoPE
# ------------------------------------------------------------------------------

def rmsnorm(x: torch.Tensor, eps: float) -> torch.Tensor:
    def _norm(y: torch.Tensor) -> torch.Tensor:
        return y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
    return _norm(x.float()).type_as(x)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm(x, self.eps) * self.weight

def apply_scaling(freqs: torch.Tensor, scale_factor: float, high_freq_factor: float) -> torch.Tensor:
    low_freq_factor = 1.0
    old_context_len = 8192.0
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs: List[torch.Tensor] = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)

def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float,
    use_scaled: bool = False,
    scale_factor: Optional[float] = None,
    high_freq_factor: Optional[float] = None,
) -> torch.Tensor:
    base = torch.arange(0, dim, 2)[: (dim // 2)].float() / dim
    freqs = 1.0 / (theta ** base)  # [dim/2]
    if use_scaled and scale_factor is not None and high_freq_factor is not None:
        freqs = apply_scaling(freqs, scale_factor, high_freq_factor)
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)  # [end]
    freqs = torch.outer(t, freqs)  # [end, dim/2]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64 [end, dim/2]
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
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
# 4) Vision pipeline: Conv2D patching + PixelShuffle adapter + MLP
# ------------------------------------------------------------------------------

class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

class ColumnParallelConv2dPatch(nn.Module):
    """
    Unfold-based patch embedding followed by ColumnParallelLinear.
    Input:  (B, C, H, W)
    Output: (B, N, D)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]],
        bias: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        self._unfold = nn.Unfold(kernel_size=kernel_size, stride=stride)
        self._linear = ColumnParallelLinear(
            in_channels * kernel_size[0] * kernel_size[1], out_channels, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._unfold(x).permute(0, 2, 1)
        x = self._linear(x)
        return x

def pixel_shuffle_op(input_x: torch.Tensor, ps_ratio: float) -> torch.Tensor:
    n, w, h, c = input_x.size()
    input_x = input_x.view(n, w, int(h * ps_ratio), int(c / ps_ratio))
    input_x = input_x.permute(0, 2, 1, 3).contiguous()
    input_x = input_x.view(
        n,
        int(h * ps_ratio),
        int(w * ps_ratio),
        int(c / (ps_ratio * ps_ratio)),
    )
    input_x = input_x.permute(0, 2, 1, 3).contiguous()
    return input_x

class PixelShuffle(nn.Module):
    def __init__(self, ps_ratio: float) -> None:
        super().__init__()
        self.ps_ratio = ps_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3
        hh = ww = int(math.sqrt(x.shape[1]))
        x = x.reshape(x.shape[0], hh, ww, -1)
        x = pixel_shuffle_op(x, ps_ratio=self.ps_ratio)
        return x.reshape(x.shape[0], -1, x.shape[-1])

class SimpleMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        bias: bool = True,
        dropout: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.c_fc = ColumnParallelLinear(dim, hidden_dim, bias=bias, gather_output=False)
        self.c_proj = RowParallelLinear(hidden_dim, hidden_dim, bias=bias, input_is_parallel=True)
        self.non_linearity = act_layer()
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.c_fc(x)
        hidden = self.non_linearity(hidden)
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return self.non_linearity(self.c_proj(hidden))

class PixelShuffleMLP(nn.Module):
    def __init__(
        self,
        ps_ratio: float,
        input_dim: int,
        output_dim: int,
        add_fc: bool = False,
    ) -> None:
        super().__init__()
        self.pixel_shuffle = PixelShuffle(ps_ratio)
        self.mlp = SimpleMLP(
            int(input_dim // (ps_ratio**2)),
            output_dim,
            bias=False,
            dropout=0.0,
            act_layer=nn.GELU,
        )
        self.fc = nn.Identity()
        if add_fc:
            self.fc = ColumnParallelLinear(output_dim, output_dim, bias=False)

    def forward(self, encoded_patches: torch.Tensor) -> torch.Tensor:
        encoded_patches = self.pixel_shuffle(encoded_patches)
        return self.fc(self.mlp(encoded_patches))

# ------------------------------------------------------------------------------
# 5) Attention with KV-cache, temperature tuning, QK-norm, and RoPE
# ------------------------------------------------------------------------------

class Attention(nn.Module):
    """
    - FairScale Column/Row-parallel projections (safe fallback)
    - KV cache for autoregressive decoding
    - Optional QK RMSNorm
    - RoPE support with precomputed complex cis
    - Temperature tuning for NoPE in very long contexts
    """
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

        self.attn_temperature_tuning = args.attn_temperature_tuning
        self.floor_scale = args.floor_scale
        self.attn_scale = args.attn_scale

        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.head_dim = args.head_dim if args.head_dim is not None else (args.dim // args.n_heads)

        world_size = mp_world_size()
        self.n_local_heads = self.n_heads // world_size
        self.n_local_kv_heads = self.n_kv_heads // world_size
        self.n_rep = max(1, self.n_local_heads // max(1, self.n_local_kv_heads))

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=add_bias,
            gather_output=False,
            init_method=lambda w: w,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=add_bias,
            gather_output=False,
            init_method=lambda w: w,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=add_bias,
            gather_output=False,
            init_method=lambda w: w,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=add_bias,
            input_is_parallel=True,
            init_method=lambda w: w,
        )

        # KV cache buffers allocated lazily to match device/dtype
        self.register_buffer("cache_k", torch.empty(0), persistent=False)
        self.register_buffer("cache_v", torch.empty(0), persistent=False)
        self.norm_eps = args.norm_eps

        self._register_load_state_dict_pre_hook(self._load_hook)

    def _load_hook(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        key = prefix + "wqkv.weight"
        if key in state_dict:
            wqkv = state_dict.pop(key)
            d, r = divmod(wqkv.shape[0], self.n_heads + 2 * self.n_kv_heads)
            if r != 0:
                raise ValueError(f"shape={tuple(wqkv.shape)} not divisible by n_heads + 2*n_kv_heads")
            wq, wk, wv = wqkv.split([d * self.n_heads, d * self.n_kv_heads, d * self.n_kv_heads], dim=0)
            state_dict[prefix + "wq.weight"] = wq
            state_dict[prefix + "wk.weight"] = wk
            state_dict[prefix + "wv.weight"] = wv

    def _ensure_kv_cache(self, device: torch.device, dtype: torch.dtype, bsz: int, seqlen_total: int) -> None:
        need_alloc = (
            self.cache_k.numel() == 0
            or self.cache_k.size(0) < bsz
            or self.cache_k.size(1) < seqlen_total
            or self.cache_k.device != device
            or self.cache_k.dtype != dtype
        )
        if need_alloc:
            world_size = mp_world_size()
            new_k = torch.zeros(
                (bsz, seqlen_total, max(1, self.n_kv_heads // world_size), self.head_dim),
                device=device,
                dtype=dtype,
            )
            new_v = torch.zeros_like(new_k)
            self.cache_k = new_k
            self.cache_v = new_v

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: Optional[torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        if self.use_rope and freqs_cis is not None:
            freqs_slice = freqs_cis[start_pos : start_pos + seqlen]
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_slice)

        if self.use_qk_norm:
            xq = rmsnorm(xq, self.norm_eps)
            xk = rmsnorm(xk, self.norm_eps)

        if self.attn_temperature_tuning and not self.use_rope:
            seq_positions = torch.arange(start_pos, start_pos + seqlen, device=xq.device, dtype=torch.float32)
            attn_scales = torch.log(torch.floor((seq_positions + 1.0) / self.floor_scale) + 1.0) * self.attn_scale + 1.0
            attn_scales = attn_scales.view(1, seqlen, 1, 1).to(xq.dtype)  # keep dtype consistent
            xq = xq * attn_scales

        self._ensure_kv_cache(device=x.device, dtype=x.dtype, bsz=bsz, seqlen_total=self.args.max_seq_len)
        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        xk = self.cache_k[:bsz, : start_pos + seqlen]
        xv = self.cache_v[:bsz, : start_pos + seqlen]

        xq, xk, xv = [t.transpose(1, 2) for t in (xq, xk, xv)]
        if self.n_rep > 1:
            xk = xk.repeat_interleave(self.n_rep, dim=1)
            xv = xv.repeat_interleave(self.n_rep, dim=1)

        attn_out = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=mask, dropout_p=0.0)
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        out = self.wo(attn_out)
        return out

# ------------------------------------------------------------------------------
# 6) FFN + Transformer blocks
# ------------------------------------------------------------------------------

class _FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.c_fc = ColumnParallelLinear(dim, hidden_dim, bias=True, gather_output=False, init_method=lambda w: w)
        self.c_proj = RowParallelLinear(hidden_dim, dim, bias=True, input_is_parallel=True, init_method=lambda w: w)
        self.non_linearity = act_layer()
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.c_fc(x)
        hidden = self.non_linearity(hidden)
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return self.c_proj(hidden)

class TransformerBlock(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        use_rope: bool,
        gated: bool = False,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        d_model = args.dim
        self.attn = Attention(args, use_qk_norm=args.use_qk_norm, use_rope=use_rope, add_bias=True)
        self.ln_1 = LayerNorm(d_model)
        hidden_dim = int(mlp_ratio * d_model)
        if args.multiple_of > 1:
            hidden_dim = (hidden_dim + args.multiple_of - 1) // args.multiple_of * args.multiple_of
        self.mlp = _FeedForward(dim=d_model, hidden_dim=hidden_dim, dropout=0.0, act_layer=act_layer)
        self.ln_2 = LayerNorm(d_model)
        self.gated = gated
        if gated:
            self.gate_attn = nn.Parameter(torch.zeros(1))
            self.gate_ffn = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freq_cis: Optional[torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        gate_attn = 1.0 if not self.gated else self.gate_attn.tanh()
        gate_ffn = 1.0 if not self.gated else self.gate_ffn.tanh()
        x = x + gate_attn * self.attn(self.ln_1(x), start_pos=start_pos, freqs_cis=freq_cis, mask=mask)
        x = x + gate_ffn * self.mlp(self.ln_2(x))
        return x

# ------------------------------------------------------------------------------
# 7) Vision Encoder with dynamic pos-emb + RoPE
# ------------------------------------------------------------------------------

class PackingIndex:
    Z = 0
    Y = 1
    X = 2
    TIME = 3
    HEIGHT = 4
    WIDTH = 5
    IDX = 6
    BATCH_IDX = 7
    NUM_METADATA = 8
    ID_CLS_TOKEN = -2
    ID_PAD_TOKEN = -1

ENCODER_MAX_BATCH_SIZE = 32
ENCODER_MAX_SEQ_LEN = 1024

class _Transformer(nn.Module):
    """
    Lightweight transformer for VisionEncoder using the same Attention implementation.
    """
    def __init__(
        self,
        dim: int,
        layers: int,
        heads: int,
        max_batch_size: int,
        max_seq_len: int,
        mlp_ratio: float = 4.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        gated: bool = False,
    ) -> None:
        super().__init__()
        args = ModelArgs(
            dim=dim,
            n_layers=1,
            n_heads=heads,
            n_kv_heads=heads,
            vocab_size=1,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
        )
        self.resblocks = nn.ModuleList(
            [
                TransformerBlock(
                    args=args,
                    use_rope=True,
                    gated=gated,
                    act_layer=act_layer,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor, freq_cis: Optional[torch.Tensor] = None) -> torch.Tensor:
        start_pos = 0
        for r in self.resblocks:
            x = r(x, start_pos=start_pos, freq_cis=freq_cis, mask=None)
        return x

class VisionEncoder(nn.Module):
    def __init__(
        self,
        image_size: Tuple[int, int],
        patch_size: Tuple[int, int],
        dim: int,
        layers: int,
        heads: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = (self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1])
        self._dim = dim
        self._heads = heads

        self.conv1 = ColumnParallelConv2dPatch(
            in_channels=3,
            out_channels=dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        scale = dim ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(dim))
        self.positional_embedding_vlm = nn.Parameter(
            scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, dim)
        )
        self.ln_pre = LayerNorm(dim)
        self.ln_post = LayerNorm(dim)

        self.transformer = _Transformer(
            dim,
            layers,
            heads,
            ENCODER_MAX_BATCH_SIZE,
            ENCODER_MAX_SEQ_LEN,
            mlp_ratio,
            act_layer=nn.GELU,
        )

        # Build packed indices to support dynamic resampling of positional embeddings
        image_h, image_w = self.image_size
        patch_h, patch_w = self.patch_size
        idx_h, idx_w = image_h // patch_h, image_w // patch_w
        img_idx = torch.arange(image_h * image_w // (patch_h * patch_w), dtype=torch.int32).reshape(idx_h * idx_w, 1)
        img_idx = torch.cat([img_idx, img_idx[:1]], dim=0)
        img_idx[-1, -1] = PackingIndex.ID_CLS_TOKEN

        packed_img_idx = torch.empty(
            img_idx.shape[0],
            img_idx.shape[1],
            PackingIndex.NUM_METADATA - 1,
            dtype=torch.int32,
        )
        packed_img_idx[:, :, PackingIndex.Y] = img_idx // idx_w
        packed_img_idx[:, :, PackingIndex.X] = img_idx % idx_w
        packed_img_idx[:, :, PackingIndex.HEIGHT].fill_(idx_h)
        packed_img_idx[:, :, PackingIndex.WIDTH].fill_(idx_w)
        packed_img_idx[:, :, PackingIndex.IDX] = img_idx
        packed_img_idx = packed_img_idx.reshape(1, -1, PackingIndex.NUM_METADATA - 1)
        self.packed_img_idx = packed_img_idx  # for pos-emb resampling

        # Precompute 2D-aware RoPE cis for visual tokens (stored as buffer)
        rope_dim_half = dim // heads // 2
        rope_freq = self._get_rope_freqs(rope_dim_half)
        freqs_x = self._compute_rope_freqs(rope_freq, packed_img_idx[:, :, PackingIndex.X] + 1)
        freqs_y = self._compute_rope_freqs(rope_freq, packed_img_idx[:, :, PackingIndex.Y] + 1)
        freqs = torch.cat([freqs_x, freqs_y], dim=-1).float().contiguous()[..., ::2]
        freqs = freqs.masked_fill(packed_img_idx[:, :, PackingIndex.IDX, None] < 0, 0)
        freq_cis = torch.view_as_complex(torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)).squeeze(0)
        self.register_buffer("freq_cis", freq_cis, persistent=False)

        self._register_load_state_dict_pre_hook(self._load_hook)

    def _get_rope_freqs(self, dim: int, theta: float = 10000.0) -> torch.Tensor:
        base = torch.arange(0, dim, 2)[: (dim // 2)].float() / dim
        return 1.0 / (theta ** base)

    @torch.amp.autocast("cuda", enabled=False)
    def _compute_rope_freqs(self, freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        freqs = einsum("..., f -> ... f", t.type(freqs.dtype), freqs)
        freqs = freqs.repeat_interleave(2, dim=-1)
        return freqs

    def _load_hook(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        orig_pos_embed = state_dict.get(prefix + "positional_embedding")
        if orig_pos_embed is not None and orig_pos_embed.shape[-2:] != self.positional_embedding_vlm.shape[-2:]:
            raise ValueError(
                f"Positional embedding shape {orig_pos_embed.shape} != expected {self.positional_embedding_vlm.shape}"
            )

        batch_size, token_per_image, _ = self.packed_img_idx.shape
        idx = self.packed_img_idx.reshape(batch_size * token_per_image, 1, -1)
        total_windows, window_size, _ = idx.shape

        grid = (
            (idx[:, :, [PackingIndex.X, PackingIndex.Y]] / idx[:, :, [PackingIndex.WIDTH, PackingIndex.HEIGHT]]) * 2 - 1
        )[None, ...]

        if orig_pos_embed is not None:
            posemb = orig_pos_embed[1:].view(1, self.grid_size[0], self.grid_size[1], -1).permute(0, 3, 1, 2).contiguous()
            posemb = posemb.to(device=grid.device, dtype=grid.dtype)
            sample = F.grid_sample(posemb, grid, padding_mode="zeros")
            sample = sample.view(-1, total_windows, window_size).permute(1, 2, 0).contiguous()
            sample = torch.where(
                idx[:, :, PackingIndex.IDX, None] == PackingIndex.ID_CLS_TOKEN,
                orig_pos_embed[0].view(1, 1, -1).to(device=sample.device, dtype=sample.dtype),
                sample,
            )
            new_pos_embed = sample.reshape(batch_size, token_per_image, -1)
            state_dict[prefix + "positional_embedding_vlm"] = new_pos_embed.squeeze(0)

    def _valid_freq_cis(self, device: torch.device) -> torch.Tensor:
        """
        Ensure freq_cis is complex (recompute if it was cast to real by a global .to(dtype=...)).
        """
        fc = getattr(self, "freq_cis", None)
        if fc is not None and torch.is_complex(fc):
            return fc.to(device)
        # Recompute
        rope_dim_half = self._dim // self._heads // 2
        rope_freq = self._get_rope_freqs(rope_dim_half)
        freqs_x = self._compute_rope_freqs(rope_freq, self.packed_img_idx[:, :, PackingIndex.X] + 1)
        freqs_y = self._compute_rope_freqs(rope_freq, self.packed_img_idx[:, :, PackingIndex.Y] + 1)
        freqs = torch.cat([freqs_x, freqs_y], dim=-1).float().contiguous()[..., ::2]
        freqs = freqs.masked_fill(self.packed_img_idx[:, :, PackingIndex.IDX, None] < 0, 0)
        fc = torch.view_as_complex(torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)).squeeze(0).to(device)
        return fc

    def _apply_class_embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat(
            [
                x,
                self.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            ],
            dim=1,
        )
        return x

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Accept [B, T, C, H, W] or [B, M, T, C, H, W]; unify to [B*M*T, C, H, W]
        if images.ndim == 5:
            num_concurrent_media = 1
            bsz, num_chunks, nch, h, w = images.shape
        else:
            bsz, num_concurrent_media, num_chunks, nch, h, w = images.shape
        images = images.reshape(bsz * num_concurrent_media * num_chunks, nch, h, w)

        x = self.conv1(images)
        _, ntok, dim = x.shape
        x = x.reshape(bsz * num_concurrent_media * num_chunks, ntok, dim)

        x = self._apply_class_embedding(x)
        ntok += 1

        if self.positional_embedding_vlm is not None:
            x = x + self.positional_embedding_vlm.to(x.dtype)

        x = x.reshape(bsz * num_concurrent_media, num_chunks, ntok, dim)
        x = self.ln_pre(x)
        x = x.view(bsz * num_concurrent_media, -1, dim)

        freq_cis = self._valid_freq_cis(images.device)
        x = self.transformer(x, freq_cis=freq_cis)
        x = self.ln_post(x)

        x = x[:, :-1, :]  # drop CLS token output
        return x

# ------------------------------------------------------------------------------
# 8) Vision Embeddings + scatter back into sequence
# ------------------------------------------------------------------------------

def scatter_embeddings(
    image_batch: List[List[torch.Tensor]],
    image_mask: torch.Tensor,
    h_image: torch.Tensor,
    encoded_patches_proj: torch.Tensor,
) -> torch.Tensor:
    """
    Place projected image patch embeddings into the target [B, T, D] sequence using a boolean image_mask [B, T].
    Ensures mask expanded to [T, D] for masked_scatter_ to prevent size mismatch errors.
    """
    num_images_per_sequence = [sum(image.size(0) for image in sample_images) for sample_images in image_batch]
    assert sum(num_images_per_sequence) == encoded_patches_proj.size(0), (
        f"{sum(num_images_per_sequence)=} != {encoded_patches_proj.shape=}"
    )

    assert not torch.isnan(encoded_patches_proj).any()
    encoded_patches_list = encoded_patches_proj.split(num_images_per_sequence, dim=0)
    B, T, D = h_image.shape

    for index in range(B):
        encoded_patches_per_sample = encoded_patches_list[index]
        sample_image_mask = image_mask[index]  # [T]

        if encoded_patches_per_sample.numel() == 0:
            continue

        src = encoded_patches_per_sample.contiguous().view(-1, encoded_patches_per_sample.size(-1)).to(h_image.dtype)
        n_tokens_to_fill = int(sample_image_mask.sum().item())
        if n_tokens_to_fill == 0:
            continue
        assert n_tokens_to_fill <= src.size(0), (
            f"mask requests {n_tokens_to_fill} tokens but only {src.size(0)} available"
        )

        mask_2d = sample_image_mask.to(torch.bool).view(T, 1).expand(T, D)
        assert src[:n_tokens_to_fill, :].numel() == int(mask_2d.sum().item()), (
            f"source elements {src[:n_tokens_to_fill, :].numel()} != masked target elements {int(mask_2d.sum().item())}"
        )
        h_image[index].masked_scatter_(mask_2d, src[:n_tokens_to_fill, :])

    return h_image

class VisionEmbeddings(nn.Module):
    def __init__(self, args: VisionArgs) -> None:
        super().__init__()
        self.args = args
        image_size = args.image_size
        patch_size = args.patch_size
        self.vision_encoder = VisionEncoder(
            image_size=(image_size.height, image_size.width),
            patch_size=(patch_size.height, patch_size.width),
            dim=args.dim,
            layers=args.n_layers,
            heads=args.n_heads,
            mlp_ratio=args.mlp_ratio,
        )
        self.vision_adapter = PixelShuffleMLP(
            ps_ratio=args.pixel_shuffle_ratio,
            input_dim=args.dim,
            output_dim=args.output_dim,
        )
        self.output_dim = args.output_dim
        self._register_load_state_dict_pre_hook(self._load_hook)

    def _load_hook(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        original_sd = self.state_dict()
        for k in list(state_dict.keys()):
            if k.startswith(prefix) and len(state_dict[k].shape) == 1 and state_dict[k].shape[0] == 0:
                state_dict[k] = state_dict[k].reshape(original_sd[k[len(prefix):]].shape)

    def _get_empty_sequence(self, h: torch.Tensor) -> torch.Tensor:
        return torch.zeros(h.shape[0], h.shape[1], self.output_dim, device=h.device, dtype=h.dtype)

    def forward(
        self,
        image_batch: List[List[torch.Tensor]],
        image_mask: torch.Tensor,
        h_ref: torch.Tensor,
    ) -> torch.Tensor:
        images_flattened = [img for sample in image_batch for img in sample]
        if len(images_flattened) == 0:
            return self._get_empty_sequence(h_ref)
        images_flattened = torch.vstack(images_flattened).unsqueeze(1).to(h_ref.dtype).to(h_ref.device)
        embedding = self.vision_encoder(images_flattened)
        projected_embedding = self.vision_adapter(embedding).to(h_ref.dtype)
        h_image = self._get_empty_sequence(h_ref)
        return scatter_embeddings(image_batch, image_mask, h_image, projected_embedding)

# ------------------------------------------------------------------------------
# 9) Text stack + Multimodal model
# ------------------------------------------------------------------------------

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, dim))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.embedding(x, self.weight)

def build_causal_mask(
    bsz: int,
    seqlen: int,
    start_pos: int,
    total_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    q_positions = torch.arange(start_pos, start_pos + seqlen, device=device)
    k_positions = torch.arange(0, total_len, device=device)
    causal = (q_positions[:, None] < (k_positions[None, :])).unsqueeze(0).unsqueeze(0)
    causal = causal.expand(bsz, 1, seqlen, total_len)
    return causal

class TextTransformer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.layers = nn.ModuleList()
        for i in range(args.n_layers):
            use_rope = True
            if args.nope_layer_interval is not None and args.nope_layer_interval > 0:
                if (i + 1) % args.nope_layer_interval == 0:
                    use_rope = False
            self.layers.append(
                TransformerBlock(args=args, use_rope=use_rope, gated=False, act_layer=nn.GELU, mlp_ratio=4.0)
            )
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        # Precompute text RoPE cis (complex). If casted by .to(dtype), we'll repair in forward.
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                dim=args.head_dim if args.head_dim is not None else (args.dim // args.n_heads),
                end=args.max_seq_len,
                theta=args.rope_theta,
                use_scaled=args.use_scaled_rope,
                scale_factor=args.rope_scaling_factor,
                high_freq_factor=args.rope_high_freq_factor,
            ),
            persistent=False,
        )

    def _valid_freq_cis(self, device: torch.device) -> torch.Tensor:
        fc = getattr(self, "freqs_cis", None)
        if fc is not None and torch.is_complex(fc):
            return fc.to(device)
        # Recompute if buffer got cast to real by a global .to(dtype)
        args = self.args
        fc = precompute_freqs_cis(
            dim=args.head_dim if args.head_dim is not None else (args.dim // args.n_heads),
            end=args.max_seq_len,
            theta=args.rope_theta,
            use_scaled=args.use_scaled_rope,
            scale_factor=args.rope_scaling_factor,
            high_freq_factor=args.rope_high_freq_factor,
        ).to(device)
        return fc

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        use_causal_mask: bool = False,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        freq_cis = self._valid_freq_cis(x.device)
        mask = None
        total_k_len = start_pos + seqlen
        if use_causal_mask and start_pos == 0:
            mask = build_causal_mask(bsz, seqlen, start_pos, total_k_len, x.device, x.dtype)
        for layer in self.layers:
            if isinstance(layer.attn, Attention) and layer.attn.use_rope:
                x = layer(x, start_pos=start_pos, freq_cis=freq_cis, mask=mask)
            else:
                x = layer(x, start_pos=start_pos, freq_cis=None, mask=mask)
        return self.norm(x)

class MultiModalTransformer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.text_emb = TokenEmbedding(args.vocab_size, args.dim)
        self.vision = VisionEmbeddings(args.vision_args) if args.vision_args is not None else None
        self.text_stack = TextTransformer(args)
        self.lm_head = nn.Linear(args.dim, args.vocab_size, bias=False)
        if args.tie_embeddings:
            self.lm_head.weight = self.text_emb.weight

        if args.quantization_args is not None and args.quantization_args.enabled:
            rich_print("[yellow]QuantizationArgs enabled: using dtype or ptq stubs.[/yellow]")
        if args.lora_args is not None and args.lora_args.enabled:
            rich_print("[yellow]LoRAArgs enabled: integration stubs available (no-op by default).[/yellow]")

    def fuse_vision(
        self,
        h_text: torch.Tensor,
        image_batch: Optional[List[List[torch.Tensor]]],
        image_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.vision is None or image_batch is None or image_mask is None:
            return h_text
        h_image = self.vision(image_batch, image_mask, h_text)
        return h_text + h_image

    def forward(
        self,
        tokens: torch.Tensor,
        start_pos: int = 0,
        image_batch: Optional[List[List[torch.Tensor]]] = None,
        image_mask: Optional[torch.Tensor] = None,
        use_causal_mask: bool = False,
    ) -> torch.Tensor:
        h_text = self.text_emb(tokens)  # [B, T, D]
        h = self.fuse_vision(h_text, image_batch=image_batch, image_mask=image_mask)
        logits = self.lm_head(self.text_stack(h, start_pos=start_pos, use_causal_mask=use_causal_mask))
        if VERBOSE:
            rich_kv_panel("Forward Shapes", {
                "tokens": tokens.shape, "tokens.dtype": tokens.dtype,
                "hidden": h.shape, "hidden.dtype": h.dtype,
                "logits": logits.shape, "logits.dtype": logits.dtype,
                "mp_world_size": mp_world_size(), "fs_ready": _FS_READY
            })
        return logits

# ------------------------------------------------------------------------------
# 10) Extensibility Hooks (LoRA / Quant / MoE) - Stubs
# ------------------------------------------------------------------------------

def apply_lora_stub(model: nn.Module, lora_args: LoRAArgs) -> None:
    if not lora_args.enabled:
        return
    rich_print(f"[green]LoRA enabled (rank={lora_args.rank}, alpha={lora_args.alpha}). Stub no-op applied.[/green]")

def apply_quantization_stub(model: nn.Module, quant_args: QuantizationArgs) -> None:
    if not quant_args.enabled:
        return
    rich_print(f"[green]Quantization enabled (dtype={quant_args.dtype}). Stub no-op applied.[/green]")

def apply_moe_stub(model: nn.Module, moe_args: MoEArgs) -> None:
    if not moe_args.enabled:
        return
    rich_print(f"[green]MoE enabled (experts={moe_args.num_experts}, top_k={moe_args.top_k}). Stub no-op applied.[/green]")

# ------------------------------------------------------------------------------
# 11) Utilities: parameter counting, dtype reports
# ------------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_summary(model: nn.Module, name: str = "Model") -> None:
    if not VERBOSE:
        return
    kv = {
        "name": name,
        "params (M)": round(count_parameters(model) / 1e6, 3),
        "fairscale": _FAIRSCALE_IMPORTED,
        "fairscale_initialized": _FS_READY,
        "mp world_size": mp_world_size(),
    }
    rich_kv_panel("Model Summary", kv)

def dtype_report(model: nn.Module, title: str = "Dtype Report") -> None:
    if not VERBOSE:
        return
    dtypes = {}
    for _, p in model.named_parameters():
        dtypes.setdefault(str(p.dtype), 0)
        dtypes[str(p.dtype)] += p.numel()
    rich_kv_panel(title, {k: f"{v/1e6:.3f}M params" for k, v in dtypes.items()})

# ------------------------------------------------------------------------------
# 12) Examples: Training, Decoding, BF16 end-to-end
# ------------------------------------------------------------------------------

def _example_build_args(
    vocab_size: int = 50257,
    seq_len: int = 256,
    vision: bool = True,
) -> ModelArgs:
    v_args = VisionArgs(
        image_size=Size(height=224, width=224),
        patch_size=Size(height=16, width=16),
        dim=384,
        n_layers=4,
        n_heads=6,
        mlp_ratio=4.0,
        output_dim=512,   # must match text dim below
        pixel_shuffle_ratio=2.0,
    ) if vision else None

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
        attention_chunk_size=None,
        rope_theta=1_000_000.0,
        use_scaled_rope=True,
        rope_scaling_factor=16.0,
        rope_high_freq_factor=1.0,
        nope_layer_interval=4,        # every 4th layer uses NoPE (temperature tuning can apply)
        use_qk_norm=True,
        attn_temperature_tuning=True,
        floor_scale=8192.0,
        attn_scale=0.1,
        vision_args=v_args,
        moe_args=MoEArgs(enabled=False),
        quantization_args=QuantizationArgs(enabled=False),
        lora_args=LoRAArgs(enabled=False),
        max_batch_size=8,
        max_seq_len=seq_len,
        tie_embeddings=True,
    )
    return ModelArgs(**m_args.model_dump())

def _example_synthetic_batch(
    bsz: int = 2,
    T: int = 64,
    vocab_size: int = 50257,
    with_vision: bool = True,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, Optional[List[List[torch.Tensor]]], Optional[torch.Tensor]]:
    tokens = torch.randint(0, vocab_size, (bsz, T), device=device)
    if not with_vision:
        return tokens, None, None

    image_batch: List[List[torch.Tensor]] = []
    image_mask = torch.zeros(bsz, T, dtype=torch.bool, device=device)
    K = min(16, T // 2)
    for b in range(bsz):
        img = torch.randn(1, 3, 224, 224, device=device, dtype=dtype)  # num_tiles=1
        image_batch.append([img])
        image_mask[b, 1 : K + 1] = True
    return tokens, image_batch, image_mask

def _example_train_step():
    global VERBOSE
    VERBOSE = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = _example_build_args(vocab_size=32000, seq_len=256, vision=True)
    model = MultiModalTransformer(args).to(device)
    model_summary(model, "MultimodalTransformer (Train)")
    dtype_report(model, "Initial Dtypes")

    tokens, image_batch, image_mask = _example_synthetic_batch(
        bsz=2, T=64, vocab_size=args.vocab_size, with_vision=True, device=device, dtype=torch.float32
    )
    logits = model(tokens, start_pos=0, image_batch=image_batch, image_mask=image_mask, use_causal_mask=True)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tokens.view(-1))
    rich_kv_panel("Train Step", {"loss": float(loss.item())})

def _example_decode_step():
    global VERBOSE
    VERBOSE = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = _example_build_args(vocab_size=32000, seq_len=256, vision=True)
    model = MultiModalTransformer(args).to(device)
    model_summary(model, "DecodeModel")
    dtype_report(model, "Decode Dtypes")

    B, T = 2, 32
    tokens, image_batch, image_mask = _example_synthetic_batch(
        bsz=B, T=T, vocab_size=args.vocab_size, with_vision=True, device=device, dtype=torch.float32
    )
    logits = model(tokens, start_pos=0, image_batch=image_batch, image_mask=image_mask, use_causal_mask=False)
    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

    tokens_step2 = next_token
    logits2 = model(tokens_step2, start_pos=T, image_batch=None, image_mask=None, use_causal_mask=False)
    next_token2 = torch.argmax(logits2[:, -1, :], dim=-1)
    rich_kv_panel("Decode Step", {"t1_next_token": next_token.squeeze().tolist(), "t2_next_token": next_token2.tolist()})

def _example_bf16_end2end():
    """
    Optional: Run end-to-end in BF16 (CUDA recommended).
    Note: Global model.to(dtype=bf16) may cast complex buffers to real and print a warning.
    We re-validate cis in forward paths to maintain correctness.
    """
    global VERBOSE
    VERBOSE = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        rich_print("[red]BF16 example is best on CUDA; skipping on CPU.[/red]")
        return

    args = _example_build_args(vocab_size=32000, seq_len=128, vision=True)
    model = MultiModalTransformer(args).to(device=device, dtype=torch.bfloat16)
    model_summary(model, "BF16 Multimodal")
    dtype_report(model, "BF16 Dtypes")

    tokens, image_batch, image_mask = _example_synthetic_batch(
        bsz=2, T=48, vocab_size=args.vocab_size, with_vision=True, device=device, dtype=torch.bfloat16
    )
    logits = model(tokens, start_pos=0, image_batch=image_batch, image_mask=image_mask, use_causal_mask=True)
    # CE expects float32 logits; cast for numerical stability
    loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), tokens.view(-1))
    rich_kv_panel("BF16 Train Step", {"loss": float(loss.item())})

if __name__ == "__main__":
    # Example runs: toggle as needed
    _example_train_step()
    _example_decode_step()
    _example_bf16_end2end()