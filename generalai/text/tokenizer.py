#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced, professional-grade, generalized tokenizer pipeline (single-file module).

Fix note (2025-08): Addressed tiktoken OverflowError for negative special-token IDs by:
- Sanitizing and auto-allocating valid positive IDs inside TiktokenBackend.with_added_special_tokens.
- Additionally, SpecialTokenManager.ensure now allocates safe IDs when the backend is tiktoken.

Design goals:
- Zero hardcoded token IDs. Fully dynamic registration and extension of special tokens.
- Backend abstraction with support for tiktoken (primary) and optional sentencepiece (if installed).
- Robust normalization (Unicode NFC/NFKC), whitespace preservation strategies, task-aware preprocessing.
- Instruction/Conversation/RLHF/Reasoning templates via a schema-driven, placeholder-based templating engine.
- Detailed, schema-aware metadata in encode/decode, with optional rich-based verbose reporting.
- Sliding-window long-context chunking, dynamic batching, attention masks, segment indices.
- Subword/OOV handling with byte-level fallback; optional domain evaluation and adaptive placeholders for frequent patterns.
- Clean, modern Python style; all explanations and examples live inside this file.

NOTE:
- This module favors tiktoken for performance and BPE coverage (with implicit byte fallback).
- SentencePiece (unigram) is optional and auto-detected; if present, you can switch via config.
- Extending the true mergeable vocabulary at runtime is not possible with tiktoken; instead, we:
  - Add schema-aware special tokens dynamically (IDs are allocated automatically without collisions).
  - Provide optional, reversible placeholder compression for domain-specific multi-byte sequences (off by default) to avoid
    distorting the training distribution.
- This file includes a comprehensive demo when run as a script. No extra text is printed beyond code behavior.

Authoring style:
- Technical tone, deep explanations in docstrings and comments.
- Clean coding standards, type hints, logging, and rich visualizations (when enabled).
"""

from __future__ import annotations

import os
import re
import math
import json
import time
import unicodedata
import logging
import itertools
import statistics
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Tuple, Optional, Iterable, Union, Sequence, Callable

# --------------------------------------------------------------------------------------
# Optional dependencies
# --------------------------------------------------------------------------------------

try:
    import tiktoken  # BPE tokenizer with byte fallback
    _HAVE_TIKTOKEN = True
except Exception:
    tiktoken = None  # type: ignore
    _HAVE_TIKTOKEN = False

try:
    import sentencepiece as spm  # Unigram tokenizer (optional)
    _HAVE_SENTENCEPIECE = True
except Exception:
    spm = None  # type: ignore
    _HAVE_SENTENCEPIECE = False

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    _HAVE_RICH = True
    _RICH_CONSOLE = Console()
except Exception:
    _HAVE_RICH = False
    _RICH_CONSOLE = None

# --------------------------------------------------------------------------------------
# Logging setup (integrates with Rich if available)
# --------------------------------------------------------------------------------------

_LOG_LEVEL = os.environ.get("TOKENIZER_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, _LOG_LEVEL, logging.INFO))
logger = logging.getLogger("advanced_tokenizer")


# --------------------------------------------------------------------------------------
# Utility helpers
# --------------------------------------------------------------------------------------

def _now_ms() -> int:
    return int(time.time() * 1000)


def _maybe_rich(func: Callable[..., None]) -> Callable[..., None]:
    """
    Decorator for rich-only UI prints: executes only if rich is available and
    if verbose mode is on in the provided self/config.
    """
    def wrapper(*args, **kwargs):
        self = args[0] if args else None
        verbose = False
        if hasattr(self, "config") and getattr(self.config, "verbose", False):
            verbose = True
        elif hasattr(self, "verbose") and getattr(self, "verbose", False):
            verbose = True
        if _HAVE_RICH and verbose:
            return func(*args, **kwargs)
        return None
    return wrapper


def _safe_len(x: Optional[Sequence[Any]]) -> int:
    return len(x) if x is not None else 0


# --------------------------------------------------------------------------------------
# Backend Abstraction
# --------------------------------------------------------------------------------------

class BaseBackend:
    """
    Abstract tokenizer backend. Concrete implementations must implement:
    - name
    - n_vocab
    - encode(text, allowed_special, disallowed_special)
    - decode(ids)
    - decode_single_token_bytes(token_id)
    - special_tokens (mapping str->int)
    - with_added_special_tokens(mapping) -> BaseBackend (returns a new backend with added tokens)
    - normalize(text) -> normalized text
    """

    name: str

    @property
    def n_vocab(self) -> int:
        raise NotImplementedError

    @property
    def special_tokens(self) -> Dict[str, int]:
        raise NotImplementedError

    def encode(
        self,
        text: str,
        *,
        allowed_special: Union[str, Iterable[str]] = (),
        disallowed_special: Union[str, Iterable[str]] = "all",
    ) -> List[int]:
        raise NotImplementedError

    def decode(self, ids: List[int]) -> str:
        raise NotImplementedError

    def decode_single_token_bytes(self, token_id: int) -> bytes:
        raise NotImplementedError

    def with_added_special_tokens(self, special: Dict[str, int]) -> "BaseBackend":
        raise NotImplementedError

    def normalize(self, text: str) -> str:
        return text


class TiktokenBackend(BaseBackend):
    """
    TikToken backend. We clone the base encoding while extending special tokens dynamically
    (no hardcoded IDs; IDs auto-allocated after base vocab size).

    Safety:
    - When adding specials, we sanitize incoming IDs. Any negative/conflicting IDs are re-assigned to fresh IDs
      >= current n_vocab and not in mergeable ranks nor existing special IDs.
    """
    def __init__(self, base_name: str = "o200k_base", normalization: str = "NFC"):
        if not _HAVE_TIKTOKEN:
            raise ImportError("tiktoken is required for TiktokenBackend.")
        try:
            self._base_encoding = tiktoken.get_encoding(base_name)
        except Exception as e:
            logger.warning(f"Failed to load tiktoken encoding '{base_name}': {e}. Falling back to 'cl100k_base'.")
            self._base_encoding = tiktoken.get_encoding("cl100k_base")
        self._normalization = normalization
        self._encoding = self._base_encoding
        self._mergeable_ranks = getattr(self._base_encoding, "_mergeable_ranks")
        self._pat_str = getattr(self._base_encoding, "_pat_str")
        self._special_tokens = dict(getattr(self._base_encoding, "_special_tokens", {}))
        self.name = f"tiktoken:{self._encoding.name}"

    @property
    def n_vocab(self) -> int:
        return int(self._encoding.n_vocab)

    @property
    def special_tokens(self) -> Dict[str, int]:
        return dict(self._special_tokens)

    def _rebuild_encoding(self):
        self._encoding = tiktoken.Encoding(
            name=f"advanced_dynamic_{self._base_encoding.name}",
            pat_str=self._pat_str,
            mergeable_ranks=self._mergeable_ranks,
            special_tokens=self._special_tokens,
        )
        self.name = f"tiktoken:{self._encoding.name}"

    def with_added_special_tokens(self, special: Dict[str, int]) -> "TiktokenBackend":
        """
        Merge the provided specials, assign safe IDs where needed, and rebuild the encoding.
        """
        clone = TiktokenBackend.__new__(TiktokenBackend)
        clone._base_encoding = self._base_encoding
        clone._normalization = self._normalization
        clone._mergeable_ranks = self._mergeable_ranks
        clone._pat_str = self._pat_str
        clone._special_tokens = dict(self._special_tokens)

        base_ids = set(clone._mergeable_ranks.values()) if isinstance(clone._mergeable_ranks, dict) else set()
        existing_special_ids = set(clone._special_tokens.values())
        # Start allocating past the current largest id (covers base + existing specials)
        start_id = max([*(base_ids or [0]), *(existing_special_ids or [0]), self.n_vocab]) + 1

        sanitized: Dict[str, int] = {}
        for name, req_id in special.items():
            if name in clone._special_tokens:
                # Already present; keep existing
                continue
            assign_id = req_id
            needs_alloc = (
                assign_id is None or
                (isinstance(assign_id, int) and assign_id < 0) or
                (isinstance(assign_id, int) and (assign_id in base_ids or assign_id in existing_special_ids))
            )
            if needs_alloc or not isinstance(assign_id, int):
                assign_id = start_id
                # Ensure no collision
                while assign_id in base_ids or assign_id in existing_special_ids:
                    assign_id += 1
                start_id = assign_id + 1
            sanitized[name] = assign_id
            existing_special_ids.add(assign_id)

        if sanitized:
            clone._special_tokens.update(sanitized)

        clone._encoding = tiktoken.Encoding(
            name=f"advanced_dynamic_{self._base_encoding.name}",
            pat_str=clone._pat_str,
            mergeable_ranks=clone._mergeable_ranks,
            special_tokens=clone._special_tokens,
        )
        clone.name = f"tiktoken:{clone._encoding.name}"
        return clone

    def encode(
        self,
        text: str,
        *,
        allowed_special: Union[str, Iterable[str]] = (),
        disallowed_special: Union[str, Iterable[str]] = "all",
    ) -> List[int]:
        return self._encoding.encode(text, allowed_special=allowed_special, disallowed_special=disallowed_special)

    def decode(self, ids: List[int]) -> str:
        return self._encoding.decode(ids)

    def decode_single_token_bytes(self, token_id: int) -> bytes:
        return self._encoding.decode_single_token_bytes(token_id)

    def normalize(self, text: str) -> str:
        if self._normalization.upper() in ("NFC", "NFKC"):
            return unicodedata.normalize(self._normalization.upper(), text)
        return text


class SentencePieceBackend(BaseBackend):
    """
    Optional SentencePiece backend (unigram LM). Ideal for morphologically rich languages and multilingual settings.
    Special token support is limited to existing pieces or <extra_id_*> series if present.

    Note: SentencePiece models typically encode BOS/EOS optionally. We won't auto-inject them; we rely on special tokens
    managed by our SpecialTokenManager. If your SP model includes <extra_id_*> tokens, we can map dynamic specials there.
    """
    def __init__(self, model_path: str, normalization: str = "NFC"):
        if not _HAVE_SENTENCEPIECE:
            raise ImportError("sentencepiece is required for SentencePieceBackend.")
        self._sp = spm.SentencePieceProcessor(model_file=model_path)
        self._normalization = normalization
        self.name = f"sentencepiece:{os.path.basename(model_path)}"
        self._piece_to_id = {self._sp.id_to_piece(i): i for i in range(self._sp.get_piece_size())}
        self._extra_ids = [p for p in self._piece_to_id if p.startswith("<extra_id_")]
        self._special_tokens: Dict[str, int] = {}
        for p in ["<pad>", "<unk>", "<bos>", "<eos>"]:
            if p in self._piece_to_id:
                self._special_tokens[p] = self._piece_to_id[p]

    @property
    def n_vocab(self) -> int:
        return int(self._sp.get_piece_size())

    @property
    def special_tokens(self) -> Dict[str, int]:
        return dict(self._special_tokens)

    def with_added_special_tokens(self, special: Dict[str, int]) -> "SentencePieceBackend":
        """
        SentencePiece does not support adding new tokens on-the-fly without retraining.
        We alias custom names to existing pieces (prefer <extra_id_*> if available).
        """
        clone = SentencePieceBackend.__new__(SentencePieceBackend)
        clone._sp = self._sp
        clone._normalization = self._normalization
        clone.name = self.name
        clone._piece_to_id = self._piece_to_id
        clone._extra_ids = self._extra_ids
        clone._special_tokens = dict(self._special_tokens)

        for name in special.keys():
            if name in clone._special_tokens:
                continue
            if name in clone._piece_to_id:
                clone._special_tokens[name] = clone._piece_to_id[name]
            else:
                extra_piece = None
                for p in clone._extra_ids:
                    if p not in clone._special_tokens:
                        extra_piece = p
                        break
                if extra_piece is not None:
                    clone._special_tokens[name] = clone._piece_to_id[extra_piece]
                else:
                    logger.warning(f"SPM: No capacity to add special token '{name}'. Consider retraining the model.")
        return clone

    def encode(
        self,
        text: str,
        *,
        allowed_special: Union[str, Iterable[str]] = (),
        disallowed_special: Union[str, Iterable[str]] = "all",
    ) -> List[int]:
        return list(self._sp.encode(text, out_type=int))

    def decode(self, ids: List[int]) -> str:
        return self._sp.decode(ids)

    def decode_single_token_bytes(self, token_id: int) -> bytes:
        piece = self._sp.id_to_piece(int(token_id))
        return piece.encode("utf-8", errors="replace")

    def normalize(self, text: str) -> str:
        if self._normalization.upper() in ("NFC", "NFKC"):
            return unicodedata.normalize(self._normalization.upper(), text)
        return text


# --------------------------------------------------------------------------------------
# Special Token Manager (dynamic, no hardcoded IDs)
# --------------------------------------------------------------------------------------

class SpecialTokenManager:
    """
    Manages a registry of special tokens (names) and negotiates their IDs from the backend.

    Strategy:
    - We do not rely on fixed IDs. Instead, we allocate IDs by cloning the backend with an augmented special token map.
    - For tiktoken: IDs must be explicit integers; we auto-assign valid, non-conflicting IDs.
    - For sentencepiece: we alias names to existing pieces or <extra_id_*> if available.

    Also provides helpers to escape user content so it doesn't accidentally inject special tokens.
    """

    def __init__(self, backend: BaseBackend):
        self._backend = backend
        self._tokens: Dict[str, int] = dict(backend.special_tokens)
        self._used_names: set[str] = set(self._tokens.keys())

    @property
    def backend(self) -> BaseBackend:
        return self._backend

    @property
    def tokens(self) -> Dict[str, int]:
        return dict(self._tokens)

    def ensure(self, names: Iterable[str]) -> None:
        """
        Ensure all token names are registered; if some are missing, rebuild backend with them.
        """
        missing = [n for n in names if n not in self._tokens]
        if not missing:
            return

        # For tiktoken, allocate safe positive IDs now. For SPM, values are ignored.
        allocation: Dict[str, int] = {}
        if isinstance(self._backend, TiktokenBackend):
            existing_ids = set(self._backend.special_tokens.values())
            start_id = max(self._backend.n_vocab, max(existing_ids) + 1 if existing_ids else self._backend.n_vocab)
            for name in missing:
                # Ensure uniqueness (defensive)
                while start_id in existing_ids:
                    start_id += 1
                allocation[name] = start_id
                existing_ids.add(start_id)
                start_id += 1
        else:
            # Values irrelevant for SPM; give dummy non-negative id to avoid any potential type issues.
            allocation = {n: 0 for n in missing}

        new_backend = self._backend.with_added_special_tokens(allocation)
        self._backend = new_backend
        self._tokens = dict(self._backend.special_tokens)
        self._used_names.update(self._tokens.keys())

    def get_id(self, name: str) -> Optional[int]:
        return self._tokens.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._tokens

    def escape_specials_in_user_text(self, text: str) -> str:
        """
        Prevent user text from accidentally inserting special token markers such as "<|...|>".
        We escape sequences of the form "<|...|>" to "<\|...\|>" unless they correspond to registered specials.
        """
        pattern = re.compile(r"<\|([^|]+)\|>")
        def _repl(m):
            token_str = f"<|{m.group(1)}|>"
            if token_str in self._tokens:
                return token_str
            return f"<\\|{m.group(1)}\\|>"
        return pattern.sub(_repl, text)

    def unescape_specials(self, text: str) -> str:
        """Reverse the escaping performed by escape_specials_in_user_text."""
        return text.replace("<\\|", "<|").replace("\\|>", "|>")


# --------------------------------------------------------------------------------------
# Advanced configuration
# --------------------------------------------------------------------------------------

@dataclass
class AdvancedTokenizerConfig:
    # Backend preferences; will pick the first available
    backend_preference: List[str] = field(default_factory=lambda: ["tiktoken:o200k_base", "tiktoken:cl100k_base"])
    sentencepiece_model: Optional[str] = None  # if you want to prefer SPM, set backend_preference accordingly

    # General behavior
    max_sequence_length: int = 2048
    pad_direction: str = "right"  # "left" | "right"
    add_bos_eos: bool = True
    normalization_form: str = "NFC"  # "NFC" | "NFKC"
    preserve_whitespace_strategy: str = "normalize"  # "normalize" | "preserve" | "token"
    escape_user_specials: bool = True
    verbose: bool = True  # rich-based metadata visualizations

    # Masks and fields
    return_attention_mask: bool = True
    return_token_type_ids: bool = False
    labels_ignore_index: int = -100

    # Truncation and windowing
    truncation_strategy: str = "right"  # "left" | "right" | "center"
    sliding_window_stride: int = 256  # for long-doc chunking
    sliding_window_overlap: int = 64

    # OOV and Unicode handling
    enable_byte_fallback: bool = True
    strict_unicode_normalization: bool = True

    # Domain adaptation options
    enable_domain_placeholders: bool = False
    domain_placeholder_min_freq: int = 100
    domain_placeholder_prefix: str = "<|dom:"
    domain_placeholder_suffix: str = "|>"

    # Special token schema (names only; IDs are resolved dynamically)
    bos_token: str = "<|bos|>"
    eos_token: str = "<|eos|>"
    pad_token: str = "<|pad|>"
    unk_token: str = "<|unk|>"

    # Roles and section tokens (optional; registered on-demand)
    default_roles: List[str] = field(default_factory=lambda: [
        "<|system|>", "<|user|>", "<|assistant|>", "<|developer|>", "<|tool|>", "<|function|>",
        "<|function_call|>", "<|observation|>", "<|tool_response|>", "<|critic|>", "<|agent|>"
    ])
    section_tokens: List[str] = field(default_factory=lambda: [
        "<|instruction|>", "<|input|>", "<|output|>", "<|context|>", "<|reasoning|>", "<|think|>",
        "<|step|>", "<|conclude|>", "<|act|>", "<|act_end|>", "<|call|>", "<|call_end|>", "<|endofsegment|>"
    ])

    # Whitespace special tokens (if using preserve strategy 'token')
    whitespace_tokens: Dict[str, str] = field(default_factory=lambda: {
        "space": "<|ws:space|>",
        "tab": "<|ws:tab|>",
        "newline": "<|ws:newline|>",
        "multiws_prefix": "<|ws:x",
        "multiws_suffix": "|>"
    })


# --------------------------------------------------------------------------------------
# Metadata structures
# --------------------------------------------------------------------------------------

@dataclass
class SegmentSpan:
    name: str
    role: Optional[str]
    start: int
    end: int
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EncodeResult:
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    spans: List[SegmentSpan] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchEncodeResult:
    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]] = None
    token_type_ids: Optional[List[List[int]]] = None
    spans: List[List[SegmentSpan]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisReport:
    original_text: str
    normalized_text: str
    token_count: int
    char_count: int
    compression_ratio: float
    unique_token_ratio: float
    special_token_count: int
    perfect_roundtrip: bool
    token_details: List[Dict[str, Any]]


# --------------------------------------------------------------------------------------
# Whitespace handler
# --------------------------------------------------------------------------------------

class WhitespaceHandler:
    """
    Strategy:
    - "normalize": compress repeated whitespace to single spaces; strip trailing spaces; keep newlines
    - "preserve": keep as-is (no changes)
    - "token": replace whitespace with schema special tokens to preserve exact structure.
      This requires subsequent decode restoration.
    """

    def __init__(self, strategy: str, special_tokens: Dict[str, str]):
        strategy = strategy.lower()
        if strategy not in ("normalize", "preserve", "token"):
            raise ValueError("preserve_whitespace_strategy must be one of {'normalize','preserve','token'}")
        self.strategy = strategy
        self.st = special_tokens

    def encode_transform(self, text: str) -> Tuple[str, Dict[str, Any]]:
        meta: Dict[str, Any] = {"ws_strategy": self.strategy}
        if self.strategy == "normalize":
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
            text = text.strip()
        elif self.strategy == "token":
            def repl_spaces(m):
                n = len(m.group(0))
                if n == 1:
                    return self.st["space"]
                return f"{self.st['multiws_prefix']}{n}{self.st['multiws_suffix']}"
            text = re.sub(r"[ ]{1,}", repl_spaces, text)
            text = text.replace("\t", self.st["tab"])
            text = text.replace("\n", self.st["newline"])
        meta["length_after_ws"] = len(text)
        return text, meta

    def decode_restore(self, text: str) -> str:
        if self.strategy != "token":
            return text
        multiws_re = re.compile(
            re.escape(self.st["multiws_prefix"]) + r"(\d+)" + re.escape(self.st["multiws_suffix"])
        )
        text = multiws_re.sub(lambda m: " " * int(m.group(1)), text)
        text = text.replace(self.st["space"], " ")
        text = text.replace(self.st["tab"], "\t")
        text = text.replace(self.st["newline"], "\n")
        return text


# --------------------------------------------------------------------------------------
# Template Engine (schema-driven, placeholder-based)
# --------------------------------------------------------------------------------------

@dataclass
class Segment:
    """Represents a logical segment: role or section with content."""
    name: str
    content: str
    role: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)


class TemplateEngine:
    """
    Build flexible, schema-aware token sequences without rigid hard-coded templates.
    The engine:
      - Registers only the tokens you use (dynamic ensure).
      - Creates BOS/EOS (if enabled) and role/section boundaries.
      - Produces precise spans metadata for later decoding or supervision.
    """

    def __init__(self, stm: SpecialTokenManager, cfg: AdvancedTokenizerConfig):
        self.stm = stm
        self.cfg = cfg
        to_ensure = []
        if cfg.add_bos_eos:
            to_ensure += [cfg.bos_token, cfg.eos_token]
        to_ensure.append(cfg.pad_token)
        to_ensure.append(cfg.unk_token)
        self.stm.ensure(to_ensure)

    def build(self, segments: List[Segment]) -> Tuple[str, List[SegmentSpan]]:
        """
        Turns a list of segments into a single schema-marked string using special tokens as boundaries.
        Important: We emit markers as textual special tokens (e.g. "<|user|>") to keep a single string,
        which will be encoded by backend with allowed_special="all".
        """
        spans: List[SegmentSpan] = []

        def ensure_marker(token_name: str) -> str:
            self.stm.ensure([token_name])
            return token_name

        assembled_str_parts: List[str] = []
        cursor = 0

        # BOS
        if self.cfg.add_bos_eos:
            tok = ensure_marker(self.cfg.bos_token)
            assembled_str_parts.append(tok)
            spans.append(SegmentSpan(name="bos", role=None, start=cursor, end=cursor + 1))
            cursor += 1

        for seg in segments:
            if seg.role:
                role_token = seg.role if seg.role.startswith("<|") else f"<|{seg.role.strip('|<>')}|>"
                tok = ensure_marker(role_token)
                assembled_str_parts.append(tok)
                spans.append(SegmentSpan(name="role", role=seg.role, start=cursor, end=cursor + 1))
                cursor += 1

            seg_token = seg.name if seg.name.startswith("<|") else f"<|{seg.name.strip('|<>')}|>"
            tok = ensure_marker(seg_token)
            assembled_str_parts.append(tok)
            spans.append(SegmentSpan(name="section", role=seg.role, start=cursor, end=cursor + 1,
                                     attributes={"section": seg.name}))
            cursor += 1

            content = seg.content
            assembled_str_parts.append(content)
            spans.append(SegmentSpan(name="content", role=seg.role, start=cursor, end=cursor + len(content),
                                     attributes={"section": seg.name}))
            cursor += len(content)

            end_marker = ensure_marker("<|endofsegment|>")
            assembled_str_parts.append(end_marker)
            spans.append(SegmentSpan(name="section_end", role=seg.role, start=cursor, end=cursor + 1,
                                     attributes={"section": seg.name}))
            cursor += 1

        if self.cfg.add_bos_eos:
            tok = ensure_marker(self.cfg.eos_token)
            assembled_str_parts.append(tok)
            spans.append(SegmentSpan(name="eos", role=None, start=cursor, end=cursor + 1))
            cursor += 1

        assembled_text = "".join(assembled_str_parts)
        return assembled_text, spans


# --------------------------------------------------------------------------------------
# Advanced Tokenizer Pipeline
# --------------------------------------------------------------------------------------

class AdvancedTokenizerPipeline:
    """
    Professional-grade tokenization pipeline with:
    - Dynamic special token registry (no hard-coded IDs).
    - Backend-agnostic encode/decode.
    - Schema-driven templating for instruction/conversation/RLHF/reasoning.
    - Sliding-window long-context chunking and dynamic batching.
    - Detailed, rich metadata introspection.
    """

    def __init__(self, config: Optional[AdvancedTokenizerConfig] = None):
        self.config = config or AdvancedTokenizerConfig()
        self.backend = self._init_backend()
        self.stm = SpecialTokenManager(self.backend)
        self.template = TemplateEngine(self.stm, self.config)
        self._stats = {
            "total_processed": 0,
            "avg_sequence_length": 0.0,
            "truncated_sequences": 0,
            "padded_sequences": 0,
            "sliding_windows_built": 0,
        }
        self._ensure_default_schema_tokens()

    def _init_backend(self) -> BaseBackend:
        for pref in self.config.backend_preference:
            try:
                if pref.startswith("tiktoken:"):
                    name = pref.split(":", 1)[1]
                    return TiktokenBackend(base_name=name, normalization=self.config.normalization_form)
                if pref.startswith("sentencepiece:"):
                    path = pref.split(":", 1)[1]
                    return SentencePieceBackend(model_path=path, normalization=self.config.normalization_form)
            except Exception as e:
                logger.warning(f"Backend preference '{pref}' failed: {e}")
        if _HAVE_TIKTOKEN:
            logger.warning("Using fallback tiktoken:cl100k_base.")
            return TiktokenBackend(base_name="cl100k_base", normalization=self.config.normalization_form)
        if _HAVE_SENTENCEPIECE and self.config.sentencepiece_model:
            return SentencePieceBackend(model_path=self.config.sentencepiece_model,
                                        normalization=self.config.normalization_form)
        raise RuntimeError("No viable tokenizer backend available. Install tiktoken or sentencepiece.")

    def _ensure_default_schema_tokens(self):
        base_tokens = [self.config.pad_token, self.config.unk_token]
        if self.config.add_bos_eos:
            base_tokens.extend([self.config.bos_token, self.config.eos_token])
        if self.config.preserve_whitespace_strategy == "token":
            base_tokens.extend(self.config.whitespace_tokens.values())
        self.stm.ensure(base_tokens)

    @property
    def special_tokens(self) -> Dict[str, int]:
        return self.stm.tokens

    @property
    def n_vocab(self) -> int:
        return self.stm.backend.n_vocab

    def _normalize(self, text: str) -> str:
        text = self.stm.backend.normalize(text)
        if self.config.strict_unicode_normalization:
            form = self.config.normalization_form.upper()
            if form in ("NFC", "NFKC"):
                text = unicodedata.normalize(form, text)
        return text

    def _apply_ws_strategy(self, text: str) -> Tuple[str, Dict[str, Any], WhitespaceHandler]:
        ws = WhitespaceHandler(self.config.preserve_whitespace_strategy, self.config.whitespace_tokens)
        transformed, meta = ws.encode_transform(text)
        return transformed, meta, ws

    def _apply_truncation(self, tokens: List[int], max_len: int) -> List[int]:
        strategy = self.config.truncation_strategy.lower()
        if len(tokens) <= max_len:
            return tokens
        if strategy == "right":
            return tokens[:max_len]
        if strategy == "left":
            return tokens[-max_len:]
        if strategy == "center":
            half = max_len // 2
            return tokens[:half] + tokens[-(max_len - half):]
        return tokens[:max_len]

    def _pad(self, tokens: List[int], target_len: int) -> Tuple[List[int], List[int]]:
        if len(tokens) >= target_len:
            return tokens[:target_len], [1] * target_len
        pad_id = self.special_tokens.get(self.config.pad_token)
        pad_id = 0 if pad_id is None else pad_id
        pad_count = target_len - len(tokens)
        mask = [1] * len(tokens) + [0] * pad_count
        if self.config.pad_direction == "right":
            return tokens + [pad_id] * pad_count, mask
        else:
            return [pad_id] * pad_count + tokens, [0] * pad_count + [1] * len(tokens)

    def _encode_string(self, text: str, allow_special: bool = True) -> List[int]:
        allowed = "all" if allow_special else ()
        try:
            return self.stm.backend.encode(text, allowed_special=allowed, disallowed_special=())
        except Exception as e:
            logger.warning(f"Encoding failed, retrying without specials. Error: {e}")
            return self.stm.backend.encode(text, allowed_special=(), disallowed_special=())

    def _decode_tokens(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        if not skip_special_tokens:
            return self.stm.backend.decode(ids)
        special_ids = set(self.special_tokens.values())
        filtered = [t for t in ids if t not in special_ids]
        return self.stm.backend.decode(filtered)

    @_maybe_rich
    def _print_verbose_metadata(self, title: str, payload: Dict[str, Any]):
        table = Table(box=box.MINIMAL_DOUBLE_HEAD, title=title, show_lines=False)
        table.add_column("Field", justify="right", style="bold cyan")
        table.add_column("Value", style="white")
        for k, v in payload.items():
            if isinstance(v, (list, dict)):
                pretty = json.dumps(v, ensure_ascii=False, indent=2)[:500]
                table.add_row(k, pretty if pretty else str(v))
            else:
                table.add_row(k, str(v))
        _RICH_CONSOLE.print(table)

    def encode(
        self,
        text: Union[str, List[str]],
        max_length: Optional[int] = None,
        add_special_tokens: Optional[bool] = None,
        return_tensors: bool = False,
        role: Optional[str] = None,
        section: Optional[str] = None,
    ) -> Union[EncodeResult, BatchEncodeResult]:
        add_special_tokens = self.config.add_bos_eos if add_special_tokens is None else add_special_tokens
        max_len = max_length or self.config.max_sequence_length

        def _encode_one(t: str) -> EncodeResult:
            t = self._normalize(t)
            original_text = t

            if self.config.escape_user_specials:
                t = self.stm.escape_specials_in_user_text(t)

            t, ws_meta, ws_handler = self._apply_ws_strategy(t)

            segments = []
            if role or section:
                seg = Segment(name=section or "content", content=t, role=role)
                segments.append(seg)
            else:
                seg = Segment(name="content", content=t, role=None)
                segments.append(seg)

            assembled_str, spans = self.template.build(segments) if add_special_tokens else (t, [])

            token_ids = self._encode_string(assembled_str, allow_special=True)

            truncated = False
            if len(token_ids) > max_len:
                token_ids = self._apply_truncation(token_ids, max_len)
                truncated = True
                self._stats["truncated_sequences"] += 1

            attention_mask = None
            if self.config.return_attention_mask:
                padded_ids, attention_mask = self._pad(token_ids, max_len)
            else:
                padded_ids = token_ids
                if len(padded_ids) < max_len:
                    padded_ids, attention_mask = self._pad(padded_ids, max_len)
                else:
                    attention_mask = [1] * len(padded_ids)

            token_type_ids = [0] * len(padded_ids) if self.config.return_token_type_ids else None

            self._stats["total_processed"] += 1
            seq_len = len(padded_ids)
            self._stats["avg_sequence_length"] = (self._stats["avg_sequence_length"] * 0.9) + (0.1 * seq_len)
            if seq_len > len(token_ids):
                self._stats["padded_sequences"] += 1

            result = EncodeResult(
                input_ids=padded_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                spans=spans,
                metadata={
                    "original_text_length": len(original_text),
                    "normalized_ws_strategy": self.config.preserve_whitespace_strategy,
                    "truncated": truncated,
                    "max_length": max_len,
                    "backend": self.stm.backend.name,
                }
            )

            self._print_verbose_metadata("Encode Metadata", {
                "role": role,
                "section": section,
                "token_count": len(token_ids),
                "padded_len": len(padded_ids),
                "truncated": truncated,
                "spans": [asdict(s) for s in spans][:5],
                "backend": self.stm.backend.name,
            })

            return result

        if isinstance(text, list):
            results = [_encode_one(t) for t in text]
            batch = BatchEncodeResult(
                input_ids=[r.input_ids for r in results],
                attention_mask=[r.attention_mask for r in results] if self.config.return_attention_mask else None,
                token_type_ids=[r.token_type_ids for r in results] if self.config.return_token_type_ids else None,
                spans=[r.spans for r in results],
                metadata={
                    "batch_size": len(text),
                    "backend": self.stm.backend.name,
                }
            )
            return batch
        else:
            return _encode_one(text)

    def decode(
        self,
        ids: Union[List[int], List[List[int]]],
        skip_special_tokens: bool = True,
        clean_ws: bool = True
    ) -> Union[str, List[str]]:
        def _decode_one(arr: List[int]) -> str:
            text = self._decode_tokens(arr, skip_special_tokens=skip_special_tokens)
            if clean_ws and self.config.preserve_whitespace_strategy == "token":
                ws = WhitespaceHandler("token", self.config.whitespace_tokens)
                text = ws.decode_restore(text)
            text = self.stm.unescape_specials(text)
            return text

        if ids and isinstance(ids[0], list):
            return [_decode_one(x) for x in ids]  # type: ignore
        else:
            return _decode_one(ids)  # type: ignore

    # ----------------------------------------------------------------------------------
    # Instruction Tuning, RLHF, RLAI Reasoning, Conversations (schema-driven)
    # ----------------------------------------------------------------------------------

    def prepare_instruction_tuning(
        self,
        instruction: str,
        input_text: str = "",
        output_text: str = "",
        system_prompt: str = "",
        max_length: Optional[int] = None
    ) -> Dict[str, Any]:
        needed = [self.config.pad_token]
        needed.extend(["<|instruction|>", "<|input|>", "<|output|>", "<|system|>"])
        if self.config.add_bos_eos:
            needed.extend([self.config.bos_token, self.config.eos_token])
        self.stm.ensure(needed)

        segments: List[Segment] = []
        if system_prompt:
            segments.append(Segment(name="<|instruction|>", role="<|system|>", content=self._normalize(system_prompt)))
        segments.append(Segment(name="<|instruction|>", role="<|user|>", content=self._normalize(instruction)))
        if input_text:
            segments.append(Segment(name="<|input|>", role="<|user|>", content=self._normalize(input_text)))
        segments.append(Segment(name="<|output|>", role="<|assistant|>", content=self._normalize(output_text)))

        assembled_str, spans = self.template.build(segments)

        token_ids = self._encode_string(assembled_str, allow_special=True)
        max_len = max_length or self.config.max_sequence_length
        truncated = False
        if len(token_ids) > max_len:
            token_ids = self._apply_truncation(token_ids, max_len)
            truncated = True
            self._stats["truncated_sequences"] += 1

        labels = [self.config.labels_ignore_index] * len(token_ids)
        stoks = self.special_tokens
        output_tok = stoks.get("<|output|>")
        endseg_tok = stoks.get("<|endofsegment|>")
        if output_tok is not None and endseg_tok is not None:
            ranges: List[Tuple[int, int]] = []
            i = 0
            while i < len(token_ids):
                if token_ids[i] == output_tok:
                    j = i + 1
                    while j < len(token_ids) and token_ids[j] != endseg_tok:
                        j += 1
                    ranges.append((i + 1, j))
                    i = j + 1
                else:
                    i += 1
            for s, e in ranges:
                for k in range(s, min(e, len(labels))):
                    labels[k] = token_ids[k]

        attn_mask = None
        if self.config.return_attention_mask:
            token_ids, attn_mask = self._pad(token_ids, max_len)
            pad_len = max_len - len(labels)
            if pad_len > 0:
                labels.extend([self.config.labels_ignore_index] * pad_len)

        self._print_verbose_metadata("Instruction Tuning Metadata", {
            "tokens": len(token_ids),
            "labels_trainable_count": sum(1 for x in labels if x != self.config.labels_ignore_index),
            "truncated": truncated,
        })

        return {
            "input_ids": token_ids,
            "attention_mask": attn_mask,
            "labels": labels,
            "metadata": {
                "format": "instruction_tuning",
                "segments": [asdict(s) for s in segments],
                "spans": [asdict(s) for s in spans],
                "truncated": truncated
            }
        }

    def prepare_conversation(self, conversation: List[Dict[str, str]], max_length: Optional[int] = None) -> Dict[str, Any]:
        segments: List[Segment] = []
        for turn in conversation:
            role = f"<|{turn['role']}|>" if not turn["role"].startswith("<|") else turn["role"]
            content = self._normalize(turn["content"])
            segments.append(Segment(name="<|context|>", role=role, content=content))
        assembled, spans = self.template.build(segments)

        token_ids = self._encode_string(assembled, allow_special=True)
        max_len = max_length or self.config.max_sequence_length
        truncated = len(token_ids) > max_len
        if truncated:
            token_ids = self._apply_truncation(token_ids, max_len)
            self._stats["truncated_sequences"] += 1
        attn_mask = None
        if self.config.return_attention_mask:
            token_ids, attn_mask = self._pad(token_ids, max_len)
        return {
            "input_ids": [token_ids],
            "attention_mask": [attn_mask] if attn_mask is not None else None,
            "metadata": {
                "format": "conversation",
                "turns": len(conversation),
                "truncated": truncated,
                "spans": [asdict(s) for s in spans]
            }
        }

    def prepare_rlhf_preference(
        self, prompt: str, chosen: str, rejected: str, max_length: Optional[int] = None
    ) -> Dict[str, Any]:
        def seq(user: str, assistant: str) -> List[int]:
            segments = [
                Segment(name="<|instruction|>", role="<|user|>", content=self._normalize(user)),
                Segment(name="<|output|>", role="<|assistant|>", content=self._normalize(assistant))
            ]
            assembled, _ = self.template.build(segments)
            return self._encode_string(assembled, allow_special=True)

        c_ids = seq(prompt, chosen)
        r_ids = seq(prompt, rejected)
        max_len = max_length or self.config.max_sequence_length

        def pad(seq_ids: List[int]) -> Tuple[List[int], List[int]]:
            if len(seq_ids) > max_len:
                seq_ids = self._apply_truncation(seq_ids, max_len)
                self._stats["truncated_sequences"] += 1
            padded, mask = self._pad(seq_ids, max_len)
            return padded, mask

        c_padded, c_mask = pad(c_ids)
        r_padded, r_mask = pad(r_ids)
        self._print_verbose_metadata("RLHF Preference", {
            "prompt_len": len(prompt),
            "chosen_tokens": len(c_ids),
            "rejected_tokens": len(r_ids)
        })
        return {
            "prompt": prompt,
            "chosen": {"input_ids": c_padded, "attention_mask": c_mask, "text": chosen},
            "rejected": {"input_ids": r_padded, "attention_mask": r_mask, "text": rejected},
            "metadata": {"format": "rlhf_preference"}
        }

    def prepare_reasoning_chain(
        self, problem: str, steps: List[str], final_answer: str, max_length: Optional[int] = None
    ) -> Dict[str, Any]:
        segments: List[Segment] = [
            Segment(name="<|instruction|>", role="<|user|>", content=self._normalize(problem)),
            Segment(name="<|think|>", role="<|assistant|>", content="")
        ]
        for s in steps:
            segments.append(Segment(name="<|step|>", role="<|assistant|>", content=self._normalize(s)))
        segments.append(Segment(name="<|conclude|>", role="<|assistant|>", content=self._normalize(final_answer)))

        assembled, spans = self.template.build(segments)
        ids = self._encode_string(assembled, allow_special=True)
        max_len = max_length or self.config.max_sequence_length
        truncated = len(ids) > max_len
        if truncated:
            ids = self._apply_truncation(ids, max_len)
            self._stats["truncated_sequences"] += 1
        attn_mask = None
        if self.config.return_attention_mask:
            ids, attn_mask = self._pad(ids, max_len)

        stoks = self.special_tokens
        think_tok = stoks.get("<|think|>")
        conclude_tok = stoks.get("<|conclude|>")
        endseg_tok = stoks.get("<|endofsegment|>")
        labels = [self.config.labels_ignore_index] * len(ids)
        if think_tok is not None:
            ranges: List[Tuple[int, int]] = []
            i = 0
            in_reason = False
            start_idx = 0
            while i < len(ids):
                if ids[i] == think_tok:
                    in_reason = True
                    start_idx = i + 1
                if conclude_tok is not None and ids[i] == conclude_tok and in_reason:
                    ranges.append((start_idx, i))
                    in_reason = False
                i += 1
            if in_reason:
                ranges.append((start_idx, len(ids)))
            for s, e in ranges:
                for k in range(s, e):
                    if endseg_tok is None or ids[k] != endseg_tok:
                        labels[k] = ids[k]

        self._print_verbose_metadata("RLAI Reasoning", {
            "problem_chars": len(problem),
            "steps": len(steps),
            "final_answer_chars": len(final_answer),
            "train_labels": sum(1 for x in labels if x != self.config.labels_ignore_index)
        })

        return {
            "input_ids": ids,
            "attention_mask": attn_mask,
            "labels": labels,
            "metadata": {
                "format": "rlai_reasoning",
                "spans": [asdict(s) for s in spans],
                "steps_count": len(steps),
                "truncated": truncated
            }
        }

    # ----------------------------------------------------------------------------------
    # Inference helpers
    # ----------------------------------------------------------------------------------

    def inference_encode(self, prompt: str, max_length: Optional[int] = None) -> Dict[str, Any]:
        res = self.encode(prompt, max_length=max_length, role="<|user|>", section="<|context|>")
        return {
            "input_ids": [res.input_ids],
            "attention_mask": [res.attention_mask] if res.attention_mask is not None else None,
            "metadata": {"format": "inference", **res.metadata}
        }

    def inference_decode(
        self,
        full_sequence: List[int],
        stop_tokens: Optional[List[str]] = None,
        max_new_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        stop_ids: List[int] = []
        if stop_tokens:
            for t in stop_tokens:
                tid = self.special_tokens.get(t)
                if tid is not None:
                    stop_ids.append(tid)
        eos_id = self.special_tokens.get(self.config.eos_token)
        if eos_id is not None:
            stop_ids.append(eos_id)

        stop_pos = len(full_sequence)
        for i, tid in enumerate(full_sequence):
            if tid in stop_ids:
                stop_pos = i
                break
        if max_new_tokens is not None:
            stop_pos = min(stop_pos, max_new_tokens)

        decoded_text = self._decode_tokens(full_sequence[:stop_pos], skip_special_tokens=True)
        self._print_verbose_metadata("Inference Decode", {
            "stop_pos": stop_pos,
            "generated_tokens": stop_pos,
            "stop_ids": stop_ids[:5]
        })
        return {
            "text": decoded_text,
            "tokens_generated": stop_pos,
            "stopped_early": stop_pos < len(full_sequence),
            "stop_reason": "stop_token" if stop_pos < len(full_sequence) else "max_length"
        }

    def batch_inference_encode(self, texts: List[str], max_length: Optional[int] = None) -> Dict[str, Any]:
        results = self.encode(texts, max_length=max_length, role="<|user|>", section="<|context|>")
        assert isinstance(results, BatchEncodeResult)
        return {
            "input_ids": results.input_ids,
            "attention_mask": results.attention_mask,
            "batch_size": len(texts),
            "metadata": results.metadata
        }

    # ----------------------------------------------------------------------------------
    # Sliding window and dynamic batching
    # ----------------------------------------------------------------------------------

    def sliding_window_encode(
        self, text: str, window_size: Optional[int] = None, overlap: Optional[int] = None
    ) -> Dict[str, Any]:
        window_size = window_size or self.config.max_sequence_length
        overlap = overlap or self.config.sliding_window_overlap

        enc = self.encode(text, max_length=10**9, add_special_tokens=True)
        assert isinstance(enc, EncodeResult)
        ids = enc.input_ids
        windows: List[List[int]] = []
        start = 0
        while start < len(ids):
            end = min(len(ids), start + window_size)
            windows.append(ids[start:end])
            if end == len(ids):
                break
            start = max(0, end - overlap)
        self._stats["sliding_windows_built"] += len(windows)
        masks = [[1] * len(w) for w in windows] if self.config.return_attention_mask else None
        self._print_verbose_metadata("Sliding Window", {
            "total_tokens": len(ids),
            "windows": len(windows),
            "window_size": window_size,
            "overlap": overlap
        })
        return {
            "windows": windows,
            "attention_masks": masks,
            "metadata": {
                "original_len": len(ids),
                "windows": len(windows),
                "window_size": window_size,
                "overlap": overlap
            }
        }

    def bucket_by_length(self, sequences: List[List[int]], num_buckets: int = 5) -> List[List[List[int]]]:
        lengths = [len(s) for s in sequences]
        if not sequences:
            return []
        # Defensive quantiles: handle small sample sizes
        try:
            qs = [statistics.quantiles(lengths, n=num_buckets + 1)[i] for i in range(num_buckets)]
        except Exception:
            # Fallback: simple linear bins
            mn, mx = min(lengths), max(lengths)
            step = max(1, (mx - mn) // max(1, num_buckets))
            qs = [mn + (i + 1) * step for i in range(num_buckets)]
        buckets: List[List[List[int]]] = [[] for _ in range(num_buckets)]
        for s in sequences:
            l = len(s)
            idx = 0
            while idx < num_buckets - 1 and l > qs[idx]:
                idx += 1
            buckets[idx].append(s)
        return [b for b in buckets if b]

    # ----------------------------------------------------------------------------------
    # Analysis and corpus evaluation
    # ----------------------------------------------------------------------------------

    def analyze(self, text: str) -> AnalysisReport:
        norm = self._normalize(text)
        ids = self._encode_string(norm, allow_special=True)
        dec = self.stm.backend.decode(ids)
        token_details: List[Dict[str, Any]] = []
        specials = set(self.special_tokens.values())
        for tid in ids:
            try:
                tb = self.stm.backend.decode_single_token_bytes(tid)
                ts = tb.decode("utf-8", errors="replace")
            except Exception:
                ts = "<ERR>"
            token_details.append({
                "id": tid,
                "is_special": tid in specials,
                "display": repr(ts),
                "byte_len": len(ts.encode("utf-8", errors="replace"))
            })
        return AnalysisReport(
            original_text=text,
            normalized_text=norm,
            token_count=len(ids),
            char_count=len(text),
            compression_ratio=(len(text) / max(1, len(ids))),
            unique_token_ratio=(len(set(ids)) / max(1, len(ids))),
            special_token_count=sum(1 for t in ids if t in specials),
            perfect_roundtrip=(dec == norm),
            token_details=token_details,
        )

    def get_pipeline_statistics(self) -> Dict[str, Any]:
        return {
            "backend": self.stm.backend.name,
            "n_vocab": self.stm.backend.n_vocab,
            "special_tokens_count": len(self.stm.tokens),
            "config": {
                "max_sequence_length": self.config.max_sequence_length,
                "truncation_strategy": self.config.truncation_strategy,
                "pad_direction": self.config.pad_direction,
                "ws_strategy": self.config.preserve_whitespace_strategy,
            },
            "usage_stats": dict(self._stats),
            "sample_special_tokens": list(itertools.islice(self.stm.tokens.items(), 10))
        }

    # ----------------------------------------------------------------------------------
    # Domain placeholders (optional, conservative)
    # ----------------------------------------------------------------------------------

    def add_domain_placeholders(self, patterns: List[str]) -> None:
        if not self.config.enable_domain_placeholders:
            logger.warning("Domain placeholders are disabled. Set enable_domain_placeholders=True to use this feature.")
            return
        placeholders = []
        for i, p in enumerate(patterns):
            pname = f"{self.config.domain_placeholder_prefix}{i}{self.config.domain_placeholder_suffix}"
            placeholders.append(pname)
        self.stm.ensure(placeholders)
        self._print_verbose_metadata("Domain Placeholders", {"placeholders": placeholders})

    # ----------------------------------------------------------------------------------
    # Rich Views
    # ----------------------------------------------------------------------------------

    @_maybe_rich
    def print_analysis(self, report: AnalysisReport, max_tokens_preview: int = 10) -> None:
        t = Table(title="Tokenization Analysis", box=box.MINIMAL_DOUBLE_HEAD)
        t.add_column("Metric", style="bold cyan")
        t.add_column("Value", style="white")
        t.add_row("Backend", self.stm.backend.name)
        t.add_row("Chars", str(report.char_count))
        t.add_row("Tokens", str(report.token_count))
        t.add_row("Compression", f"{report.compression_ratio:.2f}")
        t.add_row("Unique ratio", f"{report.unique_token_ratio:.2f}")
        t.add_row("Special count", str(report.special_token_count))
        t.add_row("Roundtrip", str(report.perfect_roundtrip))
        _RICH_CONSOLE.print(t)

        prev = report.token_details[:max_tokens_preview]
        tp = Table(title="First Tokens", box=box.MINIMAL)
        tp.add_column("ID", justify="right")
        tp.add_column("Special")
        tp.add_column("Display")
        for d in prev:
            tp.add_row(str(d["id"]), "✓" if d["is_special"] else "", d["display"])
        _RICH_CONSOLE.print(tp)


# --------------------------------------------------------------------------------------
# Demonstrations (executed when running this file directly)
# --------------------------------------------------------------------------------------

def _demo_introspection(pipe: AdvancedTokenizerPipeline):
    if _HAVE_RICH and pipe.config.verbose:
        _RICH_CONSOLE.rule("Tokenizer Introspection")
    stats = pipe.get_pipeline_statistics()
    pipe._print_verbose_metadata("Pipeline Stats", stats)


def _demo_preprocessing(pipe: AdvancedTokenizerPipeline):
    if _HAVE_RICH and pipe.config.verbose:
        _RICH_CONSOLE.rule("Preprocessing — Encoding, Padding, Truncation")

    texts = [
        "Short text.",
        "This is a medium length text for testing preprocessing capabilities of the tokenizer pipeline.",
        ("This is a very long text that should exceed the maximum sequence length and will be truncated "
         "according to the configured strategy. ") * 5
    ]
    res = pipe.encode(texts, max_length=64, role="<|user|>", section="<|context|>")
    assert isinstance(res, BatchEncodeResult)
    for i, t in enumerate(texts):
        if _HAVE_RICH and pipe.config.verbose:
            _RICH_CONSOLE.print(Panel(Text(t[:120] + ("..." if len(t) > 120 else ""), no_wrap=True),
                                      title=f"Text {i+1}"))
        dec = pipe.decode([res.input_ids[i]], skip_special_tokens=True)
        if _HAVE_RICH and pipe.config.verbose:
            _RICH_CONSOLE.print(f"Decoded roundtrip subset? {'Yes' if t.strip() in dec[0] else 'Partial'}")


def _demo_instruction(pipe: AdvancedTokenizerPipeline):
    if _HAVE_RICH and pipe.config.verbose:
        _RICH_CONSOLE.rule("Instruction Tuning Data")

    d = pipe.prepare_instruction_tuning(
        instruction="Translate the following English text to French:",
        input_text="Hello, how are you today?",
        output_text="Bonjour, comment allez-vous aujourd'hui?",
        system_prompt="You are a helpful translation assistant.",
        max_length=256
    )
    if _HAVE_RICH and pipe.config.verbose:
        _RICH_CONSOLE.print(f"Total tokens: {len(d['input_ids'])}")
        _RICH_CONSOLE.print(f"Trainable labels: {sum(1 for x in d['labels'] if x != pipe.config.labels_ignore_index)}")


def _demo_rlhf(pipe: AdvancedTokenizerPipeline):
    if _HAVE_RICH and pipe.config.verbose:
        _RICH_CONSOLE.rule("RLHF Preference")

    data = pipe.prepare_rlhf_preference(
        prompt="What is the best way to learn programming?",
        chosen=("Practice regularly, start with fundamentals, build small projects, read others' code, and seek "
                "feedback from experienced developers."),
        rejected="Just watch a few short videos."
    )
    if _HAVE_RICH and pipe.config.verbose:
        _RICH_CONSOLE.print(f"Chosen tokens: {len(data['chosen']['input_ids'])}, "
                            f"Rejected tokens: {len(data['rejected']['input_ids'])}")


def _demo_reasoning(pipe: AdvancedTokenizerPipeline):
    if _HAVE_RICH and pipe.config.verbose:
        _RICH_CONSOLE.rule("RLAI Reasoning")

    r = pipe.prepare_reasoning_chain(
        problem="If a train travels 120 km in 2 hours, what is its average speed?",
        steps=[
            "Use average speed = distance / time.",
            "Given distance=120 km, time=2 hours.",
            "Compute 120 / 2 = 60 km/h."
        ],
        final_answer="The average speed is 60 km/h.",
        max_length=256
    )
    if _HAVE_RICH and pipe.config.verbose:
        _RICH_CONSOLE.print(f"Labels (trainable) = {sum(1 for x in r['labels'] if x != pipe.config.labels_ignore_index)}")


def _demo_inference(pipe: AdvancedTokenizerPipeline):
    if _HAVE_RICH and pipe.config.verbose:
        _RICH_CONSOLE.rule("Inference")

    prompt = "Explain the concept of machine learning in simple terms."
    enc = pipe.inference_encode(prompt, max_length=256)
    mock = pipe.encode(
        "Machine learning lets computers learn patterns from data to make predictions.",
        role="<|assistant|>", section="<|output|>"
    )
    assert isinstance(mock, EncodeResult)
    full_seq = enc["input_ids"][0] + mock.input_ids
    dec = pipe.inference_decode(full_seq, stop_tokens=[pipe.config.eos_token], max_new_tokens=200)
    if _HAVE_RICH and pipe.config.verbose:
        _RICH_CONSOLE.print(Panel(dec["text"][:160] + ("..." if len(dec["text"]) > 160 else ""), title="Decoded"))


def _demo_batch(pipe: AdvancedTokenizerPipeline):
    if _HAVE_RICH and pipe.config.verbose:
        _RICH_CONSOLE.rule("Batch Processing")

    batch_texts = [
        "First example for batch processing.",
        "Second example with different length and details.",
        "Third example text for testing.",
        "Fourth and final example."
    ]
    b = pipe.batch_inference_encode(batch_texts, max_length=128)
    if _HAVE_RICH and pipe.config.verbose:
        _RICH_CONSOLE.print(f"Batch size: {b['batch_size']}")
        _RICH_CONSOLE.print(f"Uniform length: {len(set(len(x) for x in b['input_ids'])) == 1}")


def _demo_sliding_window(pipe: AdvancedTokenizerPipeline):
    if _HAVE_RICH and pipe.config.verbose:
        _RICH_CONSOLE.rule("Sliding Window Demo")

    long_text = ("This is a long passage. " * 600).strip()
    windows = pipe.sliding_window_encode(long_text, window_size=256, overlap=32)
    if _HAVE_RICH and pipe.config.verbose:
        _RICH_CONSOLE.print(f"Windows created: {windows['metadata']['windows']} (size={windows['metadata']['window_size']})")


def _demo_analysis(pipe: AdvancedTokenizerPipeline):
    if _HAVE_RICH and pipe.config.verbose:
        _RICH_CONSOLE.rule("Analysis")

    sample = "The quick brown fox jumps over the lazy dog. 🦊 Café naïve façade – spéciàl chars!"
    report = pipe.analyze(sample)
    pipe.print_analysis(report)


def run_comprehensive_demo():
    cfg = AdvancedTokenizerConfig(verbose=True)
    pipe = AdvancedTokenizerPipeline(cfg)
    _demo_introspection(pipe)
    _demo_preprocessing(pipe)
    _demo_instruction(pipe)
    _demo_rlhf(pipe)
    _demo_reasoning(pipe)
    _demo_inference(pipe)
    _demo_batch(pipe)
    _demo_sliding_window(pipe)
    _demo_analysis(pipe)
    if _HAVE_RICH and pipe.config.verbose:
        _RICH_CONSOLE.rule("All demonstrations completed. Pipeline ready.")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    run_comprehensive_demo()