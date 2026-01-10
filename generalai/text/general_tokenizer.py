#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced, generalized, dynamic tokenizer toolkit (single-file .py)

Overview
--------
This file implements a production-grade, highly dynamic tokenization toolkit that adapts
to arbitrary Hugging Face tokenizers (AutoTokenizer) and covers nuanced needs of modern LLM usage:
- Works with any pretrained tokenizer (zero hard-coded templates).
- Introspects tokenizer capabilities dynamically (chat template availability, fast/slow, vocab, specials).
- Encodes rich conversational structures: system/user/assistant, context, instruction, reasoning, tool/function-call, agentic roles, custom roles, and response nature.
- Provides exact, metadata-rich tracing of character spans and token spans for each segment, where possible.
- Offers robust normalization (NFKC), whitespace-preserving modes, Unicode safety, and visibility options.
- Minimizes OOV/Unicode fragmentation impact with optional domain adaptation via train_new_from_iterator.
- Addresses common pitfalls: OOV, whitespace loss, truncation risk, padding waste, byte fallback fragmentation,
  special-token distortion, language bias, Unicode corruption, rigid templates, corpus dependence.

Design principles
-----------------
- Zero hard-coded, single-purpose templates. All schemas are flexible, placeholder-driven, and inspectable.
- “Minimal injection” policy for special markers: we avoid adding special tokens unless explicitly requested.
- Metadata-first: every encode path can emit exhaustive structured telemetry via the rich console.
- Compatible with chat and non-chat tokenizers, fast or slow.
- Safe defaults for pad/bos/eos and truncation sides, with transparent reporting.

Quick start
-----------
- Requires: transformers, rich (and optionally regex for grapheme).
- Run: python advanced_tokenizer.py --model gpt2 --demo all
- Toggle verbose metadata: --verbose true|false
- Try chat-style encoding even for non-chat models (uses fallback labeled template).

Implementation notes
--------------------
- For chat-capable models with a native chat_template, you can opt to use it.
  For complete segment-boundary tracing, the fallback template is recommended.
- Domain adaptation retrains a new tokenizer head from a corpus iterator (fast tokenizers only).
- Sliding-window encoding with stride prevents truncation of critical regions.
- Dynamic batching/padding via tokenizer.pad ensures minimal padding waste and valid attention masks.
- Whitespace preservation mode encodes visible whitespace tags in-band (schema tokens can be added if requested).
- Unicode normalization (NFKC) is applied consistently to avoid corruption.

Examples
--------
- Basic multilingual + emoji encode/decode with offsets and token spans.
- Chat multi-role message encoding with segment boundary tracing.
- Function/tool-call role blocks.
- Sliding window encoding for long contexts.
- Dynamic batching and padding.
- Optional domain adaptation on a small synthetic corpus.

"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import textwrap
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

# Optional regex module for Unicode grapheme clustering; fallback to re if missing
try:
    import regex as regx  # supports \X
    _HAS_REGEX = True
except Exception:  # pragma: no cover
    import re as regx
    _HAS_REGEX = False

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.json import JSON
from rich.text import Text
from rich.pretty import Pretty
from rich import box

try:
    from transformers import AutoTokenizer, PreTrainedTokenizerBase
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "This script requires 'transformers'. Install via: pip install transformers"
    ) from exc


# ---------------------------
# Data models and structures
# ---------------------------

@dataclass
class Message:
    """
    Generic chat message structure compatible with both chat and non-chat models.
    - role: "system" | "user" | "assistant" | "tool" | "function" | "context" | "instruction" | "reasoning" | custom
    - name: optional tag for tools/functions/agents
    - content: text payload for the segment
    - meta: free-form metadata for tracing or agentic hints
    """
    role: str
    content: str
    name: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SegmentBoundary:
    """
    Represents a semantically labeled span in both character and token space.
    Computed best-effort using offsets from fast tokenizers; falls back gracefully.
    """
    tag: str
    char_start: int
    char_end: int
    tok_start: Optional[int] = None
    tok_end: Optional[int] = None


@dataclass
class EncodeSummary:
    """
    Encapsulation of one encoded sequence with telemetry for debugging/tracing and reconstruction.
    """
    input_ids: List[int]
    attention_mask: List[int]
    tokens: List[str]
    special_tokens_mask: Optional[List[int]] = None
    type_ids: Optional[List[int]] = None
    offsets: Optional[List[Tuple[int, int]]] = None  # character offsets per token (fast tokenizers)
    text: Optional[str] = None
    segments: List[SegmentBoundary] = field(default_factory=list)
    truncated: bool = False
    overflow: int = 0
    model_max_length: Optional[int] = None


@dataclass
class WindowedEncoding:
    """
    Stores a list of sliding windows on a long sequence.
    """
    windows: List[EncodeSummary]
    stride: int
    window_size: int
    total_len: int


# ---------------------------
# Utility helpers
# ---------------------------

def nfkc_normalize(text: str) -> str:
    """
    Apply Unicode NFKC normalization to reduce ambiguity and avoid Unicode corruption.
    """
    return unicodedata.normalize("NFKC", text)


def visible_whitespace(text: str) -> str:
    """
    Convert whitespace into visible in-band markers that remain tokenizable,
    helping preserve structure while avoiding custom special tokens by default.
    """
    # Replace sequences to minimize injection. Example tags:
    # ␠ for space, ␉ for tab, ␤ for newline; these are visible whitespace symbols.
    # Many tokenizers will split these reasonably without needing added tokens.
    text = text.replace("\t", "␉")
    # Preserve CRLF pairs first
    text = text.replace("\r\n", "␍␊")
    text = text.replace("\r", "␍")
    text = text.replace("\n", "␊")
    # Optionally handle runs of spaces
    # Use a compact visible char per space to avoid inflating sequence length drastically.
    return text.replace(" ", "␠")


def restore_visible_whitespace(text: str) -> str:
    """
    Reverse of visible_whitespace: turn markers back into actual whitespace.
    """
    text = text.replace("␉", "\t")
    text = text.replace("␍␊", "\r\n")
    text = text.replace("␍", "\r")
    text = text.replace("␊", "\n")
    return text.replace("␠", " ")


def graphemes(text: str) -> List[str]:
    """
    Split text into Unicode grapheme clusters for robust character-level fallback where needed.
    """
    if _HAS_REGEX:
        return regx.findall(r"\X", text)
    # Fallback: naive split, not as accurate but keeps code standalone
    return list(text)


def chunk_tokens(seq: List[int], size: int, stride: int) -> List[Tuple[int, int]]:
    """
    Compute start/end token-index ranges for sliding windows.
    """
    ranges = []
    start = 0
    n = len(seq)
    while start < n:
        end = min(start + size, n)
        ranges.append((start, end))
        if end == n:
            break
        start = max(start + size - stride, 0)
        if start <= 0 and size < n:
            # ensure forward progress
            start = size - stride
    return ranges


def safe_len(x: Optional[Sequence[Any]]) -> int:
    return len(x) if x is not None else 0


# ---------------------------
# Special schema (minimal injection)
# ---------------------------

@dataclass
class SpecialSchema:
    """
    A flexible, minimal-injection schema for labeling conversational or structural
    segments in-band. By default, uses textual delim strings rather than adding
    new special tokens (to avoid distorting model distributions).

    If you WANT to add as special tokens, set add_as_special=True when passing to the manager.
    """
    # Delimiter tokens (textual, by default)
    begin: str = "<|BEGIN|>"
    end: str = "<|END|>"

    # Role labels
    role_begin_fmt: str = "<|{role}_BEGIN|>"
    role_end_fmt: str = "<|{role}_END|>"

    # Explicit semantic segments
    begin_context: str = "<|CONTEXT_BEGIN|>"
    end_context: str = "<|CONTEXT_END|>"
    begin_instruction: str = "<|INSTRUCTION_BEGIN|>"
    end_instruction: str = "<|INSTRUCTION_END|>"
    begin_reasoning: str = "<|REASONING_BEGIN|>"
    end_reasoning: str = "<|REASONING_END|>"
    begin_response: str = "<|RESPONSE_BEGIN|>"
    end_response: str = "<|RESPONSE_END|>"

    # Tool/function-call
    begin_tool: str = "<|TOOL_BEGIN|>"
    end_tool: str = "<|TOOL_END|>"
    begin_function: str = "<|FUNCTION_BEGIN|>"
    end_function: str = "<|FUNCTION_END|>"

    # Whether to attempt registering these as special tokens on the tokenizer
    add_as_special: bool = False

    def delim_for_role(self, role: str) -> Tuple[str, str]:
        return self.role_begin_fmt.format(role=role.upper()), self.role_end_fmt.format(role=role.upper())

    def all_tokens(self) -> List[str]:
        roles = ["SYSTEM", "USER", "ASSISTANT", "TOOL", "FUNCTION", "CONTEXT", "INSTRUCTION", "REASONING", "RESPONSE"]
        tokens = [
            self.begin, self.end,
            self.begin_context, self.end_context,
            self.begin_instruction, self.end_instruction,
            self.begin_reasoning, self.end_reasoning,
            self.begin_response, self.end_response,
            self.begin_tool, self.end_tool,
            self.begin_function, self.end_function,
        ]
        for r in roles:
            tokens.append(self.role_begin_fmt.format(role=r))
            tokens.append(self.role_end_fmt.format(role=r))
        return list(dict.fromkeys(tokens))  # unique preserve order


# ---------------------------
# Verbose reporter (rich)
# ---------------------------

class Reporter:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.console = Console(highlight=True, force_jupyter=False)

    def rule(self, title: str) -> None:
        if self.enabled:
            self.console.rule(Text(title, style="bold cyan"))

    def panel(self, title: str, content: Union[str, Dict[str, Any], Any], style: str = "green") -> None:
        if not self.enabled:
            return
        if isinstance(content, (dict, list)):
            rendered = JSON.from_data(content, indent=2, ensure_ascii=False)
        else:
            rendered = Text(str(content))
        self.console.print(Panel(rendered, title=title, title_align="left", border_style=style))

    def table(self, title: str, columns: List[str], rows: List[List[Any]], style: str = "cyan") -> None:
        if not self.enabled:
            return
        t = Table(title=title, title_style="bold", box=box.SIMPLE_HEAVY)
        for c in columns:
            t.add_column(c)
        for r in rows:
            t.add_row(*[str(x) for x in r])
        self.console.print(t)

    def pretty(self, obj: Any, title: Optional[str] = None, style: str = "magenta") -> None:
        if not self.enabled:
            return
        content = Pretty(obj, expand_all=False)
        if title:
            self.console.print(Panel(content, title=title, title_align="left", border_style=style))
        else:
            self.console.print(content)

    def warn(self, msg: str) -> None:
        if self.enabled:
            self.console.print(f"[yellow]Warning:[/yellow] {msg}")

    def info(self, msg: str) -> None:
        if self.enabled:
            self.console.print(f"[blue]Info:[/blue] {msg}")

    def error(self, msg: str) -> None:
        if self.enabled:
            self.console.print(f"[red]Error:[/red] {msg}")


# ---------------------------
# Core manager
# ---------------------------

class AdvancedTokenizer:
    """
    Generalized tokenizer manager that wraps any HF AutoTokenizer with:
    - Normalization, whitespace preservation, Unicode safeguards.
    - Chat/non-chat templating with labeled segments and precise span metadata.
    - Sliding window encoding, padding-aware dynamic batching.
    - Domain adaptation via train_new_from_iterator.
    - Rich metadata reporting for transparency.

    Notes on special tokens:
    - By default, we avoid adding new special tokens (to prevent distribution shift).
    - You may opt-in to register schema delimiters as special tokens via schema.add_as_special=True.
    - Pad token is ensured (falls back to eos if missing), to avoid padding waste and attention mask issues.
    """

    def __init__(
        self,
        model_name_or_path: str,
        *,
        schema: Optional[SpecialSchema] = None,
        prefer_fallback_template: bool = True,
        normalize_nfkc: bool = True,
        preserve_ws: bool = False,
        verbose: bool = True,
    ) -> None:
        self.reporter = Reporter(verbose)
        self.model_name = model_name_or_path
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.is_fast = bool(getattr(self.tokenizer, "is_fast", False))
        self.normalize_nfkc = normalize_nfkc
        self.preserve_ws = preserve_ws
        self.prefer_fallback_template = prefer_fallback_template
        self.schema = schema or SpecialSchema()
        self._register_schema_tokens_if_requested()
        self._ensure_pad_token()
        self._introspect()

    # ---- internal setup ----

    def _register_schema_tokens_if_requested(self) -> None:
        if not self.schema.add_as_special:
            return
        # Avoid duplication; only add tokens not present
        existing = set(self.tokenizer.get_vocab().keys()) | set(self.tokenizer.all_special_tokens)
        to_add = [tok for tok in self.schema.all_tokens() if tok not in existing]
        if to_add:
            self.reporter.info(f"Registering {len(to_add)} schema tokens as special tokens.")
            self.tokenizer.add_special_tokens({"additional_special_tokens": to_add})
        else:
            self.reporter.info("All schema tokens already present; no additions needed.")

    def _ensure_pad_token(self) -> None:
        if self.tokenizer.pad_token is None:
            # Prefer eos as pad to avoid random new tokens
            eos = self.tokenizer.eos_token
            if eos is not None:
                self.tokenizer.pad_token = eos
                self.reporter.info(f"pad_token not set; using eos_token='{eos}' as pad.")
            else:
                # Fall back to unk if eos missing
                unk = self.tokenizer.unk_token
                if unk is None:
                    # Last resort: add a [PAD] token minimally
                    self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                    self.reporter.warn("Created new [PAD] token; embeddings extended (may affect pretrained behavior).")
                else:
                    self.tokenizer.pad_token = unk
                    self.reporter.info(f"pad_token not set; using unk_token='{unk}' as pad.")

    def _introspect(self) -> None:
        tok = self.tokenizer
        summary = {
            "name_or_path": getattr(tok, "name_or_path", "<unknown>"),
            "is_fast": bool(getattr(tok, "is_fast", False)),
            "vocab_size": getattr(tok, "vocab_size", None),
            "model_max_length": getattr(tok, "model_max_length", None),
            "padding_side": getattr(tok, "padding_side", None),
            "truncation_side": getattr(tok, "truncation_side", None),
            "bos_token": tok.bos_token,
            "eos_token": tok.eos_token,
            "unk_token": tok.unk_token,
            "pad_token": tok.pad_token,
            "all_special_tokens": tok.all_special_tokens,
            "has_chat_template": hasattr(tok, "chat_template") and tok.chat_template is not None,
        }
        self.reporter.panel("Tokenizer Introspection", summary, style="cyan")

    # ---- normalization / whitespace ----

    def _preprocess_text(self, text: str) -> str:
        original = text
        if self.normalize_nfkc:
            text = nfkc_normalize(text)
        if self.preserve_ws:
            # Visible whitespace; keep structure; reversible via restore_visible_whitespace
            text = visible_whitespace(text)
        # Avoid collapsing leading/trailing whitespace by normalization loss
        if text != original:
            self.reporter.info("Applied normalization/whitespace transformations to input text.")
        return text

    def _postprocess_text(self, text: str) -> str:
        if self.preserve_ws:
            text = restore_visible_whitespace(text)
        return text

    # ---- templating ----

    def _fallback_chat_template(self, messages: List[Message]) -> Tuple[str, List[SegmentBoundary]]:
        """
        Build a labeled string using the schema delimiters without registering special tokens,
        capturing exact char spans for each role/content block.
        """
        segs: List[SegmentBoundary] = []
        parts: List[str] = []
        cursor = 0

        def append_with_span(tag: str, s: str) -> None:
            nonlocal cursor
            start = cursor
            parts.append(s)
            cursor += len(s)
            segs.append(SegmentBoundary(tag=tag, char_start=start, char_end=cursor))

        # Begin entire dialog
        append_with_span("BEGIN", self.schema.begin + "\n")

        for msg in messages:
            role_begin, role_end = self.schema.delim_for_role(msg.role)
            header = f"{role_begin}\n"
            append_with_span(f"{msg.role.upper()}_BEGIN", header)

            name_line = ""
            if msg.name:
                name_line = f"@name: {msg.name}\n"
                append_with_span(f"{msg.role.upper()}_NAME", name_line)

            # Optional meta
            if msg.meta:
                meta_json = json.dumps(msg.meta, ensure_ascii=False)
                meta_line = f"@meta: {meta_json}\n"
                append_with_span(f"{msg.role.upper()}_META", meta_line)

            # Body content
            body = msg.content
            body = self._preprocess_text(body)
            append_with_span(f"{msg.role.upper()}_CONTENT", body + "\n")

            # Role end
            append_with_span(f"{msg.role.upper()}_END", role_end + "\n")

        # End entire dialog
        append_with_span("END", self.schema.end)
        final_text = "".join(parts)
        return final_text, segs

    def _maybe_native_chat_template(self, messages: List[Message], add_generation_prompt: bool = False) -> Optional[str]:
        tok = self.tokenizer
        if not (hasattr(tok, "apply_chat_template") and tok.chat_template and not self.prefer_fallback_template):
            return None
        # Translate Message -> dict format expected by HF chat templates
        mlist = []
        for m in messages:
            d = {"role": m.role, "content": self._preprocess_text(m.content)}
            if m.name:
                d["name"] = m.name
            if m.meta:
                d["meta"] = m.meta
            mlist.append(d)
        try:
            return tok.apply_chat_template(
                mlist,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception as exc:
            self.reporter.warn(f"apply_chat_template failed; using fallback. Reason: {exc}")
            return None

    # ---- encode/decode primitives ----

    def encode_text(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = True,
        return_offsets: bool = True,
    ) -> EncodeSummary:
        """
        Encode a raw text string with optional truncation, returning detailed metadata.
        """
        tok = self.tokenizer
        text_in = self._preprocess_text(text)
        kw = dict(
            text=text_in,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            return_tensors=None,
        )
        if max_length is not None:
            kw["max_length"] = max_length
        if self.is_fast and return_offsets:
            kw["return_offsets_mapping"] = True

        encoded = tok.encode_plus(**kw)
        # Ensure required keys exist
        input_ids: List[int] = list(encoded["input_ids"])
        attn: List[int] = list(encoded.get("attention_mask", [1] * len(input_ids)))
        st_mask: Optional[List[int]] = None
        if "special_tokens_mask" in encoded:
            st_mask = list(encoded["special_tokens_mask"])
        tokens = tok.convert_ids_to_tokens(input_ids)
        offsets = encoded.get("offset_mapping", None)
        truncated = False
        overflow = 0

        if "num_truncated_tokens" in encoded:
            overflow = int(encoded["num_truncated_tokens"])
            truncated = overflow > 0
        elif max_length is not None and len(input_ids) > max_length:
            truncated = True
            overflow = len(input_ids) - max_length

        return EncodeSummary(
            input_ids=input_ids,
            attention_mask=attn,
            tokens=tokens,
            special_tokens_mask=st_mask,
            offsets=offsetsets_to_list(offsets),
            text=text_in,
            segments=[],
            truncated=truncated,
            overflow=overflow,
            model_max_length=getattr(tok, "model_max_length", None),
        )

    def encode_messages(
        self,
        messages: List[Message],
        *,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = True,
        prefer_native_template: Optional[bool] = None,
        add_generation_prompt: bool = False,
    ) -> EncodeSummary:
        """
        Encode a message list either via native chat template (when available) or our fallback labeled template.
        Returns a fully annotated EncodeSummary with segment boundaries where possible.
        """
        if prefer_native_template is None:
            prefer_native_template = not self.prefer_fallback_template

        text: Optional[str] = None
        segments: List[SegmentBoundary] = []
        if prefer_native_template:
            text = self._maybe_native_chat_template(messages, add_generation_prompt=add_generation_prompt)

        if text is None:
            text, segments = self._fallback_chat_template(messages)

        enc = self.encode_text(
            text,
            add_special_tokens=add_special_tokens,
            max_length=max_length,
            truncation=truncation,
            return_offsets=True,
        )

        # Compute token span boundaries per segment via offsets mapping (fast tokenizers)
        if enc.offsets:
            for seg in segments:
                tok_span = offsets_to_token_span(enc.offsets, seg.char_start, seg.char_end)
                seg.tok_start, seg.tok_end = tok_span
        enc.segments = segments

        # Report metadata
        self._report_encode("Chat/Structured Encoding", enc, messages=messages)
        return enc

    def decode(self, input_ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        """
        Decode token ids back to text with optional postprocessing (restore whitespace when enabled).
        """
        text = self.tokenizer.decode(input_ids, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=False)
        return self._postprocess_text(text)

    # ---- sliding window and dynamic batching ----

    def sliding_window_encode(
        self,
        text_or_messages: Union[str, List[Message]],
        *,
        window_size: int,
        stride: int,
        add_special_tokens: bool = True,
        truncation: bool = False,
    ) -> WindowedEncoding:
        """
        Create overlapping windows to mitigate truncation risk.
        - For messages: windows are created over the tokenized full sequence for simplicity.
        - Returns windowed encodings with consistent attention masks.
        """
        if isinstance(text_or_messages, str):
            base = self.encode_text(
                text_or_messages,
                add_special_tokens=add_special_tokens,
                max_length=None,
                truncation=False,
                return_offsets=True,
            )
        else:
            base = self.encode_messages(
                text_or_messages,
                add_special_tokens=add_special_tokens,
                max_length=None,
                truncation=False,
            )

        ids = base.input_ids
        ranges = chunk_tokens(ids, window_size, stride)
        wins: List[EncodeSummary] = []

        for (s, e) in ranges:
            slice_ids = ids[s:e]
            # Build attention mask fully on; no padding within window
            attn = [1] * len(slice_ids)
            tokens = self.tokenizer.convert_ids_to_tokens(slice_ids)
            # Map offsets if known
            offsets = None
            if base.offsets:
                offsets = base.offsets[s:e]
            wins.append(
                EncodeSummary(
                    input_ids=slice_ids,
                    attention_mask=attn,
                    tokens=tokens,
                    offsets=offsets,
                    text=None,
                    segments=[],
                    truncated=False,
                    overflow=0,
                    model_max_length=getattr(self.tokenizer, "model_max_length", None),
                )
            )

        meta = {
            "total_tokens": len(ids),
            "windows": len(wins),
            "window_size": window_size,
            "stride": stride,
        }
        self.reporter.panel("Sliding Window Metadata", meta, style="blue")
        return WindowedEncoding(windows=wins, stride=stride, window_size=window_size, total_len=len(ids))

    def pad_batch(self, batch: List[Dict[str, Any]], pad_to_multiple_of: Optional[int] = None) -> Dict[str, Any]:
        """
        Dynamic batching with minimal padding waste using tokenizer.pad.
        Provide a list of dicts with 'input_ids' and (optional) 'attention_mask', etc.
        """
        padded = self.tokenizer.pad(
            batch,
            padding=True,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=None,
        )
        # Telemetry
        max_len = max(len(x) for x in padded["input_ids"])
        rows = []
        for i, ids in enumerate(padded["input_ids"]):
            pad_count = max_len - len([z for z in batch[i]["input_ids"]])
            rows.append([i, len(batch[i]["input_ids"]), max_len, pad_count])
        self.reporter.table("Batch Padding Summary", ["idx", "orig_len", "padded_len", "pad_added"], rows)
        return padded

    # ---- domain adaptation ----

    def adapt_from_iterator(
        self,
        text_iterator: Iterable[str],
        new_vocab_size: int = 64000,
        min_frequency: int = 2,
        special_tokens: Optional[List[str]] = None,
    ) -> Optional[PreTrainedTokenizerBase]:
        """
        Domain adaptation: train a new tokenizer on top of an iterator.
        - Only available for fast tokenizers with a train_new_from_iterator method.
        - This does not alter the underlying model weights; intended for experimentation or building a new head.
        - For safety, we do not auto-assign this new tokenizer as active; we return it for explicit opt-in.

        Returns the new tokenizer instance or None if unsupported.
        """
        tok = self.tokenizer
        if not hasattr(tok, "train_new_from_iterator"):
            self.reporter.warn("Tokenizer does not support train_new_from_iterator; adaptation skipped.")
            return None

        self.reporter.info(f"Training new tokenizer from iterator (vocab_size={new_vocab_size}, min_freq={min_frequency})...")
        try:
            new_tok = tok.train_new_from_iterator(
                iterator=(self._preprocess_text(x) for x in text_iterator),
                vocab_size=new_vocab_size,
                min_frequency=min_frequency,
                special_tokens=special_tokens or tok.all_special_tokens,
            )
        except Exception as exc:
            self.reporter.error(f"Adaptation failed: {exc}")
            return None

        # Basic sanity report
        summary = {
            "orig_vocab": getattr(tok, "vocab_size", None),
            "new_vocab": getattr(new_tok, "vocab_size", None),
            "orig_fast": bool(getattr(tok, "is_fast", False)),
            "new_fast": bool(getattr(new_tok, "is_fast", False)),
        }
        self.reporter.panel("Adaptation Result", summary, style="magenta")
        return new_tok

    # ---- analysis ----

    def _report_encode(self, title: str, enc: EncodeSummary, messages: Optional[List[Message]] = None) -> None:
        table_rows = []
        num_specials = sum(enc.special_tokens_mask or []) if enc.special_tokens_mask else 0
        table_rows.append(["tokens", len(enc.input_ids)])
        table_rows.append(["special_tokens", num_specials])
        table_rows.append(["truncated", enc.truncated])
        table_rows.append(["overflow", enc.overflow])
        table_rows.append(["model_max_length", enc.model_max_length])
        self.reporter.table(title + " · Summary", ["metric", "value"], table_rows, style="green")

        # Segment spans
        if enc.segments:
            rows = []
            for s in enc.segments:
                rows.append([
                    s.tag,
                    f"{s.char_start}:{s.char_end}",
                    f"{s.tok_start}:{s.tok_end}" if s.tok_start is not None else "n/a",
                ])
            self.reporter.table("Segment Boundaries", ["tag", "char_span", "tok_span"], rows, style="cyan")

        # Token preview
        preview = {
            "first_32_tokens": enc.tokens[:32],
            "last_32_tokens": enc.tokens[-32:],
        }
        self.reporter.panel("Token Preview", preview, style="blue")


# ---------------------------
# Offsets helpers
# ---------------------------

def offsets_to_token_span(offsets: List[Tuple[int, int]], char_start: int, char_end: int) -> Tuple[int, int]:
    """
    Given HF offsets (per token) and a character span [char_start, char_end),
    return best-fit token indices [tok_start, tok_end) for that span.
    """
    tok_start = None
    tok_end = None
    for i, (s, e) in enumerate(offsets):
        if s <= char_start < e or (char_start == e and e > s):
            tok_start = i if tok_start is None else tok_start
        if s < char_end <= e or (char_end == s and e > s):
            tok_end = i + 1
            break
        if s >= char_start and e <= char_end:
            tok_start = i if tok_start is None else tok_start
            tok_end = i + 1
    if tok_start is None:
        # find first token after char_start
        for i, (s, e) in enumerate(offsets):
            if s >= char_start:
                tok_start = i
                break
    if tok_end is None:
        # find first token whose end >= char_end
        for i, (s, e) in enumerate(offsets):
            if e >= char_end:
                tok_end = i + 1
                break
    if tok_start is None:
        tok_start = 0
    if tok_end is None:
        tok_end = len(offsets)
    return tok_start, tok_end


def offsetsets_to_list(offsets: Any) -> Optional[List[Tuple[int, int]]]:
    if offsets is None:
        return None
    return [(int(s), int(e)) for (s, e) in offsets]


# ---------------------------
# Demonstrations (examples)
# ---------------------------

def demo_basic(at: AdvancedTokenizer) -> None:
    at.reporter.rule("Demo · Basic Multilingual + Emoji Encode/Decode")
    sample = "Hello, 世界 🌍🚀\nTabs\tand spaces  preserved?"
    enc = at.encode_text(sample, add_special_tokens=True, max_length=None, truncation=False)
    at._report_encode("Basic Encoding", enc)
    dec = at.decode(enc.input_ids, skip_special_tokens=False)
    at.reporter.panel("Decoded Text (postprocessed)", dec, style="green")


def demo_chat(at: AdvancedTokenizer) -> None:
    at.reporter.rule("Demo · Chat Roles, Instructions, Reasoning, Tools")
    messages = [
        Message(role="system", content="You are a reliable assistant. Follow the given instruction schema."),
        Message(role="context", content="Customer profile: locale=de-DE, loyalty_tier=gold"),
        Message(role="instruction", content="Summarize the user's request and propose next steps."),
        Message(role="user", content="I need to reschedule my appointment from 3pm to next Monday morning."),
        Message(role="reasoning", content="Extract time delta and constraints; consider timezone; produce 2 alternatives."),
        Message(role="assistant", content="Sure. Let me check available slots..."),
        Message(role="tool", name="calendar.search_slots", content='{"day":"Monday","period":"morning"}', meta={"kind": "function_call"}),
        Message(role="assistant", content="Two slots found: 9:30 and 11:00. Which do you prefer?"),
    ]
    enc = at.encode_messages(messages, add_special_tokens=True, max_length=512, truncation=True)
    # decode for demo
    text = at.decode(enc.input_ids, skip_special_tokens=False)
    at.reporter.panel("Round-trip Decoded Chat Text", text, style="green")


def demo_sliding_window(at: AdvancedTokenizer) -> None:
    at.reporter.rule("Demo · Sliding Window Encoding on Long Text")
    long_text = " ".join(["[para] " + ("lorem ipsum " * 50)] * 20)
    windows = at.sliding_window_encode(long_text, window_size=128, stride=48, add_special_tokens=True)
    # Print first 2 windows
    for i, w in enumerate(windows.windows[:2]):
        at._report_encode(f"Window #{i}", w)


def demo_dynamic_batch(at: AdvancedTokenizer) -> None:
    at.reporter.rule("Demo · Dynamic Batching + Padding")
    texts = [
        "Short.",
        "A bit longer sentence with more tokens.",
        "This one is much, much longer and will require more padding when batched together. " * 2,
    ]
    encs = [at.encode_text(t, add_special_tokens=True, truncation=False) for t in texts]
    features = [{"input_ids": e.input_ids, "attention_mask": e.attention_mask} for e in encs]
    padded = at.pad_batch(features, pad_to_multiple_of=8)
    at.reporter.panel("Batch Padded Shapes", {k: [len(x) for x in v] for k, v in padded.items()}, style="magenta")


def demo_adaptation(at: AdvancedTokenizer) -> None:
    at.reporter.rule("Demo · Domain Adaptation (Small Synthetic Corpus)")
    corpus = [
        # code-like, emojis, domain jargon
        "def tokenize_über(text: str) -> List[str]: pass  # 🧪🚀",
        "OrderID: ABC-12345-XYZ; status=PENDING; retry_count=3",
        "GraphQL{query: { product(id: \"SKU-Ω-9000\") { price currency } }}",
        "Συνάρτηση f(x)=x²+1; 测试; prueba; тест",
        "function callTool(name, args){ return { ok:true, data:args }; }",
    ] * 100  # replicate to enforce frequencies

    new_tok = at.adapt_from_iterator(corpus, new_vocab_size=32000, min_frequency=2)
    if new_tok is not None:
        # Quick comparison encode on a domain string
        test = "callTool(\"compute\", {vector: [1,2,3]}) // Ω 🚀"
        base = at.encode_text(test, truncation=False)
        new_ids = new_tok.encode(test, add_special_tokens=True)
        at.reporter.table(
            "Adaptation Compare",
            ["metric", "base", "adapted"],
            [
                ["vocab_size", getattr(at.tokenizer, "vocab_size", None), getattr(new_tok, "vocab_size", None)],
                ["token_count", len(base.input_ids), len(new_ids)],
            ],
        )
        at.reporter.panel("Adapted Tokens (preview)", new_tok.convert_ids_to_tokens(new_ids)[:48], style="yellow")


# ---------------------------
# CLI
# ---------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Advanced generalized tokenizer toolkit")
    p.add_argument("--model", type=str, default="gpt2", help="Model name or path for AutoTokenizer")
    p.add_argument("--verbose", type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=True, help="Enable rich verbose metadata")
    p.add_argument("--preserve-ws", type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=False, help="Preserve whitespace with visible markers")
    p.add_argument("--normalize-nfkc", type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=True, help="Apply NFKC normalization")
    p.add_argument("--fallback-template", type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=True, help="Prefer fallback labeled chat template")
    p.add_argument("--add-schema-specials", type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=False, help="Register schema delimiters as special tokens (not recommended by default)")
    p.add_argument("--demo", type=str, default="all", choices=["all", "basic", "chat", "window", "batch", "adapt"], help="Which demo(s) to run")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    schema = SpecialSchema(add_as_special=args.add_schema_specials)
    at = AdvancedTokenizer(
        model_name_or_path=args.model,
        schema=schema,
        prefer_fallback_template=args.fallback_template,
        normalize_nfkc=args.normalize_nfkc,
        preserve_ws=args.preserve_ws,
        verbose=args.verbose,
    )

    # Run demos
    if args.demo in ("all", "basic"):
        demo_basic(at)
    if args.demo in ("all", "chat"):
        demo_chat(at)
    if args.demo in ("all", "window"):
        demo_sliding_window(at)
    if args.demo in ("all", "batch"):
        demo_dynamic_batch(at)
    if args.demo in ("all", "adapt"):
        demo_adaptation(at)

    # Final note on common pitfalls and how this script addresses them (printed as metadata)
    pitfalls = [
        ("OOV handling", "Prefer subword/BPE; optional domain adaptation via train_new_from_iterator"),
        ("Whitespace loss", "Optional visible WS markers; normalize consistently"),
        ("Truncation risk", "Sliding windows with stride; report overflow"),
        ("Padding waste", "tokenizer.pad with attention masks; dynamic batching"),
        ("Byte fragmentation", "NFKC normalization; optional adaptation to include frequent multibyte units"),
        ("Special token distortion", "Minimal injection; add specials only when requested; schema-aware delimiters"),
        ("Language bias", "Unicode NFKC; domain adaptation; char/grapheme awareness (regex)"),
        ("Unicode corruption", "Normalize NFKC consistently; preserve markers"),
        ("Rigid templates", "Fallback labeled, placeholder-driven schema; native chat template optional"),
        ("Corpus dependence", "Adapt/evaluate cross-corpus via iterator training and metadata introspection"),
    ]
    if args.verbose:
        table = Table(title="Pitfalls · Mitigations", box=box.MINIMAL_DOUBLE_HEAD)
        table.add_column("Drawback")
        table.add_column("Mitigation")
        for k, v in pitfalls:
            table.add_row(k, v)
        Console().print(table)


if __name__ == "__main__":
    main()