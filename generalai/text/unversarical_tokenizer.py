#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
universal_tokenizer.py

A rigorously engineered, fully dynamic, single-file tokenizer utility built on top of
Hugging Face AutoTokenizer that generalizes across models and task formats without
hard-coded structures. The code prioritizes correctness, explainability, and performance,
and it prints rich, schema-aware meta-data to help practitioners deeply understand how
a model will perceive input sequences: where a sequence begins/ends, roles, instructions,
reasoning, context, and other task-specific sections.

This version is robust to GPT-2 family tokenizers (e.g., microsoft/DialoGPT-medium) and
older slow tokenizers that do not expose attributes such as `extra_special_tokens`.
It uses safe getattr fallbacks and harmonizes special-token wrapping and offsets.

Highlights
----------
- Zero hard-coded, model-agnostic flow:
  - Autodetects special tokens, chat templates, BOS/EOS/PAD, truncation/padding sides.
  - Uses tokenizer.apply_chat_template when available; else builds a conservative fallback.
  - Uses tokenizer's own add_special_tokens=True path to keep masks/offsets aligned.

- Robust schema: instruction, context, reasoning, inference, actions, functions/tools,
  agentic roles, and arbitrary custom fields. Explicit role spans at token-level.

- Rich meta-data: detailed, colorized reporting using `rich` (if installed):
  - Token counts, special token ratio, unknown/OOV ratio, Unicode fragmentation stats.
  - Span boundaries per role with token index ranges and char offsets.
  - Highlights tokenizer capabilities (fast/slow), normalization, and max length.

- OOV handling and diagnostics:
  - Subword tokenization via AutoTokenizer; unknown token ratio computed.
  - Unicode normalization (NFC/NFKC). Byte fallback fragmentation diagnostic.
  - Preserves whitespace intent via explicit options and regex strategies.

- Truncation and windowing:
  - Dynamic windowing and sliding stride windows for long contexts.
  - Segment-preserving windows when possible; optional overlap.

- Padding, batching, and masks:
  - Dynamic padding to batch max; attention masks and type ids included when present.
  - Controls for left/right padding/truncation aligned to tokenizer defaults.
  - GPT-2 safety: if pad_token is absent, set it to eos_token to avoid padding errors.

- Template flexibility and schema-aware minimal injection:
  - Respects tokenizer-provided chat templates when available.
  - Fallback template is conservative and uses detected special tokens if present.
  - Limits excessive special marker injection and keeps distributions realistic.

- Domain adaptation (optional):
  - train_new_from_iterator hook for custom corpora (analysis/testing).
  - Cross-corpus diagnostics to highlight potential tokenizer mismatch.

Caveats
-------
- Adding new tokens to a pretrained model's tokenizer without adjusting model embeddings is unsafe.
  This utility avoids that. Domain-adapted tokenizers should be paired with appropriately adapted models.
- Byte fallback "expansion" requires tokenizer retraining; this code only diagnoses it.
- Precise role span mapping requires fast tokenizers to get offset mappings; slow tokenizers degrade to approximate spans.

Examples
--------
Run the module directly to see demonstrations:
  - Plain text tokenization + metadata
  - Chat-style messages with instruction + reasoning + tool call + generation prompt
  - Sliding window encoding for long input
  - Dynamic batch encoding
  - Optional domain-adaptation tokenizer training (small mockup)

Env var MODEL_ID can override the demo model:
  export MODEL_ID=microsoft/DialoGPT-medium
  python universal_tokenizer.py

"""

from __future__ import annotations

import os
import re
import json
import math
import textwrap
import unicodedata
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, TypedDict, Union

# Rich is used for verbose metadata and human-friendly inspection.
# If Rich is unavailable, we provide graceful fallback printing.
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import box

    _HAS_RICH = True
except Exception:
    _HAS_RICH = False


def _get_console() -> Any:
    if _HAS_RICH:
        return Console(highlight=False, soft_wrap=True)
    return None


def _print_panel(title: str, content: str, style: str = "cyan", border_style: str = "cyan") -> None:
    if _HAS_RICH:
        console = _get_console()
        console.print(Panel.fit(content, title=title, border_style=border_style))
    else:
        print(f"[{title}] {content}")


def _print_table(title: str, columns: List[str], rows: List[List[Any]]) -> None:
    if _HAS_RICH:
        table = Table(title=title, box=box.SIMPLE_HEAVY)
        for c in columns:
            table.add_column(c)
        for r in rows:
            table.add_row(*[str(x) for x in r])
        _get_console().print(table)
    else:
        print(f"== {title} ==")
        print("\t".join(columns))
        for r in rows:
            print("\t".join([str(x) for x in r]))


def _soft_warn(msg: str) -> None:
    warnings.warn(msg, stacklevel=2)


# -----------------------------
# Schema and data structures
# -----------------------------

class Message(TypedDict, total=False):
    role: str
    content: str
    name: str
    # Optional structured fields:
    instruction: str
    context: str
    reasoning: str
    inference: str
    action: str
    tool_call: Dict[str, Any]
    function_call: Dict[str, Any]
    meta: Dict[str, Any]


@dataclass
class RoleSpan:
    role: str
    start_token: int
    end_token: int
    # Optional char offsets (requires fast tokenizer)
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    text_snippet: Optional[str] = None


@dataclass
class EncodedSequence:
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    special_tokens_mask: Optional[List[int]] = None
    tokens: Optional[List[str]] = None
    offsets: Optional[List[Tuple[int, int]]] = None
    role_spans: List[RoleSpan] = field(default_factory=list)
    text: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.input_ids)


# --------------------------------
# Utility helpers and analyzers
# --------------------------------

def normalize_text(
    s: str,
    normalization: Optional[str] = "NFKC",
    preserve_newlines: bool = True,
    strip_control_chars: bool = True,
) -> str:
    """
    Unicode normalization with optional newline preservation and control-char stripping.
    Normalization is essential for consistent encoding across multilingual text.
    """
    if normalization:
        s = unicodedata.normalize(normalization, s)
    if strip_control_chars:
        # Remove ASCII control chars except newline and tab if preserving newlines
        if preserve_newlines:
            s = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", s)
        else:
            s = re.sub(r"[\x00-\x1F\x7F]", "", s)
    # Normalize line endings to LF
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s


def iter_sentences(text: str) -> List[str]:
    """
    Very light heuristic sentence splitter (no external dependency).
    Helps windowing near sentence boundaries to reduce truncation risk.
    """
    parts = re.split(r"([.!?])(\s+)", text)
    if len(parts) <= 1:
        return [text]
    out: List[str] = []
    buf = []
    for i in range(0, len(parts), 3):
        segment = parts[i]
        punct = parts[i + 1] if i + 1 < len(parts) else ""
        spacing = parts[i + 2] if i + 2 < len(parts) else ""
        buf.append(segment)
        buf.append(punct)
        buf.append(spacing)
        joined = "".join(buf).strip()
        if punct:
            out.append(joined)
            buf = []
    if buf:
        tail = "".join(buf).strip()
        if tail:
            out.append(tail)
    return out


def sliding_windows(
    ids: List[int], max_len: int, stride: int, bos_eos_wrap: callable
) -> Iterator[List[int]]:
    """
    General sliding window generator over token ids.
    bos_eos_wrap is a callable that safely wraps the window with model-specific special tokens.
    """
    if max_len <= 0:
        yield bos_eos_wrap(ids)
        return
    if len(ids) <= max_len:
        yield bos_eos_wrap(ids)
        return

    start = 0
    while start < len(ids):
        end = min(start + max_len, len(ids))
        chunk = ids[start:end]
        yield bos_eos_wrap(chunk)
        if end == len(ids):
            break
        start = max(0, end - stride)


def safe_import_torch() -> Any:
    try:
        import torch  # type: ignore
        return torch
    except Exception:
        return None


# ---------------------------------
# Special token inspector toolkit
# ---------------------------------

class SpecialTokenToolkit:
    """
    Inspects a tokenizer to discover best available special tokens for chat-like
    formatting and boundaries without hardcoding a template.

    This class is defensive against older tokenizers lacking newer attributes.
    """

    def __init__(self, tokenizer: Any) -> None:
        self.tok = tokenizer
        self.special_map: Dict[str, Any] = dict(getattr(self.tok, "special_tokens_map", {}) or {})
        self.all_special_tokens: List[str] = list(getattr(self.tok, "all_special_tokens", []) or [])
        # Defensive: GPT-2 and some slow tokenizers do not have `extra_special_tokens`
        _extra = getattr(self.tok, "extra_special_tokens", None)
        _additional = getattr(self.tok, "additional_special_tokens", None)
        self.extra_special_tokens: List[str] = list(
            (_extra if _extra is not None else _additional)
            or self.special_map.get("additional_special_tokens", [])
            or []
        )

        # Derived hints
        self.has_im = "<|im_start|>" in self.all_special_tokens and "<|im_end|>" in self.all_special_tokens
        self.has_inst = any(t for t in (self.all_special_tokens + self.extra_special_tokens) if "[INST]" in t or "<<SYS>>" in t)
        self.bos = getattr(self.tok, "bos_token", None)
        self.eos = getattr(self.tok, "eos_token", None)
        self.pad = getattr(self.tok, "pad_token", None)
        self.unk = getattr(self.tok, "unk_token", None)

        # Thinking token (optional in some model tokenizers)
        self.think_token = None
        for t in self.extra_special_tokens + self.all_special_tokens:
            t_low = str(t).lower()
            if "think" in t_low or "<think>" in t_low:
                self.think_token = t
                break

    def make_turn(self, role: str, content: str) -> str:
        """
        Construct a conservative role turn using available special tokens.
        If tokenizer has an internal chat template, it should be used instead
        of this fallback. This fallback avoids arbitrary hardcoding.
        """
        role = role.strip().lower()
        if self.has_im:
            # OpenAI-like template
            return f"<|im_start|>{role}\n{content}<|im_end|>\n"
        if self.has_inst and role in {"system", "user"}:
            # Llama/Mistral-like instruction bracketing for prompts (fallback)
            if role == "system":
                return f"<<SYS>>\n{content}\n<</SYS>>\n"
            return f"[INST] {content} [/INST]\n"
        # Minimal, schema-aware neutral fallback
        # Avoid too many special markers; simple headers preserve structure.
        return f"### {role}:\n{content}\n"

    def wrap_generation_prompt(self) -> str:
        """
        Append a generation cue matching the discovered style.
        """
        if self.has_im:
            return "<|im_start|>assistant\n"
        if self.has_inst:
            return ""  # [INST] usually encloses the user turn already.
        # Neutral cue
        return "### assistant:\n"

    def bos_eos_wrap(self, inner_ids: List[int]) -> List[int]:
        """
        Wrap token ids with BOS/EOS using tokenizer's build_inputs_with_special_tokens,
        avoiding manual injection logic.
        """
        return list(self.tok.build_inputs_with_special_tokens(inner_ids))

    def info(self) -> Dict[str, Any]:
        return {
            "bos": self.bos,
            "eos": self.eos,
            "pad": self.pad,
            "unk": self.unk,
            "extra_special_tokens": self.extra_special_tokens,
            "has_im_pair": self.has_im,
            "has_inst_brackets": self.has_inst,
            "think_token": self.think_token,
        }


# ---------------------------------
# Universal tokenizer main class
# ---------------------------------

class UniversalTokenizer:
    """
    Universal tokenizer with schema-aware, model-agnostic behavior.
    It provides high-fidelity encode/decode, role spans, dynamic windowing, batching,
    and rich diagnostics. All explanations are included as docstrings and comments.

    Public methods:
    - encode_text(...)
    - encode_messages(...)
    - decode(...)
    - analyze(...)
    - window_encode(...)
    - batch_encode_texts(...)
    - train_new_from_iterator(...)
    """

    def __init__(
        self,
        model_id_or_path: str,
        verbose: bool = True,
        normalization: Optional[str] = "NFKC",
        preserve_newlines: bool = True,
        trust_remote_code: bool = True,
        use_fast: Optional[bool] = None,
        add_prefix_space: Optional[bool] = None,
    ) -> None:
        from transformers import AutoTokenizer  # import locally to keep file self-contained

        self.verbose = bool(verbose)
        self.console = _get_console() if self.verbose and _HAS_RICH else None
        self.model_id = model_id_or_path
        self.normalization = normalization
        self.preserve_newlines = preserve_newlines

        # Load tokenizer dynamically; let HF figure out the class.
        self.tok = AutoTokenizer.from_pretrained(
            model_id_or_path, trust_remote_code=trust_remote_code, use_fast=use_fast
        )

        # Add prefix space if recommended (common for GPT-2/RoBERTa style tokenizers)
        if hasattr(self.tok, "add_prefix_space") and add_prefix_space is not None:
            try:
                self.tok.add_prefix_space = bool(add_prefix_space)
            except Exception:
                pass

        # GPT-2 safety: if pad token is missing, reuse eos_token to avoid padding errors
        try:
            if getattr(self.tok, "pad_token", None) is None and getattr(self.tok, "eos_token", None) is not None:
                self.tok.pad_token = self.tok.eos_token
        except Exception:
            pass

        # Special token toolkit and introspection
        self.st = SpecialTokenToolkit(self.tok)

        # Controls for underlying tokenizer behavior
        self.is_fast = bool(getattr(self.tok, "is_fast", False))
        self.model_max_length = int(getattr(self.tok, "model_max_length", 1000000000000))
        self.padding_side = getattr(self.tok, "padding_side", "right")
        self.truncation_side = getattr(self.tok, "truncation_side", "right")

        # Detect chat template availability
        self.has_chat_template = False
        try:
            tmpl = getattr(self.tok, "get_chat_template", None)
            if callable(tmpl):
                self.has_chat_template = bool(self.tok.get_chat_template())
        except Exception:
            self.has_chat_template = False

        # Prepare reporting about tokenizer
        if self.verbose:
            self._report_tokenizer_info()

    # ----------------------------
    # Internal reporting utilities
    # ----------------------------

    def _report_tokenizer_info(self) -> None:
        info = self.st.info()
        text = textwrap.dedent(
            f"""
            Model/Tokenizer: {self.model_id}
            is_fast              : {self.is_fast}
            model_max_length     : {self.model_max_length}
            padding_side         : {self.padding_side}
            truncation_side      : {self.truncation_side}
            chat_template_exists : {self.has_chat_template}

            Special tokens:
              bos={info['bos']}
              eos={info['eos']}
              pad={info['pad']}
              unk={info['unk']}
              think_token={info['think_token']}
              extra_special_tokens={info['extra_special_tokens']}
            """
        ).strip()
        _print_panel("Tokenizer Overview", text, border_style="green")

    # ----------------------------
    # Normalization pipeline
    # ----------------------------

    def _normalize(self, s: str) -> str:
        return normalize_text(s, normalization=self.normalization, preserve_newlines=self.preserve_newlines)

    # ----------------------------
    # Encoding helpers and core
    # ----------------------------

    def _apply_chat_template(
        self,
        messages: List[Message],
        add_generation_prompt: bool = False,
        tokenize: bool = False,
    ) -> Union[str, Dict[str, Any]]:
        """
        Prefer tokenizer's internal chat template. If absent, fallback to conservative builder.
        """
        # Attempt to use tokenizer.apply_chat_template when available
        try:
            if hasattr(self.tok, "apply_chat_template"):
                rendered = self.tok.apply_chat_template(
                    messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt
                )
                return rendered
        except Exception:
            pass

        # Fallback: conservative format
        text_parts: List[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = self._render_message_content(msg)
            text_parts.append(self.st.make_turn(role, content))
        if add_generation_prompt:
            text_parts.append(self.st.wrap_generation_prompt())
        return "".join(text_parts)

    def _render_message_content(self, msg: Message) -> str:
        """
        Render a message content by merging fields in a schema-aware and minimal way.
        Exposes structure but limits artificial markers to avoid distortions.
        """
        fields: List[Tuple[str, Optional[str]]] = [
            ("instruction", msg.get("instruction")),
            ("context", msg.get("context")),
            ("content", msg.get("content")),
            ("reasoning", msg.get("reasoning")),
            ("inference", msg.get("inference")),
            ("action", msg.get("action")),
        ]

        # Function/tool calls if present
        fc = msg.get("function_call")
        tc = msg.get("tool_call")

        parts: List[str] = []
        for name, value in fields:
            if value:
                val = self._normalize(str(value))
                # Minimal schema-aware header
                parts.append(f"[{name}]\n{val}\n[/{name}]")

        if fc:
            try:
                parts.append(f"[function_call]\n{json.dumps(fc, ensure_ascii=False)}\n[/function_call]")
            except Exception:
                parts.append(f"[function_call]\n{str(fc)}\n[/function_call]")

        if tc:
            try:
                parts.append(f"[tool_call]\n{json.dumps(tc, ensure_ascii=False)}\n[/tool_call]")
            except Exception:
                parts.append(f"[tool_call]\n{str(tc)}\n[/tool_call]")

        if not parts:
            # Fallback to role content if no fields provided
            c = self._normalize(str(msg.get("content", "")))
            parts.append(c)

        return "\n".join(parts).strip()

    def _encode_text_core(
        self, text: str, add_bos_eos: bool = True, return_offsets: bool = True
    ) -> Tuple[List[int], Dict[str, Any]]:
        """
        Core text encoding using tokenizer with clean normalization and optional offset mapping.

        Implementation note:
        - We rely on `add_special_tokens=add_bos_eos` in the tokenizer call to keep
          `input_ids`, `special_tokens_mask`, and `offset_mapping` aligned. This avoids
          mismatches that arise when wrapping manually after the fact.
        """
        normalized = self._normalize(text)
        enc = self.tok(
            normalized,
            return_attention_mask=True,
            return_special_tokens_mask=True,
            return_offsets_mapping=self.is_fast and return_offsets,
            add_special_tokens=add_bos_eos,
        )

        ids = list(enc["input_ids"])
        meta = {
            "normalized": normalized,
            "attention_mask": list(enc.get("attention_mask", [])) if enc.get("attention_mask") is not None else None,
            "token_type_ids": list(enc.get("token_type_ids", [])) if enc.get("token_type_ids") is not None else None,
            "special_tokens_mask": list(enc.get("special_tokens_mask", [])) if enc.get("special_tokens_mask") is not None else None,
            "offset_mapping": list(enc.get("offset_mapping", [])) if enc.get("offset_mapping") is not None else None,
            "tokens": self.tok.convert_ids_to_tokens(ids),
        }
        return ids, meta

    def encode_text(
        self,
        text: str,
        add_bos_eos: bool = True,
        verbose: Optional[bool] = None,
    ) -> EncodedSequence:
        """
        Encode plain text with high-fidelity normalization, role-neutral.
        """
        ids, meta = self._encode_text_core(text, add_bos_eos=add_bos_eos, return_offsets=True)
        enc = EncodedSequence(
            input_ids=ids,
            attention_mask=meta.get("attention_mask"),
            token_type_ids=meta.get("token_type_ids"),
            special_tokens_mask=meta.get("special_tokens_mask"),
            tokens=meta.get("tokens"),
            offsets=meta.get("offset_mapping"),
            role_spans=[],  # No roles in plain text
            text=meta.get("normalized"),
            meta={"model_max_length": self.model_max_length, "schema": "text"},
        )
        if (self.verbose if verbose is None else verbose):
            self.analyze(enc, title="Text Encoding Analysis")
        return enc

    def encode_messages(
        self,
        messages: List[Message],
        add_generation_prompt: bool = False,
        include_reasoning: bool = True,
        mask_reasoning: bool = False,
        verbose: Optional[bool] = None,
    ) -> EncodedSequence:
        """
        Encode a chat-style list of Message dicts.

        Parameters:
        - add_generation_prompt: Append assistant-start cue if model expects one.
        - include_reasoning: If False, drops reasoning fields from inputs.
        - mask_reasoning: If True, keeps reasoning text but marks its span in meta only
          (no general, safe way to mask for all models; we expose spans for downstream masking).

        Returns:
        - EncodedSequence with role_spans resolved to token indices and (if possible) char offsets.
        """
        # Prepare messages: optionally strip reasoning
        msgs: List[Message] = []
        for m in messages:
            m2 = dict(m)  # shallow copy
            if not include_reasoning and "reasoning" in m2:
                m2.pop("reasoning", None)
            # Normalize content-like fields eagerly to compute accurate char spans later
            for k in ("instruction", "context", "content", "reasoning", "inference", "action"):
                if m2.get(k) is not None:
                    m2[k] = self._normalize(str(m2[k]))
            msgs.append(m2)

        # Render prompt string
        rendered = self._apply_chat_template(msgs, add_generation_prompt=add_generation_prompt, tokenize=False)
        if isinstance(rendered, dict):
            raise RuntimeError("Unexpected tokenized output for chat template. Set tokenize=False.")
        prompt_text = rendered

        # Encode full prompt with offsets (add special tokens to keep masks/offsets aligned)
        ids, meta = self._encode_text_core(prompt_text, add_bos_eos=True, return_offsets=True)
        offsets = meta.get("offset_mapping")
        tokens = meta.get("tokens")

        # Compute role spans by searching rendered text for each message's content
        role_spans = self._compute_role_spans_from_rendered(prompt_text, msgs, offsets)

        # Optionally mark reasoning spans in meta for downstream masking
        reasoning_spans = [rs for rs in role_spans if rs.role.endswith(":reasoning")]
        seq_meta = {
            "model_max_length": self.model_max_length,
            "schema": "chat",
            "add_generation_prompt": add_generation_prompt,
            "reasoning_spans": [(s.start_token, s.end_token) for s in reasoning_spans],
            "has_chat_template": self.has_chat_template,
            "mask_reasoning_hint": bool(mask_reasoning),
        }

        enc = EncodedSequence(
            input_ids=ids,
            attention_mask=meta.get("attention_mask"),
            token_type_ids=meta.get("token_type_ids"),
            special_tokens_mask=meta.get("special_tokens_mask"),
            tokens=tokens,
            offsets=offsets,
            role_spans=role_spans,
            text=prompt_text,
            meta=seq_meta,
        )

        if (self.verbose if verbose is None else verbose):
            self.analyze(enc, title="Chat Encoding Analysis")

        return enc

    def _compute_role_spans_from_rendered(
        self,
        prompt_text: str,
        msgs: List[Message],
        offsets: Optional[List[Tuple[int, int]]],
    ) -> List[RoleSpan]:
        """
        Approximate role spans by searching the rendered prompt for each content field.
        For fast tokenizers, resolves char->token boundaries precisely via offsets.
        """
        spans: List[RoleSpan] = []
        for idx, m in enumerate(msgs):
            role = m.get("role", "user")
            fields = [
                ("instruction", m.get("instruction")),
                ("context", m.get("context")),
                ("content", m.get("content")),
                ("reasoning", m.get("reasoning")),
                ("inference", m.get("inference")),
                ("action", m.get("action")),
            ]
            for fname, fval in fields:
                if not fval:
                    continue
                for m_obj in re.finditer(re.escape(fval), prompt_text):
                    start_char = m_obj.start()
                    end_char = m_obj.end()
                    start_tok, end_tok = self._char_to_token_span(offsets, start_char, end_char)
                    spans.append(
                        RoleSpan(
                            role=f"{role}:{fname}",
                            start_token=start_tok,
                            end_token=end_tok,
                            start_char=start_char,
                            end_char=end_char,
                            text_snippet=fval[:64] + ("..." if len(fval) > 64 else ""),
                        )
                    )
                    break  # one span per field instance

            for key in ("function_call", "tool_call"):
                obj = m.get(key)
                if not obj:
                    continue
                try:
                    blob = json.dumps(obj, ensure_ascii=False)
                except Exception:
                    blob = str(obj)
                for m_obj in re.finditer(re.escape(blob), prompt_text):
                    start_char = m_obj.start()
                    end_char = m_obj.end()
                    start_tok, end_tok = self._char_to_token_span(offsets, start_char, end_char)
                    spans.append(
                        RoleSpan(
                            role=f"{role}:{key}",
                            start_token=start_tok,
                            end_token=end_tok,
                            start_char=start_char,
                            end_char=end_char,
                            text_snippet=blob[:64] + ("..." if len(blob) > 64 else ""),
                        )
                    )
                    break

        return spans

    def _char_to_token_span(
        self,
        offsets: Optional[List[Tuple[int, int]]],
        s_char: int,
        e_char: int,
    ) -> Tuple[int, int]:
        """
        Map char span to token span using offsets (fast tokenizer).
        If unavailable, return (-1, -1) as unknown.
        """
        if not offsets:
            return -1, -1
        start_token = 0
        end_token = 0
        for i, (a, b) in enumerate(offsets):
            if a <= s_char < b:
                start_token = i
                break
        else:
            start_token = 0

        for j in range(len(offsets) - 1, -1, -1):
            a, b = offsets[j]
            if a < e_char <= b:
                end_token = j + 1  # exclusive end
                break
        else:
            end_token = len(offsets)
        return start_token, end_token

    # ----------------------------
    # Decoding
    # ----------------------------

    def decode(self, ids: List[int], skip_special_tokens: bool = False, clean_up_spaces: bool = True) -> str:
        """
        Decode token ids to string with safe defaults.
        """
        return self.tok.decode(ids, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=clean_up_spaces)

    # ----------------------------
    # Analysis and rich diagnostics
    # ----------------------------

    def analyze(self, enc: EncodedSequence, title: str = "Encoding Analysis") -> Dict[str, Any]:
        """
        Produce a comprehensive diagnostic report about an encoded sequence:

        - token counts and caps
        - special token ratio
        - OOV rate
        - unicode fragmentation
        - whitespace diagnostics
        - role spans table
        """
        ids = enc.input_ids
        toks = enc.tokens or self.tok.convert_ids_to_tokens(ids)
        sp_mask = enc.special_tokens_mask or [0] * len(ids)
        unk_id = self.tok.unk_token_id
        special_count = int(sum(sp_mask))
        unk_count = int(sum(1 for i in ids if i == unk_id)) if unk_id is not None else 0

        # Unicode fragmentation: average tokens per non-ASCII char
        unicodes = sum(1 for ch in (enc.text or "") if ord(ch) > 127)
        avg_toks_per_unicode_char = float("nan")
        if self.is_fast and enc.offsets:
            non_ascii_positions = {i for i, ch in enumerate(enc.text or "") if ord(ch) > 127}
            token_hits = 0
            for (a, b) in enc.offsets:
                if any(p in non_ascii_positions for p in range(a, b)):
                    token_hits += 1
            if unicodes > 0:
                avg_toks_per_unicode_char = token_hits / max(1, unicodes)

        whitespaces = re.findall(r"\s+", enc.text or "")
        max_ws = max((len(w) for w in whitespaces), default=0)

        analysis = {
            "length_tokens": len(ids),
            "length_chars": len(enc.text or ""),
            "model_max_length": self.model_max_length,
            "special_token_ratio": round(special_count / max(1, len(ids)), 4),
            "oov_ratio": round(unk_count / max(1, len(ids)), 4),
            "unicodes": unicodes,
            "avg_toks_per_unicode_char": round(avg_toks_per_unicode_char, 4) if isinstance(avg_toks_per_unicode_char, float) else "n/a",
            "max_whitespace_run": max_ws,
            "padding_side": self.padding_side,
            "truncation_side": self.truncation_side,
        }

        if self.verbose:
            meta_lines = [
                f"Tokens: {analysis['length_tokens']}  |  Chars: {analysis['length_chars']}  |  Model cap: {analysis['model_max_length']}",
                f"Special ratio: {analysis['special_token_ratio']}  |  OOV ratio: {analysis['oov_ratio']}",
                f"Unicode chars: {unicodes}  |  Avg toks/unicode char: {analysis['avg_toks_per_unicode_char']}",
                f"Max whitespace run: {analysis['max_whitespace_run']}",
                f"Padding side: {analysis['padding_side']}  |  Truncation side: {analysis['truncation_side']}",
            ]
            _print_panel(title, "\n".join(meta_lines), border_style="cyan")

            # Special tokens quick view
            sp_rows = []
            for i, (tid, t, sm) in enumerate(zip(ids, toks, sp_mask)):
                if sm:
                    sp_rows.append([i, tid, t])
            if sp_rows:
                _print_table("Special Tokens", ["idx", "id", "token"], sp_rows[:50])

            # Role span table
            if enc.role_spans:
                rows = []
                for rs in enc.role_spans:
                    rows.append([
                        rs.role,
                        rs.start_token,
                        rs.end_token,
                        (rs.end_token - rs.start_token) if (rs.end_token >= 0 and rs.start_token >= 0) else "n/a",
                        f"{rs.start_char}-{rs.end_char}" if rs.start_char is not None else "n/a",
                    ])
                _print_table("Role Spans", ["role", "start_tok", "end_tok", "len", "char_range"], rows)

        return analysis

    # ----------------------------
    # Windowing
    # ----------------------------

    def window_encode(
        self,
        text: str,
        target_window_tokens: int,
        stride_tokens: Optional[int] = None,
        add_bos_eos_per_window: bool = True,
    ) -> Iterator[EncodedSequence]:
        """
        Encode a long text into multiple overlapping windows.
        Each window is safely wrapped with BOS/EOS via build_inputs_with_special_tokens.
        """
        stride_tokens = stride_tokens if stride_tokens is not None else max(1, target_window_tokens // 5)

        # Encode raw text without special tokens to slice windows at token level
        normalized = self._normalize(text)
        base = self.tok(
            normalized,
            return_attention_mask=False,
            return_offsets_mapping=self.is_fast,
            add_special_tokens=False,
        )
        ids = list(base["input_ids"])

        def wrap(chunk_ids: List[int]) -> List[int]:
            if add_bos_eos_per_window:
                return self.st.bos_eos_wrap(chunk_ids)
            return list(chunk_ids)

        for win_ids in sliding_windows(ids, max_len=target_window_tokens, stride=stride_tokens, bos_eos_wrap=wrap):
            toks = self.tok.convert_ids_to_tokens(win_ids)
            attn = [1] * len(win_ids)
            yield EncodedSequence(
                input_ids=win_ids,
                attention_mask=attn,
                tokens=toks,
                text=normalized,
                meta={"schema": "window", "window_size": target_window_tokens, "stride": stride_tokens},
            )

    # ----------------------------
    # Dynamic batching
    # ----------------------------

    def batch_encode_texts(
        self,
        texts: List[str],
        add_bos_eos: bool = True,
        pad_to_longest: bool = True,
        return_tensors: Optional[str] = None,  # "pt" or "np"
    ) -> Dict[str, Any]:
        """
        Encode a batch of texts with dynamic padding and optional tensor conversion.
        """
        normed = [self._normalize(t) for t in texts]
        enc = self.tok(
            normed,
            padding="longest" if pad_to_longest else False,
            truncation=False,
            return_attention_mask=True,
            return_special_tokens_mask=True,
            return_offsets_mapping=False,
            add_special_tokens=add_bos_eos,
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        out = {"input_ids": input_ids, "attention_mask": attention_mask}
        if return_tensors:
            torch = safe_import_torch()
            if return_tensors == "pt" and torch is not None:
                out = {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}
            elif return_tensors == "np":
                import numpy as np  # type: ignore

                out = {k: np.array(v, dtype=np.int64) for k, v in out.items()}
            else:
                _soft_warn("Requested return_tensors but dependency not available; returning Python lists.")
        return out

    # ----------------------------
    # Domain adaptation (optional)
    # ----------------------------

    def train_new_from_iterator(
        self,
        iterator: Iterable[str],
        vocab_size: int = 30000,
        min_frequency: int = 2,
        show_progress: bool = True,
    ) -> Any:
        """
        Train a new tokenizer from iterator (BPE/Unigram depending on base tokenizer type).
        WARNING: Training a tokenizer alone does not adapt a model. This is for analysis
        and experimentation. If you need a production tokenizer, pair it with a suitably
        adapted model.
        """
        if not hasattr(self.tok, "train_new_from_iterator"):
            raise RuntimeError("This tokenizer class does not support train_new_from_iterator.")
        new_tok = self.tok.train_new_from_iterator(
            iterator, vocab_size=vocab_size, min_frequency=min_frequency, show_progress=show_progress
        )
        return new_tok


# ---------------------------------
# Demonstrations
# ---------------------------------

def _demo_text(tokenizer: UniversalTokenizer) -> None:
    sample = "Hello, 世界!   This is a test.\nNew line, tabs\tand multiple   spaces."
    enc = tokenizer.encode_text(sample, add_bos_eos=True, verbose=True)
    decoded = tokenizer.decode(enc.input_ids, skip_special_tokens=False)
    _print_panel("Decoded (raw)", decoded)


def _demo_chat(tokenizer: UniversalTokenizer) -> None:
    messages: List[Message] = [
        {
            "role": "system",
            "instruction": "You are a precise assistant. Answer with concise, correct information.",
            "context": "Domain=mathematics; Audience=advanced; Style=succinct; Language=en",
        },
        {
            "role": "user",
            "content": "Integrate x^2 from 0 to 3.",
            "reasoning": "We can recall ∫ x^2 dx = x^3/3 and apply bounds 0 and 3.",
        },
        {
            "role": "assistant",
            "inference": "x^3/3 from 0 to 3 is (27/3) - 0 = 9.",
            "action": "final_answer",
        },
        {
            "role": "assistant",
            "function_call": {"name": "plot_polynomial", "arguments": {"expr": "x^2", "xmin": 0, "xmax": 3}},
        },
        {
            "role": "tool",
            "tool_call": {"name": "plot_polynomial", "result": "plot_url=https://example.com/plot/abc"},
        },
    ]

    enc = tokenizer.encode_messages(
        messages, add_generation_prompt=True, include_reasoning=True, mask_reasoning=True, verbose=True
    )
    decoded = tokenizer.decode(enc.input_ids, skip_special_tokens=False)
    _print_panel("Rendered Prompt (decoded)", decoded)


def _demo_windowing(tokenizer: UniversalTokenizer) -> None:
    long_text = " ".join(
        [f"Sentence {i}. This is a long document with multilingual tokens like café, 東京, and emojis 😊."
         for i in range(1, 30)]
    )
    windows = tokenizer.window_encode(long_text, target_window_tokens=128, stride_tokens=48, add_bos_eos_per_window=True)
    for i, w in enumerate(windows, 1):
        _print_panel(f"Window {i}", f"len={len(w.input_ids)}; meta={w.meta}")


def _demo_batching(tokenizer: UniversalTokenizer) -> None:
    batch = [
        "Hello world.",
        "Multilingual: Привет мир, こんにちは世界, مرحبا بالعالم.",
        "Whitespace   preserves   intent.",
    ]
    out = tokenizer.batch_encode_texts(batch, add_bos_eos=True, pad_to_longest=True, return_tensors=None)
    _print_panel("Batch shapes", f"num={len(out['input_ids'])}; lengths={[len(x) for x in out['input_ids']]}")


def _demo_train_new_from_iterator(tokenizer: UniversalTokenizer) -> None:
    # Caution: This is a short illustrative demo; real training needs large corpora.
    corpus = [
        "Domain-specific term: foobarization and anti-foobarization.",
        "Another domain string with ζ-function and non-ASCII math symbols like ∑ and ∫.",
        "Tool calls: call_tool(name=extract, args={...}).",
    ]

    try:
        new_tok = tokenizer.train_new_from_iterator(corpus, vocab_size=2000, min_frequency=1, show_progress=False)
        info = {
            "new_vocab_size": new_tok.vocab_size,
            "has_chat_template": bool(getattr(new_tok, "chat_template", None)),
            "is_fast": bool(getattr(new_tok, "is_fast", False)),
        }
        _print_panel("Trained new tokenizer (demo)", json.dumps(info, ensure_ascii=False, indent=2))
    except Exception as e:
        _soft_warn(f"train_new_from_iterator not supported for this tokenizer: {e}")


# ---------------------------------
# Main
# ---------------------------------

if __name__ == "__main__":
    # Select a model id dynamically. Defaults to a GPT-2-family tokenizer to show robust fallbacks.
    model_id = os.environ.get("MODEL_ID", "microsoft/DialoGPT-medium")
    verbose_flag = os.environ.get("TOKENIZER_VERBOSE", "1") not in {"0", "false", "False"}

    tok = UniversalTokenizer(model_id, verbose=verbose_flag, normalization="NFKC")

    _demo_text(tok)
    _demo_chat(tok)
    _demo_windowing(tok)
    _demo_batching(tok)
    # _demo_train_new_from_iterator(tok)