#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio Loader + Matplotlib Visualizations (rich-enabled, verbose)

This single-file module provides:
- Robust loading of audio from:
    - Local path (absolute/relative), including file:// URIs
    - HTTP/HTTPS URLs
    - In-memory bytes (WAV/AIFF via stdlib)
- A clean, well-documented pipeline that uses only the provided audio API:
    from audio import Audio, mel_filter_bank, hertz_to_mel, mel_to_hertz
- Rich-powered metadata panels and progress feedback
- High-quality Matplotlib visualizations:
    - Waveform (multi-channel aware)
    - Linear-frequency spectrogram (dB-scaled)
    - Mel-spectrogram (using the provided mel_filter_bank)
    - Mel filter bank response curves
- Sidecar JSON metadata for each saved figure

Notes for coders:
- Explanations are embedded as docstrings and comments.
- Figures are saved under ./figures (override via --plot-dir).
- This script avoids unnecessary third-party dependencies:
    - Decoding uses stdlib (wave/aifc) when Audio API methods are not present.
    - If the Audio class already has capable loaders (e.g., from_file/url/path/bytes),
      those are used preferentially via hasattr checks.
- The "quick demo" provided in the spec is embedded under __main__ and extended
  with visualizations.

CLI examples:
    python audio_vis.py --source ./example.wav --plot-dir out
    python audio_vis.py --source file:///absolute/path/to/file.wav
    python audio_vis.py --source https://example.com/audio.wav
    python audio_vis.py --stdin-bytes wav --plot-dir out  < some.wav
    AUDIO_VERBOSE=1 python audio_vis.py --source ./example.wav

Authoring style:
- Technical tone, best-practice code style, detailed exception handling,
  and clear, reusable components for next-generation coders.

"""

from __future__ import annotations

import os
import io
import re
import sys
import json
import math
import base64
import aifc
import wave
import time
import types
import typing as t
import warnings
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

# Force a non-interactive backend for headless environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.progress import track
from rich import box

# Only allowed API from the provided code spec
from audio import Audio, mel_filter_bank, hertz_to_mel, mel_to_hertz


# --------------------------------------------------------------------------------------
# Global console and helpers
# --------------------------------------------------------------------------------------

_console = Console(record=False, highlight=False, emoji=True)

def _is_truthy(val: str | None) -> bool:
    """Return True if val looks like a truthy string (1,true,yes,on), case-insensitive."""
    if val is None:
        return False
    return str(val).strip().lower() in {"1", "true", "yes", "on", "y", "t"}

def _rich_panel_for_meta(meta: dict, title: str = "Meta", border_style: str = "blue") -> Panel:
    """Render dictionary metadata as a rich Panel."""
    try:
        pretty = json.dumps(meta, indent=2, default=str)
    except Exception:
        pretty = repr(meta)
    return Panel(Text(pretty, style="bold"), title=title, border_style=border_style)

def _now_str() -> str:
    """Timestamp string for filenames."""
    return time.strftime("%Y%m%d-%H%M%S")


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class PlotConfig:
    """Configuration for spectral analysis and figure output."""
    n_fft: int = 1024
    hop_length: int | None = None  # default: n_fft // 4
    win: str = "hann"
    n_mels: int = 64
    fmin: float = 20.0
    fmax_ratio: float = 0.5  # fmax = sr * fmax_ratio
    top_db: float = 80.0
    cmap: str = "magma"
    figsize_wave: tuple[float, float] = (12, 4.0)
    figsize_spec: tuple[float, float] = (12, 5.5)
    dpi: int = 140
    facecolor: str = "white"

DEFAULT_CFG = PlotConfig()


# --------------------------------------------------------------------------------------
# Audio sample extraction helpers (robust to unknown Audio API internals)
# --------------------------------------------------------------------------------------

def get_audio_samples(audio: Audio) -> np.ndarray:
    """
    Try to extract the underlying numpy array from an Audio instance
    without relying on a specific internal attribute name.
    Returns a contiguous float32 array in shape (n,) or (n, ch).
    Raises if not found.
    """
    candidates = ["samples", "data", "array", "y", "values", "_samples", "_data"]
    for name in candidates:
        arr = getattr(audio, name, None)
        if isinstance(arr, np.ndarray):
            return _to_float32(arr)

    # Try callable converters if exposed
    converters = ["to_numpy", "to_ndarray", "to_array"]
    for name in converters:
        fn = getattr(audio, name, None)
        if callable(fn):
            arr = fn()
            if isinstance(arr, np.ndarray):
                return _to_float32(arr)

    # As a last resort, check __array__ protocol
    try:
        arr = np.asarray(audio)  # type: ignore[arg-type]
        if isinstance(arr, np.ndarray):
            return _to_float32(arr)
    except Exception:
        pass

    raise AttributeError("Cannot extract numpy samples from Audio; "
                         "expected an ndarray-like attribute or converter.")

def _to_float32(x: np.ndarray) -> np.ndarray:
    """Convert to float32 numpy array and ensure C-contiguous memory."""
    if np.issubdtype(x.dtype, np.floating):
        arr = x.astype(np.float32, copy=False)
    else:
        # int -> float range [-1, 1] if common PCM types; else normalize by max
        info = np.iinfo(x.dtype) if np.issubdtype(x.dtype, np.integer) else None
        if info:
            max_abs = max(abs(info.min), abs(info.max))
            arr = (x.astype(np.float32) / max_abs)
        else:
            arr = x.astype(np.float32)
    return np.ascontiguousarray(arr)


# --------------------------------------------------------------------------------------
# Minimal audio decoding from stdlib (WAV/AIFF) if Audio API lacks its own loader
# --------------------------------------------------------------------------------------

def _is_url(s: str) -> bool:
    try:
        p = urllib.parse.urlparse(s)
        return p.scheme in {"http", "https"}
    except Exception:
        return False

def _is_file_uri(s: str) -> bool:
    try:
        p = urllib.parse.urlparse(s)
        return p.scheme == "file"
    except Exception:
        return False

def _path_from_file_uri(uri: str) -> Path:
    p = urllib.parse.urlparse(uri)
    return Path(urllib.request.url2pathname(p.path))

def _fetch_bytes_from_source(source: str | bytes) -> bytes:
    """
    Fetch bytes from:
      - bytes input (returned as-is)
      - 'file://...' URI
      - local filesystem path
      - http(s) URL
    """
    if isinstance(source, (bytes, bytearray, memoryview)):
        return bytes(source)

    assert isinstance(source, str)
    if _is_file_uri(source):
        path = _path_from_file_uri(source)
        return path.read_bytes()

    if _is_url(source):
        with urllib.request.urlopen(source) as r:
            return r.read()

    # treat as local path
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {source}")
    return path.read_bytes()

def _decode_wav_bytes(b: bytes) -> tuple[np.ndarray, int, int]:
    """
    Decode WAV bytes using stdlib 'wave'.
    Returns: (float32 ndarray shape (n, ch), sample_rate, channels)
    """
    with wave.open(io.BytesIO(b), "rb") as wf:
        n_channels = wf.getnchannels()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(n_frames)

    # Interpret PCM width
    if sampwidth == 1:  # 8-bit unsigned PCM
        x = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        x = (x - 128.0) / 128.0
    elif sampwidth == 2:  # 16-bit
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 3:  # 24-bit packed
        # Unpack 3-byte little-endian to int32
        a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        b32 = (a[:, 0].astype(np.int32) |
               (a[:, 1].astype(np.int32) << 8) |
               (a[:, 2].astype(np.int32) << 16))
        # Sign-extend
        mask = 1 << 23
        b32 = (b32 ^ mask) - mask
        x = b32.astype(np.float32) / (1 << 23)
    elif sampwidth == 4:  # 32-bit PCM
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    if n_channels > 1:
        x = x.reshape(-1, n_channels)
    return x.astype(np.float32, copy=False), sr, n_channels

def _decode_aiff_bytes(b: bytes) -> tuple[np.ndarray, int, int]:
    """
    Decode AIFF/AIFF-C via stdlib 'aifc'.
    Returns: (float32 ndarray shape (n, ch), sample_rate, channels)
    """
    with aifc.open(io.BytesIO(b), "rb") as af:
        n_channels = af.getnchannels()
        sr = int(af.getframerate())
        n_frames = af.getnframes()
        sampwidth = af.getsampwidth()
        raw = af.readframes(n_frames)

    # For AIFF, data is big-endian PCM
    if sampwidth == 1:
        x = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        x = (x - 128.0) / 128.0
    elif sampwidth == 2:
        x = np.frombuffer(raw, dtype=">i2").astype(np.float32) / 32768.0
    elif sampwidth == 3:
        a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        b32 = ((a[:, 0].astype(np.int32) << 16) |
               (a[:, 1].astype(np.int32) << 8) |
               (a[:, 2].astype(np.int32)))
        mask = 1 << 23
        b32 = (b32 ^ mask) - mask
        x = b32.astype(np.float32) / (1 << 23)
    elif sampwidth == 4:
        x = np.frombuffer(raw, dtype=">i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported AIFF sample width: {sampwidth} bytes")

    if n_channels > 1:
        x = x.reshape(-1, n_channels)
    return x.astype(np.float32, copy=False), sr, n_channels


# --------------------------------------------------------------------------------------
# Loader facade: prefer Audio API, fallback to stdlib decoders
# --------------------------------------------------------------------------------------

@dataclass
class LoadedAudio:
    """Container for loaded audio and attached numpy samples for plotting."""
    audio: Audio
    samples: np.ndarray  # float32, shape (n,) or (n, ch)
    sample_rate: int

def load_audio_any(
    source: str | bytes,
    *,
    mono: bool | None = None,
    fmt: str | None = None,
    source_tag: str | None = None,
    meta: dict | None = None,
    _verbose: bool = True,
) -> LoadedAudio:
    """
    Load audio from path/URL/bytes using the provided Audio API when possible.
    Fallback to stdlib WAV/AIFF decoders.

    Returns LoadedAudio with:
      - audio: Audio instance
      - samples: float32 ndarray (mono or multichannel)
      - sample_rate: int

    Parameters:
      mono: If True/False and API supports it, ensure mono on load;
            otherwise, perform a mixdown in this function.
      fmt: Optional format hint (e.g., "WAV")
      source_tag: Short label to attach to meta
      meta: Additional metadata dict
    """
    meta = dict(meta or {})
    if source_tag:
        meta.setdefault("source_tag", source_tag)

    # Prefer Audio API load methods if available
    if isinstance(source, (bytes, bytearray, memoryview)):
        b = bytes(source)

        # 1) Has a direct bytes loader?
        if hasattr(Audio, "from_bytes") and callable(getattr(Audio, "from_bytes")):
            audio = Audio.from_bytes(b, mono=bool(mono) if mono is not None else None, fmt=fmt, source="bytes", meta=meta)  # type: ignore[arg-type]
            samples = get_audio_samples(audio)
            return LoadedAudio(audio=audio, samples=samples, sample_rate=int(getattr(audio, "sample_rate", 0) or 0))

        # 2) Try stdlib decoders
        try:
            if b[:4] == b"RIFF" and b[8:12] == b"WAVE":
                samples, sr, _ = _decode_wav_bytes(b)
                audio = Audio(samples, sr, fmt="WAV", mono=False if samples.ndim == 2 else True, source="bytes-wav", meta=meta)
                # Apply mono if requested
                if mono is True and samples.ndim == 2:
                    samples = samples.mean(axis=1, keepdims=False)
                    audio = audio.ensure_mono() if hasattr(audio, "ensure_mono") else Audio(samples, sr, fmt="WAV", mono=True, source="bytes-wav-mix", meta=meta)
                return LoadedAudio(audio=audio, samples=samples, sample_rate=sr)
        except Exception as e:
            if _verbose:
                _console.print(Panel(Text(str(e), style="bold red"), title="WAV Decode Failed", border_style="red"))

        # 3) AIFF?
        try:
            if b[:4] in (b"FORM",):
                samples, sr, _ = _decode_aiff_bytes(b)
                audio = Audio(samples, sr, fmt="AIFF", mono=False if samples.ndim == 2 else True, source="bytes-aiff", meta=meta)
                if mono is True and samples.ndim == 2:
                    samples = samples.mean(axis=1, keepdims=False)
                    audio = audio.ensure_mono() if hasattr(audio, "ensure_mono") else Audio(samples, sr, fmt="AIFF", mono=True, source="bytes-aiff-mix", meta=meta)
                return LoadedAudio(audio=audio, samples=samples, sample_rate=sr)
        except Exception as e:
            if _verbose:
                _console.print(Panel(Text(str(e), style="bold red"), title="AIFF Decode Failed", border_style="red"))

        # 4) Last resort: Raw PCM via provided API
        # If caller intentionally gives raw PCM16 mono, they should use Audio.from_raw_audio.
        raise ValueError("Unsupported in-memory bytes format; provide WAV/AIFF bytes or use Audio.from_raw_audio payload.")

    # String source: URL or path or file://
    assert isinstance(source, str)
    if hasattr(Audio, "from_path") and callable(getattr(Audio, "from_path")):
        try:
            audio = Audio.from_path(source, mono=bool(mono) if mono is not None else None, fmt=fmt, source="path/url", meta=meta)  # type: ignore[arg-type]
            samples = get_audio_samples(audio)
            return LoadedAudio(audio=audio, samples=samples, sample_rate=int(getattr(audio, "sample_rate", 0) or 0))
        except Exception as e:
            if _verbose:
                _console.print(Panel(Text(str(e), style="bold yellow"), title="Audio.from_path failed; fallback decoding", border_style="yellow"))

    if hasattr(Audio, "from_file") and callable(getattr(Audio, "from_file")):
        try:
            audio = Audio.from_file(source, mono=bool(mono) if mono is not None else None, fmt=fmt, source="file", meta=meta)  # type: ignore[arg-type]
            samples = get_audio_samples(audio)
            return LoadedAudio(audio=audio, samples=samples, sample_rate=int(getattr(audio, "sample_rate", 0) or 0))
        except Exception as e:
            if _verbose:
                _console.print(Panel(Text(str(e), style="bold yellow"), title="Audio.from_file failed; fallback decoding", border_style="yellow"))

    # Fallback: fetch bytes and try stdlib decoders
    b = _fetch_bytes_from_source(source)
    try:
        samples, sr, _ = _decode_wav_bytes(b)
        audio = Audio(samples, sr, fmt="WAV", mono=False if samples.ndim == 2 else True, source="wav", meta=meta)
    except Exception:
        samples, sr, _ = _decode_aiff_bytes(b)
        audio = Audio(samples, sr, fmt="AIFF", mono=False if samples.ndim == 2 else True, source="aiff", meta=meta)

    if mono is True and samples.ndim == 2:
        samples = samples.mean(axis=1, keepdims=False)
        audio = audio.ensure_mono() if hasattr(audio, "ensure_mono") else Audio(samples, sr, fmt=fmt or "WAV", mono=True, source="mixdown", meta=meta)

    return LoadedAudio(audio=audio, samples=samples, sample_rate=sr)


# --------------------------------------------------------------------------------------
# Spectral analysis (STFT, dB scaling) and plotting utilities
# --------------------------------------------------------------------------------------

def _window_fn(name: str, n: int) -> np.ndarray:
    """Window generator (currently hann only for deterministic behavior)."""
    name = name.lower()
    if name in {"hann", "hanning"}:
        return np.hanning(n).astype(np.float32)
    if name in {"hamming"}:
        return np.hamming(n).astype(np.float32)
    raise ValueError(f"Unsupported window: {name}")

def _frame_signal(y: np.ndarray, frame_length: int, hop: int, center: bool = True) -> np.ndarray:
    """Create a 2D array of frames [frame_length, n_frames] using stride tricks."""
    if center:
        pad = frame_length // 2
        y = np.pad(y, (pad, pad), mode="reflect")
    n_frames = 1 + (len(y) - frame_length) // hop
    if n_frames <= 0:
        # Short signal: pad to at least 1 frame
        needed = frame_length + (frame_length // 2) * 2
        pad_width = max(0, needed - len(y))
        y = np.pad(y, (0, pad_width), mode="constant")
        n_frames = 1 + (len(y) - frame_length) // hop
    strides = (y.strides[0], y.strides[0] * hop)
    shape = (frame_length, n_frames)
    frames = np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides)
    return np.ascontiguousarray(frames)

def stft_mag(y: np.ndarray, n_fft: int, hop_length: int | None = None, win: str = "hann") -> np.ndarray:
    """
    Real STFT magnitude |FFT| with rfft on mono array y (float32).
    Returns shape (1 + n_fft//2, n_frames).
    """
    if hop_length is None:
        hop_length = n_fft // 4
    w = _window_fn(win, n_fft)
    frames = _frame_signal(y, n_fft, hop_length, center=True) * w[:, None]
    spec = np.fft.rfft(frames, n=n_fft, axis=0)
    mag = np.abs(spec).astype(np.float32)
    return mag

def power_to_db(S: np.ndarray, ref: float | t.Callable[[np.ndarray], float] = 1.0, amin: float = 1e-10, top_db: float = 80.0) -> np.ndarray:
    """
    Convert power spectrogram (S) to decibel units.
    """
    if callable(ref):
        ref_value = float(ref(S))
    else:
        ref_value = float(ref)
    S = np.maximum(S, amin)
    log_spec = 10.0 * np.log10(S) - 10.0 * np.log10(ref_value)
    log_spec = np.maximum(log_spec, log_spec.max() - float(top_db))
    return log_spec.astype(np.float32)

def amplitude_to_db(S: np.ndarray, ref: float | t.Callable[[np.ndarray], float] = 1.0, amin: float = 1e-10, top_db: float = 80.0) -> np.ndarray:
    """
    Convert amplitude spectrogram (|FFT|) to dB.
    """
    if callable(ref):
        ref_value = float(ref(S))
    else:
        ref_value = float(ref)
    S = np.maximum(S, amin)
    log_spec = 20.0 * np.log10(S) - 20.0 * np.log10(ref_value)
    log_spec = np.maximum(log_spec, log_spec.max() - float(top_db))
    return log_spec.astype(np.float32)

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def _safe_name(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9\-_\.]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "audio"

def _dump_meta_sidecar(path: Path, meta: dict) -> None:
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        _console.print(Panel(Text(str(e), style="bold yellow"), title=f"Meta sidecar write failed: {path.name}", border_style="yellow"))

def _save_figure(fig: plt.Figure, out_dir: Path, basename: str, meta: dict, cfg: PlotConfig) -> Path:
    _ensure_dir(out_dir)
    fname = f"{_safe_name(basename)}.png"
    fpath = out_dir / fname
    fig.savefig(fpath.as_posix(), dpi=cfg.dpi, facecolor=cfg.facecolor, bbox_inches="tight")
    sidecar = out_dir / f"{_safe_name(basename)}.json"
    _dump_meta_sidecar(sidecar, meta)
    return fpath

def _format_duration(n_samples: int, sr: int) -> str:
    sec = n_samples / float(sr) if sr else 0.0
    return f"{sec:.3f}s"

def _chan_count(x: np.ndarray) -> int:
    return 1 if x.ndim == 1 else x.shape[1]

def _to_mono(x: np.ndarray) -> np.ndarray:
    return x if x.ndim == 1 else x.mean(axis=1)


# --------------------------------------------------------------------------------------
# Plotting functions
# --------------------------------------------------------------------------------------

def plot_waveform(samples: np.ndarray, sr: int, title: str, out_dir: Path, cfg: PlotConfig = DEFAULT_CFG) -> Path:
    """
    Plot time-domain waveform. Supports mono or multi-channel.
    """
    n = samples.shape[0]
    ch = _chan_count(samples)
    t_axis = np.arange(n, dtype=np.float32) / float(sr)

    fig, ax = plt.subplots(figsize=cfg.figsize_wave, dpi=cfg.dpi)
    ax.set_title(f"{title} — Waveform\nsr={sr}Hz, duration={_format_duration(n, sr)}, channels={ch}", fontsize=12)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, linestyle="--", alpha=0.4)

    if ch == 1:
        ax.plot(t_axis, samples if samples.ndim == 1 else samples[:, 0], color="#1f77b4", lw=1.0, label="Mono")
    else:
        for ci in range(ch):
            ax.plot(t_axis, samples[:, ci], lw=0.9, label=f"Channel {ci+1}")
        ax.legend(loc="upper right", frameon=True)

    # Tight x-limits with small margin
    ax.set_xlim(0.0, max(t_axis[-1], 1e-6))

    meta = {
        "figure": "waveform",
        "title": title,
        "sr": sr,
        "n_samples": n,
        "duration_sec": n / float(sr) if sr > 0 else None,
        "channels": ch,
        "figsize": cfg.figsize_wave,
        "dpi": cfg.dpi,
    }
    path = _save_figure(fig, out_dir, f"{title}_waveform", meta, cfg)
    plt.close(fig)
    return path

def plot_spectrogram(samples: np.ndarray, sr: int, title: str, out_dir: Path, cfg: PlotConfig = DEFAULT_CFG) -> Path:
    """
    Plot a magnitude spectrogram in dB for mono or each channel separately (stacked vertically).
    """
    ch = _chan_count(samples)
    n_fft = cfg.n_fft
    hop = cfg.hop_length or (n_fft // 4)

    if ch == 1:
        xs = [samples if samples.ndim == 1 else samples[:, 0]]
        labels = ["Mono"]
        rows = 1
    else:
        xs = [samples[:, ci] for ci in range(ch)]
        labels = [f"Ch {i+1}" for i in range(ch)]
        rows = ch

    fig, axes = plt.subplots(rows, 1, figsize=cfg.figsize_spec, dpi=cfg.dpi, sharex=True)
    if rows == 1:
        axes = [axes]  # normalize

    for i, (y, label) in enumerate(zip(xs, labels)):
        mag = stft_mag(y, n_fft=n_fft, hop_length=hop, win=cfg.win)
        db = amplitude_to_db(mag, ref=np.max, top_db=cfg.top_db)

        freqs = np.linspace(0, sr / 2.0, mag.shape[0], dtype=np.float32)
        times = np.arange(mag.shape[1]) * (hop / float(sr))

        ax = axes[i]
        im = ax.pcolormesh(times, freqs, db, shading="auto", cmap=cfg.cmap)
        cbar = fig.colorbar(im, ax=ax, shrink=0.95, pad=0.01)
        cbar.set_label("Magnitude (dB)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(f"{label} Spectrogram (n_fft={n_fft}, hop={hop})", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.25)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{title} — Spectrogram (Linear Frequency)", fontsize=12)

    meta = {
        "figure": "spectrogram",
        "title": title,
        "sr": sr,
        "channels": ch,
        "n_fft": n_fft,
        "hop_length": hop,
        "top_db": cfg.top_db,
        "cmap": cfg.cmap,
        "figsize": cfg.figsize_spec,
        "dpi": cfg.dpi,
    }
    path = _save_figure(fig, out_dir, f"{title}_spectrogram", meta, cfg)
    plt.close(fig)
    return path

def plot_mel_spectrogram(samples: np.ndarray, sr: int, title: str, out_dir: Path, cfg: PlotConfig = DEFAULT_CFG) -> Path:
    """
    Plot Mel-spectrogram in dB. Uses the provided mel_filter_bank implementation.
    For multi-channel audio, a mono mix is used for a concise visualization.
    """
    y_mono = _to_mono(samples)
    n_fft = cfg.n_fft
    hop = cfg.hop_length or (n_fft // 4)
    fmax = sr * cfg.fmax_ratio

    mag = stft_mag(y_mono, n_fft=n_fft, hop_length=hop, win=cfg.win)        # (1 + n_fft//2, frames)
    power = (mag ** 2).astype(np.float32)
    mfb = mel_filter_bank(
        sample_rate=sr,
        n_fft=n_fft,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=fmax,
        htk=False,
        norm="slaney",
        dtype="float64",
    )  # shape (n_mels, 1 + n_fft//2)
    mel_power = (mfb @ power).astype(np.float32)
    mel_db = power_to_db(mel_power, ref=np.max, top_db=cfg.top_db)

    times = np.arange(mel_db.shape[1]) * (hop / float(sr))
    mel_bins = np.arange(cfg.n_mels)

    fig, ax = plt.subplots(1, 1, figsize=cfg.figsize_spec, dpi=cfg.dpi)
    im = ax.pcolormesh(times, mel_bins, mel_db, shading="auto", cmap=cfg.cmap)
    cbar = fig.colorbar(im, ax=ax, shrink=0.95, pad=0.01)
    cbar.set_label("Power (dB)")
    ax.set_ylabel("Mel bin")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"{title} — Mel-Spectrogram (n_mels={cfg.n_mels}, n_fft={n_fft}, hop={hop})", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.25)

    meta = {
        "figure": "mel_spectrogram",
        "title": title,
        "sr": sr,
        "n_fft": n_fft,
        "hop_length": hop,
        "n_mels": cfg.n_mels,
        "fmin": cfg.fmin,
        "fmax": fmax,
        "top_db": cfg.top_db,
        "cmap": cfg.cmap,
        "figsize": cfg.figsize_spec,
        "dpi": cfg.dpi,
    }
    path = _save_figure(fig, out_dir, f"{title}_mel_spectrogram", meta, cfg)
    plt.close(fig)
    return path

def plot_mel_filter_bank_curves(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float, title: str, out_dir: Path, cfg: PlotConfig = DEFAULT_CFG) -> Path:
    """Plot the mel filter bank responses across frequency bins."""
    mfb = mel_filter_bank(
        sample_rate=sr,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        htk=False,
        norm="slaney",
        dtype="float64",
    )
    freqs = np.linspace(0.0, sr / 2.0, 1 + n_fft // 2, dtype=np.float64)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi=cfg.dpi)
    for i in range(n_mels):
        ax.plot(freqs, mfb[i, :], lw=0.8, alpha=0.8)
    ax.set_title(f"{title} — Mel Filter Bank Curves\nsr={sr}Hz, n_fft={n_fft}, n_mels={n_mels}, [{fmin:.1f}, {fmax:.1f}] Hz", fontsize=12)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Gain")
    ax.grid(True, linestyle="--", alpha=0.35)

    meta = {
        "figure": "mel_filter_bank_curves",
        "title": title,
        "sr": sr,
        "n_fft": n_fft,
        "n_mels": n_mels,
        "fmin": fmin,
        "fmax": fmax,
        "figsize": (12, 6),
        "dpi": cfg.dpi,
    }
    path = _save_figure(fig, out_dir, f"{title}_mel_filters", meta, cfg)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------
# High-level orchestration: analyze + plot everything for an Audio object
# --------------------------------------------------------------------------------------

def analyze_and_plot(
    loaded: LoadedAudio,
    label: str,
    *,
    out_dir: Path,
    cfg: PlotConfig = DEFAULT_CFG,
    verbose: bool = True,
) -> dict[str, Path]:
    """
    Produce and save a suite of plots for the given loaded audio.
    Returns dict of figure names to saved paths.
    """
    audio = loaded.audio
    x = loaded.samples
    sr = int(loaded.sample_rate)

    ch = _chan_count(x)
    meta = {
        "label": label,
        "sr": sr,
        "duration_sec": x.shape[0] / float(sr) if sr > 0 else None,
        "channels": ch,
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "source": getattr(audio, "source", None),
        "format": getattr(audio, "fmt", None),
        "meta": getattr(audio, "meta", None),
    }
    if verbose:
        _console.print(_rich_panel_for_meta(meta, title=f"Analyze: {label}", border_style="green"))

    saved: dict[str, Path] = {}
    for desc, fn in track(
        [
            ("waveform", plot_waveform),
            ("spectrogram", plot_spectrogram),
            ("mel_spectrogram", plot_mel_spectrogram),
        ],
        description=f"[cyan]Rendering plots for: {label}",
    ):
        try:
            path = fn(x, sr, label, out_dir, cfg)  # type: ignore[misc]
            saved[desc] = path
        except Exception as e:
            _console.print(Panel(Text(str(e), style="bold red"), title=f"{desc} plot failed", border_style="red"))

    # Also render mel filter bank curves once per sample rate setting
    try:
        fmax = sr * cfg.fmax_ratio
        fb_path = plot_mel_filter_bank_curves(sr, cfg.n_fft, cfg.n_mels, cfg.fmin, fmax, f"{label}", out_dir, cfg)
        saved["mel_filter_bank"] = fb_path
    except Exception as e:
        _console.print(Panel(Text(str(e), style="bold yellow"), title="Mel filter bank plot failed", border_style="yellow"))

    return saved


# --------------------------------------------------------------------------------------
# CLI parsing
# --------------------------------------------------------------------------------------

def _parse_args(argv: list[str]) -> dict:
    import argparse

    p = argparse.ArgumentParser(description="Load audio from path/URL/bytes and generate detailed visualizations.")
    p.add_argument("--source", "-s", action="append", default=[], help="Path/URL/file:// to audio. Repeatable.")
    p.add_argument("--stdin-bytes", choices=["wav", "aiff"], help="Read audio bytes from stdin (format hint required).")
    p.add_argument("--mono", action="store_true", help="Force mono mixdown after loading.")
    p.add_argument("--plot-dir", default="figures", help="Directory to save figures (default: ./figures)")
    p.add_argument("--n-fft", type=int, default=DEFAULT_CFG.n_fft, help="FFT size")
    p.add_argument("--n-mels", type=int, default=DEFAULT_CFG.n_mels, help="Number of mel bands")
    p.add_argument("--top-db", type=float, default=DEFAULT_CFG.top_db, help="Top dB range for spectrograms")
    p.add_argument("--cmap", default=DEFAULT_CFG.cmap, help="Matplotlib colormap (e.g., 'magma', 'viridis')")
    p.add_argument("--no-demo", action="store_true", help="Skip the built-in quick demo pipeline.")
    p.add_argument("--title-suffix", default="", help="Optional suffix to append to plot titles.")
    args = p.parse_args(argv)

    return {
        "sources": args.source,
        "stdin_format": args.stdin_bytes,
        "mono": args.mono,
        "plot_dir": args.plot_dir,
        "n_fft": args.n_fft,
        "n_mels": args.n_mels,
        "top_db": args.top_db,
        "cmap": args.cmap,
        "run_demo": not args.no_demo,
        "title_suffix": args.title_suffix,
    }


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Enable verbose panels when run as a script (can override via env)
    Audio.set_verbose(_is_truthy(os.getenv("AUDIO_VERBOSE", "1")))

    # Parse CLI
    opts = _parse_args(sys.argv[1:])
    plot_dir = Path(opts["plot_dir"]).resolve()
    _ensure_dir(plot_dir)

    # Update default plotting config from CLI
    cfg = PlotConfig(
        n_fft=int(opts["n_fft"]),
        n_mels=int(opts["n_mels"]),
        top_db=float(opts["top_db"]),
        cmap=str(opts["cmap"]),
        hop_length=None,  # default n_fft//4
    )

    _console.rule("[bold green]Audio Loader + Visualization Pipeline")

    # If stdin bytes are provided, load them
    stdin_loaded: list[LoadedAudio] = []
    if opts["stdin_format"] is not None:
        b = sys.stdin.buffer.read()
        fmt_hint = "WAV" if opts["stdin_format"] == "wav" else "AIFF"
        try:
            loaded = load_audio_any(
                b,
                mono=opts["mono"],
                fmt=fmt_hint,
                source_tag=f"stdin-{fmt_hint.lower()}",
                meta={"note": "loaded from stdin bytes"},
                _verbose=True,
            )
            stdin_loaded.append(loaded)
            _console.print(Panel(Text(f"Loaded {fmt_hint} from stdin: sr={loaded.sample_rate}, shape={loaded.samples.shape}", style="cyan"),
                                 title="stdin load", border_style="cyan"))
        except Exception as e:
            _console.print(Panel(Text(str(e), style="bold red"), title="stdin load failed", border_style="red"))

    # Load provided sources
    loaded_list: list[LoadedAudio] = []
    for s in opts["sources"]:
        try:
            loaded = load_audio_any(
                s,
                mono=opts["mono"],
                fmt=None,
                source_tag="cli-source",
                meta={"cli": True, "original": s},
                _verbose=True,
            )
            loaded_list.append(loaded)
            _console.print(Panel(Text(f"Loaded: sr={loaded.sample_rate}, shape={loaded.samples.shape}", style="cyan"),
                                 title=f"source: {s}", border_style="cyan"))
        except Exception as e:
            _console.print(Panel(Text(str(e), style="bold red"), title=f"Load failed: {s}", border_style="red"))

    # Analyze and plot CLI-provided audio
    for i, loaded in enumerate(stdin_loaded + loaded_list):
        label = f"input_{i+1}"
        if opts["title_suffix"]:
            label = f"{label}_{opts['title_suffix']}"
        analyze_and_plot(loaded, label, out_dir=plot_dir, cfg=cfg, verbose=True)

    # ------------------------------------------------------------
    # Built-in Quick Demo as provided (extended with visualizations)
    # ------------------------------------------------------------
    if opts["run_demo"]:
        _console.rule("[bold green]Audio Class Quick Demo")

        # 1) Synthesize a 440 Hz sine, 1 second, mono
        sr = 22050
        t = np.linspace(0, 1.0, int(sr * 1.0), endpoint=False)
        sine = 0.2 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

        sine_audio = Audio(sine, sr, fmt="WAV", mono=True, source="synth", meta={"note": "440 Hz sine"})
        sine_audio.describe("Sine 440 Hz")

        # Visualize
        try:
            loaded = LoadedAudio(audio=sine_audio, samples=sine, sample_rate=sr)
            analyze_and_plot(loaded, "demo_sine_440Hz", out_dir=plot_dir, cfg=cfg, verbose=True)
        except Exception as e:
            _console.print(Panel(Text(str(e), style="bold red"), title="Plotting failed: Sine 440", border_style="red"))

        # 2) Mixdown from stereo to mono
        stereo = np.stack([sine, 0.2 * np.sin(2 * np.pi * 660.0 * t)], axis=1)  # (n, 2)
        stereo_audio = Audio(stereo, sr, fmt="WAV", mono=False, source="synth", meta={"note": "stereo test"})
        stereo_audio.describe("Stereo (original)")
        mono_audio = stereo_audio.ensure_mono()
        mono_audio.describe("Stereo -> Mono Mixdown")

        # Visualize stereo
        try:
            loaded = LoadedAudio(audio=stereo_audio, samples=stereo, sample_rate=sr)
            analyze_and_plot(loaded, "demo_stereo", out_dir=plot_dir, cfg=cfg, verbose=True)
        except Exception as e:
            _console.print(Panel(Text(str(e), style="bold red"), title="Plotting failed: Stereo", border_style="red"))

        # Visualize mono
        try:
            mono_samples = get_audio_samples(mono_audio)
            loaded = LoadedAudio(audio=mono_audio, samples=mono_samples, sample_rate=sr)
            analyze_and_plot(loaded, "demo_stereo_mono", out_dir=plot_dir, cfg=cfg, verbose=True)
        except Exception as e:
            _console.print(Panel(Text(str(e), style="bold red"), title="Plotting failed: Mono", border_style="red"))

        # 3) Resample with soxr (if available)
        try:
            sine_16k = sine_audio.resample(16000, quality="HQ", verbose=True)
            sine_16k.describe("Resampled to 16k")

            # Visualize resampled
            try:
                sine_16k_samples = get_audio_samples(sine_16k)
                loaded = LoadedAudio(audio=sine_16k, samples=sine_16k_samples, sample_rate=int(getattr(sine_16k, "sample_rate", 16000)))
                analyze_and_plot(loaded, "demo_sine_16k", out_dir=plot_dir, cfg=cfg, verbose=True)
            except Exception as e:
                _console.print(Panel(Text(str(e), style="bold red"), title="Plotting failed: Resampled 16k", border_style="red"))

        except ImportError as e:
            _console.print(Panel(Text(str(e), style="bold yellow"), title="Resample Skipped", border_style="yellow"))

        # 4) Base64 round-trip (in-memory)
        b64_wav = sine_audio.to_base64(fmt="WAV", subtype="PCM_16", verbose=True)
        restored = Audio.from_base64(b64_wav, mono=True, fmt="WAV", source="b64-restore", meta={"note": "roundtrip"})
        restored.describe("Restored from b64")

        # Visualize restored
        try:
            restored_samples = get_audio_samples(restored)
            loaded = LoadedAudio(audio=restored, samples=restored_samples, sample_rate=sr)
            analyze_and_plot(loaded, "demo_restored_b64", out_dir=plot_dir, cfg=cfg, verbose=True)
        except Exception as e:
            _console.print(Panel(Text(str(e), style="bold red"), title="Plotting failed: Restored b64", border_style="red"))

        # 5) Mel filter bank demonstration (float64 path, matching your usage)
        n_fft = cfg.n_fft
        mel_filters = mel_filter_bank(
            sample_rate=sine_audio.sample_rate,
            n_fft=n_fft,
            n_mels=40,
            fmin=20.0,
            fmax=sr / 2,
            htk=False,
            norm="slaney",
            dtype="float64",
        )
        _console.print(Panel(Text(f"Mel filter bank shape: {mel_filters.shape}, dtype: {mel_filters.dtype}", style="cyan"),
                             title="Mel Filter Bank", border_style="cyan"))

        # Visualize mel filter bank curves for sr=22050
        try:
            fb_path = plot_mel_filter_bank_curves(sr, n_fft, 40, 20.0, sr/2.0, "demo_mel_fb_sr22050", out_dir=plot_dir, cfg=cfg)
            _console.print(Panel(Text(f"Saved: {fb_path}", style="green"), title="Mel FB Curves", border_style="green"))
        except Exception as e:
            _console.print(Panel(Text(str(e), style="bold red"), title="Mel FB plot failed", border_style="red"))

        # 6) RawAudio protocol: PCM_S16LE mono samples
        pcm16 = (sine * 32767.0).astype(np.int16).tobytes()
        raw_payload = {"samples": pcm16, "sample_rate": sr, "channels": 1, "encoding": "PCM_S16LE"}
        raw_audio = Audio.from_raw_audio(raw_payload, mono=True, source="raw", meta={"desc": "PCM16 from sine"})
        raw_audio.describe("RawAudio/PCM16")

        # Visualize raw-audio
        try:
            raw_samples = get_audio_samples(raw_audio)
            loaded = LoadedAudio(audio=raw_audio, samples=raw_samples, sample_rate=sr)
            analyze_and_plot(loaded, "demo_raw_pcm16", out_dir=plot_dir, cfg=cfg, verbose=True)
        except Exception as e:
            _console.print(Panel(Text(str(e), style="bold red"), title="Plotting failed: Raw PCM16", border_style="red"))

        # 7) Scalar robustness checks for mel conversions (covers earlier traceback case)
        m20 = hertz_to_mel(20.0)          # scalar in -> scalar-like out
        f_back = mel_to_hertz(m20)
        _console.print(_rich_panel_for_meta({
            "mels@20Hz": float(np.asarray(m20)),
            "hz@mel(m20)": float(np.asarray(f_back)),
        }, title="Mel/Hz Scalar Roundtrip"))

        # 8) Negative frequency robustness (common if users pass np.fft.fftfreq outputs)
        test_fft_freqs = np.array([-512.0, -1.0, 0.0, 1.0, 1000.0], dtype=np.float64)
        mels_vals = hertz_to_mel(test_fft_freqs)  # should not raise; negatives clamped to 0
        hz_back = mel_to_hertz(mels_vals)
        _console.print(_rich_panel_for_meta({
            "fft_freqs_input": test_fft_freqs.tolist(),
            "mels_vals": np.asarray(mels_vals).tolist(),
            "hz_back": np.asarray(hz_back).tolist(),
        }, title="Negative Frequency Handling"))

        _console.rule("[bold green]Done")

    # Final note in logs: where to find figures
    _console.print(Panel(Text(str(plot_dir), style="bold green"), title="Figures saved under", border_style="green"))