#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Audio Mono-Channel Utilities, Robust Loaders, Resampling (soxr), and Mel Filter Bank (Slaney)

You want clean, standard-compliant, future-proof audio utilities with a technical tone and rich verbosity.
This single-file module delivers a focused, high-signal API with detailed inline documentation and examples.

What’s inside
- Class Audio (mono-channel control first)
  - __init__: Initialize with waveform, sample rate, and output container format (e.g., "WAV").
              Internally standardizes to float32; optional mono mixdown by averaging channels.
  - __repr__: Concise summary of key metadata.
  - _check_valid: Type/shape/format validation.
  - duration: Property in seconds.
  - from_url: Load via HTTP/HTTPS or file:// URL.
  - from_base64: Decode base64 container audio and load.
  - from_file: Load from absolute/relative path or file://.
  - from_bytes: Load from in-memory container bytes.
  - to_base64: Encode waveform to container bytes and return base64.
  - from_raw_audio: Build from raw PCM bytes or ndarray ("RawAudio protocol").
  - resample: High-quality resampling via soxr (pip install soxr).
  - ensure_mono/mixdown: Mono control.
  - metadata/describe: Rich panels for meta-data when verbose is enabled.

- Standalone Functions
  - hertz_to_mel: Hz -> Mel (Slaney by default; HTK if requested).
    Robust for scalar and array inputs. Gracefully handles negatives by clamping to 0 (audio-domain friendly).
  - mel_to_hertz: Mel -> Hz (inverse; scalar and array safe).
  - _create_triangular_filter_bank: Triangular filters in Hz domain with Slaney area normalization.
  - mel_filter_bank: Build Mel filter bank (n_mels x (1 + n_fft//2)) for power spectra.

Design and Conventions
- Internal numeric type for waveform: float32 in [-1, 1].
- Input acceptance: int PCM and uint PCM are normalized to float32.
- Mono control: mixdown at construction time by default (mono=True) unless disabled.
- File/URL/base64/bytes decoding via soundfile (libsndfile). Your system must support the given container.
  MP3 support depends on libsndfile build; for MP3 consider ffmpeg or transcode to wav/flac.
- Verbose rich panels:
  - Set AUDIO_VERBOSE=1 in environment, or call Audio.set_verbose(True).
  - Per-call verbose=True also works.
- Mel scale:
  - Default Slaney scale with piecewise linear/log mapping and "Slaney" area normalization (2/(f_right - f_left)).
  - htk=True switches to HTK mel equations.

Install dependencies (examples)
  pip install numpy soundfile soxr rich requests
"""

from __future__ import annotations

import base64
import io
import math
import os
import warnings
from typing import Any, Dict, Mapping, Optional, Union
from urllib.parse import urlparse, unquote

import numpy as np
import soundfile as sf  # libsndfile backend

try:
    import requests  # for from_url
except Exception:  # pragma: no cover
    requests = None  # type: ignore

try:
    import soxr  # high-quality resampling
except Exception:  # pragma: no cover
    soxr = None  # type: ignore

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.text import Text
from rich.pretty import Pretty


# --------------------------------------------------------------------------------------
# Global verbosity toggle via environment variable or code
# --------------------------------------------------------------------------------------

def _is_truthy(x: Any) -> bool:
    return str(x).strip().lower() in {"1", "true", "t", "yes", "y", "on"}

VERBOSE_DEFAULT = _is_truthy(os.getenv("AUDIO_VERBOSE", "0"))
_console = Console()


# --------------------------------------------------------------------------------------
# Utility helpers
# --------------------------------------------------------------------------------------

def _infer_format_from_path(path: str) -> Optional[str]:
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    mapping = {
        "wav": "WAV",
        "flac": "FLAC",
        "ogg": "OGG",
        "opus": "OGG",  # container OGG (OPUS subtype)
        "oga": "OGG",
        "aif": "AIFF",
        "aiff": "AIFF",
        "aifc": "AIFF",
        "caf": "CAF",
        "mp3": "MP3",  # soundfile may not support MP3 on your system
        "snd": "AU",
        "au": "AU",
    }
    return mapping.get(ext, None)


def _default_subtype_for(fmt: str) -> Optional[str]:
    fmt = fmt.upper()
    # Subtype selection for writing; None means let soundfile decide
    # - WAV: 16-bit PCM by default
    # - OGG: default to VORBIS; use subtype="OPUS" if you want OPUS
    # - FLAC: default None (libsndfile picks an appropriate FLAC subtype)
    # - AIFF/CAF/AU: default 16-bit PCM
    if fmt == "WAV":
        return "PCM_16"
    if fmt == "OGG":
        return "VORBIS"
    if fmt in {"AIFF", "CAF", "AU"}:
        return "PCM_16"
    return None


def _as_mono(x: np.ndarray) -> np.ndarray:
    # Input shape: (n,) or (n, ch). Output shape: (n,)
    if x.ndim == 1:
        return x
    return np.mean(x, axis=1, dtype=np.float32)


def _safe_int_to_float32(x: np.ndarray) -> np.ndarray:
    if np.issubdtype(x.dtype, np.floating):
        return x.astype(np.float32, copy=False)
    # Convert common PCM integers to float32 in [-1, 1]
    if np.issubdtype(x.dtype, np.signedinteger):
        bits = x.dtype.itemsize * 8
        max_val = float(2 ** (bits - 1))
        return (x.astype(np.float32) / max_val).clip(-1.0, 1.0)
    if np.issubdtype(x.dtype, np.unsignedinteger):
        bits = x.dtype.itemsize * 8
        max_val = float(2 ** bits)
        return ((x.astype(np.float32) / max_val) * 2.0 - 1.0).clip(-1.0, 1.0)
    raise TypeError(f"Unsupported dtype for conversion: {x.dtype}")


def _fft_frequencies(samplerate: int, n_fft: int) -> np.ndarray:
    """
    Frequencies for real FFT bins (0..Nyquist), size 1 + n_fft//2.
    Avoids negative frequencies entirely (suitable for magnitude or power spectra from rFFT).
    """
    return np.linspace(0.0, samplerate / 2.0, num=1 + n_fft // 2, endpoint=True, dtype=np.float64)


def _rich_panel_for_meta(meta: Mapping[str, Any], title: str = "Audio Meta"):
    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("Field", style="bold cyan", no_wrap=True)
    table.add_column("Value", style="white")
    for k, v in meta.items():
        table.add_row(str(k), Pretty(v, overflow="ignore"))
    return Panel(table, title=title, border_style="green")


# --------------------------------------------------------------------------------------
# Audio class
# --------------------------------------------------------------------------------------

class Audio:
    """
    Immutable-ish audio container focusing on mono-channel control, robust I/O, and utilities.

    Attributes
    - waveform: np.ndarray, float32, shape (n,) for mono or (n, channels) for multi
    - sample_rate: int (Hz)
    - format: str: default output container format on write/to_base64 (e.g., "WAV", "FLAC", "OGG")
    - source: Optional[str]: "file", "url", "bytes", "base64", "raw", or user-defined
    - meta: Optional[dict]: Free-form metadata

    Mono-channel control
    - By default, loaders and __init__ can mixdown to mono (mono=True).
    - Set mono=False to preserve channels; call .ensure_mono() when needed.

    Verbosity and Rich output
    - Global toggle: Audio.set_verbose(True/False) or env AUDIO_VERBOSE=1
    - Most constructors accept verbose=True to force metadata panels.

    Implementation notes
    - This class does not perform file writing to disk. Use to_base64() or soundfile.write externally.
    """

    # Class-wide verbose flag (Rich panels)
    _VERBOSE: bool = VERBOSE_DEFAULT

    def __init__(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        fmt: str = "WAV",
        *,
        mono: bool = True,
        copy: bool = False,
        source: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        verbose: Optional[bool] = None,
    ) -> None:
        x = np.array(waveform, copy=copy)
        x = _safe_int_to_float32(x)
        self.sample_rate = int(sample_rate)
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")

        if x.ndim == 1:
            pass
        elif x.ndim == 2:
            # Accept (n, ch)
            pass
        else:
            raise ValueError(f"waveform must be 1D or 2D, got shape {x.shape}")

        if mono and x.ndim == 2:
            x = _as_mono(x)

        self.waveform: np.ndarray = x.astype(np.float32, copy=False)
        self.format: str = str(fmt).upper()
        self.source: Optional[str] = source
        self.meta: Dict[str, Any] = dict(meta or {})
        self._check_valid()

        if self._resolve_verbose(verbose):
            self.describe()

    # -------------------------- core dunder --------------------------

    def __repr__(self) -> str:
        ch = self.channels
        shape = tuple(self.waveform.shape)
        return (
            f"Audio(sr={self.sample_rate}Hz, duration={self.duration:.3f}s, "
            f"shape={shape}, channels={ch}, dtype={self.waveform.dtype.name}, format='{self.format}')"
        )

    # -------------------------- validation --------------------------

    def _check_valid(self) -> None:
        if not isinstance(self.waveform, np.ndarray):
            raise TypeError("waveform must be a numpy.ndarray")
        if self.waveform.dtype != np.float32:
            self.waveform = self.waveform.astype(np.float32, copy=False)
        if self.waveform.ndim not in (1, 2):
            raise ValueError(f"waveform must be 1D or 2D, got shape {self.waveform.shape}")
        if self.waveform.size == 0:
            raise ValueError("waveform is empty")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not isinstance(self.format, str):
            raise TypeError("format must be a string like 'WAV', 'FLAC', 'OGG'")
        self.format = self.format.upper()

    # -------------------------- properties --------------------------

    @property
    def channels(self) -> int:
        return 1 if self.waveform.ndim == 1 else int(self.waveform.shape[1])

    @property
    def duration(self) -> float:
        n = self.waveform.shape[0]
        return float(n) / float(self.sample_rate)

    # -------------------------- verbosity ---------------------------

    @classmethod
    def set_verbose(cls, value: bool) -> None:
        cls._VERBOSE = bool(value)

    @classmethod
    def get_verbose(cls) -> bool:
        return bool(cls._VERBOSE)

    def _resolve_verbose(self, verbose: Optional[bool]) -> bool:
        return self._VERBOSE if verbose is None else bool(verbose)

    # -------------------------- metadata & rich ----------------------

    def metadata(self) -> Dict[str, Any]:
        x = self.waveform
        dur = self.duration
        peak = float(np.max(np.abs(x)))
        rms = float(np.sqrt(np.mean(x**2)))
        meta = {
            "shape": tuple(x.shape),
            "channels": self.channels,
            "dtype": x.dtype.name,
            "sample_rate": self.sample_rate,
            "duration_sec": round(dur, 6),
            "format": self.format,
            "source": self.source,
            "rms": round(rms, 8),
            "peak": round(peak, 8),
        }
        if self.meta:
            meta["extra_meta"] = dict(self.meta)
        return meta

    def describe(self, title: str = "Audio Meta", verbose: Optional[bool] = None) -> None:
        """Pretty-print metadata panel using rich."""
        if not self._resolve_verbose(verbose):
            return
        panel = _rich_panel_for_meta(self.metadata(), title=title)
        _console.print(panel)

    # -------------------------- utilities ---------------------------

    def ensure_mono(self) -> "Audio":
        """Return a mono-mixed copy (averaging channels)."""
        if self.channels == 1:
            return Audio(self.waveform.copy(), self.sample_rate, self.format, mono=True, source=self.source, meta=self.meta)
        return Audio(_as_mono(self.waveform), self.sample_rate, self.format, mono=True, source=self.source, meta=self.meta)

    mixdown = ensure_mono  # alias

    # -------------------------- factories ---------------------------

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        mono: bool = True,
        timeout: float = 30.0,
        headers: Optional[Dict[str, str]] = None,
        fmt: Optional[str] = None,
        source: Optional[str] = "url",
        meta: Optional[Dict[str, Any]] = None,
        verbose: Optional[bool] = None,
    ) -> "Audio":
        """
        Download audio via HTTP/HTTPS or load from file:// URL and construct an Audio instance.

        - If scheme is file://, defers to from_file.
        - Otherwise, requires 'requests' to fetch bytes.
        """
        parsed = urlparse(url)
        if parsed.scheme == "file":
            local_path = unquote(parsed.path)
            return cls.from_file(local_path, mono=mono, fmt=fmt, source=source, meta=meta, verbose=verbose)

        if parsed.scheme not in {"http", "https"}:
            raise ValueError(f"Unsupported URL scheme: {parsed.scheme or '(none)'}")

        if requests is None:
            raise ImportError("requests is required for from_url; please pip install requests")

        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.content
        audio = cls.from_bytes(data, mono=mono, fmt=fmt, source=source, meta=meta, verbose=verbose)
        audio.meta.setdefault("url", url)
        return audio

    @classmethod
    def from_base64(
        cls,
        b64: Union[str, bytes],
        *,
        mono: bool = True,
        fmt: Optional[str] = None,
        source: Optional[str] = "base64",
        meta: Optional[Dict[str, Any]] = None,
        verbose: Optional[bool] = None,
    ) -> "Audio":
        """
        Decode a base64-encoded audio container (e.g., WAV/FLAC/OGG) into an Audio instance.
        """
        if isinstance(b64, str):
            b64 = b64.strip()
            # Strip common data-URI prefix if present
            if "," in b64 and ";base64" in b64[:128]:
                b64 = b64.split(",", 1)[1]
            data = base64.b64decode(b64)
        else:
            data = base64.b64decode(b64)

        return cls.from_bytes(data, mono=mono, fmt=fmt, source=source, meta=meta, verbose=verbose)

    @classmethod
    def from_file(
        cls,
        path: str,
        *,
        mono: bool = True,
        fmt: Optional[str] = None,
        source: Optional[str] = "file",
        meta: Optional[Dict[str, Any]] = None,
        verbose: Optional[bool] = None,
    ) -> "Audio":
        """
        Read audio from a local file path (supports absolute/relative and 'file://' URLs).
        Uses soundfile to decode via libsndfile.
        """
        if path.startswith("file://"):
            path = unquote(urlparse(path).path)

        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        # Decode as float32, always_2d for uniform shape (frames, channels)
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        used_fmt = fmt or _infer_format_from_path(path) or "WAV"
        x = data if not mono else _as_mono(data)

        audio = cls(x, sr, used_fmt, mono=False, source=source, meta=meta, verbose=verbose)
        audio.meta.setdefault("path", os.path.abspath(path))
        return audio

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        mono: bool = True,
        fmt: Optional[str] = None,
        source: Optional[str] = "bytes",
        meta: Optional[Dict[str, Any]] = None,
        verbose: Optional[bool] = None,
    ) -> "Audio":
        """
        Decode from raw container bytes using soundfile (libsndfile).
        Supports formats that your libsndfile build can decode (e.g., WAV, FLAC, OGG).
        """
        with sf.SoundFile(io.BytesIO(data)) as f:
            sr = int(f.samplerate)
            x = f.read(dtype="float32", always_2d=True)
            container_fmt = f.format or "WAV"
            subtype = f.subtype or None

        used_fmt = (fmt or container_fmt or "WAV").upper()
        x = x if not mono else _as_mono(x)
        audio = cls(x, sr, used_fmt, mono=False, source=source, meta=meta, verbose=verbose)
        audio.meta.setdefault("container_format", container_fmt)
        if subtype:
            audio.meta.setdefault("container_subtype", subtype)
        return audio

    @classmethod
    def from_raw_audio(
        cls,
        payload: Mapping[str, Any],
        *,
        mono: bool = True,
        fmt: Optional[str] = None,
        source: Optional[str] = "raw",
        meta: Optional[Dict[str, Any]] = None,
        verbose: Optional[bool] = None,
    ) -> "Audio":
        """
        Construct an Audio instance from "RawAudio protocol" data.

        Supported payload fields (examples)
          {
            "samples": np.ndarray or bytes (PCM),
            "sample_rate": 48000,
            "channels": 1 or 2,            # optional; default inferred
            "encoding": "float32"          # or "PCM_S16LE", "PCM_U8", etc.
            "dtype": "float32"             # optional if samples is ndarray
            "endianness": "little"         # optional for PCM bytes
          }

        Cases
        - samples is np.ndarray: converted to float32 in [-1, 1] if integer. If 2D, expected shape (n, ch).
        - samples is bytes: must provide encoding like "PCM_S16LE"/"PCM_U8"/"FLOAT32"/etc.
        """
        if "samples" not in payload or "sample_rate" not in payload:
            raise KeyError("payload must contain 'samples' and 'sample_rate'")

        sr = int(payload["sample_rate"])
        enc = str(payload.get("encoding", "")).upper()
        ch_hint = payload.get("channels", None)

        samples = payload["samples"]

        if isinstance(samples, np.ndarray):
            x = _safe_int_to_float32(samples)
            if x.ndim == 1:
                if ch_hint and int(ch_hint) not in (None, 1):
                    warnings.warn("channels hint ignored: ndarray is 1D; assuming mono.")
            elif x.ndim == 2:
                if ch_hint and int(ch_hint) != x.shape[1]:
                    warnings.warn(f"channels hint ({ch_hint}) does not match ndarray shape {x.shape}; using shape's channel count.")
            else:
                raise ValueError("ndarray samples must be 1D or 2D")
        elif isinstance(samples, (bytes, bytearray, memoryview)):
            buf = bytes(samples)
            if enc in {"PCM_S16LE", "PCM_16", "PCM16", "S16LE"}:
                x = np.frombuffer(buf, dtype="<i2").astype(np.float32) / 32768.0
            elif enc in {"PCM_S24LE", "S24LE"}:
                b = np.frombuffer(buf, dtype=np.uint8)
                if len(b) % 3 != 0:
                    raise ValueError("PCM_S24LE bytes length must be divisible by 3")
                b = b.reshape(-1, 3)
                signed = (b[:, 2] & 0x80) != 0
                out = (b[:, 0].astype(np.int32)
                       | (b[:, 1].astype(np.int32) << 8)
                       | (b[:, 2].astype(np.int32) << 16))
                out = out.astype(np.int32)
                out[signed] |= ~0xFFFFFF  # sign extend top bits
                x = (out.astype(np.float32) / (2 ** 23))
            elif enc in {"PCM_S32LE", "S32LE"}:
                x = np.frombuffer(buf, dtype="<i4").astype(np.float32) / (2 ** 31)
            elif enc in {"PCM_U8", "U8"}:
                x = np.frombuffer(buf, dtype=np.uint8).astype(np.float32) / 255.0
                x = x * 2.0 - 1.0
            elif enc in {"FLOAT32", "F32LE"}:
                x = np.frombuffer(buf, dtype="<f4").astype(np.float32)
            elif enc in {"FLOAT64", "F64LE"}:
                x = np.frombuffer(buf, dtype="<f8").astype(np.float32)
            else:
                raise ValueError(
                    f"Unsupported or unspecified raw encoding '{enc}'. "
                    f"Provide 'encoding' like 'PCM_S16LE', 'PCM_U8', 'FLOAT32', etc."
                )

            if ch_hint is not None:
                ch = int(ch_hint)
                if ch <= 0:
                    raise ValueError("channels must be positive")
                if x.size % ch != 0:
                    raise ValueError(f"Number of samples ({x.size}) not divisible by channels ({ch})")
                x = x.reshape(-1, ch)
        else:
            raise TypeError("samples must be ndarray or bytes-like")

        used_fmt = (fmt or "WAV").upper()
        x = x if not mono else _as_mono(x)
        audio = cls(x, sr, used_fmt, mono=False, source=source, meta=meta, verbose=verbose)
        audio.meta.setdefault("raw_encoding", enc or ("ndarray/" + str(samples.dtype) if isinstance(samples, np.ndarray) else "bytes"))
        return audio

    # -------------------------- encoding ----------------------------

    def to_base64(
        self,
        *,
        fmt: Optional[str] = None,
        subtype: Optional[str] = None,
        verbose: Optional[bool] = None,
    ) -> str:
        """
        Encode the current waveform to a base64 audio container (via soundfile).
        - fmt: container format (default self.format: e.g., "WAV", "FLAC", "OGG")
        - subtype: data subtype (e.g., "PCM_16", "VORBIS", "OPUS"). If None, a sensible default is chosen.
        Returns: base64 string (no data-URI prefix)
        """
        used_fmt = (fmt or self.format or "WAV").upper()
        used_subtype = subtype or _default_subtype_for(used_fmt)

        buf = io.BytesIO()
        data = self.waveform  # shape: (frames,) or (frames, channels)
        try:
            sf.write(buf, data, self.sample_rate, format=used_fmt, subtype=used_subtype)
        except Exception as e:
            msg = f"soundfile.write failed for fmt={used_fmt}, subtype={used_subtype}. Error: {e}"
            raise RuntimeError(msg)

        raw_bytes = buf.getvalue()
        b64 = base64.b64encode(raw_bytes).decode("ascii")

        if self._resolve_verbose(verbose):
            info = {
                "format": used_fmt,
                "subtype": used_subtype,
                "bytes": len(raw_bytes),
                "base64_len": len(b64),
                "duration_sec": round(self.duration, 6),
                "sample_rate": self.sample_rate,
                "channels": self.channels,
            }
            _console.print(_rich_panel_for_meta(info, title="to_base64 Output"))

        return b64

    # -------------------------- resample ----------------------------

    def resample(
        self,
        new_rate: int,
        *,
        quality: str = "HQ",
        dtype: str = "float32",
        verbose: Optional[bool] = None,
    ) -> "Audio":
        """
        Resample audio using soxr for high-quality resampling.

        - new_rate: target sample rate (Hz)
        - quality: soxr quality profile ("HQ", "VHQ", "MQ", "LQ")
        - dtype: output dtype ("float32" recommended)
        Returns: new Audio instance with resampled waveform and updated sample_rate.

        Requires: pip install soxr
        """
        new_rate = int(new_rate)
        if new_rate <= 0:
            raise ValueError("new_rate must be positive")

        if soxr is None:
            raise ImportError("soxr is required for resample(). Install with: pip install soxr")

        x = self.waveform
        y = soxr.resample(x, self.sample_rate, new_rate, quality=quality)
        out = Audio(y, new_rate, self.format, mono=False, source=self.source, meta=self.meta, verbose=False)

        if self._resolve_verbose(verbose):
            info = {
                "old_sr": self.sample_rate,
                "new_sr": new_rate,
                "old_shape": tuple(x.shape),
                "new_shape": tuple(y.shape),
                "quality": quality,
                "dtype_out": dtype,
            }
            _console.print(_rich_panel_for_meta(info, title="Resample"))

        return out


# --------------------------------------------------------------------------------------
# Slaney/HTK mel conversions and filter banks
# --------------------------------------------------------------------------------------

def hertz_to_mel(frequencies: Union[float, np.ndarray], *, htk: bool = False) -> np.ndarray:
    """
    Convert frequency in Hz to mels.

    - htk=False: Slaney's Auditory Toolbox mel scale (piecewise linear/log)
      Constants:
        f_sp = 200/3 Hz/mel
        min_log_hz = 1000 Hz
        min_log_mel = min_log_hz / f_sp
        logstep = ln(6.4) / 27
      Note: Slaney's mel(1000 Hz) ~= 15, not 1000.
    - htk=True: HTK mel: mel = 2595 * log10(1 + f/700)

    Robustness:
    - Accepts scalar and array inputs.
    - Any negative Hz (e.g., from np.fft.fftfreq) are clamped to 0 (physically, negative frequencies
      mirror positive ones in real signals).

    Returns np.ndarray (float64).
    """
    f = np.asarray(frequencies, dtype=np.float64)

    # Clamp to non-negative to avoid domain/log issues and to support pipelines that pass full fftfreq grids.
    if np.any(f < 0):
        f = np.where(f < 0.0, 0.0, f)

    # Scalar fast path
    if f.ndim == 0:
        fv = float(f)
        if htk:
            mv = 2595.0 * math.log10(1.0 + fv / 700.0)
            return np.asarray(mv, dtype=np.float64)
        f_sp = 200.0 / 3.0
        min_log_hz = 1000.0
        min_log_mel = min_log_hz / f_sp  # 15
        logstep = math.log(6.4) / 27.0
        if fv < min_log_hz:
            mv = fv / f_sp
        else:
            mv = min_log_mel + math.log(max(fv, 1e-20) / min_log_hz) / logstep
        return np.asarray(mv, dtype=np.float64)

    # Vector path
    if htk:
        return 2595.0 * np.log10(1.0 + f / 700.0)

    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp  # 15
    logstep = np.log(6.4) / 27.0

    linear = f / f_sp
    above = f >= min_log_hz
    log_mels = min_log_mel + np.log(np.maximum(f, 1e-20) / min_log_hz) / logstep
    mels = np.where(above, log_mels, linear)
    return mels.astype(np.float64, copy=False)


def mel_to_hertz(mels: Union[float, np.ndarray], *, htk: bool = False) -> np.ndarray:
    """
    Convert mels to frequency in Hz.

    - htk=False: Inverse of Slaney scale.
    - htk=True: Inverse of HTK formula.

    Robustness:
    - Handles scalar and array inputs.
    - Output is guaranteed non-negative.

    Returns np.ndarray (float64).
    """
    m = np.asarray(mels, dtype=np.float64)

    # Scalar fast path
    if m.ndim == 0:
        mv = float(m)
        if htk:
            fv = 700.0 * (10.0 ** (mv / 2595.0) - 1.0)
            return np.asarray(max(fv, 0.0), dtype=np.float64)

        f_sp = 200.0 / 3.0
        min_log_hz = 1000.0
        min_log_mel = min_log_hz / f_sp  # 15
        logstep = math.log(6.4) / 27.0

        if mv < min_log_mel:
            fv = f_sp * mv
        else:
            fv = min_log_hz * math.exp(logstep * (mv - min_log_mel))
        return np.asarray(max(fv, 0.0), dtype=np.float64)

    # Vector path
    if htk:
        hz = 700.0 * (10.0 ** (m / 2595.0) - 1.0)
        return np.maximum(hz, 0.0, dtype=np.float64)

    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp  # 15
    logstep = np.log(6.4) / 27.0

    below = m < min_log_mel
    f_linear = f_sp * m
    f_log = min_log_hz * np.exp(logstep * (m - min_log_mel))
    hz = np.where(below, f_linear, f_log)
    return np.maximum(hz, 0.0, dtype=np.float64)


def _create_triangular_filter_bank(
    fft_freqs_hz: np.ndarray,  # shape (n_fft_bins,)
    band_edges_hz: np.ndarray,  # shape (n_mels + 2,)
    *,
    slaney_norm: bool = True,
) -> np.ndarray:
    """
    Generate triangular filters in Hz domain given FFT bin center frequencies and Mel band edges (Hz).

    - fft_freqs_hz: e.g., np.linspace(0, sr/2, 1 + n_fft//2)
    - band_edges_hz: frequency values (Hz) corresponding to mel-space evenly spaced points (n_mels + 2)
    - slaney_norm=True: scale filters to have approximately constant area (2/(f_right - f_left))

    Returns: filter bank matrix, shape (n_mels, n_fft_bins) in float64.
    """
    fft_freqs = np.asanyarray(fft_freqs_hz, dtype=np.float64)
    edges = np.asanyarray(band_edges_hz, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 3:
        raise ValueError("band_edges_hz must be 1D array with size >= 3")

    n_mels = edges.size - 2
    n_bins = fft_freqs.size
    fb = np.zeros((n_mels, n_bins), dtype=np.float64)

    # Ensure strictly non-degenerate edges
    if np.any(np.diff(edges) <= 0):
        raise ValueError("band_edges_hz must be strictly increasing")

    for i in range(n_mels):
        left = edges[i]
        center = edges[i + 1]
        right = edges[i + 2]

        # Rising and falling slopes; guard against divide-by-zero
        denom_l = max(center - left, 1e-12)
        denom_r = max(right - center, 1e-12)
        rising = (fft_freqs - left) / denom_l
        falling = (right - fft_freqs) / denom_r

        tri = np.maximum(0.0, np.minimum(rising, falling))
        fb[i, :] = tri

    if slaney_norm:
        # Scale by 2/(f_right - f_left) per Slaney to equalize area
        denom = (edges[2:] - edges[:-2])
        denom = np.where(denom <= 1e-12, 1e-12, denom)
        enorm = 2.0 / denom
        fb *= enorm[:, np.newaxis]

    return fb


def mel_filter_bank(
    sample_rate: int,
    n_fft: int,
    *,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    htk: bool = False,
    norm: Optional[str] = "slaney",
    dtype: str = "float32",
) -> np.ndarray:
    """
    Create a mel filter bank (n_mels, 1 + n_fft//2) mapping power spectrum to mel bands.

    Parameters
    - sample_rate: audio sampling rate
    - n_fft: FFT size used to compute spectrogram
    - n_mels: number of mel bands
    - fmin: minimum frequency (Hz) (will be clamped to [0, sr/2])
    - fmax: maximum frequency (Hz), default sr/2
    - htk: use HTK mel (if True). If False (default), use Slaney.
    - norm: if "slaney", perform Slaney area normalization; if None, no normalization.
    - dtype: output dtype (e.g., "float32" or "float64")

    Returns
    - Mel filter bank matrix of shape (n_mels, 1 + n_fft//2)

    Notes
    - This implementation only uses the non-negative, real FFT frequency grid (0..Nyquist).
      If your pipeline uses np.fft.fftfreq (which yields negative bins), compute your spectrogram
      using rFFT (e.g., np.fft.rfft) and use this filter bank as-is.
    """
    if n_fft <= 0:
        raise ValueError("n_fft must be positive")

    sr = int(sample_rate)
    nyq = sr / 2.0
    if fmax is None:
        fmax = nyq

    # Clamp fmin/fmax into valid range and validate ordering
    fmin = max(0.0, float(fmin))
    fmax = min(float(fmax), nyq)
    if not (0.0 <= fmin < fmax <= nyq + 1e-9):
        raise ValueError(f"Invalid fmin/fmax: fmin={fmin}, fmax={fmax}, sr={sr}")

    # Compute FFT bin frequencies (non-negative only)
    fft_freqs = _fft_frequencies(sr, n_fft)  # (1 + n_fft//2,)

    # Mel-space: n_mels + 2 band edges
    m_min = hertz_to_mel(fmin, htk=htk)
    m_max = hertz_to_mel(fmax, htk=htk)
    m_points = np.linspace(np.asarray(m_min, dtype=np.float64),
                           np.asarray(m_max, dtype=np.float64),
                           num=n_mels + 2)
    hz_points = mel_to_hertz(m_points, htk=htk)

    filter_bank = _create_triangular_filter_bank(fft_freqs, hz_points, slaney_norm=(norm == "slaney"))
    return filter_bank.astype(dtype, copy=False)


# --------------------------------------------------------------------------------------
# Examples and quick tests (executed if run as a script)
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Enable verbose panels when run as a script (can override via env)
    Audio.set_verbose(_is_truthy(os.getenv("AUDIO_VERBOSE", "1")))
    _console.rule("[bold green]Audio Class Quick Demo")

    # 1) Synthesize a 440 Hz sine, 1 second, mono
    sr = 22050
    t = np.linspace(0, 1.0, int(sr * 1.0), endpoint=False)
    sine = 0.2 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

    sine_audio = Audio(sine, sr, fmt="WAV", mono=True, source="synth", meta={"note": "440 Hz sine"})
    sine_audio.describe("Sine 440 Hz")

    # 2) Mixdown from stereo to mono
    stereo = np.stack([sine, 0.2 * np.sin(2 * np.pi * 660.0 * t)], axis=1)  # (n, 2)
    stereo_audio = Audio(stereo, sr, fmt="WAV", mono=False, source="synth", meta={"note": "stereo test"})
    stereo_audio.describe("Stereo (original)")
    mono_audio = stereo_audio.ensure_mono()
    mono_audio.describe("Stereo -> Mono Mixdown")

    # 3) Resample with soxr (if available)
    try:
        sine_16k = sine_audio.resample(16000, quality="HQ", verbose=True)
        sine_16k.describe("Resampled to 16k")
    except ImportError as e:
        _console.print(Panel(Text(str(e), style="bold yellow"), title="Resample Skipped", border_style="yellow"))

    # 4) Base64 round-trip (in-memory)
    b64_wav = sine_audio.to_base64(fmt="WAV", subtype="PCM_16", verbose=True)
    restored = Audio.from_base64(b64_wav, mono=True, fmt="WAV", source="b64-restore", meta={"note": "roundtrip"})
    restored.describe("Restored from b64")

    # 5) Mel filter bank demonstration (float64 path, matching your usage)
    n_fft = 1024
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

    # 6) RawAudio protocol: PCM_S16LE mono samples
    pcm16 = (sine * 32767.0).astype(np.int16).tobytes()
    raw_payload = {"samples": pcm16, "sample_rate": sr, "channels": 1, "encoding": "PCM_S16LE"}
    raw_audio = Audio.from_raw_audio(raw_payload, mono=True, source="raw", meta={"desc": "PCM16 from sine"})
    raw_audio.describe("RawAudio/PCM16")

    # 7) Scalar robustness checks for mel conversions (covers your earlier traceback case)
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
        "fft_freqs_input": test_fft_freqs,
        "mels_vals": mels_vals,
        "hz_back": hz_back,
    }, title="Negative Frequency Handling"))

    _console.rule("[bold green]Done")