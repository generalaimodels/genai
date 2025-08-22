#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio Transform Visualizer — robust, chunk-aware, single-file .py tool with version-compat fixes.

Author: "Best Coder" tutor mode
Tone: Technical, modern Pythonic style, strong coding standards and detailed inline explanations.

Highlights (updated per feedback):
- Chunk-aware processing for long audio (1s to 10min+):
  - Stream-safe Spectrogram/MelSpectrogram computation with configurable chunk and overlap.
  - Conservatively applies full-buffer for transforms where chunk-stitching is non-trivial.
- Torchaudio API compatibility shims:
  - Handles signature differences across versions for LFCC, Loudness, Speed, SpecAugment, MVDR, SoudenMVDR, PitchShift, TimeStretch, Vad.
  - InverseMelScale robust fallback via Mel filterbank pseudoinverse when least-squares is ill-conditioned.
- Rich-based verbose metadata for each transform:
  - Shapes, dtypes, timings, output image paths, and fallbacks used.
- Output per transform:
  - A folder per transform under outdir/, with 1024x1024 "original.png" and "<TransformName>.png".
  - Proper legends, labeled axes, titles, and colorbars for 2D.

Usage examples:
- Basic, verbose run on a local file (with chunking):
  python audio_transform_visualizer.py --source ./audio.wav --outdir ./transform_outputs --verbose true --chunk-seconds 30 --chunk-overlap 0.5
- From HTTP URL, limit to 90s for speed:
  python audio_transform_visualizer.py --source https://example.com/sample.wav --max-seconds 90
- From file:// URL and resample to 16 kHz:
  python audio_transform_visualizer.py --source file:///absolute/path/to/audio.wav --resample-to 16000
- From stdin bytes:
  cat audio.wav | python audio_transform_visualizer.py --source -

Dependencies:
- Python 3.9+
- torch, torchaudio
- matplotlib
- rich
- requests (optional; urllib fallback)
- numpy
"""

from __future__ import annotations

import argparse
import base64
import io
import math
import os
import sys
import time
import typing as T
from dataclasses import dataclass
from urllib.parse import urlparse

import torch
import torchaudio
from torchaudio.transforms import (
    AddNoise,              # def forward(self, waveform: torch.Tensor, noise: torch.Tensor, snr: torch.Tensor, lengths: T.Optional[torch.Tensor] = None) -> torch.Tensor:
    AmplitudeToDB,         # forward(self, x: torch.Tensor) -> torch.Tensor  # Convert amplitude spectrogram to decibel (dB) scale
    ComputeDeltas,         # forward(self, specgram: torch.Tensor) -> torch.Tensor  # Compute delta coefficients of spectrogram
    Convolve,              # forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor  # Convolve two 1D signals
    Deemphasis,            # forward(self, waveform: torch.Tensor) -> torch.Tensor # Apply deemphasis filter to waveform
    Fade,                  # forward(self, waveform: torch.Tensor) -> torch.Tensor  # Apply fade in/out
    FFTConvolve,           # forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor  # FFT-based convolution
    FrequencyMasking,      # forward(self, specgram: torch.Tensor, mask_param: int) -> torch.Tensor  # Apply frequency masking to spectrogram
    GriffinLim,            # forward(self, specgram: torch.Tensor) -> torch.Tensor:  # Reconstruct waveform from magnitude spectrogram
    InverseMelScale,       # forward(self, melspec: torch.Tensor) -> torch.Tensor:  # Convert mel spectrogram back to linear
    InverseSpectrogram,    # forward(self, spectrogram: torch.Tensor, length: T.Optional[int] = None) -> torch.Tensor:  # Inverse STFT to waveform
    LFCC,                  # forward(self, waveform: torch.Tensor) -> torch.Tensor:  # Compute Linear Frequency Cepstral Coefficients
    Loudness,              # forward(self, wavefrom: torch.Tensor) -> torch.Tensor  # Measure perceptual loudness
    MFCC,                  # forward(self, waveform: torch.Tensor) -> torch.Tensor:  # Compute Mel Frequency Cepstral Coefficients
    MVDR,                  # forward(self, specgram: torch.Tensor, mask_s: torch.Tensor, mask_n: T.Optional[torch.Tensor] = None) -> torch.Tensor:  # Apply MVDR beamforming
    MelScale,              # forward(self, specgram: torch.Tensor) -> torch.Tensor:  # Convert linear spectrogram to Mel scale
    MelSpectrogram,        # forward(self, x_mu: torch.Tensor) -> torch.Tensor: # Compute Mel-scaled spectrogram
    MuLawDecoding,         # forward(self, x_mu: torch.Tensor) -> torch.Tensor  # Decode waveform from mu-law encoding
    MuLawEncoding,         # forward(self, x: torch.Tensor) -> torch.Tensor:  # Mu-law encode waveform
    PSD,                   # forward(self, specgram: torch.Tensor, mask: T.Optional[torch.Tensor] = None):  # Estimate Power Spectral Density
    PitchShift,            # forward(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor  # Apply pitch shift
    Preemphasis,           # forward(self, waveform: torch.Tensor) -> torch.Tensor: # Apply preemphasis filter
    RTFMVDR,               # def forward(self,specgram: torch.Tensor,rtf: torch.Tensor,psd_n: torch.Tensor,reference_channel: T.Union[int, torch.Tensor],diagonal_loading: bool = True,diag_eps: float = 1e-7,eps: float = 1e-8,) -> torch.Tensor:
    Resample,              # forward(self, waveform: torch.Tensor) -> torch.Tensor:
    SlidingWindowCmn,      # forward(self, specgram: torch.Tensor) -> torch.Tensor:
    SoudenMVDR,            # forward(self,specgram: torch.Tensor,psd_s: torch.Tensor,psd_n: torch.Tensor,reference_channel: T.Union[int, torch.Tensor],diagonal_loading: bool = True,diag_eps: float = 1e-7,eps: float = 1e-8,) -> torch.Tensor:
    SpecAugment,           # forward(self, specgram: torch.Tensor) -> torch.Tensor  # Apply SpecAugment (time + frequency masking)
    SpectralCentroid,      # forward(self, waveform: torch.Tensor) -> torch.Tensor  # Compute spectral centroid
    Spectrogram,           # forward(self, waveform: torch.Tensor) -> torch.Tensor  # Compute spectrogram (STFT)
    Speed,                 # forward(self, waveform, lengths: T.Optional[torch.Tensor] = None) -> T.Tuple[torch.Tensor, T.Optional[torch.Tensor]]:
    SpeedPerturbation,     # forward(self, waveform: torch.Tensor, lengths: T.Optional[torch.Tensor] = None) -> T.Tuple[torch.Tensor, T.Optional[torch.Tensor]]:
    TimeMasking,          
    TimeStretch,           # forward(self, complex_specgrams: Tensor, overriding_rate: T.Optional[float] = None) -> Tensor:
    Vad,                   # forward(self, waveform: torch.Tensor) -> torch.Tensor  # Apply Voice Activity Detection
    Vol,                   # forward(self, waveform: torch.Tensor) -> torch.Tensor  # Adjust volume
)
import torchaudio.functional as F

# Third-party utilities
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import requests
except Exception:
    requests = None

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text


# =========================
# Configuration structures
# =========================

@dataclass
class AppConfig:
    source: str | None
    outdir: str = "transform_outputs"
    verbose: bool = True
    resample_to: int | None = None
    max_seconds: float | None = 600.0  # up to 10 min
    device: str = "cpu"
    image_size_px: int = 1024
    dpi: int = 100
    seed: int = 42
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    top_db: float = 80.0
    num_mels: int = 128
    num_mfcc: int = 40
    num_lfcc: int = 40
    out_audio: bool = False
    enable_beamforming: bool = True
    time_stretch_rate: float = 1.2
    pitch_semitones: float = 3.0
    speed_factor: float = 1.1
    fade_in_seconds: float = 0.2
    fade_out_seconds: float = 0.2
    snr_db: float = 10.0
    vol_gain_db: float = 6.0
    frequency_mask_param: int = 24
    time_mask_param: int = 30
    # Chunk-aware additions
    chunk_seconds: float | None = 30.0   # set None to disable chunking
    chunk_overlap_seconds: float = 0.5   # overlap for streaming-friendly transforms
    # Beamforming workload cap (avoid giant tensors)
    bf_max_seconds: float = 30.0


@dataclass
class TransformResult:
    name: str
    ok: bool
    in_shape: T.Optional[T.Tuple[int, ...]] = None
    out_shape: T.Optional[T.Tuple[int, ...]] = None
    in_dtype: T.Optional[torch.dtype] = None
    out_dtype: T.Optional[torch.dtype] = None
    plot_type: str = "waveform"
    original_path: str | None = None
    transformed_path: str | None = None
    note: str | None = None
    elapsed_ms: float | None = None
    error: str | None = None


@dataclass
class AudioData:
    waveform: torch.Tensor  # [C, T]
    sample_rate: int
    path_or_desc: str


class SafeRNG:
    def __init__(self, seed: int):
        self._gen = torch.Generator(device="cpu")
        self._gen.manual_seed(seed)
    def rand(self, *shape: int) -> torch.Tensor:
        return torch.rand(shape, generator=self._gen)
    def randn(self, *shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=self._gen)


# ===========
# Utilities
# ===========

console = Console(highlight=False, soft_wrap=True)

def log(panel_title: str, content: str, style: str = "cyan"):
    if app_cfg.verbose:
        console.print(Panel.fit(Text(content), title=panel_title, border_style=style))

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()

def secs_to_samples(seconds: float, sr: int) -> int:
    return int(round(seconds * sr))

def trim_to_max_seconds(waveform: torch.Tensor, sample_rate: int, max_seconds: float | None) -> torch.Tensor:
    if max_seconds is None:
        return waveform
    max_len = secs_to_samples(max_seconds, sample_rate)
    if waveform.shape[-1] > max_len:
        return waveform[..., :max_len]
    return waveform

def normalize_waveform(wf: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    peak = wf.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
    return (wf / peak).clamp(-1.0, 1.0)

def descr_tensor(x: torch.Tensor) -> str:
    return f"shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}, min={x.min().item():.4f}, max={x.max().item():.4f}"

def resolve_local_path_from_file_url(url: str) -> str:
    parsed = urlparse(url)
    return parsed.path

def load_from_url(url: str) -> bytes:
    if requests:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.content
    import urllib.request
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read()

def decode_base64_data(b64: str) -> bytes:
    if b64.startswith("base64://"):
        b64 = b64[len("base64://"):]
    return base64.b64decode(b64)

def load_audio_from_source(cfg: AppConfig) -> AudioData:
    """
    Load audio from:
    - local path | file:// URL | http(s):// URL
    - stdin ("-")
    - base64://... (inline base64 content)
    Returns float32 waveform [C, T] in [-1, 1] and sample_rate.
    """
    source = cfg.source
    if source is None:
        sr = 44100
        dur = 8.0
        t = torch.linspace(0, dur, int(sr * dur), dtype=torch.float32)
        sig = 0.45 * torch.sin(2 * math.pi * 220 * t) + 0.35 * torch.sin(2 * math.pi * 440 * t)
        wf = normalize_waveform(sig.unsqueeze(0))
        log("Info", "No source provided. Using synthesized 8s sine mix.")
        return AudioData(waveform=wf, sample_rate=sr, path_or_desc="synthetic:sines")
    try:
        if source == "-":
            raw = sys.stdin.buffer.read()
            wf, sr = torchaudio.load(io.BytesIO(raw))
            return AudioData(waveform=normalize_waveform(wf), sample_rate=sr, path_or_desc="stdin-bytes")
        if source.startswith("base64://"):
            raw = decode_base64_data(source)
            wf, sr = torchaudio.load(io.BytesIO(raw))
            return AudioData(waveform=normalize_waveform(wf), sample_rate=sr, path_or_desc="base64")
        if source.startswith("http://") or source.startswith("https://"):
            raw = load_from_url(source)
            wf, sr = torchaudio.load(io.BytesIO(raw))
            return AudioData(waveform=normalize_waveform(wf), sample_rate=sr, path_or_desc=source)
        if source.startswith("file://"):
            path = resolve_local_path_from_file_url(source)
            wf, sr = torchaudio.load(path)
            return AudioData(waveform=normalize_waveform(wf), sample_rate=sr, path_or_desc=path)
        wf, sr = torchaudio.load(source)
        return AudioData(waveform=normalize_waveform(wf), sample_rate=sr, path_or_desc=os.path.abspath(source))
    except Exception as e:
        # Optional fallback via soundfile if available
        try:
            import soundfile as sf
            if source.startswith(("http://", "https://")):
                raw = load_from_url(source)
                data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
            elif source.startswith("file://"):
                path = resolve_local_path_from_file_url(source)
                data, sr = sf.read(path, dtype="float32", always_2d=True)
            else:
                data, sr = sf.read(source, dtype="float32", always_2d=True)
            wf = torch.from_numpy(data.T.copy())
            return AudioData(waveform=normalize_waveform(wf), sample_rate=sr, path_or_desc=source)
        except Exception as e2:
            raise RuntimeError(f"Failed to load audio from {source}. torchaudio error={e}. fallback error={e2}") from e2

def maybe_resample(audio: AudioData, target_sr: int | None) -> AudioData:
    if target_sr is None or audio.sample_rate == target_sr:
        return audio
    resampler = Resample(orig_freq=audio.sample_rate, new_freq=target_sr)
    wf = resampler(audio.waveform)
    return AudioData(waveform=wf, sample_rate=target_sr, path_or_desc=audio.path_or_desc)

def fix_duration(audio: AudioData, max_seconds: float | None) -> AudioData:
    wf = trim_to_max_seconds(audio.waveform, audio.sample_rate, max_seconds)
    return AudioData(waveform=wf, sample_rate=audio.sample_rate, path_or_desc=audio.path_or_desc)

def with_device(audio: AudioData, device: str) -> AudioData:
    wf = audio.waveform.to(device)
    return AudioData(waveform=wf, sample_rate=audio.sample_rate, path_or_desc=audio.path_or_desc)


# ======================
# Chunking infrastructure
# ======================

class Chunker:
    """
    Simple chunker for long audio. It yields overlapped chunks for stream-friendly computations.
    Overlap helps reduce boundary artifacts for frame-based transforms.
    """
    def __init__(self, sr: int, chunk_seconds: float | None, overlap_seconds: float):
        self.sr = sr
        self.chunk_samples = None if chunk_seconds is None else max(1, secs_to_samples(chunk_seconds, sr))
        self.overlap_samples = max(0, secs_to_samples(overlap_seconds, sr))
    def iter_chunks(self, x: torch.Tensor) -> T.Iterable[tuple[int, int, torch.Tensor]]:
        # x: [C, T]
        Tlen = x.shape[-1]
        if self.chunk_samples is None or self.chunk_samples >= Tlen:
            yield 0, Tlen, x
            return
        step = max(1, self.chunk_samples - self.overlap_samples)
        start = 0
        while start < Tlen:
            end = min(Tlen, start + self.chunk_samples)
            yield start, end, x[..., start:end]
            if end == Tlen:
                break
            start += step


# ======================
# Plotting helpers
# ======================

def figure_1024(title: str, dpi: int) -> plt.Figure:
    size_in = 1024 / dpi
    fig = plt.figure(figsize=(size_in, size_in), dpi=dpi, constrained_layout=True)
    fig.suptitle(title, fontsize=12)
    return fig

def save_figure(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

def plot_waveform_image(waveform: torch.Tensor, sample_rate: int, title: str, out_path: str, dpi: int) -> None:
    fig = figure_1024(title, dpi)
    ax = fig.add_subplot(111)
    num_ch, length = waveform.shape
    t_axis = torch.linspace(0, length / sample_rate, length, dtype=torch.float32).cpu().numpy()
    for ch in range(num_ch):
        ax.plot(t_axis, waveform[ch].detach().cpu().numpy(), linewidth=0.75, label=f"ch{ch}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=8)
    save_figure(fig, out_path)

def plot_line_image(series: torch.Tensor, title: str, out_path: str, dpi: int, x_label: str = "Frame") -> None:
    data = series.detach().cpu()
    if data.ndim == 1:
        series_list = [data]
        labels = ["value"]
    elif data.ndim == 2:
        series_list = [data[i] for i in range(data.shape[0])]
        labels = [f"ch{i}" for i in range(data.shape[0])]
    else:
        flat = data.reshape(-1, data.shape[-1])
        series_list = [flat[i] for i in range(min(flat.shape[0], 6))]
        labels = [f"series{i}" for i in range(len(series_list))]
    fig = figure_1024(title, dpi)
    ax = fig.add_subplot(111)
    x_vals = np.arange(series_list[0].numel())
    for i, s in enumerate(series_list):
        ax.plot(x_vals, s.numpy(), linewidth=0.9, label=labels[i])
    ax.set_xlabel(x_label)
    ax.set_ylabel("Value")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=8)
    save_figure(fig, out_path)

def plot_heatmap_image(
    matrix: torch.Tensor, title: str, out_path: str, dpi: int, x_label: str = "Frame", y_label: str = "Bin", add_colorbar: bool = True
) -> None:
    mat = matrix.detach().cpu()
    if mat.ndim == 3:
        mat = mat[0]
    if mat.ndim == 1:
        mat = mat.unsqueeze(0)
    fig = figure_1024(title, dpi)
    ax = fig.add_subplot(111)
    im = ax.imshow(mat.numpy(), origin="lower", aspect="auto", interpolation="none", cmap="magma")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(False)
    if add_colorbar:
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    save_figure(fig, out_path)

def placeholder_image(msg: str, title: str, out_path: str, dpi: int) -> None:
    fig = figure_1024(title, dpi)
    ax = fig.add_subplot(111)
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=11)
    ax.set_axis_off()
    save_figure(fig, out_path)


# ======================
# Version-compat helpers
# ======================

def compat_LFCC(sample_rate: int, n_lfcc: int, n_fft: int, hop_length: int) -> LFCC:
    """
    LFCC signature varies across versions:
    - Newer: LFCC(sample_rate=..., n_lfcc=..., n_fft=..., hop_length=...)
    - Older: LFCC(sample_rate=..., n_lfcc=...)
    - Some:  LFCC(sample_rate=..., n_lfcc=..., speckwargs={'n_fft':..., 'hop_length':...})
    """
    try:
        return LFCC(sample_rate=sample_rate, n_lfcc=n_lfcc)
    except TypeError:
        try:
            return LFCC(sample_rate=sample_rate, n_lfcc=n_lfcc)
        except TypeError:
            return LFCC(sample_rate=sample_rate, n_lfcc=n_lfcc, speckwargs={'n_fft': n_fft, 'hop_length': hop_length})

def compat_Loudness(sample_rate: int, n_fft: int, hop_length: int) -> Loudness | None:
    """
    Newer: Loudness(sample_rate=..., n_fft=..., hop_length=...)
    Older: Loudness(sample_rate=...)
    """
    try:
        return Loudness(sample_rate=sample_rate)
    except TypeError:
        try:
            return Loudness(sample_rate=sample_rate)
        except Exception:
            return None  # fallback will be RMS proxy

def compat_SpecAugment(freq_mask_param: int, time_mask_param: int) -> SpecAugment | None:
    """
    Newer: SpecAugment(time_warp=False, time_mask_param=..., freq_mask_param=...)
    Older: SpecAugment(time_mask_param=..., freq_mask_param=...)
    If both unavailable, fallback to FrequencyMasking + TimeMasking pipeline.
    """
    try:
        return SpecAugment(time_warp=False, time_mask_param=time_mask_param, freq_mask_param=freq_mask_param)
    except TypeError:
        try:
            return SpecAugment(time_mask_param=time_mask_param, freq_mask_param=freq_mask_param)
        except TypeError:
            return None

def compat_Speed(sample_rate: int, factor: float) -> Speed:
    """
    Try multiple init signatures; forward can return (y, lengths) or just y.
    """
    last_ex = None
    for args in (
        dict(factor=factor),
        dict(),
        dict(sample_rate=sample_rate, factor=factor),
    ):
        try:
            return Speed(**args)
        except Exception as e:
            last_ex = e
            continue
    raise last_ex

def compat_PitchShift(sample_rate: int, n_steps: float, x: torch.Tensor) -> torch.Tensor:
    """
    Newer: PitchShift(sample_rate=..., n_steps=...)(x)
    Older: tr = PitchShift(n_steps=...); tr(x, sample_rate)
    """
    try:
        return PitchShift(sample_rate=sample_rate, n_steps=n_steps)(x)
    except Exception:
        tr = PitchShift(n_steps=n_steps)
        return tr(x, sample_rate)

def compat_MVDR_call(spec_c: torch.Tensor, mask_s: torch.Tensor, mask_n: torch.Tensor) -> torch.Tensor:
    """
    MVDR(diagonal_loading=True) vs MVDR() only; also handle mask rank [F,T] vs [C,F,T]
    """
    def ensure_mask3(mask: torch.Tensor, C: int) -> torch.Tensor:
        if mask.ndim == 2:
            return mask.unsqueeze(0).expand(C, -1, -1)
        return mask
    C = spec_c.shape[0]
    for init_kwargs in (dict(diagonal_loading=True), dict()):
        try:
            return MVDR(**init_kwargs)(spec_c, mask_s, mask_n)
        except TypeError:
            try:
                return MVDR(**init_kwargs)(spec_c, ensure_mask3(mask_s, C), ensure_mask3(mask_n, C))
            except Exception:
                continue
        except Exception:
            continue
    raise RuntimeError("MVDR call failed for all compatible signatures")

def compat_SoudenMVDR_call(spec_c: torch.Tensor, psd_s: torch.Tensor, psd_n: torch.Tensor, ref_ch: int) -> torch.Tensor:
    for init_kwargs in (dict(diagonal_loading=True), dict()):
        try:
            return SoudenMVDR(**init_kwargs)(spec_c, psd_s, psd_n, reference_channel=ref_ch)
        except TypeError:
            try:
                return SoudenMVDR(**init_kwargs)(spec_c, psd_s, psd_n, ref_ch)
            except Exception:
                continue
        except Exception:
            continue
    raise RuntimeError("SoudenMVDR call failed for all compatible signatures")

def inverse_mel_scale_fallback(melspec: torch.Tensor, n_stft: int, n_mels: int, sample_rate: int) -> torch.Tensor:
    """
    Robust pseudoinverse fallback: spec ≈ (pinv(Fb).T) @ melspec, where Fb is Mel filterbank [F, M].
    Accepts melspec [C, M, T], returns approx linear [C, F, T].
    """
    device = melspec.device
    Fb = None
    try:
        Fb = F.melscale_fbanks(
            n_freqs=n_stft, f_min=0.0, f_max=float(sample_rate) / 2.0, n_mels=n_mels,
            sample_rate=sample_rate, norm="slaney", mel_scale="htk"
        )
    except Exception:
        Fb = F.create_fb_matrix(
            n_stft, f_min=0.0, f_max=float(sample_rate) / 2.0, n_mels=n_mels, sample_rate=sample_rate
        )
    Fb = Fb.to(device=device, dtype=melspec.dtype)
    # Ensure Fb is [F, M]
    if Fb.shape[0] != n_stft or Fb.shape[1] != n_mels:
        if Fb.T.shape[0] == n_stft and Fb.T.shape[1] == n_mels:
            Fb = Fb.T
        else:
            raise RuntimeError(f"Unexpected fbanks shape {tuple(Fb.shape)} for n_stft={n_stft}, n_mels={n_mels}")
    W = torch.pinverse(Fb).T  # [F, M]
    C = melspec.shape[0]
    Tlen = melspec.shape[-1]
    spec_est = torch.zeros((C, n_stft, Tlen), dtype=melspec.dtype, device=device)
    for c in range(C):
        spec_est[c] = W @ melspec[c]
    return spec_est


# ========================
# Transform Demo Routines
# ========================

class TransformSuite:
    def __init__(self, cfg: AppConfig, audio: AudioData, rng: SafeRNG):
        self.cfg = cfg
        self.audio = audio
        self.rng = rng
        self.chunker = Chunker(audio.sample_rate, cfg.chunk_seconds, cfg.chunk_overlap_seconds)

    # ---------- Spectral builders (streaming-aware) ----------

    def _spec_stream(self, waveform: torch.Tensor, power: float | None = 2.0, return_complex: bool = False) -> torch.Tensor:
        n_fft, hop_length, win_length = self.cfg.n_fft, self.cfg.hop_length, self.cfg.win_length
        spec_t = Spectrogram(
            n_fft=n_fft, hop_length=hop_length, win_length=win_length, pad=0, window_fn=torch.hann_window,
            power=power, return_complex=return_complex
        )
        chunks: list[torch.Tensor] = []
        for _, _, x in self.chunker.iter_chunks(waveform):
            chunks.append(spec_t(x))
        return torch.cat(chunks, dim=-1)  # concat along time frames

    def _spec(self, waveform: torch.Tensor, power: float | None = 2.0, return_complex: bool = False) -> torch.Tensor:
        # Decide streaming vs full
        if self.cfg.chunk_seconds is not None and waveform.shape[-1] > secs_to_samples(self.cfg.chunk_seconds, self.audio.sample_rate):
            return self._spec_stream(waveform, power=power, return_complex=return_complex)
        spec_t = Spectrogram(
            n_fft=self.cfg.n_fft, hop_length=self.cfg.hop_length, win_length=self.cfg.win_length, pad=0,
            window_fn=torch.hann_window, power=power, return_complex=return_complex
        )
        return spec_t(waveform)

    def _melspec(self, waveform: torch.Tensor, power: float = 2.0) -> torch.Tensor:
        mel_t = MelSpectrogram(
            sample_rate=self.audio.sample_rate, n_fft=self.cfg.n_fft, hop_length=self.cfg.hop_length,
            win_length=self.cfg.win_length, n_mels=self.cfg.num_mels, power=power
        )
        chunks: list[torch.Tensor] = []
        for _, _, x in self.chunker.iter_chunks(waveform):
            chunks.append(mel_t(x))
        return torch.cat(chunks, dim=-1)

    def _mfcc(self, waveform: torch.Tensor) -> torch.Tensor:
        mfcc = MFCC(
            sample_rate=self.audio.sample_rate, n_mfcc=self.cfg.num_mfcc,
            melkwargs=dict(n_fft=self.cfg.n_fft, hop_length=self.cfg.hop_length, n_mels=self.cfg.num_mels)
        )
        chunks: list[torch.Tensor] = []
        for _, _, x in self.chunker.iter_chunks(waveform):
            chunks.append(mfcc(x))
        return torch.cat(chunks, dim=-1)

    def _lfcc(self, waveform: torch.Tensor) -> torch.Tensor:
        lfcc = compat_LFCC(self.audio.sample_rate, self.cfg.num_lfcc, self.cfg.n_fft, self.cfg.hop_length)
        chunks: list[torch.Tensor] = []
        for _, _, x in self.chunker.iter_chunks(waveform):
            chunks.append(lfcc(x))
        return torch.cat(chunks, dim=-1)

    def _beamforming_inputs(self) -> T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Construct demo inputs for MVDR/PSD/SoudenMVDR/RTFMVDR:
        - Use at most bf_max_seconds for tractable memory.
        - 2-channel waveform: ch0 original, ch1 delayed+noise
        - complex STFT
        """
        wf = trim_to_max_seconds(self.audio.waveform, self.audio.sample_rate, self.cfg.bf_max_seconds)
        sr = self.audio.sample_rate
        delay_samples = max(1, int(0.002 * sr))  # 2 ms
        pad = torch.zeros((1, delay_samples), device=wf.device, dtype=wf.dtype)
        ch1 = torch.cat([pad, wf[0:1, :-delay_samples]], dim=-1)
        ch1 = ch1 + 0.02 * self.rng.randn(*ch1.shape).to(wf.device)
        mc = torch.cat([wf[0:1], ch1], dim=0)  # [2, T]
        spec_c = self._spec(mc, return_complex=True)
        mag = spec_c.abs().mean(dim=0)  # [F, T]
        thresh = mag.mean() * 0.7
        mask_s = (mag >= thresh).float()
        mask_n = 1.0 - mask_s
        ref_ch = 0
        return spec_c, mask_s, mask_n, ref_ch

    # ---------- Plot pair helpers ----------

    def _save_pair_waveform(self, name: str, x_in: torch.Tensor, x_out: torch.Tensor) -> T.Tuple[str, str]:
        outdir = os.path.join(self.cfg.outdir, name)
        ensure_dir(outdir)
        p0 = os.path.join(outdir, "original.png")
        p1 = os.path.join(outdir, f"{name}.png")
        plot_waveform_image(x_in, self.audio.sample_rate, f"{name} — original waveform", p0, self.cfg.dpi)
        plot_waveform_image(x_out, self.audio.sample_rate, f"{name} — after", p1, self.cfg.dpi)
        return p0, p1

    def _save_pair_heatmap(self, name: str, x_in: torch.Tensor, x_out: torch.Tensor, y_label: str) -> T.Tuple[str, str]:
        outdir = os.path.join(self.cfg.outdir, name)
        ensure_dir(outdir)
        p0 = os.path.join(outdir, "original.png")
        p1 = os.path.join(outdir, f"{name}.png")
        plot_heatmap_image(x_in, f"{name} — original", p0, self.cfg.dpi, x_label="Frame", y_label=y_label)
        plot_heatmap_image(x_out, f"{name} — after", p1, self.cfg.dpi, x_label="Frame", y_label=y_label)
        return p0, p1

    def _save_pair_line(self, name: str, x_in: torch.Tensor, x_out: torch.Tensor, x_label: str) -> T.Tuple[str, str]:
        outdir = os.path.join(self.cfg.outdir, name)
        ensure_dir(outdir)
        p0 = os.path.join(outdir, "original.png")
        p1 = os.path.join(outdir, f"{name}.png")
        plot_line_image(x_in, f"{name} — original", p0, self.cfg.dpi, x_label=x_label)
        plot_line_image(x_out, f"{name} — after", p1, self.cfg.dpi, x_label=x_label)
        return p0, p1

    def _save_placeholder_pair(self, name: str, in_msg: str, out_msg: str) -> T.Tuple[str, str]:
        outdir = os.path.join(self.cfg.outdir, name)
        ensure_dir(outdir)
        p0 = os.path.join(outdir, "original.png")
        p1 = os.path.join(outdir, f"{name}.png")
        placeholder_image(in_msg, f"{name} — original (placeholder)", p0, self.cfg.dpi)
        placeholder_image(out_msg, f"{name} — after (placeholder)", p1, self.cfg.dpi)
        return p0, p1

    # ---------- Individual transform demos (robust) ----------

    def run_AddNoise(self) -> TransformResult:
        name = "AddNoise"
        t0 = time.perf_counter()
        try:
            x = self.audio.waveform
            noise = self.rng.randn(*x.shape).to(x.device)
            snr = torch.tensor(self.cfg.snr_db, device=x.device, dtype=x.dtype)
            try:
                y = AddNoise()(x, noise, snr)
            except Exception:
                y = AddNoise()(x, noise, snr.unsqueeze(0))
            p0, p1 = self._save_pair_waveform(name, x, y)
            return TransformResult(name=name, ok=True, in_shape=tuple(x.shape), out_shape=tuple(y.shape),
                                   in_dtype=x.dtype, out_dtype=y.dtype, plot_type="waveform",
                                   original_path=p0, transformed_path=p1,
                                   elapsed_ms=1000 * (time.perf_counter() - t0), note=f"SNR={self.cfg.snr_db} dB")
        except Exception as e:
            p0, p1 = self._save_placeholder_pair(name, "AddNoise input placeholder", f"Exception: {e}")
            return TransformResult(name=name, ok=False, error=str(e), original_path=p0, transformed_path=p1)

    def run_AmplitudeToDB(self) -> TransformResult:
        name = "AmplitudeToDB"
        t0 = time.perf_counter()
        try:
            spec_amp = self._spec(self.audio.waveform, power=1.0)
            atodb = AmplitudeToDB(stype="magnitude", top_db=self.cfg.top_db)
            spec_db = atodb(spec_amp)
            p0, p1 = self._save_pair_heatmap(name, spec_amp[0], spec_db[0], "Freq Bin")
            return TransformResult(name=name, ok=True, in_shape=tuple(spec_amp.shape), out_shape=tuple(spec_db.shape),
                                   in_dtype=spec_amp.dtype, out_dtype=spec_db.dtype, plot_type="heatmap",
                                   original_path=p0, transformed_path=p1, elapsed_ms=1000*(time.perf_counter()-t0),
                                   note=f"top_db={self.cfg.top_db}")
        except Exception as e:
            p0, p1 = self._save_placeholder_pair(name, "Amplitude spectrogram", f"Exception: {e}")
            return TransformResult(name=name, ok=False, error=str(e), original_path=p0, transformed_path=p1)

    def run_ComputeDeltas(self) -> TransformResult:
        name = "ComputeDeltas"
        t0 = time.perf_counter()
        try:
            feats = self._mfcc(self.audio.waveform)
            deltas = ComputeDeltas(win_length=5)(feats)
            p0, p1 = self._save_pair_heatmap(name, feats[0], deltas[0], "MFCC Coeff")
            return TransformResult(name=name, ok=True, in_shape=tuple(feats.shape), out_shape=tuple(deltas.shape),
                                   in_dtype=feats.dtype, out_dtype=deltas.dtype, plot_type="heatmap",
                                   original_path=p0, transformed_path=p1, elapsed_ms=1000*(time.perf_counter()-t0))
        except Exception as e:
            p0, p1 = self._save_placeholder_pair(name, "MFCC features", f"Exception: {e}")
            return TransformResult(name=name, ok=False, error=str(e), original_path=p0, transformed_path=p1)

    def _design_lowpass_ir(self, taps: int = 63, cutoff_frac: float = 0.1) -> torch.Tensor:
        n = torch.arange(taps, dtype=torch.float32) - (taps - 1) / 2
        h = torch.where(n == 0, torch.tensor(2 * math.pi * cutoff_frac), torch.sin(2 * math.pi * cutoff_frac * n) / n)
        window = torch.hamming_window(taps, periodic=False)
        h = h * window
        h = h / h.sum()
        return h.to(self.audio.waveform.device).unsqueeze(0)

    def run_Convolve(self) -> TransformResult:
        name = "Convolve"
        t0 = time.perf_counter()
        try:
            x = self.audio.waveform
            ir = self._design_lowpass_ir(63, 0.12).expand(x.shape[0], -1)
            y = Convolve()(x, ir)
            p0, p1 = self._save_pair_waveform(name, x, y)
            return TransformResult(name=name, ok=True, in_shape=tuple(x.shape), out_shape=tuple(y.shape),
                                   in_dtype=x.dtype, out_dtype=y.dtype, plot_type="waveform",
                                   original_path=p0, transformed_path=p1, elapsed_ms=1000*(time.perf_counter()-t0),
                                   note="Low-pass FIR via direct convolution")
        except Exception as e:
            p0, p1 = self._save_placeholder_pair(name, "Waveform", f"Exception: {e}")
            return TransformResult(name=name, ok=False, error=str(e), original_path=p0, transformed_path=p1)

    def run_FFTConvolve(self) -> TransformResult:
        name = "FFTConvolve"
        t0 = time.perf_counter()
        try:
            x = self.audio.waveform
            ir = self._design_lowpass_ir(127, 0.08).expand(x.shape[0], -1)
            y = FFTConvolve()(x, ir)
            p0, p1 = self._save_pair_waveform(name, x, y)
            return TransformResult(name=name, ok=True, in_shape=tuple(x.shape), out_shape=tuple(y.shape),
                                   in_dtype=x.dtype, out_dtype=y.dtype, plot_type="waveform",
                                   original_path=p0, transformed_path=p1, elapsed_ms=1000*(time.perf_counter()-t0),
                                   note="Low-pass FIR via FFT-based convolution")
        except Exception as e:
            p0, p1 = self._save_placeholder_pair(name, "Waveform", f"Exception: {e}")
            return TransformResult(name=name, ok=False, error=str(e), original_path=p0, transformed_path=p1)

    def run_Deemphasis(self) -> TransformResult:
        name = "Deemphasis"
        t0 = time.perf_counter()
        try:
            x = self.audio.waveform
            pre = Preemphasis()(x)
            y = Deemphasis()(pre)
            p0, p1 = self._save_pair_waveform(name, pre, y)
            return TransformResult(name=name, ok=True, in_shape=tuple(pre.shape), out_shape=tuple(y.shape),
                                   in_dtype=pre.dtype, out_dtype=y.dtype, plot_type="waveform",
                                   original_path=p0, transformed_path=p1, elapsed_ms=1000*(time.perf_counter()-t0),
                                   note="Original=preemphasized; After=deemphasized (restored)")
        except Exception as e:
            p0, p1 = self._save_placeholder_pair(name, "Preemphasized waveform", f"Exception: {e}")
            return TransformResult(name=name, ok=False, error=str(e), original_path=p0, transformed_path=p1)

    def run_Fade(self) -> TransformResult:
        name = "Fade"
        t0 = time.perf_counter()
        try:
            x = self.audio.waveform
            fin = secs_to_samples(self.cfg.fade_in_seconds, self.audio.sample_rate)
            fout = secs_to_samples(self.cfg.fade_out_seconds, self.audio.sample_rate)
            y = Fade(fade_in_len=fin, fade_out_len=fout, fade_shape="linear")(x)
            p0, p1 = self._save_pair_waveform(name, x, y)
            return TransformResult(name=name, ok=True, in_shape=tuple(x.shape), out_shape=tuple(y.shape),
                                   in_dtype=x.dtype, out_dtype=y.dtype, plot_type="waveform",
                                   original_path=p0, transformed_path=p1, elapsed_ms=1000*(time.perf_counter()-t0),
                                   note=f"fade_in={self.cfg.fade_in_seconds}s, fade_out={self.cfg.fade_out_seconds}s")
        except Exception as e:
            p0, p1 = self._save_placeholder_pair(name, "Waveform", f"Exception: {e}")
            return TransformResult(name=name, ok=False, error=str(e), original_path=p0, transformed_path=p1)

    def run_FrequencyMasking(self) -> TransformResult:
        name = "FrequencyMasking"
        t0 = time.perf_counter()
        try:
            spec = self._spec(self.audio.waveform, power=1.0)
            fm = FrequencyMasking(freq_mask_param=self.cfg.frequency_mask_param)
            after = fm(spec)
            p0, p1 = self._save_pair_heatmap(name, spec[0], after[0], "Freq Bin")
            return TransformResult(name=name, ok=True, in_shape=tuple(spec.shape), out_shape=tuple(after.shape),
                                   in_dtype=spec.dtype, out_dtype=after.dtype, plot_type="heatmap",
                                   original_path=p0, transformed_path=p1, elapsed_ms=1000*(time.perf_counter()-t0),
                                   note=f"freq_mask_param={self.cfg.frequency_mask_param}")
        except Exception as e:
            p0, p1 = self._save_placeholder_pair(name, "Spectrogram", f"Exception: {e}")
            return TransformResult(name=name, ok=False, error=str(e), original_path=p0, transformed_path=p1)

    def run_GriffinLim(self) -> TransformResult:
        name = "GriffinLim"
        t0 = time.perf_counter()
        try:
            mag_spec = self._spec(self.audio.waveform, power=1.0)
            gl = GriffinLim(n_fft=self.cfg.n_fft, hop_length=self.cfg.hop_length, win_length=self.cfg.win_length)
            recon = gl(mag_spec)
            p0, p1 = self._save_pair_waveform(name, self.audio.waveform, recon)
            return TransformResult(name=name, ok=True, in_shape=tuple(mag_spec.shape), out_shape=tuple(recon.shape),
                                   in_dtype=mag_spec.dtype, out_dtype=recon.dtype, plot_type="waveform",
                                   original_path=p0, transformed_path=p1, elapsed_ms=1000*(time.perf_counter()-t0),
                                   note="Reconstructed waveform from magnitude spectrogram")
        except Exception as e:
            p0, p1 = self._save_placeholder_pair(name, "Magnitude spectrogram", f"Exception: {e}")
            return TransformResult(name=name, ok=False, error=str(e), original_path=p0, transformed_path=p1)

    def run_InverseMelScale(self) -> TransformResult:
        name = "InverseMelScale"
        t0 = time.perf_counter()
        try:
            lin_spec = self._spec(self.audio.waveform, power=1.0)
            mel = MelScale(n_mels=self.cfg.num_mels, sample_rate=self.audio.sample_rate, n_stft=self.cfg.n_fft // 2 + 1)(lin_spec)
            # Try transform; if it fails (rank-deficient), fallback to pseudoinverse.
            try:
                inv = InverseMelScale(n_stft=self.cfg.n_fft // 2 + 1, n_mels=self.cfg.num_mels, sample_rate=self.audio.sample_rate)(mel)
                note = "InverseMelScale (transform)"
            except Exception as ex_inv:
                inv = inverse_mel_scale_fallback(mel, n_stft=self.cfg.n_fft // 2 + 1, n_mels=self.cfg.num_mels, sample_rate=self.audio.sample_rate)
                note = f"Fallback via Mel pseudoinverse due to: {ex_inv}"
            p0, p1 = self._save_pair_heatmap(name, mel[0], inv[0], "Freq/Mel Bin")
            return TransformResult(name=name, ok=True, in_shape=tuple(mel.shape), out_shape=tuple(inv.shape),
                                   in_dtype=mel.dtype, out_dtype=inv.dtype, plot_type="heatmap",
                                   original_path=p0, transformed_path=p1, elapsed_ms=1000*(time.perf_counter()-t0), note=note)
        except Exception as e:
            p0, p1 = self._save_placeholder_pair(name, "Mel spectrogram", f"Exception: {e}")
            return TransformResult(name=name, ok=False, error=str(e), original_path=p0, transformed_path=p1)

    def run_InverseSpectrogram(self) -> TransformResult:
        name = "InverseSpectrogram"
        t0 = time.perf_counter()
        try:
            spec_c = self._spec(self.audio.waveform, return_complex=True)
            inv = InverseSpectrogram(n_fft=self.cfg.n_fft, hop_length=self.cfg.hop_length, win_length=self.cfg.win_length)
            recon = inv(spec_c)
            p0, p1 = self._save_pair_waveform(name, self.audio.waveform, recon)
            return TransformResult(name=name, ok=True, in_shape=tuple(spec_c.shape), out_shape=tuple(recon.shape),
                                   in_dtype=spec_c.dtype, out_dtype=recon.dtype, plot_type="waveform",
                                   original_path=p0, transformed_path=p1, elapsed_ms=1000*(time.perf_counter()-t0),
                                   note="Inverse STFT from complex spectrogram")
        except Exception as e:
            p0, p1 = self._save_placeholder_pair(name, "Complex spectrogram", f"Exception: {e}")
            return TransformResult(name=name, ok=False, error=str(e), original_path=p0, transformed_path=p1)

    def run_LFCC(self) -> TransformResult:
        name = "LFCC"
        t0 = time.perf_counter()
        try:
            lfcc = self._lfcc(self.audio.waveform)
            p0, p1 = self._save_pair_heatmap(name, lfcc[0], lfcc[0], "LFCC")
            return TransformResult(name=name, ok=True, in_shape=tuple(lfcc.shape), out_shape=tuple(lfcc.shape),
                                   in_dtype=lfcc.dtype, out_dtype=lfcc.dtype, plot_type="heatmap",
                                   original_path=p0, transformed_path=p1, elapsed_ms=1000*(time.perf_counter()-t0))
        except Exception as e:
            p0, p1 = self._save_placeholder_pair(name, "LFCC", f"Exception: {e}")
            return TransformResult(name=name, ok=False, error=str(e), original_path=p0, transformed_path=p1)

    def run_Loudness(self) -> TransformResult:
        name = "Loudness"
        t0 = time.perf_counter()
        try:
            tr = compat_Loudness(self.audio.sample_rate, self.cfg.n_fft, self.cfg.hop_length)
            if tr is not None:
                loud = tr(self.audio.waveform)
                # Original comparator: RMS per frame
                spec = self._spec(self.audio.waveform, power=2.0)
                rms = spec.mean(dim=1).sqrt().squeeze(0)
                p0, p1 = self._save_pair_line(name, rms, loud, "Frame")
                note = "Loudness transform"
            else:
                # Fallback: RMS proxy
                spec = self._spec(self.audio.waveform, power=2.0)
                rms = spec.mean(dim=1).sqrt().squeeze(0)
                loud = rms
                p0, p1 = self._save_pair_line(name, rms, loud, "Frame")
                note = "Fallback to RMS proxy (Loudness unavailable)"
            return TransformResult(name=name, ok=True, in_shape=tuple(self.audio.waveform.shape), out_shape=tuple(loud.shape),
                                   in_dtype=self.audio.waveform.dtype, out_dtype=loud.dtype, plot_type="line",
                                   original_path=p0, transformed_path=p1, elapsed_ms=1000*(time.perf_counter()-t0),
                                   note=note)
        except Exception as e:
            p0, p1 = self._save_placeholder_pair(name, "RMS approx", f"Exception: {e}")
            return TransformResult(name=name, ok=False, error=str(e), original_path=p0, transformed_path=p1)

    def run_MFCC(self) -> TransformResult:
        name = "MFCC"
        t0 = time.perf_counter()
        try:
            m = self._mfcc(self.audio.waveform)
            p0, p1 = self._save_pair_heatmap(name, m[0], m[0], "MFCC")
            return TransformResult(name=name, ok=True, in_shape=tuple(m.shape), out_shape=tuple(m.shape),
                                   in_dtype=m.dtype, out_dtype=m.dtype, plot_type="heatmap",
                                   original_path=p0, transformed_path=p1, elapsed_ms=1000*(time.perf_counter()-t0))
        except Exception as e:
            p0, p1 = self._save_placeholder_pair(name, "MFCC", f"Exception: {e}")
            return TransformResult(name=name, ok=False, error=str(e), original_path=p0, transformed_path=p1)

    def run_MVDR(self) -> TransformResult:
        name = "MVDR"
        t0 = time.perf_counter()
        try:
            if not self.cfg.enable_beamforming:
                raise RuntimeError("Beamforming disabled by config")
            spec_c, mask_s, mask_n, _ = self._beamforming_inputs()
            y = compat_MVDR_call(spec_c, mask_s, mask_n)
            mag_in = spec_c.abs().mean(dim=0)
            mag_out = (y.abs() if torch.is_complex(y) else y).mean(dim=0) if y.ndim == 3 else (y.abs() if torch.is_complex(y) else y)
            p0, p1 = self._save_pair_heatmap(name, mag_in, mag_out, "Freq Bin")
            return TransformResult(name=name, ok=True, in_shape=tuple(spec_c.shape), out_shape=tuple(y.shape),
                                   in_dtype=spec_c.dtype, out_dtype=y.dtype, plot_type="heatmap",
                                   original_path=p0, transformed_path=p1, elapsed_ms=1000*(time.perf_counter()-t0))
        except Exception as e:
            p0, p1 = self._save_placeholder_pair(name, "Beamforming input", f"Exception: {e}")
            return TransformResult(name=name, ok=False, error=str(e), original_path=p0, transformed_path=p1)

    def run_MelScale(self) -> TransformResult:
        name = "MelScale"
        t0 = time.perf_counter()
        try:
            spec = self._spec(self.audio.waveform, power=1.0)
            mel = MelScale(n_mels=self.cfg.num_mels, sample_rate=self.audio.sample_rate, n_stft=self.cfg.n_fft // 2 + 1)(spec)
            p0, p1 = self._save_pair_heatmap(name, spec[0], mel[0], "Freq/Mel")
            return TransformResult(name=name, ok=True, in_shape=tuple(spec.shape), out_shape=tuple(mel.shape),
                                   in_dtype=spec.dtype, out_dtype=mel.dtype, plot_type="heatmap",
                                   original_path=p0, transformed_path=p1, elapsed_ms=1000*(time.perf_counter()-t0))
        except Exception as e:
            p0, p1 = self._save_placeholder_pair(name, "Spectrogram", f"Exception: {e}")
            return TransformResult(name=name, ok=False, error=str(e), original_path=p0, transformed_path=p1)

    def run_MelSpectrogram(self) -> TransformResult:
        name = "MelSpectrogram"
        t0 = time.perf_counter()
        try:
            mel = self._melspec(self.audio.waveform, power=2.0)
            p0, p1 = self._save_pair_heatmap(name, mel[0], mel[0], "Mel Bin")
            return TransformResult(name=name, ok=True, in_shape=tuple(mel.shape), out_shape=tuple(mel.shape),
                                   in_dtype=mel.dtype, out_dtype=mel.dtype, plot_type="heatmap",
                                   original_path=p0, transformed_path=p1, elapsed_ms=1000*(time.perf_counter()-t0))
        except Exception as e:
            p0, p1 = self._save_placeholder_pair(name, "MelSpectrogram", f"Exception: {e}")
            return TransformResult(name=name, ok=False, error=str(e), original_path=p0, transformed_path=p1)

    def run_MuLawPair(self) -> T.List[TransformResult]:
        results: T.List[TransformResult] = []
        # Encoding
        name_enc = "MuLawEncoding"
        t0 = time.perf_counter()
        try:
            x = self.audio.waveform
            enc = MuLawEncoding(quantization_channels=256)(x)
            outdir = os.path.join(self.cfg.outdir, name_enc)
            ensure_dir(outdir)
            p0 = os.path.join(outdir, "original.png")
            p1 = os.path.join(outdir, f"{name_enc}.png")
            plot_waveform_image(x, self.audio.sample_rate, f"{name_enc} — original", p0, self.cfg.dpi)
            plot_line_image(enc.float(), f"{name_enc} — encoded indices", p1, self.cfg.dpi, x_label="Sample idx")
            results.append(TransformResult(name=name_enc, ok=True, in_shape=tuple(x.shape), out_shape=tuple(enc.shape),
                                           in_dtype=x.dtype, out_dtype=enc.dtype, plot_type="line",
                                           original_path=p0, transformed_path=p1, elapsed_ms=1000*(time.perf_counter()-t0)))
        except Exception as e:
            p0, p1 = self._save_placeholder_pair(name_enc, "Waveform", f"Exception: {e}")
            results.append(TransformResult(name=name_enc, ok=False, error=str(e), original_path=p0, transformed_path=p1))
        # Decoding
        name_dec = "MuLawDecoding"
        t1 = time.perf_counter()
        try:
            x = self.audio.waveform
            enc = MuLawEncoding(quantization_channels=256)(x)
            dec = MuLawDecoding(quantization_channels=256)(enc)
            p0, p1 = self._save_pair_waveform(name_dec, x, dec)
            results.append(TransformResult(name=name_dec, ok=True, in_shape=tuple(enc.shape), out_shape=tuple(dec.shape),
                                           in_dtype=enc.dtype, out_dtype=dec.dtype, plot_type="waveform",
                                           original_path=p0, transformed_path=p1, elapsed_ms=1000*(time.perf_counter()-t1)))
        except Exception as e:
            p0, p1 = self._save_placeholder_pair(name_dec, "MuLaw indices", f"Exception: {e}")
            results.append(TransformResult(name=name_dec, ok=False, error=str(e), original_path=p0, transformed_path=p1))
        return results

    def run_PSD(self) -> TransformResult:
        name = "PSD"
        t0 = time.perf_counter()
        try:
            spec_c, mask_s, _, _ = self._beamforming_inputs()
            psd = PSD()(spec_c, mask_s)
            arr = psd
            if arr.ndim == 3:
                if arr.shape[0] == self.cfg.n_fft // 2 + 1:
                    mat = arr.reshape(arr.shape[0], -1)
                    y_label = "Freq Bin"
                elif arr.shape[-1] == self.cfg.n_fft // 2 + 1:
                    mat = arr.permute(2, 0, 1).reshape(arr.shape[-1], -1)
                    y_label = "Freq Bin"
                else:
                    mat = arr.reshape(arr.shape[0], -1); y_label = "Dim"
            else:
                mat = arr; y_label = "Dim"
            p0, p1 = self._save_pair_heatmap(name, mat.abs().T, mat.abs().T, y_label)
            return TransformResult(name=name, ok=True, in_shape=tuple(spec_c.shape), out_shape=tuple(psd.shape),
                                   in_dtype=spec_c.dtype, out_dtype=psd.dtype, plot_type="heatmap",
                                   original_path=p0, transformed_path=p1, elapsed_ms=1000*(time.perf_counter()-t0),
                                   note="PSD visualized as unfolded covariance vs. frequency")
        except Exception as e:
            p0, p1 = self._save_placeholder_pair(name, "Complex spec", f"Exception: {e}")
            return TransformResult(name=name, ok=False, error=str(e), original_path=p0, transformed_path=p1)

    def run_PitchShift(self) -> TransformResult:
        name = "PitchShift"
        t0 = time.perf_counter()
        try:
            x = self.audio.waveform
            y = compat_PitchShift(self.audio.sample_rate, self.cfg.pitch_semitones, x)
            p0, p1 = self._save_pair_waveform(name, x, y)
            return TransformResult(name=name, ok=True, in_shape=tuple(x.shape), out_shape=tuple(y.shape),
                                   in_dtype=x.dtype, out_dtype=y.dtype, plot_type="waveform",
                                   original_path=p0, transformed_path=p1, elapsed_ms=1000*(time.perf_counter()-t0),
                                   note=f"n_steps={self.cfg.pitch_semitones}")
        except Exception as e:
            p0, p1 = self._save_placeholder_pair(name, "Waveform", f"Exception: {e}")
            return TransformResult(name=name, ok=False, error=str(e), original_path=p0, transformed_path=p1)

    def run_Preemphasis(self) -> TransformResult:
        name = "Preemphasis"
        t0 = time.perf_counter()
        try:
            x = self.audio.waveform
            y = Preemphasis()(x)
            p0, p1 = self._save_pair_waveform(name, x, y)
            return TransformResult(name=name, ok=True, in_shape=tuple(x.shape), out_shape=tuple(y.shape),
                                   in_dtype=x.dtype, out_dtype=y.dtype, plot_type="waveform",
                                   original_path=p0, transformed_path=p1, elapsed_ms=1000*(time.perf_counter()-t0))
        except Exception as e:
            p0, p1 = self._save_placeholder_pair(name, "Waveform", f"Exception: {e}")
            return TransformResult(name=name, ok=False, error=str(e), original_path=p0, transformed_path=p1)

    def run_RTFMVDR(self) -> TransformResult:
        name = "RTFMVDR"
        t0 = time.perf_counter()
        try:
            if not self.cfg.enable_beamforming:
                raise RuntimeError("Beamforming disabled by config")
            spec_c, mask_s, mask_n, ref_ch = self._beamforming_inputs()
            psd_n = PSD()(spec_c, mask_n)
            C = spec_c.shape[0]
            rtf = torch.zeros((self.cfg.n_fft // 2 + 1, C), dtype=torch.complex64, device=spec_c.device)
            rtf[:, ref_ch] = 1 + 0j
            y = RTFMVDR()(spec_c, rtf, psd_n, reference_channel=ref_ch)
            mag_in = spec_c.abs().mean(dim=0)
            mag_out = (y.abs() if torch.is_complex(y) else y).mean(dim=0) if y.ndim == 3 else (y.abs() if torch.is_complex(y) else y)
            p0, p1 = self._save_pair_heatmap(name, mag_in, mag_out, "Freq Bin")
            return TransformResult(name=name, ok=True, in_shape=tuple(spec_c.shape), out_shape=tuple(y.shape),
                                   in_dtype=spec_c.dtype, out_dtype=y.dtype, plot_type="heatmap",
                                   original_path=p0, transformed_path=p1, elapsed_ms=1000*(time.perf_counter()-t0),
                                   note=f"Ref ch={ref_ch}, naive RTF")
        except Exception as e:
            p0, p1 = self._save_placeholder_pair(name, "Beamforming input", f"Exception: {e}")
            return TransformResult(name=name, ok=False, error=str(e), original_path=p0, transformed_path=p1)

    def run_Resample(self) -> TransformResult:
        name = "Resample"
        t0 = time.perf_counter()
        try:
            x = self.audio.waveform
            target_sr = 16000 if self.audio.sample_rate != 16000 else 22050
            resampler = Resample(orig_freq=self.audio.sample_rate, new_freq=target_sr)
            y = resampler(x)
            outdir = os.path.join(self.cfg.outdir, name)
            ensure_dir(outdir)
            p0 = os.path.join(outdir, "original.png")
            p1 = os.path.join(outdir, f"{name}.png")
            plot_waveform_image(x, self.audio.sample_rate, f"{name} — original ({self.audio.sample_rate} Hz)", p0, self.cfg.dpi)
            plot_waveform_image(y, target_sr, f"{name} — resampled ({target_sr} Hz)", p1, self.cfg.dpi)
            return TransformResult(name=name, ok=True, in_shape=tuple(x.shape), out_shape=tuple(y.shape),
                                   in_dtype=x.dtype, out_dtype=y.dtype, plot_type="waveform",
                                   original_path=p0, transformed_path=p1, elapsed_ms=1000*(time.perf_counter()-t0),
                                   note=f"{self.audio.sample_rate} -> {target_sr}")
        except Exception as e:
            p0, p1 = self._save_placeholder_pair(name, "Waveform", f"Exception: {e}")
            return TransformResult(name=name, ok=False, error=str(e), original_path=p0, transformed_path=p1)

    def run_SlidingWindowCmn(self) -> TransformResult:
        name = "SlidingWindowCmn"
        t0 = time.perf_counter()
        try:
            feats = self._mfcc(self.audio.waveform)
            norm = SlidingWindowCmn(cmn_window=600, min_cmn_window=100, center=True)(feats)
            p0, p1 = self._save_pair_heatmap(name, feats[0], norm[0], "MFCC")
            return TransformResult(name=name, ok=True, in_shape=tuple(feats.shape), out_shape=tuple(norm.shape),
                                   in_dtype=feats.dtype, out_dtype=norm.dtype, plot_type="heatmap",
                                   original_path=p0, transformed_path=p1, elapsed_ms=1000*(time.perf_counter()-t0))
        except Exception as e:
            p0, p1 = self._save_placeholder_pair(name, "MFCC", f"Exception: {e}")
            return TransformResult(name=name, ok=False, error=str(e), original_path=p0, transformed_path=p1)

    def run_SoudenMVDR(self) -> TransformResult:
        name = "SoudenMVDR"
        t0 = time.perf_counter()
        try:
            if not self.cfg.enable_beamforming:
                raise RuntimeError("Beamforming disabled by config")
            spec_c, mask_s, mask_n, ref_ch = self._beamforming_inputs()
            psd_s = PSD()(spec_c, mask_s)
            psd_n = PSD()(spec_c, mask_n)
            y = compat_SoudenMVDR_call(spec_c, psd_s, psd_n, ref_ch)
            mag_in = spec_c.abs().mean(dim=0)
            mag_out = (y.abs() if torch.is_complex(y) else y).mean(dim=0) if y.ndim == 3 else (y.abs() if torch.is_complex(y) else y)
            p0, p1 = self._save_pair_heatmap(name, mag_in, mag_out, "Freq Bin")
            return TransformResult(name=name, ok=True, in_shape=tuple(spec_c.shape), out_shape=tuple(y.shape),
                                   in_dtype=spec_c.dtype, out_dtype=y.dtype, plot_type="heatmap",
                                   original_path=p0, transformed_path=p1, elapsed_ms=1000*(time.perf_counter()-t0),
                                   note=f"Ref ch={ref_ch}")
        except Exception as e:
            p0, p1 = self._save_placeholder_pair(name, "Beamforming input", f"Exception: {e}")
            return TransformResult(name=name, ok=False, error=str(e), original_path=p0, transformed_path=p1)

    def run_SpecAugment(self) -> TransformResult:
        name = "SpecAugment"
        t0 = time.perf_counter()
        try:
            spec = self._spec(self.audio.waveform, power=1.0)
            tr = compat_SpecAugment(self.cfg.frequency_mask_param, self.cfg.time_mask_param)
            if tr is not None:
                aug = tr(spec)
                note = "SpecAugment"
            else:
                # Fallback: TimeMasking + FrequencyMasking composition
                tm = TimeMasking(time_mask_param=self.cfg.time_mask_param)
                fm = FrequencyMasking(freq_mask_param=self.cfg.frequency_mask_param)
                aug = fm(tm(spec))
                note = "Fallback: TimeMasking + FrequencyMasking"
            p0, p1 = self._save_pair_heatmap(name, spec[0], aug[0], "Freq Bin")
            return TransformResult(name=name, ok=True, in_shape=tuple(spec.shape), out_shape=tuple(aug.shape),
                                   in_dtype=spec.dtype, out_dtype=aug.dtype, plot_type="heatmap",
                                   original_path=p0, transformed_path=p1, elapsed_ms=1000*(time.perf_counter()-t0),
                                   note=note)
        except Exception as e:
            p0, p1 = self._save_placeholder_pair(name, "Spectrogram", f"Exception: {e}")
            return TransformResult(name=name, ok=False, error=str(e), original_path=p0, transformed_path=p1)

    def run_SpectralCentroid(self) -> TransformResult:
        name = "SpectralCentroid"
        t0 = time.perf_counter()
        try:
            sc = SpectralCentroid(sample_rate=self.audio.sample_rate, n_fft=self.cfg.n_fft, hop_length=self.cfg.hop_length)(self.audio.waveform)
            spec = self._spec(self.audio.waveform, power=1.0)
            maxbin = spec.argmax(dim=1).float().squeeze(0)
            p0, p1 = self._save_pair_line(name, maxbin, sc, "Frame")
            return TransformResult(name=name, ok=True, in_shape=tuple(self.audio.waveform.shape), out_shape=tuple(sc.shape),
                                   in_dtype=self.audio.waveform.dtype, out_dtype=sc.dtype, plot_type="line",
                                   original_path=p0, transformed_path=p1, elapsed_ms=1000*(time.perf_counter()-t0))
        except Exception as e:
            p0, p1 = self._save_placeholder_pair(name, "Spec peak proxy", f"Exception: {e}")
            return TransformResult(name=name, ok=False, error=str(e), original_path=p0, transformed_path=p1)

    def run_Spectrogram(self) -> TransformResult:
        name = "Spectrogram"
        t0 = time.perf_counter()
        try:
            spec = self._spec(self.audio.waveform, power=2.0)
            p0, p1 = self._save_pair_heatmap(name, spec[0], spec[0], "Freq Bin")
            return TransformResult(name=name, ok=True, in_shape=tuple(self.audio.waveform.shape), out_shape=tuple(spec.shape),
                                   in_dtype=self.audio.waveform.dtype, out_dtype=spec.dtype, plot_type="heatmap",
                                   original_path=p0, transformed_path=p1, elapsed_ms=1000*(time.perf_counter()-t0))
        except Exception as e:
            p0, p1 = self._save_placeholder_pair(name, "Waveform", f"Exception: {e}")
            return TransformResult(name=name, ok=False, error=str(e), original_path=p0, transformed_path=p1)

    def run_Speed(self) -> TransformResult:
        name = "Speed"
        t0 = time.perf_counter()
        try:
            x = self.audio.waveform
            tr = compat_Speed(self.audio.sample_rate, self.cfg.speed_factor)
            # Forward variants
            try:
                y, _ = tr(x, None)
            except Exception:
                try:
                    y, _ = tr(x)
                except Exception:
                    y = tr(x)
            p0, p1 = self._save_pair_waveform(name, x, y)
            return TransformResult(name=name, ok=True, in_shape=tuple(x.shape), out_shape=tuple(y.shape),
                                   in_dtype=x.dtype, out_dtype=y.dtype, plot_type="waveform",
                                   original_path=p0, transformed_path=p1, elapsed_ms=1000*(time.perf_counter()-t0),
                                   note=f"factor={self.cfg.speed_factor}")
        except Exception as e:
            p0, p1 = self._save_placeholder_pair(name, "Waveform", f"Exception: {e}\nNote: Speed API varies.")
            return TransformResult(name=name, ok=False, error=str(e), original_path=p0, transformed_path=p1)

    def run_SpeedPerturbation(self) -> TransformResult:
        name = "SpeedPerturbation"
        t0 = time.perf_counter()
        try:
            x = self.audio.waveform
            try:
                tr = SpeedPerturbation(sample_rate=self.audio.sample_rate, factors=[0.9, 1.0, 1.1])
            except Exception:
                tr = SpeedPerturbation(self.audio.sample_rate, [0.9, 1.0, 1.1])
            try:
                y, _ = tr(x, None)
            except Exception:
                y = tr(x)
            p0, p1 = self._save_pair_waveform(name, x, y)
            return TransformResult(name=name, ok=True, in_shape=tuple(x.shape), out_shape=tuple(y.shape),
                                   in_dtype=x.dtype, out_dtype=y.dtype, plot_type="waveform",
                                   original_path=p0, transformed_path=p1, elapsed_ms=1000*(time.perf_counter()-t0))
        except Exception as e:
            p0, p1 = self._save_placeholder_pair(name, "Waveform", f"Exception: {e}\nNote: SpeedPerturbation API varies.")
            return TransformResult(name=name, ok=False, error=str(e), original_path=p0, transformed_path=p1)

    def run_TimeMasking(self) -> TransformResult:
        name = "TimeMasking"
        t0 = time.perf_counter()
        try:
            spec = self._spec(self.audio.waveform, power=1.0)
            tm = TimeMasking(time_mask_param=self.cfg.time_mask_param)
            after = tm(spec)
            p0, p1 = self._save_pair_heatmap(name, spec[0], after[0], "Freq Bin")
            return TransformResult(name=name, ok=True, in_shape=tuple(spec.shape), out_shape=tuple(after.shape),
                                   in_dtype=spec.dtype, out_dtype=after.dtype, plot_type="heatmap",
                                   original_path=p0, transformed_path=p1, elapsed_ms=1000*(time.perf_counter()-t0),
                                   note=f"time_mask_param={self.cfg.time_mask_param}")
        except Exception as e:
            p0, p1 = self._save_placeholder_pair(name, "Spectrogram", f"Exception: {e}")
            return TransformResult(name=name, ok=False, error=str(e), original_path=p0, transformed_path=p1)

    def run_TimeStretch(self) -> TransformResult:
        name = "TimeStretch"
        t0 = time.perf_counter()
        try:
            spec_c = self._spec(self.audio.waveform, return_complex=True)
            try:
                ts = TimeStretch(hop_length=self.cfg.hop_length, n_freq=self.cfg.n_fft // 2 + 1, fixed_rate=None)
                y = ts(spec_c, overriding_rate=self.cfg.time_stretch_rate)
            except TypeError:
                # Some versions set rate in init
                ts = TimeStretch(hop_length=self.cfg.hop_length, n_freq=self.cfg.n_fft // 2 + 1, fixed_rate=self.cfg.time_stretch_rate)
                y = ts(spec_c)
            mag_in = spec_c.abs().mean(dim=0)
            mag_out = y.abs().mean(dim=0) if torch.is_complex(y) else y.mean(dim=0)
            p0, p1 = self._save_pair_heatmap(name, mag_in, mag_out, "Freq Bin")
            return TransformResult(name=name, ok=True, in_shape=tuple(spec_c.shape), out_shape=tuple(y.shape),
                                   in_dtype=spec_c.dtype, out_dtype=y.dtype, plot_type="heatmap",
                                   original_path=p0, transformed_path=p1, elapsed_ms=1000*(time.perf_counter()-t0),
                                   note=f"rate={self.cfg.time_stretch_rate}")
        except Exception as e:
            p0, p1 = self._save_placeholder_pair(name, "Complex spec", f"Exception: {e}")
            return TransformResult(name=name, ok=False, error=str(e), original_path=p0, transformed_path=p1)

    def run_Vad(self) -> TransformResult:
        name = "Vad"
        t0 = time.perf_counter()
        try:
            try:
                tr = Vad(sample_rate=self.audio.sample_rate)
            except Exception:
                tr = Vad()
            y = tr(self.audio.waveform)
            p0, p1 = self._save_pair_waveform(name, self.audio.waveform, y)
            return TransformResult(name=name, ok=True, in_shape=tuple(self.audio.waveform.shape), out_shape=tuple(y.shape),
                                   in_dtype=self.audio.waveform.dtype, out_dtype=y.dtype, plot_type="waveform",
                                   original_path=p0, transformed_path=p1, elapsed_ms=1000*(time.perf_counter()-t0),
                                   note="Voice activity detection")
        except Exception as e:
            p0, p1 = self._save_placeholder_pair(name, "Waveform", f"Exception: {e}\nNote: Vad API varies.")
            return TransformResult(name=name, ok=False, error=str(e), original_path=p0, transformed_path=p1)

    def run_Vol(self) -> TransformResult:
        name = "Vol"
        t0 = time.perf_counter()
        try:
            x = self.audio.waveform
            y = Vol(gain=self.cfg.vol_gain_db, gain_type="db")(x)
            p0, p1 = self._save_pair_waveform(name, x, y)
            return TransformResult(name=name, ok=True, in_shape=tuple(x.shape), out_shape=tuple(y.shape),
                                   in_dtype=x.dtype, out_dtype=y.dtype, plot_type="waveform",
                                   original_path=p0, transformed_path=p1, elapsed_ms=1000*(time.perf_counter()-t0),
                                   note=f"{self.cfg.vol_gain_db} dB")
        except Exception as e:
            p0, p1 = self._save_placeholder_pair(name, "Waveform", f"Exception: {e}")
            return TransformResult(name=name, ok=False, error=str(e), original_path=p0, transformed_path=p1)

    # ---------- Orchestrator ----------

    def run_all(self) -> T.List[TransformResult]:
        results: T.List[TransformResult] = []
        runners: T.List[T.Callable[[], TransformResult | T.List[TransformResult]]] = [
            self.run_AddNoise,
            self.run_AmplitudeToDB,
            self.run_ComputeDeltas,
            self.run_Convolve,
            self.run_FFTConvolve,
            self.run_Deemphasis,
            self.run_Fade,
            self.run_FrequencyMasking,
            self.run_GriffinLim,
            self.run_InverseMelScale,
            self.run_InverseSpectrogram,
            self.run_LFCC,
            self.run_Loudness,
            self.run_MFCC,
            self.run_MVDR,
            self.run_MelScale,
            self.run_MelSpectrogram,
            lambda: self.run_MuLawPair(),
            self.run_PSD,
            self.run_PitchShift,
            self.run_Preemphasis,
            self.run_RTFMVDR,
            self.run_Resample,
            self.run_SlidingWindowCmn,
            self.run_SoudenMVDR,
            self.run_SpecAugment,
            self.run_SpectralCentroid,
            self.run_Spectrogram,
            self.run_Speed,
            self.run_SpeedPerturbation,
            self.run_TimeMasking,
            self.run_TimeStretch,
            self.run_Vad,
            self.run_Vol,
        ]
        for fn in runners:
            try:
                res = fn()
                if isinstance(res, list):
                    results.extend(res)
                else:
                    results.append(res)
            except Exception as e:
                name = fn.__name__.replace("run_", "")
                p0, p1 = self._save_placeholder_pair(name, "N/A", f"Fatal exception: {e}")
                results.append(TransformResult(name=name, ok=False, error=str(e), original_path=p0, transformed_path=p1))
        return results


# =======================
# Reporting via Rich
# =======================

def report_overview(cfg: AppConfig, audio: AudioData):
    if not cfg.verbose:
        return
    table = Table(title="Run Configuration & Input Audio", show_header=True, header_style="bold magenta")
    table.add_column("Field", style="bold cyan")
    table.add_column("Value", overflow="fold")
    table.add_row("Source", audio.path_or_desc)
    table.add_row("Channels", str(audio.waveform.shape[0]))
    table.add_row("Samples", str(audio.waveform.shape[-1]))
    table.add_row("Sample Rate", f"{audio.sample_rate} Hz")
    table.add_row("Duration", f"{audio.waveform.shape[-1] / audio.sample_rate:.2f} s")
    table.add_row("Device", cfg.device)
    table.add_row("Torch", torch.__version__)
    table.add_row("Torchaudio", torchaudio.__version__ if hasattr(torchaudio, "__version__") else "unknown")
    table.add_row("Output Dir", os.path.abspath(cfg.outdir))
    table.add_row("Image Size", f"{cfg.image_size_px}x{cfg.image_size_px}")
    table.add_row("DPI", str(cfg.dpi))
    table.add_row("Chunk seconds", str(cfg.chunk_seconds))
    table.add_row("Chunk overlap", f"{cfg.chunk_overlap_seconds}s")
    console.print(table)

def report_results(results: T.List[TransformResult]):
    if not app_cfg.verbose:
        return
    table = Table(title="Transforms Summary", show_header=True, header_style="bold green")
    table.add_column("Transform", style="bold")
    table.add_column("OK", justify="center")
    table.add_column("In->Out Shape")
    table.add_column("DType")
    table.add_column("Plot")
    table.add_column("Orig Img", overflow="fold")
    table.add_column("After Img", overflow="fold")
    table.add_column("ms", justify="right")
    table.add_column("Note/Error", overflow="fold")
    for r in results:
        ok = "✅" if r.ok else "❌"
        shapes = f"{r.in_shape} → {r.out_shape}" if r.in_shape and r.out_shape else "-"
        dtypes = f"{r.in_dtype} → {r.out_dtype}" if r.in_dtype and r.out_dtype else "-"
        table.add_row(
            r.name, ok, shapes, dtypes, r.plot_type, str(r.original_path or "-"), str(r.transformed_path or "-"),
            f"{r.elapsed_ms:.1f}" if r.elapsed_ms else "-", r.note if r.ok else (r.error or "")
        )
    console.print(table)


# =======================
# Main entry point (CLI)
# =======================

def parse_args() -> AppConfig:
    p = argparse.ArgumentParser(description="Apply torchaudio transforms, plot original and transformed images per effect, save 1024x1024 PNGs.")
    p.add_argument("--source", type=str, default=None, help="Audio source: path | file:// | http(s):// | - (stdin bytes) | base64://<...>")
    p.add_argument("--outdir", type=str, default="transform_outputs", help="Output directory root.")
    p.add_argument("--verbose", type=lambda s: s.lower() in ("1","true","yes","y"), default=True, help="Verbose rich metadata logging.")
    p.add_argument("--resample-to", type=int, default=None, help="Optional resample target frequency.")
    p.add_argument("--max-seconds", type=float, default=600.0, help="Trim audio to at most this many seconds.")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"], help="Device for tensors.")
    p.add_argument("--image-size", type=int, default=100, help="Image size (square) in pixels.")
    p.add_argument("--dpi", type=int, default=100, help="DPI for Matplotlib canvas.")
    p.add_argument("--seed", type=int, default=42, help="RNG seed.")
    p.add_argument("--disable-beamforming", action="store_true", help="Skip MVDR/RTFMVDR/SoudenMVDR/PSD demos.")
    p.add_argument("--time-stretch", type=float, default=1.2, help="TimeStretch rate.")
    p.add_argument("--pitch-steps", type=float, default=3.0, help="PitchShift semitone steps (+ up, - down).")
    p.add_argument("--speed-factor", type=float, default=1.1, help="Speed transform factor.")
    p.add_argument("--fade-in", type=float, default=0.2, help="Fade-in seconds.")
    p.add_argument("--fade-out", type=float, default=0.2, help="Fade-out seconds.")
    p.add_argument("--snr", type=float, default=10.0, help="AddNoise SNR in dB.")
    p.add_argument("--vol-db", type=float, default=6.0, help="Vol gain in dB.")
    p.add_argument("--freq-mask", type=int, default=24, help="FrequencyMasking param.")
    p.add_argument("--time-mask", type=int, default=30, help="TimeMasking param.")
    # Chunking
    p.add_argument("--chunk-seconds", type=float, default=30.0, help="Chunk length in seconds for streaming-friendly transforms (None to disable).")
    p.add_argument("--chunk-overlap", type=float, default=0.5, help="Overlap between chunks in seconds.")
    # Beamforming budget
    p.add_argument("--bf-max-seconds", type=float, default=30.0, help="Limit seconds processed for beamforming demos.")
    args = p.parse_args()

    chunk_seconds = args.chunk_seconds
    if isinstance(chunk_seconds, float) and chunk_seconds <= 0:
        chunk_seconds = None

    return AppConfig(
        source=args.source,
        outdir=args.outdir,
        verbose=args.verbose,
        resample_to=args.resample_to,
        max_seconds=args.max_seconds,
        device=args.device,
        image_size_px=args.image_size,
        dpi=args.dpi,
        seed=args.seed,
        enable_beamforming=not args.disable_beamforming,
        time_stretch_rate=args.time_stretch,
        pitch_semitones=args.pitch_steps,
        speed_factor=args.speed_factor,
        fade_in_seconds=args.fade_in,
        fade_out_seconds=args.fade_out,
        snr_db=args.snr,
        vol_gain_db=args.vol_db,
        frequency_mask_param=args.freq_mask,
        time_mask_param=args.time_mask,
        chunk_seconds=chunk_seconds,
        chunk_overlap_seconds=args.chunk_overlap,
        bf_max_seconds=args.bf_max_seconds,
    )

def main():
    global app_cfg
    app_cfg = parse_args()
    console.rule("[bold]Audio Transform Visualizer")
    rng = SafeRNG(app_cfg.seed)

    # Load audio
    audio = load_audio_from_source(app_cfg)
    # Optional resample and trim
    audio = maybe_resample(audio, app_cfg.resample_to)
    audio = fix_duration(audio, app_cfg.max_seconds)
    audio = with_device(audio, app_cfg.device)
    ensure_dir(app_cfg.outdir)

    report_overview(app_cfg, audio)

    # Apply suite
    suite = TransformSuite(app_cfg, audio, rng)
    results = suite.run_all()
    report_results(results)

    if app_cfg.verbose:
        ok_count = sum(1 for r in results if r.ok)
        total = len(results)
        console.rule()
        console.print(f"Completed {ok_count}/{total} transforms. Output root: {os.path.abspath(app_cfg.outdir)}", style="bold cyan")
        console.rule()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        console.print_exception()
        sys.exit(1)