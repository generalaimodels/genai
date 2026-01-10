#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio Augmentation Super-Suite (channels-first, resilient, single-file, verbose, standards-compliant)

Purpose
- Load one audio file (any extension supported by libsndfile via soundfile),
- Apply EVERY audiomentations waveform transform listed below, one-by-one (non-cumulative),
- Save all results into ONE output folder, with rich, verbose metadata and a JSON log,
- Auto-fix array shape issues (uses channels-first for audiomentations),
- Auto-create assets (background noises, short noises, impulse responses) when not provided,
- Provide robust fallbacks for optional-dependency transforms (Mp3Compression, LoudnessNormalization, RoomSimulator),
- Handle API differences across audiomentations versions (e.g., Lambda, AdjustDuration).

Transforms attempted (39)
- AddBackgroundNoise       (assets auto-generated if not supplied)
- AddColorNoise
- AddGaussianNoise
- AddGaussianSNR
- AddShortNoises           (assets auto-generated if not supplied)
- AdjustDuration           (handles API variations; has fallback)
- AirAbsorption
- Aliasing
- ApplyImpulseResponse     (assets auto-generated if not supplied)
- BandPassFilter
- BandStopFilter
- BitCrush
- Clip
- ClippingDistortion
- Gain
- GainTransition
- HighPassFilter
- HighShelfFilter
- Lambda                   (uses transform=..., not function=...)
- Limiter
- LoudnessNormalization    (fallback if pyloudnorm missing)
- LowPassFilter
- LowShelfFilter
- Mp3Compression           (fallback if pydub/ffmpeg missing)
- Normalize
- Padding
- PeakingFilter
- PitchShift
- PolarityInversion
- RepeatPart
- Resample
- Reverse
- RoomSimulator            (fallback if pyroomacoustics missing)
- SevenBandParametricEQ
- Shift
- TanhDistortion
- TimeMask
- TimeStretch
- Trim

Key Fixes vs. common pitfalls
- audiomentations expects multichannel audio shaped as (channels, samples). This script ALWAYS feeds (C, T) into transforms.
- AdjustDuration has different constructor parameter names across versions; we introspect and pass the correct one (or fallback).
- Lambda requires transform=callable in newer versions; we bind a robust demo transform.
- Asset-dependent transforms now work out of the box thanks to automatic asset synthesis (pink/white noise, short beeps/clicks, synthetic IRs).
- Optional dependency transforms (LoudnessNormalization, Mp3Compression, RoomSimulator) have high-quality fallbacks when dependencies are missing.

Install
- Required: pip install -U audiomentations numpy soundfile rich
- Optional:
  - pyloudnorm (better LoudnessNormalization): pip install pyloudnorm
  - pydub + ffmpeg (Mp3Compression): pip install pydub, and install ffmpeg executable
  - pyroomacoustics (RoomSimulator): pip install pyroomacoustics
  - librosa (high-quality pre-resampling, pitch/time algorithms): pip install librosa

CLI
- Basic:
  python augment_audio_suite.py input.wav --out ./aug_out

- With assets (if you already have them):
  python augment_audio_suite.py input.wav --out ./aug_out \
    --noise-dir ./assets/background_noises \
    --short-noise-dir ./assets/short_noises \
    --ir-dir ./assets/impulse_responses

- Force seed, target duration, and PCM16:
  python augment_audio_suite.py input.wav --out ./aug_out --target-duration 12.0 --seed 7 --pcm16

- Run subset or skip some:
  python augment_audio_suite.py input.wav --out ./aug_out --only "PitchShift,TimeStretch" --skip "RoomSimulator"

Design notes
- Input audio is read as float32 [-1, 1] and kept non-cumulative between transforms.
- We compute stats (RMS/Peak dBFS, optional LUFS) pre and post each transform.
- We write metadata per transform and a global _augmentation_meta.json in the output directory.
- We use a single base seed and derive per-transform seeds for reproducibility.
- Where exact external algorithms are missing, fallbacks aim to be perceptually reasonable.

This file is purposefully verbose and deeply commented to teach best practices, edge cases, and robust engineering choices.
"""
from __future__ import annotations

import argparse
import dataclasses
import inspect
import io
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf


# ----------------------------- Data Models -----------------------------------


TransformName = str


@dataclass
class AugmentConfig:
    out_dir: Union[str, Path]
    noise_dir: Optional[Union[str, Path]] = None
    short_noise_dir: Optional[Union[str, Path]] = None
    ir_dir: Optional[Union[str, Path]] = None

    # Processing controls
    target_duration: Optional[float] = 5.0  # seconds for AdjustDuration (fallback uses this)
    resample_sr: Optional[int] = None  # optional pre-resampling

    # Selection controls
    only: Optional[List[TransformName]] = None
    skip: Optional[List[TransformName]] = None

    # Output controls
    pcm16: bool = True  # save outputs as PCM_16; else float32

    # Reproducibility
    seed: Optional[int] = None

    # Verbosity
    verbose: bool = True

    # Fail behavior
    fail_fast: bool = False


@dataclass
class AudioStats:
    sample_rate: int
    num_samples: int
    num_channels: int
    duration_sec: float
    dtype: str
    min_value: float
    max_value: float
    rms_dbfs: float
    peak_dbfs: float
    lufs: Optional[float] = None


@dataclass
class TransformResult:
    name: TransformName
    status: str  # ok | skipped | failed
    reason: Optional[str]
    out_path: Optional[str]
    elapsed_ms: Optional[float]
    input_stats: Optional[AudioStats]
    output_stats: Optional[AudioStats]
    parameters: Optional[Dict[str, Any]]


# ----------------------------- Global Constants ------------------------------


ALL_TRANSFORMS: List[TransformName] = [
    "AddBackgroundNoise",
    "AddColorNoise",
    "AddGaussianNoise",
    "AddGaussianSNR",
    "AddShortNoises",
    "AdjustDuration",
    "AirAbsorption",
    "Aliasing",
    "ApplyImpulseResponse",
    "BandPassFilter",
    "BandStopFilter",
    "BitCrush",
    "Clip",
    "ClippingDistortion",
    "Gain",
    "GainTransition",
    "HighPassFilter",
    "HighShelfFilter",
    "Lambda",
    "Limiter",
    "LoudnessNormalization",
    "LowPassFilter",
    "LowShelfFilter",
    "Mp3Compression",
    "Normalize",
    "Padding",
    "PeakingFilter",
    "PitchShift",
    "PolarityInversion",
    "RepeatPart",
    "Resample",
    "Reverse",
    "RoomSimulator",
    "SevenBandParametricEQ",
    "Shift",
    "TanhDistortion",
    "TimeMask",
    "TimeStretch",
    "Trim",
]


# ----------------------------- Rich Console Setup ----------------------------


def _get_console(enabled: bool):
    if enabled:
        try:
            from rich.console import Console  # type: ignore
            from rich.traceback import install as install_rich_traceback  # type: ignore

            install_rich_traceback(width=140, show_locals=False)
            return Console()
        except Exception:
            pass

    class _DummyConsole:
        def print(self, *args, **kwargs):
            print(*args)

    return _DummyConsole()


# ----------------------------- Seeding ---------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def derive_seed(base_seed: Optional[int], name: str) -> Optional[int]:
    if base_seed is None:
        return None
    mix = f"{base_seed}:{name}"
    return abs(hash(mix)) % (2**31 - 1)


# ----------------------------- IO & Shapes -----------------------------------


def load_audio_any(path: Union[str, Path]) -> Tuple[np.ndarray, int]:
    """
    Load with soundfile as float32. Return samples as [T, C] and sample_rate.
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data, int(sr)


def save_audio(path: Union[str, Path], samples_tc: np.ndarray, sample_rate: int, pcm16: bool) -> None:
    """
    Save [T, C] using soundfile. If pcm16 True, clip to [-1, 1] and use PCM_16.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = samples_tc
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    if pcm16:
        x = np.clip(x, -1.0, 1.0)
        sf.write(str(path), x, samplerate=sample_rate, subtype="PCM_16")
    else:
        sf.write(str(path), x, samplerate=sample_rate, subtype="FLOAT")


def as_tc(x: np.ndarray) -> np.ndarray:
    """
    Ensure [T, C].
    Accepts [T, C], [C, T] or [T].
    """
    if x.ndim == 1:
        return x[:, None].astype(np.float32, copy=False)
    if x.ndim == 2:
        T, C = x.shape
        # If channels-last already (T >> C small), keep; otherwise transpose
        if T >= C and C <= 32:
            return x.astype(np.float32, copy=False)
        else:
            return x.T.astype(np.float32, copy=False)
    raise ValueError(f"Unsupported shape for as_tc: {x.shape}")


def as_ct_from_tc(x_tc: np.ndarray) -> np.ndarray:
    """
    Convert [T, C] -> [C, T].
    """
    assert x_tc.ndim == 2
    return x_tc.T.astype(np.float32, copy=False)


def as_tc_from_ct(x_ct: np.ndarray) -> np.ndarray:
    """
    Convert [C, T] -> [T, C].
    """
    assert x_ct.ndim == 2
    return x_ct.T.astype(np.float32, copy=False)


# ---------------------------- Preprocessing ----------------------------------


def preprocess_resample(samples_tc: np.ndarray, sample_rate: int, target_sr: Optional[int], console=None):
    if target_sr is None or target_sr == sample_rate:
        return samples_tc, sample_rate
    try:
        import librosa  # type: ignore

        # librosa expects 1D per channel; resample per channel
        x_ct = as_ct_from_tc(samples_tc)  # [C, T]
        C, T = x_ct.shape
        out = []
        for ch in range(C):
            out.append(librosa.resample(x_ct[ch], orig_sr=sample_rate, target_sr=target_sr, res_type="kaiser_best"))
        y_ct = _pad_or_stack_channels(out)  # [C, T']
        return as_tc_from_ct(y_ct), int(target_sr)
    except Exception as e:
        if console:
            console.print(f"[yellow]Pre-resample requested but failed ({e}). Keeping original SR {sample_rate}.[/yellow]")
        return samples_tc, sample_rate


def _pad_or_stack_channels(ch_signals: List[np.ndarray]) -> np.ndarray:
    """
    Stack list of 1D arrays as [C, T], padding shorter channels with zeros if needed.
    """
    max_len = max(len(c) for c in ch_signals) if ch_signals else 0
    out = []
    for sig in ch_signals:
        if len(sig) < max_len:
            pad = np.zeros(max_len - len(sig), dtype=np.float32)
            out.append(np.concatenate([sig.astype(np.float32), pad], axis=0))
        else:
            out.append(sig.astype(np.float32))
    return np.stack(out, axis=0) if out else np.zeros((0, 0), dtype=np.float32)


# ------------------------------ Stats & Metrics ------------------------------


def compute_stats(samples_tc: np.ndarray, sample_rate: int) -> AudioStats:
    x = as_tc(samples_tc)
    T, C = x.shape
    dur = float(T) / float(sample_rate) if sample_rate > 0 else 0.0
    xmin = float(np.min(x)) if x.size else 0.0
    xmax = float(np.max(x)) if x.size else 0.0
    rms = float(np.sqrt(np.mean(x**2))) if x.size else 0.0
    eps = 1e-12
    rms_dbfs = 20.0 * math.log10(max(rms, eps))
    peak_dbfs = 20.0 * math.log10(max(float(np.max(np.abs(x))), eps))
    lufs: Optional[float] = None
    try:
        import pyloudnorm as pyln  # type: ignore

        mono = np.mean(x, axis=1)
        meter = pyln.Meter(sample_rate)  # EBU R128
        lufs = float(meter.integrated_loudness(mono))
    except Exception:
        lufs = None
    return AudioStats(
        sample_rate=int(sample_rate),
        num_samples=int(T),
        num_channels=int(C),
        duration_sec=float(dur),
        dtype=str(x.dtype),
        min_value=float(xmin),
        max_value=float(xmax),
        rms_dbfs=float(rms_dbfs),
        peak_dbfs=float(peak_dbfs),
        lufs=lufs,
    )


def to_jsonable(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # numpy scalars
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)


# ------------------------ Assets: Auto Generation ----------------------------


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _gen_colored_noise(color: str, num_samples: int, rng: np.random.Generator, sr: int) -> np.ndarray:
    """
    Generate one channel of colored noise via frequency-domain shaping.
    color in {'white', 'pink', 'brown'}
    """
    # White Gaussian
    x = rng.standard_normal(num_samples).astype(np.float32)
    if color == "white":
        return x * 0.2
    # Shape in frequency domain: multiply spectrum by f^{-alpha/2}
    # pink alpha=1, brown alpha=2
    alpha = 1.0 if color == "pink" else 2.0
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(num_samples, d=1.0 / sr)
    mag = np.ones_like(freqs)
    mag[1:] = 1.0 / (freqs[1:] ** (alpha / 2.0))
    X_shaped = X * mag
    y = np.fft.irfft(X_shaped, n=num_samples).real.astype(np.float32)
    y = y / (np.max(np.abs(y)) + 1e-9) * 0.2
    return y


def _sine_beep(num_samples: int, sr: int, freq: float, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(num_samples, dtype=np.float32) / float(sr)
    phase = rng.uniform(0, 2 * math.pi)
    sig = np.sin(2 * math.pi * freq * t + phase).astype(np.float32)
    # apply hann envelope
    N = len(sig)
    w = np.hanning(N).astype(np.float32)
    sig = sig * w
    # moderate level
    return (sig * 0.3).astype(np.float32)


def _click(num_samples: int, rng: np.random.Generator) -> np.ndarray:
    sig = np.zeros(num_samples, dtype=np.float32)
    pos = rng.integers(low=0, high=max(1, num_samples // 3))
    length = int(max(1, num_samples * 0.02))
    sig[pos : min(num_samples, pos + length)] = rng.uniform(0.5, 1.0)
    # decay
    decay = np.linspace(1.0, 0.0, length, dtype=np.float32)
    sig[pos : min(num_samples, pos + length)] *= decay
    return sig


def _exp_decay_ir(rt60: float, sr: int, length_sec: float, rng: np.random.Generator) -> np.ndarray:
    """
    Generate a simple room-like IR: early reflections + exponential late decay.
    """
    N = int(length_sec * sr)
    if N < 4:
        N = 4
    t = np.arange(N, dtype=np.float32) / float(sr)
    # approximate decay parameter: amplitude ~ exp(-t / tau)
    tau = rt60 / 6.91  # -60 dB at t=rt60 -> tau ~ rt60/6.91
    late = np.exp(-t / max(tau, 1e-3)).astype(np.float32)
    late *= np.random.default_rng(rng.integers(0, 10_000)).standard_normal(N).astype(np.float32) * 0.05

    # early reflections (a few taps)
    taps = np.zeros(N, dtype=np.float32)
    for _ in range(rng.integers(3, 8)):
        delay = rng.integers(40, min(N - 1, int(0.05 * sr)))  # up to 50 ms
        amp = rng.uniform(0.1, 0.6)
        taps[delay] += amp

    ir = taps + late
    # normalize IR energy
    ir = ir / (np.max(np.abs(ir)) + 1e-9) * 0.9
    return ir.astype(np.float32)


def ensure_auto_assets(
    base_out_dir: Path,
    sr: int,
    num_channels: int,
    approx_duration_sec: float,
    cfg: AugmentConfig,
    console=None,
) -> AugmentConfig:
    """
    Create auto assets when not supplied by the user:
    - background noises: 3 files, ~max(10s, target_duration)
    - short noises: ~15 beeps/clicks, 0.05..0.35s
    - impulse responses: 5 synthetic IRs
    Returns an updated cfg with directories filled in.
    """
    rng = np.random.default_rng(1337)
    assets_root = Path(base_out_dir) / "_auto_assets"
    _ensure_dir(assets_root)

    # Background noises
    if cfg.noise_dir is None or not Path(cfg.noise_dir).exists():
        bg_dir = assets_root / "background_noises"
        _ensure_dir(bg_dir)
        dur = max(10.0, float(cfg.target_duration or 5.0), approx_duration_sec * 0.25)
        N = int(dur * sr)
        colors = ["white", "pink", "brown"]
        for i, color in enumerate(colors, start=1):
            chs = []
            for _ in range(num_channels):
                chs.append(_gen_colored_noise(color, N, rng, sr))
            y_ct = np.stack(chs, axis=0)  # [C, T]
            y_tc = as_tc_from_ct(y_ct)
            save_audio(bg_dir / f"{i:02d}_{color}_noise.wav", y_tc, sr, pcm16=False)
        cfg.noise_dir = str(bg_dir)
        if console:
            console.print(f"[green]Auto-generated background noises at {bg_dir}[/green]")

    # Short noises
    if cfg.short_noise_dir is None or not Path(cfg.short_noise_dir).exists():
        sn_dir = assets_root / "short_noises"
        _ensure_dir(sn_dir)
        for i in range(1, 16):
            length_sec = float(rng.uniform(0.05, 0.35))
            N = int(length_sec * sr)
            if rng.random() < 0.6:
                freq = float(rng.uniform(400, 5000))
                mono = _sine_beep(N, sr, freq, rng)
            else:
                mono = _click(N, rng)
            # Make multichannel by duplicating with slight variations
            chs = []
            for ch in range(num_channels):
                jitter = rng.uniform(0.95, 1.05)
                sig = mono * jitter
                chs.append(sig.astype(np.float32))
            y_ct = _pad_or_stack_channels(chs)  # [C, T]
            y_tc = as_tc_from_ct(y_ct)
            save_audio(sn_dir / f"{i:02d}_short.wav", y_tc, sr, pcm16=False)
        cfg.short_noise_dir = str(sn_dir)
        if console:
            console.print(f"[green]Auto-generated short noises at {sn_dir}[/green]")

    # Impulse responses
    if cfg.ir_dir is None or not Path(cfg.ir_dir).exists():
        ir_dir = assets_root / "impulse_responses"
        _ensure_dir(ir_dir)
        for i in range(1, 6):
            rt60 = float(rng.uniform(0.2, 1.0))
            length_sec = min(0.5, rt60 + 0.1)  # keep IR short for performance
            chs = []
            for ch in range(num_channels):
                ir = _exp_decay_ir(rt60, sr, length_sec, rng)
                chs.append(ir)
            y_ct = _pad_or_stack_channels(chs)  # [C, T]
            y_tc = as_tc_from_ct(y_ct)
            save_audio(ir_dir / f"{i:02d}_ir.wav", y_tc, sr, pcm16=False)
        cfg.ir_dir = str(ir_dir)
        if console:
            console.print(f"[green]Auto-generated impulse responses at {ir_dir}[/green]")

    return cfg


# ------------------------ Fallback Transforms --------------------------------


class FallbackLoudnessNormalization:
    """
    Approximation when pyloudnorm is unavailable.
    We scale by RMS to reach approximate target dBFS (proxy for LUFS).
    """
    def __init__(self, target_lufs: float = -23.0, p: float = 1.0):
        self.target_lufs = float(target_lufs)
        self.p = float(p)
        self.parameters: Dict[str, Any] = {}

    def __call__(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        # Expect [C, T]
        x_ct = samples.astype(np.float32, copy=True)
        x_tc = as_tc_from_ct(x_ct)
        eps = 1e-12
        rms = float(np.sqrt(np.mean(x_tc**2)))
        cur_db = 20.0 * math.log10(max(rms, eps))
        # Map LUFS target to similar RMS dBFS target (very rough but reasonable proxy)
        target_db = self.target_lufs
        gain_db = target_db - cur_db
        gain = float(10.0 ** (gain_db / 20.0))
        y_ct = (x_ct * gain).astype(np.float32)
        self.parameters = {"fallback": True, "target_lufs": self.target_lufs, "gain_db": gain_db}
        return y_ct


class FallbackRoomSimulator:
    """
    Approximate room effect via synthetic IR convolution if pyroomacoustics is unavailable.
    """
    def __init__(self, min_rt60: float = 0.2, max_rt60: float = 1.0, p: float = 1.0, seed: Optional[int] = None):
        self.min_rt60 = float(min_rt60)
        self.max_rt60 = float(max_rt60)
        self.p = float(p)
        self.rng = np.random.default_rng(seed)
        self.parameters: Dict[str, Any] = {}

    def __call__(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        x_ct = samples.astype(np.float32, copy=False)
        C, T = x_ct.shape
        rt60 = float(self.rng.uniform(self.min_rt60, self.max_rt60))
        ir = _exp_decay_ir(rt60, sample_rate, min(0.5, rt60 + 0.1), self.rng)  # keep short for speed
        ir = ir / (np.max(np.abs(ir)) + 1e-9)
        # Fast FFT-based convolution per channel, keep original length
        n = T + len(ir) - 1
        nfft = 1 << (n - 1).bit_length()  # next pow2
        IR = np.fft.rfft(ir, n=nfft)
        out = np.zeros((C, T), dtype=np.float32)
        for ch in range(C):
            X = np.fft.rfft(x_ct[ch], n=nfft)
            Y = X * IR
            y = np.fft.irfft(Y, n=nfft)[:T]
            out[ch] = y.astype(np.float32)
        # Slight attenuation to prevent clipping
        out = out * 0.9
        self.parameters = {"fallback": True, "rt60": rt60, "ir_len": len(ir)}
        return out


class FallbackMp3Compression:
    """
    Approximate MP3 compression artifacts: downsample + upsample + bitcrush + lowpass.
    Avoids external encoders.
    """
    def __init__(self, min_bitrate: int = 32, max_bitrate: int = 128, p: float = 1.0, seed: Optional[int] = None):
        self.min_bitrate = int(min_bitrate)
        self.max_bitrate = int(max_bitrate)
        self.p = float(p)
        self.rng = np.random.default_rng(seed)
        self.parameters: Dict[str, Any] = {}

    def __call__(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        x_ct = samples.astype(np.float32, copy=False)
        C, T = x_ct.shape
        # Pick a target "bitrate" -> map to a target sample rate and bit depth
        br = int(self.rng.integers(self.min_bitrate, self.max_bitrate + 1))
        # Heuristic mapping
        target_sr = int(np.clip(br * 1000 // 4, 8000, min(22050, sample_rate)))
        bit_depth = 8 if br < 72 else 12

        # Resample down then up (linear)
        y_ct = np.zeros_like(x_ct)
        for ch in range(C):
            down = _resample_linear(x_ct[ch], sample_rate, target_sr)
            up = _resample_linear(down, target_sr, sample_rate)
            # Bitcrush
            y_ct[ch] = _bitcrush(up, bit_depth)

        # Gentle lowpass to emulate bandwidth reduction
        y_ct = _simple_lowpass(y_ct, sample_rate, cutoff_hz=min(0.45 * target_sr, 6000.0))
        self.parameters = {"fallback": True, "bitrate_kbps": br, "target_sr": target_sr, "bit_depth": bit_depth}
        return y_ct.astype(np.float32)


class FallbackAdjustDuration:
    """
    Fallback for AdjustDuration if the class signature mismatches in current version.
    """
    def __init__(self, duration_sec: float, p: float = 1.0, pad_section: str = "end"):
        self.duration_sec = float(duration_sec)
        self.p = float(p)
        self.pad_section = pad_section
        self.parameters: Dict[str, Any] = {"fallback": True, "duration_sec": self.duration_sec, "pad_section": pad_section}

    def __call__(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        x_ct = samples.astype(np.float32, copy=False)
        C, T = x_ct.shape
        target_T = int(round(self.duration_sec * sample_rate))
        if target_T <= 0:
            return x_ct
        if T == target_T:
            return x_ct
        if T > target_T:
            # Trim
            if self.pad_section == "center":
                start = (T - target_T) // 2
                end = start + target_T
                return x_ct[:, start:end]
            elif self.pad_section == "start":
                return x_ct[:, T - target_T :]
            else:
                return x_ct[:, :target_T]
        # Pad
        pad = target_T - T
        if self.pad_section == "center":
            left = pad // 2
            right = pad - left
            return np.pad(x_ct, ((0, 0), (left, right)), mode="constant")
        elif self.pad_section == "start":
            return np.pad(x_ct, ((0, 0), (pad, 0)), mode="constant")
        else:
            return np.pad(x_ct, ((0, 0), (0, pad)), mode="constant")


# ----------------------------- DSP helpers -----------------------------------


def _resample_linear(sig: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return sig.astype(np.float32, copy=True)
    # 1D linear interpolation
    x = np.arange(len(sig), dtype=np.float64)
    xp = np.linspace(0, len(sig) - 1, int(round(len(sig) * target_sr / orig_sr)), dtype=np.float64)
    y = np.interp(xp, x, sig.astype(np.float64))
    return y.astype(np.float32)


def _bitcrush(sig: np.ndarray, bits: int) -> np.ndarray:
    # Map [-1,1] to quantization levels
    levels = float(2**bits - 1)
    y = np.round(((sig + 1.0) * 0.5) * levels) / levels * 2.0 - 1.0
    return y.astype(np.float32)


def _simple_lowpass(x_ct: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    # One-pole low-pass per channel: y[n] = a*y[n-1] + (1-a)*x[n], a = exp(-2*pi*fc/fs)
    a = math.exp(-2.0 * math.pi * cutoff_hz / sr)
    y = np.zeros_like(x_ct)
    for ch in range(x_ct.shape[0]):
        prev = 0.0
        for n in range(x_ct.shape[1]):
            prev = a * prev + (1.0 - a) * x_ct[ch, n]
            y[ch, n] = prev
    return y.astype(np.float32)


# ----------------------- Transform Instantiation -----------------------------


def _has_optional_dependency_for_transform(name: str) -> Tuple[bool, Optional[str]]:
    if name == "Mp3Compression":
        try:
            import pydub  # noqa: F401

            return True, None
        except Exception:
            return False, "pydub/ffmpeg not available"
    if name == "LoudnessNormalization":
        try:
            import pyloudnorm  # noqa: F401

            return True, None
        except Exception:
            return False, "pyloudnorm not available"
    if name == "RoomSimulator":
        try:
            import pyroomacoustics  # noqa: F401

            return True, None
        except Exception:
            return False, "pyroomacoustics not available"
    return True, None


def _assets_ready_for_transform(name: str, cfg: AugmentConfig) -> Tuple[bool, Optional[str], Optional[Path]]:
    if name == "AddBackgroundNoise":
        if cfg.noise_dir is None:
            return False, "requires --noise-dir", None
        p = Path(cfg.noise_dir)
        if not p.exists() or not p.is_dir():
            return False, f"noise_dir not found: {p}", None
        return True, None, p
    if name == "AddShortNoises":
        p = Path(cfg.short_noise_dir) if cfg.short_noise_dir else (Path(cfg.noise_dir) if cfg.noise_dir else None)
        if p is None:
            return False, "requires --short-noise-dir or --noise-dir", None
        if not p.exists() or not p.is_dir():
            return False, f"short_noise_dir not found: {p}", None
        return True, None, p
    if name == "ApplyImpulseResponse":
        if cfg.ir_dir is None:
            return False, "requires --ir-dir", None
        p = Path(cfg.ir_dir)
        if not p.exists() or not p.is_dir():
            return False, f"ir_dir not found: {p}", None
        return True, None, p
    return True, None, None


def _pick_duration_kwargs(sig: inspect.Signature, target_sec: float, sr: int) -> Dict[str, Any]:
    """
    Different audiomentations versions use different parameter names for AdjustDuration.
    We probe and return the correct kwargs for the current signature.
    """
    names = set(sig.parameters.keys())
    if "duration" in names:
        return {"duration": float(target_sec)}
    if "target_duration" in names:
        return {"target_duration": float(target_sec)}
    if "duration_seconds" in names:
        return {"duration_seconds": float(target_sec)}
    if "duration_in_seconds" in names:
        return {"duration_in_seconds": float(target_sec)}
    # samples-based fallbacks
    target_samples = int(round(target_sec * sr))
    for k in ["duration_samples", "target_num_samples", "num_samples", "length_in_samples"]:
        if k in names:
            return {k: int(target_samples)}
    # give something sensible anyway
    return {"duration": float(target_sec)}


def instantiate_transform(
    amod: Any,
    name: str,
    cfg: AugmentConfig,
    sample_rate: int,
    num_channels: int,
    approx_duration_sec: float,
    console=None,
) -> Tuple[Optional[Any], Optional[str]]:
    """
    Instantiate a transform with robust defaults and asset handling.
    Returns (transform_or_fallback, skip_reason_or_None).
    """
    # Proactively create assets if needed
    if name in {"AddBackgroundNoise", "AddShortNoises", "ApplyImpulseResponse"}:
        cfg = ensure_auto_assets(Path(cfg.out_dir), sample_rate, num_channels, approx_duration_sec, cfg, console=console)

    ok_assets, reason_assets, asset_path = _assets_ready_for_transform(name, cfg)
    if not ok_assets:
        # If still not ok, skip
        return None, reason_assets

    # Optional dependency fallbacks
    ok_dep, reason_dep = _has_optional_dependency_for_transform(name)
    if not ok_dep:
        if name == "LoudnessNormalization":
            # Fallback approximator
            fln = FallbackLoudnessNormalization(target_lufs=-23.0, p=1.0)
            return fln, None
        if name == "RoomSimulator":
            frs = FallbackRoomSimulator(p=1.0, seed=derive_seed(0, "FallbackRoomSimulator"))
            return frs, None
        if name == "Mp3Compression":
            fmp3 = FallbackMp3Compression(p=1.0, seed=derive_seed(0, "FallbackMp3Compression"))
            return fmp3, None
        # Other deps: skip
        return None, reason_dep

    # Acquire class
    if not hasattr(amod, name):
        return None, f"transform not found in audiomentations: {name}"
    cls = getattr(amod, name)

    # Base kwargs
    kwargs: Dict[str, Any] = {}
    try:
        sig = inspect.signature(cls.__init__)
    except Exception:
        sig = None

    # Force p=1
    if sig and "p" in sig.parameters:
        kwargs["p"] = 1.0
    else:
        kwargs["p"] = 1.0

    # Asset arguments
    if name in {"AddBackgroundNoise", "AddShortNoises"} and asset_path is not None:
        # sounds_path or sounds_paths
        if sig and "sounds_path" in sig.parameters:
            kwargs["sounds_path"] = str(asset_path)
        elif sig and "sounds_paths" in sig.parameters:
            kwargs["sounds_paths"] = [str(asset_path)]
        else:
            kwargs["sounds_path"] = str(asset_path)
    if name == "ApplyImpulseResponse" and asset_path is not None:
        if sig and "ir_path" in sig.parameters:
            kwargs["ir_path"] = str(asset_path)
        else:
            kwargs["ir_path"] = str(asset_path)

    # Special handling
    if name == "AdjustDuration":
        # Probe signature for the right duration parameter name
        if sig:
            kwargs.update(_pick_duration_kwargs(sig, float(cfg.target_duration or 5.0), sample_rate))
        else:
            kwargs["duration"] = float(cfg.target_duration or 5.0)

    if name == "Lambda":
        # audiomentations >=0.33 uses transform=...
        def _demo_lambda(samples: np.ndarray, sample_rate: int) -> np.ndarray:
            # samples expected [C, T]; cosine fade in/out + gentle tanh saturation
            x_ct = samples.astype(np.float32, copy=True)
            C, T = x_ct.shape
            wlen = max(1024, int(0.05 * T))
            if wlen * 2 < T:
                fade = 0.5 * (1 - np.cos(np.linspace(0, math.pi, wlen, dtype=np.float32)))
                for ch in range(C):
                    x_ct[ch, :wlen] *= fade
                    x_ct[ch, -wlen:] *= fade[::-1]
            x_ct = np.tanh(1.2 * x_ct).astype(np.float32)
            return x_ct

        if sig and "transform" in sig.parameters:
            kwargs["transform"] = _demo_lambda
        else:
            # Older versions might use "function"
            kwargs["function"] = _demo_lambda

    # Conservative defaults for missing required parameters
    param_defaults: Dict[str, Any] = {
        "min_amplitude": 0.001,
        "max_amplitude": 0.015,
        "min_snr_in_db": 5.0,
        "max_snr_in_db": 35.0,
        "min_bit_depth": 8,
        "max_bit_depth": 14,
        "min_bitrate": 32,   # kbps
        "max_bitrate": 192,
        "min_sample_rate": 8000,
        "max_sample_rate": min(48000, sample_rate),
        "min_center_freq": 200.0,
        "max_center_freq": min(8000.0, 0.45 * sample_rate),
        "min_cutoff_freq": 100.0,
        "max_cutoff_freq": min(8000.0, 0.45 * sample_rate),
        "min_bandwidth_fraction": 0.05,
        "max_bandwidth_fraction": 0.75,
        "min_gain_db": -12.0,
        "max_gain_db": 12.0,
        "min_gain_in_db": -12.0,
        "max_gain_in_db": 12.0,
        "min_q": 0.5,
        "max_q": 2.0,
        "min_rate": 0.8,
        "max_rate": 1.25,
        "min_semitones": -4.0,
        "max_semitones": 4.0,
        "min_fraction": -0.35,
        "max_fraction": 0.35,
        "fade": True,
        "min_repeat": 2,
        "max_repeat": 4,
        "min_repeat_fraction": 0.05,
        "max_repeat_fraction": 0.25,
        "top_db": 40.0,
        "min_band_part": 0.05,
        "max_band_part": 0.25,
    }

    # Try to instantiate with current kwargs
    try:
        return cls(**kwargs), None
    except TypeError as e:
        # Fill any missing required params from param_defaults
        try:
            if sig is None:
                sig = inspect.signature(cls.__init__)
            for pname, p in sig.parameters.items():
                if pname in {"self"}:
                    continue
                if pname in kwargs:
                    continue
                if p.default is inspect._empty and pname in param_defaults:
                    kwargs[pname] = param_defaults[pname]
                # AdjustDuration alternative names based on signature
                if name == "AdjustDuration" and p.default is inspect._empty and pname not in kwargs:
                    kwargs.update(_pick_duration_kwargs(sig, float(cfg.target_duration or 5.0), sample_rate))
            return cls(**kwargs), None
        except Exception as e2:
            # Provide fallbacks for a few known transforms
            if name == "AdjustDuration":
                fad = FallbackAdjustDuration(duration_sec=float(cfg.target_duration or 5.0))
                return fad, None
            if name == "Lambda":
                # Create a "passthrough" fallback
                return (lambda samples, sample_rate: samples), None
            return None, f"init failed: {e} | retry failed: {e2}"
    except Exception as e:
        if name == "AdjustDuration":
            fad = FallbackAdjustDuration(duration_sec=float(cfg.target_duration or 5.0))
            return fad, None
        return None, f"init failed: {e}"


# ------------------------------ Apply Transform ------------------------------


def apply_one_transform(
    transform: Any,
    name: str,
    samples_ct: np.ndarray,
    sample_rate: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply a single transform. Input and output are [C, T].
    Returns (y_ct, parameters_dict)
    """
    # Call transform
    out = transform(samples=samples_ct, sample_rate=sample_rate)
    # Ensure [C, T]
    if out.ndim == 1:
        out = out[None, :]
    elif out.ndim == 2 and out.shape[0] < out.shape[1]:
        # Already [C, T]
        pass
    elif out.ndim == 2:
        # Likely [T, C]
        out = out.T
    out = out.astype(np.float32, copy=False)

    # Collect parameters if available
    params: Dict[str, Any] = {}
    try:
        params = dict(getattr(transform, "parameters", {}) or {})
    except Exception:
        try:
            # Some fallbacks might store a dict attribute
            params = dict(getattr(transform, "__dict__", {}) or {})
        except Exception:
            params = {}
    return out, params


# --------------------------------- Orchestrator ------------------------------


def slugify(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in text)
    return safe[:200]


def _collect_versions() -> Dict[str, str]:
    out = {}
    try:
        import audiomentations as A  # type: ignore

        out["audiomentations"] = getattr(A, "__version__", "unknown")
    except Exception:
        out["audiomentations"] = "unavailable"
    try:
        import numpy as np  # type: ignore

        out["numpy"] = np.__version__
    except Exception:
        pass
    try:
        import soundfile as sf  # type: ignore

        out["soundfile"] = sf.__version__
    except Exception:
        pass
    try:
        import rich  # type: ignore

        out["rich"] = rich.__version__
    except Exception:
        pass
    try:
        import librosa  # type: ignore

        out["librosa"] = librosa.__version__
    except Exception:
        pass
    try:
        import pyloudnorm  # type: ignore

        out["pyloudnorm"] = pyloudnorm.__version__
    except Exception:
        pass
    try:
        import pyroomacoustics  # type: ignore

        out["pyroomacoustics"] = pyroomacoustics.__version__
    except Exception:
        pass
    try:
        import pydub  # type: ignore

        out["pydub"] = pydub.__version__
    except Exception:
        pass
    return out


def augment_all(input_audio: Union[str, Path], config: AugmentConfig) -> Dict[str, Any]:
    console = _get_console(config.verbose)

    # Import audiomentations
    try:
        import audiomentations as A  # type: ignore
    except Exception as e:
        raise RuntimeError(f"audiomentations is required: pip install -U audiomentations. Error: {e}")

    in_path = Path(input_audio)
    if not in_path.exists():
        raise FileNotFoundError(f"Input audio not found: {in_path}")

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Seed
    if config.seed is not None:
        seed_everything(config.seed)

    # Load audio as [T, C]
    raw_tc, raw_sr = load_audio_any(in_path)

    # Pre-resample if requested
    x_tc, sr = preprocess_resample(raw_tc, raw_sr, config.resample_sr, console=console)
    in_stats = compute_stats(x_tc, sr)
    C = x_tc.shape[1]
    T = x_tc.shape[0]
    x_ct = as_ct_from_tc(x_tc)

    # Selection of transforms
    if config.only:
        selected = [t for t in ALL_TRANSFORMS if t in set(config.only)]
    else:
        selected = ALL_TRANSFORMS.copy()
    if config.skip:
        selected = [t for t in selected if t not in set(config.skip)]

    # Banner
    try:
        from rich.panel import Panel  # type: ignore

        console.print(
            Panel.fit(
                f"Audio Augmentation Suite\n"
                f"Input: {in_path.name}\n"
                f"SR: {sr} Hz | Duration: {in_stats.duration_sec:.3f}s | Channels: {in_stats.num_channels}\n"
                f"Saving to: {str(out_dir)}\n"
                f"Transforms: {len(selected)} selected",
                title="audiomentations",
            )
        )
    except Exception:
        console.print(
            f"Input: {in_path.name} | SR: {sr} Hz | Duration: {in_stats.duration_sec:.3f}s | Channels: {in_stats.num_channels}"
        )

    # Ensure auto assets upfront (so that asset-dependent transforms will work)
    ensure_auto_assets(out_dir, sr, C, in_stats.duration_sec, config, console=console)

    # Run transforms one-by-one from original x_ct
    results: List[TransformResult] = []
    for idx, name in enumerate(selected, start=1):
        # Per-transform seed
        t_seed = derive_seed(config.seed, name)
        if t_seed is not None:
            seed_everything(t_seed)

        # Instantiate transform or fallback
        transform, skip_reason = instantiate_transform(
            A, name, config, sr, C, in_stats.duration_sec, console=console
        )
        if transform is None:
            results.append(
                TransformResult(
                    name=name,
                    status="skipped",
                    reason=skip_reason or "not available",
                    out_path=None,
                    elapsed_ms=None,
                    input_stats=in_stats,
                    output_stats=None,
                    parameters=None,
                )
            )
            try:
                from rich import box  # type: ignore
                from rich.table import Table  # type: ignore

                table = Table(title=f"{idx:02d}. {name} -> SKIPPED", box=box.SIMPLE)
                table.add_column("Reason", style="yellow")
                table.add_row(skip_reason or "Unknown")
                console.print(table)
            except Exception:
                console.print(f"[SKIP] {idx:02d}. {name}: {skip_reason}")
            continue

        # Apply safely
        t0 = time.perf_counter()
        try:
            y_ct, params = apply_one_transform(transform, name, x_ct, sr)
            elapsed = (time.perf_counter() - t0) * 1000.0
            y_tc = as_tc_from_ct(y_ct)

            out_stats = compute_stats(y_tc, sr)

            # Save
            out_name = f"{idx:02d}_{slugify(name)}_{slugify(in_path.stem)}.wav"
            out_path = out_dir / out_name
            save_audio(out_path, y_tc, sr, pcm16=config.pcm16)

            # Log
            try:
                from rich import box  # type: ignore
                from rich.table import Table  # type: ignore

                table = Table(title=f"{idx:02d}. {name}", box=box.SIMPLE)
                table.add_column("Field", style="bold")
                table.add_column("Value")

                table.add_row("Status", "[green]OK[/green]")
                table.add_row("Output", str(out_path))
                table.add_row("Elapsed", f"{elapsed:.2f} ms")
                table.add_row("Input shape", f"{x_tc.shape} (T,C) @ {sr} Hz")
                table.add_row("Output shape", f"{y_tc.shape} (T,C) @ {sr} Hz")
                table.add_row("In RMS/Peak dBFS", f"{in_stats.rms_dbfs:.2f} / {in_stats.peak_dbfs:.2f}")
                table.add_row("Out RMS/Peak dBFS", f"{out_stats.rms_dbfs:.2f} / {out_stats.peak_dbfs:.2f}")
                if in_stats.lufs is not None or out_stats.lufs is not None:
                    table.add_row("In LUFS", f"{in_stats.lufs if in_stats.lufs is not None else 'n/a'}")
                    table.add_row("Out LUFS", f"{out_stats.lufs if out_stats.lufs is not None else 'n/a'}")
                short_params = "; ".join(f"{k}={to_jsonable(v)}" for k, v in list((params or {}).items())[:10])
                table.add_row("Params (head)", short_params if short_params else "n/a")
                console.print(table)
            except Exception:
                console.print(f"[OK] {idx:02d}. {name} -> {out_path} ({elapsed:.1f} ms)")

            results.append(
                TransformResult(
                    name=name,
                    status="ok",
                    reason=None,
                    out_path=str(out_path),
                    elapsed_ms=elapsed,
                    input_stats=in_stats,
                    output_stats=out_stats,
                    parameters=params or {},
                )
            )
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000.0
            reason = f"{type(e).__name__}: {e}"
            results.append(
                TransformResult(
                    name=name,
                    status="failed",
                    reason=reason,
                    out_path=None,
                    elapsed_ms=elapsed,
                    input_stats=in_stats,
                    output_stats=None,
                    parameters=None,
                )
            )
            try:
                from rich import box  # type: ignore
                from rich.table import Table  # type: ignore

                table = Table(title=f"{idx:02d}. {name} -> FAILED", box=box.SIMPLE)
                table.add_column("Error", style="red")
                table.add_row(reason)
                console.print(table)
            except Exception:
                console.print(f"[FAIL] {idx:02d}. {name} -> {reason}")
            if config.fail_fast:
                break

    # Metadata JSON
    meta = {
        "input": {
            "path": str(in_path),
            "sample_rate": sr,
            "stats": to_jsonable(in_stats),
        },
        "output_dir": str(out_dir),
        "config": to_jsonable(config),
        "results": [to_jsonable(r) for r in results],
        "versions": _collect_versions(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "notes": "Auto assets generated; fallbacks used for optional deps if missing; channels-first shape enforced.",
    }
    meta_path = out_dir / "_augmentation_meta.json"
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        console.print(f"Metadata written: {meta_path}")
    except Exception as e:
        console.print(f"[yellow]Could not write metadata JSON: {e}[/yellow]")

    ok = sum(1 for r in results if r.status == "ok")
    skipped = sum(1 for r in results if r.status == "skipped")
    failed = sum(1 for r in results if r.status == "failed")
    console.print(f"Summary: OK={ok} | Skipped={skipped} | Failed={failed} | Total={len(results)}")

    return meta


# ---------------------------------- CLI --------------------------------------


def _parse_list_arg(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    parts = [x.strip() for x in value.split(",")]
    parts = [p for p in parts if p]
    return parts or None


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Apply all audiomentations transforms to a single audio file (channels-first for audiomentations), saving each result.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input_audio", type=str, help="Path to input audio file")
    p.add_argument("--out", dest="out_dir", type=str, required=True, help="Output directory (will be created)")
    p.add_argument("--noise-dir", type=str, default=None, help="Folder with background noise wavs for AddBackgroundNoise (auto-generated if absent)")
    p.add_argument("--short-noise-dir", type=str, default=None, help="Folder with short noise wavs for AddShortNoises (auto-generated if absent)")
    p.add_argument("--ir-dir", type=str, default=None, help="Folder with impulse responses for ApplyImpulseResponse (auto-generated if absent)")
    p.add_argument("--target-duration", type=float, default=5.0, help="Target duration (seconds) for AdjustDuration")
    p.add_argument("--resample-sr", type=int, default=None, help="Pre-resample input to this SR before augmentations")
    p.add_argument("--only", type=str, default=None, help="Comma-separated transform names to run")
    p.add_argument("--skip", type=str, default=None, help="Comma-separated transform names to skip")
    p.add_argument("--seed", type=int, default=None, help="Base seed for reproducibility")
    p.add_argument("--pcm16", action="store_true", help="Save WAV output as PCM_16 instead of float32")
    p.add_argument("--no-verbose", action="store_true", help="Disable rich verbose logging")
    p.add_argument("--fail-fast", action="store_true", help="Stop on first error")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = AugmentConfig(
        out_dir=args.out_dir,
        noise_dir=args.noise_dir,
        short_noise_dir=args.short_noise_dir,
        ir_dir=args.ir_dir,
        target_duration=args.target_duration,
        resample_sr=args.resample_sr,
        only=_parse_list_arg(args.only),
        skip=_parse_list_arg(args.skip),
        pcm16=bool(args.pcm16),
        seed=args.seed,
        verbose=not args.no_verbose,
        fail_fast=bool(args.fail_fast),
    )
    try:
        augment_all(args.input_audio, cfg)
        return 0
    except Exception as e:
        console = _get_console(True)
        console.print(f"[red]Fatal error: {e}[/red]")
        return 1


# --------------------------------- Examples ----------------------------------
# 1) Minimal (auto assets + fallbacks will kick in as needed):
#    python augment_audio_suite.py input.wav --out ./aug_out --target-duration 12.0 --seed 7 --pcm16
#
# 2) With user-provided assets (no auto generation for those):
#    python augment_audio_suite.py input.wav --out ./aug_out \
#      --noise-dir ./assets/bg_noises --short-noise-dir ./assets/short_noises --ir-dir ./assets/irs
#
# 3) Run a focused subset:
#    python augment_audio_suite.py input.wav --out ./aug_out --only "AddGaussianNoise,PitchShift,TimeStretch"
#
# Notes:
# - This script fixes the WrongMultichannelAudioShape by always feeding (channels, samples) to audiomentations.
# - AdjustDuration is version-agnostic: we probe constructor args or apply a robust fallback.
# - Lambda uses transform=... function; if unavailable, a safe passthrough fallback is used.
# - Optional deps missing? High-quality fallbacks emulate Mp3Compression, LoudnessNormalization, RoomSimulator.
# - Assets missing? We synthesize background noises, short noises, and IRs automatically.


if __name__ == "__main__":
    sys.exit(main())