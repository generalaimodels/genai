#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generalized Video class for robust, flexible, and efficient video data handling.

Highlights:
- Universal loaders for many sources: local paths, file:// URIs, http(s), base64, bytes, NumPy, OpenCV,
  screen capture, camera, frame subsequences, clipboard, remote cloud storage (via fsspec),
  archives (zip/tar), IP/URL streams, TensorFlow/PyTorch tensors, HDF5 datasets, and raw binary files.
- User-level control over:
  - Time slicing: start_time/end_time (seconds), frame slicing: start_frame/end_frame, stride, max_frames.
  - Resolution: width/height resize with aspect handling and interpolation choice.
  - Cropping regions: x, y, width, height.
  - Color space: bgr/rgb/gray; dtype and normalization to [0, 1].
  - FPS resampling via frame index sampling or exact timestamps.
  - Progress bars and verbose metadata with the 'rich' module.
- Sub-clip operations that return new Video objects.
- Unified interface to iterate frames, get single frames, convert to NumPy/Torch/TensorFlow, and save.

Note:
- The implementation uses OpenCV (cv2) as the primary decoding backend.
- Optional integrations are supported when libraries are available: requests, fsspec, mss, h5py, torch, tensorflow, pyperclip, etc.
- When optional packages are missing for a method, a clear, actionable exception is raised.
- All examples below are safe to run; heavy/network examples are guarded to avoid failures in restricted environments.

Usage quick glance (see more examples at bottom under if __name__ == "__main__"):
    from pathlib import Path

    # Local file
    v = Video.from_local_path("example.mp4", verbose=True)
    v.describe()  # pretty metadata via rich
    arr = v.to_numpy(start_time=0.5, end_time=2.5, resize=(320, 240), color_space="rgb")

    # Sub-clip (by time)
    sub = v.subclip(start_time=1.0, end_time=3.0)
    sub.save("clip.mp4", fps=24, codec="mp4v", color_space="bgr")

    # Streaming camera (device 0)
    cam = Video.from_camera(0, verbose=True)
    for i, frame in enumerate(cam.iter_frames(max_frames=30, resize=(640, 480))):
        pass  # process frames here

    # Numpy frames
    import numpy as np
    frames = np.random.randint(0, 255, (60, 120, 160, 3), dtype=np.uint8)
    nv = Video.from_numpy(frames, fps=30)
    nv.save("random.mp4", fps=30)


Author: A meticulous, technically rigorous, and deeply detailed coding tutor.
"""

from __future__ import annotations

import base64
import contextlib
import io
import math
import os
import re
import sys
import tarfile
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterable, List, Literal, Optional, Tuple, Union
from urllib.parse import urlparse, unquote

# Hard dependencies (primary backend)
try:
    import cv2  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("OpenCV 'cv2' is required for this module. Please `pip install opencv-python`.") from e

# Soft/optional dependencies (loaded lazily)
try:
    import numpy as np  # widely available in most environments
except Exception as e:  # pragma: no cover
    raise RuntimeError("NumPy is required for this module. Please `pip install numpy`.") from e

# Optional rich console for verbose metadata and progress.
# If rich is unavailable, the module still works but with plain fallback.
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
    from rich import box
except Exception:  # pragma: no cover
    Console = None
    Table = None
    Panel = None
    Progress = None
    BarColumn = None
    TimeElapsedColumn = None
    TimeRemainingColumn = None
    MofNCompleteColumn = None
    box = None


# ---------------------------------------------
# Utilities and configuration
# ---------------------------------------------

def _console() -> Optional["Console"]:
    if Console is None:
        return None
    return Console()

def _rich_available() -> bool:
    return Console is not None

def _print_if_verbose(enabled: bool, renderable: Any) -> None:
    if enabled and _rich_available():
        _console().print(renderable)

def _interp_to_cv2(interp: str) -> int:
    interp = interp.lower()
    mapping = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "bilinear": cv2.INTER_LINEAR,
        "area": cv2.INTER_AREA,
        "cubic": cv2.INTER_CUBIC,
        "bicubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    return mapping.get(interp, cv2.INTER_AREA)

def _color_space_request_to_cv2_code(requested: str, src_bgr: bool = True) -> Optional[int]:
    requested = requested.lower()
    if requested in ("bgr", "rgb", "gray", "greyscale", "grayscale"):
        if requested == "bgr":
            return None if src_bgr else cv2.COLOR_RGB2BGR
        if requested == "rgb":
            return cv2.COLOR_BGR2RGB if src_bgr else None
        if requested in ("gray", "greyscale", "grayscale"):
            return cv2.COLOR_BGR2GRAY if src_bgr else cv2.COLOR_RGB2GRAY
    return None

def _mime_to_suffix(mime: str) -> str:
    table = {
        "video/mp4": ".mp4",
        "video/quicktime": ".mov",
        "video/x-msvideo": ".avi",
        "video/x-matroska": ".mkv",
        "video/webm": ".webm",
        "video/ogg": ".ogv",
        "application/octet-stream": ".bin",
    }
    return table.get(mime.lower(), ".mp4")

def _guess_suffix_from_path(path: str) -> str:
    # fallback when not base64 header; guess from URL or file path
    parsed = urlparse(path)
    candidate = Path(unquote(parsed.path)).suffix
    return candidate if candidate else ".mp4"

def _ensure_parent_dir(path: Path) -> None:
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

def _coerce_bool(x: Any, default: bool = False) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() in ("1", "true", "yes", "y", "on")
    return default

def _dtype_from_any(obj: Any, default: "np.dtype" = np.uint8) -> "np.dtype":
    try:
        return np.dtype(obj)
    except Exception:
        return default

def _parse_shape_with_optional_dtype(shape: Tuple) -> Tuple[Tuple[int, ...], "np.dtype"]:
    """
    Accepts:
      - (T, H, W, C)
      - (T, C, H, W)
      - (T, H, W, C, 'uint8') or dtype object at end
    Returns (shape_without_dtype, dtype)
    """
    if len(shape) >= 5:
        last = shape[-1]
        try:
            dtype = _dtype_from_any(last)
            dims = tuple(int(x) for x in shape[:-1])  # may raise ValueError if not ints
            return dims, dtype
        except Exception:
            pass
    # else assume dtype default
    dims = tuple(int(x) for x in shape)
    return dims, np.uint8

def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default

def _fourcc_to_string(fourcc_float: float) -> str:
    try:
        fourcc_int = int(fourcc_float)
        chars = [
            chr((fourcc_int >> 0) & 0xFF),
            chr((fourcc_int >> 8) & 0xFF),
            chr((fourcc_int >> 16) & 0xFF),
            chr((fourcc_int >> 24) & 0xFF),
        ]
        return "".join(chars)
    except Exception:
        return "unknown"

def _is_streaming_url(url: str) -> bool:
    # Heuristics for URL streams
    url = url.lower()
    return any(p in url for p in (".m3u8", "rtsp://", "rtsps://", "rtmp://", "udp://", "tcp://"))

def _time_to_frame_index(time_sec: float, fps: float) -> int:
    return int(round(time_sec * fps))

@contextlib.contextmanager
def _open_tempfile_with_bytes(data: bytes, suffix: str = ".mp4") -> Generator[Path, None, None]:
    """
    Writes bytes to a temporary file and yields its Path.
    Note: For a long-lived Video object, we often need the temp file to survive beyond this context;
          the Video class keeps its own temp files list and cleans up in __del__.
    """
    fd, name = tempfile.mkstemp(suffix=suffix, prefix="video_tmp_")
    os.close(fd)
    p = Path(name)
    with p.open("wb") as f:
        f.write(data)
    try:
        yield p
    finally:
        # The caller may choose to unlink later. We don't delete here because the caller
        # might need the file longer than this context.
        pass

def _requests_download(url: str) -> bytes:
    try:
        import requests  # type: ignore
    except Exception as e:
        raise RuntimeError("`requests` is required to download http(s) resources. Please `pip install requests`.") from e

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        chunks = []
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                chunks.append(chunk)
        return b"".join(chunks)

def _fsspec_read(uri: str) -> bytes:
    try:
        import fsspec  # type: ignore
    except Exception as e:
        raise RuntimeError("`fsspec` is required to read remote storage URIs. Please `pip install fsspec[s3,gcs,abfs]`.") from e
    with fsspec.open(uri, "rb") as f:
        return f.read()

def _clipboard_text() -> Optional[str]:
    # Try pyperclip first
    try:
        import pyperclip  # type: ignore
        txt = pyperclip.paste()
        if isinstance(txt, str) and txt.strip():
            return txt.strip()
    except Exception:
        pass
    # Try Tkinter
    try:
        import tkinter as tk  # type: ignore
        r = tk.Tk()
        r.withdraw()
        txt = r.clipboard_get()
        r.destroy()
        if isinstance(txt, str) and txt.strip():
            return txt.strip()
    except Exception:
        pass
    return None

def _ensure_tuple_resize(resize: Optional[Union[Tuple[int, int], Tuple[None, None]]]) -> Optional[Tuple[Optional[int], Optional[int]]]:
    if resize is None:
        return None
    if isinstance(resize, (tuple, list)) and len(resize) == 2:
        w, h = resize
        return (int(w) if w is not None else None, int(h) if h is not None else None)
    raise ValueError("resize must be a tuple (width, height) or None")

def _safe_sleep(seconds: float) -> None:
    if seconds <= 0:
        return
    try:
        time.sleep(seconds)
    except Exception:
        pass

def _np_color_channels(arr: np.ndarray) -> int:
    if arr.ndim == 3:
        return 1 if arr.shape[2] in (1,) else arr.shape[2]
    if arr.ndim == 2:
        return 1
    if arr.ndim == 4:
        return 1 if arr.shape[-1] in (1,) else arr.shape[-1]
    return 3

def _cv2_read_capture_meta(cap: "cv2.VideoCapture") -> Dict[str, Any]:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fourcc = _fourcc_to_string(cap.get(cv2.CAP_PROP_FOURCC) or 0)
    duration = float(frame_count / fps) if fps > 0 and frame_count > 0 else None
    return {
        "width": width,
        "height": height,
        "fps": fps if fps > 0 else None,
        "frame_count": frame_count if frame_count > 0 else None,
        "fourcc": fourcc,
        "duration": duration,
    }

def _coerce_rgb(arr: np.ndarray, src_is_bgr: bool, color_space: str) -> np.ndarray:
    code = _color_space_request_to_cv2_code(color_space, src_bgr=src_is_bgr)
    if code is None:
        return arr
    out = cv2.cvtColor(arr, code)
    if out.ndim == 2:
        out = out[:, :, None]  # standardize shape for gray
    return out

def _normalize_to_dtype(arr: np.ndarray, normalize: bool, dtype: Optional["np.dtype"]) -> np.ndarray:
    if normalize:
        arr = arr.astype(np.float32, copy=False) / 255.0
    if dtype is not None:
        target = np.dtype(dtype)
        if arr.dtype != target:
            arr = arr.astype(target, copy=False)
    return arr

def _resize_frame(frame: np.ndarray, resize: Optional[Tuple[Optional[int], Optional[int]]], keep_aspect: bool, interp: str) -> np.ndarray:
    if resize is None:
        return frame
    w, h = resize
    if w is None and h is None:
        return frame
    if w is not None and h is not None and not keep_aspect:
        return cv2.resize(frame, (w, h), interpolation=_interp_to_cv2(interp))
    # keep aspect
    H, W = frame.shape[:2]
    if w is None:
        scale = h / float(H)
        newW = max(1, int(round(W * scale)))
        newH = h
    elif h is None:
        scale = w / float(W)
        newW = w
        newH = max(1, int(round(H * scale)))
    else:
        # both provided -> letterbox within
        scale = min(w / float(W), h / float(H))
        newW = max(1, int(round(W * scale)))
        newH = max(1, int(round(H * scale)))
    resized = cv2.resize(frame, (newW, newH), interpolation=_interp_to_cv2(interp))
    if w is not None and h is not None and (newW != w or newH != h):
        # pad to target (letterbox) with black
        canvas = np.zeros((h, w, resized.shape[2]), dtype=resized.dtype)
        off_x = (w - newW) // 2
        off_y = (h - newH) // 2
        canvas[off_y:off_y + newH, off_x:off_x + newW] = resized
        return canvas
    return resized

def _crop_frame(frame: np.ndarray, crop: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    if not crop:
        return frame
    x, y, w, h = crop
    H, W = frame.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x0 >= x1 or y0 >= y1:
        return frame  # invalid crop -> ignore
    return frame[y0:y1, x0:x1]

def _valid_frame_index(i: int, total: Optional[int]) -> bool:
    if total is None:
        return i >= 0
    return 0 <= i < total

def _ensure_4d(frames: np.ndarray) -> np.ndarray:
    if frames.ndim == 3:
        return frames[None, ...]
    if frames.ndim == 2:
        return frames[None, :, :, None]
    if frames.ndim == 4:
        return frames
    raise ValueError("Frames ndarray must be (T,H,W,C) or (H,W,C) or (H,W).")

def _maybe_to_uint8(frames: np.ndarray) -> np.ndarray:
    if frames.dtype == np.uint8:
        return frames
    if issubclass(frames.dtype.type, np.floating):
        frames = np.clip(frames, 0.0, 1.0)
        return (frames * 255.0 + 0.5).astype(np.uint8)
    frames = frames.astype(np.uint8, copy=False)
    return frames

# ---------------------------------------------
# Errors and Config
# ---------------------------------------------

class VideoError(Exception):
    """Custom exception for Video errors."""

@dataclass
class VideoMeta:
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    frame_count: Optional[int] = None
    duration: Optional[float] = None
    codec: Optional[str] = None
    source: Optional[str] = None
    colorspace: str = "bgr"  # bgr (opencv default), rgb, gray
    backend: str = "opencv"

@dataclass
class DecodeOptions:
    # Frame/time range
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    stride: int = 1
    max_frames: Optional[int] = None
    timestamps: Optional[List[float]] = None  # explicit seconds at which to sample one frame

    # Geometry, color, dtype
    resize: Optional[Tuple[Optional[int], Optional[int]]] = None  # (width, height), each can be None
    keep_aspect: bool = True
    interpolation: str = "area"
    crop: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    color_space: str = "bgr"
    normalize: bool = False
    dtype: Optional["np.dtype"] = None

    # Resampling
    target_fps: Optional[float] = None  # if provided, sample to approximate this FPS, requires known src fps

    # UX
    progress: bool = False
    return_timestamps: bool = False

@dataclass
class VideoConfig:
    verbose: bool = True
    tmp_dir: Optional[Path] = None

# ---------------------------------------------
# Core Video class
# ---------------------------------------------

class Video:
    """
    A unified, feature-rich interface for working with video content regardless of the source.

    Instances represent a video "source", not necessarily all frames loaded into memory.
    Decode on demand via iter_frames()/to_numpy()/get_frame(), with powerful user controls.

    Important notes:
    - colorspace defaults to "bgr" to match OpenCV. Explicitly request "rgb" if needed.
    - For streaming sources (camera, url streams, screen capture), random access seeking is not supported.
      Use sequential reading and sampling.
    - Many loaders may store a temporary file on disk to allow OpenCV decoding. The Video object cleans up
      its own temporary files on destruction.

    Common operations:
    - describe(): prints metadata using rich (if verbose=True).
    - iter_frames(...): generator with controlled decoding.
    - get_frame(index=..., time=...): single frame extraction.
    - to_numpy(...): decode into a NumPy array in memory.
    - subclip(...): return a new Video (in-memory by default) representing a segment.
    - save(path, ...): write frames to a new video file with chosen codec/fps.
    """

    # ------------------ Construction ------------------

    def __init__(
        self,
        source_type: str,
        meta: VideoMeta,
        *,
        path: Optional[Path] = None,
        url: Optional[str] = None,
        capture: Optional["cv2.VideoCapture"] = None,
        frames: Optional[np.ndarray] = None,
        live: bool = False,
        producer: Optional[Callable[..., Iterable[np.ndarray]]] = None,
        config: Optional[VideoConfig] = None,
        temp_paths: Optional[List[Path]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.source_type = source_type  # e.g., 'path', 'url', 'numpy', 'opencv', 'screen', 'camera', ...
        self.meta = meta
        self._path = path
        self._url = url
        self._capture = capture
        self._frames = frames  # (T,H,W,C) in memory, uint8 or float
        self._live = live  # streaming/live source (camera/screen/url_stream)
        self._producer = producer  # for screen capture or custom generator
        self._owned_capture = False
        self._temp_paths: List[Path] = temp_paths or []
        self._config = config or VideoConfig()
        self._extra = extra or {}

        if self._config.verbose:
            self._verbose_initial_report()

    # ------------- Class methods (loaders) -------------

    @classmethod
    def from_local_path(cls, path: str, *, verbose: bool = True) -> "Video":
        p = Path(path).expanduser().resolve()
        if not p.exists():
            raise VideoError(f"File not found: {p}")
        meta = cls._probe_path(p)
        meta.source = str(p)
        cfg = VideoConfig(verbose=verbose)
        return cls("path", meta, path=p, config=cfg)

    @classmethod
    def from_file_uri(cls, uri: str, *, verbose: bool = True) -> "Video":
        parsed = urlparse(uri)
        if parsed.scheme != "file":
            raise VideoError(f"Invalid file URI scheme (expected file://): {uri}")
        p = Path(unquote(parsed.path)).expanduser()
        if not p.is_absolute():
            # Handle cases like file://relative/path which are unusual. Normalize if possible.
            p = p.resolve()
        return cls.from_local_path(str(p), verbose=verbose)

    @classmethod
    def from_http_url(cls, url: str, *, verbose: bool = True) -> "Video":
        """
        Download and decode an http(s) resource as a full file.
        For real-time streams (RTSP/RTMP/m3u8), use from_url_stream.
        """
        if _is_streaming_url(url):
            raise VideoError("Detected streaming URL. Use from_url_stream(url) for live streams.")
        data = _requests_download(url)
        suffix = _guess_suffix_from_path(url)
        temp_path = cls._bytes_to_tempfile(data, suffix)
        meta = cls._probe_path(temp_path)
        meta.source = url
        return cls("http", meta, path=temp_path, url=url, config=VideoConfig(verbose=verbose), temp_paths=[temp_path])

    @classmethod
    def from_base64(cls, data: str, *, verbose: bool = True) -> "Video":
        """
        Accepts raw base64 or data URLs like: data:video/mp4;base64,AAA...
        """
        match = re.match(r"^data:(?P<mime>[^;]+);base64,(?P<b64>.+)$", data, flags=re.IGNORECASE | re.DOTALL)
        if match:
            mime = match.group("mime")
            b64 = match.group("b64")
            suffix = _mime_to_suffix(mime)
            raw = base64.b64decode(b64)
        else:
            suffix = ".mp4"
            raw = base64.b64decode(data)
        temp_path = cls._bytes_to_tempfile(raw, suffix)
        meta = cls._probe_path(temp_path)
        meta.source = "base64"
        return cls("base64", meta, path=temp_path, config=VideoConfig(verbose=verbose), temp_paths=[temp_path])

    @classmethod
    def from_bytes(cls, data: bytes, *, verbose: bool = True, suffix: str = ".mp4") -> "Video":
        temp_path = cls._bytes_to_tempfile(data, suffix)
        meta = cls._probe_path(temp_path)
        meta.source = "bytes"
        return cls("bytes", meta, path=temp_path, config=VideoConfig(verbose=verbose), temp_paths=[temp_path])

    @classmethod
    def from_numpy(cls, frames: np.ndarray, *, fps: Optional[float] = 30.0, color_space: str = "rgb", verbose: bool = True) -> "Video":
        frames = _ensure_4d(frames)
        H, W, C = frames.shape[1], frames.shape[2], frames.shape[3]
        meta = VideoMeta(width=W, height=H, fps=fps, frame_count=frames.shape[0], duration=(frames.shape[0] / fps if fps else None), colorspace=color_space, source="numpy")
        return cls("numpy", meta, frames=frames, config=VideoConfig(verbose=verbose))

    @classmethod
    def from_opencv(cls, cv2_capture: "cv2.VideoCapture", *, verbose: bool = True) -> "Video":
        if not isinstance(cv2_capture, cv2.VideoCapture):
            raise VideoError("from_opencv expects a cv2.VideoCapture instance.")
        if not cv2_capture.isOpened():
            raise VideoError("Provided cv2.VideoCapture is not opened.")
        meta_info = _cv2_read_capture_meta(cv2_capture)
        meta = VideoMeta(
            width=meta_info["width"],
            height=meta_info["height"],
            fps=meta_info["fps"],
            frame_count=meta_info["frame_count"],
            duration=meta_info["duration"],
            codec=meta_info["fourcc"],
            colorspace="bgr",
            source="opencv",
        )
        vid = cls("opencv", meta, capture=cv2_capture, config=VideoConfig(verbose=verbose))
        vid._owned_capture = False  # do not close capture on delete since user provided it
        return vid

    @classmethod
    def from_screen_capture(cls, os: str = "", *, region: Optional[Tuple[int, int, int, int]] = None, verbose: bool = True) -> "Video":
        """
        Screen capture via mss (if available).
        region: optional (x,y,w,h)
        """
        try:
            import mss  # type: ignore
            import mss.tools  # type: ignore
        except Exception as e:
            raise VideoError("Screen capture requires `mss`. Please `pip install mss`.") from e

        monitor_region = region  # use region as provided
        # producer yields frames in RGB
        def producer() -> Iterable[np.ndarray]:
            with mss.mss() as sct:
                mon = sct.monitors[1]  # full screen default
                if monitor_region:
                    x, y, w, h = monitor_region
                    mon = {"top": y, "left": x, "width": w, "height": h}
                while True:
                    img = sct.grab(mon)  # raw BGRA
                    frame = np.array(img)  # (H, W, 4)
                    frame = frame[:, :, :3]  # drop alpha
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR) if frame.shape[2] == 4 else frame
                    yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # produce RGB frames

        # Meta unknown; will infer on first frame read in iter, set placeholder
        meta = VideoMeta(width=None, height=None, fps=None, frame_count=None, duration=None, codec=None, colorspace="rgb", source="screen")
        return cls("screen", meta, live=True, producer=producer, config=VideoConfig(verbose=verbose))

    @classmethod
    def from_camera(cls, device_index: int = 0, *, verbose: bool = True) -> "Video":
        cap = cv2.VideoCapture(device_index)
        if not cap.isOpened():
            raise VideoError(f"Cannot open camera device {device_index}.")
        meta_info = _cv2_read_capture_meta(cap)
        meta = VideoMeta(
            width=meta_info["width"],
            height=meta_info["height"],
            fps=meta_info["fps"],
            frame_count=None,  # live
            duration=None,
            codec=meta_info["fourcc"],
            colorspace="bgr",
            source=f"camera:{device_index}",
        )
        vid = cls("camera", meta, capture=cap, live=True, config=VideoConfig(verbose=verbose))
        vid._owned_capture = True
        return vid

    @classmethod
    def from_video_frame_sequence(cls, video_path: str, frame_range: Tuple[int, int], *, verbose: bool = True) -> "Video":
        start, end = frame_range
        base = cls.from_local_path(video_path, verbose=False)
        frames = base.to_numpy(start_frame=start, end_frame=end, color_space="rgb")
        meta = VideoMeta(width=frames.shape[2], height=frames.shape[1], fps=base.meta.fps, frame_count=frames.shape[0], duration=(frames.shape[0] / base.meta.fps if base.meta.fps else None), colorspace="rgb", source=f"{video_path}[{start}:{end}]")
        return cls("numpy", meta, frames=frames, config=VideoConfig(verbose=verbose))

    @classmethod
    def from_clipboard(cls, *, verbose: bool = True) -> "Video":
        txt = _clipboard_text()
        if not txt:
            raise VideoError("Clipboard does not contain usable content. Expected a file path, URL, file:// URI, or base64 data URL.")
        # Decide what it is
        s = txt.strip()
        if s.startswith("file://"):
            return cls.from_file_uri(s, verbose=verbose)
        if s.startswith("http://") or s.startswith("https://"):
            # Downloaded file resource
            return cls.from_http_url(s, verbose=verbose)
        if s.startswith("data:") and ";base64," in s:
            return cls.from_base64(s, verbose=verbose)
        # Plain path?
        p = Path(s).expanduser()
        if p.exists():
            return cls.from_local_path(str(p), verbose=verbose)
        # Maybe base64 raw
        try:
            base64.b64decode(s)
            return cls.from_base64(s, verbose=verbose)
        except Exception:
            pass
        raise VideoError("Clipboard content not recognized as a valid video path/URL/base64.")

    @classmethod
    def from_remote_storage(cls, uri: str, *, verbose: bool = True) -> "Video":
        """
        Supports URIs via fsspec: s3://bucket/key, gs://bucket/key, az://container/blob, abfs://, etc.
        """
        data = _fsspec_read(uri)
        suffix = _guess_suffix_from_path(uri)
        temp_path = cls._bytes_to_tempfile(data, suffix)
        meta = cls._probe_path(temp_path)
        meta.source = uri
        return cls("remote", meta, path=temp_path, config=VideoConfig(verbose=verbose), temp_paths=[temp_path])

    @classmethod
    def from_zip(cls, archive_path: str, filename: str, *, verbose: bool = True) -> "Video":
        p = Path(archive_path).expanduser().resolve()
        if not p.exists():
            raise VideoError(f"Zip archive not found: {p}")
        with zipfile.ZipFile(p, "r") as zf:
            if filename not in zf.namelist():
                raise VideoError(f"File '{filename}' not found inside zip archive.")
            data = zf.read(filename)
        suffix = Path(filename).suffix or ".mp4"
        temp_path = cls._bytes_to_tempfile(data, suffix)
        meta = cls._probe_path(temp_path)
        meta.source = f"zip://{p}!/{filename}"
        return cls("zip", meta, path=temp_path, config=VideoConfig(verbose=verbose), temp_paths=[temp_path])

    @classmethod
    def from_tar(cls, tar_path: str, member: str, *, verbose: bool = True) -> "Video":
        p = Path(tar_path).expanduser().resolve()
        if not p.exists():
            raise VideoError(f"Tar archive not found: {p}")
        with tarfile.open(p, "r:*") as tf:
            try:
                member_info = tf.getmember(member)
            except KeyError:
                raise VideoError(f"Member '{member}' not found in tar archive.") from None
            data = tf.extractfile(member_info).read()  # type: ignore
        suffix = Path(member).suffix or ".mp4"
        temp_path = cls._bytes_to_tempfile(data, suffix)
        meta = cls._probe_path(temp_path)
        meta.source = f"tar://{p}!/{member}"
        return cls("tar", meta, path=temp_path, config=VideoConfig(verbose=verbose), temp_paths=[temp_path])

    @classmethod
    def from_url_stream(cls, url: str, *, verbose: bool = True) -> "Video":
        if not _is_streaming_url(url) and not url.lower().startswith(("http://", "https://")):
            raise VideoError("from_url_stream expects a streaming URL (rtsp/rtmp/m3u8/udp/tcp) or an HTTP URL that serves a stream.")
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            raise VideoError(f"Cannot open stream: {url}")
        meta_info = _cv2_read_capture_meta(cap)
        meta = VideoMeta(
            width=meta_info["width"],
            height=meta_info["height"],
            fps=meta_info["fps"],
            frame_count=None,
            duration=None,
            codec=meta_info["fourcc"],
            colorspace="bgr",
            source=f"stream:{url}",
        )
        vid = cls("stream", meta, capture=cap, live=True, config=VideoConfig(verbose=verbose))
        vid._owned_capture = True
        return vid

    @classmethod
    def from_tensorflow_tensor(cls, tf_tensor: Any, *, verbose: bool = True, color_space: str = "rgb", fps: Optional[float] = 30.0) -> "Video":
        # Accepts EagerTensor or numpy array-like with shape (T,H,W,C) or (T,C,H,W)
        try:
            import tensorflow as tf  # type: ignore
            if isinstance(tf_tensor, tf.Tensor):
                arr = tf_tensor.numpy()
            else:
                arr = np.array(tf_tensor)
        except Exception:
            # If TF not installed or not a tensor, try generic numpy conversion
            arr = np.array(tf_tensor)
        arr = _ensure_4d(arr)
        if arr.shape[1] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            # Probably (T,C,H,W) -> (T,H,W,C)
            arr = np.transpose(arr, (0, 2, 3, 1))
        meta = VideoMeta(width=arr.shape[2], height=arr.shape[1], fps=fps, frame_count=arr.shape[0], duration=(arr.shape[0] / fps if fps else None), colorspace=color_space, source="tensorflow")
        return cls("numpy", meta, frames=arr, config=VideoConfig(verbose=verbose))

    @classmethod
    def from_torch_tensor(cls, torch_tensor: Any, *, verbose: bool = True, color_space: str = "rgb", fps: Optional[float] = 30.0) -> "Video":
        try:
            import torch  # type: ignore
            if isinstance(torch_tensor, torch.Tensor):
                arr = torch_tensor.detach().cpu().numpy()
            else:
                arr = np.array(torch_tensor)
        except Exception:
            arr = np.array(torch_tensor)
        arr = _ensure_4d(arr)
        # Common PyTorch layout (T,C,H,W)
        if arr.shape[1] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (0, 2, 3, 1))
        meta = VideoMeta(width=arr.shape[2], height=arr.shape[1], fps=fps, frame_count=arr.shape[0], duration=(arr.shape[0] / fps if fps else None), colorspace=color_space, source="torch")
        return cls("numpy", meta, frames=arr, config=VideoConfig(verbose=verbose))

    @classmethod
    def from_hdf5(cls, hdf_path: str, dataset: str, *, verbose: bool = True, color_space: str = "rgb", fps: Optional[float] = 30.0) -> "Video":
        try:
            import h5py  # type: ignore
        except Exception as e:
            raise VideoError("from_hdf5 requires `h5py`. Please `pip install h5py`.") from e
        p = Path(hdf_path).expanduser().resolve()
        if not p.exists():
            raise VideoError(f"HDF5 file not found: {p}")
        with h5py.File(p, "r") as f:
            if dataset not in f:
                raise VideoError(f"Dataset '{dataset}' not found in HDF5 file.")
            arr = f[dataset][...]  # load into memory
        arr = _ensure_4d(np.array(arr))
        # Try to fix layout if (T,C,H,W)
        if arr.shape[1] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (0, 2, 3, 1))
        meta = VideoMeta(width=arr.shape[2], height=arr.shape[1], fps=fps, frame_count=arr.shape[0], duration=(arr.shape[0] / fps if fps else None), colorspace=color_space, source=f"hdf5:{p}:{dataset}")
        return cls("numpy", meta, frames=arr, config=VideoConfig(verbose=verbose))

    @classmethod
    def from_binary_file(cls, bin_path: str, shape: Tuple, *, verbose: bool = True, color_space: str = "rgb", fps: Optional[float] = 30.0) -> "Video":
        p = Path(bin_path).expanduser().resolve()
        if not p.exists():
            raise VideoError(f"Binary file not found: {p}")
        dims, dtype = _parse_shape_with_optional_dtype(shape)
        arr = np.fromfile(str(p), dtype=dtype)
        expected = int(np.prod(dims))
        if arr.size != expected:
            raise VideoError(f"Binary size mismatch: expected {expected} elements, got {arr.size}.")
        arr = arr.reshape(dims)
        arr = _ensure_4d(arr)
        if arr.shape[1] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (0, 2, 3, 1))
        meta = VideoMeta(width=arr.shape[2], height=arr.shape[1], fps=fps, frame_count=arr.shape[0], duration=(arr.shape[0] / fps if fps else None), colorspace=color_space, source=f"binary:{p}")
        return cls("numpy", meta, frames=arr, config=VideoConfig(verbose=verbose))

    # ------------------- Core API ----------------------

    def describe(self) -> None:
        """Pretty-print metadata for this video (rich)."""
        if not _rich_available():
            print(self.meta)
            return
        table = Table(title="Video Metadata", box=box.SIMPLE_HEAVY)
        table.add_column("Field", justify="right")
        table.add_column("Value", overflow="fold")
        info = {
            "source_type": self.source_type,
            "source": self.meta.source or "",
            "backend": self.meta.backend,
            "codec": self.meta.codec or "",
            "width": str(self.meta.width or "unknown"),
            "height": str(self.meta.height or "unknown"),
            "fps": f"{self.meta.fps:.3f}" if self.meta.fps else "unknown",
            "frame_count": str(self.meta.frame_count or "unknown"),
            "duration_sec": f"{self.meta.duration:.3f}" if self.meta.duration else "unknown",
            "colorspace": self.meta.colorspace,
            "live": str(self._live),
        }
        for k, v in info.items():
            table.add_row(k, str(v))
        _console().print(table)

    def iter_frames(
        self,
        *,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        stride: int = 1,
        max_frames: Optional[int] = None,
        timestamps: Optional[List[float]] = None,
        resize: Optional[Tuple[Optional[int], Optional[int]]] = None,
        keep_aspect: bool = True,
        interpolation: str = "area",
        crop: Optional[Tuple[int, int, int, int]] = None,
        color_space: str = "bgr",
        normalize: bool = False,
        dtype: Optional["np.dtype"] = None,
        target_fps: Optional[float] = None,
        progress: bool = False,
        return_timestamps: bool = False,
    ) -> Iterable[Union[np.ndarray, Tuple[np.ndarray, float]]]:
        """
        Iterates frames with fine-grained control. Yields frames as NumPy arrays (H,W,C).
        If return_timestamps=True, yields (frame, timestamp_sec).

        Notes:
        - For live sources, only sequential reading is supported; timestamps/target_fps/start/end seeking are best-effort.
        - For file sources, random access is used when needed (timestamps or resampling), leveraging CAP_PROP_POS_FRAMES (or POS_MSEC).
        """
        opts = DecodeOptions(
            start_time=start_time,
            end_time=end_time,
            start_frame=start_frame,
            end_frame=end_frame,
            stride=stride,
            max_frames=max_frames,
            timestamps=timestamps[:] if timestamps else None,
            resize=_ensure_tuple_resize(resize),
            keep_aspect=keep_aspect,
            interpolation=interpolation,
            crop=crop,
            color_space=color_space,
            normalize=normalize,
            dtype=_dtype_from_any(dtype) if dtype is not None else None,
            target_fps=target_fps,
            progress=progress,
            return_timestamps=return_timestamps,
        )

        # Routing per source type
        if self.source_type in ("numpy",):
            yield from self._iter_frames_numpy(opts)
        elif self.source_type in ("path", "http", "bytes", "base64", "zip", "tar", "remote"):
            yield from self._iter_frames_file(opts)
        elif self.source_type in ("opencv", "camera", "stream"):
            yield from self._iter_frames_capture(opts)
        elif self.source_type in ("screen",):
            yield from self._iter_frames_screen(opts)
        else:
            raise VideoError(f"iter_frames not supported for source_type={self.source_type}.")

    def to_numpy(self, **kwargs: Any) -> np.ndarray:
        """
        Decode frames into a NumPy array. Accepts same kwargs as iter_frames.
        Returns array of shape (T, H, W, C).
        """
        frames: List[np.ndarray] = []
        timestamps: List[float] = []
        for item in self.iter_frames(**kwargs, return_timestamps=True):
            frame, ts = item  # type: ignore
            frames.append(frame)
            timestamps.append(ts)
        if not frames:
            return np.empty((0, 0, 0, 3), dtype=np.uint8)
        arr = np.stack(frames, axis=0)
        # Optionally attach timestamps
        self._extra["last_to_numpy_timestamps"] = timestamps
        return arr

    def get_frame(self, *, index: Optional[int] = None, time_sec: Optional[float] = None, color_space: str = "bgr", resize: Optional[Tuple[Optional[int], Optional[int]]] = None, interpolation: str = "area") -> np.ndarray:
        """
        Extract a single frame by index or timestamp (seconds).
        """
        if index is None and time_sec is None:
            raise VideoError("Provide either index or time_sec.")
        if self.source_type == "numpy":
            arr = self._frames  # type: ignore
            if index is None:
                if self.meta.fps is None:
                    raise VideoError("Unknown fps; cannot map time to frame index.")
                index = _time_to_frame_index(time_sec, self.meta.fps)
            if not _valid_frame_index(index, arr.shape[0]):
                raise VideoError(f"Frame index out of range: {index}")
            frame = arr[index]
            frame = _coerce_rgb(frame, src_is_bgr=(self.meta.colorspace == "bgr"), color_space=color_space)
            frame = _resize_frame(frame, _ensure_tuple_resize(resize), keep_aspect=True, interp=interpolation)
            return frame

        # File/capture-based
        if time_sec is not None and self.meta.fps is not None:
            index = _time_to_frame_index(time_sec, self.meta.fps)
        if index is None:
            # fallback using pos_msec if fps unknown
            with self._open_capture() as cap:
                cap.set(cv2.CAP_PROP_POS_MSEC, float(time_sec) * 1000.0)  # type: ignore
                ok, frame = cap.read()
                if not ok:
                    raise VideoError("Failed to read frame at requested time.")
                frame = _coerce_rgb(frame, src_is_bgr=True, color_space=color_space)
                frame = _resize_frame(frame, _ensure_tuple_resize(resize), keep_aspect=True, interp=interpolation)
                return frame

        # By index
        with self._open_capture() as cap:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(index))
            ok, frame = cap.read()
            if not ok:
                raise VideoError(f"Failed to read frame index {index}.")
            frame = _coerce_rgb(frame, src_is_bgr=True, color_space=color_space)
            frame = _resize_frame(frame, _ensure_tuple_resize(resize), keep_aspect=True, interp=interpolation)
            return frame

    def subclip(
        self,
        *,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        stride: int = 1,
        resize: Optional[Tuple[Optional[int], Optional[int]]] = None,
        keep_aspect: bool = True,
        interpolation: str = "area",
        crop: Optional[Tuple[int, int, int, int]] = None,
        color_space: str = "rgb",
        normalize: bool = False,
        dtype: Optional["np.dtype"] = None,
        target_fps: Optional[float] = None,
        max_frames: Optional[int] = None,
        progress: bool = False,
    ) -> "Video":
        """
        Returns a new in-memory Video containing the specified segment and transformations.
        This is robust and convenient for downstream operations.
        """
        arr = self.to_numpy(
            start_time=start_time,
            end_time=end_time,
            start_frame=start_frame,
            end_frame=end_frame,
            stride=stride,
            max_frames=max_frames,
            resize=resize,
            keep_aspect=keep_aspect,
            interpolation=interpolation,
            crop=crop,
            color_space=color_space,
            normalize=normalize,
            dtype=dtype,
            target_fps=target_fps,
            progress=progress,
        )
        # fps handling for new clip
        fps = self.meta.fps
        if target_fps is not None:
            fps = target_fps
        meta = VideoMeta(
            width=(arr.shape[2] if arr.ndim == 4 else None),
            height=(arr.shape[1] if arr.ndim == 4 else None),
            fps=fps,
            frame_count=(arr.shape[0] if arr.ndim == 4 else None),
            duration=(arr.shape[0] / fps if (arr.ndim == 4 and fps) else None),
            colorspace=color_space,
            source=f"{self.meta.source or self.source_type}[subclip]",
        )
        return Video("numpy", meta, frames=arr, config=VideoConfig(verbose=self._config.verbose))

    def save(
        self,
        path: str,
        *,
        fps: Optional[float] = None,
        codec: str = "mp4v",
        color_space: str = "bgr",
        quality: Optional[int] = None,
        **iter_kwargs: Any,
    ) -> Path:
        """
        Save the decoded frames to a video file.

        Args:
            path: Output file path.
            fps: frames per second for the output. Defaults to self.meta.fps or 30 if unknown.
            codec: fourcc code (e.g., 'mp4v', 'avc1', 'XVID', 'MJPG').
            color_space: expected codec color space (bgr or rgb). Frames will be converted to match writer requirements.
            quality: For certain codecs, setting CAP_PROP_QUALITY may help (OpenCV support varies).
            iter_kwargs: Any iter_frames arguments to control what frames are saved.

        Returns:
            pathlib.Path to the written file.
        """
        out_path = Path(path).expanduser()
        _ensure_parent_dir(out_path)
        # Determine resolution from first frame
        iterator = self.iter_frames(color_space=color_space, **iter_kwargs)
        first = None
        try:
            first = next(iter(iterator))
        except StopIteration:
            raise VideoError("No frames to save.")
        if isinstance(first, tuple):
            first = first[0]
        H, W = first.shape[:2]
        # Determine fps
        use_fps = fps or self.meta.fps or 30.0
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(out_path), fourcc, use_fps, (W, H), isColor=(first.shape[2] != 1))

        if not writer.isOpened():
            raise VideoError(f"Failed to open VideoWriter for {out_path} with codec '{codec}'.")

        # Write first frame, ensuring BGR for writer
        f0 = first
        if color_space.lower() == "rgb":
            f0 = cv2.cvtColor(f0, cv2.COLOR_RGB2BGR)
        elif color_space.lower() in ("gray", "grayscale", "greyscale") and f0.shape[2] == 1:
            f0 = f0[:, :, 0]
        writer.write(_maybe_to_uint8(f0))

        count = 1
        # Continue with the rest
        for item in iterator:
            if isinstance(item, tuple):
                frame = item[0]
            else:
                frame = item
            if color_space.lower() == "rgb":
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif color_space.lower() in ("gray", "grayscale", "greyscale") and frame.shape[2] == 1:
                frame = frame[:, :, 0]
            writer.write(_maybe_to_uint8(frame))
            count += 1

        writer.release()

        if self._config.verbose and _rich_available():
            panel = Panel.fit(f"Saved {count} frames to {out_path} at {use_fps} FPS with codec={codec}.", title="Save Complete")
            _console().print(panel)
        return out_path

    # ---------------- Internal mechanics ----------------

    def _iter_frames_numpy(self, opts: DecodeOptions) -> Iterable[Union[np.ndarray, Tuple[np.ndarray, float]]]:
        arr = _ensure_4d(self._frames)  # type: ignore
        T = arr.shape[0]
        # Determine index range
        if opts.timestamps:
            # Only possible if meta.fps known; else approximate equally
            if self.meta.fps is None:
                raise VideoError("Timestamps requested but fps unknown for numpy source.")
            indices = [min(max(_time_to_frame_index(t, self.meta.fps), 0), T - 1) for t in opts.timestamps]
        else:
            start_idx = opts.start_frame if opts.start_frame is not None else 0
            end_idx = opts.end_frame if opts.end_frame is not None else (T - 1)
            if opts.start_time is not None:
                if self.meta.fps is None:
                    raise VideoError("start_time/end_time require known fps.")
                start_idx = max(start_idx, _time_to_frame_index(opts.start_time, self.meta.fps))
            if opts.end_time is not None:
                if self.meta.fps is None:
                    raise VideoError("start_time/end_time require known fps.")
                end_idx = min(end_idx, _time_to_frame_index(opts.end_time, self.meta.fps))
            indices = list(range(max(0, start_idx), min(T - 1, end_idx) + 1, max(1, opts.stride)))

            if opts.target_fps is not None and self.meta.fps is not None and opts.target_fps > 0:
                # Resample indices to target_fps
                step = self.meta.fps / opts.target_fps
                resampled = []
                i = indices[0] if indices else 0
                stop = indices[-1] if indices else -1
                while i <= stop:
                    resampled.append(int(round(i)))
                    i += step
                indices = sorted(set([j for j in resampled if 0 <= j < T]))

        if opts.max_frames is not None:
            indices = indices[:opts.max_frames]

        progress = None
        task_id = None
        if opts.progress and _rich_available():
            progress = Progress(
                "[progress.description]{task.description}",
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                transient=True,
            )
            progress.start()
            task_id = progress.add_task("Decoding (numpy)", total=len(indices))

        for k, idx in enumerate(indices):
            frame = arr[idx]
            ts = (idx / self.meta.fps) if self.meta.fps else float(k)
            # Transform
            frame = _crop_frame(frame, opts.crop)
            frame = _resize_frame(frame, opts.resize, keep_aspect=opts.keep_aspect, interp=opts.interpolation)
            frame = _coerce_rgb(frame, src_is_bgr=(self.meta.colorspace == "bgr"), color_space=opts.color_space)
            frame = _normalize_to_dtype(frame, opts.normalize, opts.dtype)
            if progress is not None and task_id is not None:
                progress.advance(task_id, 1)
            yield (frame, ts) if opts.return_timestamps else frame

        if progress is not None:
            progress.stop()

    def _iter_frames_file(self, opts: DecodeOptions) -> Iterable[Union[np.ndarray, Tuple[np.ndarray, float]]]:
        with self._open_capture() as cap:
            meta = _cv2_read_capture_meta(cap)
            fps = meta["fps"]
            total = meta["frame_count"]
            if opts.timestamps:
                # Sample exact timestamps
                times = opts.timestamps
                if fps is None:
                    # fallback to POS_MSEC
                    for t in times:
                        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
                        ok, frame = cap.read()
                        if not ok:
                            break
                        frame = self._transform_frame(frame, opts, src_is_bgr=True)
                        yield (frame, t) if opts.return_timestamps else frame
                    return
                else:
                    # Use frame positioning
                    for t in times:
                        idx = _time_to_frame_index(t, fps)
                        if not _valid_frame_index(idx, total):
                            continue
                        cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
                        ok, frame = cap.read()
                        if not ok:
                            break
                        frame = self._transform_frame(frame, opts, src_is_bgr=True)
                        ts = idx / fps
                        yield (frame, ts) if opts.return_timestamps else frame
                    return

            # Determine index range for sequential reading
            if fps is not None:
                start_idx = opts.start_frame if opts.start_frame is not None else 0
                end_idx = opts.end_frame if opts.end_frame is not None else (total - 1 if total else None)
                if opts.start_time is not None:
                    start_idx = max(start_idx, _time_to_frame_index(opts.start_time, fps))
                if opts.end_time is not None:
                    end_idx_raw = _time_to_frame_index(opts.end_time, fps)
                    end_idx = min(end_idx if end_idx is not None else end_idx_raw, end_idx_raw)
                if start_idx and start_idx > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_idx))
                    current_idx = start_idx
                else:
                    current_idx = _safe_int(cap.get(cv2.CAP_PROP_POS_FRAMES), 0)
            else:
                # fps unknown, fallback sequential
                current_idx = _safe_int(cap.get(cv2.CAP_PROP_POS_FRAMES), 0)
                end_idx = opts.end_frame

            # Resampling: build a set of indices if target_fps is requested and fps known
            resample_indices = None
            if opts.target_fps is not None and fps is not None and opts.target_fps > 0:
                step = fps / opts.target_fps
                resample_indices = set()
                i = (opts.start_frame or current_idx)
                last_idx = (end_idx if end_idx is not None else (total - 1 if total else i + 999999))
                while i <= last_idx:
                    resample_indices.add(int(round(i)))
                    i += step

            count = 0
            # Progress setup
            progress = None
            task_id = None
            total_prog = (end_idx - current_idx + 1) if (end_idx is not None and end_idx >= current_idx) else (total - current_idx if total else None)
            if opts.progress and _rich_available() and total_prog:
                progress = Progress(
                    "[progress.description]{task.description}",
                    BarColumn(),
                    MofNCompleteColumn(),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                    transient=True,
                )
                progress.start()
                task_id = progress.add_task("Decoding (file)", total=total_prog)

            while True:
                if end_idx is not None and current_idx > end_idx:
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                # Sampling logic
                take = True
                if opts.stride > 1 and ((current_idx - (opts.start_frame or 0)) % opts.stride != 0):
                    take = False
                if resample_indices is not None and current_idx not in resample_indices:
                    take = False
                if take:
                    out = self._transform_frame(frame, opts, src_is_bgr=True)
                    ts = (current_idx / fps) if fps else float(count)
                    yield (out, ts) if opts.return_timestamps else out
                    count += 1
                    if opts.max_frames is not None and count >= opts.max_frames:
                        break
                current_idx += 1
                if progress is not None and task_id is not None:
                    progress.advance(task_id, 1)
            if progress is not None:
                progress.stop()

    def _iter_frames_capture(self, opts: DecodeOptions) -> Iterable[Union[np.ndarray, Tuple[np.ndarray, float]]]:
        # For open capture (opencv/camera/stream). Seeking support depends on source.
        cap = self._capture
        if cap is None or not cap.isOpened():
            with self._open_capture() as fresh_cap:
                yield from self._iter_frames_capture_impl(fresh_cap, opts)
        else:
            yield from self._iter_frames_capture_impl(cap, opts)

    def _iter_frames_capture_impl(self, cap: "cv2.VideoCapture", opts: DecodeOptions) -> Iterable[Union[np.ndarray, Tuple[np.ndarray, float]]]:
        meta = _cv2_read_capture_meta(cap)
        fps = meta["fps"]
        total = meta["frame_count"]
        # Live or unknown fps: sequential only
        current_idx = _safe_int(cap.get(cv2.CAP_PROP_POS_FRAMES), 0)
        # If start_time/frame given and we can seek (not live stream)
        if not self._live and (opts.start_time is not None or opts.start_frame is not None) and total is not None:
            if opts.start_frame is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(max(0, opts.start_frame)))
                current_idx = max(0, opts.start_frame)
            elif opts.start_time is not None and fps is not None:
                current_idx = max(0, _time_to_frame_index(opts.start_time, fps))
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(current_idx))
        # End bound
        if not self._live:
            end_idx = None
            if opts.end_frame is not None:
                end_idx = min(opts.end_frame, (total - 1) if total is not None else opts.end_frame)
            elif opts.end_time is not None and fps is not None:
                end_idx = _time_to_frame_index(opts.end_time, fps)
        else:
            end_idx = None

        count = 0
        progress = None
        task_id = None
        # Only show progress when we know an end index
        if opts.progress and _rich_available() and end_idx is not None and end_idx >= current_idx:
            progress = Progress(
                "[progress.description]{task.description}",
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                transient=True,
            )
            progress.start()
            task_id = progress.add_task("Decoding (capture)", total=(end_idx - current_idx + 1))

        # Resampling if fps known
        resample_indices = None
        if opts.target_fps is not None and fps is not None and opts.target_fps > 0 and not self._live:
            step = fps / opts.target_fps
            resample_indices = set()
            i = (opts.start_frame or current_idx)
            last_idx = end_idx if end_idx is not None else ((total - 1) if total else i + 999999)
            while i <= last_idx:
                resample_indices.add(int(round(i)))
                i += step

        while True:
            if end_idx is not None and current_idx > end_idx:
                break
            ok, frame = cap.read()
            if not ok:
                break
            take = True
            if not self._live:
                # stride
                if opts.stride > 1 and ((current_idx - (opts.start_frame or 0)) % opts.stride != 0):
                    take = False
                if resample_indices is not None and current_idx not in resample_indices:
                    take = False
            if take:
                out = self._transform_frame(frame, opts, src_is_bgr=True)
                ts = (current_idx / fps) if fps else float(count)
                yield (out, ts) if opts.return_timestamps else out
                count += 1
                if opts.max_frames is not None and count >= opts.max_frames:
                    break
            current_idx += 1
            if progress is not None and task_id is not None:
                progress.advance(task_id, 1)
        if progress is not None:
            progress.stop()

    def _iter_frames_screen(self, opts: DecodeOptions) -> Iterable[Union[np.ndarray, Tuple[np.ndarray, float]]]:
        if self._producer is None:
            raise VideoError("Screen capture producer not available.")
        gen = self._producer()
        t0 = time.time()
        count = 0
        for frame in gen:
            # frame is RGB from producer
            ts = time.time() - t0
            # Transform
            frame = _crop_frame(frame, opts.crop)
            frame = _resize_frame(frame, opts.resize, keep_aspect=opts.keep_aspect, interp=opts.interpolation)
            frame = _coerce_rgb(frame, src_is_bgr=False, color_space=opts.color_space)
            frame = _normalize_to_dtype(frame, opts.normalize, opts.dtype)
            yield (frame, ts) if opts.return_timestamps else frame
            count += 1
            if opts.max_frames is not None and count >= opts.max_frames:
                break

    def _transform_frame(self, frame: np.ndarray, opts: DecodeOptions, *, src_is_bgr: bool) -> np.ndarray:
        frame = _crop_frame(frame, opts.crop)
        frame = _resize_frame(frame, opts.resize, keep_aspect=opts.keep_aspect, interp=opts.interpolation)
        frame = _coerce_rgb(frame, src_is_bgr=src_is_bgr, color_space=opts.color_space)
        frame = _normalize_to_dtype(frame, opts.normalize, opts.dtype)
        return frame

    @contextlib.contextmanager
    def _open_capture(self) -> Generator["cv2.VideoCapture", None, None]:
        """
        Context manager to open a cv2.VideoCapture based on the source.
        Ensures release after use for file-based sources.
        """
        if self.source_type in ("path", "http", "bytes", "base64", "zip", "tar", "remote"):
            if not self._path:
                raise VideoError("No underlying file path for capture.")
            cap = cv2.VideoCapture(str(self._path))
            if not cap.isOpened():
                raise VideoError(f"Failed to open video file at {self._path}.")
            try:
                yield cap
            finally:
                cap.release()
        elif self.source_type in ("opencv", "camera", "stream"):
            if self._capture is None or not self._capture.isOpened():
                raise VideoError("Capture not available or not open.")
            yield self._capture
        else:
            raise VideoError("Cannot open capture for this source type.")

    def _verbose_initial_report(self) -> None:
        if not _rich_available():
            return
        table = Table(title="Video Source Created", box=box.SIMPLE)
        table.add_column("Key", justify="right")
        table.add_column("Value", overflow="fold")
        table.add_row("source_type", self.source_type)
        table.add_row("source", self.meta.source or "")
        table.add_row("backend", self.meta.backend)
        table.add_row("codec", self.meta.codec or "")
        table.add_row("resolution", f"{self.meta.width}x{self.meta.height}")
        table.add_row("fps", f"{self.meta.fps}" if self.meta.fps else "unknown")
        table.add_row("frame_count", f"{self.meta.frame_count}" if self.meta.frame_count is not None else "unknown")
        table.add_row("duration", f"{self.meta.duration:.3f}s" if self.meta.duration is not None else "unknown")
        _console().print(table)

    @staticmethod
    def _probe_path(p: Path) -> VideoMeta:
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            raise VideoError(f"OpenCV cannot open: {p}")
        meta_info = _cv2_read_capture_meta(cap)
        cap.release()
        return VideoMeta(
            width=meta_info["width"],
            height=meta_info["height"],
            fps=meta_info["fps"],
            frame_count=meta_info["frame_count"],
            duration=meta_info["duration"],
            codec=meta_info["fourcc"],
            colorspace="bgr",
            source=str(p),
        )

    @staticmethod
    def _bytes_to_tempfile(data: bytes, suffix: str = ".mp4") -> Path:
        fd, name = tempfile.mkstemp(suffix=suffix, prefix="vsrc_")
        os.close(fd)
        p = Path(name)
        with p.open("wb") as f:
            f.write(data)
        return p

    def __del__(self) -> None:
        # Cleanup temp files and owned capture
        try:
            if self._owned_capture and self._capture is not None:
                self._capture.release()
            for p in self._temp_paths:
                try:
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass
        except Exception:
            pass

# ---------------------------------------------
# Examples and self-test (safe to run)
# ---------------------------------------------

if __name__ == "__main__":
    # Toggle to False to suppress rich metadata and progress
    VERBOSE = True
    video = Video.from_http_url("http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4")
    video.save("video/out_test.mp4")

    # # 1) Create from NumPy and explore options
    # T, H, W = 48, 120, 160
    # rng = np.random.default_rng(42)
    # # synthetic RGB frames [0..255]
    # frames = rng.integers(0, 256, size=(T, H, W, 3), dtype=np.uint8)
    # video = Video.from_numpy(frames, fps=24.0, color_space="rgb", verbose=VERBOSE)
    # video.describe()

    # # Iterate with resizing, cropping, color conversion and normalization
    # sample = []
    # for f, ts in video.iter_frames(
    #     start_frame=5,
    #     end_frame=20,
    #     stride=3,
    #     resize=(96, 72),
    #     interpolation="area",
    #     crop=(10, 10, 120, 90),
    #     color_space="rgb",
    #     normalize=True,
    #     return_timestamps=True,
    #     progress=True,
    # ):
    #     sample.append((f, ts))
    # if _rich_available():
    #     _console().print(Panel.fit(f"Iterated {len(sample)} frames with transformations.", title="Demo 1"))

    # # 2) Subclip by time and save (in-memory numpy -> mp4)
    # sub = video.subclip(start_time=0.25, end_time=1.0, resize=(128, 96), color_space="rgb", progress=True)
    # out_path = sub.save(str(Path(tempfile.gettempdir()) / "demo_clip.mp4"), fps=12, codec="mp4v", color_space="rgb")
    # if _rich_available():
    #     _console().print(Panel.fit(f"Wrote demo clip to: {out_path}", title="Demo 2"))

    # # 3) Single frame extraction by time and index
    # try:
    #     f_by_time = video.get_frame(time_sec=0.5, color_space="rgb", resize=(64, 48))
    #     f_by_index = video.get_frame(index=10, color_space="rgb")
    #     if _rich_available():
    #         _console().print(Panel.fit(f"Frames extracted: time-based {f_by_time.shape}, index-based {f_by_index.shape}", title="Demo 3"))
    # except VideoError as e:
    #     print(f"get_frame demo error: {e}")

    # # 4) Frame subsequence from a local file (safe path check)
    # # This part tries to find any small MP4 in the current working directory to demonstrate local path loading.
    # found_local = None
    # for ext in (".mp4", ".mov", ".avi", ".mkv", ".webm", ".ogv"):
    #     for p in Path(".").glob(f"*{ext}"):
    #         found_local = p
    #         break
    #     if found_local:
    #         break

    # if found_local:
    #     try:
    #         v_local = Video.from_local_path(str(found_local), verbose=VERBOSE)
    #         v_local.describe()
    #         seq = v_local.to_numpy(start_frame=0, end_frame=min(15, (v_local.meta.frame_count or 16) - 1), color_space="rgb", progress=True)
    #         if _rich_available():
    #             _console().print(Panel.fit(f"Loaded subsequence from local file: {found_local}, got array {seq.shape}", title="Demo 4"))
    #     except Exception as e:
    #         print(f"Local file demo error: {e}")
    # else:
    #     if _rich_available():
    #         _console().print(Panel.fit("No local video found to demo from_local_path; skipping.", title="Demo 4", subtitle="Place an MP4 in the current directory to run this demo."))

    # # 5) Clipboard demo (best-effort; may not be available in sandbox)
    # try:
    #     # Only run if we actually have something in the clipboard
    #     txt = _clipboard_text()
    #     if txt and (txt.startswith("http") or txt.startswith("file://") or Path(txt).expanduser().exists() or txt.startswith("data:")):
    #         v_clip = Video.from_clipboard(verbose=VERBOSE)
    #         v_clip.describe()
    #         head = v_clip.to_numpy(max_frames=3, color_space="rgb", progress=True)
    #         if _rich_available():
    #             _console().print(Panel.fit(f"Clipboard video head shape: {head.shape}", title="Demo 5"))
    #     else:
    #         if _rich_available():
    #             _console().print(Panel.fit("Clipboard demo skipped: no usable video content detected in clipboard.", title="Demo 5"))
    # except Exception as e:
    #     print(f"Clipboard demo error: {e}")

    # # 6) HDF5/Tensor/Torch demos (only if libs available)
    # # TensorFlow
    # try:
    #     import tensorflow as tf  # type: ignore  # noqa
    #     tf_arr = np.clip(rng.normal(0.5, 0.2, size=(12, 64, 96, 3)), 0, 1).astype(np.float32)
    #     tf_video = Video.from_tensorflow_tensor(tf_arr, verbose=VERBOSE, fps=10.0)
    #     head = tf_video.to_numpy(max_frames=4, color_space="rgb")
    #     if _rich_available():
    #         _console().print(Panel.fit(f"TF tensor video head: {head.shape}", title="Demo 6A"))
    # except Exception:
    #     if _rich_available():
    #         _console().print(Panel.fit("TensorFlow not available; skipping TF demo.", title="Demo 6A"))

    # # PyTorch
    # try:
    #     import torch  # type: ignore  # noqa
    #     torch_arr = torch.from_numpy(np.random.randint(0, 255, size=(10, 3, 64, 64), dtype=np.uint8))
    #     torch_video = Video.from_torch_tensor(torch_arr, verbose=VERBOSE, fps=15.0)
    #     head = torch_video.to_numpy(max_frames=5, color_space="rgb")
    #     if _rich_available():
    #         _console().print(Panel.fit(f"Torch tensor video head: {head.shape}", title="Demo 6B"))
    # except Exception:
    #     if _rich_available():
    #         _console().print(Panel.fit("PyTorch not available; skipping Torch demo.", title="Demo 6B"))

    # # 7) Screen and Camera demos are intentionally skipped by default due to environment constraints.
    # # Uncomment to try locally:
    # # screen = Video.from_screen_capture(verbose=VERBOSE)
    # # for i, (frame, ts) in enumerate(screen.iter_frames(max_frames=5, return_timestamps=True, color_space="rgb")):
    # #     pass
    # # cam = Video.from_camera(0, verbose=VERBOSE)
    # # for i, frame in enumerate(cam.iter_frames(max_frames=10, resize=(320, 240))):
    # #     pass

    # if _rich_available():
    #     _console().print(Panel.fit("All demos finished.", title="Done"))