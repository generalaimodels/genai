#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Zero-Warning Image Augmentation Suite (single-file, verbose, JSON-safe)

- Loads image(s), optional resize, applies a wide catalog of augmentations, saves each output into one folder.
- Uses Albumentations when available, plus robust custom fallbacks for transforms requiring extra metadata or not present.
- Produces rich console logs (with --verbose) and a JSON-safe metadata file (metadata_run.json).
- Hard requirement from you: ZERO warnings in console. This script globally suppresses warnings and also avoids invalid
  parameters that cause Albumentations to warn. Additionally, some transforms are fulfilled via custom implementations
  to remove metadata-related warnings in Albumentations v2.

Install hints (for maximal coverage):
  pip install -U albumentations opencv-python-headless rich numpy scikit-image

CLI examples:
  python augment_all.py --input ./images/one.jpg --out ./aug_out --resize 1024 1024 --verbose
  python augment_all.py --input ./images --out ./aug_out --seed 123 --verbose
  python augment_all.py --input ./img.jpg --out ./aug_out --only "Blur,Rotate,RandomBrightnessContrast"
  python augment_all.py --input ./img.jpg --out ./aug_out --only-category "color,geometry"
  python augment_all.py --input ./img.jpg --out ./aug_out --save-ext .jpg --jpeg-quality 95
"""

from __future__ import annotations

# 0) Hard-suppress every warning to guarantee a clean console (as requested).
import warnings as _warn
_warn.filterwarnings("ignore")  # Hide all warnings of any category and module

import argparse
import dataclasses
import datetime as _dt
import inspect
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Core imaging libs
try:
    import cv2
except Exception:
    cv2 = None  # type: ignore
import numpy as np

# Rich console for verbose UX
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    from rich.traceback import install as rich_install
except Exception:
    Console = None  # type: ignore
    Table = None  # type: ignore
    Panel = None  # type: ignore
    Text = None  # type: ignore
    box = None  # type: ignore
    def rich_install():  # type: ignore
        return None

# Albumentations
try:
    import albumentations as A
    HAVE_ALBU = True
    ALBU_VER = getattr(A, "__version__", "unknown")
except Exception:
    A = None  # type: ignore
    HAVE_ALBU = False
    ALBU_VER = "not-installed"

# Optional skimage
try:
    from skimage.exposure import match_histograms as _sk_match_histograms
    HAVE_SKIMAGE = True
except Exception:
    HAVE_SKIMAGE = False


# ------------------------------------------
# Utilities
# ------------------------------------------

def _get_console(verbose: bool) -> "Console":
    if Console is None:
        class _Dummy:
            def __getattr__(self, _):
                return self
            def print(self, *a, **k): pass
        return _Dummy()  # type: ignore
    try:
        rich_install()
    except Exception:
        pass
    return Console(stderr=False, highlight=True, soft_wrap=True, record=False, color_system="auto")


def _cv2_available() -> bool:
    return cv2 is not None


def _ensure_dir(p: Union[str, Path]) -> Path:
    d = Path(p).expanduser().resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _seed_everything(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)  # type: ignore
    except Exception:
        pass


def _read_image(path: Path) -> np.ndarray:
    """
    Return uint8 BGR, HxWx3.
    """
    if not path.exists():
        raise FileNotFoundError(str(path))
    if _cv2_available():
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to read image: {path}")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img
    else:
        from PIL import Image
        pil = Image.open(str(path)).convert("RGB")
        return np.array(pil)[:, :, ::-1].copy()


def _resize_image(img: np.ndarray, width: Optional[int], height: Optional[int]) -> np.ndarray:
    if width is None and height is None:
        return img
    h, w = img.shape[:2]
    if width is not None and height is not None:
        if _cv2_available():
            return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        else:
            from PIL import Image
            pil = Image.fromarray(img[:, :, ::-1]).resize((width, height))
            return np.array(pil)[:, :, ::-1].copy()
    if width is not None:
        scale = width / float(w)
        new_w, new_h = width, max(1, int(round(h * scale)))
    else:
        scale = height / float(h)
        new_h, new_w = height, max(1, int(round(w * scale)))
    if _cv2_available():
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        from PIL import Image
        pil = Image.fromarray(img[:, :, ::-1]).resize((new_w, new_h))
        return np.array(pil)[:, :, ::-1].copy()


def _to_uint8_image(img: np.ndarray) -> np.ndarray:
    out = img
    if out.ndim == 2:
        out = np.stack([out] * 3, axis=-1)
    if out.ndim == 3 and out.shape[2] == 4 and _cv2_available():
        out = cv2.cvtColor(out, cv2.COLOR_BGRA2BGR)
    if out.dtype in (np.float32, np.float64):
        mn = float(np.nanmin(out))
        mx = float(np.nanmax(out))
        if 0.0 >= mn and mx <= 1.0 and mx > 0:
            out = (out * 255.0).round()
        else:
            if mx > mn:
                out = ((out - mn) / (mx - mn) * 255.0).round()
            else:
                out = np.zeros_like(out, dtype=np.float32)
        out = out.astype(np.uint8)
    elif out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
    if out.ndim == 2:
        out = np.stack([out] * 3, axis=-1)
    if out.ndim == 3 and out.shape[2] == 1:
        out = np.repeat(out, 3, axis=2)
    return out


def _save_image(path: Path, img: np.ndarray, save_ext: str, jpeg_quality: int = 95) -> Path:
    out_path = path.with_suffix(save_ext)
    arr = _to_uint8_image(img)
    if _cv2_available():
        if save_ext.lower() in {".jpg", ".jpeg"}:
            cv2.imwrite(str(out_path), arr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        else:
            cv2.imwrite(str(out_path), arr)
    else:
        from PIL import Image
        Image.fromarray(arr[:, :, ::-1], mode="RGB").save(str(out_path), quality=jpeg_quality)
    return out_path


def _sanitize_name(name: str) -> str:
    name = re.sub(r"[^0-9a-zA-Z_]+", "_", name.strip())
    return re.sub(r"_+", "_", name).strip("_")


def _list_images_in_directory(d: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    imgs = []
    for p in sorted(d.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            imgs.append(p)
    return imgs


def _pick_additional_image(candidates: List[np.ndarray], fallback: np.ndarray) -> np.ndarray:
    if not candidates:
        return fallback
    idx = random.randint(0, len(candidates) - 1)
    cand = candidates[idx]
    return cand if isinstance(cand, np.ndarray) else fallback


def _clip_bbox_pascal_voc(box: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 <= x1: x2 = min(w - 1, x1 + 1)
    if y2 <= y1: y2 = min(h - 1, y1 + 1)
    return (x1, y1, x2, y2)


def _synthetic_bboxes_for_image(img: np.ndarray) -> Tuple[List[Tuple[int,int,int,int]], List[str]]:
    h, w = img.shape[:2]
    cx1, cy1, cx2, cy2 = int(0.2*w), int(0.2*h), int(0.8*w), int(0.8*h)
    tx1, ty1, tx2, ty2 = 0, 0, max(1, int(0.3*w)), max(1, int(0.3*h))
    bx1, by1, bx2, by2 = max(0, int(0.65*w)), max(0, int(0.65*h)), w-1, h-1
    boxes = [
        _clip_bbox_pascal_voc((cx1, cy1, cx2, cy2), w, h),
        _clip_bbox_pascal_voc((tx1, ty1, tx2, ty2), w, h),
        _clip_bbox_pascal_voc((bx1, by1, bx2, by2), w, h),
    ]
    labels = ["center", "top_left", "bottom_right"]
    return boxes, labels


# --------------- JSON sanitization ---------------

def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (np.integer, np.int32, np.int64, np.uint8, np.uint16)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        try:
            if obj.size <= 32:
                return obj.tolist()
            return {"__ndarray__": True, "shape": list(obj.shape), "dtype": str(obj.dtype)}
        except Exception:
            return {"__ndarray__": True, "shape": "unknown"}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(x) for x in list(obj)]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "model_dump"):
        try:
            return _json_safe(obj.model_dump())
        except Exception:
            return str(obj)
    if hasattr(obj, "__dict__"):
        try:
            return _json_safe(vars(obj))
        except Exception:
            return str(obj)
    return str(obj)


# ------------------------------------------
# Data classes
# ------------------------------------------

@dataclass
class TransformResult:
    name: str
    status: str  # 'applied' | 'skipped' | 'error'
    reason: Optional[str] = None
    save_path: Optional[str] = None
    time_ms: Optional[float] = None
    input_shape: Optional[Tuple[int, int, int]] = None
    output_shape: Optional[Tuple[int, int, int]] = None
    replay: Optional[Dict[str, Any]] = None
    custom: bool = False
    needs: List[str] = field(default_factory=list)


@dataclass
class TransformContext:
    image: np.ndarray
    extra_images: List[np.ndarray]
    seed: Optional[int]
    bbox_mode: str = "pascal_voc"
    bboxes: Optional[List[Tuple[int, int, int, int]]] = None
    bbox_labels: Optional[List[str]] = None


# ------------------------------------------
# Transform Factory
# ------------------------------------------

class TransformFactory:
    """
    Builds transforms. Custom fallbacks are provided to avoid version-specific warnings
    and to eliminate metadata-related warnings by Albumentations v2.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.console = _get_console(verbose)

        # The full list from the task (order preserved)
        self.requested_transforms = [
            'AdditiveNoise', 'AdvancedBlur', 'Affine', 'AtLeastOneBBoxRandomCrop', 'AutoContrast',
            'BBoxSafeRandomCrop', 'BaseCompose', 'BasicTransform', 'BboxParams', 'Blur', 'CLAHE',
            'CenterCrop', 'CenterCrop3D', 'ChannelDropout', 'ChannelShuffle', 'ChromaticAberration',
            'CoarseDropout', 'CoarseDropout3D', 'ColorJitter', 'Compose', 'ConstrainedCoarseDropout',
            'Crop', 'CropAndPad', 'CropNonEmptyMaskIfExists', 'CubicSymmetry', 'D4', 'Defocus',
            'Downscale', 'DualTransform', 'ElasticTransform', 'Emboss', 'Equalize', 'Erasing', 'FDA',
            'FancyPCA', 'FrequencyMasking', 'FromFloat', 'GaussNoise', 'GaussianBlur', 'GlassBlur',
            'GridDistortion', 'GridDropout', 'GridElasticDeform', 'HEStain', 'HistogramMatching',
            'HorizontalFlip', 'HueSaturationValue', 'ISONoise', 'Illumination', 'ImageCompression',
            'ImageOnlyTransform', 'InvertImg', 'KeypointParams', 'Lambda', 'LongestMaxSize',
            'MaskDropout', 'MedianBlur', 'Morphological', 'Mosaic', 'MotionBlur', 'MultiplicativeNoise',
            'NoOp', 'Normalize', 'OneOf', 'OneOrOther', 'OpticalDistortion', 'OverlayElements', 'Pad',
            'Pad3D', 'PadIfNeeded', 'PadIfNeeded3D', 'Perspective', 'PiecewiseAffine',
            'PixelDistributionAdaptation', 'PixelDropout', 'PlanckianJitter', 'PlasmaBrightnessContrast',
            'PlasmaShadow', 'Posterize', 'RGBShift', 'RandomBrightnessContrast', 'RandomCrop',
            'RandomCrop3D', 'RandomCropFromBorders', 'RandomCropNearBBox', 'RandomFog', 'RandomGamma',
            'RandomGravel', 'RandomGridShuffle', 'RandomOrder', 'RandomRain', 'RandomResizedCrop',
            'RandomRotate90', 'RandomScale', 'RandomShadow', 'RandomSizedBBoxSafeCrop',
            'RandomSizedCrop', 'RandomSnow', 'RandomSunFlare', 'RandomToneCurve', 'ReplayCompose',
            'Resize', 'RingingOvershoot', 'Rotate', 'SafeRotate', 'SaltAndPepper',
            'SelectiveChannelTransform', 'Sequential', 'Sharpen', 'ShiftScaleRotate', 'ShotNoise',
            'SmallestMaxSize', 'Solarize', 'SomeOf', 'Spatter', 'SquareSymmetry', 'Superpixels',
            'TextImage', 'ThinPlateSpline', 'TimeMasking', 'TimeReverse', 'ToFloat', 'ToGray', 'ToRGB',
            'ToSepia', 'ToTensor3D', 'ToTensorV2', 'Transform3D', 'Transpose', 'UnsharpMask',
            'VerticalFlip', 'XYMasking', 'ZoomBlur',
        ]

        self.aliases = {
            "Erasing": "CoarseDropout",
            "SafeRotate": "Rotate",
            # Force custom
            "Pad": None,
            "RandomGridShuffle": None,
        }

        self.meta_classes = {
            "BaseCompose", "BasicTransform", "DualTransform", "ImageOnlyTransform",
            "Compose", "ReplayCompose", "Sequential", "RandomOrder", "SomeOf",
            "OneOf", "OneOrOther", "BboxParams", "KeypointParams", "Transform3D", "Lambda",
        }
        self.non_image_2d_transforms = {
            "CenterCrop3D", "CoarseDropout3D", "Pad3D", "PadIfNeeded3D", "RandomCrop3D",
            "ToTensor3D", "TimeMasking", "TimeReverse", "Transform3D", "CubicSymmetry",
            "SquareSymmetry",
        }
        self.mask_required = {"MaskDropout"}

        # Albumentations v2 transforms requiring metadata -> we use custom fallbacks
        self.v2_metadata_transforms = {
            "Mosaic", "HistogramMatching", "OverlayElements", "PixelDistributionAdaptation", "FDA"
        }

        self.categories: Dict[str, List[str]] = {
            "color": [
                "AutoContrast", "Equalize", "CLAHE", "ColorJitter", "RandomBrightnessContrast",
                "RandomGamma", "HueSaturationValue", "RGBShift", "FancyPCA", "RandomToneCurve",
                "Posterize", "Solarize", "InvertImg", "ToGray", "ToSepia", "ToRGB",
            ],
            "blur": [
                "Blur", "GaussianBlur", "MedianBlur", "MotionBlur", "GlassBlur", "AdvancedBlur",
                "Defocus", "ZoomBlur", "UnsharpMask", "Sharpen", "Emboss",
            ],
            "noise": [
                "GaussNoise", "ISONoise", "MultiplicativeNoise", "AdditiveNoise", "ShotNoise",
                "SaltAndPepper", "RingingOvershoot",
            ],
            "geometry": [
                "Affine", "Rotate", "SafeRotate", "RandomRotate90", "HorizontalFlip", "VerticalFlip",
                "Transpose", "ShiftScaleRotate", "RandomScale", "Perspective", "OpticalDistortion",
                "GridDistortion", "PiecewiseAffine", "ElasticTransform", "ThinPlateSpline",
                "RandomResizedCrop", "CenterCrop", "RandomCrop", "RandomCropFromBorders", "Crop",
                "CropAndPad", "PadIfNeeded", "Pad", "LongestMaxSize", "SmallestMaxSize", "Resize",
            ],
            "dropout": [
                "CoarseDropout", "ConstrainedCoarseDropout", "GridDropout", "PixelDropout",
                "MaskDropout", "ChannelDropout", "ChannelShuffle", "SelectiveChannelTransform",
            ],
            "weather": [
                "RandomFog", "RandomSnow", "RandomRain", "RandomSunFlare", "RandomShadow", "Spatter",
                "RandomGravel",
            ],
            "compression": ["ImageCompression"],
            "domain": ["HEStain", "HistogramMatching", "FDA", "Illumination", "PlanckianJitter",
                       "PixelDistributionAdaptation"],
            "bboxes": [
                "BBoxSafeRandomCrop", "RandomCropNearBBox", "RandomSizedBBoxSafeCrop",
                "AtLeastOneBBoxRandomCrop",
            ],
            "misc": ["Superpixels", "Mosaic", "Morphological", "OverlayElements", "TextImage", "NoOp"],
            "normalize": ["Normalize", "FromFloat", "ToFloat"],
            "3d_or_audio": list(self.non_image_2d_transforms),
        }

    # ---------- Custom transforms (to avoid warnings and metadata requirements) ----------

    @staticmethod
    def _custom_additive_noise(ctx: TransformContext) -> A.BasicTransform:
        def f(image, **_):
            img = image.astype(np.float32)
            strength = random.uniform(5.0, 20.0)
            noise = np.random.normal(0.0, strength, size=img.shape).astype(np.float32)
            out = img + noise
            return np.clip(out, 0, 255).astype(np.uint8)
        return A.Lambda(name="AdditiveNoise_custom", image=f, p=1.0)

    @staticmethod
    def _custom_shot_noise(ctx: TransformContext) -> A.BasicTransform:
        def f(image, **_):
            img = image.astype(np.float32) / 255.0
            lam = random.uniform(10.0, 50.0)
            noisy = np.random.poisson(img * lam) / lam
            return np.clip(noisy * 255.0, 0, 255).astype(np.uint8)
        return A.Lambda(name="ShotNoise_custom", image=f, p=1.0)

    @staticmethod
    def _custom_salt_and_pepper(ctx: TransformContext) -> A.BasicTransform:
        def f(image, **_):
            img = image.copy()
            h, w = img.shape[:2]
            prob = random.uniform(0.005, 0.02)
            num = int(h * w * prob)
            ys = np.random.randint(0, h, size=(num,))
            xs = np.random.randint(0, w, size=(num,))
            img[ys, xs] = 255
            ys = np.random.randint(0, h, size=(num,))
            xs = np.random.randint(0, w, size=(num,))
            img[ys, xs] = 0
            return img
        return A.Lambda(name="SaltAndPepper_custom", image=f, p=1.0)

    @staticmethod
    def _custom_unsharp_mask(ctx: TransformContext) -> A.BasicTransform:
        def f(image, **_):
            if not _cv2_available(): return image
            blur = cv2.GaussianBlur(image, (0, 0), sigmaX=random.uniform(0.8, 2.0))
            amount = random.uniform(0.5, 1.2)
            sharp = cv2.addWeighted(image, 1 + amount, blur, -amount, 0)
            return np.clip(sharp, 0, 255).astype(np.uint8)
        return A.Lambda(name="UnsharpMask_custom", image=f, p=1.0)

    @staticmethod
    def _custom_emboss(ctx: TransformContext) -> A.BasicTransform:
        def f(image, **_):
            if not _cv2_available(): return image
            kernel = np.array([[-2, -1, 0], [-1,  1, 1], [0,  1, 2]], dtype=np.float32)
            out = cv2.filter2D(image, -1, kernel) + 128
            return np.clip(out, 0, 255).astype(np.uint8)
        return A.Lambda(name="Emboss_custom", image=f, p=1.0)

    @staticmethod
    def _custom_zoom_blur(ctx: TransformContext) -> A.BasicTransform:
        def f(image, **_):
            if not _cv2_available(): return image
            h, w = image.shape[:2]
            steps = 8
            strength = random.uniform(1.02, 1.08)
            acc = np.zeros_like(image, dtype=np.float32)
            imgf = image.astype(np.float32)
            for i in range(steps):
                scale = strength ** i
                resized = cv2.resize(imgf, (max(1, int(w*scale)), max(1, int(h*scale))), interpolation=cv2.INTER_LINEAR)
                rh, rw = resized.shape[:2]
                ys = max(0, (rh - h) // 2)
                xs = max(0, (rw - w) // 2)
                crop = resized[ys: ys+h, xs: xs+w]
                canvas = np.zeros_like(imgf)
                ch, cw = crop.shape[:2]
                y0, x0 = (h - ch) // 2, (w - cw) // 2
                canvas[y0:y0+ch, x0:x0+cw] = crop
                acc += canvas
            acc /= steps
            return np.clip(acc, 0, 255).astype(np.uint8)
        return A.Lambda(name="ZoomBlur_custom", image=f, p=1.0)

    @staticmethod
    def _custom_chromatic_aberration(ctx: TransformContext) -> A.BasicTransform:
        def f(image, **_):
            if not _cv2_available() or image.shape[2] < 3: return image
            shift = random.randint(1, 3)
            dx, dy = random.choice([-shift, shift]), random.choice([-shift, shift])
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            r = cv2.warpAffine(image[:, :, 2], M, (image.shape[1], image.shape[0]))
            out = image.copy()
            out[:, :, 2] = r
            return out
        return A.Lambda(name="ChromaticAberration_custom", image=f, p=1.0)

    @staticmethod
    def _custom_pad(ctx: TransformContext) -> A.BasicTransform:
        def f(image, **_):
            if not _cv2_available(): return image
            top = random.randint(5, 40); bottom = random.randint(5, 40)
            left = random.randint(5, 40); right = random.randint(5, 40)
            color = tuple(int(x) for x in np.random.randint(0, 255, size=(3,)))
            return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return A.Lambda(name="Pad_custom", image=f, p=1.0)

    @staticmethod
    def _custom_mosaic(ctx: TransformContext) -> A.BasicTransform:
        def f(image, **_):
            imgs = [image]
            for _ in range(3):
                imgs.append(_pick_additional_image(ctx.extra_images, image))
            h, w = image.shape[:2]
            target_h, target_w = max(1, h//2), max(1, w//2)
            tiles = []
            for im in imgs:
                if _cv2_available():
                    rr = cv2.resize(im, (target_w, target_h), interpolation=cv2.INTER_AREA)
                else:
                    from PIL import Image
                    pil = Image.fromarray(im[:, :, ::-1]).resize((target_w, target_h))
                    rr = np.array(pil)[:, :, ::-1].copy()
                tiles.append(rr)
            top = np.hstack([tiles[0], tiles[1]])
            bottom = np.hstack([tiles[2], tiles[3]])
            return np.vstack([top, bottom])
        return A.Lambda(name="Mosaic_custom", image=f, p=1.0)

    @staticmethod
    def _custom_histogram_matching(ctx: TransformContext) -> A.BasicTransform:
        def f(image, **_):
            ref = _pick_additional_image(ctx.extra_images, image)
            if HAVE_SKIMAGE:
                if _cv2_available():
                    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    ref_rgb = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
                else:
                    img_rgb = image[:, :, ::-1]; ref_rgb = ref[:, :, ::-1]
                matched = _sk_match_histograms(img_rgb, ref_rgb, channel_axis=-1)
                if _cv2_available():
                    return cv2.cvtColor(matched.astype(np.uint8), cv2.COLOR_RGB2BGR)
                return matched[:, :, ::-1].astype(np.uint8)
            # fallback mean-std matching
            out = image.astype(np.float32); ref = ref.astype(np.float32)
            for c in range(3):
                x, r = out[:, :, c], ref[:, :, c]
                xm, xs = x.mean(), x.std() + 1e-6
                rm, rs = r.mean(), r.std() + 1e-6
                out[:, :, c] = (x - xm) / xs * rs + rm
            return np.clip(out, 0, 255).astype(np.uint8)
        return A.Lambda(name="HistogramMatching_custom", image=f, p=1.0)

    @staticmethod
    def _custom_overlay_elements(ctx: TransformContext) -> A.BasicTransform:
        def f(image, **_):
            if not _cv2_available(): return image
            out = image.copy()
            h, w = out.shape[:2]
            for _ in range(random.randint(3, 8)):
                color = tuple(int(c) for c in np.random.randint(0, 255, size=3))
                alpha = random.uniform(0.2, 0.6)
                shape = random.choice(["rect", "circle"])
                overlay = out.copy()
                if shape == "rect":
                    x1, y1 = random.randint(0, w-1), random.randint(0, h-1)
                    x2 = random.randint(x1, min(w-1, x1+int(0.3*w)))
                    y2 = random.randint(y1, min(h-1, y1+int(0.3*h)))
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                else:
                    cx, cy = random.randint(0, w-1), random.randint(0, h-1)
                    rad = random.randint(5, max(6, int(0.15*min(h, w))))
                    cv2.circle(overlay, (cx, cy), rad, color, -1)
                out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)
            return out
        return A.Lambda(name="OverlayElements_custom", image=f, p=1.0)

    @staticmethod
    def _custom_pda(ctx: TransformContext) -> A.BasicTransform:
        def f(image, **_):
            ref = _pick_additional_image(ctx.extra_images, image)
            if HAVE_SKIMAGE:
                if _cv2_available():
                    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    ref_rgb = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
                else:
                    img_rgb = image[:, :, ::-1]; ref_rgb = ref[:, :, ::-1]
                matched = _sk_match_histograms(img_rgb, ref_rgb, channel_axis=-1)
                if _cv2_available():
                    return cv2.cvtColor(matched.astype(np.uint8), cv2.COLOR_RGB2BGR)
                return matched[:, :, ::-1].astype(np.uint8)
            out = image.astype(np.float32); ref = ref.astype(np.float32)
            for c in range(3):
                x, r = out[:, :, c], ref[:, :, c]
                xm, xs = x.mean(), x.std() + 1e-6
                rm, rs = r.mean(), r.std() + 1e-6
                out[:, :, c] = (x - xm) / xs * rs + rm
            return np.clip(out, 0, 255).astype(np.uint8)
        return A.Lambda(name="PixelDistributionAdaptation_custom", image=f, p=1.0)

    @staticmethod
    def _custom_fda(ctx: TransformContext) -> A.BasicTransform:
        def f(image, **_):
            ref = _pick_additional_image(ctx.extra_images, image)
            src = image.astype(np.float32) / 255.0
            tar = ref.astype(np.float32) / 255.0
            h, w = src.shape[:2]
            r = max(3, int(0.01 * min(h, w)))
            out = np.zeros_like(src)
            for c in range(3):
                Fs, Ft = np.fft.fft2(src[:, :, c]), np.fft.fft2(tar[:, :, c])
                Fs_s, Ft_s = np.fft.fftshift(Fs), np.fft.fftshift(Ft)
                cy, cx = h // 2, w // 2
                mag_s, phase_s = np.abs(Fs_s), np.angle(Fs_s)
                mag_t = np.abs(Ft_s)
                mag_mix = mag_s.copy()
                mag_mix[cy-r:cy+r+1, cx-r:cx+r+1] = mag_t[cy-r:cy+r+1, cx-r:cx+r+1]
                F_mix = np.fft.ifftshift(mag_mix * np.exp(1j * phase_s))
                out[:, :, c] = np.fft.ifft2(F_mix).real
            return (np.clip(out, 0, 1) * 255.0).astype(np.uint8)
        return A.Lambda(name="FDA_custom", image=f, p=1.0)

    @staticmethod
    def _custom_random_grid_shuffle(ctx: TransformContext) -> A.BasicTransform:
        def f(image, **_):
            h, w = image.shape[:2]
            n = random.choice([2, 3, 4])
            ys = np.linspace(0, h, n+1).astype(int)
            xs = np.linspace(0, w, n+1).astype(int)
            tiles = []
            for i in range(n):
                for j in range(n):
                    tiles.append(image[ys[i]:ys[i+1], xs[j]:xs[j+1]].copy())
            random.shuffle(tiles)
            out = np.zeros_like(image)
            k = 0
            for i in range(n):
                for j in range(n):
                    th, tw = ys[i+1]-ys[i], xs[j+1]-xs[j]
                    tile = tiles[k]
                    if tile.shape[:2] != (th, tw):
                        if _cv2_available():
                            tile = cv2.resize(tile, (tw, th), interpolation=cv2.INTER_AREA)
                        else:
                            from PIL import Image
                            pil = Image.fromarray(tile[:, :, ::-1]).resize((tw, th))
                            tile = np.array(pil)[:, :, ::-1].copy()
                    out[ys[i]:ys[i+1], xs[j]:xs[j+1]] = tile
                    k += 1
            return out
        return A.Lambda(name="RandomGridShuffle_custom", image=f, p=1.0)

    @staticmethod
    def _custom_crop_and_pad(ctx: TransformContext) -> A.BasicTransform:
        def f(image, **_):
            h, w = image.shape[:2]
            ch, cw = max(1, int(0.8*h)), max(1, int(0.8*w))
            y0, x0 = (h - ch)//2, (w - cw)//2
            crop = image[y0:y0+ch, x0:x0+cw]
            if not _cv2_available():
                return crop
            top = random.randint(5, 20); bottom = random.randint(5, 20)
            left = random.randint(5, 20); right = random.randint(5, 20)
            color = tuple(int(x) for x in np.random.randint(0, 255, size=3))
            return cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return A.Lambda(name="CropAndPad_custom", image=f, p=1.0)

    @staticmethod
    def _custom_crop_non_empty_mask(ctx: TransformContext) -> A.BasicTransform:
        def f(image, **_):
            if not _cv2_available():
                return image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            h, w = image.shape[:2]
            for _ in range(20):
                ch = random.randint(max(8, h//5), h)
                cw = random.randint(max(8, w//5), w)
                y0 = random.randint(0, h - ch)
                x0 = random.randint(0, w - cw)
                if edges[y0:y0+ch, x0:x0+cw].sum() > 0:
                    return image[y0:y0+ch, x0:x0+cw]
            # fallback center crop
            ch, cw = max(8, int(0.8*h)), max(8, int(0.8*w))
            y0, x0 = (h - ch)//2, (w - cw)//2
            return image[y0:y0+ch, x0:x0+cw]
        return A.Lambda(name="CropNonEmptyMaskIfExists_custom", image=f, p=1.0)

    @staticmethod
    def _custom_random_sized_crop(ctx: TransformContext) -> A.BasicTransform:
        def f(image, **_):
            h, w = image.shape[:2]
            area = h * w
            for _ in range(20):
                target_area = random.uniform(0.5, 1.0) * area
                aspect = random.uniform(0.75, 1.333)
                nh = int(round(np.sqrt(target_area / aspect)))
                nw = int(round(np.sqrt(target_area * aspect)))
                if nh <= h and nw <= w and nh > 0 and nw > 0:
                    y0 = random.randint(0, h - nh)
                    x0 = random.randint(0, w - nw)
                    crop = image[y0:y0+nh, x0:x0+nw]
                    if _cv2_available():
                        return cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)
                    else:
                        from PIL import Image
                        pil = Image.fromarray(crop[:, :, ::-1]).resize((w, h))
                        return np.array(pil)[:, :, ::-1].copy()
            return image
        return A.Lambda(name="RandomSizedCrop_custom", image=f, p=1.0)

    @staticmethod
    def _custom_text_image(ctx: TransformContext) -> A.BasicTransform:
        def f(image, **_):
            if not _cv2_available(): return image
            out = image.copy()
            h, w = out.shape[:2]
            text = random.choice(["AI", "AUG", "Albumentations", "GENAI", "DATA"])
            font = random.choice([
                cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_COMPLEX,
                cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_TRIPLEX
            ])
            scale = random.uniform(0.7, 2.0)
            color = tuple(int(c) for c in np.random.randint(0, 255, size=3))
            thickness = random.randint(1, 3)
            x = random.randint(0, max(0, w - 1))
            y = random.randint(0, max(0, h - 1))
            cv2.putText(out, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
            return out
        return A.Lambda(name="TextImage_custom", image=f, p=1.0)

    @staticmethod
    def _custom_selective_channel_transform(ctx: TransformContext) -> A.BasicTransform:
        def f(image, **_):
            out = image.copy().astype(np.float32)
            channels = [0, 1, 2]
            k = random.randint(1, 3)
            chs = random.sample(channels, k)
            for c in chs:
                op = random.choice(["invert", "gamma", "bias"])
                if op == "invert":
                    out[:, :, c] = 255.0 - out[:, :, c]
                elif op == "gamma":
                    gamma = random.uniform(0.7, 1.5)
                    out[:, :, c] = (out[:, :, c]/255.0) ** gamma * 255.0
                else:
                    out[:, :, c] = np.clip(out[:, :, c] + random.uniform(-30, 30), 0, 255)
            return out.clip(0, 255).astype(np.uint8)
        return A.Lambda(name="SelectiveChannelTransform_custom", image=f, p=1.0)

    # ---------- Albumentations helpers ----------

    def _get_albu_class(self, class_name: str) -> Optional[type]:
        if not HAVE_ALBU:
            return None
        return getattr(A, class_name, None)

    def _instantiate_minimal(self, name: str, cls: type, ctx: TransformContext) -> A.BasicTransform:
        """
        Instantiate with minimal, warning-free params (avoid deprecated/invalid args).
        """
        H, W = ctx.image.shape[:2]
        short = min(H, W)
        crop_h = min(H, max(8, int(short * 0.8)))
        crop_w = min(W, max(8, int(short * 0.8)))
        x1 = max(0, (W - crop_w) // 2)
        y1 = max(0, (H - crop_h) // 2)
        x2, y2 = min(W, x1 + crop_w), min(H, y1 + crop_h)

        n = name

        # Geometry/crops
        if n == "CenterCrop":      return cls(height=crop_h, width=crop_w, p=1.0)
        if n == "RandomCrop":      return cls(height=crop_h, width=crop_w, p=1.0)
        if n == "RandomResizedCrop":
            # Albumentations v2 requires size=(h,w)
            return cls(size=(crop_h, crop_w), scale=(0.5, 1.0), ratio=(0.75, 1.333), p=1.0)
        if n == "Crop":            return cls(x_min=x1, y_min=y1, x_max=x2, y_max=y2, p=1.0)
        if n == "Resize":          return cls(height=max(8, int(H * 0.8)), width=max(8, int(W * 0.8)), p=1.0)
        if n == "PadIfNeeded":     return cls(min_height=H + 16, min_width=W + 16, p=1.0)
        if n == "SmallestMaxSize": return cls(max_size=max(8, int(short * 0.75)), p=1.0)
        if n == "LongestMaxSize":  return cls(max_size=max(8, int(short * 0.75)), p=1.0)

        # Rotations/flips
        if n in ("Rotate", "SafeRotate"):
            border_mode = cv2.BORDER_REFLECT101 if _cv2_available() else 4
            return self._get_albu_class("Rotate")(limit=20, border_mode=border_mode, p=1.0)
        if n in ("RandomRotate90", "HorizontalFlip", "VerticalFlip", "Transpose"):
            return cls(p=1.0)

        # BBox-related
        if n == "BBoxSafeRandomCrop":         return cls(p=1.0)
        if n == "RandomSizedBBoxSafeCrop":    return cls(height=crop_h, width=crop_w, p=1.0)
        if n == "AtLeastOneBBoxRandomCrop":   return cls(height=crop_h, width=crop_w, p=1.0)
        if n == "RandomCropNearBBox":         return cls(max_part_shift=0.3, p=1.0)

        # Warps/distortions
        if n in ("ElasticTransform", "ThinPlateSpline", "PiecewiseAffine", "OpticalDistortion",
                 "GridDistortion", "Affine", "ShiftScaleRotate", "RandomScale", "Perspective"):
            return cls(p=1.0)
        if n == "GridElasticDeform":
            return cls(num_grid_xy=(4, 4), magnitude=5, p=1.0)

        # Color/intensity
        if n in ("CLAHE", "Equalize", "AutoContrast", "ColorJitter", "RandomBrightnessContrast", "RandomGamma",
                 "HueSaturationValue", "RGBShift", "FancyPCA", "Posterize", "Solarize", "InvertImg",
                 "ToGray", "ToRGB", "ToSepia", "RandomToneCurve"):
            return cls(p=1.0)

        # Blur/sharpen
        if n in ("Blur", "GaussianBlur", "MedianBlur", "MotionBlur", "GlassBlur", "AdvancedBlur", "Defocus",
                 "Sharpen", "Emboss", "ZoomBlur"):
            return cls(p=1.0)

        # Noise
        if n in ("GaussNoise", "ISONoise", "MultiplicativeNoise", "RingingOvershoot"):
            # Avoid version-specific params: use defaults only
            return cls(p=1.0)

        # Dropout/channel ops
        if n in ("CoarseDropout", "ConstrainedCoarseDropout", "GridDropout", "PixelDropout", "ChannelDropout",
                 "ChannelShuffle", "SelectiveChannelTransform"):
            return cls(p=1.0)

        # Weather/effects
        if n in ("RandomFog", "RandomSnow", "RandomRain", "RandomSunFlare", "RandomShadow", "Spatter",
                 "RandomGravel"):
            return cls(p=1.0)

        # Domain/misc
        if n in ("ImageCompression", "Normalize", "FromFloat", "ToFloat", "HEStain", "Illumination",
                 "PlanckianJitter", "PlasmaBrightnessContrast", "PlasmaShadow", "Superpixels", "Morphological",
                 "NoOp", "D4", "XYMasking", "ToTensorV2"):
            return cls(p=1.0)

        try:
            return cls(p=1.0)
        except Exception:
            raise RuntimeError(f"No default instantiation strategy for {name}")

    def _albu_or_custom(self, name: str, ctx: TransformContext) -> Tuple[Optional[A.BasicTransform], bool, Optional[str]]:
        if name in self.meta_classes:
            return (None, False, "Meta/base class (not a concrete transform)")
        if name in self.non_image_2d_transforms:
            return (None, False, "Transform requires 3D/audio/volume; not applicable to 2D images")
        if name in self.mask_required:
            return (None, False, "Transform requires a mask; not available in this task")

        alias = self.aliases.get(name, name)
        if alias is None:
            # forced custom
            if name == "Pad":
                return (self._custom_pad(ctx), True, None)
            if name == "RandomGridShuffle":
                return (self._custom_random_grid_shuffle(ctx), True, None)

        if name in self.v2_metadata_transforms:
            if name == "Mosaic":                      return (self._custom_mosaic(ctx), True, None)
            if name == "HistogramMatching":           return (self._custom_histogram_matching(ctx), True, None)
            if name == "OverlayElements":             return (self._custom_overlay_elements(ctx), True, None)
            if name == "PixelDistributionAdaptation": return (self._custom_pda(ctx), True, None)
            if name == "FDA":                         return (self._custom_fda(ctx), True, None)

        # Additional customs (to avoid instantiation failures and warnings)
        if name == "AdditiveNoise":                 return (self._custom_additive_noise(ctx), True, None)
        if name == "ShotNoise":                     return (self._custom_shot_noise(ctx), True, None)
        if name == "SaltAndPepper":                 return (self._custom_salt_and_pepper(ctx), True, None)
        if name == "UnsharpMask":                   return (self._custom_unsharp_mask(ctx), True, None)
        if name == "Emboss" and not (HAVE_ALBU and hasattr(A, "Emboss")):
            return (self._custom_emboss(ctx), True, None)
        if name == "ZoomBlur" and not (HAVE_ALBU and hasattr(A, "ZoomBlur")):
            return (self._custom_zoom_blur(ctx), True, None)
        if name == "ChromaticAberration":           return (self._custom_chromatic_aberration(ctx), True, None)
        if name == "CropAndPad":                    return (self._custom_crop_and_pad(ctx), True, None)
        if name == "CropNonEmptyMaskIfExists":      return (self._custom_crop_non_empty_mask(ctx), True, None)
        if name == "RandomSizedCrop":               return (self._custom_random_sized_crop(ctx), True, None)
        if name == "TextImage":                     return (self._custom_text_image(ctx), True, None)
        if name == "SelectiveChannelTransform" and not hasattr(A, "SelectiveChannelTransform"):
            return (self._custom_selective_channel_transform(ctx), True, None)

        if HAVE_ALBU:
            target = alias
            cls = self._get_albu_class(target)
            if cls is not None:
                try:
                    t = self._instantiate_minimal(name, cls, ctx)
                    return (t, False, None)
                except Exception as e:
                    return (None, False, f"Failed to instantiate: {e}")

        return (None, False, "Transform not available in this environment")

    def build(self, name: str, ctx: TransformContext) -> Tuple[Optional[A.BasicTransform], bool, Optional[str]]:
        return self._albu_or_custom(name, ctx)


# ------------------------------------------
# Executor
# ------------------------------------------

class AugmentationExecutor:
    def __init__(self, out_dir: Path, save_ext: str = ".png", jpeg_quality: int = 95, verbose: bool = True):
        self.out_dir = _ensure_dir(out_dir)
        self.save_ext = save_ext if save_ext.startswith(".") else f".{save_ext}"
        self.jpeg_quality = jpeg_quality
        self.verbose = verbose
        self.console = _get_console(verbose)
        self.factory = TransformFactory(verbose=verbose)

    def _transform_expects_bboxes(self, tname: str) -> bool:
        # Include ConstrainedCoarseDropout to avoid "no bboxes/mask provided" warnings
        return tname in {
            "BBoxSafeRandomCrop", "RandomCropNearBBox", "RandomSizedBBoxSafeCrop",
            "AtLeastOneBBoxRandomCrop", "ConstrainedCoarseDropout"
        }

    def apply_all(
        self,
        images: List[np.ndarray],
        image_names: List[str],
        only_transforms: Optional[List[str]] = None,
        only_categories: Optional[List[str]] = None,
        seed: Optional[int] = None,
        resize_to: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        _seed_everything(seed)
        start = time.time()

        if resize_to is not None:
            rw, rh = resize_to
            images = [_resize_image(img, width=rw, height=rh) for img in images]

        transforms_to_run = self.factory.requested_transforms[:]
        if only_transforms:
            include = set([t.strip() for t in only_transforms if t.strip()])
            transforms_to_run = [t for t in transforms_to_run if t in include]
        if only_categories:
            catset = set([c.strip().lower() for c in only_categories if c.strip()])
            selected = set()
            for cat in catset:
                for k, v in self.factory.categories.items():
                    if k.lower() == cat:
                        selected.update(v)
            transforms_to_run = [t for t in transforms_to_run if t in selected]

        extra_images = [im for im in images[1:]] if len(images) > 1 else []

        run_results: Dict[str, Any] = {
            "metadata_version": "2.0",
            "started_at": _now_iso(),
            "seed": seed,
            "resize": resize_to,
            "library": {
                "albumentations": ALBU_VER,
                "opencv": cv2.__version__ if _cv2_available() else "not-installed",
                "numpy": np.__version__,
                "rich": "installed" if Console is not None else "not-installed",
                "skimage": "installed" if HAVE_SKIMAGE else "not-installed",
                "python": sys.version,
            },
            "results": [],
        }

        status_table = None
        if self.verbose and Table is not None:
            status_table = Table("Image", "Transform", "Status", "Reason/Note", "Time(ms)", "Saved", box=box.SIMPLE_HEAVY)
            self.console.print(Panel(Text("Augmentation run starting", style="bold cyan"), title="Info", border_style="cyan"))

        for img, img_name in zip(images, image_names):
            bboxes, labels = _synthetic_bboxes_for_image(img)
            ctx = TransformContext(image=img, extra_images=extra_images, seed=seed, bboxes=bboxes, bbox_labels=labels)
            for tname in transforms_to_run:
                res = self._apply_one_transform(img, img_name, tname, ctx)
                run_results["results"].append(dataclasses.asdict(res))
                if status_table is not None:
                    status_table.add_row(
                        img_name,
                        tname,
                        res.status,
                        res.reason or "",
                        f"{res.time_ms:.1f}" if res.time_ms is not None else "",
                        os.path.basename(res.save_path) if res.save_path else "",
                    )

        if status_table is not None:
            self.console.print(status_table)

        run_results["finished_at"] = _now_iso()
        run_results["elapsed_sec"] = round(time.time() - start, 3)

        meta_path = self.out_dir / "metadata_run.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(run_results), f, indent=2)

        if self.verbose:
            total = len(run_results["results"])
            applied = sum(1 for r in run_results["results"] if r["status"] == "applied")
            skipped = sum(1 for r in run_results["results"] if r["status"] == "skipped")
            errors = sum(1 for r in run_results["results"] if r["status"] == "error")
            self.console.print(Panel.fit(
                Text(f"Completed. Metadata: {meta_path}\nTransforms: total={total}, applied={applied}, skipped={skipped}, errors={errors}", style="bold green" if errors == 0 else "bold yellow"),
                title="Summary",
                border_style="green" if errors == 0 else "yellow",
            ))

        return run_results

    def _apply_one_transform(self, img: np.ndarray, img_name: str, tname: str, ctx: TransformContext) -> TransformResult:
        res = TransformResult(name=tname, status="skipped", input_shape=tuple(img.shape))

        trans, is_custom, reason = self.factory.build(tname, ctx)
        if trans is None:
            res.status = "skipped"
            res.reason = reason or "Unavailable"
            return res

        if not HAVE_ALBU:
            res.status = "skipped"
            res.reason = "Albumentations not installed; only custom transforms can run"
            return res

        # BBoxes if needed (also for ConstrainedCoarseDropout to avoid warnings)
        bbox_params = None
        if ctx.bboxes is not None and self._transform_expects_bboxes(tname):
            bbox_params = A.BboxParams(format=ctx.bbox_mode, label_fields=["bbox_labels"])

        comp = A.ReplayCompose([trans], bbox_params=bbox_params, p=1.0)
        data: Dict[str, Any] = {"image": img}
        if bbox_params is not None:
            data["bboxes"] = ctx.bboxes
            data["bbox_labels"] = ctx.bbox_labels or ["bbox"] * len(ctx.bboxes)
            if tname == "RandomCropNearBBox":
                data["cropping_bbox"] = ctx.bboxes[0]

        t0 = time.time()
        try:
            out = comp(**data)
            out_img = out["image"]
            res.replay = out.get("replay", None)
            res.status = "applied"
            res.custom = is_custom

            if isinstance(out_img, np.ndarray):
                save_img = out_img
            else:
                try:
                    import torch
                    if isinstance(out_img, torch.Tensor):
                        arr = out_img.detach().cpu().numpy()
                        if arr.ndim == 3 and arr.shape[0] in (1, 3):
                            arr = np.transpose(arr, (1, 2, 0))
                        save_img = (arr * 255.0).clip(0, 255).astype(np.uint8)
                    else:
                        save_img = np.array(out_img)
                except Exception:
                    save_img = None

            if save_img is None:
                res.status = "skipped"
                res.reason = "Produced a non-numpy output; skipping save"
                return res

            save_img = _to_uint8_image(save_img)
            res.output_shape = tuple(save_img.shape)
            res.time_ms = (time.time() - t0) * 1000.0

            base = Path(img_name).stem
            out_name = f"{_sanitize_name(base)}__{_sanitize_name(tname)}"
            save_path = self.out_dir / out_name
            save_path = _save_image(save_path, save_img, self.save_ext, self.jpeg_quality)
            res.save_path = str(save_path)
            return res

        except Exception as e:
            res.status = "error"
            res.reason = f"{type(e).__name__}: {e}"
            return res


# ------------------------------------------
# CLI
# ------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="augment_all.py",
        description="Apply a large catalog of augmentations to image(s) with zero warnings and JSON-safe metadata.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", type=str, required=True, help="Path to image file or directory with images")
    p.add_argument("--out", type=str, required=True, help="Output directory (augmented images will be saved here)")
    p.add_argument("--resize", type=int, nargs=2, metavar=("W", "H"), help="Optional resize WxH before augmentations")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument("--verbose", action="store_true", help="Verbose rich console")
    p.add_argument("--save-ext", type=str, default=".png", help="Output extension (.png or .jpg)")
    p.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality for .jpg/.jpeg")
    p.add_argument("--only", type=str, default=None, help="Comma-separated transform names to apply")
    p.add_argument("--only-category", type=str, default=None, help="Comma-separated categories (e.g., color,geometry)")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    console = _get_console(args.verbose)

    inp = Path(args.input).expanduser().resolve()
    out_dir = _ensure_dir(args.out)

    images: List[np.ndarray] = []
    image_names: List[str] = []

    if inp.is_file():
        images.append(_read_image(inp))
        image_names.append(inp.name)
    elif inp.is_dir():
        paths = _list_images_in_directory(inp)
        if not paths:
            if args.verbose:
                console.print("[yellow]No images found in the input directory[/yellow]")
            return 2
        for p in paths:
            try:
                images.append(_read_image(p))
                image_names.append(p.name)
            except Exception as e:
                if args.verbose:
                    console.print(f"[red]Failed to read {p}[/red]: {e}")
        if not images:
            if args.verbose:
                console.print("[red]No readable images found.[/red]")
            return 2
    else:
        if args.verbose:
            console.print(f"[red]Input path does not exist:[/red] {inp}")
        return 2

    resize_to = None
    if args.resize:
        resize_to = (int(args.resize[0]), int(args.resize[1]))

    only_transforms = None
    if args.only:
        only_transforms = [s.strip() for s in args.only.split(",") if s.strip()]
    only_categories = None
    if args.only_category:
        only_categories = [s.strip() for s in args.only_category.split(",") if s.strip()]

    if args.verbose:
        banner = Text(f"Augmentation Session — {len(images)} image(s), output: {str(out_dir)}", style="bold white")
        console.print(Panel(banner, title="Start", border_style="blue"))
        if only_transforms:
            console.print(f"[cyan]Only transforms:[/cyan] {only_transforms}")
        if only_categories:
            console.print(f"[cyan]Only categories:[/cyan] {only_categories}")
        if resize_to:
            console.print(f"[cyan]Resize:[/cyan] {resize_to}")
        if not HAVE_ALBU:
            console.print("[yellow]Albumentations is not installed. Only custom subset will be available.[/yellow]")

    executor = AugmentationExecutor(
        out_dir=out_dir,
        save_ext=args.save_ext,
        jpeg_quality=args.jpeg_quality,
        verbose=args.verbose,
    )

    executor.apply_all(
        images=images,
        image_names=image_names,
        only_transforms=only_transforms,
        only_categories=only_categories,
        seed=args.seed,
        resize_to=resize_to,
    )
    return 0


if __name__ == "__main__":
    # Example quick test (uncomment & adjust):
    # sys.argv = ["", "--input", "path/to/image.jpg", "--out", "./aug_out", "--resize", "1024", "1024", "--verbose"]
    sys.exit(main())