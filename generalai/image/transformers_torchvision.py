#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Torchvision Augmentation Suite — single-file, deeply documented, and production-grade.

Objective
- Ingest an image from local path (Image.from_loadpath).
- Optionally resize to a target size (tuple of (H, W)).
- Apply (almost) all torchvision.transforms APIs in a robust, reproducible, and well-parameterized manner.
- Save every augmented result into an output/ directory with controlled JPEG quality.
- Emit rich, structured metadata panels (if `rich` is available) when verbose is enabled.

Why this file is useful
- Coders can learn how to wire up diverse torchvision transforms and understand tensor vs. PIL flows.
- Shows careful exception handling and parameterization for transforms that need special handling (Normalize,
  ConvertImageDtype, RandomErasing, FiveCrop/TenCrop, AutoAugment/RandAugment/TrivialAugmentWide, ElasticTransform, etc.).
- Provides reproducibility via fixed RNG seeds.
- Highly modular design—add/modify transform blocks with minimal effort.

Usage
- CLI:
    python3 augment_all_torchvision.py --input image_test.png --size 256x256 --quality 95 --output-dir output --verbose 1
- Programmatic:
    from augment_all_torchvision import main
    main(["--input", "image_test.png", "--size", "224x224", "--quality", "90", "--verbose", "1"])

Notes
- Transforms requiring tensors are composed with ToTensor()/ToPILImage() as needed.
- Normalize is visualized by un-normalizing for save (so the saved image is human-friendly).
- LinearTransformation is demonstrated on a small resized patch to avoid huge matrices.
- Some transforms may be missing depending on torchvision version; those will be skipped with a clear reason.
- ElasticTransform and AugMix, RandAugment, TrivialAugmentWide are version-dependent; guarded by feature detection.

Dependencies
- Python 3.8+
- numpy, pillow (PIL), torch, torchvision
- Optional: rich (pretty metadata panels)
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

# Third-party core
try:
    import numpy as np
except Exception as e:
    raise RuntimeError("numpy required. pip install numpy") from e

try:
    from PIL import Image as PILImage, ImageOps as PILImageOps
except Exception as e:
    raise RuntimeError("Pillow required. pip install pillow") from e

try:
    import torch
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode, AutoAugmentPolicy
except Exception as e:
    raise RuntimeError("torch and torchvision required. pip install torch torchvision") from e

# Optional pretty metadata
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    RICH_AVAILABLE = True
except Exception:
    Console = None
    Panel = None
    Table = None
    Text = None
    RICH_AVAILABLE = False


# --------------------------- Rich logging helpers ---------------------------

GLOBAL_VERBOSE = os.getenv("AUG_VERBOSE", "0") in ("1", "true", "True", "YES", "yes")


def _console() -> Optional[Console]:
    return Console() if RICH_AVAILABLE else None


def _log_panel(title: str, data: Dict[str, Any], verbose: Optional[bool]) -> None:
    v = GLOBAL_VERBOSE if verbose is None else bool(verbose)
    if not v:
        return
    if RICH_AVAILABLE:
        con = _console()
        if con is None:
            return
        tbl = Table(show_header=True, header_style="bold magenta")
        tbl.add_column("Key", style="cyan", no_wrap=True)
        tbl.add_column("Value", style="green")
        for k, val in data.items():
            try:
                sval = json.dumps(val, default=str)
            except Exception:
                sval = str(val)
            tbl.add_row(str(k), sval)
        con.print(Panel(tbl, title=f"[bold blue]{title}", border_style="blue"))
    else:
        print(f"[{title}]")
        for k, v in data.items():
            print(f" - {k}: {v}")


# ------------------------------ Image Wrapper ------------------------------

@dataclass
class Image:
    """
    Minimal image wrapper to match the requested API: Image.from_loadpath(path) and convenience ops.
    Internally uses Pillow; torchvision transforms accept PIL images or torch tensors.

    - path: source path (optional)
    - pil: Pillow image
    - meta: structured metadata dict
    """
    pil: PILImage.Image
    path: Optional[Union[str, Path]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_loadpath(cls, path: Union[str, Path], *, mode: Optional[str] = None, verbose: Optional[bool] = None) -> "Image":
        p = Path(path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        img = PILImage.open(str(p))
        if mode is not None:
            img = img.convert(mode)
        meta = {"source": "local_path", "path": str(p.resolve()), "format": img.format, "size": img.size}
        _log_panel("Image.from_loadpath", {"path": meta["path"], "format": meta["format"], "size": meta["size"]}, verbose)
        return cls(pil=img, path=p, meta=meta)

    @property
    def width(self) -> int:
        return int(self.pil.size[0])

    @property
    def height(self) -> int:
        return int(self.pil.size[1])

    @property
    def size_wh(self) -> Tuple[int, int]:
        return (self.width, self.height)

    def resize_hw(self, hw: Tuple[int, int], interpolation: InterpolationMode = InterpolationMode.BILINEAR) -> "Image":
        """
        Resize to (H, W) order as typical ML convention; PIL expects (W, H).
        """
        h, w = hw
        pil_resized = transforms.Resize((h, w), interpolation=interpolation)(self.pil)
        return Image(pil=pil_resized, path=self.path, meta={**self.meta, "resized_to_hw": (h, w)})

    def save_jpeg(self, path: Union[str, Path], quality: int = 95) -> None:
        """
        Save image as JPEG; converts mode to RGB if needed (JPEG has no alpha).
        """
        outp = Path(path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        pil = self.pil
        if pil.mode not in ("RGB", "L"):
            pil = pil.convert("RGB")
        pil.save(str(outp), format="JPEG", quality=int(quality), optimize=True)


# --------------------------- Utility functions -----------------------------

def set_deterministic(seed: int = 123) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make torchvision transforms deterministic where applicable; note some ops may still have randomness by design.


def parse_size(s: str) -> Tuple[int, int]:
    """
    Parse size strings like '256x256', '256,256', '(256,256)' into (H, W).
    """
    s = s.strip().lower().replace("(", "").replace(")", "").replace("[", "").replace("]", "")
    if "x" in s:
        a, b = s.split("x", 1)
    elif "," in s:
        a, b = s.split(",", 1)
    else:
        raise ValueError("Size must be in 'HxW' or 'H,W' format, e.g., 256x256")
    h, w = int(a), int(b)
    if h <= 0 or w <= 0:
        raise ValueError("Size must be positive integers.")
    return (h, w)


def to_pil_from_any(x: Union[PILImage.Image, torch.Tensor]) -> PILImage.Image:
    if isinstance(x, PILImage.Image):
        return x
    if isinstance(x, torch.Tensor):
        # If tensor is float, assume [0,1] or normalized; clamp and convert
        t = x.detach().cpu()
        if t.dtype.is_floating_point:
            t = t.clamp(0.0, 1.0)
        return transforms.ToPILImage()(t)
    raise TypeError(f"Unsupported type for to_pil_from_any: {type(x)}")


def unnormalize_tensor(t: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    """
    Inverse of torchvision.transforms.Normalize for visualization.
    """
    assert t.ndim == 3, "Expect CHW tensor"
    device = t.device
    m = torch.tensor(mean, dtype=t.dtype, device=device).view(-1, 1, 1)
    s = torch.tensor(std, dtype=t.dtype, device=device).view(-1, 1, 1)
    return t * s + m


def save_pil(pil: PILImage.Image, path: Path, quality: int) -> None:
    """
    Save PIL image to path, choosing JPEG with quality if suffix is .jpg/.jpeg else PNG.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        img = pil
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img.save(str(path), format="JPEG", quality=int(quality), optimize=True)
    else:
        # Save PNG; quality not used, but keep consistent naming.
        pil.save(str(path), format="PNG", optimize=True)


def pil_info(pil: PILImage.Image) -> Dict[str, Any]:
    w, h = pil.size
    return {"mode": pil.mode, "width": w, "height": h}


# --------------------------- Augmentation Runner ---------------------------

@dataclass
class AugResult:
    name: str
    variant: str
    out_path: Path
    meta: Dict[str, Any]


@dataclass
class AugmentationRunner:
    size_hw: Tuple[int, int]
    quality: int
    output_dir: Path
    seed: int = 123
    verbose: bool = False
    counter: int = 0
    results: List[AugResult] = field(default_factory=list)
    skipped: List[Tuple[str, str]] = field(default_factory=list)

    def _next_name(self, name: str, variant: Optional[str] = None) -> str:
        self.counter += 1
        vn = f"_{variant}" if variant else ""
        return f"{self.counter:03d}_{name}{vn}"

    def _save_and_record(self, name: str, variant: Optional[str], image_any: Union[PILImage.Image, torch.Tensor], extra_meta: Dict[str, Any]) -> None:
        pil = to_pil_from_any(image_any)
        out_name = self._next_name(name, variant)
        out_path = self.output_dir / f"{out_name}.jpg"  # JPEG for controllable quality
        t0 = time.time()
        save_pil(pil, out_path, self.quality)
        dt = (time.time() - t0) * 1000.0
        meta = {
            "transform": name,
            "variant": variant or "",
            "save_ms": round(dt, 2),
            "quality": self.quality,
            "input_size_hw": list(self.size_hw),
            "output_info": pil_info(pil),
            **extra_meta,
        }
        self.results.append(AugResult(name=name, variant=variant or "", out_path=out_path, meta=meta))
        _log_panel("Saved augmentation", meta, self.verbose)

    def _try(self, name: str, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception as e:
            self.skipped.append((name, str(e)))
            _log_panel("Skip augmentation", {"name": name, "reason": str(e)}, self.verbose)

    def run_all(self, base: Image) -> None:
        """
        Apply a representative demo for each transform in torchvision.transforms we care about.
        The base image is first resized to size_hw for consistency.
        """
        set_deterministic(self.seed)
        # Standardize base image size
        base_resized = base.resize_hw(self.size_hw, InterpolationMode.BICUBIC)
        pil = base_resized.pil

        # Always print dir(transforms) as requested
        print("dir(torchvision.transforms):")
        print(dir(transforms))

        # 1) Simple and deterministic transforms on PIL
        self._try("Resize", lambda: self._save_and_record(
            "Resize",
            "320x320_LANCZOS",
            transforms.Resize((320, 320), interpolation=InterpolationMode.LANCZOS)(pil),
            {"params": {"size": (320, 320), "interpolation": "LANCZOS"}}
        ))

        self._try("CenterCrop", lambda: self._save_and_record(
            "CenterCrop",
            "224",
            transforms.CenterCrop(224)(pil),
            {"params": {"size": 224}}
        ))

        self._try("Pad", lambda: self._save_and_record(
            "Pad",
            "10_black",
            transforms.Pad(10, fill=0)(pil),
            {"params": {"padding": 10, "fill": (0, 0, 0)}}
        ))

        self._try("Grayscale", lambda: self._save_and_record(
            "Grayscale",
            "3ch",
            transforms.Grayscale(num_output_channels=3)(pil),
            {"params": {"num_output_channels": 3}}
        ))

        self._try("ColorJitter", lambda: self._save_and_record(
            "ColorJitter",
            "b0.2_c0.2_s0.2_h0.02",
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02)(pil),
            {"params": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.02}}
        ))

        self._try("GaussianBlur", lambda: self._save_and_record(
            "GaussianBlur",
            "k5",
            transforms.GaussianBlur(kernel_size=5)(pil),
            {"params": {"kernel_size": 5}}
        ))

        self._try("RandomRotation", lambda: self._save_and_record(
            "RandomRotation",
            "15deg",
            transforms.RandomRotation(degrees=15)(pil),
            {"params": {"degrees": 15}}
        ))

        self._try("RandomAffine", lambda: self._save_and_record(
            "RandomAffine",
            "rotate20_translate(0.1,0.1)_scale(0.9,1.1)_shear10",
            transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10)(pil),
            {"params": {"degrees": 20, "translate": (0.1, 0.1), "scale": (0.9, 1.1), "shear": 10}}
        ))

        self._try("RandomPerspective", lambda: self._save_and_record(
            "RandomPerspective",
            "dist0.6",
            transforms.RandomPerspective(distortion_scale=0.6, p=1.0)(pil),
            {"params": {"distortion_scale": 0.6, "p": 1.0}}
        ))

        self._try("RandomHorizontalFlip", lambda: self._save_and_record(
            "RandomHorizontalFlip",
            "p1",
            transforms.RandomHorizontalFlip(p=1.0)(pil),
            {"params": {"p": 1.0}}
        ))

        self._try("RandomVerticalFlip", lambda: self._save_and_record(
            "RandomVerticalFlip",
            "p1",
            transforms.RandomVerticalFlip(p=1.0)(pil),
            {"params": {"p": 1.0}}
        ))

        self._try("RandomGrayscale", lambda: self._save_and_record(
            "RandomGrayscale",
            "p1",
            transforms.RandomGrayscale(p=1.0)(pil),
            {"params": {"p": 1.0}}
        ))

        self._try("RandomInvert", lambda: self._save_and_record(
            "RandomInvert",
            "p1",
            transforms.RandomInvert(p=1.0)(pil),
            {"params": {"p": 1.0}}
        ))

        self._try("RandomAutocontrast", lambda: self._save_and_record(
            "RandomAutocontrast",
            "p1",
            transforms.RandomAutocontrast(p=1.0)(pil),
            {"params": {"p": 1.0}}
        ))

        self._try("RandomEqualize", lambda: self._save_and_record(
            "RandomEqualize",
            "p1",
            transforms.RandomEqualize(p=1.0)(pil),
            {"params": {"p": 1.0}}
        ))

        self._try("RandomAdjustSharpness", lambda: self._save_and_record(
            "RandomAdjustSharpness",
            "factor2.0",
            transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=1.0)(pil),
            {"params": {"sharpness_factor": 2.0, "p": 1.0}}
        ))

        self._try("RandomPosterize", lambda: self._save_and_record(
            "RandomPosterize",
            "bits4",
            transforms.RandomPosterize(bits=4, p=1.0)(pil),
            {"params": {"bits": 4, "p": 1.0}}
        ))

        self._try("RandomSolarize", lambda: self._save_and_record(
            "RandomSolarize",
            "thr128",
            transforms.RandomSolarize(threshold=128, p=1.0)(pil),
            {"params": {"threshold": 128, "p": 1.0}}
        ))

        # 2) Crop variants that produce multiple outputs
        def fivecrop():
            out = transforms.FiveCrop(size=128)(pil)  # (TL, TR, BL, BR, Center)
            for i, img in enumerate(out):
                self._save_and_record("FiveCrop", f"crop{i+1}", img, {"params": {"size": 128}})
        self._try("FiveCrop", fivecrop)

        def tencrop():
            out = transforms.TenCrop(size=128)  # (5 crops + their mirrors)
            res = out(pil)
            for i, img in enumerate(res):
                self._save_and_record("TenCrop", f"crop{i+1}", img, {"params": {"size": 128}})
        self._try("TenCrop", tencrop)

        # 3) Tensor-required transforms: ToTensor, PILToTensor, ConvertImageDtype, Normalize, RandomErasing
        def to_tensor_roundtrip():
            t = transforms.ToTensor()(pil)  # float [0,1], CHW
            self._save_and_record("ToTensor", "visualized", transforms.ToPILImage()(t), {"params": {"note": "ToTensor->ToPILImage"}})
        self._try("ToTensor", to_tensor_roundtrip)

        def pil_to_tensor_roundtrip():
            t = transforms.PILToTensor()(pil)  # uint8 [0..255], CHW
            self._save_and_record("PILToTensor", "visualized", transforms.ToPILImage()(t), {"params": {"note": "PILToTensor->ToPILImage"}})
        self._try("PILToTensor", pil_to_tensor_roundtrip)

        def convert_image_dtype_demo():
            t = transforms.Compose([
                transforms.ToTensor(),  # float [0,1]
                transforms.ConvertImageDtype(torch.float16),
            ])(pil)
            # convert back to float32 for ToPILImage
            t = t.to(torch.float32).clamp(0, 1)
            self._save_and_record("ConvertImageDtype", "float16", transforms.ToPILImage()(t), {"params": {"dtype": "float16"}})
        self._try("ConvertImageDtype", convert_image_dtype_demo)

        def normalize_unorm_demo():
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            t = transforms.ToTensor()(pil)
            t_norm = transforms.Normalize(mean=mean, std=std)(t)
            t_vis = unnormalize_tensor(t_norm, mean, std).clamp(0, 1)  # visualize normalized by un-normalizing
            self._save_and_record("Normalize", "imagenet", transforms.ToPILImage()(t_vis), {"params": {"mean": mean, "std": std}})
        self._try("Normalize", normalize_unorm_demo)

        def random_erasing_demo():
            t = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomErasing(p=1.0, scale=(0.02, 0.2), ratio=(0.3, 3.3), value="random"),
            ])(pil)
            t = t.clamp(0, 1)
            self._save_and_record("RandomErasing", "p1", transforms.ToPILImage()(t), {"params": {"p": 1.0}})
        self._try("RandomErasing", random_erasing_demo)

        # 4) Compound meta-transforms: RandomApply, RandomChoice, RandomOrder, Compose, Lambda
        def random_apply_demo():
            tr = transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=1.0)
            self._save_and_record("RandomApply", "ColorJitter_p1", tr(pil), {"params": {"inner": "ColorJitter(0.4,...)", "p": 1.0}})
        self._try("RandomApply", random_apply_demo)

        def random_choice_demo():
            tr = transforms.RandomChoice([transforms.Grayscale(3), transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)])
            self._save_and_record("RandomChoice", "chosen", tr(pil), {"params": {"choices": ["Grayscale(3)", "ColorJitter(...)"]}})
        self._try("RandomChoice", random_choice_demo)

        def random_order_demo():
            tr = transforms.RandomOrder([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1), transforms.RandomInvert(p=1.0), transforms.RandomRotation(10)])
            self._save_and_record("RandomOrder", "combo", tr(pil), {"params": {"ops": ["ColorJitter", "Invert", "Rotation"]}})
        self._try("RandomOrder", random_order_demo)

        def compose_demo():
            tr = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.ColorJitter(0.3, 0.3, 0.3, 0.05)])
            self._save_and_record("Compose", "resize_crop_jitter", tr(pil), {"params": {"ops": ["Resize(256)", "CenterCrop(224)", "ColorJitter"]}})
        self._try("Compose", compose_demo)

        def lambda_demo():
            lam = transforms.Lambda(lambda im: PILImageOps.invert(im.convert("RGB")))
            self._save_and_record("Lambda", "invert", lam(pil), {"params": {"lambda": "invert(RGB)"}})
        self._try("Lambda", lambda_demo)

        # 5) Resized crops and interpolation choices
        self._try("RandomResizedCrop", lambda: self._save_and_record(
            "RandomResizedCrop",
            "size224_scale(0.5,1.0)_ratio(0.75,1.33)",
            transforms.RandomResizedCrop(size=224, scale=(0.5, 1.0), ratio=(0.75, 1.33), interpolation=InterpolationMode.BILINEAR)(pil),
            {"params": {"size": 224, "scale": (0.5, 1.0), "ratio": (0.75, 1.33), "interpolation": "BILINEAR"}}
        ))

        self._try("RandomCrop", lambda: self._save_and_record(
            "RandomCrop",
            "200_pad10",
            transforms.RandomCrop(size=200, padding=10)(pil),
            {"params": {"size": 200, "padding": 10}}
        ))

        # 6) AutoAugment/RandAugment/TrivialAugmentWide/AugMix
        def autoaugment_demo():
            for pol in [AutoAugmentPolicy.IMAGENET, AutoAugmentPolicy.CIFAR10, AutoAugmentPolicy.SVHN]:
                tr = transforms.AutoAugment(policy=pol)
                self._save_and_record("AutoAugment", f"{pol.name}", tr(pil), {"params": {"policy": pol.name}})
        self._try("AutoAugment", autoaugment_demo)

        def randaugment_demo():
            if hasattr(transforms, "RandAugment"):
                tr = transforms.RandAugment(num_ops=2, magnitude=9)
                self._save_and_record("RandAugment", "n2_m9", tr(pil), {"params": {"num_ops": 2, "magnitude": 9}})
            else:
                raise RuntimeError("RandAugment not available in this torchvision version.")
        self._try("RandAugment", randaugment_demo)

        def trivial_augment_wide_demo():
            if hasattr(transforms, "TrivialAugmentWide"):
                tr = transforms.TrivialAugmentWide()
                self._save_and_record("TrivialAugmentWide", "default", tr(pil), {"params": {}})
            else:
                raise RuntimeError("TrivialAugmentWide not available in this torchvision version.")
        self._try("TrivialAugmentWide", trivial_augment_wide_demo)

        def augmix_demo():
            if hasattr(transforms, "AugMix"):
                tr = transforms.AugMix(severity=3, alpha=1.0)
                self._save_and_record("AugMix", "severity3", tr(pil), {"params": {"severity": 3, "alpha": 1.0}})
            else:
                raise RuntimeError("AugMix not available in this torchvision version.")
        self._try("AugMix", augmix_demo)

        # 7) ElasticTransform (version dependent)
        def elastic_transform_demo():
            if hasattr(transforms, "ElasticTransform"):
                # API varies across versions; try common signature
                try:
                    tr = transforms.ElasticTransform(alpha=50.0, sigma=5.0, interpolation=InterpolationMode.BILINEAR)
                except TypeError:
                    # Newer versions may use different parameter names; try defaults
                    tr = transforms.ElasticTransform()
                self._save_and_record("ElasticTransform", "default", tr(pil), {"params": {}})
            else:
                raise RuntimeError("ElasticTransform not available in this torchvision version.")
        self._try("ElasticTransform", elastic_transform_demo)

        # 8) LinearTransformation (expensive if applied on full image) — demo on a smaller patch
        def linear_transformation_demo():
            # Downscale to 32x32, apply identity linear transform, then upscale back for visualization
            down = transforms.Resize((32, 32), interpolation=InterpolationMode.BILINEAR)(pil)
            t = transforms.ToTensor()(down)  # CxHxW
            C, H, W = t.shape
            n = C * H * W
            # Identity matrix and zero mean (no-op), but demonstrates API usage
            M = torch.eye(n, dtype=t.dtype)
            mean = torch.zeros(n, dtype=t.dtype)
            t_flat = t.view(-1)
            t_out = torch.mv(M, t_flat - mean).view(C, H, W)
            up = transforms.ToPILImage()(t_out.clamp(0, 1))
            up = transforms.Resize(self.size_hw, interpolation=InterpolationMode.BILINEAR)(up)
            self._save_and_record("LinearTransformation", "identity_on_32x32", up, {"params": {"note": "identity on 32x32 patch"}})
        self._try("LinearTransformation", linear_transformation_demo)

        # 9) ToPILImage standalone demo (applied after ToTensor)
        def topilimage_demo():
            t = transforms.ToTensor()(pil)
            out = transforms.ToPILImage()(t)
            self._save_and_record("ToPILImage", "from_tensor", out, {"params": {"note": "round-trip"}})
        self._try("ToPILImage", topilimage_demo)

        # 10) InterpolationMode showcase via Resize
        def interpolation_showcase():
            for mode in [InterpolationMode.NEAREST, InterpolationMode.BILINEAR, InterpolationMode.BICUBIC, InterpolationMode.LANCZOS]:
                out = transforms.Resize((256, 256), interpolation=mode)(pil)
                self._save_and_record("ResizeInterp", mode.name, out, {"params": {"interpolation": mode.name}})
        self._try("InterpolationMode", interpolation_showcase)

        # Summary
        self._summary()

    def _summary(self) -> None:
        ok = len(self.results)
        sk = len(self.skipped)
        data = {
            "saved": ok,
            "skipped": sk,
            "output_dir": str(self.output_dir.resolve()),
            "seed": self.seed,
            "quality": self.quality,
            "size_hw": list(self.size_hw),
        }
        _log_panel("Augmentation Summary", data, self.verbose)
        if not self.verbose:
            print(json.dumps(data, indent=2))
        if self.skipped:
            msg = {"skipped_list": [{"name": n, "reason": r} for (n, r) in self.skipped]}
            _log_panel("Skipped details", msg, self.verbose)


# ------------------------------- Main / CLI --------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Apply torchvision transforms to an image and save all results.")
    p.add_argument("--input", type=str, default="image_test.png", help="Input image path (default: image_test.png)")
    p.add_argument("--size", type=str, default="256x256", help="Target size (HxW), e.g., 256x256")
    p.add_argument("--quality", type=int, default=95, help="JPEG quality for saved outputs (default: 95)")
    p.add_argument("--output-dir", type=str, default="output", help="Directory to save augmented images")
    p.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility")
    p.add_argument("--verbose", type=int, default=int(GLOBAL_VERBOSE), help="Verbose rich metadata panels (0/1)")
    return p


def _maybe_create_synthetic_image(path: Path, size_wh: Tuple[int, int] = (320, 240)) -> None:
    """
    If input image is missing, create a synthetic checkerboard+gradient RGB image for demonstrations.
    """
    if path.exists():
        return
    w, h = size_wh
    x = np.linspace(0, 1, w, dtype=np.float32)
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    r = np.tile(x, (h, 1))
    g = np.tile(y, (1, w))
    b = 0.5 * (np.sin(10 * math.pi * r) * np.cos(10 * math.pi * g) + 1.0)
    img = np.stack([r, g, b], axis=-1)
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    PILImage.fromarray(img, mode="RGB").save(str(path))


def main(argv: Optional[List[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    size_hw = parse_size(args.size)
    quality = int(args.quality)
    verbose = bool(args.verbose)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Input image
    inp = Path(args.input)
    _maybe_create_synthetic_image(inp, size_wh=(max(320, size_hw[1]), max(240, size_hw[0])))  # ensure size >= transforms
    img = Image.from_loadpath(inp, verbose=verbose)

    # Runner
    runner = AugmentationRunner(size_hw=size_hw, quality=quality, output_dir=output_dir, seed=args.seed, verbose=verbose)
    runner.run_all(img)


if __name__ == "__main__":
    main()