import argparse
import io
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import torch

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

try:
    from transformers import (
        AutoConfig,
        AutoModelForZeroShotObjectDetection,
        AutoProcessor,
        CLIPModel,
        CLIPProcessor,
        SiglipModel,
        SiglipProcessor,
    )
    from transformers.utils import logging as hf_logging

    TRANSFORMERS_AVAILABLE = True
    hf_logging.set_verbosity_error()
except Exception:
    TRANSFORMERS_AVAILABLE = False

try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

try:
    from pdf2image import convert_from_path

    PDF2IMAGE_AVAILABLE = True
except Exception:
    PDF2IMAGE_AVAILABLE = False

try:
    import fitz

    PYMUPDF_AVAILABLE = True
except Exception:
    PYMUPDF_AVAILABLE = False

try:
    import open_clip

    OPEN_CLIP_AVAILABLE = True
except Exception:
    OPEN_CLIP_AVAILABLE = False

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

console = Console()


class PipelineType(Enum):
    YOLO = "yolo"
    GROUNDING_DINO = "grounding_dino"
    OWL_VIT = "owl_vit"
    CLIP = "clip"


@dataclass
class BBox:
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    def to_xyxy(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.x2, self.y2

    @staticmethod
    def from_xyxy(xmin: float, ymin: float, xmax: float, ymax: float) -> "BBox":
        xmin, ymin, xmax, ymax = float(xmin), float(ymin), float(xmax), float(ymax)
        x1 = int(round(min(xmin, xmax)))
        y1 = int(round(min(ymin, ymax)))
        w = int(round(max(1, abs(xmax - xmin))))
        h = int(round(max(1, abs(ymax - ymin))))
        return BBox(x=x1, y=y1, w=w, h=h)

    def to_dict(self) -> Dict[str, int]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h, "x2": self.x2, "y2": self.y2}


@dataclass
class Detection:
    id: str
    class_name: str
    score: float
    bbox: BBox
    pipeline: str
    page_no: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessedPage:
    page_no: int
    width: int
    height: int
    image_path: str
    detections: Dict[str, List[Detection]] = field(default_factory=dict)


def _uuid() -> str:
    return uuid.uuid4().hex[:8]


def _device_pref() -> str:
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _to_device(tensors: Dict[str, Any], device: str) -> Dict[str, Any]:
    if not TORCH_AVAILABLE:
        return tensors
    out = {}
    for k, v in tensors.items():
        if hasattr(v, "to"):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _image_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _get_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", 16)
    except Exception:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", 16)
        except Exception:
            return ImageFont.load_default()


def _measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    try:
        tb = draw.textbbox((0, 0), text, font=font)
        return tb[2] - tb[0], tb[3] - tb[1]
    except Exception:
        return draw.textsize(text, font=font)


def _nullcontext():
    class _NC:
        def __enter__(self): return None

        def __exit__(self, exc_type, exc, tb): return False

    return _NC()


def safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


def _expand_box(b: BBox, w_img: int, h_img: int, ratio: float) -> BBox:
    cx = b.x + b.w / 2.0
    cy = b.y + b.h / 2.0
    nw = int(round(b.w * (1.0 + ratio)))
    nh = int(round(b.h * (1.0 + ratio)))
    x1 = int(round(cx - nw / 2.0))
    y1 = int(round(cy - nh / 2.0))
    x1 = max(0, min(x1, w_img - 1))
    y1 = max(0, min(y1, h_img - 1))
    x2 = max(1, min(x1 + nw, w_img))
    y2 = max(1, min(y1 + nh, h_img))
    return BBox.from_xyxy(x1, y1, x2, y2)


def _box_area(b: BBox) -> int:
    return max(0, b.w) * max(0, b.h)


def _iou(b1: BBox, b2: BBox) -> float:
    ax1, ay1, ax2, ay2 = b1.to_xyxy()
    bx1, by1, bx2, by2 = b2.to_xyxy()
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = _box_area(b1) + _box_area(b2) - inter
    if ua <= 0:
        return 0.0
    return inter / ua


def _nms(dets: List[Detection], iou_thr: float) -> List[Detection]:
    if not dets:
        return []
    dets_sorted = sorted(dets, key=lambda d: d.score, reverse=True)
    kept: List[Detection] = []
    for d in dets_sorted:
        ok = True
        for k in kept:
            if _iou(d.bbox, k.bbox) >= iou_thr:
                ok = False
                break
        if ok:
            kept.append(d)
    return kept


def _soft_nms(dets: List[Detection], iou_thr: float, sigma: float = 0.5, score_thresh: float = 0.001) -> List[Detection]:
    if not dets:
        return []
    dets_sorted = sorted(dets, key=lambda d: d.score, reverse=True)
    res: List[Detection] = []
    while dets_sorted:
        m = dets_sorted.pop(0)
        res.append(m)
        survivors: List[Detection] = []
        for d in dets_sorted:
            overlap = _iou(m.bbox, d.bbox)
            if overlap > iou_thr:
                d = Detection(
                    id=d.id,
                    class_name=d.class_name,
                    score=d.score * np.exp(-overlap * overlap / sigma),
                    bbox=d.bbox,
                    pipeline=d.pipeline,
                    page_no=d.page_no,
                    metadata=d.metadata,
                )
            if d.score >= score_thresh:
                survivors.append(d)
        dets_sorted = sorted(survivors, key=lambda d: d.score, reverse=True)
    return res


def _wbf(dets: List[Detection], iou_thr: float) -> List[Detection]:
    if not dets:
        return []
    dets_sorted = sorted(dets, key=lambda d: d.score, reverse=True)
    used = [False] * len(dets_sorted)
    fused: List[Detection] = []
    for i, di in enumerate(dets_sorted):
        if used[i]:
            continue
        cluster = [i]
        used[i] = True
        for j in range(i + 1, len(dets_sorted)):
            if used[j]:
                continue
            if _iou(di.bbox, dets_sorted[j].bbox) >= iou_thr:
                used[j] = True
                cluster.append(j)
        if len(cluster) == 1:
            fused.append(di)
        else:
            xs1, ys1, xs2, ys2, ws = [], [], [], [], []
            classes: Dict[str, float] = {}
            pipelines: Dict[str, float] = {}
            page_no = di.page_no
            max_score = 0.0
            for idx in cluster:
                d = dets_sorted[idx]
                x1, y1, x2, y2 = d.bbox.to_xyxy()
                xs1.append(x1 * d.score)
                ys1.append(y1 * d.score)
                xs2.append(x2 * d.score)
                ys2.append(y2 * d.score)
                ws.append(d.score)
                classes[d.class_name] = classes.get(d.class_name, 0.0) + d.score
                pipelines[d.pipeline] = pipelines.get(d.pipeline, 0.0) + d.score
                page_no = d.page_no if d.page_no is not None else page_no
                if d.score > max_score:
                    max_score = d.score
            s = max(1e-6, float(sum(ws)))
            fb = BBox.from_xyxy(sum(xs1) / s, sum(ys1) / s, sum(xs2) / s, sum(ys2) / s)
            c = max(classes.items(), key=lambda kv: kv[1])[0]
            p = max(pipelines.items(), key=lambda kv: kv[1])[0]
            fused.append(Detection(id=_uuid(), class_name=c, score=max_score, bbox=fb, pipeline=f"wbf:{p}", page_no=page_no))
    return fused


def _cluster_and_fuse(dets: List[Detection], algo: str, iou_thr: float) -> List[Detection]:
    if not dets:
        return []
    if algo == "nms":
        return _nms(dets, iou_thr)
    if algo == "soft-nms":
        return _soft_nms(dets, iou_thr)
    if algo == "wbf":
        return _wbf(dets, iou_thr)
    return dets


def _load_processor(cls, model_id: str, use_fast: bool = True):
    if not TRANSFORMERS_AVAILABLE:
        return None
    try:
        return cls.from_pretrained(model_id, use_fast=use_fast, trust_remote_code=True)
    except TypeError:
        return cls.from_pretrained(model_id, trust_remote_code=True)
    except Exception:
        return cls.from_pretrained(model_id)


class BaseDetector:
    def __init__(self, name: str, device: Optional[str] = None):
        self.name = name
        self.device = device or _device_pref()

    def detect(self, image: Image.Image, **kwargs) -> List[Detection]:
        raise NotImplementedError


class YOLODetector(BaseDetector):
    def __init__(self, model_path: str, device: Optional[str] = None, conf: float = 0.25, iou: float = 0.45, imgsz: Any = "auto"):
        super().__init__("yolo", device)
        if not YOLO_AVAILABLE:
            raise RuntimeError("ultralytics not installed")
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz

    def _auto_imgsz(self, image_size: Tuple[int, int]) -> int:
        w, h = image_size
        base = max(768, min(2048, max(w, h)))
        return int((base + 31) // 32) * 32

    def detect(self, image: Image.Image, **kwargs) -> List[Detection]:
        conf = kwargs.get("conf", self.conf)
        iou = kwargs.get("iou", self.iou)
        imgsz = kwargs.get("imgsz", self.imgsz)
        if imgsz == "auto":
            imgsz = self._auto_imgsz(image.size)
        results = self.model.predict(image, conf=conf, iou=iou, imgsz=imgsz, device=self.device, agnostic_nms=True, verbose=False)
        detections: List[Detection] = []
        for r in results:
            if not hasattr(r, "boxes") or r.boxes is None:
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            for i, (box, score, cls_id) in enumerate(zip(boxes, scores, classes)):
                x1, y1, x2, y2 = box
                bbox = BBox.from_xyxy(x1, y1, x2, y2)
                class_name = self.model.names.get(int(cls_id), str(cls_id))
                detections.append(Detection(id=f"yolo_{_uuid()}_{i}", class_name=class_name, score=float(score), bbox=bbox, pipeline="yolo"))
        return detections


class OpenVocabularyDetector(BaseDetector):
    def __init__(self, model_id: str, device: Optional[str] = None, min_threshold: float = 0.15):
        super().__init__("open_vocabulary", device)
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers not installed")
        self.model_id = model_id
        self.min_threshold = min_threshold
        self.config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        self.processor = _load_processor(AutoProcessor, model_id, use_fast=True)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, trust_remote_code=True)
        if TORCH_AVAILABLE:
            self.model.to(self.device)
        self.model.eval()
        self.model_type = getattr(self.config, "model_type", "").lower()

    def detect(self, image: Image.Image, queries: Optional[List[str]] = None, threshold: Optional[float] = None, **kwargs) -> List[Detection]:
        if queries is None:
            queries = ["signature", "handwritten signature", "ink stamp", "round stamp", "square stamp", "official seal", "company seal", "document", "text"]
        if threshold is None:
            threshold = self.min_threshold
        threshold = max(threshold, self.min_threshold)
        if "grounding" in self.model_type and "dino" in self.model_type:
            text_str = " ".join([q.strip().rstrip(".") + "." for q in queries])
            inputs = self.processor(images=image, text=text_str, return_tensors="pt")
        else:
            inputs = self.processor(text=queries, images=image, return_tensors="pt", padding=True)
        if TORCH_AVAILABLE:
            inputs = _to_device(inputs, self.device)
        with (torch.inference_mode() if TORCH_AVAILABLE else _nullcontext()):
            outputs = self.model(**inputs)
        target_sizes = None
        if TORCH_AVAILABLE:
            target_sizes = torch.tensor([image.size[::-1]], device=self.device)
        post = None
        if hasattr(self.processor, "post_process_grounded_object_detection"):
            post = getattr(self.processor, "post_process_grounded_object_detection")
        elif hasattr(self.processor, "post_process_object_detection"):
            post = getattr(self.processor, "post_process_object_detection")
        if post is None:
            return []
        processed = post(outputs=outputs, target_sizes=target_sizes)[0]
        boxes = processed.get("boxes")
        scores = processed.get("scores")
        labels = None
        if "grounding" in self.model_type and "dino" in self.model_type:
            labels = processed.get("text_labels", [""] * (len(scores) if scores is not None else 0))
        else:
            label_ids = processed.get("labels")
            if label_ids is not None:
                lab = label_ids.detach().cpu().numpy().tolist()
                labels = [queries[int(i)] for i in lab]
        if boxes is None or scores is None or labels is None:
            return []
        boxes_np = boxes.detach().cpu().numpy()
        scores_np = scores.detach().cpu().numpy()
        detections: List[Detection] = []
        for i, (box, score, label) in enumerate(zip(boxes_np, scores_np, labels)):
            if float(score) < threshold:
                continue
            x1, y1, x2, y2 = box
            bbox = BBox.from_xyxy(x1, y1, x2, y2)
            detections.append(Detection(id=f"ov_{_uuid()}_{i}", class_name=str(label), score=float(score), bbox=bbox, pipeline=("grounding_dino" if "grounding" in self.model_type and "dino" in self.model_type else "owl_vit")))
        return detections


class ZeroShotMatcher(BaseDetector):
    def __init__(self, model_id: str, backend: str = "auto", device: Optional[str] = None, labels: Optional[List[str]] = None, prompts: Optional[List[str]] = None, oc_model_name: str = "ViT-B-32", oc_pretrained: str = "laion2b_s34b_b79k"):
        super().__init__("clip", device)
        self.backend = backend
        self.model_id = model_id
        self.labels = labels or ["signature", "stamp", "seal"]
        self.prompts = prompts or ["a handwritten signature", "an ink stamp", "an official seal"]
        self.oc_model_name = oc_model_name
        self.oc_pretrained = oc_pretrained
        self._init_backend()

    def _init_backend(self):
        self.is_open_clip = False
        self.is_clip = False
        self.is_siglip = False
        if self.backend == "open-clip" or (self.backend == "auto" and OPEN_CLIP_AVAILABLE and ("laion" in self.model_id.lower() or "openclip" in self.model_id.lower())):
            self.is_open_clip = True
            self.oc_model, _, self.oc_preprocess = open_clip.create_model_and_transforms(self.oc_model_name, pretrained=self.oc_pretrained, device=self.device)
            self.oc_tokenizer = open_clip.get_tokenizer(self.oc_model_name)
            self.oc_model.eval()
            return
        if TRANSFORMERS_AVAILABLE:
            try:
                self.processor = CLIPProcessor.from_pretrained(self.model_id, use_fast=False, trust_remote_code=True)
                self.model = CLIPModel.from_pretrained(self.model_id, trust_remote_code=True)
                if TORCH_AVAILABLE:
                    self.model.to(self.device)
                self.model.eval()
                self.is_clip = True
                return
            except Exception:
                pass
            try:
                self.processor = SiglipProcessor.from_pretrained(self.model_id, use_fast=False, trust_remote_code=True)
                self.model = SiglipModel.from_pretrained(self.model_id, trust_remote_code=True)
                if TORCH_AVAILABLE:
                    self.model.to(self.device)
                self.model.eval()
                self.is_siglip = True
                return
            except Exception:
                pass
        raise RuntimeError("No compatible zero-shot image-text backend available")

    def encode_text(self, texts: List[str]) -> Any:
        if not TORCH_AVAILABLE:
            raise RuntimeError("Torch required")
        if self.is_open_clip:
            with torch.inference_mode():
                tok = self.oc_tokenizer(texts)
                if hasattr(tok, "to"):
                    tok = tok.to(self.device)
                return self.oc_model.encode_text(tok).float()
        if self.is_clip or self.is_siglip:
            if hasattr(self.processor, "tokenizer"):
                tok = self.processor.tokenizer(texts, padding=True, return_tensors="pt")
                tok = _to_device(tok, self.device)
                with torch.inference_mode():
                    return self.model.get_text_features(**tok).float()
            raise RuntimeError("Tokenizer unavailable for text encoding")
        raise RuntimeError("No backend")

    def encode_images(self, images: List[Image.Image]) -> Any:
        if not TORCH_AVAILABLE:
            raise RuntimeError("Torch required")
        if self.is_open_clip:
            with torch.inference_mode():
                tensors = [self.oc_preprocess(img.convert("RGB")).unsqueeze(0) for img in images]
                batch = torch.cat(tensors, dim=0).to(self.device)
                return self.oc_model.encode_image(batch).float()
        if self.is_clip or self.is_siglip:
            if hasattr(self.processor, "image_processor"):
                ip = self.processor.image_processor
                ip_inputs = ip(images=images, return_tensors="pt")
                ip_inputs = _to_device(ip_inputs, self.device)
                with torch.inference_mode():
                    return self.model.get_image_features(**ip_inputs).float()
            raise RuntimeError("Image processor unavailable for image encoding")
        raise RuntimeError("No backend")

    def logits(self, image_emb: Any, text_emb: Any) -> Any:
        if not TORCH_AVAILABLE:
            raise RuntimeError("Torch required")
        image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        sim = image_emb @ text_emb.t()
        if self.is_clip and hasattr(self.model, "logit_scale"):
            sim = sim * self.model.logit_scale.exp()
        return sim

    def classify_regions(self, image: Image.Image, regions: List[Detection], topk: int = 1) -> List[Dict[str, Any]]:
        crops: List[Image.Image] = []
        kept_regions: List[Detection] = []
        for r in regions:
            x1, y1, x2, y2 = r.bbox.to_xyxy()
            if x2 <= x1 or y2 <= y1:
                continue
            crops.append(image.crop((x1, y1, x2, y2)).convert("RGB"))
            kept_regions.append(r)
        if not crops:
            return []
        if not TORCH_AVAILABLE:
            return []
        text_emb = self.encode_text(self.prompts)
        img_emb = self.encode_images(crops)
        with torch.inference_mode():
            logits = self.logits(img_emb, text_emb)
            probs = logits.softmax(dim=-1).detach().cpu().numpy()
        results: List[Dict[str, Any]] = []
        for i, row in enumerate(probs):
            if topk == 1:
                idx = int(np.argmax(row))
                results.append({"class": self.labels[idx], "score": float(row[idx]), "region": kept_regions[i]})
            else:
                idxs = np.argsort(-row)[:topk]
                results.append({"classes": [self.labels[int(j)] for j in idxs], "scores": [float(row[int(j)]) for j in idxs], "region": kept_regions[i]})
        return results

    def _generate_proposals(self, image: Image.Image, max_props: int = 512) -> List[Detection]:
        w, h = image.size
        scales = [0.10, 0.14, 0.18, 0.22, 0.28]
        proposals: List[Detection] = []
        pid = 0
        for s in scales:
            size = int(round(s * min(w, h)))
            size = max(96, min(size, min(w, h)))
            stride = max(32, size // 2)
            for y in range(0, max(1, h - size + 1), stride):
                for x in range(0, max(1, w - size + 1), stride):
                    proposals.append(Detection(id=f"prop_{pid}", class_name="proposal", score=1.0, bbox=BBox(x=x, y=y, w=size, h=size), pipeline="proposal"))
                    pid += 1
        if len(proposals) > max_props:
            step = max(1, len(proposals) // max_props)
            proposals = proposals[::step][:max_props]
        return proposals

    def detect(self, image: Image.Image, proposals: Optional[List[Detection]] = None, threshold: float = 0.18, topk: int = 1, **kwargs) -> List[Detection]:
        if proposals is None:
            proposals = self._generate_proposals(image)
        classified = self.classify_regions(image, proposals, topk=topk)
        detections: List[Detection] = []
        for i, item in enumerate(classified):
            c = item.get("class")
            s = item.get("score", 0.0)
            if c is None or s < threshold:
                continue
            detections.append(Detection(id=f"clip_{_uuid()}_{i}", class_name=c, score=s, bbox=item["region"].bbox, pipeline="clip"))
        return detections


class DocumentProcessor:
    def __init__(self, input_path: str, output_dir: Path, pipelines_config: Dict[str, Dict[str, Any]], aggregation: str = "wbf", aggregation_iou: float = 0.55, refine_pad: float = 0.06, verbose: bool = False):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.pipelines_config = pipelines_config
        self.pipelines: Dict[str, BaseDetector] = {}
        self.aggregation = aggregation
        self.aggregation_iou = aggregation_iou
        self.refine_pad = refine_pad
        self.verbose = verbose
        self._initialize_pipelines()

    def _initialize_pipelines(self):
        for name, config in self.pipelines_config.items():
            if not config.get("enabled", False):
                continue
            t = config["type"]
            if t == PipelineType.YOLO.value:
                if YOLO_AVAILABLE:
                    self.pipelines[name] = YOLODetector(model_path=config["model_path"], device=config.get("device"), conf=config.get("conf", 0.25), iou=config.get("iou", 0.45), imgsz=config.get("imgsz", "auto"))
            elif t == PipelineType.GROUNDING_DINO.value:
                if TRANSFORMERS_AVAILABLE:
                    self.pipelines[name] = OpenVocabularyDetector(model_id=config["model_id"], device=config.get("device"), min_threshold=config.get("min_threshold", 0.15))
            elif t == PipelineType.OWL_VIT.value:
                if TRANSFORMERS_AVAILABLE:
                    self.pipelines[name] = OpenVocabularyDetector(model_id=config["model_id"], device=config.get("device"), min_threshold=config.get("min_threshold", 0.15))
            elif t == PipelineType.CLIP.value:
                if TRANSFORMERS_AVAILABLE or OPEN_CLIP_AVAILABLE:
                    self.pipelines[name] = ZeroShotMatcher(model_id=config.get("model_id", "openai/clip-vit-base-patch32"), backend=config.get("backend", "auto"), device=config.get("device"), labels=config.get("labels"), prompts=config.get("prompts"), oc_model_name=config.get("open_clip_model", "ViT-B-32"), oc_pretrained=config.get("open_clip_pretrained", "laion2b_s34b_b79k"))

    def process(self) -> Dict[str, Any]:
        if self.input_path.suffix.lower() == ".pdf":
            return self._process_pdf()
        return self._process_image()

    def _process_pdf(self) -> Dict[str, Any]:
        pages = list(self._rasterize_pdf())
        results: List[ProcessedPage] = []
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), TimeElapsedColumn(), transient=not self.verbose, console=console) as progress:
            task = progress.add_task("Processing PDF pages", total=len(pages))
            for page_no, image in pages:
                page_result = self._process_single_image(image, page_no)
                results.append(page_result)
                progress.advance(task)
        return self._save_results(results)

    def _process_image(self) -> Dict[str, Any]:
        image = Image.open(self.input_path).convert("RGB")
        page_result = self._process_single_image(image, page_no=1)
        return self._save_results([page_result])

    def _process_single_image(self, image: Image.Image, page_no: int) -> ProcessedPage:
        w, h = image.size
        base_page_dir = self.output_dir / f"page_{page_no:04d}"
        base_page_dir.mkdir(parents=True, exist_ok=True)
        image_path = base_page_dir / "original.png"
        image.save(image_path)
        page_result = ProcessedPage(page_no=page_no, width=w, height=h, image_path=str(image_path))
        all_detections: List[Detection] = []
        refined_dets: List[Detection] = []
        for pipeline_name, pipeline in self.pipelines.items():
            pipeline_cfg = self.pipelines_config.get(pipeline_name, {})
            pipeline_dir = base_page_dir / pipeline_name
            pipeline_dir.mkdir(exist_ok=True)
            skip_direct = pipeline_cfg.get("type") == PipelineType.CLIP.value and pipeline_cfg.get("mode") == "refine"
            if skip_direct:
                page_result.detections[pipeline_name] = []
                continue
            try:
                if isinstance(pipeline, OpenVocabularyDetector):
                    queries = pipeline_cfg.get("queries")
                    threshold = pipeline_cfg.get("min_threshold")
                    detections = pipeline.detect(image, queries=queries, threshold=threshold)
                elif isinstance(pipeline, YOLODetector):
                    detections = pipeline.detect(image)
                elif isinstance(pipeline, ZeroShotMatcher):
                    detections = pipeline.detect(image, threshold=pipeline_cfg.get("threshold", 0.18))
                else:
                    detections = pipeline.detect(image)
                detections = [self._clamp_detection(det, w, h, page_no) for det in detections]
                page_result.detections[pipeline_name] = detections
                all_detections.extend(detections)
                annotated = self._annotate_image(image, detections)
                (pipeline_dir / "annotated.png").write_bytes(_image_to_png_bytes(annotated))
                for i, det in enumerate(detections):
                    x1, y1, x2, y2 = det.bbox.to_xyxy()
                    if x2 > x1 and y2 > y1:
                        crop = image.crop((x1, y1, x2, y2))
                        crop_path = pipeline_dir / f"crop_{i:04d}_{safe_name(det.class_name)}.png"
                        crop.save(crop_path)
                result_json = {
                    "page_no": page_no,
                    "pipeline": pipeline_name,
                    "detections": [{"id": det.id, "class": det.class_name, "score": det.score, "bbox": det.bbox.to_dict()} for det in detections],
                }
                (pipeline_dir / "results.json").write_text(json.dumps(result_json, indent=2))
            except Exception as e:
                logging.exception(f"Pipeline {pipeline_name} failed: {e}")
                page_result.detections[pipeline_name] = []
        if "clip" in self.pipelines and self.pipelines_config.get("clip", {}).get("mode") == "refine":
            try:
                clip_model: ZeroShotMatcher = self.pipelines["clip"]  # type: ignore
                proposals = self._aggregate_proposals(all_detections, w, h)
                classified = clip_model.classify_regions(image, proposals)
                for i, item in enumerate(classified):
                    det = Detection(
                        id=f"clip_refine_{_uuid()}_{i}",
                        class_name=item["class"],
                        score=item["score"],
                        bbox=item["region"].bbox,
                        pipeline="clip_refine",
                        page_no=page_no,
                    )
                    refined_dets.append(det)
                page_result.detections["clip_refine"] = refined_dets
                clip_ref_dir = base_page_dir / "clip_refine"
                clip_ref_dir.mkdir(exist_ok=True)
                annotated_ref = self._annotate_image(image, refined_dets)
                (clip_ref_dir / "annotated.png").write_bytes(_image_to_png_bytes(annotated_ref))
                (clip_ref_dir / "results.json").write_text(json.dumps({"page_no": page_no, "pipeline": "clip_refine", "detections": [{"id": d.id, "class": d.class_name, "score": d.score, "bbox": d.bbox.to_dict()} for d in refined_dets]}, indent=2))
            except Exception as e:
                logging.exception(f"CLIP refinement failed: {e}")
                page_result.detections["clip_refine"] = []
        try:
            page_all = all_detections + refined_dets
            page_annotated = self._annotate_image(image, page_all)
            (base_page_dir / "page_annotated.png").write_bytes(_image_to_png_bytes(page_annotated))
        except Exception as e:
            logging.exception(f"Failed to save page_annotated: {e}")
        return page_result

    def _aggregate_proposals(self, detections: List[Detection], w: int, h: int) -> List[Detection]:
        if not detections:
            return []
        proposals = []
        for d in detections:
            proposals.append(Detection(id=d.id, class_name="proposal", score=d.score, bbox=_expand_box(d.bbox, w, h, self.refine_pad), pipeline="proposal", page_no=d.page_no))
        fused = _cluster_and_fuse(proposals, self.aggregation, self.aggregation_iou)
        return fused

    def _rasterize_pdf(self) -> Iterable[Tuple[int, Image.Image]]:
        pdf_path = self.input_path
        if PYMUPDF_AVAILABLE:
            doc = fitz.open(str(pdf_path))
            for i in range(len(doc)):
                page = doc[i]
                mat = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data)).convert("RGB")
                yield i + 1, img
        elif PDF2IMAGE_AVAILABLE:
            pages = convert_from_path(str(pdf_path), dpi=200)
            for i, page in enumerate(pages):
                yield i + 1, page.convert("RGB")
        else:
            raise RuntimeError("No PDF rasterization library available")

    def _annotate_image(self, image: Image.Image, detections: List[Detection]) -> Image.Image:
        img = image.convert("RGB").copy()
        draw = ImageDraw.Draw(img)
        font = _get_font()
        class_colors = {"signature": (255, 200, 0), "stamp": (0, 200, 255), "seal": (255, 0, 100), "document": (0, 255, 0), "text": (255, 255, 0)}
        pipeline_fallback = {"yolo": (255, 255, 255), "grounding_dino": (255, 128, 0), "owl_vit": (128, 255, 0), "clip": (128, 128, 255), "clip_refine": (255, 0, 255), "proposal": (128, 128, 128)}
        w, h = img.size
        for det in detections:
            cname = det.class_name.lower()
            color = class_colors.get(cname, pipeline_fallback.get(det.pipeline, (255, 255, 255)))
            x1, y1, x2, y2 = det.bbox.to_xyxy()
            x1 = max(0, min(int(x1), w - 1))
            y1 = max(0, min(int(y1), h - 1))
            x2 = max(0, min(int(x2), w - 1))
            y2 = max(0, min(int(y2), h - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            label = f"{det.class_name} {det.score:.2f} ({det.pipeline})"
            tw, th = _measure_text(draw, label, font)
            ty = y1 - th - 4
            if ty < 0:
                ty = y1 + 4
                if ty + th + 4 > h:
                    ty = max(0, h - th - 4)
            draw.rectangle([x1, ty, x1 + tw + 6, ty + th + 4], fill=(0, 0, 0))
            draw.text((x1 + 3, ty + 2), label, fill=color, font=font)
        return img

    def _clamp_detection(self, det: Detection, w: int, h: int, page_no: int) -> Detection:
        x1, y1, x2, y2 = det.bbox.to_xyxy()
        x1 = max(0, min(int(x1), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        x2 = max(0, min(int(x2), w - 1))
        y2 = max(0, min(int(y2), h - 1))
        det.bbox = BBox.from_xyxy(x1, y1, x2, y2)
        det.page_no = page_no
        return det

    def _save_results(self, results: List[ProcessedPage]) -> Dict[str, Any]:
        summary = {
            "input": str(self.input_path),
            "output_dir": str(self.output_dir),
            "timestamp": datetime.now().isoformat(),
            "pipelines": list(self.pipelines.keys()),
            "pages": [],
        }
        for page_result in results:
            page_summary = {"page_no": page_result.page_no, "size": {"width": page_result.width, "height": page_result.height}, "detections_by_pipeline": {}}
            for pipeline_name, detections in page_result.detections.items():
                page_summary["detections_by_pipeline"][pipeline_name] = {"count": len(detections), "classes": sorted(list(set(d.class_name for d in detections)))}
            summary["pages"].append(page_summary)
        summary_path = self.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        return summary


def parse_args():
    p = argparse.ArgumentParser(description="Document detection pipeline")
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="output")
    p.add_argument("--pipelines", nargs="+", default=["yolo"], choices=["yolo", "grounding_dino", "owl_vit", "clip", "all"])
    p.add_argument("--yolo_model", default="yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.15)
    p.add_argument("--iou", type=float, default=0.25)
    p.add_argument("--imgsz", default="auto")
    p.add_argument("--grounding_dino_model", default="IDEA-Research/grounding-dino-base")
    p.add_argument("--owl_vit_model", default="google/owlv2-base-patch16-ensemble")
    p.add_argument("--queries", default="signature,stamp,seal,document,text")
    p.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    p.add_argument("--clip_mode", choices=["standalone", "refine"], default="refine")
    p.add_argument("--clip_backend", choices=["auto", "clip", "siglip", "open-clip"], default="auto")
    p.add_argument("--labels", default="signature,stamp,seal")
    p.add_argument("--prompts", default="a handwritten signature,an ink stamp,an official seal")
    p.add_argument("--open_clip_model", default="ViT-B-32")
    p.add_argument("--open_clip_pretrained", default="laion2b_s34b_b79k")
    p.add_argument("--aggregation", choices=["wbf", "nms", "soft-nms", "none"], default="wbf")
    p.add_argument("--aggregation_iou", type=float, default=0.55)
    p.add_argument("--refine_pad", type=float, default=0.06)
    p.add_argument("--device", default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=(logging.DEBUG if args.verbose else logging.INFO), format="%(message)s", datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True, console=console)])
    pipelines_to_run = args.pipelines
    if "all" in pipelines_to_run:
        pipelines_to_run = ["yolo", "grounding_dino", "owl_vit", "clip"]
    imgsz = args.imgsz
    try:
        if isinstance(imgsz, str) and imgsz.lower() != "auto":
            imgsz = int(imgsz)
    except Exception:
        imgsz = "auto"
    queries = [q.strip() for q in args.queries.split(",") if q.strip()]
    labels = [q.strip() for q in args.labels.split(",") if q.strip()]
    prompts = [q.strip() for q in args.prompts.split(",") if q.strip()]
    pipelines_config: Dict[str, Dict[str, Any]] = {}
    if "yolo" in pipelines_to_run:
        pipelines_config["yolo"] = {"enabled": True, "type": PipelineType.YOLO.value, "model_path": args.yolo_model, "device": args.device, "conf": args.conf, "iou": args.iou, "imgsz": imgsz}
    if "grounding_dino" in pipelines_to_run:
        pipelines_config["grounding_dino"] = {"enabled": True, "type": PipelineType.GROUNDING_DINO.value, "model_id": args.grounding_dino_model, "device": args.device, "min_threshold": args.conf, "queries": queries}
    if "owl_vit" in pipelines_to_run:
        pipelines_config["owl_vit"] = {"enabled": True, "type": PipelineType.OWL_VIT.value, "model_id": args.owl_vit_model, "device": args.device, "min_threshold": args.conf, "queries": queries}
    if "clip" in pipelines_to_run:
        pipelines_config["clip"] = {
            "enabled": True,
            "type": PipelineType.CLIP.value,
            "model_id": args.clip_model,
            "device": args.device,
            "mode": args.clip_mode,
            "backend": args.clip_backend,
            "labels": labels,
            "prompts": prompts,
            "threshold": args.conf,
            "open_clip_model": args.open_clip_model,
            "open_clip_pretrained": args.open_clip_pretrained,
        }
    processor = DocumentProcessor(input_path=args.input, output_dir=Path(args.output), pipelines_config=pipelines_config, aggregation=args.aggregation, aggregation_iou=args.aggregation_iou, refine_pad=args.refine_pad, verbose=args.verbose)
    try:
        summary = processor.process()
        logging.info(json.dumps(summary, indent=2))
    except Exception as e:
        logging.exception(f"Processing failed: {e}")
        raise


if __name__ == "__main__":
    main()