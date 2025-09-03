import argparse
import json
import io
import logging
import math
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2

    CV2_AVAILABLE = True
except:
    CV2_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except:
    TORCH_AVAILABLE = False

try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except:
    YOLO_AVAILABLE = False

try:
    from transformers import (
        AutoConfig,
        AutoProcessor,
        AutoModelForZeroShotObjectDetection,
        CLIPModel,
        CLIPProcessor,
    )

    TRANSFORMERS_AVAILABLE = True
except:
    TRANSFORMERS_AVAILABLE = False

try:
    import fitz

    PYMUPDF_AVAILABLE = True
except:
    PYMUPDF_AVAILABLE = False

try:
    from pdf2image import convert_from_path

    PDF2IMAGE_AVAILABLE = True
except:
    PDF2IMAGE_AVAILABLE = False


class PipelineType(Enum):
    YOLO = "yolo"
    OPEN_VOCAB = "open_vocabulary"
    CLIP = "clip"
    CV = "cv"


@dataclass(frozen=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def as_xyxy(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def as_int(self) -> Tuple[int, int, int, int]:
        return (int(round(self.x1)), int(round(self.y1)), int(round(self.x2)), int(round(self.y2)))

    def w(self) -> float:
        return max(0.0, self.x2 - self.x1)

    def h(self) -> float:
        return max(0.0, self.y2 - self.y1)

    def area(self) -> float:
        return self.w() * self.h()

    def clamp(self, w: int, h: int) -> "BBox":
        x1 = float(min(max(0.0, self.x1), max(0, w - 1)))
        y1 = float(min(max(0.0, self.y1), max(0, h - 1)))
        x2 = float(min(max(x1 + 1.0, self.x2), w))
        y2 = float(min(max(y1 + 1.0, self.y2), h))
        return BBox(x1, y1, x2, y2)

    def pad_to_square(self, ratio: float = 1.0, center: bool = True) -> "BBox":
        w = self.w()
        h = self.h()
        s = max(w, h) * ratio
        if center:
            cx = (self.x1 + self.x2) * 0.5
            cy = (self.y1 + self.y2) * 0.5
            x1 = cx - s * 0.5
            y1 = cy - s * 0.5
            x2 = cx + s * 0.5
            y2 = cy + s * 0.5
            return BBox(x1, y1, x2, y2)
        x2 = self.x1 + s
        y2 = self.y1 + s
        return BBox(self.x1, self.y1, x2, y2)

    def expand(self, px: float) -> "BBox":
        return BBox(self.x1 - px, self.y1 - px, self.x2 + px, self.y2 + px)


@dataclass
class Detection:
    id: str
    class_name: str
    score: float
    bbox: BBox
    pipeline: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PageResult:
    page_no: int
    width: int
    height: int
    image_path: str
    detections_by_pipeline: Dict[str, List[Detection]] = field(default_factory=dict)
    merged_detections: List[Detection] = field(default_factory=list)


class Geometry:
    @staticmethod
    def iou(a: BBox, b: BBox) -> float:
        x1 = max(a.x1, b.x1)
        y1 = max(a.y1, b.y1)
        x2 = min(a.x2, b.x2)
        y2 = min(a.y2, b.y2)
        iw = max(0.0, x2 - x1)
        ih = max(0.0, y2 - y1)
        inter = iw * ih
        ua = a.area() + b.area() - inter + 1e-9
        return float(inter / ua)

    @staticmethod
    def iou_matrix(boxes: np.ndarray, target: np.ndarray) -> np.ndarray:
        xx1 = np.maximum(boxes[:, 0], target[0])
        yy1 = np.maximum(boxes[:, 1], target[1])
        xx2 = np.minimum(boxes[:, 2], target[2])
        yy2 = np.minimum(boxes[:, 3], target[3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        a_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        a_t = (target[2] - target[0]) * (target[3] - target[1])
        return inter / (a_boxes + a_t - inter + 1e-9)

    @staticmethod
    def diou(a: np.ndarray, bs: np.ndarray) -> np.ndarray:
        ious = Geometry.iou_matrix(bs, a)
        cx, cy = (a[0] + a[2]) * 0.5, (a[1] + a[3]) * 0.5
        cxs, cys = (bs[:, 0] + bs[:, 2]) * 0.5, (bs[:, 1] + bs[:, 3]) * 0.5
        rho2 = (cx - cxs) ** 2 + (cy - cys) ** 2
        x1 = np.minimum(a[0], bs[:, 0])
        y1 = np.minimum(a[1], bs[:, 1])
        x2 = np.maximum(a[2], bs[:, 2])
        y2 = np.maximum(a[3], bs[:, 3])
        c2 = (x2 - x1) ** 2 + (y2 - y1) ** 2 + 1e-9
        return ious - rho2 / c2


class BoxOps:
    @staticmethod
    def to_array(dets: List[Detection]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        if not dets:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), []
        b = np.array([d.bbox.as_xyxy() for d in dets], dtype=np.float32)
        s = np.array([d.score for d in dets], dtype=np.float32)
        l = [d.class_name for d in dets]
        return b, s, l

    @staticmethod
    def greedy_nms(boxes: np.ndarray, scores: np.ndarray, thr: float) -> np.ndarray:
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            ious = Geometry.iou_matrix(boxes[order[1:]], boxes[i])
            inds = np.where(ious <= thr)[0]
            order = order[inds + 1]
        return np.array(keep, dtype=np.int32)

    @staticmethod
    def diou_nms(boxes: np.ndarray, scores: np.ndarray, thr: float) -> np.ndarray:
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            dious = Geometry.diou(boxes[i], boxes[order[1:]])
            inds = np.where(dious <= thr)[0]
            order = order[inds + 1]
        return np.array(keep, dtype=np.int32)

    @staticmethod
    def soft_nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float, sigma: float, method: str, score_thr: float) -> Tuple[np.ndarray, np.ndarray]:
        boxes = boxes.copy()
        scores = scores.copy()
        idxs = np.arange(scores.shape[0])
        kept = []
        while scores.size > 0:
            m = int(np.argmax(scores))
            kept.append((idxs[m], boxes[m].copy(), scores[m].copy()))
            if scores.size == 1:
                break
            b0 = boxes[m]
            s0 = scores[m]
            b = np.concatenate([boxes[:m], boxes[m + 1 :]], 0)
            s = np.concatenate([scores[:m], scores[m + 1 :]], 0)
            ix = np.concatenate([idxs[:m], idxs[m + 1 :]], 0)
            ious = Geometry.iou_matrix(b, b0)
            if method == "linear":
                w = np.ones_like(ious)
                mask = ious > iou_thr
                w[mask] = 1 - ious[mask]
            elif method == "gaussian":
                w = np.exp(-((ious**2) / sigma))
            else:
                mask = ious <= iou_thr
                boxes = b[mask]
                scores = s[mask]
                idxs = ix[mask]
                continue
            s = s * w
            valid = s > score_thr
            boxes = b[valid]
            scores = s[valid]
            idxs = ix[valid]
        if not kept:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)
        kept_idx = np.array([k[0] for k in kept], dtype=np.int32)
        kept_scores = np.array([k[2] for k in kept], dtype=np.float32)
        o = kept_scores.argsort()[::-1]
        return kept_idx[o], kept_scores[o]

    @staticmethod
    def weighted_boxes_fusion(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> Tuple[np.ndarray, np.ndarray]:
        if boxes.shape[0] == 0:
            return boxes, scores
        order = scores.argsort()[::-1]
        boxes = boxes[order]
        scores = scores[order]
        used = np.zeros(len(scores), dtype=bool)
        fused_boxes = []
        fused_scores = []
        for i in range(len(scores)):
            if used[i]:
                continue
            cluster = [i]
            for j in range(i + 1, len(scores)):
                if used[j]:
                    continue
                if Geometry.iou_matrix(boxes[j:j+1], boxes[i])[0] >= iou_thr:
                    cluster.append(j)
            used[cluster] = True
            w = scores[cluster]
            wsum = float(np.sum(w) + 1e-12)
            b = boxes[cluster]
            fb = np.sum(b * w[:, None], axis=0) / wsum
            sc = float(np.mean(scores[cluster]))
            fused_boxes.append(fb)
            fused_scores.append(sc)
        fb = np.vstack(fused_boxes).astype(np.float32) if fused_boxes else np.zeros((0, 4), dtype=np.float32)
        fs = np.array(fused_scores, dtype=np.float32) if fused_scores else np.zeros((0,), dtype=np.float32)
        return fb, fs

    @staticmethod
    def box_voting(seed: np.ndarray, boxes: np.ndarray, weights: np.ndarray, iou_thr: float) -> np.ndarray:
        ious = Geometry.iou_matrix(boxes, seed)
        mask = ious >= iou_thr
        if not np.any(mask):
            return seed
        w = weights[mask]
        wsum = float(np.sum(w) + 1e-12)
        return np.sum(boxes[mask] * w[:, None], axis=0) / wsum


class Labeling:
    @staticmethod
    def refined_label(name: str, mapping: Optional[Dict[str, str]] = None) -> str:
        n = name.strip().lower()
        aliases = {
            "signature": "signature",
            "sign": "signature",
            "autograph": "signature",
            "sig": "signature",
            "seal": "seal",
            "company seal": "seal",
            "official seal": "seal",
            "square seal": "seal",
            "round seal": "seal",
            "stamp": "stamp",
            "rubber stamp": "stamp",
            "round stamp": "stamp",
            "chop": "seal",
            "ink seal": "seal",
            "ink stamp": "stamp",
            "black ink seal": "seal",
            "black ink stamp": "stamp",
        }
        if mapping:
            for k, v in mapping.items():
                aliases[k.strip().lower()] = v.strip().lower()
        if n in aliases:
            return aliases[n]
        for k, v in aliases.items():
            if k in n:
                return v
        return n

    @staticmethod
    def color_for(name: str) -> Tuple[int, int, int]:
        base = {
            "seal": (255, 64, 64),
            "stamp": (64, 192, 255),
            "signature": (255, 200, 0),
            "document": (200, 200, 200),
        }
        n = name.lower()
        if n in base:
            return base[n]
        h = hash(n)
        r = (h & 255)
        g = (h >> 8) & 255
        b = (h >> 16) & 255
        return (int(r), int(g), int(b))


class Transform:
    @staticmethod
    def hflip(image: np.ndarray) -> np.ndarray:
        return image[:, ::-1, :]

    @staticmethod
    def vflip(image: np.ndarray) -> np.ndarray:
        return image[::-1, :, :]

    @staticmethod
    def rot90(image: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(np.rot90(image, 1))

    @staticmethod
    def rot270(image: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(np.rot90(image, -1))

    @staticmethod
    def map_boxes_hflip(boxes: np.ndarray, w: int) -> np.ndarray:
        b = boxes.copy()
        b[:, 0], b[:, 2] = w - boxes[:, 2], w - boxes[:, 0]
        return b

    @staticmethod
    def map_boxes_vflip(boxes: np.ndarray, h: int) -> np.ndarray:
        b = boxes.copy()
        b[:, 1], b[:, 3] = h - boxes[:, 3], h - boxes[:, 1]
        return b

    @staticmethod
    def map_boxes_rot90(boxes: np.ndarray, w: int, h: int) -> np.ndarray:
        b = boxes.copy()
        x1, y1, x2, y2 = b[:, 0].copy(), b[:, 1].copy(), b[:, 2].copy(), b[:, 3].copy()
        b[:, 0] = y1
        b[:, 1] = w - x2
        b[:, 2] = y2
        b[:, 3] = w - x1
        return b

    @staticmethod
    def map_boxes_rot270(boxes: np.ndarray, w: int, h: int) -> np.ndarray:
        b = boxes.copy()
        x1, y1, x2, y2 = b[:, 0].copy(), b[:, 1].copy(), b[:, 2].copy(), b[:, 3].copy()
        b[:, 0] = h - y2
        b[:, 1] = x1
        b[:, 2] = h - y1
        b[:, 3] = x2
        return b


class BaseDetector:
    def __init__(self, name: str, device: Optional[str] = None):
        self.name = name
        if TORCH_AVAILABLE:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device or "cpu"

    def detect(self, image: Image.Image) -> List[Detection]:
        raise NotImplementedError


class YOLODetector(BaseDetector):
    def __init__(self, model_path: str, device: Optional[str] = None, conf: float = 0.25, iou: float = 0.45, max_det: int = 300, agnostic: bool = True, tta: bool = True, imgsz_list: Optional[List[int]] = None, conf_seq: Optional[List[float]] = None):
        super().__init__("yolo", device)
        if not YOLO_AVAILABLE:
            raise RuntimeError("ultralytics not installed")
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.agnostic = agnostic
        self.tta = tta
        self.imgsz_list = imgsz_list or [1024, 1280, 1536]
        c0 = max(0.05, conf)
        c1 = max(0.05, conf * 0.75)
        self.conf_seq = conf_seq or sorted(list(dict.fromkeys([conf, c1, 0.05])), reverse=True)
        self.names = self._names()

    def _names(self) -> Dict[int, str]:
        names = getattr(self.model, "names", {})
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, list):
            return {i: str(v) for i, v in enumerate(names)}
        return {}

    def _predict_single(self, img: np.ndarray, conf: float, imgsz: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r = self.model.predict(img, conf=conf, iou=self.iou, imgsz=imgsz, device=self.device, verbose=False, agnostic_nms=self.agnostic, max_det=self.max_det)
        if not r:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32)
        b, s, c = [], [], []
        for o in r:
            if not hasattr(o, "boxes") or o.boxes is None or o.boxes.xyxy is None:
                continue
            bx = o.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
            sc = o.boxes.conf.detach().cpu().numpy().astype(np.float32).reshape(-1)
            cl = o.boxes.cls.detach().cpu().numpy().astype(np.int32).reshape(-1)
            b.append(bx)
            s.append(sc)
            c.append(cl)
        if not b:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32)
        b = np.concatenate(b, 0)
        s = np.concatenate(s, 0)
        c = np.concatenate(c, 0)
        return b, s, c

    def _predict_with_tta(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        H, W = image_bgr.shape[:2]
        boxes_all = []
        scores_all = []
        clss_all = []
        augs = [("none", image_bgr)]
        if self.tta:
            augs += [("h", Transform.hflip(image_bgr)), ("v", Transform.vflip(image_bgr)), ("r90", Transform.rot90(image_bgr)), ("r270", Transform.rot270(image_bgr))]
        for imgsz in self.imgsz_list:
            for cf in self.conf_seq:
                for tag, aug in augs:
                    b, s, c = self._predict_single(aug, conf=cf, imgsz=imgsz)
                    if b.shape[0] == 0:
                        continue
                    if tag == "h":
                        b = Transform.map_boxes_hflip(b, W)
                    elif tag == "v":
                        b = Transform.map_boxes_vflip(b, H)
                    elif tag == "r90":
                        b = Transform.map_boxes_rot90(b, W, H)
                    elif tag == "r270":
                        b = Transform.map_boxes_rot270(b, W, H)
                    boxes_all.append(b)
                    scores_all.append(s)
                    clss_all.append(c)
        if not boxes_all:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32)
        boxes = np.concatenate(boxes_all, 0)
        scores = np.concatenate(scores_all, 0)
        clss = np.concatenate(clss_all, 0)
        return boxes, scores, clss

    def detect(self, image: Image.Image) -> List[Detection]:
        if not CV2_AVAILABLE:
            return []
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        b, s, c = self._predict_with_tta(img_bgr)
        if b.shape[0] == 0:
            return []
        b, s = BoxOps.weighted_boxes_fusion(b, s, iou_thr=0.55)
        keep = BoxOps.greedy_nms(b, s, thr=self.iou)
        b = b[keep]
        s = s[keep]
        dets = []
        for i in range(b.shape[0]):
            x1, y1, x2, y2 = b[i].tolist()
            cid = int(-1)
            cname = "object"
            dets.append(Detection(id=f"y_{i}", class_name=cname, score=float(s[i]), bbox=BBox(x1, y1, x2, y2), pipeline="yolo", meta={"raw": True}))
        return dets


class OpenVocabularyDetector(BaseDetector):
    def __init__(self, model_id: str, device: Optional[str] = None, min_threshold: float = 0.15, queries: Optional[List[str]] = None):
        super().__init__("open_vocabulary", device)
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers not installed")
        self.model_id = model_id
        self.min_threshold = min_threshold
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        self.model.to(self.device).eval()
        self.config = AutoConfig.from_pretrained(model_id)
        self.is_grounding_dino = self._is_gdino()
        self.queries = queries or ["square seal", "seal", "stamp", "round stamp", "company seal", "official seal", "signature"]

    def _is_gdino(self) -> bool:
        mt = getattr(self.config, "model_type", "").lower()
        return "grounding" in mt and "dino" in mt

    def detect(self, image: Image.Image) -> List[Detection]:
        if self.is_grounding_dino:
            text = " ".join([q.strip() + "." for q in self.queries])
            inputs = self.processor(images=image, text=text, return_tensors="pt")
        else:
            inputs = self.processor(text=self.queries, images=image, return_tensors="pt", padding=True)
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs)
        ts = torch.tensor([image.size[::-1]], device=self.device)
        if self.is_grounding_dino:
            processed = self.processor.post_process_grounded_object_detection(outputs=outputs, target_sizes=ts)[0]
            boxes = processed["boxes"].detach().cpu().numpy()
            scores = processed["scores"].detach().cpu().numpy()
            labels = processed.get("text_labels", [""] * len(scores))
        else:
            processed = self.processor.post_process_object_detection(outputs=outputs, target_sizes=ts, threshold=self.min_threshold)[0]
            boxes = processed["boxes"].detach().cpu().numpy()
            scores = processed["scores"].detach().cpu().numpy()
            label_ids = processed["labels"].detach().cpu().numpy().astype(int).tolist()
            labels = [self.queries[i] if 0 <= i < len(self.queries) else "object" for i in label_ids]
        dets = []
        for i, (box, sc, lb) in enumerate(zip(boxes, scores, labels)):
            if float(sc) < self.min_threshold:
                continue
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            dets.append(Detection(id=f"ov_{i}", class_name=str(lb), score=float(sc), bbox=BBox(x1, y1, x2, y2), pipeline="open_vocabulary"))
        return dets


class CLIPVerifier(BaseDetector):
    def __init__(self, model_id: str = "openai/clip-vit-base-patch32", device: Optional[str] = None):
        super().__init__("clip", device)
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers not installed")
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id)
        self.model.to(self.device).eval()
        self.labels = ["square seal", "seal", "stamp", "signature", "background"]
        self.prompts = ["a square official ink seal", "an official seal", "an ink stamp", "a handwritten signature", "background"]

    def score_regions(self, image: Image.Image, dets: List[Detection]) -> List[Tuple[int, float, str]]:
        if not dets:
            return []
        crops = []
        for d in dets:
            x1, y1, x2, y2 = d.bbox.as_int()
            crop = image.crop((x1, y1, x2, y2)).convert("RGB")
            crops.append(crop)
        inputs = self.processor(text=self.prompts, images=crops, return_tensors="pt", padding=True)
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.inference_mode():
            out = self.model(**inputs)
        probs = out.logits_per_image.softmax(dim=-1).detach().cpu().numpy()
        results = []
        for i in range(probs.shape[0]):
            j = int(np.argmax(probs[i]))
            results.append((i, float(probs[i, j]), self.labels[j]))
        return results

    def detect(self, image: Image.Image) -> List[Detection]:
        return []


class CVSealDetector(BaseDetector):
    def __init__(self):
        super().__init__("cv", device="cpu")

    def _detect_red_masks(self, bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lower1 = np.array([0, 60, 40], dtype=np.uint8)
        upper1 = np.array([10, 255, 255], dtype=np.uint8)
        lower2 = np.array([170, 60, 40], dtype=np.uint8)
        upper2 = np.array([180, 255, 255], dtype=np.uint8)
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.medianBlur(mask, 3)
        return mask

    def _detect_blue_masks(self, bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lower = np.array([90, 60, 40], dtype=np.uint8)
        upper = np.array([135, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.medianBlur(mask, 3)
        return mask

    def _quad_score(self, cnt: np.ndarray) -> Tuple[float, float, float]:
        area = float(cv2.contourArea(cnt))
        if area <= 0:
            return 0.0, 0.0, 0.0
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        w = float(rect[1][0])
        h = float(rect[1][1])
        if w <= 1 or h <= 1:
            return 0.0, 0.0, 0.0
        ratio = min(w, h) / max(w, h)
        hull = cv2.convexHull(cnt)
        s = float(cv2.contourArea(hull))
        solidity = area / (s + 1e-9)
        quadness = 1.0 if len(approx) == 4 else max(0.0, 1.0 - abs(len(approx) - 4) * 0.15)
        return ratio, solidity, quadness

    def _round_score(self, cnt: np.ndarray) -> float:
        area = float(cv2.contourArea(cnt))
        if area <= 0:
            return 0.0
        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            return 0.0
        circ = 4 * math.pi * area / (peri * peri + 1e-9)
        return float(np.clip(circ, 0.0, 1.0))

    def _extract_candidates(self, mask: np.ndarray, bgr: np.ndarray, min_area: int) -> List[Tuple[BBox, float, str]]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        H, W = mask.shape[:2]
        cands = []
        for c in contours:
            a = cv2.contourArea(c)
            if a < max(min_area, 0.0005 * W * H):
                continue
            x, y, w, h = cv2.boundingRect(c)
            b = BBox(float(x), float(y), float(x + w), float(y + h))
            ar = min(w, h) / max(w, h)
            ratio, solidity, quadness = self._quad_score(c)
            rscore = self._round_score(c)
            edge = cv2.Canny(mask[y : y + h, x : x + w], 40, 120)
            ed = float(np.mean(edge > 0))
            qscore = 0.35 * ratio + 0.35 * quadness + 0.2 * solidity + 0.1 * ed
            score = float(np.clip(qscore, 0.0, 1.0))
            label = "seal" if ar >= 0.7 else "stamp"
            if rscore > 0.55 and ar >= 0.7:
                label = "stamp"
                score = max(score, float(rscore))
            cands.append((b, score, label))
        return cands

    def _hough_round(self, bgr: np.ndarray) -> List[Tuple[BBox, float, str]]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 1.2)
        rows = gray.shape[0]
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 16, param1=80, param2=28, minRadius=max(8, rows // 100), maxRadius=rows // 3)
        out = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for c in circles[0, :]:
                cx, cy, r = int(c[0]), int(c[1]), int(c[2])
                b = BBox(float(cx - r), float(cy - r), float(cx + r), float(cy + r))
                out.append((b, 0.6, "stamp"))
        return out

    def detect(self, image: Image.Image) -> List[Detection]:
        if not CV2_AVAILABLE:
            return []
        bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        H, W = bgr.shape[:2]
        mask_r = self._detect_red_masks(bgr)
        mask_b = self._detect_blue_masks(bgr)
        c1 = self._extract_candidates(mask_r, bgr, min_area=int(0.0008 * W * H))
        c2 = self._extract_candidates(mask_b, bgr, min_area=int(0.0008 * W * H))
        hr = self._hough_round(bgr)
        allc = c1 + c2 + hr
        dets = []
        for i, (b, sc, lb) in enumerate(allc):
            dets.append(Detection(id=f"cv_{i}", class_name=lb, score=float(sc), bbox=b, pipeline="cv"))
        return dets


class SquareRefiner:
    @staticmethod
    def refine(image: Image.Image, det: Detection) -> Detection:
        if not CV2_AVAILABLE:
            return det
        if Labeling.refined_label(det.class_name) not in {"seal", "stamp"}:
            return det
        bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        H, W = bgr.shape[:2]
        x1, y1, x2, y2 = det.bbox.as_int()
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(W, x2)
        y2 = min(H, y2)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return det
        crop = bgr[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, np.array([0, 40, 30], np.uint8), np.array([10, 255, 255], np.uint8))
        m2 = cv2.inRange(hsv, np.array([170, 40, 30], np.uint8), np.array([180, 255, 255], np.uint8))
        mask = cv2.bitwise_or(m1, m2)
        if np.mean(mask > 0) < 0.03:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 7)
        mask = cv2.medianBlur(mask, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = -1.0
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 0.005 * mask.shape[0] * mask.shape[1]:
                continue
            rect = cv2.minAreaRect(c)
            w, h = rect[1][0], rect[1][1]
            if w <= 1 or h <= 1:
                continue
            ratio = min(w, h) / max(w, h)
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.03 * peri, True)
            quadness = 1.0 if len(approx) == 4 else max(0.0, 1.0 - abs(len(approx) - 4) * 0.2)
            hull = cv2.convexHull(c)
            solidity = cv2.contourArea(c) / (cv2.contourArea(hull) + 1e-9)
            score = 0.4 * ratio + 0.4 * quadness + 0.2 * solidity
            if score > best_score:
                best_score = score
                bx, by, bw, bh = cv2.boundingRect(c)
                best = (bx, by, bw, bh)
        if best is None:
            return det
        bx, by, bw, bh = best
        nx1, ny1, nx2, ny2 = x1 + bx, y1 + by, x1 + bx + bw, y1 + by + bh
        refined = BBox(float(nx1), float(ny1), float(nx2), float(ny2))
        refined = refined.pad_to_square(ratio=1.0, center=True).clamp(W, H)
        sc = float(min(1.0, det.score * (0.8 + 0.2 * best_score)))
        return Detection(id=det.id, class_name=det.class_name, score=sc, bbox=refined, pipeline=det.pipeline, meta=dict(det.meta, refined=True))


class Merger:
    @staticmethod
    def group_by_label(dets: List[Detection], mapping: Optional[Dict[str, str]], group_by_refined: bool) -> Dict[str, List[Detection]]:
        groups: Dict[str, List[Detection]] = {}
        for d in dets:
            k = Labeling.refined_label(d.class_name, mapping) if group_by_refined else d.class_name
            groups.setdefault(k, []).append(d)
        return groups

    @staticmethod
    def fuse(dets: List[Detection], iou_thr: float, method: str, mapping: Optional[Dict[str, str]], group_by_refined: bool) -> List[Detection]:
        if not dets:
            return []
        groups = Merger.group_by_label(dets, mapping, group_by_refined)
        out: List[Detection] = []
        for k, dl in groups.items():
            b, s, labels = BoxOps.to_array(dl)
            if b.shape[0] == 0:
                continue
            if method == "wbf":
                fb, fs = BoxOps.weighted_boxes_fusion(b, s, iou_thr=iou_thr)
                for i in range(fb.shape[0]):
                    bx = fb[i].tolist()
                    out.append(Detection(id=f"m_{k}_{i}", class_name=k, score=float(fs[i]), bbox=BBox(*bx), pipeline="merge"))
            elif method == "soft_linear":
                keep, ns = BoxOps.soft_nms(b, s, iou_thr=iou_thr, sigma=0.5, method="linear", score_thr=1e-3)
                for idx, sc in zip(keep.tolist(), ns.tolist()):
                    bx = b[idx].tolist()
                    out.append(Detection(id=f"m_{k}_{idx}", class_name=k, score=float(sc), bbox=BBox(*bx), pipeline="merge"))
            elif method == "soft_gaussian":
                keep, ns = BoxOps.soft_nms(b, s, iou_thr=iou_thr, sigma=0.5, method="gaussian", score_thr=1e-3)
                for idx, sc in zip(keep.tolist(), ns.tolist()):
                    bx = b[idx].tolist()
                    out.append(Detection(id=f"m_{k}_{idx}", class_name=k, score=float(sc), bbox=BBox(*bx), pipeline="merge"))
            elif method == "diou":
                keep = BoxOps.diou_nms(b, s, thr=iou_thr)
                for idx in keep.tolist():
                    bx = b[idx].tolist()
                    out.append(Detection(id=f"m_{k}_{idx}", class_name=k, score=float(s[idx]), bbox=BBox(*bx), pipeline="merge"))
            else:
                keep = BoxOps.greedy_nms(b, s, thr=iou_thr)
                for idx in keep.tolist():
                    bx = b[idx].tolist()
                    out.append(Detection(id=f"m_{k}_{idx}", class_name=k, score=float(s[idx]), bbox=BBox(*bx), pipeline="merge"))
        out.sort(key=lambda d: d.score, reverse=True)
        return out

    @staticmethod
    def reclassify_with_clip(image: Image.Image, clip: Optional[CLIPVerifier], dets: List[Detection], min_bg_margin: float = 0.05) -> List[Detection]:
        if clip is None or not dets:
            return dets
        idx_map = {i: d for i, d in enumerate(dets)}
        scores = clip.score_regions(image, dets)
        if not scores:
            return dets
        out = []
        for i, sc, lab in scores:
            d0 = idx_map[i]
            if lab == "background" and sc > max(0.5, d0.score + min_bg_margin):
                continue
            cname = Labeling.refined_label(lab)
            out.append(Detection(id=d0.id, class_name=cname, score=float(max(d0.score, sc)), bbox=d0.bbox, pipeline=d0.pipeline, meta=dict(d0.meta, clip_label=lab, clip_score=sc)))
        return out


class Visualizer:
    @staticmethod
    def draw(image: Image.Image, dets: List[Detection]) -> Image.Image:
        img = image.copy()
        draw = ImageDraw.Draw(img)
        W, H = image.size
        t = max(2, int(round(0.004 * max(W, H))))
        try:
            font = ImageFont.truetype("arial.ttf", max(12, int(round(0.012 * max(W, H)))))
        except:
            font = ImageFont.load_default()
        for d in dets:
            x1, y1, x2, y2 = d.bbox.as_xyxy()
            xi1, yi1, xi2, yi2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
            c = Labeling.color_for(Labeling.refined_label(d.class_name))
            draw.rectangle([xi1, yi1, xi2, yi2], outline=c, width=t)
            label = f"{Labeling.refined_label(d.class_name)} {d.score:.3f}"
            tw, th = draw.textlength(label, font=font), font.size
            y0 = max(0, yi1 - th - 4)
            draw.rectangle([xi1, y0, xi1 + int(tw) + 6, y0 + th + 4], fill=c)
            draw.text((xi1 + 3, y0 + 2), label, fill=(255, 255, 255), font=font)
        return img


class Rasterizer:
    @staticmethod
    def rasterize_pdf(path: Path) -> Generator[Tuple[int, Image.Image], None, None]:
        if PYMUPDF_AVAILABLE:
            doc = fitz.open(str(path))
            for i in range(len(doc)):
                page = doc[i]
                mat = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=mat)
                img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                yield i + 1, img
            return
        if PDF2IMAGE_AVAILABLE:
            pages = convert_from_path(str(path), dpi=200)
            for i, p in enumerate(pages):
                yield i + 1, p.convert("RGB")
            return
        raise RuntimeError("No PDF rasterization backend available")


class DatasetResolver:
    @staticmethod
    def resolve(source: str) -> List[Tuple[Optional[int], Image.Image, str]]:
        p = Path(source)
        out: List[Tuple[Optional[int], Image.Image, str]] = []
        if p.is_dir():
            exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".pdf"}
            paths = sorted([x for x in p.rglob("*") if x.suffix.lower() in exts])
            for x in paths:
                if x.suffix.lower() == ".pdf":
                    for page_no, img in Rasterizer.rasterize_pdf(x):
                        out.append((page_no, img, f"{x.stem}_page_{page_no:04d}"))
                else:
                    img = Image.open(x).convert("RGB")
                    out.append((None, img, x.stem))
            return out
        if p.is_file():
            if p.suffix.lower() == ".pdf":
                for page_no, img in Rasterizer.rasterize_pdf(p):
                    out.append((page_no, img, f"{p.stem}_page_{page_no:04d}"))
            else:
                img = Image.open(p).convert("RGB")
                out.append((None, img, p.stem))
            return out
        raise FileNotFoundError(str(source))


class Orchestrator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.out_dir = Path(args.output)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.mapping = self._load_mapping(args.label_map)
        self.pipelines: Dict[str, BaseDetector] = {}
        self.clip_verifier: Optional[CLIPVerifier] = None
        self._init_pipelines()

    def _load_mapping(self, path: Optional[str]) -> Optional[Dict[str, str]]:
        if not path:
            return None
        p = Path(path)
        if not p.exists():
            return None
        m = json.loads(p.read_text(encoding="utf-8"))
        return {str(k).strip().lower(): str(v).strip().lower() for k, v in m.items()}

    def _init_pipelines(self) -> None:
        if "yolo" in self.args.pipelines and YOLO_AVAILABLE and CV2_AVAILABLE:
            imgsz_list = self._parse_int_list(self.args.yolo_imgsz)
            conf_seq = self._parse_conf_seq(self.args.conf, self.args.min_conf)
            self.pipelines["yolo"] = YOLODetector(self.args.yolo_model, device=self.args.device, conf=self.args.conf, iou=self.args.iou, max_det=self.args.max_det, agnostic=True, tta=not self.args.no_tta, imgsz_list=imgsz_list, conf_seq=conf_seq)
        if "open_vocabulary" in self.args.pipelines and TRANSFORMERS_AVAILABLE:
            self.pipelines["open_vocabulary"] = OpenVocabularyDetector(self.args.ov_model, device=self.args.device, min_threshold=self.args.ov_conf, queries=self._queries())
        if "cv" in self.args.pipelines and CV2_AVAILABLE:
            self.pipelines["cv"] = CVSealDetector()
        if "clip" in self.args.pipelines and TRANSFORMERS_AVAILABLE:
            self.clip_verifier = CLIPVerifier(self.args.clip_model, device=self.args.device)

    def _queries(self) -> List[str]:
        q = ["square seal", "seal", "stamp", "round stamp", "company seal", "official seal", "signature"]
        if self.args.ov_queries:
            ex = [t.strip() for t in self.args.ov_queries.split(",") if t.strip()]
            q = list(dict.fromkeys(q + ex))
        return q

    def _parse_int_list(self, s: str) -> List[int]:
        vals = []
        for t in s.split(","):
            t = t.strip()
            if not t:
                continue
            try:
                vals.append(int(t))
            except:
                pass
        vals = [x for x in vals if x > 0]
        return sorted(list(dict.fromkeys(vals))) if vals else [1280]

    def _parse_conf_seq(self, conf: float, min_conf: float) -> List[float]:
        c0 = max(min_conf, conf)
        c1 = max(min_conf, conf * 0.75)
        c2 = min_conf
        return sorted(list(dict.fromkeys([c0, c1, c2])), reverse=True)

    def _filter_labels(self, dets: List[Detection]) -> List[Detection]:
        if not self.args.allowed_labels:
            return dets
        allow = {a.strip().lower() for a in self.args.allowed_labels.split(",") if a.strip()}
        out = []
        for d in dets:
            rn = Labeling.refined_label(d.class_name, self.mapping)
            if rn in allow or d.class_name.lower() in allow:
                out.append(d)
        return out if out else dets

    def _refine_square(self, image: Image.Image, dets: List[Detection]) -> List[Detection]:
        out = []
        for d in dets:
            out.append(SquareRefiner.refine(image, d))
        return out

    def _annotate_and_save(self, image: Image.Image, dets: List[Detection], tag: str, stem: str) -> Path:
        ann = Visualizer.draw(image, dets)
        p = self.out_dir / f"{stem}_{tag}.png"
        ann.save(p)
        return p

    def _detection_to_json(self, dets: List[Detection]) -> List[Dict[str, Any]]:
        items = []
        for d in dets:
            x1, y1, x2, y2 = d.bbox.as_int()
            items.append(
                {
                    "id": d.id,
                    "class": d.class_name,
                    "refined_class": Labeling.refined_label(d.class_name, self.mapping),
                    "score": float(d.score),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "w": max(0, x2 - x1), "h": max(0, y2 - y1)},
                    "pipeline": d.pipeline,
                    "meta": d.meta,
                }
            )
        return items

    def process(self) -> None:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
        items = DatasetResolver.resolve(self.args.input)
        for page_no, image, stem in items:
            W, H = image.size
            page_dir = self.out_dir / stem
            page_dir.mkdir(parents=True, exist_ok=True)
            orig_path = page_dir / "original.png"
            image.save(orig_path)
            per_pipeline: Dict[str, List[Detection]] = {}
            for name, pipe in self.pipelines.items():
                dets = pipe.detect(image)
                dets = [Detection(id=f"{name}_{i}", class_name=d.class_name, score=d.score, bbox=d.bbox.clamp(W, H), pipeline=name, meta=d.meta) for i, d in enumerate(dets)]
                if self.args.square_refine:
                    dets = self._refine_square(image, dets)
                dets = self._filter_labels(dets)
                per_pipeline[name] = dets
                ann_path = self._annotate_and_save(image, dets, tag=name, stem=stem)
                res = {"page": page_no, "pipeline": name, "image": str(orig_path), "annotated": str(ann_path), "detections": self._detection_to_json(dets)}
                (page_dir / f"{name}.json").write_text(json.dumps(res, ensure_ascii=False, separators=(",", ":")))
            merged = [d for v in per_pipeline.values() for d in v]
            if self.clip_verifier is not None and merged:
                merged = Merger.reclassify_with_clip(image, self.clip_verifier, merged)
            merged = Merger.fuse(merged, iou_thr=self.args.merge_iou, method=self.args.merge_method, mapping=self.mapping, group_by_refined=True)
            if self.args.square_refine:
                merged = self._refine_square(image, merged)
            ann_path = self._annotate_and_save(image, merged, tag="merged", stem=stem)
            merged_res = {"page": page_no, "pipeline": "merged", "image": str(orig_path), "annotated": str(ann_path), "detections": self._detection_to_json(merged)}
            (page_dir / f"merged.json").write_text(json.dumps(merged_res, ensure_ascii=False, separators=(",", ":")))

    @staticmethod
    def available_pipelines() -> List[str]:
        ps = []
        if YOLO_AVAILABLE and CV2_AVAILABLE:
            ps.append("yolo")
        if TRANSFORMERS_AVAILABLE:
            ps.append("open_vocabulary")
            ps.append("clip")
        if CV2_AVAILABLE:
            ps.append("cv")
        return ps


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, default="runs/seal_detection")
    p.add_argument("--pipelines", type=str, default="yolo,cv,open_vocabulary,clip")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--yolo_model", type=str, default="yolov8x.pt")
    p.add_argument("--yolo_imgsz", type=str, default="1024,1280,1536")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--min_conf", type=float, default=0.05)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--max_det", type=int, default=300)
    p.add_argument("--no_tta", action="store_true")
    p.add_argument("--ov_model", type=str, default="IDEA-Research/grounding-dino-base")
    p.add_argument("--ov_conf", type=float, default=0.15)
    p.add_argument("--ov_queries", type=str, default="")
    p.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32")
    p.add_argument("--merge_method", type=str, default="wbf", choices=["wbf", "nms", "soft_linear", "soft_gaussian", "diou"])
    p.add_argument("--merge_iou", type=float, default=0.55)
    p.add_argument("--square_refine", action="store_true")
    p.add_argument("--allowed_labels", type=str, default="")
    p.add_argument("--label_map", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    req = [t.strip() for t in args.pipelines.split(",") if t.strip()]
    avail = Orchestrator.available_pipelines()
    args.pipelines = [r for r in req if r in avail]
    if not args.pipelines:
        args.pipelines = avail
    orch = Orchestrator(args)
    orch.process()


if __name__ == "__main__":
    main()