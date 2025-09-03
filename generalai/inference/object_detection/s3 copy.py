import argparse
import json
import logging
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Optional imports with availability flags
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = None  # ensure name exists

try:
    from transformers import (
        AutoConfig,
        AutoProcessor,
        AutoModelForZeroShotObjectDetection,
        CLIPModel,
        CLIPProcessor,
    )
    TRANSFORMERS_AVAILABLE = True
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
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except Exception:
    PYMUPDF_AVAILABLE = False


class PipelineType(Enum):
    YOLO = "yolo"
    GROUNDING_DINO = "grounding_dino"
    OWL_VIT = "owl_vit"
    CLIP = "clip"  # can be standalone or "refine" mode


@dataclass
class BBox:
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self):
        return self.x + self.w

    @property
    def y2(self):
        return self.y + self.h

    def to_xyxy(self):
        return self.x, self.y, self.x2, self.y2

    @staticmethod
    def from_xyxy(xmin, ymin, xmax, ymax):
        xmin, ymin, xmax, ymax = float(xmin), float(ymin), float(xmax), float(ymax)
        return BBox(
            x=int(round(min(xmin, xmax))),
            y=int(round(min(ymin, ymax))),
            w=int(round(max(1, abs(xmax - xmin)))),
            h=int(round(max(1, abs(ymax - ymin)))),
        )

    def to_dict(self):
        return {
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
            "x2": self.x2,
            "y2": self.y2,
        }


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


class BaseDetector:
    def __init__(self, name, device=None):
        self.name = name
        # pick device safely even if torch is missing
        if device is not None:
            self.device = device
        else:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"

    def detect(self, image, **kwargs):
        raise NotImplementedError

    def get_output_dir(self, base_dir):
        return base_dir / self.name


class YOLODetector(BaseDetector):
    def __init__(self, model_path, device=None, conf=0.25, iou=0.45, imgsz="auto"):
        super().__init__("yolo", device)
        if not YOLO_AVAILABLE:
            raise RuntimeError("ultralytics not installed")
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz  # can be int or "auto"

    def _auto_imgsz(self, image_size):
        # choose a reasonable imgsz based on input image dims, multiple of 32
        w, h = image_size
        base = max(640, min(1920, max(w, h)))
        # round up to nearest multiple of 32
        return int((base + 31) // 32) * 32

    def detect(self, image, **kwargs):
        # ultralytics can handle PIL.Image directly; avoid cv2 dependency
        conf = kwargs.get("conf", self.conf)
        iou = kwargs.get("iou", self.iou)
        imgsz = kwargs.get("imgsz", self.imgsz)
        if imgsz == "auto":
            imgsz = self._auto_imgsz(image.size)

        results = self.model.predict(
            image,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=self.device,
            verbose=False,
        )

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

                det = Detection(
                    id=f"yolo_{i}",
                    class_name=class_name,
                    score=float(score),
                    bbox=bbox,
                    pipeline="yolo",
                )
                detections.append(det)

        return detections


class OpenVocabularyDetector(BaseDetector):
    def __init__(self, model_id, device=None, min_threshold=0.15):
        super().__init__("open_vocabulary", device)
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers not installed")

        self.model_id = model_id
        self.min_threshold = min_threshold

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        if TORCH_AVAILABLE:
            self.model.to(self.device)
        self.model.eval()

        self.config = AutoConfig.from_pretrained(model_id)
        self.is_grounding_dino = self._detect_grounding_dino()

    def _detect_grounding_dino(self):
        mt = getattr(self.config, "model_type", "").lower()
        return "grounding" in mt and "dino" in mt

    def detect(self, image, queries=None, threshold=None, **kwargs):
        if queries is None:
            queries = ["signature.","red signature." "stamp.", "seal.", "square_stamp.", "cricle stamp."]

        if threshold is None:
            threshold = self.min_threshold
        threshold = max(threshold, self.min_threshold)

        if self.is_grounding_dino:
            # GroundingDINO expects a single text string with periods
            text_str = " ".join([q.strip() + "." for q in queries])
            inputs = self.processor(images=image, text=text_str, return_tensors="pt")
        else:
            # OWL-ViT style: batch text prompts
            inputs = self.processor(text=queries, images=image, return_tensors="pt", padding=True)

        if TORCH_AVAILABLE:
            inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with (torch.inference_mode() if TORCH_AVAILABLE else _nullcontext()):
            outputs = self.model(**inputs)

        # target_sizes expects (h, w)
        target_sizes = None
        if TORCH_AVAILABLE:
            target_sizes = torch.tensor([image.size[::-1]], device=self.device)
        else:
            # if torch not available, transformers won't run anyway; guard above ensures torch is there
            pass

        if self.is_grounding_dino:
            processed = self.processor.post_process_grounded_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
              
            )[0]
            boxes = processed.get("boxes")
            scores = processed.get("scores")
            labels = processed.get("text_labels", [""] * (len(scores) if scores is not None else 0))
        else:
            processed = self.processor.post_process_grounded_object_detection(
                outputs=outputs, target_sizes=target_sizes
            )[0]
            boxes = processed.get("boxes")
            scores = processed.get("scores")
            label_ids = processed.get("labels")
            labels = [queries[int(idx)] for idx in label_ids.cpu().numpy()]

        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()

        detections: List[Detection] = []
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            if score < threshold:
                continue

            x1, y1, x2, y2 = box
            bbox = BBox.from_xyxy(x1, y1, x2, y2)

            det = Detection(
                id=f"ovd_{i}",
                class_name=str(label),
                score=float(score),
                bbox=bbox,
                pipeline=("grounding_dino" if self.is_grounding_dino else "owl_vit"),
            )
            detections.append(det)

        return detections


class CLIPClassifier(BaseDetector):
    def __init__(self, model_id="openai/clip-vit-base-patch32", device=None):
        super().__init__("clip", device)
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers not installed")

        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id)
        if TORCH_AVAILABLE:
            self.model.to(self.device)
        self.model.eval()

        self.labels = ["signature", "stamp", "seal", ]
        self.prompts = [
            "a handwritten signature",
            "an ink stamp",
            "an official seal",
         
        ]

    def classify_regions(self, image, regions: List[Detection], **kwargs):
        crops = []
        for region in regions:
            x1, y1, x2, y2 = region.bbox.to_xyxy()
            crop = image.crop((x1, y1, x2, y2))
            crops.append(crop.convert("RGB"))

        if not crops:
            return []

        inputs = self.processor(text=self.prompts, images=crops, return_tensors="pt", padding=True)
        if TORCH_AVAILABLE:
            inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with (torch.inference_mode() if TORCH_AVAILABLE else _nullcontext()):
            outputs = self.model(**inputs)

        probs = outputs.logits_per_image.softmax(dim=-1).detach().cpu().numpy()

        results = []
        for i, row in enumerate(probs):
            best_idx = int(np.argmax(row))
            # Skip "background"
            if self.labels[best_idx] != "background":
                results.append(
                    {
                        "region": regions[i],
                        "class": self.labels[best_idx],
                        "score": float(row[best_idx]),
                    }
                )

        return results

    def detect(self, image, proposals: Optional[List[Detection]] = None, **kwargs):
        # standalone detection: use naive sliding-window proposals
        if proposals is None:
            proposals = self._generate_proposals(image)

        classified = self.classify_regions(image, proposals)

        detections: List[Detection] = []
        for i, item in enumerate(classified):
            det = Detection(
                id=f"clip_{i}",
                class_name=item["class"],
                score=item["score"],
                bbox=item["region"].bbox,
                pipeline="clip",
            )
            detections.append(det)

        return detections

    def _generate_proposals(self, image: Image.Image):
        w, h = image.size
        proposals = []
        stride = max(128, min(w, h) // 8)
        size = max(256, min(w, h) // 4)

        count = 0
        for y in range(0, max(1, h - size + 1), stride):
            for x in range(0, max(1, w - size + 1), stride):
                bbox = BBox(x=x, y=y, w=size, h=size)
                det = Detection(
                    id=f"prop_{count}",
                    class_name="proposal",
                    score=1.0,
                    bbox=bbox,
                    pipeline="proposal",
                )
                proposals.append(det)
                count += 1

        return proposals


class DocumentProcessor:
    def __init__(self, input_path, output_dir, pipelines_config):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.pipelines_config = pipelines_config
        self.pipelines: Dict[str, BaseDetector] = {}
        self._initialize_pipelines()

    def _initialize_pipelines(self):
        for name, config in self.pipelines_config.items():
            if not config.get("enabled", False):
                continue

            pipeline_type = config["type"]

            if pipeline_type == PipelineType.YOLO.value:
                if YOLO_AVAILABLE:
                    self.pipelines[name] = YOLODetector(
                        model_path=config["model_path"],
                        device=config.get("device"),
                        conf=config.get("conf", 0.25),
                        iou=config.get("iou", 0.45),
                        imgsz=config.get("imgsz", "auto"),
                    )
                else:
                    logging.warning(f"YOLO not available, skipping {name}")

            elif pipeline_type == PipelineType.GROUNDING_DINO.value:
                if TRANSFORMERS_AVAILABLE:
                    self.pipelines[name] = OpenVocabularyDetector(
                        model_id=config["model_id"],
                        device=config.get("device"),
                        min_threshold=config.get("min_threshold", 0.15),
                    )
                else:
                    logging.warning(f"Transformers not available, skipping {name}")

            elif pipeline_type == PipelineType.OWL_VIT.value:
                if TRANSFORMERS_AVAILABLE:
                    self.pipelines[name] = OpenVocabularyDetector(
                        model_id=config["model_id"],
                        device=config.get("device"),
                        min_threshold=config.get("min_threshold", 0.15),
                    )
                else:
                    logging.warning(f"Transformers not available, skipping {name}")

            elif pipeline_type == PipelineType.CLIP.value:
                if TRANSFORMERS_AVAILABLE:
                    self.pipelines[name] = CLIPClassifier(
                        model_id=config.get("model_id", "openai/clip-vit-base-patch32"),
                        device=config.get("device"),
                    )
                else:
                    logging.warning(f"Transformers not available, skipping {name}")

    def process(self):
        if self.input_path.suffix.lower() == ".pdf":
            return self._process_pdf()
        else:
            return self._process_image()

    def _process_pdf(self):
        pages = list(self._rasterize_pdf())
        results = []
        for page_no, image in pages:
            logging.info(f"Processing page {page_no}")
            page_result = self._process_single_image(image, page_no)
            results.append(page_result)
        return self._save_results(results)

    def _process_image(self):
        image = Image.open(self.input_path).convert("RGB")
        page_result = self._process_single_image(image, page_no=1)
        return self._save_results([page_result])

    def _process_single_image(self, image, page_no):
        w, h = image.size

        base_page_dir = self.output_dir / f"page_{page_no:04d}"
        base_page_dir.mkdir(parents=True, exist_ok=True)

        image_path = base_page_dir / "original.png"
        image.save(image_path)

        page_result = ProcessedPage(page_no=page_no, width=w, height=h, image_path=str(image_path))

        all_detections: List[Detection] = []
        refined_dets: List[Detection] = []

        for pipeline_name, pipeline in self.pipelines.items():
            logging.info(f"Running {pipeline_name} on page {page_no}")

            pipeline_cfg = self.pipelines_config.get(pipeline_name, {})
            pipeline_dir = base_page_dir / pipeline_name
            pipeline_dir.mkdir(exist_ok=True)

            try:
                # Special handling: if CLIP is in "refine" mode, skip direct detect here
                if pipeline_cfg.get("type") == PipelineType.CLIP.value and pipeline_cfg.get("mode") == "refine":
                    logging.info("Skipping standalone CLIP detect (refine mode enabled).")
                    page_result.detections[pipeline_name] = []
                    continue

                # For OV detectors, pass custom queries/threshold if provided
                if isinstance(pipeline, OpenVocabularyDetector):
                    queries = pipeline_cfg.get("queries")
                    threshold = pipeline_cfg.get("min_threshold")
                    detections = pipeline.detect(image, queries=queries, threshold=threshold)
                elif isinstance(pipeline, YOLODetector):
                    detections = pipeline.detect(image)
                elif isinstance(pipeline, CLIPClassifier):
                    # standalone clip detection
                    detections = pipeline.detect(image)
                else:
                    detections = pipeline.detect(image)

                # attach page_no, clamp boxes, save per-pipeline annotated + crops
                detections = [self._clamp_detection(det, w, h, page_no) for det in detections]
                page_result.detections[pipeline_name] = detections
                all_detections.extend(detections)

                annotated = self._annotate_image(image, detections)
                (pipeline_dir / "annotated.png").write_bytes(self._image_to_png_bytes(annotated))

                # save crops
                for i, det in enumerate(detections):
                    x1, y1, x2, y2 = det.bbox.to_xyxy()
                    if x2 > x1 and y2 > y1:
                        crop = image.crop((x1, y1, x2, y2))
                        crop_path = pipeline_dir / f"crop_{i:04d}_{safe_name(det.class_name)}.png"
                        crop.save(crop_path)

                # save results json per pipeline
                result_json = {
                    "page_no": page_no,
                    "pipeline": pipeline_name,
                    "detections": [
                        {
                            "id": det.id,
                            "class": det.class_name,
                            "score": det.score,
                            "bbox": det.bbox.to_dict(),
                        }
                        for det in detections
                    ],
                }
                (pipeline_dir / "results.json").write_text(json.dumps(result_json, indent=2))

            except Exception as e:
                logging.exception(f"Pipeline {pipeline_name} failed: {e}")
                page_result.detections[pipeline_name] = []

        # Optional CLIP refinement if clip pipeline is enabled in refine mode
        if "clip" in self.pipelines and self.pipelines_config.get("clip", {}).get("mode") == "refine":
            try:
                clip_model: CLIPClassifier = self.pipelines["clip"]  # type: ignore
                if all_detections:
                    classified = clip_model.classify_regions(image, all_detections)
                    for i, item in enumerate(classified):
                        det = Detection(
                            id=f"clip_refine_{i}",
                            class_name=item["class"],
                            score=item["score"],
                            bbox=item["region"].bbox,
                            pipeline="clip_refine",
                            page_no=page_no,
                        )
                        refined_dets.append(det)

                    page_result.detections["clip_refine"] = refined_dets

                    # save refinement outputs
                    clip_ref_dir = base_page_dir / "clip_refine"
                    clip_ref_dir.mkdir(exist_ok=True)
                    annotated_ref = self._annotate_image(image, refined_dets)
                    (clip_ref_dir / "annotated.png").write_bytes(self._image_to_png_bytes(annotated_ref))

                    (clip_ref_dir / "results.json").write_text(
                        json.dumps(
                            {
                                "page_no": page_no,
                                "pipeline": "clip_refine",
                                "detections": [
                                    {
                                        "id": d.id,
                                        "class": d.class_name,
                                        "score": d.score,
                                        "bbox": d.bbox.to_dict(),
                                    }
                                    for d in refined_dets
                                ],
                            },
                            indent=2,
                        )
                    )
            except Exception as e:
                logging.exception(f"CLIP refinement failed: {e}")
                page_result.detections["clip_refine"] = []

        # Save a single page-level annotated image across all pipelines (including refined)
        try:
            page_all = all_detections + refined_dets
            page_annotated = self._annotate_image(image, page_all)
            (base_page_dir / "page_annotated.png").write_bytes(self._image_to_png_bytes(page_annotated))
        except Exception as e:
            logging.exception(f"Failed to save page_annotated: {e}")

        return page_result

    def _rasterize_pdf(self):
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
            pages = convert_from_path(str(pdf_path), dpi=150)
            for i, page in enumerate(pages):
                yield i + 1, page.convert("RGB")

        else:
            raise RuntimeError("No PDF rasterization library available (install PyMuPDF or pdf2image)")

    def _annotate_image(self, image: Image.Image, detections: List[Detection]):
        img = image.convert("RGB").copy()
        draw = ImageDraw.Draw(img)
        font = self._get_font()

        # colors per known class; fallback uses pipeline-based color
        class_colors = {
            "signature": (255, 200, 0),
            "stamp": (0, 200, 255),
            "seal": (255, 0, 100),
            "document": (0, 255, 0),
            "text": (255, 255, 0),
        }
        pipeline_fallback = {
            "yolo": (255, 255, 255),
            "grounding_dino": (255, 128, 0),
            "owl_vit": (128, 255, 0),
            "clip": (128, 128, 255),
            "clip_refine": (255, 0, 255),
            "proposal": (128, 128, 128),
        }

        w, h = img.size

        for det in detections:
            cname = det.class_name.lower()
            color = class_colors.get(cname, pipeline_fallback.get(det.pipeline, (255, 255, 255)))
            x1, y1, x2, y2 = det.bbox.to_xyxy()

            # clamp to image bounds
            x1 = max(0, min(int(x1), w - 1))
            y1 = max(0, min(int(y1), h - 1))
            x2 = max(0, min(int(x2), w - 1))
            y2 = max(0, min(int(y2), h - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            label = f"{det.class_name} {det.score:.2f} ({det.pipeline})"
            tw, th = self._measure_text(draw, label, font)

            # place label above box if possible, otherwise below
            ty = y1 - th - 4
            if ty < 0:
                ty = y1 + 4
                if ty + th + 4 > h:
                    ty = max(0, h - th - 4)

            # background for text (black) and text in color
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

    def _save_results(self, results: List[ProcessedPage]):
        summary = {
            "input": str(self.input_path),
            "output_dir": str(self.output_dir),
            "timestamp": datetime.now().isoformat(),
            "pipelines": list(self.pipelines.keys()),
            "pages": [],
        }

        for page_result in results:
            page_summary = {
                "page_no": page_result.page_no,
                "size": {"width": page_result.width, "height": page_result.height},
                "detections_by_pipeline": {},
            }

            for pipeline_name, detections in page_result.detections.items():
                page_summary["detections_by_pipeline"][pipeline_name] = {
                    "count": len(detections),
                    "classes": sorted(list(set(d.class_name for d in detections))),
                }

            summary["pages"].append(page_summary)

        summary_path = self.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))

        return summary

    @staticmethod
    def _image_to_png_bytes(img: Image.Image) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    @staticmethod
    def _get_font():
        # Try a few common fonts, then default
        try:
            return ImageFont.truetype("arial.ttf", 16)
        except Exception:
            try:
                return ImageFont.truetype("DejaVuSans.ttf", 16)
            except Exception:
                return ImageFont.load_default()

    @staticmethod
    def _measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
        try:
            tb = draw.textbbox((0, 0), text, font=font)
            return tb[2] - tb[0], tb[3] - tb[1]
        except Exception:
            return draw.textsize(text, font=font)


def _nullcontext():
    # Fallback no-op context manager for when torch is unavailable
    class _NC:
        def __enter__(self): return None
        def __exit__(self, exc_type, exc, tb): return False
    return _NC()


def safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


def parse_args():
    parser = argparse.ArgumentParser(description="Document detection pipeline (YOLO, GroundingDINO, OWL-ViT, CLIP)")
    parser.add_argument("--input", required=True, help="Input PDF or image path")
    parser.add_argument("--output", default="output", help="Output directory")

    parser.add_argument(
        "--pipelines",
        nargs="+",
        default=["yolo"],
        choices=["yolo", "grounding_dino", "owl_vit", "clip", "all"],
        help="Pipelines to run",
    )

    # YOLO options
    parser.add_argument("--yolo_model", default="yolov8n.pt", help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence/threshold (also used by OV models)")
    parser.add_argument("--iou", type=float, default=0.25, help="IoU threshold for YOLO")
    parser.add_argument("--imgsz", default="auto", help="YOLO image size (int) or 'auto'")

    # Open-vocab models
    parser.add_argument("--grounding_dino_model", default="IDEA-Research/grounding-dino-base",
                        help="Grounding DINO model ID")
    parser.add_argument("--owl_vit_model", default="google/owlv2-base-patch16-ensemble",
                        help="OWL-ViT model ID")
    parser.add_argument("--queries", default="signature,stamp,seal",
                        help="Comma-separated queries for open-vocab detectors")

    # CLIP options
    parser.add_argument("--clip_model", default="openai/clip-vit-base-patch32", help="CLIP model ID")
    parser.add_argument("--clip_mode", choices=["standalone", "refine"], default="refine",
                        help="CLIP usage: standalone detector or refine other detections")

    # Device
    parser.add_argument("--device", default=None, help="Device (e.g., cuda, cpu)")

    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    pipelines_to_run = args.pipelines
    if "all" in pipelines_to_run:
        pipelines_to_run = ["yolo", "grounding_dino", "owl_vit", "clip"]

    # Parse imgsz to int if provided
    imgsz = args.imgsz
    try:
        if isinstance(imgsz, str) and imgsz.lower() != "auto":
            imgsz = int(imgsz)
    except Exception:
        imgsz = "auto"

    # Parse queries
    queries = [q.strip() for q in args.queries.split(",") if q.strip()]

    pipelines_config: Dict[str, Dict[str, Any]] = {}

    if "yolo" in pipelines_to_run:
        pipelines_config["yolo"] = {
            "enabled": True,
            "type": PipelineType.YOLO.value,
            "model_path": args.yolo_model,
            "device": args.device,
            "conf": args.conf,
            "iou": args.iou,
            "imgsz": imgsz,
        }

    if "grounding_dino" in pipelines_to_run:
        pipelines_config["grounding_dino"] = {
            "enabled": True,
            "type": PipelineType.GROUNDING_DINO.value,
            "model_id": args.grounding_dino_model,
            "device": args.device,
            "min_threshold": args.conf,
            "queries": queries,
        }

    if "owl_vit" in pipelines_to_run:
        pipelines_config["owl_vit"] = {
            "enabled": True,
            "type": PipelineType.OWL_VIT.value,
            "model_id": args.owl_vit_model,
            "device": args.device,
            "min_threshold": args.conf,
            "queries": queries,
        }

    if "clip" in pipelines_to_run:
        pipelines_config["clip"] = {
            "enabled": True,
            "type": PipelineType.CLIP.value,
            "model_id": args.clip_model,
            "device": args.device,
            "mode": args.clip_mode,  # "refine" or "standalone"
        }

    processor = DocumentProcessor(
        input_path=args.input, output_dir=Path(args.output), pipelines_config=pipelines_config
    )

    try:
        summary = processor.process()
        logging.info(f"Processing complete. Results saved to {args.output}")
        logging.info(f"Summary: {json.dumps(summary, indent=2)}")
    except Exception as e:
        logging.exception(f"Processing failed: {e}")
        raise


if __name__ == "__main__":
    main()