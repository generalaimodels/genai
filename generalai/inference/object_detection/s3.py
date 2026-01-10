
import argparse
import json
import hashlib
import time
import logging
import sys
import re
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Generator
from datetime import datetime
from enum import Enum
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import io
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
    from transformers import (
        AutoConfig,
        AutoProcessor,
        AutoModelForZeroShotObjectDetection,
        CLIPModel,
        CLIPProcessor
    )
    TRANSFORMERS_AVAILABLE = True
except:
    TRANSFORMERS_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except:
    YOLO_AVAILABLE = False

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except:
    PDF2IMAGE_AVAILABLE = False

try:
    import pypdfium2 as pdfium
    PYPDFIUM_AVAILABLE = True
except:
    PYPDFIUM_AVAILABLE = False

try:
    import fitz
    PYMUPDF_AVAILABLE = True
except:
    PYMUPDF_AVAILABLE = False


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
    def x2(self):
        return self.x + self.w

    @property
    def y2(self):
        return self.y + self.h

    def to_xyxy(self):
        return self.x, self.y, self.x2, self.y2

    @staticmethod
    def from_xyxy(xmin, ymin, xmax, ymax):
        return BBox(
            x=int(round(xmin)),
            y=int(round(ymin)),
            w=int(round(max(1, xmax - xmin))),
            h=int(round(max(1, ymax - ymin)))
        )

    def to_dict(self):
        return {
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
            "x2": self.x2,
            "y2": self.y2
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
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def detect(self, image, **kwargs):
        raise NotImplementedError

    def get_output_dir(self, base_dir):
        return base_dir / self.name


class YOLODetector(BaseDetector):
    def __init__(self, model_path, device=None, conf=0.25, iou=0.45):
        super().__init__("yolo", device)
        if not YOLO_AVAILABLE:
            raise RuntimeError("ultralytics not installed")
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou

    def detect(self, image, **kwargs):
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        conf = kwargs.get("conf", self.conf)
        iou = kwargs.get("iou", self.iou)
        imgsz = kwargs.get("imgsz", 1280)
        
        results = self.model.predict(
            image,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=self.device,
            verbose=False
        )
        
        detections = []
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
                    pipeline="yolo"
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
        self.model.to(self.device).eval()
        
        self.config = AutoConfig.from_pretrained(model_id)
        self.is_grounding_dino = self._detect_grounding_dino()

    def _detect_grounding_dino(self):
        mt = getattr(self.config, "model_type", "").lower()
        return "grounding" in mt and "dino" in mt

    @torch.inference_mode()
    def detect(self, image, queries=None, threshold=0.15, **kwargs):
        if queries is None:
            queries = ["signature", "stamp", "seal", "document", "text"]
        
        threshold = max(threshold, self.min_threshold)
        
        if self.is_grounding_dino:
            text_str = " ".join([q.strip() + "." for q in queries])
            inputs = self.processor(images=image, text=text_str, return_tensors="pt")
        else:
            inputs = self.processor(text=queries, images=image, return_tensors="pt", padding=True)
        
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        outputs = self.model(**inputs)
        
        target_sizes = torch.tensor([image.size[::-1]], device=self.device)
        
        if self.is_grounding_dino:
            processed = self.processor.post_process_grounded_object_detection(
                outputs=outputs,
                target_sizes=target_sizes
            )[0]
        else:
            processed = self.processor.post_process_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=threshold
            )[0]
        
        boxes = processed["boxes"].cpu().numpy()
        scores = processed["scores"].cpu().numpy()
        
        if self.is_grounding_dino:
            labels = processed.get("text_labels", [""] * len(scores))
        else:
            label_ids = processed["labels"].cpu().numpy()
            labels = [queries[int(idx)] for idx in label_ids]
        
        detections = []
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
                pipeline="open_vocabulary"
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
        self.model.to(self.device).eval()
        
        self.labels = ["signature", "stamp", "seal", "document", "background"]
        self.prompts = [
            "a handwritten signature",
            "an ink stamp",
            "an official seal",
            "a document",
            "background"
        ]

    @torch.inference_mode()
    def classify_regions(self, image, regions, **kwargs):
        crops = []
        for region in regions:
            x1, y1, x2, y2 = region.bbox.to_xyxy()
            crop = image.crop((x1, y1, x2, y2))
            crops.append(crop.convert("RGB"))
        
        if not crops:
            return []
        
        inputs = self.processor(
            text=self.prompts,
            images=crops,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        
        outputs = self.model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=-1).cpu().numpy()
        
        results = []
        for i, row in enumerate(probs):
            best_idx = int(np.argmax(row))
            if self.labels[best_idx] != "background":
                results.append({
                    "region": regions[i],
                    "class": self.labels[best_idx],
                    "score": float(row[best_idx])
                })
        
        return results

    def detect(self, image, proposals=None, **kwargs):
        if proposals is None:
            proposals = self._generate_proposals(image)
        
        classified = self.classify_regions(image, proposals)
        
        detections = []
        for i, item in enumerate(classified):
            det = Detection(
                id=f"clip_{i}",
                class_name=item["class"],
                score=item["score"],
                bbox=item["region"].bbox,
                pipeline="clip"
            )
            detections.append(det)
        
        return detections

    def _generate_proposals(self, image):
        w, h = image.size
        proposals = []
        stride = max(128, min(w, h) // 8)
        size = max(256, min(w, h) // 4)
        
        for y in range(0, h - size, stride):
            for x in range(0, w - size, stride):
                bbox = BBox(x=x, y=y, w=size, h=size)
                det = Detection(
                    id=f"prop_{len(proposals)}",
                    class_name="proposal",
                    score=1.0,
                    bbox=bbox,
                    pipeline="proposal"
                )
                proposals.append(det)
        
        return proposals


class DocumentProcessor:
    def __init__(self, input_path, output_dir, pipelines_config):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.pipelines_config = pipelines_config
        self.pipelines = {}
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
                        iou=config.get("iou", 0.45)
                    )
                else:
                    logging.warning(f"YOLO not available, skipping {name}")
            
            elif pipeline_type == PipelineType.GROUNDING_DINO.value:
                if TRANSFORMERS_AVAILABLE:
                    self.pipelines[name] = OpenVocabularyDetector(
                        model_id=config["model_id"],
                        device=config.get("device"),
                        min_threshold=config.get("min_threshold", 0.15)
                    )
                else:
                    logging.warning(f"Transformers not available, skipping {name}")
            
            elif pipeline_type == PipelineType.OWL_VIT.value:
                if TRANSFORMERS_AVAILABLE:
                    self.pipelines[name] = OpenVocabularyDetector(
                        model_id=config["model_id"],
                        device=config.get("device"),
                        min_threshold=config.get("min_threshold", 0.15)
                    )
                else:
                    logging.warning(f"Transformers not available, skipping {name}")
            
            elif pipeline_type == PipelineType.CLIP.value:
                if TRANSFORMERS_AVAILABLE:
                    self.pipelines[name] = CLIPClassifier(
                        model_id=config.get("model_id", "openai/clip-vit-base-patch32"),
                        device=config.get("device")
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
        
        page_result = ProcessedPage(
            page_no=page_no,
            width=w,
            height=h,
            image_path=str(image_path)
        )
        
        for pipeline_name, pipeline in self.pipelines.items():
            logging.info(f"Running {pipeline_name} on page {page_no}")
            
            pipeline_dir = base_page_dir / pipeline_name
            pipeline_dir.mkdir(exist_ok=True)
            
            try:
                detections = pipeline.detect(image)
                page_result.detections[pipeline_name] = detections
                
                annotated = self._annotate_image(image, detections)
                annotated_path = pipeline_dir / "annotated.png"
                annotated.save(annotated_path)
                
                for i, det in enumerate(detections):
                    crop_path = pipeline_dir / f"crop_{i:04d}_{det.class_name}.png"
                    x1, y1, x2, y2 = det.bbox.to_xyxy()
                    crop = image.crop((x1, y1, x2, y2))
                    crop.save(crop_path)
                
                result_json = {
                    "page_no": page_no,
                    "pipeline": pipeline_name,
                    "detections": [
                        {
                            "id": det.id,
                            "class": det.class_name,
                            "score": det.score,
                            "bbox": det.bbox.to_dict()
                        }
                        for det in detections
                    ]
                }
                
                json_path = pipeline_dir / "results.json"
                json_path.write_text(json.dumps(result_json, indent=2))
                
            except Exception as e:
                logging.error(f"Pipeline {pipeline_name} failed: {e}")
        
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
            raise RuntimeError("No PDF rasterization library available")

    def _annotate_image(self, image, detections):
        img = image.copy()
        draw = ImageDraw.Draw(img)
        
        colors = {
            "signature": (255, 200, 0),
            "stamp": (0, 200, 255),
            "seal": (255, 0, 100)
        }
        
        for det in detections:
            color = colors.get(det.class_name.lower(), (255, 255, 255))
            x1, y1, x2, y2 = det.bbox.to_xyxy()
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            label = f"{det.class_name} {det.score:.2f}"
            draw.text((x1, y1 - 20), label, fill=color)
        
        return img

    def _save_results(self, results):
        summary = {
            "input": str(self.input_path),
            "output_dir": str(self.output_dir),
            "timestamp": datetime.now().isoformat(),
            "pipelines": list(self.pipelines.keys()),
            "pages": []
        }
        
        for page_result in results:
            page_summary = {
                "page_no": page_result.page_no,
                "size": {"width": page_result.width, "height": page_result.height},
                "detections_by_pipeline": {}
            }
            
            for pipeline_name, detections in page_result.detections.items():
                page_summary["detections_by_pipeline"][pipeline_name] = {
                    "count": len(detections),
                    "classes": list(set(d.class_name for d in detections))
                }
            
            summary["pages"].append(page_summary)
        
        summary_path = self.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        
        return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input PDF or image path")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--pipelines", nargs="+", default=["yolo"], 
                       choices=["yolo", "grounding_dino", "owl_vit", "clip", "all"],
                       help="Pipelines to run")
    parser.add_argument("--yolo_model", default="yolov8n.pt", help="YOLO model path")
    parser.add_argument("--grounding_dino_model", default="IDEA-Research/grounding-dino-base",
                       help="Grounding DINO model ID")
    parser.add_argument("--owl_vit_model", default="google/owlv2-base-patch16-ensemble",
                       help="OWL-ViT model ID")
    parser.add_argument("--clip_model", default="openai/clip-vit-base-patch32",
                       help="CLIP model ID")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    return parser.parse_args()


def main():
    args = parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    
    pipelines_to_run = args.pipelines
    if "all" in pipelines_to_run:
        pipelines_to_run = ["yolo", "grounding_dino", "owl_vit", "clip"]
    
    pipelines_config = {}
    
    if "yolo" in pipelines_to_run:
        pipelines_config["yolo"] = {
            "enabled": True,
            "type": PipelineType.YOLO.value,
            "model_path": args.yolo_model,
            "device": args.device,
            "conf": args.conf,
            "iou": args.iou
        }
    
    if "grounding_dino" in pipelines_to_run:
        pipelines_config["grounding_dino"] = {
            "enabled": True,
            "type": PipelineType.GROUNDING_DINO.value,
            "model_id": args.grounding_dino_model,
            "device": args.device,
            "min_threshold": args.conf
        }
    
    if "owl_vit" in pipelines_to_run:
        pipelines_config["owl_vit"] = {
            "enabled": True,
            "type": PipelineType.OWL_VIT.value,
            "model_id": args.owl_vit_model,
            "device": args.device,
            "min_threshold": args.conf
        }
    
    if "clip" in pipelines_to_run:
        pipelines_config["clip"] = {
            "enabled": True,
            "type": PipelineType.CLIP.value,
            "model_id": args.clip_model,
            "device": args.device
        }
    
    processor = DocumentProcessor(
        input_path=args.input,
        output_dir=Path(args.output),
        pipelines_config=pipelines_config
    )
    
    try:
        summary = processor.process()
        logging.info(f"Processing complete. Results saved to {args.output}")
        logging.info(f"Summary: {json.dumps(summary, indent=2)}")
    except Exception as e:
        logging.error(f"Processing failed: {e}")
        raise


if __name__ == "__main__":
    main()
