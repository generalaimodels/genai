import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass(frozen=True)
class Detection:
    xyxy: Tuple[float, float, float, float]
    score: float
    class_id: int
    class_name: str


class YoloRunner:
    def __init__(self, model_path: str, device: Optional[str]):
        self.model = YOLO(model_path)
        self.device = device if device else None
        self.names = self._names()

    def _names(self) -> Dict[int, str]:
        names = getattr(self.model, "names", {})
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, list):
            return {i: str(v) for i, v in enumerate(names)}
        return {}

    def predict(self, image: np.ndarray, conf: float, iou: float, imgsz: int, augment: bool, agnostic: bool, max_det: int) -> List[Detection]:
        results = self.model.predict(image, conf=conf, iou=iou, imgsz=imgsz, device=self.device, augment=augment, agnostic_nms=agnostic, verbose=False, max_det=max_det)
        out: List[Detection] = []
        for r in results:
            if not hasattr(r, "boxes") or r.boxes is None or r.boxes.xyxy is None:
                continue
            bxyxy = r.boxes.xyxy.cpu().numpy().astype(np.float32)
            confs = r.boxes.conf.cpu().numpy().astype(np.float32).reshape(-1)
            clss = r.boxes.cls.cpu().numpy().astype(np.int32).reshape(-1)
            for (x1, y1, x2, y2), sc, cid in zip(bxyxy, confs, clss):
                out.append(Detection((float(x1), float(y1), float(x2), float(y2)), float(sc), int(cid), self.names.get(int(cid), str(cid))))
        return out


def load_image(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return img


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_xyxy_array(dets: List[Detection]) -> np.ndarray:
    if not dets:
        return np.zeros((0, 4), dtype=np.float32)
    return np.array([d.xyxy for d in dets], dtype=np.float32)


def to_scores_array(dets: List[Detection]) -> np.ndarray:
    if not dets:
        return np.zeros((0,), dtype=np.float32)
    return np.array([d.score for d in dets], dtype=np.float32)


def iou_matrix(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - inter + 1e-9
    return inter / union


def diou_score(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    ious = iou_matrix(box, boxes)
    cx, cy = (box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5
    cxs, cys = (boxes[:, 0] + boxes[:, 2]) * 0.5, (boxes[:, 1] + boxes[:, 3]) * 0.5
    rho2 = (cx - cxs) ** 2 + (cy - cys) ** 2
    x1 = np.minimum(box[0], boxes[:, 0])
    y1 = np.minimum(box[1], boxes[:, 1])
    x2 = np.maximum(box[2], boxes[:, 2])
    y2 = np.maximum(box[3], boxes[:, 3])
    c2 = (x2 - x1) ** 2 + (y2 - y1) ** 2 + 1e-9
    return ious - rho2 / c2


def greedy_nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = iou_matrix(boxes[i], boxes[order[1:]])
        inds = np.where(ious <= iou_thr)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=np.int32)


def diou_nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        dious = diou_score(boxes[i], boxes[order[1:]])
        inds = np.where(dious <= iou_thr)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=np.int32)


def soft_nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float, sigma: float, score_thr: float, method: str) -> Tuple[np.ndarray, np.ndarray]:
    boxes = boxes.copy()
    scores = scores.copy()
    N = scores.shape[0]
    indices = np.arange(N)
    keep = []
    for _ in range(N):
        if scores.size == 0:
            break
        m = np.argmax(scores)
        max_box = boxes[m].copy()
        max_score = scores[m].copy()
        max_index = indices[m].copy()
        keep.append((max_index, max_box, max_score))
        if len(scores) == 1:
            break
        boxes_m = np.concatenate([boxes[:m], boxes[m + 1:]], axis=0)
        scores_m = np.concatenate([scores[:m], scores[m + 1:]], axis=0)
        indices_m = np.concatenate([indices[:m], indices[m + 1:]], axis=0)
        ious = iou_matrix(max_box, boxes_m)
        if method == "linear":
            weight = np.ones_like(ious)
            mask = ious > iou_thr
            weight[mask] = 1 - ious[mask]
        elif method == "gaussian":
            weight = np.exp(-(ious ** 2) / sigma)
        else:
            mask = ious <= iou_thr
            boxes = boxes_m[mask]
            scores = scores_m[mask]
            indices = indices_m[mask]
            continue
        scores_m = scores_m * weight
        valid = scores_m > score_thr
        boxes = boxes_m[valid]
        scores = scores_m[valid]
        indices = indices_m[valid]
    if not keep:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32)
    kept_indices = np.array([k[0] for k in keep], dtype=np.int32)
    kept_scores = np.array([k[2] for k in keep], dtype=np.float32)
    ord2 = kept_scores.argsort()[::-1]
    return kept_indices[ord2], kept_scores[ord2]


def weighted_boxes_fusion(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> Tuple[np.ndarray, np.ndarray]:
    if boxes.shape[0] == 0:
        return boxes, scores
    order = scores.argsort()[::-1]
    boxes = boxes[order]
    scores = scores[order]
    fused_boxes = []
    fused_scores = []
    used = np.zeros(len(scores), dtype=bool)
    for i in range(len(scores)):
        if used[i]:
            continue
        cluster = [i]
        for j in range(i + 1, len(scores)):
            if used[j]:
                continue
            if iou_matrix(boxes[i], boxes[j:j + 1])[0] >= iou_thr:
                cluster.append(j)
        used[cluster] = True
        w = scores[cluster]
        w_sum = np.sum(w) + 1e-12
        b = boxes[cluster]
        fused = np.sum(b * w[:, None], axis=0) / w_sum
        conf = float(np.mean(scores[cluster]))
        fused_boxes.append(fused)
        fused_scores.append(conf)
    fb = np.vstack(fused_boxes).astype(np.float32) if fused_boxes else np.zeros((0, 4), dtype=np.float32)
    fs = np.array(fused_scores, dtype=np.float32) if fused_scores else np.zeros((0,), dtype=np.float32)
    return fb, fs


def apply_suppression(dets: List[Detection], method: str, iou_thr: float, group_by: str = "class_id", mapping: Optional[Dict[str, str]] = None) -> List[Detection]:
    if not dets or method == "yolo":
        return dets
    boxes = to_xyxy_array(dets)
    scores = to_scores_array(dets)
    if group_by == "refined":
        keys = np.array([refined_label(d.class_name, mapping).lower() for d in dets])
        uniq = np.unique(keys)
        kept: List[Detection] = []
        for key in uniq:
            idx = np.where(keys == key)[0]
            b = boxes[idx]
            s = scores[idx]
            originals = [dets[i] for i in idx.tolist()]
            if method == "nms":
                keep_idx = greedy_nms(b, s, iou_thr)
                kept.extend([originals[k] for k in keep_idx.tolist()])
            elif method == "diou":
                keep_idx = diou_nms(b, s, iou_thr)
                kept.extend([originals[k] for k in keep_idx.tolist()])
            elif method == "soft_linear":
                keep_idx, new_scores = soft_nms(b, s, iou_thr=iou_thr, sigma=0.5, score_thr=1e-3, method="linear")
                for k, sc in zip(keep_idx.tolist(), new_scores.tolist()):
                    d = originals[k]
                    kept.append(Detection(d.xyxy, sc, d.class_id, d.class_name))
            elif method == "soft_gaussian":
                keep_idx, new_scores = soft_nms(b, s, iou_thr=iou_thr, sigma=0.5, score_thr=1e-3, method="gaussian")
                for k, sc in zip(keep_idx.tolist(), new_scores.tolist()):
                    d = originals[k]
                    kept.append(Detection(d.xyxy, sc, d.class_id, d.class_name))
            elif method == "wbf":
                fb, fs = weighted_boxes_fusion(b, s, iou_thr)
                for bx, sc in zip(fb.tolist(), fs.tolist()):
                    d0 = originals[0]
                    kept.append(Detection(tuple(bx), sc, d0.class_id, d0.class_name))
        kept.sort(key=lambda d: d.score, reverse=True)
        return kept
    class_ids = np.array([d.class_id for d in dets], dtype=np.int32)
    kept: List[Detection] = []
    for cid in np.unique(class_ids):
        idx = np.where(class_ids == cid)[0]
        b = boxes[idx]
        s = scores[idx]
        originals = [dets[i] for i in idx.tolist()]
        if method == "nms":
            keep_idx = greedy_nms(b, s, iou_thr)
            kept.extend([originals[k] for k in keep_idx.tolist()])
        elif method == "diou":
            keep_idx = diou_nms(b, s, iou_thr)
            kept.extend([originals[k] for k in keep_idx.tolist()])
        elif method == "soft_linear":
            keep_idx, new_scores = soft_nms(b, s, iou_thr=iou_thr, sigma=0.5, score_thr=1e-3, method="linear")
            for k, sc in zip(keep_idx.tolist(), new_scores.tolist()):
                d = originals[k]
                kept.append(Detection(d.xyxy, sc, d.class_id, d.class_name))
        elif method == "soft_gaussian":
            keep_idx, new_scores = soft_nms(b, s, iou_thr=iou_thr, sigma=0.5, score_thr=1e-3, method="gaussian")
            for k, sc in zip(keep_idx.tolist(), new_scores.tolist()):
                d = originals[k]
                kept.append(Detection(d.xyxy, sc, d.class_id, d.class_name))
        elif method == "wbf":
            fb, fs = weighted_boxes_fusion(b, s, iou_thr)
            for bx, sc in zip(fb.tolist(), fs.tolist()):
                kept.append(Detection(tuple(bx), sc, int(cid), originals[0].class_name))
    kept.sort(key=lambda d: d.score, reverse=True)
    return kept


def refined_label(name: str, mapping: Optional[Dict[str, str]]) -> str:
    n = name.strip().lower()
    aliases = {
        "signature": "signature",
        "sign": "signature",
        "autograph": "signature",
        "sig": "signature",
        "seal": "seal",
        "company seal": "seal",
        "official seal": "seal",
        "stamp": "stamp",
        "rubber stamp": "stamp",
        "round stamp": "stamp",
    }
    if mapping:
        for k, v in mapping.items():
            aliases[k.strip().lower()] = v.strip().lower()
    if n in aliases:
        return aliases[n]
    for k, v in aliases.items():
        if k in n:
            return v
    return name


def filter_by_labels(dets: List[Detection], allowed: Optional[Iterable[str]], mapping: Optional[Dict[str, str]], fuzzy: bool, retain_all_if_empty: bool) -> List[Detection]:
    if not allowed:
        return dets
    s = {a.strip().lower() for a in allowed if a.strip()}
    picked = []
    for d in dets:
        dn = d.class_name.lower()
        if dn in s:
            picked.append(d)
            continue
        if fuzzy:
            if any(a in dn for a in s):
                picked.append(d)
                continue
            rn = refined_label(d.class_name, mapping).lower()
            if rn in s or any(a in rn for a in s):
                picked.append(d)
    if retain_all_if_empty and not picked:
        return dets
    return picked


def color_for_class(name: str) -> Tuple[int, int, int]:
    palette = {
        "signature": (56, 168, 255),
        "seal": (40, 200, 120),
        "stamp": (255, 64, 64),
    }
    n = name.lower()
    if n in palette:
        return palette[n]
    h = hash(n)
    r = (h & 255)
    g = (h >> 8) & 255
    b = (h >> 16) & 255
    return (int(b), int(g), int(r))


def draw_label(img: np.ndarray, x1: int, y1: int, text: str, color: Tuple[int, int, int], thickness: int) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, min(img.shape[1], img.shape[0]) / 1000.0)
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    th2 = int(th * 1.2)
    y0 = max(0, y1 - th2 - 2)
    cv2.rectangle(img, (x1, y0), (x1 + tw + 6, y0 + th2 + 2), color, -1)
    cv2.putText(img, text, (x1 + 3, y0 + th), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def annotate_image(image: np.ndarray, dets: List[Detection], mapping: Optional[Dict[str, str]]) -> np.ndarray:
    annotated = image.copy()
    t = max(2, int(round(0.002 * max(image.shape[0], image.shape[1]))))
    for d in dets:
        x1, y1, x2, y2 = d.xyxy
        xi1, yi1, xi2, yi2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        rc = refined_label(d.class_name, mapping)
        color = color_for_class(rc)
        cv2.rectangle(annotated, (xi1, yi1), (xi2, yi2), color, t)
        label = f"{rc} {d.score:.3f}"
        draw_label(annotated, xi1, yi1, label, color, max(1, t - 1))
    return annotated


def bbox_dict(x1: float, y1: float, x2: float, y2: float) -> Dict[str, int]:
    xi1, yi1, xi2, yi2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
    w = max(0, xi2 - xi1)
    h = max(0, yi2 - yi1)
    return {"x1": xi1, "y1": yi1, "x2": xi2, "y2": yi2, "w": w, "h": h}


def hex_color(color: Tuple[int, int, int]) -> str:
    b, g, r = color
    return f"#{r:02X}{g:02X}{b:02X}"


def build_result_json(image_path: Path, annotated_path: Optional[Path], dets: List[Detection], mapping: Optional[Dict[str, str]]) -> Dict:
    items = []
    for d in dets:
        rc = refined_label(d.class_name, mapping)
        c = color_for_class(rc)
        items.append(
            {
                "class": d.class_name,
                "refined_class": rc,
                "score": float(d.score),
                "bbox": bbox_dict(*d.xyxy),
                "color": hex_color(c),
            }
        )
    payload = {
        "image": str(image_path),
        "page_annotated": str(annotated_path) if annotated_path else None,
        "detections": items,
    }
    return payload


def parse_label_map(path: Optional[str]) -> Optional[Dict[str, str]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    m = json.loads(p.read_text(encoding="utf-8"))
    return {str(k).strip().lower(): str(v).strip().lower() for k, v in m.items()}


def resolve_sources(source: str) -> List[Path]:
    p = Path(source)
    if p.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        return sorted([x for x in p.rglob("*") if x.suffix.lower() in exts])
    if p.is_file():
        return [p]
    paths = [Path(s) for s in sorted(map(str, Path().glob(source)))]
    return [x for x in paths if x.exists() and x.is_file()]


def parse_int_list(v: str) -> List[int]:
    if not v:
        return [1280]
    vals = []
    for t in v.split(","):
        t = t.strip()
        if not t:
            continue
        try:
            vals.append(int(t))
        except:
            pass
    vals = [x for x in vals if x > 0]
    if not vals:
        vals = [1280]
    return sorted(list(dict.fromkeys(vals)))


def aggregate_passes(passes: List[List[Detection]], method: str, iou_thr: float, mapping: Optional[Dict[str, str]], group_by_refined: bool) -> List[Detection]:
    merged = [d for pl in passes for d in pl]
    if not merged:
        return []
    group_key = "refined" if group_by_refined else "class_id"
    fused = apply_suppression(merged, method, iou_thr=iou_thr, group_by=group_key, mapping=mapping)
    return fused


def process_image(runner: YoloRunner, image_path: Path, out_dir: Path, allowed_labels: Optional[List[str]], mapping: Optional[Dict[str, str]], nms_method: str, iou_suppress: float, imgsz_list: List[int], conf_seq: List[float], use_tta: bool, save_image: bool, save_json: bool, fuzzy_filter: bool, retain_all_if_empty: bool, group_by_refined: bool, agnostic_nms: bool, max_det: int) -> None:
    image = load_image(image_path)
    passes: List[List[Detection]] = []
    for sz in imgsz_list:
        for cf in conf_seq:
            dets = runner.predict(image, conf=cf, iou=iou_suppress if nms_method == "yolo" else 0.7, imgsz=sz, augment=use_tta, agnostic=agnostic_nms, max_det=max_det)
            passes.append(dets)
    dets = aggregate_passes(passes, "wbf", iou_thr=0.55, mapping=mapping, group_by_refined=group_by_refined)
    dets = apply_suppression(dets, nms_method, iou_thr=iou_suppress, group_by="refined" if group_by_refined else "class_id", mapping=mapping)
    filtered = filter_by_labels(dets, allowed_labels, mapping, fuzzy=fuzzy_filter, retain_all_if_empty=retain_all_if_empty)
    dets_final = filtered if filtered else dets
    annotated_path = None
    if save_image:
        annotated = annotate_image(image, dets_final, mapping)
        ensure_dir(out_dir)
        annotated_path = out_dir / f"{image_path.stem}_annotated.png"
        _, buf = cv2.imencode(".png", annotated, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        annotated_path.write_bytes(buf)
    if save_json:
        result = build_result_json(image_path, annotated_path, dets_final, mapping)
        ensure_dir(out_dir)
        json_path = out_dir / f"{image_path.stem}.json"
        json_path.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")))


def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="runs/page_annotations")
    parser.add_argument("--allowed_labels", type=str, default="")
    parser.add_argument("--label_map", type=str, default=None)
    parser.add_argument("--nms_method", type=str, default="yolo", choices=["yolo", "nms", "diou", "soft_linear", "soft_gaussian", "wbf"])
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--min_conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz_list", type=str, default="1024,1280,1536")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--save_image", action="store_true")
    parser.add_argument("--save_json", action="store_true")
    parser.add_argument("--fuzzy_filter", action="store_true")
    parser.add_argument("--no_retain_if_empty", action="store_true")
    parser.add_argument("--group_by_refined", action="store_true")
    parser.add_argument("--agnostic_nms", action="store_true")
    parser.add_argument("--max_det", type=int, default=300)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    mapping = parse_label_map(args.label_map)
    allowed = [a.strip() for a in args.allowed_labels.split(",") if a.strip()] if args.allowed_labels else None
    imgsz_list = parse_int_list(args.imgsz_list)
    conf_seq = sorted(list(dict.fromkeys([max(args.min_conf, args.conf), max(args.min_conf, args.conf * 0.75), args.min_conf])), reverse=True)
    runner = YoloRunner(args.model, args.device if args.device else None)
    sources = resolve_sources(args.source)
    for p in sources:
        process_image(
            runner=runner,
            image_path=p,
            out_dir=out_dir,
            allowed_labels=allowed,
            mapping=mapping,
            nms_method=args.nms_method,
            iou_suppress=args.iou,
            imgsz_list=imgsz_list,
            conf_seq=conf_seq,
            use_tta=args.tta or True,
            save_image=args.save_image or True,
            save_json=args.save_json or True,
            fuzzy_filter=args.fuzzy_filter or True,
            retain_all_if_empty=not args.no_retain_if_empty or True,
            group_by_refined=args.group_by_refined or True,
            agnostic_nms=args.agnostic_nms or True,
            max_det=max(50, args.max_det),
        )


if __name__ == "__main__":
    main()