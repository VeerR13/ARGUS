"""
eval_real.py — Real-validation pipeline for YOLOv8 + DeepSORT vehicle detection.

Matches CVAT YOLO 1.1 annotations against live model inference and produces
a suite of JSON reports, a histogram PNG, and a multi-page PDF summary.

Usage
-----
    python eval_real.py \\
        --annotations annotation_tasks/cvat_export/ \\
        --footage unseen_test_videos/ \\
        --model models/week2_retrained.pt \\
        --output reports_validated/

If --annotations is empty or missing the script prints a waiting message,
writes a placeholder report, and exits 0.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

DEFAULT_CLASS_NAMES: dict[int, str] = {
    0: "car",
    1: "motorcycle",
    2: "bus",
    3: "truck",
}

IOU_THRESHOLD = 0.5
FP_RATE_FLAG_THRESHOLD = 0.05      # 5 %
LAG_OFFSET_FLAG_THRESHOLD = 15.0   # pixels
RECALL_FLAG_THRESHOLD = 0.85
PRECISION_FLAG_THRESHOLD = 0.85

SCORE_WEIGHTS = {
    "mean_iou": 0.25,
    "phantom_fp_rate": 0.25,
    "recall": 0.20,
    "precision": 0.15,
    "bbox_lag": 0.15,
}

SCENARIO_KEYWORDS: list[tuple[str, str]] = [
    ("night", "night"),
    ("highway", "highway"),
    ("accident", "accident"),
    ("clip", "accident"),
]


# ---------------------------------------------------------------------------
# Scenario helper
# ---------------------------------------------------------------------------

def scenario_from_name(name: str) -> str:
    """Return scenario label derived from a filename stem."""
    lower = name.lower()
    for keyword, label in SCENARIO_KEYWORDS:
        if keyword in lower:
            return label
    return "dashcam"


# ---------------------------------------------------------------------------
# Annotation loading
# ---------------------------------------------------------------------------

def _parse_yolo_line(line: str) -> tuple[int, float, float, float, float] | None:
    """Parse a single YOLO annotation line.

    Returns (class_id, cx, cy, w, h) or None if the line is malformed.
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        class_id = int(parts[0])
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        return class_id, cx, cy, w, h
    except ValueError:
        return None


def load_annotations(annotation_dir: Path) -> dict[str, list[tuple[int, float, float, float, float]]]:
    """Load all YOLO .txt annotation files from *annotation_dir*.

    Returns a mapping of stem → list of (class_id, cx, cy, w, h).
    Skips obj.data, obj.names, classes.txt, and any file without digit content.
    """
    skip_names = {"obj.data", "obj.names", "classes.txt", "data.yaml"}
    annotations: dict[str, list[tuple[int, float, float, float, float]]] = {}

    for txt_path in sorted(annotation_dir.rglob("*.txt")):
        if txt_path.name in skip_names:
            continue
        boxes: list[tuple[int, float, float, float, float]] = []
        try:
            text = txt_path.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            parsed = _parse_yolo_line(line)
            if parsed is not None:
                boxes.append(parsed)
        # Use the stem as the key (matches image filename without extension)
        annotations[txt_path.stem] = boxes

    return annotations


def load_class_names(annotation_dir: Path) -> dict[int, str]:
    """Attempt to read class names from obj.names or data.yaml.

    Falls back to DEFAULT_CLASS_NAMES when neither file is present.
    """
    # Try obj.names
    names_path = annotation_dir / "obj.names"
    if names_path.exists():
        lines = [l.strip() for l in names_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        return {i: name for i, name in enumerate(lines)}

    # Try data.yaml (ultralytics format)
    yaml_path = annotation_dir / "data.yaml"
    if yaml_path.exists():
        try:
            import yaml  # type: ignore[import]
            with yaml_path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            if isinstance(data, dict) and "names" in data:
                raw = data["names"]
                if isinstance(raw, list):
                    return {i: name for i, name in enumerate(raw)}
                if isinstance(raw, dict):
                    return {int(k): v for k, v in raw.items()}
        except Exception:
            pass

    return DEFAULT_CLASS_NAMES


# ---------------------------------------------------------------------------
# IoU utilities
# ---------------------------------------------------------------------------

def _yolo_to_xyxy(
    cx: float, cy: float, w: float, h: float,
    img_w: int = FRAME_WIDTH, img_h: int = FRAME_HEIGHT,
) -> tuple[float, float, float, float]:
    """Convert normalised YOLO box to pixel-space xyxy."""
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return x1, y1, x2, y2


def _iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """Compute IoU between two xyxy boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def _centroid_offset(
    box_pred: tuple[float, float, float, float],
    box_gt: tuple[float, float, float, float],
) -> float:
    """Euclidean distance between centroids of two xyxy boxes (pixels)."""
    pred_cx = (box_pred[0] + box_pred[2]) / 2.0
    pred_cy = (box_pred[1] + box_pred[3]) / 2.0
    gt_cx = (box_gt[0] + box_gt[2]) / 2.0
    gt_cy = (box_gt[1] + box_gt[3]) / 2.0
    return float(np.hypot(pred_cx - gt_cx, pred_cy - gt_cy))


# ---------------------------------------------------------------------------
# Greedy matching
# ---------------------------------------------------------------------------

def greedy_match(
    pred_boxes: list[tuple[float, float, float, float]],
    gt_boxes: list[tuple[float, float, float, float]],
    threshold: float = IOU_THRESHOLD,
) -> list[tuple[int, int, float]]:
    """Greedy IoU matching: highest-IoU pairs first.

    Returns list of (pred_idx, gt_idx, iou_value) for matched pairs.
    Unmatched predictions and GTs are omitted from this list.
    """
    if not pred_boxes or not gt_boxes:
        return []

    # Build IoU matrix
    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float64)
    for pi, pb in enumerate(pred_boxes):
        for gi, gb in enumerate(gt_boxes):
            iou_matrix[pi, gi] = _iou(pb, gb)

    matched: list[tuple[int, int, float]] = []
    used_pred: set[int] = set()
    used_gt: set[int] = set()

    # Sort all pairs by descending IoU
    pairs = sorted(
        [(iou_matrix[pi, gi], pi, gi) for pi in range(len(pred_boxes)) for gi in range(len(gt_boxes))],
        reverse=True,
    )

    for iou_val, pi, gi in pairs:
        if iou_val < threshold:
            break
        if pi in used_pred or gi in used_gt:
            continue
        matched.append((pi, gi, float(iou_val)))
        used_pred.add(pi)
        used_gt.add(gi)

    return matched


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    model_path: str,
    footage_dir: Path,
    annotation_stems: set[str],
) -> dict[str, list[tuple[int, float, float, float, float]]]:
    """Run YOLO inference on .jpg frames that have a matching annotation stem.

    Returns mapping of stem → list of (class_id, cx_norm, cy_norm, w_norm, h_norm).
    """
    try:
        from ultralytics import YOLO  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is required for inference. "
            "Install with: pip install ultralytics"
        ) from exc

    model = YOLO(model_path)
    predictions: dict[str, list[tuple[int, float, float, float, float]]] = {}

    jpg_files = sorted(footage_dir.rglob("*.jpg"))
    if not jpg_files:
        print(f"  [inference] No .jpg files found in {footage_dir}")
        return predictions

    matched_files = [f for f in jpg_files if f.stem in annotation_stems]
    if not matched_files:
        print(
            f"  [inference] None of the {len(jpg_files)} .jpg files match "
            f"annotation stems. Running inference on all frames."
        )
        matched_files = jpg_files

    print(f"  [inference] Running on {len(matched_files)} frame(s) ...")

    for jpg_path in matched_files:
        import cv2  # type: ignore[import]
        frame = cv2.imread(str(jpg_path))
        if frame is None:
            print(f"  [inference] WARNING: could not read {jpg_path}")
            continue

        img_h, img_w = frame.shape[:2]

        results = model(frame, conf=0.4, iou=0.35, verbose=False)[0]
        boxes_raw = results.boxes

        frame_preds: list[tuple[int, float, float, float, float]] = []
        if boxes_raw is not None and len(boxes_raw) > 0:
            for box in boxes_raw:
                cls_id = int(box.cls[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bw = x2 - x1
                bh = y2 - y1
                cx_n = (x1 + bw / 2) / img_w
                cy_n = (y1 + bh / 2) / img_h
                w_n = bw / img_w
                h_n = bh / img_h
                frame_preds.append((cls_id, cx_n, cy_n, w_n, h_n))

        predictions[jpg_path.stem] = frame_preds

    return predictions


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

class EvaluationResults:
    """Container for all per-frame evaluation data accumulated during matching."""

    def __init__(self, class_names: dict[int, str]) -> None:
        self.class_names = class_names

        # IoU
        self.all_iou_values: list[float] = []
        self.per_class_iou: dict[int, list[float]] = defaultdict(list)
        self.per_scenario_iou: dict[str, list[float]] = defaultdict(list)

        # FP / FN
        self.per_frame_fp: dict[str, int] = {}
        self.per_frame_pred_count: dict[str, int] = {}
        self.per_scenario_fp: dict[str, list[int]] = defaultdict(list)
        self.per_scenario_pred_count: dict[str, list[int]] = defaultdict(list)

        # Bbox lag (centroid offset)
        self.all_offsets: list[float] = []
        self.per_frame_offsets: dict[str, list[float]] = defaultdict(list)

        # Recall / Precision accumulators per class
        self.per_class_tp: dict[int, int] = defaultdict(int)
        self.per_class_fp: dict[int, int] = defaultdict(int)
        self.per_class_fn: dict[int, int] = defaultdict(int)

        # Global TP / FP / FN
        self.total_tp: int = 0
        self.total_fp: int = 0
        self.total_fn: int = 0


def evaluate_frame(
    stem: str,
    gt_entries: list[tuple[int, float, float, float, float]],
    pred_entries: list[tuple[int, float, float, float, float]],
    results: EvaluationResults,
    img_w: int = FRAME_WIDTH,
    img_h: int = FRAME_HEIGHT,
) -> None:
    """Accumulate per-frame metrics into *results*."""
    scenario = scenario_from_name(stem)

    gt_boxes = [_yolo_to_xyxy(cx, cy, w, h, img_w, img_h) for _, cx, cy, w, h in gt_entries]
    pred_boxes = [_yolo_to_xyxy(cx, cy, w, h, img_w, img_h) for _, cx, cy, w, h in pred_entries]

    gt_classes = [e[0] for e in gt_entries]
    pred_classes = [e[0] for e in pred_entries]

    matches = greedy_match(pred_boxes, gt_boxes, IOU_THRESHOLD)

    matched_pred_indices: set[int] = set()
    matched_gt_indices: set[int] = set()

    for pi, gi, iou_val in matches:
        matched_pred_indices.add(pi)
        matched_gt_indices.add(gi)

        results.all_iou_values.append(iou_val)
        gt_cls = gt_classes[gi]
        results.per_class_iou[gt_cls].append(iou_val)
        results.per_scenario_iou[scenario].append(iou_val)

        offset = _centroid_offset(pred_boxes[pi], gt_boxes[gi])
        results.all_offsets.append(offset)
        results.per_frame_offsets[stem].append(offset)

        # TP attributed to the GT class
        results.per_class_tp[gt_cls] += 1
        results.total_tp += 1

    # FP = predicted boxes not matched to any GT
    fp_count = len(pred_boxes) - len(matched_pred_indices)
    results.per_frame_fp[stem] = fp_count
    results.per_frame_pred_count[stem] = len(pred_boxes)
    results.per_scenario_fp[scenario].append(fp_count)
    results.per_scenario_pred_count[scenario].append(len(pred_boxes))
    results.total_fp += fp_count

    for pi in range(len(pred_boxes)):
        if pi not in matched_pred_indices:
            cls = pred_classes[pi]
            results.per_class_fp[cls] += 1

    # FN = GT boxes not matched to any prediction
    fn_count = len(gt_boxes) - len(matched_gt_indices)
    results.total_fn += fn_count
    for gi in range(len(gt_boxes)):
        if gi not in matched_gt_indices:
            cls = gt_classes[gi]
            results.per_class_fn[cls] += 1


# ---------------------------------------------------------------------------
# Report builders
# ---------------------------------------------------------------------------

def build_iou_report(results: EvaluationResults) -> dict[str, Any]:
    """Compile the IoU report dictionary."""
    overall_mean = float(np.mean(results.all_iou_values)) if results.all_iou_values else 0.0

    per_class: dict[str, dict[str, Any]] = {}
    all_class_ids = set(results.per_class_iou.keys())
    for cls_id in sorted(all_class_ids):
        vals = results.per_class_iou[cls_id]
        name = results.class_names.get(cls_id, str(cls_id))
        per_class[name] = {
            "class_id": cls_id,
            "mean_iou": float(np.mean(vals)) if vals else 0.0,
            "median_iou": float(np.median(vals)) if vals else 0.0,
            "count": len(vals),
        }

    per_scenario: dict[str, dict[str, Any]] = {}
    for scenario, vals in results.per_scenario_iou.items():
        per_scenario[scenario] = {
            "mean_iou": float(np.mean(vals)) if vals else 0.0,
            "count": len(vals),
        }

    return {
        "overall_mean_iou": overall_mean,
        "per_class": per_class,
        "per_scenario": per_scenario,
        "total_matched_pairs": len(results.all_iou_values),
    }


def build_fp_report(results: EvaluationResults) -> dict[str, Any]:
    """Compile the false-positive / phantom detection report."""
    total_preds = sum(results.per_frame_pred_count.values())
    total_fp = sum(results.per_frame_fp.values())
    overall_fp_rate = total_fp / max(total_preds, 1)

    per_frame: dict[str, dict[str, Any]] = {}
    for stem in results.per_frame_fp:
        fp = results.per_frame_fp[stem]
        n_pred = results.per_frame_pred_count[stem]
        per_frame[stem] = {
            "fp_count": fp,
            "pred_count": n_pred,
            "fp_rate": fp / max(n_pred, 1),
        }

    per_scenario: dict[str, dict[str, Any]] = {}
    flagged_scenarios: list[str] = []
    for scenario in results.per_scenario_fp:
        fp_list = results.per_scenario_fp[scenario]
        pred_list = results.per_scenario_pred_count[scenario]
        total_fp_s = sum(fp_list)
        total_pred_s = sum(pred_list)
        rate = total_fp_s / max(total_pred_s, 1)
        per_scenario[scenario] = {
            "fp_rate": rate,
            "total_fp": total_fp_s,
            "total_predictions": total_pred_s,
            "flagged": rate > FP_RATE_FLAG_THRESHOLD,
        }
        if rate > FP_RATE_FLAG_THRESHOLD:
            flagged_scenarios.append(scenario)

    return {
        "overall_fp_rate": overall_fp_rate,
        "total_fp": total_fp,
        "total_predictions": total_preds,
        "per_scenario": per_scenario,
        "flagged_scenarios": flagged_scenarios,
        "per_frame": per_frame,
    }


def build_lag_report(results: EvaluationResults) -> dict[str, Any]:
    """Compile the bounding-box centroid lag report."""
    all_off = results.all_offsets
    mean_off = float(np.mean(all_off)) if all_off else 0.0
    max_off = float(np.max(all_off)) if all_off else 0.0
    p95_off = float(np.percentile(all_off, 95)) if all_off else 0.0

    flagged_frames: list[str] = []
    per_frame: dict[str, dict[str, Any]] = {}
    for stem, offsets in results.per_frame_offsets.items():
        frame_mean = float(np.mean(offsets)) if offsets else 0.0
        flagged = frame_mean > LAG_OFFSET_FLAG_THRESHOLD
        per_frame[stem] = {
            "mean_offset_px": frame_mean,
            "max_offset_px": float(np.max(offsets)) if offsets else 0.0,
            "flagged": flagged,
        }
        if flagged:
            flagged_frames.append(stem)

    return {
        "mean_offset_px": mean_off,
        "max_offset_px": max_off,
        "p95_offset_px": p95_off,
        "total_matched_pairs": len(all_off),
        "flagged_frames": flagged_frames,
        "per_frame": per_frame,
    }


def build_recall_report(results: EvaluationResults) -> dict[str, Any]:
    """Compile the recall report."""
    total_gt = results.total_tp + results.total_fn
    overall_recall = results.total_tp / max(total_gt, 1)

    all_cls = set(results.per_class_tp) | set(results.per_class_fn)
    per_class: dict[str, dict[str, Any]] = {}
    flagged_classes: list[str] = []

    for cls_id in sorted(all_cls):
        tp = results.per_class_tp.get(cls_id, 0)
        fn = results.per_class_fn.get(cls_id, 0)
        gt_count = tp + fn
        recall = tp / max(gt_count, 1)
        name = results.class_names.get(cls_id, str(cls_id))
        flagged = recall < RECALL_FLAG_THRESHOLD
        per_class[name] = {
            "class_id": cls_id,
            "tp": tp,
            "fn": fn,
            "gt_count": gt_count,
            "recall": recall,
            "flagged": flagged,
        }
        if flagged:
            flagged_classes.append(name)

    return {
        "overall_recall": overall_recall,
        "total_tp": results.total_tp,
        "total_fn": results.total_fn,
        "per_class": per_class,
        "flagged_classes": flagged_classes,
    }


def build_precision_report(results: EvaluationResults) -> dict[str, Any]:
    """Compile the precision report."""
    total_pred = results.total_tp + results.total_fp
    overall_precision = results.total_tp / max(total_pred, 1)

    all_cls = set(results.per_class_tp) | set(results.per_class_fp)
    per_class: dict[str, dict[str, Any]] = {}
    flagged_classes: list[str] = []

    for cls_id in sorted(all_cls):
        tp = results.per_class_tp.get(cls_id, 0)
        fp = results.per_class_fp.get(cls_id, 0)
        pred_count = tp + fp
        precision = tp / max(pred_count, 1)
        name = results.class_names.get(cls_id, str(cls_id))
        flagged = precision < PRECISION_FLAG_THRESHOLD
        per_class[name] = {
            "class_id": cls_id,
            "tp": tp,
            "fp": fp,
            "pred_count": pred_count,
            "precision": precision,
            "flagged": flagged,
        }
        if flagged:
            flagged_classes.append(name)

    return {
        "overall_precision": overall_precision,
        "total_tp": results.total_tp,
        "total_fp": results.total_fp,
        "per_class": per_class,
        "flagged_classes": flagged_classes,
    }


# ---------------------------------------------------------------------------
# Incident detection accuracy
# ---------------------------------------------------------------------------

def build_incident_report(
    footage_dir: Path,
    predictions: dict[str, list[tuple[int, float, float, float, float]]],
) -> dict[str, Any]:
    """Evaluate per-video incident detection.

    A video whose name contains "accident" is expected to yield detections.
    All other videos are expected to yield 0 detections.
    """
    # Group stems by video name prefix (best-effort: split on '_frame' or use stem directly)
    # For flat .jpg frames named like videoname_frame0042.jpg we group by removing the frame suffix.
    # Also handle the case where the stem IS the video name (single image per video).

    video_detection_count: dict[str, int] = defaultdict(int)
    video_frame_count: dict[str, int] = defaultdict(int)

    jpg_files = sorted(footage_dir.rglob("*.jpg"))
    for jpg_path in jpg_files:
        stem = jpg_path.stem
        # Derive a video-level identifier by stripping trailing _frameNNNN or _NNNN
        import re
        video_id = re.sub(r"[_\-]frame\d+$", "", stem, flags=re.IGNORECASE)
        video_id = re.sub(r"[_\-]\d+$", "", video_id)
        if not video_id:
            video_id = stem

        video_frame_count[video_id] += 1
        det_count = len(predictions.get(stem, []))
        video_detection_count[video_id] += det_count

    per_video: dict[str, dict[str, Any]] = {}
    accident_videos: list[str] = []
    non_accident_videos: list[str] = []

    for vid_id in sorted(video_detection_count.keys()):
        is_accident = "accident" in vid_id.lower()
        detected = video_detection_count[vid_id] > 0
        per_video[vid_id] = {
            "expected_detections": is_accident,
            "detected": detected,
            "frame_count": video_frame_count[vid_id],
            "detection_count": video_detection_count[vid_id],
            "correct": detected == is_accident,
        }
        if is_accident:
            accident_videos.append(vid_id)
        else:
            non_accident_videos.append(vid_id)

    missed_accidents = [v for v in accident_videos if not per_video[v]["detected"]]
    false_alarms = [v for v in non_accident_videos if per_video[v]["detected"]]

    miss_rate = len(missed_accidents) / max(len(accident_videos), 1)
    false_alarm_rate = len(false_alarms) / max(len(non_accident_videos), 1)

    return {
        "miss_rate": miss_rate,
        "false_alarm_rate": false_alarm_rate,
        "accident_videos": accident_videos,
        "non_accident_videos": non_accident_videos,
        "missed_accidents": missed_accidents,
        "false_alarms": false_alarms,
        "per_video": per_video,
    }


# ---------------------------------------------------------------------------
# Aggregate score
# ---------------------------------------------------------------------------

def compute_aggregate_score(
    iou_report: dict[str, Any],
    fp_report: dict[str, Any],
    recall_report: dict[str, Any],
    precision_report: dict[str, Any],
    lag_report: dict[str, Any],
) -> dict[str, Any]:
    """Compute a 0–100 weighted aggregate score.

    Scoring sub-components
    ----------------------
    mean_iou       : raw iou * 100
    phantom_fp_rate: (1 - fp_rate) * 100
    recall         : overall_recall * 100
    precision      : overall_precision * 100
    bbox_lag       : score decays linearly from 100 at 0 px to 0 at 50 px
    """
    iou_score = iou_report["overall_mean_iou"] * 100.0
    fp_score = (1.0 - fp_report["overall_fp_rate"]) * 100.0
    recall_score = recall_report["overall_recall"] * 100.0
    precision_score = precision_report["overall_precision"] * 100.0

    mean_lag = lag_report["mean_offset_px"]
    lag_score = max(0.0, 100.0 - (mean_lag / 50.0) * 100.0)

    components = {
        "mean_iou": iou_score,
        "phantom_fp_rate": fp_score,
        "recall": recall_score,
        "precision": precision_score,
        "bbox_lag": lag_score,
    }

    aggregate = sum(components[k] * SCORE_WEIGHTS[k] for k in SCORE_WEIGHTS)

    if aggregate >= 85:
        grade = "A"
    elif aggregate >= 70:
        grade = "B"
    elif aggregate >= 55:
        grade = "C"
    else:
        grade = "D"

    return {
        "aggregate_score": round(aggregate, 2),
        "grade": grade,
        "components": {k: round(v, 2) for k, v in components.items()},
        "weights": SCORE_WEIGHTS,
    }


# ---------------------------------------------------------------------------
# Plot: IoU histogram
# ---------------------------------------------------------------------------

def save_iou_histogram(iou_values: list[float], output_path: Path) -> None:
    """Save an IoU distribution histogram as a PNG."""
    fig, ax = plt.subplots(figsize=(8, 5))

    if iou_values:
        ax.hist(iou_values, bins=20, range=(0.0, 1.0), color="#2196F3", edgecolor="white", linewidth=0.5)
        ax.axvline(float(np.mean(iou_values)), color="#F44336", linestyle="--", linewidth=1.5,
                   label=f"Mean IoU = {np.mean(iou_values):.3f}")
        ax.axvline(IOU_THRESHOLD, color="#FF9800", linestyle=":", linewidth=1.5,
                   label=f"Threshold = {IOU_THRESHOLD}")
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "No matched pairs", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="grey")

    ax.set_xlabel("IoU", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("IoU Distribution — Matched Prediction / GT Pairs", fontsize=12, fontweight="bold")
    ax.set_xlim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# PDF report
# ---------------------------------------------------------------------------

def _cover_page(pdf: PdfPages, score_data: dict[str, Any]) -> None:
    """Page 1: cover + aggregate score."""
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("#1A237E")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_facecolor("#1A237E")

    ax.text(0.5, 0.88, "ARGUS", ha="center", va="center",
            fontsize=28, fontweight="bold", color="white", transform=ax.transAxes)
    ax.text(0.5, 0.80, "Real-World Validation Report", ha="center", va="center",
            fontsize=18, color="#90CAF9", transform=ax.transAxes)

    score = score_data["aggregate_score"]
    grade = score_data["grade"]
    grade_colors = {"A": "#4CAF50", "B": "#8BC34A", "C": "#FFC107", "D": "#F44336"}
    grade_color = grade_colors.get(grade, "#9E9E9E")

    ax.text(0.5, 0.60, f"{score:.1f} / 100", ha="center", va="center",
            fontsize=52, fontweight="bold", color=grade_color, transform=ax.transAxes)
    ax.text(0.5, 0.50, f"Grade: {grade}", ha="center", va="center",
            fontsize=32, fontweight="bold", color=grade_color, transform=ax.transAxes)

    # Component breakdown
    components = score_data["components"]
    weights = score_data["weights"]
    y_start = 0.40
    labels = {
        "mean_iou": "Mean IoU",
        "phantom_fp_rate": "FP Rate (inverted)",
        "recall": "Recall",
        "precision": "Precision",
        "bbox_lag": "BBox Lag (inverted)",
    }
    for i, (key, label) in enumerate(labels.items()):
        y = y_start - i * 0.054
        w = weights[key]
        raw = components[key]
        contribution = raw * w
        ax.text(0.18, y, label, ha="left", va="center", fontsize=9,
                color="#B3E5FC", transform=ax.transAxes)
        ax.text(0.56, y, f"{raw:.1f}", ha="right", va="center", fontsize=9,
                color="white", transform=ax.transAxes)
        ax.text(0.70, y, f"x {w:.0%}", ha="right", va="center", fontsize=9,
                color="#78909C", transform=ax.transAxes)
        ax.text(0.85, y, f"= {contribution:.1f}", ha="right", va="center", fontsize=9,
                color="#FFE082", transform=ax.transAxes)

    from datetime import datetime
    ax.text(0.5, 0.06, f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}", ha="center",
            va="center", fontsize=8, color="#546E7A", transform=ax.transAxes)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _iou_histogram_page(pdf: PdfPages, histogram_path: Path) -> None:
    """Page 2: embed the IoU histogram PNG."""
    fig = plt.figure(figsize=(8.5, 11))
    ax_title = fig.add_axes([0.05, 0.90, 0.90, 0.08])
    ax_title.set_axis_off()
    ax_title.text(0.5, 0.5, "IoU Distribution", ha="center", va="center",
                  fontsize=16, fontweight="bold")

    if histogram_path.exists():
        img = plt.imread(str(histogram_path))
        ax_img = fig.add_axes([0.05, 0.35, 0.90, 0.52])
        ax_img.imshow(img)
        ax_img.set_axis_off()

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _precision_recall_table_page(
    pdf: PdfPages,
    recall_report: dict[str, Any],
    precision_report: dict[str, Any],
) -> None:
    """Page 3: per-class precision / recall table."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.set_axis_off()
    ax.set_title("Per-Class Precision & Recall", fontsize=14, fontweight="bold", pad=20)

    all_classes = sorted(
        set(recall_report["per_class"].keys()) | set(precision_report["per_class"].keys())
    )

    col_labels = ["Class", "Precision", "Recall", "TP", "FP", "FN", "P Flag", "R Flag"]
    rows: list[list[str]] = []
    cell_colors: list[list[str]] = []

    for cls_name in all_classes:
        prec_data = precision_report["per_class"].get(cls_name, {})
        rec_data = recall_report["per_class"].get(cls_name, {})

        precision = prec_data.get("precision", 0.0)
        recall = rec_data.get("recall", 0.0)
        tp = prec_data.get("tp", rec_data.get("tp", 0))
        fp = prec_data.get("fp", 0)
        fn = rec_data.get("fn", 0)
        p_flagged = prec_data.get("flagged", False)
        r_flagged = rec_data.get("flagged", False)

        row = [
            cls_name,
            f"{precision:.3f}",
            f"{recall:.3f}",
            str(tp),
            str(fp),
            str(fn),
            "YES" if p_flagged else "no",
            "YES" if r_flagged else "no",
        ]
        rows.append(row)

        flag_red = "#FFCDD2"
        flag_green = "#C8E6C9"
        p_color = flag_red if p_flagged else flag_green
        r_color = flag_red if r_flagged else flag_green
        cell_colors.append(["white", "white", "white", "white", "white", "white", p_color, r_color])

    if not rows:
        rows = [["(no data)", "-", "-", "-", "-", "-", "-", "-"]]
        cell_colors = [["white"] * 8]

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _scenario_summary_page(
    pdf: PdfPages,
    iou_report: dict[str, Any],
    fp_report: dict[str, Any],
) -> None:
    """Page 4: per-scenario summary table."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.set_axis_off()
    ax.set_title("Per-Scenario Summary", fontsize=14, fontweight="bold", pad=20)

    all_scenarios = sorted(
        set(iou_report["per_scenario"].keys()) | set(fp_report["per_scenario"].keys())
    )

    col_labels = ["Scenario", "Mean IoU", "IoU Count", "FP Rate", "FP Count", "FP Flagged"]
    rows: list[list[str]] = []
    cell_colors: list[list[str]] = []

    for sc in all_scenarios:
        iou_data = iou_report["per_scenario"].get(sc, {})
        fp_data = fp_report["per_scenario"].get(sc, {})

        mean_iou = iou_data.get("mean_iou", 0.0)
        iou_count = iou_data.get("count", 0)
        fp_rate = fp_data.get("fp_rate", 0.0)
        fp_count = fp_data.get("total_fp", 0)
        fp_flagged = fp_data.get("flagged", False)

        row = [
            sc,
            f"{mean_iou:.3f}",
            str(iou_count),
            f"{fp_rate:.3f}",
            str(fp_count),
            "YES" if fp_flagged else "no",
        ]
        rows.append(row)
        flag_color = "#FFCDD2" if fp_flagged else "#C8E6C9"
        cell_colors.append(["white", "white", "white", "white", "white", flag_color])

    if not rows:
        rows = [["(no data)", "-", "-", "-", "-", "-"]]
        cell_colors = [["white"] * 6]

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 2.0)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _incident_page(pdf: PdfPages, incident_report: dict[str, Any]) -> None:
    """Page 5: incident detection accuracy."""
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 11), gridspec_kw={"height_ratios": [1, 3]})

    # Summary metrics
    ax_summary = axes[0]
    ax_summary.set_axis_off()
    ax_summary.set_title("Incident Detection Accuracy", fontsize=14, fontweight="bold")

    miss = incident_report["miss_rate"]
    fa = incident_report["false_alarm_rate"]
    summary_text = (
        f"Miss Rate (accident videos with 0 detections): {miss:.1%}\n"
        f"False Alarm Rate (non-accident with detections): {fa:.1%}\n"
        f"Missed accidents: {incident_report['missed_accidents']}\n"
        f"False alarms:     {incident_report['false_alarms']}"
    )
    ax_summary.text(0.05, 0.6, summary_text, ha="left", va="center",
                    transform=ax_summary.transAxes, fontsize=10,
                    fontfamily="monospace",
                    bbox={"boxstyle": "round", "facecolor": "#E3F2FD", "alpha": 0.8})

    # Per-video table
    ax_table = axes[1]
    ax_table.set_axis_off()

    per_video = incident_report["per_video"]
    col_labels = ["Video", "Expected", "Detected", "Frames", "Detections", "Correct"]
    rows: list[list[str]] = []
    cell_colors: list[list[str]] = []

    for vid_id, vdata in sorted(per_video.items()):
        row = [
            vid_id[:30],
            "YES" if vdata["expected_detections"] else "no",
            "YES" if vdata["detected"] else "no",
            str(vdata["frame_count"]),
            str(vdata["detection_count"]),
            "YES" if vdata["correct"] else "NO",
        ]
        correct_color = "#C8E6C9" if vdata["correct"] else "#FFCDD2"
        rows.append(row)
        cell_colors.append(["white", "white", "white", "white", "white", correct_color])

    if not rows:
        rows = [["(no footage)", "-", "-", "-", "-", "-"]]
        cell_colors = [["white"] * 6]

    table = ax_table.table(
        cellText=rows,
        colLabels=col_labels,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.8)

    fig.tight_layout(pad=2.0)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def save_pdf_report(
    output_dir: Path,
    score_data: dict[str, Any],
    iou_report: dict[str, Any],
    fp_report: dict[str, Any],
    recall_report: dict[str, Any],
    precision_report: dict[str, Any],
    lag_report: dict[str, Any],
    incident_report: dict[str, Any],
    histogram_path: Path,
) -> Path:
    """Write the 5-page PDF evaluation report."""
    pdf_path = output_dir / "real_eval_report.pdf"
    with PdfPages(str(pdf_path)) as pdf:
        _cover_page(pdf, score_data)
        _iou_histogram_page(pdf, histogram_path)
        _precision_recall_table_page(pdf, recall_report, precision_report)
        _scenario_summary_page(pdf, iou_report, fp_report)
        _incident_page(pdf, incident_report)

        # PDF metadata
        d = pdf.infodict()
        d["Title"] = "ARGUS — Real Validation Report"
        d["Author"] = "ARGUS eval_real.py"
        d["Subject"] = "YOLOv8 + DeepSORT real-world evaluation"

    return pdf_path


# ---------------------------------------------------------------------------
# Placeholder for missing annotations
# ---------------------------------------------------------------------------

def _save_placeholder(output_dir: str) -> None:
    """Write a minimal placeholder JSON when annotations are not yet available."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    placeholder = {
        "status": "Awaiting CVAT annotations",
        "instructions": [
            "Import annotation_tasks/cvat_import.zip into CVAT",
            "Annotate all frames, then export as YOLO 1.1",
            "Place export in annotation_tasks/cvat_export/",
            "Then re-run: python eval_real.py ...",
        ],
        "reports": {
            "iou_report.json": "not generated",
            "fp_report.json": "not generated",
            "lag_report_real.json": "not generated",
            "recall_report.json": "not generated",
            "precision_report.json": "not generated",
            "incident_report.json": "not generated",
            "real_aggregate_score.json": "not generated",
            "real_eval_report.pdf": "not generated",
        },
    }

    placeholder_path = out / "placeholder_status.json"
    placeholder_path.write_text(json.dumps(placeholder, indent=2), encoding="utf-8")
    print(f"  Placeholder report saved to: {placeholder_path}")


def _annotations_ready(annotation_dir: str) -> bool:
    """Return True only when the annotation directory exists and contains at least one .txt file
    that is not a metadata file (obj.data, obj.names, etc.)."""
    skip_names = {"obj.data", "obj.names", "classes.txt", "data.yaml"}
    ann_path = Path(annotation_dir)

    if not ann_path.exists():
        return False

    for txt_file in ann_path.rglob("*.txt"):
        if txt_file.name not in skip_names:
            return True

    return False


# ---------------------------------------------------------------------------
# JSON helper
# ---------------------------------------------------------------------------

def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-validation pipeline for YOLOv8 + DeepSORT vehicle detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--annotations",
        required=True,
        help="Path to CVAT YOLO 1.1 export directory (e.g. annotation_tasks/cvat_export/)",
    )
    parser.add_argument(
        "--footage",
        required=True,
        help="Directory containing .jpg frames for inference (e.g. unseen_test_videos/)",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to YOLOv8 model weights (e.g. models/week2_retrained.pt)",
    )
    parser.add_argument(
        "--output",
        default="reports_validated/",
        help="Output directory for all reports (default: reports_validated/)",
    )
    parser.add_argument(
        "--img-width",
        type=int,
        default=FRAME_WIDTH,
        help=f"Frame width in pixels for centroid offset calculation (default: {FRAME_WIDTH})",
    )
    parser.add_argument(
        "--img-height",
        type=int,
        default=FRAME_HEIGHT,
        help=f"Frame height in pixels for centroid offset calculation (default: {FRAME_HEIGHT})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the full real-validation pipeline.

    Returns 0 on success, non-zero on unrecoverable error.
    """
    args = parse_args(argv)

    # ------------------------------------------------------------------
    # Check for annotations
    # ------------------------------------------------------------------
    if not _annotations_ready(args.annotations):
        print("=" * 60)
        print("  Awaiting CVAT annotations")
        print("  Import annotation_tasks/cvat_import.zip into CVAT")
        print("  Annotate all frames, then export as YOLO 1.1")
        print("  Place export in annotation_tasks/cvat_export/")
        print("  Then re-run: python eval_real.py ...")
        print("=" * 60)
        _save_placeholder(args.output)
        return 0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotation_dir = Path(args.annotations)
    footage_dir = Path(args.footage)

    img_w = args.img_width
    img_h = args.img_height

    print(f"\n{'=' * 60}")
    print("  ARGUS — Real Validation Pipeline")
    print(f"{'=' * 60}")
    print(f"  Annotations : {annotation_dir}")
    print(f"  Footage     : {footage_dir}")
    print(f"  Model       : {args.model}")
    print(f"  Output      : {output_dir}")
    print(f"  Frame size  : {img_w}x{img_h}")
    print(f"{'=' * 60}\n")

    # ------------------------------------------------------------------
    # Load annotations
    # ------------------------------------------------------------------
    print("[1/8] Loading annotations ...")
    class_names = load_class_names(annotation_dir)
    print(f"  Class names: {class_names}")

    annotations = load_annotations(annotation_dir)
    print(f"  Loaded annotations for {len(annotations)} frame(s).")

    if not annotations:
        print("  WARNING: Annotation files found but all were empty or malformed.")
        print("  Treating as 'awaiting annotations' and writing placeholder.")
        _save_placeholder(args.output)
        return 0

    # ------------------------------------------------------------------
    # Run inference
    # ------------------------------------------------------------------
    print("\n[2/8] Running model inference ...")
    if not footage_dir.exists():
        print(f"  WARNING: Footage directory does not exist: {footage_dir}")
        print("  Cannot run inference without footage. Exiting.")
        return 1

    predictions = run_inference(args.model, footage_dir, set(annotations.keys()))
    print(f"  Inference complete: {len(predictions)} frame(s) processed.")

    # ------------------------------------------------------------------
    # Per-frame evaluation
    # ------------------------------------------------------------------
    print("\n[3/8] Evaluating predictions against ground truth ...")
    eval_results = EvaluationResults(class_names)

    # Only evaluate frames present in both annotations and predictions
    common_stems = set(annotations.keys()) & set(predictions.keys())
    annotation_only = set(annotations.keys()) - set(predictions.keys())
    prediction_only = set(predictions.keys()) - set(annotations.keys())

    print(f"  Frames with both annotation + prediction : {len(common_stems)}")
    if annotation_only:
        print(f"  Frames with annotation only (FN source)  : {len(annotation_only)}")
    if prediction_only:
        print(f"  Frames with prediction only (FP source)  : {len(prediction_only)}")

    # Evaluate matched frames
    for stem in sorted(common_stems):
        gt_entries = annotations[stem]
        pred_entries = predictions[stem]
        evaluate_frame(stem, gt_entries, pred_entries, eval_results, img_w, img_h)

    # Account for annotation-only frames: all GT boxes are FN, no predictions
    for stem in annotation_only:
        gt_entries = annotations[stem]
        evaluate_frame(stem, gt_entries, [], eval_results, img_w, img_h)

    # Account for prediction-only frames: all predictions are FP, no GT
    for stem in prediction_only:
        pred_entries = predictions[stem]
        evaluate_frame(stem, [], pred_entries, eval_results, img_w, img_h)

    # ------------------------------------------------------------------
    # Build and save individual reports
    # ------------------------------------------------------------------
    print("\n[4/8] Building reports ...")

    iou_report = build_iou_report(eval_results)
    _write_json(output_dir / "iou_report.json", iou_report)

    fp_report = build_fp_report(eval_results)
    _write_json(output_dir / "fp_report.json", fp_report)

    lag_report = build_lag_report(eval_results)
    _write_json(output_dir / "lag_report_real.json", lag_report)

    recall_report = build_recall_report(eval_results)
    _write_json(output_dir / "recall_report.json", recall_report)

    precision_report = build_precision_report(eval_results)
    _write_json(output_dir / "precision_report.json", precision_report)

    # ------------------------------------------------------------------
    # Incident detection accuracy
    # ------------------------------------------------------------------
    print("\n[5/8] Computing incident detection accuracy ...")
    incident_report = build_incident_report(footage_dir, predictions)
    _write_json(output_dir / "incident_report.json", incident_report)

    # ------------------------------------------------------------------
    # Aggregate score
    # ------------------------------------------------------------------
    print("\n[6/8] Computing aggregate score ...")
    score_data = compute_aggregate_score(
        iou_report, fp_report, recall_report, precision_report, lag_report
    )
    _write_json(output_dir / "real_aggregate_score.json", score_data)
    print(
        f"  Aggregate score: {score_data['aggregate_score']:.1f} / 100  "
        f"(Grade: {score_data['grade']})"
    )

    # ------------------------------------------------------------------
    # IoU histogram
    # ------------------------------------------------------------------
    print("\n[7/8] Saving IoU histogram ...")
    histogram_path = output_dir / "iou_histogram.png"
    save_iou_histogram(eval_results.all_iou_values, histogram_path)
    print(f"  Saved: {histogram_path}")

    # ------------------------------------------------------------------
    # PDF report
    # ------------------------------------------------------------------
    print("\n[8/8] Generating PDF report ...")
    pdf_path = save_pdf_report(
        output_dir,
        score_data,
        iou_report,
        fp_report,
        recall_report,
        precision_report,
        lag_report,
        incident_report,
        histogram_path,
    )
    print(f"  Saved: {pdf_path}")

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  Evaluation Complete")
    print(f"{'=' * 60}")
    print(f"  Overall mean IoU      : {iou_report['overall_mean_iou']:.4f}")
    print(f"  Overall FP rate       : {fp_report['overall_fp_rate']:.4f}")
    print(f"  Overall recall        : {recall_report['overall_recall']:.4f}")
    print(f"  Overall precision     : {precision_report['overall_precision']:.4f}")
    print(f"  Mean centroid offset  : {lag_report['mean_offset_px']:.2f} px")
    print(f"  P95 centroid offset   : {lag_report['p95_offset_px']:.2f} px")
    print(f"  Incident miss rate    : {incident_report['miss_rate']:.1%}")
    print(f"  Incident FA rate      : {incident_report['false_alarm_rate']:.1%}")
    print(f"  Aggregate score       : {score_data['aggregate_score']:.1f} / 100  "
          f"[Grade {score_data['grade']}]")
    print(f"{'=' * 60}")

    if fp_report["flagged_scenarios"]:
        print(f"\n  FLAGGED — High FP rate scenarios: {fp_report['flagged_scenarios']}")
    if recall_report["flagged_classes"]:
        print(f"  FLAGGED — Low recall classes    : {recall_report['flagged_classes']}")
    if precision_report["flagged_classes"]:
        print(f"  FLAGGED — Low precision classes : {precision_report['flagged_classes']}")
    if lag_report["flagged_frames"]:
        print(f"  FLAGGED — High-lag frames       : {lag_report['flagged_frames']}")

    print(f"\n  All reports written to: {output_dir.resolve()}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
