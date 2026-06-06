"""
YOLO12x vehicle detector with temporal confirmation buffer.

Supports both pretrained and finetuned YOLO12x weights:
  - Pretrained (yolo12x.pt):         COCO class IDs (car=2, moto=3, bus=5, truck=7)
  - Finetuned  (argus_yolo12x_best.pt): custom IDs   (car=0, moto=1, bus=2, truck=3, bicycle=4)

Class mapping is resolved automatically from model.names at load time — no
manual flag needed. Inference is ~38 fps on Apple Silicon MPS; export to
TensorRT via `model.export(format='engine')` for Jetson (~70 fps).
"""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from .constants import COCO_VEHICLE_CLASSES, FINETUNED_VEHICLE_CLASSES

# How many consecutive frames a detection must appear before it is forwarded
# to the tracker. Suppresses single-frame phantoms from shadows and reflections.
_CONFIRM_MIN_FRAMES = 2
_CONFIRM_IOU_THRESH = 0.40


def _iou(a: list[int], b: list[int]) -> float:
    """Intersection-over-union for two xyxy boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


class _TemporalConfirmationBuffer:
    """
    Gate that requires a detection to appear in _CONFIRM_MIN_FRAMES consecutive
    frames (matched by IoU) before forwarding it to the tracker.
    """

    def __init__(self) -> None:
        self._candidates: list[dict] = []

    @staticmethod
    def _to_xyxy(d: dict) -> list[int]:
        x, y, w, h = d["bbox_xywh"]
        return [x, y, x + w, y + h]

    def update(self, detections: list[dict]) -> list[dict]:
        curr_boxes   = [self._to_xyxy(d) for d in detections]
        matched_cand = [False] * len(self._candidates)
        matched_curr = [False] * len(detections)

        for ci, cbox in enumerate(curr_boxes):
            best_iou, best_ti = 0.0, -1
            for ti, cand in enumerate(self._candidates):
                if matched_cand[ti]:
                    continue
                iou = _iou(cbox, cand["bbox_xyxy"])
                if iou > best_iou:
                    best_iou, best_ti = iou, ti
            if best_iou >= _CONFIRM_IOU_THRESH and best_ti >= 0:
                cand = self._candidates[best_ti]
                cand["count"]     += 1
                cand["bbox_xyxy"]  = cbox
                cand["confidence"] = detections[ci]["confidence"]
                matched_cand[best_ti] = True
                matched_curr[ci]      = True

        surviving = [c for c, m in zip(self._candidates, matched_cand) if m]
        for ci, det in enumerate(detections):
            if not matched_curr[ci]:
                surviving.append({
                    "bbox_xyxy":  curr_boxes[ci],
                    "class_id":   det["class_id"],
                    "class_name": det["class_name"],
                    "confidence": det["confidence"],
                    "count":      1,
                })
        self._candidates = surviving

        confirmed = []
        for cand in self._candidates:
            if cand["count"] >= _CONFIRM_MIN_FRAMES:
                x1, y1, x2, y2 = cand["bbox_xyxy"]
                confirmed.append({
                    "bbox_xywh":  [x1, y1, x2 - x1, y2 - y1],
                    "confidence": cand["confidence"],
                    "class_id":   cand["class_id"],
                    "class_name": cand["class_name"],
                })
        return confirmed

    @property
    def unconfirmed(self) -> list[dict]:
        """Detections still in probation — single-frame phantoms live here."""
        result = []
        for cand in self._candidates:
            if cand["count"] < _CONFIRM_MIN_FRAMES:
                x1, y1, x2, y2 = cand["bbox_xyxy"]
                result.append({
                    "bbox_xywh":  [x1, y1, x2 - x1, y2 - y1],
                    "confidence": cand["confidence"],
                    "class_id":   cand["class_id"],
                    "class_name": cand["class_name"],
                })
        return result


def _resolve_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_class_map(model: YOLO) -> dict[int, str]:
    """
    Pick the right class-ID → name mapping based on what the model knows.

    If the model was trained on COCO (80 classes), class 2 is "car".
    If it was finetuned on the ARGUS dataset (5 classes), class 0 is "car".
    We detect this from model.names so no manual flag is required.
    """
    model_names: dict[int, str] = model.names  # e.g. {0: "person", 2: "car", ...}
    if model_names.get(0, "").lower() == "car":
        return FINETUNED_VEHICLE_CLASSES
    return COCO_VEHICLE_CLASSES


class VehicleDetector:
    """
    Run YOLO12x on a BGR frame and return confirmed vehicle detections.

    Output format per detection:
        bbox_xywh  : [x_topleft, y_topleft, width, height]  (pixels)
        confidence : float
        class_id   : int   (matches the class map in use)
        class_name : str
    """

    def __init__(
        self,
        model_path: str   = "yolo12x.pt",
        confidence: float = 0.35,
        device: str       = "auto",
        imgsz: int        = 640,
        temporal_confirm: bool = True,
    ) -> None:
        self.model      = YOLO(model_path)
        self.confidence = confidence
        self.device     = _resolve_device() if device == "auto" else device
        self.imgsz      = imgsz
        self._class_map = _resolve_class_map(self.model)
        self._buffer    = _TemporalConfirmationBuffer() if temporal_confirm else None

    def detect(self, frame: np.ndarray) -> list[dict]:
        results = self.model.predict(
            frame,
            conf=self.confidence,
            device=self.device,
            imgsz=self.imgsz,
            verbose=False,
            classes=list(self._class_map.keys()),
        )

        raw: list[dict] = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in self._class_map:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                raw.append({
                    "bbox_xywh":  [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                    "confidence": float(box.conf[0]),
                    "class_id":   cls_id,
                    "class_name": self._class_map[cls_id],
                })

        return self._buffer.update(raw) if self._buffer else raw

    @property
    def phantom_candidates(self) -> list[dict]:
        """Unconfirmed (likely phantom) detections from the current buffer."""
        return self._buffer.unconfirmed if self._buffer else []
