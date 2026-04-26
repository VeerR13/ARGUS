"""
YOLO12x vehicle detector with temporal confirmation buffer.

Model: YOLO12x (ultralytics 8.4.x) — flagship as of 2025.
  - Faster and lighter than RF-DETR at equivalent accuracy on COCO
  - Native MPS / CUDA / CPU support, no manual device placement
  - TensorRT / ONNX export path for Jetson edge deployment
  - Built-in ByteTrack via model.track() — used here in detect-only mode
    so VehicleTracker handles tracking (keeps pipeline modular)

Class IDs are COCO-80, same as before (car=2, motorcycle=3, bus=5, truck=7).
No downstream changes required.
"""

import os

import numpy as np
from ultralytics import YOLO

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

CONFIRM_MIN_FRAMES = 2
CONFIRM_IOU_THRESH = 0.40


def _iou(a: list, b: list) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class TemporalConfirmationBuffer:
    """
    Filters raw detections before they reach the tracker.
    A candidate is forwarded only once matched across CONFIRM_MIN_FRAMES
    consecutive frames (IoU >= CONFIRM_IOU_THRESH). Suppresses single-frame
    phantoms from shadows, reflections, and motion blur.
    """

    def __init__(self, min_frames: int = CONFIRM_MIN_FRAMES,
                 iou_threshold: float = CONFIRM_IOU_THRESH):
        self.min_frames    = min_frames
        self.iou_threshold = iou_threshold
        self._candidates: list[dict] = []

    @staticmethod
    def _to_xyxy(d: dict) -> list:
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
            if best_iou >= self.iou_threshold and best_ti >= 0:
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
            if cand["count"] >= self.min_frames:
                x1, y1, x2, y2 = cand["bbox_xyxy"]
                confirmed.append({
                    "bbox_xywh":  [x1, y1, x2 - x1, y2 - y1],
                    "confidence": cand["confidence"],
                    "class_id":   cand["class_id"],
                    "class_name": cand["class_name"],
                })
        return confirmed

    def unconfirmed_detections(self) -> list[dict]:
        phantoms = []
        for cand in self._candidates:
            if cand["count"] < self.min_frames:
                x1, y1, x2, y2 = cand["bbox_xyxy"]
                phantoms.append({
                    "bbox_xywh":  [x1, y1, x2 - x1, y2 - y1],
                    "confidence": cand["confidence"],
                    "class_id":   cand["class_id"],
                    "class_name": cand["class_name"],
                })
        return phantoms


class VehicleDetector:
    def __init__(
        self,
        model_path: str   = "yolo12x.pt",
        confidence: float = 0.35,
        device: str       = "auto",          # "auto" → MPS on Apple Silicon, CUDA on GPU, else CPU
        temporal_confirm: bool = True,
        imgsz: int        = 640,
    ):
        if device == "auto":
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.model      = YOLO(model_path)
        self.confidence = confidence
        self.device     = device
        self.imgsz      = imgsz
        self._buffer    = TemporalConfirmationBuffer() if temporal_confirm else None

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run YOLO12x on a single BGR frame.

        Returns list of dicts:
            bbox_xywh  : [x_topleft, y_topleft, width, height]
            confidence : float
            class_id   : int
            class_name : str
        """
        results = self.model.predict(
            frame,
            conf=self.confidence,
            device=self.device,
            imgsz=self.imgsz,
            verbose=False,
            classes=list(VEHICLE_CLASSES.keys()),
        )

        raw = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in VEHICLE_CLASSES:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                raw.append({
                    "bbox_xywh":  [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                    "confidence": float(box.conf[0]),
                    "class_id":   cls_id,
                    "class_name": VEHICLE_CLASSES[cls_id],
                })

        if self._buffer is not None:
            return self._buffer.update(raw)
        return raw

    @property
    def phantom_candidates(self) -> list[dict]:
        if self._buffer is None:
            return []
        return self._buffer.unconfirmed_detections()
