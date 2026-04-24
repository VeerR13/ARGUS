"""
RF-DETR vehicle detector with temporal confirmation buffer.

Replaces YOLOv8 with RF-DETR (Roboflow Detection Transformer):
  - Transformer attention captures whole-scene context, reducing phantom detections
    from shadows and reflections that fool anchor-based grid classifiers
  - Higher COCO AP (53.3 base / 56.3 large) vs YOLOv8s (48.0)
  - Confidence threshold lowered 0.50 → 0.40: RF-DETR scores are better calibrated,
    so a lower threshold recovers occluded vehicles without flooding phantoms
  - TemporalConfirmationBuffer kept but min_frames reduced 3 → 2: transformer
    already suppresses most single-frame phantoms; 3-frame latency was cutting
    legitimate fast-approach detections

Class IDs: RF-DETR is COCO-80-class, same mapping as YOLOv8 (car=2, motorcycle=3,
bus=5, truck=7). No mapping changes required downstream.
"""

import os

import cv2
import numpy as np
from rfdetr import RFDETRBase, RFDETRLarge

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

_VEHICLE_NAMES = {"car", "motorcycle", "bus", "truck", "vehicle"}

CONFIRM_MIN_FRAMES = 2      # reduced from 3 — RF-DETR has fewer single-frame phantoms
CONFIRM_IOU_THRESH = 0.40


def _iou(a: list, b: list) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw  = max(0, ix2 - ix1)
    ih  = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class TemporalConfirmationBuffer:
    """
    Filters raw detections before they reach the tracker.

    A detection candidate is forwarded to ByteTrack only once it has been
    matched (IoU ≥ CONFIRM_IOU_THRESH) in at least CONFIRM_MIN_FRAMES
    consecutive frames.  If a candidate misses a frame, its streak resets.

    With RF-DETR (min_frames=2) the latency is one frame (~33 ms at 30 fps).
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
        curr_boxes    = [self._to_xyxy(d) for d in detections]
        matched_cand  = [False] * len(self._candidates)
        matched_curr  = [False] * len(detections)

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
        """Unconfirmed (likely phantom) detections — used by hard-negative mining."""
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
        model_path: str   = "rfdetr-base",  # "rfdetr-large" for +3 AP, ~2× slower
        confidence: float = 0.40,            # lower than YOLOv8 — RF-DETR calibrated
        device: str       = "cpu",
        temporal_confirm: bool = True,
        pretrain_weights: str | None = None,  # path to a finetuned .pth checkpoint
    ):
        large  = "large" in model_path.lower()
        kwargs = {}
        if pretrain_weights and os.path.exists(pretrain_weights):
            kwargs["pretrain_weights"] = pretrain_weights
        self.model      = RFDETRLarge(**kwargs) if large else RFDETRBase(**kwargs)
        self.confidence = confidence
        self.device     = device
        self._buffer    = TemporalConfirmationBuffer() if temporal_confirm else None

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run RF-DETR on a single BGR frame.

        With temporal_confirm=True (default), only returns detections confirmed
        across CONFIRM_MIN_FRAMES consecutive frames.

        Returns list of dicts:
            bbox_xywh  : [x_topleft, y_topleft, width, height]
            confidence : float
            class_id   : int
            class_name : str
        """
        # RF-DETR expects RGB; OpenCV delivers BGR
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        sv_dets = self.model.predict(rgb, threshold=self.confidence)

        raw = []
        n = len(sv_dets) if sv_dets is not None else 0
        for i in range(n):
            cls_id = int(sv_dets.class_id[i])
            if cls_id not in VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = sv_dets.xyxy[i]
            raw.append({
                "bbox_xywh":  [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                "confidence": float(sv_dets.confidence[i]),
                "class_id":   cls_id,
                "class_name": VEHICLE_CLASSES[cls_id],
            })

        if self._buffer is not None:
            return self._buffer.update(raw)
        return raw

    @property
    def phantom_candidates(self) -> list[dict]:
        """Unconfirmed (likely phantom) detections from the current buffer state."""
        if self._buffer is None:
            return []
        return self._buffer.unconfirmed_detections()
