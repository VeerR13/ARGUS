"""
YOLOv8 vehicle detector with temporal confirmation buffer.

Bug fixes applied:
  1. Confidence raised 0.4 → 0.5; IOU threshold set to 0.35 (tighter NMS).
  2. TemporalConfirmationBuffer: a detection must appear in min_frames consecutive
     frames (matched by IoU ≥ iou_threshold) before being passed to the tracker.
     Eliminates single-frame phantom detections from shadows, reflections, motion blur.
"""

from collections import deque

from ultralytics import YOLO

# COCO class IDs for vehicles we care about (default for yolov8s/n pretrained)
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Vehicle class names that custom-trained models may use
_VEHICLE_NAMES = {"car", "motorcycle", "bus", "truck", "vehicle"}

# ── Temporal confirmation ──────────────────────────────────────────────────────
CONFIRM_MIN_FRAMES  = 3    # detection must appear in this many consecutive frames
CONFIRM_IOU_THRESH  = 0.40  # IoU to match a detection across frames


def _iou(a: list, b: list) -> float:
    """IoU between two [x1, y1, x2, y2] boxes."""
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
    Filters raw YOLO detections before they reach the tracker.

    Each detection candidate is tracked internally.  It is only forwarded to
    DeepSORT once it has been matched (IoU ≥ CONFIRM_IOU_THRESH) in at least
    CONFIRM_MIN_FRAMES consecutive frames.  If a candidate misses a frame, its
    consecutive count resets.

    This eliminates phantom detections that fire on a single frame — shadows,
    specular reflections, motion-blur streaks — while adding only a small
    latency (CONFIRM_MIN_FRAMES − 1 frames ≈ 0.08 s at 25 fps).
    """

    def __init__(self, min_frames: int = CONFIRM_MIN_FRAMES,
                 iou_threshold: float = CONFIRM_IOU_THRESH):
        self.min_frames    = min_frames
        self.iou_threshold = iou_threshold
        # Each entry: bbox_xyxy, class_id, class_name, confidence, consecutive count
        self._candidates: list[dict] = []

    @staticmethod
    def _to_xyxy(d: dict) -> list:
        x, y, w, h = d["bbox_xywh"]
        return [x, y, x + w, y + h]

    def update(self, detections: list[dict]) -> list[dict]:
        """
        Feed this frame's raw YOLO detections.
        Returns only those confirmed across min_frames consecutive frames.
        """
        curr_boxes = [self._to_xyxy(d) for d in detections]
        matched_cand = [False] * len(self._candidates)
        matched_curr = [False] * len(detections)

        # Greedy best-IoU match: each current det → best unmatched candidate
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
                cand["count"]      += 1
                cand["bbox_xyxy"]   = cbox
                cand["confidence"]  = detections[ci]["confidence"]
                matched_cand[best_ti] = True
                matched_curr[ci]      = True

        # Unmatched candidates → drop (missed this frame, streak broken)
        surviving = [c for c, m in zip(self._candidates, matched_cand) if m]

        # Unmatched current detections → new candidates at count=1
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

        # Emit only candidates confirmed for min_frames consecutive frames
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
        """
        Return current candidates that never reached min_frames.
        Used by hard-negative mining to collect likely phantom detections.
        """
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
        model_path: str  = "yolov8s.pt",
        confidence: float = 0.50,   # raised from 0.4 — reduces low-conf phantoms
        iou: float        = 0.35,   # tighter NMS — suppresses duplicate boxes on same vehicle
        device: str       = "cpu",
        temporal_confirm: bool = True,
    ):
        self.model      = YOLO(model_path)
        self.confidence = confidence
        self.iou        = iou
        self.device     = device
        self._buffer    = TemporalConfirmationBuffer() if temporal_confirm else None
        # Build class filter from the loaded model's own names (handles both COCO
        # pretrained models and custom-trained models with different class IDs).
        self._vehicle_classes = {
            cid: name
            for cid, name in self.model.names.items()
            if name.lower() in _VEHICLE_NAMES
        } or VEHICLE_CLASSES  # fall back to COCO if model has no matching names

    def detect(self, frame) -> list[dict]:
        """
        Run detection on a single BGR frame.

        With temporal_confirm=True (default), returns only detections that have
        appeared in CONFIRM_MIN_FRAMES consecutive frames — eliminating phantoms.

        Returns list of dicts:
            bbox_xywh   : [x_topleft, y_topleft, width, height]
            confidence  : float
            class_id    : int
            class_name  : str
        """
        results = self.model(
            frame,
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )[0]

        raw = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in self._vehicle_classes:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            raw.append({
                "bbox_xywh":  [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                "confidence": float(box.conf[0]),
                "class_id":   cls_id,
                "class_name": self._vehicle_classes[cls_id],
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
