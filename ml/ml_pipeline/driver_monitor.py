"""
Driver monitoring pipeline — gaze deviation, drowsiness, phone/distraction.

Gaze:       MediaPipe head-yaw fast-path (>25°) + optional L2CS-Net for 10-25°
            ambiguous cases (pip install l2cs, MIT license).
Drowsiness: PERCLOS via Eye Aspect Ratio over a 30-frame sliding window.
            MAR (mouth aspect ratio) for yawning as secondary signal.
Phone:      YOLOv8n with State Farm fine-tuned weights (2.84 MB).
            Falls back to COCO yolov8n.pt + cell-phone class if custom weights absent.

Usage:
    monitor = DriverMonitor(use_phone_detector=True)
    events  = monitor.process_frame(frame_bgr, timestamp_ms=1000)
    closed  = monitor.flush()   # call once at end of video
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp

# ── MediaPipe landmark indices ────────────────────────────────────────────────
# Eye Aspect Ratio: outer, top-L, top-R, inner, bot-R, bot-L
_LEFT_EYE  = [33,  160, 158, 133, 153, 144]
_RIGHT_EYE = [362, 385, 387, 263, 373, 380]
# Mouth Aspect Ratio: top-lip, bot-lip, left-corner, right-corner
_MOUTH     = [13, 14, 61, 291]
# Head yaw: nose tip, left pinna, right pinna
_NOSE      = 1
_L_EAR_LM  = 234
_R_EAR_LM  = 454

# COCO class ID for "cell phone" (fallback when State Farm weights absent)
_COCO_PHONE_CLASS = 67


@dataclass
class DriverEvent:
    start_ts:   int    # milliseconds
    end_ts:     int    # milliseconds
    type:       str    # "gaze_away" | "drowsy" | "phone"
    confidence: float  # 0.0 – 1.0


def _ear(lm, ids: list[int], w: int, h: int) -> float:
    pts = [(lm[i].x * w, lm[i].y * h) for i in ids]
    v1 = math.dist(pts[1], pts[5])
    v2 = math.dist(pts[2], pts[4])
    hd = math.dist(pts[0], pts[3])
    return (v1 + v2) / (2.0 * hd) if hd > 0 else 0.0


def _mar(lm, w: int, h: int) -> float:
    top = (lm[_MOUTH[0]].x * w, lm[_MOUTH[0]].y * h)
    bot = (lm[_MOUTH[1]].x * w, lm[_MOUTH[1]].y * h)
    left = (lm[_MOUTH[2]].x * w, lm[_MOUTH[2]].y * h)
    right = (lm[_MOUTH[3]].x * w, lm[_MOUTH[3]].y * h)
    v = math.dist(top, bot)
    h_ = math.dist(left, right)
    return v / h_ if h_ > 0 else 0.0


def _yaw_deg(lm) -> float:
    nose_x  = lm[_NOSE].x
    left_x  = lm[_L_EAR_LM].x
    right_x = lm[_R_EAR_LM].x
    face_w  = right_x - left_x
    if face_w <= 0:
        return 0.0
    centre_x = (left_x + right_x) / 2.0
    return float(((nose_x - centre_x) / face_w) * 90.0)


class DriverMonitor:
    """Process driver-facing frames and emit DriverEvent records."""

    # ── Drowsiness thresholds ─────────────────────────────────────────────────
    EAR_THRESH             = 0.22   # below = eye closed
    PERCLOS_WINDOW         = 30     # frames (~1 s at 30 fps)
    PERCLOS_THRESH         = 0.80   # ≥80 % closed → drowsy
    DROWSY_MIN_MS          = 2500   # must persist before emitting
    SUNGLASSES_WINDOW      = 150    # 5 s at 30 fps
    SUNGLASSES_EAR_MAX     = 0.15   # EAR this low + no blinks = sunglasses likely

    # ── Gaze thresholds ───────────────────────────────────────────────────────
    GAZE_FAST_DEG          = 25.0   # obvious look-away, no L2CS needed
    GAZE_L2CS_DEG          = 10.0   # below this: driver is looking forward
    GAZE_MIN_MS            = 1500   # must look away ≥1.5 s before emitting

    # ── Phone detection ───────────────────────────────────────────────────────
    PHONE_CONF             = 0.60
    PHONE_PERSIST          = 3      # consecutive frames required
    PHONE_ASPECT_MIN       = 1.0    # height/width: phones are tall

    # ── Global gate ──────────────────────────────────────────────────────────
    FACE_CONF_MIN          = 0.75

    def __init__(
        self,
        use_l2cs: bool = False,
        use_phone_detector: bool = True,
        phone_model_path: str = "yolov8n-statefarm.pt",
        fps: float = 30.0,
    ) -> None:
        """
        use_l2cs          : L2CS-Net for refined gaze on ambiguous 10-25° cases.
                            Requires: pip install l2cs
        use_phone_detector: YOLOv8n phone/distraction detector.
                            Provide fine-tuned weights or falls back to COCO yolov8n.pt.
        phone_model_path  : State Farm fine-tuned YOLOv8n weights path.
                            Download: see README driver cam section.
        fps               : Source video frame rate.
        """
        self.fps = fps

        self._mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=self.FACE_CONF_MIN,
            min_tracking_confidence=0.5,
        )

        self._l2cs = None
        if use_l2cs:
            try:
                from l2cs import Pipeline as L2CSPipeline
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._l2cs = L2CSPipeline(
                    arch="ResNet50",
                    weights="L2CSNet_gaze360.pkl",
                    device=device,
                )
            except ImportError:
                print("l2cs not installed — head-pose proxy used for all angles. pip install l2cs")

        self._phone_model = None
        if use_phone_detector:
            try:
                from ultralytics import YOLO
                import pathlib
                weights = phone_model_path if pathlib.Path(phone_model_path).exists() else "yolov8n.pt"
                if weights == "yolov8n.pt":
                    print(f"Phone weights not found at {phone_model_path} — using COCO yolov8n (cell phone class)")
                self._phone_model = YOLO(weights)
                self._phone_coco_only = (weights == "yolov8n.pt")
            except Exception as e:
                print(f"Phone detector disabled: {e}")

        # Rolling buffers
        self._ear_buf:    deque[float] = deque(maxlen=self.PERCLOS_WINDOW)
        self._blink_buf:  deque[int]   = deque(maxlen=self.SUNGLASSES_WINDOW)
        self._phone_hits: int          = 0

        # Active event start timestamps
        self._gaze_start:  Optional[int] = None
        self._drowsy_start: Optional[int] = None
        self._phone_start:  Optional[int] = None
        self._last_ts:      int           = 0

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: int,
    ) -> list[DriverEvent]:
        """
        Process one BGR frame at timestamp_ms.
        Returns DriverEvents that completed this frame (start+end both resolved).
        """
        self._last_ts = timestamp_ms
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        completed: list[DriverEvent] = []

        results = self._mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            self._gaze_start = self._drowsy_start = None
            self._phone_hits = 0
            return completed

        lm = results.multi_face_landmarks[0].landmark

        # ── EAR + PERCLOS drowsiness ─────────────────────────────────────────
        avg_ear = (_ear(lm, _LEFT_EYE, w, h) + _ear(lm, _RIGHT_EYE, w, h)) / 2.0
        self._ear_buf.append(avg_ear)
        eye_closed = avg_ear < self.EAR_THRESH
        self._blink_buf.append(0 if eye_closed else 1)

        # Sunglasses gate: no blink for 5 s, EAR permanently near zero
        sunglasses = (
            len(self._blink_buf) == self.SUNGLASSES_WINDOW
            and sum(self._blink_buf) == 0
            and avg_ear < self.SUNGLASSES_EAR_MAX
        )

        if not sunglasses and len(self._ear_buf) == self.PERCLOS_WINDOW:
            perclos = sum(1 for e in self._ear_buf if e < self.EAR_THRESH) / self.PERCLOS_WINDOW
            if perclos >= self.PERCLOS_THRESH:
                if self._drowsy_start is None:
                    self._drowsy_start = timestamp_ms
                elif timestamp_ms - self._drowsy_start >= self.DROWSY_MIN_MS:
                    completed.append(DriverEvent(
                        start_ts=self._drowsy_start, end_ts=timestamp_ms,
                        type="drowsy", confidence=round(perclos, 2),
                    ))
                    self._drowsy_start = None
            else:
                self._drowsy_start = None

        # ── Gaze deviation ───────────────────────────────────────────────────
        yaw = _yaw_deg(lm)
        abs_yaw = abs(yaw)

        # Suppress EAR drowsiness at extreme profile — gaze detection still valid
        if abs_yaw >= 40.0:
            self._drowsy_start = None

        gaze_away = False
        if abs_yaw >= self.GAZE_FAST_DEG:
            gaze_away = True
        elif self._l2cs is not None and abs_yaw >= self.GAZE_L2CS_DEG:
            try:
                res = self._l2cs.step(frame_rgb)
                if res and len(res.yaw) > 0:
                    l2cs_yaw = abs(float(res.yaw[0])) * (180.0 / math.pi)
                    gaze_away = l2cs_yaw >= self.GAZE_FAST_DEG
            except Exception:
                gaze_away = abs_yaw >= self.GAZE_FAST_DEG

        if gaze_away:
            if self._gaze_start is None:
                self._gaze_start = timestamp_ms
            elif timestamp_ms - self._gaze_start >= self.GAZE_MIN_MS:
                completed.append(DriverEvent(
                    start_ts=self._gaze_start, end_ts=timestamp_ms,
                    type="gaze_away", confidence=round(min(1.0, abs_yaw / 45.0), 2),
                ))
                self._gaze_start = None
        else:
            self._gaze_start = None

        # ── Phone / distraction ──────────────────────────────────────────────
        if self._phone_model is not None:
            try:
                kwargs: dict = {"conf": self.PHONE_CONF, "verbose": False}
                if self._phone_coco_only:
                    kwargs["classes"] = [_COCO_PHONE_CLASS]
                det = self._phone_model.predict(frame_bgr, **kwargs)
                found = False
                for r in det:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        bh, bw = y2 - y1, x2 - x1
                        if bw > 0 and (bh / bw) >= self.PHONE_ASPECT_MIN:
                            found = True
                            break
                    if found:
                        break

                if found:
                    self._phone_hits += 1
                    if self._phone_start is None:
                        self._phone_start = int(timestamp_ms - (self._phone_hits / self.fps * 1000))
                else:
                    if self._phone_hits > 0 and self._phone_start is not None:
                        # emit on release rather than on persist to get exact window
                        pass
                    self._phone_hits = 0
                    self._phone_start = None

                if self._phone_hits >= self.PHONE_PERSIST and self._phone_start is not None:
                    completed.append(DriverEvent(
                        start_ts=self._phone_start, end_ts=timestamp_ms,
                        type="phone", confidence=min(1.0, round(self._phone_hits / 15.0, 2)),
                    ))
                    self._phone_hits = 0
                    self._phone_start = None

            except Exception:
                pass

        return completed

    def flush(self) -> list[DriverEvent]:
        """Close any open events at end of video."""
        ts = self._last_ts
        completed: list[DriverEvent] = []
        if self._drowsy_start is not None:
            completed.append(DriverEvent(
                start_ts=self._drowsy_start, end_ts=ts, type="drowsy", confidence=0.8
            ))
        if self._gaze_start is not None:
            completed.append(DriverEvent(
                start_ts=self._gaze_start, end_ts=ts, type="gaze_away", confidence=0.8
            ))
        self._drowsy_start = self._gaze_start = self._phone_start = None
        return completed
