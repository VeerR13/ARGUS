"""
ByteTrack vehicle tracker (IoU-only, no appearance model).

Uses supervision's ByteTrack implementation. High-confidence detections
associate first via Hungarian + Kalman; low-confidence detections are matched
against unconfirmed tracks, recovering partially-occluded vehicles that pure
SORT would lose and re-initialise as phantom tracks.

Reference: Zhang et al., "ByteTrack", ECCV 2022.
"""

from __future__ import annotations

import numpy as np
import supervision as sv



class VehicleTracker:
    def __init__(
        self,
        track_activation_threshold: float = 0.25,  # min confidence for primary association
        lost_track_buffer: int             = 30,    # frames to keep a lost track alive (≈1s @30fps)
        minimum_matching_threshold: float  = 0.80,  # IoU gate for track–detection association
        frame_rate: int                    = 30,
        minimum_consecutive_frames: int    = 2,     # frames before a tentative track is confirmed
    ):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
            minimum_consecutive_frames=minimum_consecutive_frames,
        )

    def update(self, detections: list[dict], frame: np.ndarray) -> list[dict]:
        """
        Args:
            detections : output of VehicleDetector.detect() (temporally confirmed)
            frame      : current BGR frame (unused — kept for interface compatibility)

        Returns confirmed tracks:
            track_id   : int
            bbox_ltrb  : [left, top, right, bottom]
            class_name : str
        """
        if not detections:
            self.tracker.update_with_detections(sv.Detections.empty())
            return []

        xyxy = np.array([
            [d["bbox_xywh"][0],
             d["bbox_xywh"][1],
             d["bbox_xywh"][0] + d["bbox_xywh"][2],
             d["bbox_xywh"][1] + d["bbox_xywh"][3]]
            for d in detections
        ], dtype=np.float32)

        sv_dets = sv.Detections(
            xyxy=xyxy,
            confidence=np.array([d["confidence"] for d in detections], dtype=np.float32),
            class_id=np.array([d["class_id"]   for d in detections], dtype=int),
        )

        tracked = self.tracker.update_with_detections(sv_dets)

        name_map = {d["class_id"]: d["class_name"] for d in detections}

        active = []
        for i in range(len(tracked)):
            cls_id = int(tracked.class_id[i])
            x1, y1, x2, y2 = tracked.xyxy[i]
            active.append({
                "track_id":  int(tracked.tracker_id[i]),
                "bbox_ltrb": [int(x1), int(y1), int(x2), int(y2)],
                "class_name": name_map.get(cls_id, "car"),
            })
        return active
