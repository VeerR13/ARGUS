"""
ByteTrack vehicle tracker.

Replaces DeepSORT + MobileNet appearance embeddings with ByteTrack:
  - DeepSORT root cause: MobileNet embeddings trained on pedestrians fail on vehicles
    seen from dashcam angles with motion blur and partial occlusion, producing
    phantom tracks (65.6% FP rate on dashcam footage)
  - ByteTrack is IoU-only — no appearance model — so motion blur and occlusion
    don't corrupt the association step
  - ByteTrack's "byte" strategy: high-confidence detections match first (SORT-style
    Hungarian + Kalman), then low-confidence detections are matched against
    unconfirmed tracks, recovering partially occluded vehicles that would otherwise
    be lost and re-initialised as new phantom tracks

ByteTrack paper: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating
Every Detection Box", ECCV 2022.
"""

import numpy as np
import supervision as sv

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


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

        active = []
        for i in range(len(tracked)):
            cls_id = int(tracked.class_id[i])
            x1, y1, x2, y2 = tracked.xyxy[i]
            active.append({
                "track_id":  int(tracked.tracker_id[i]),
                "bbox_ltrb": [int(x1), int(y1), int(x2), int(y2)],
                "class_name": VEHICLE_CLASSES.get(cls_id, "car"),
            })
        return active
