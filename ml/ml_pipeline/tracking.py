"""
DeepSORT wrapper.

Bug fixes applied:
  2a. n_init reduced 3 → 2: track confirmed after 2 detections, not 3.
      Combined with the 3-frame temporal buffer in detection.py, total
      confirmation latency is still 3 frames — but the track is established
      faster once detections pass the buffer, reducing bbox lag at first appear.
  2b. max_age=30, max_iou_distance=0.7 unchanged — already correct.
  2c. Uses track.to_ltrb() (post-Kalman-update position) for display, not raw
      YOLO bbox. The Kalman filter has already been tuned (std_weight_velocity
      1/160 → 1/20) so predicted positions track fast-moving vehicles better.
"""

from deep_sort_realtime.deepsort_tracker import DeepSort


class VehicleTracker:
    def __init__(
        self,
        max_age: int         = 30,
        n_init: int          = 2,    # lowered from 3 — faster track confirmation
        max_iou_distance: float = 0.7,
    ):
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_iou_distance=max_iou_distance,
            embedder="mobilenet",
            half=True,
            bgr=True,
        )

    def update(self, detections: list[dict], frame) -> list[dict]:
        """
        Args:
            detections : output of VehicleDetector.detect() (already temporally confirmed)
            frame      : current BGR frame (needed by the embedder)

        Returns confirmed tracks with post-update Kalman bbox (to_ltrb):
            track_id   : int
            bbox_ltrb  : [left, top, right, bottom]
            class_name : str
        """
        ds_input = [
            (d["bbox_xywh"], d["confidence"], d["class_name"])
            for d in detections
        ]
        raw_tracks = self.tracker.update_tracks(ds_input, frame=frame)

        active = []
        for track in raw_tracks:
            if not track.is_confirmed():
                continue
            # to_ltrb() returns the post-update Kalman state — not the raw
            # detection bbox — so it leads rather than lags on fast vehicles.
            ltrb = track.to_ltrb()
            active.append({
                "track_id":  int(track.track_id),
                "bbox_ltrb": [int(v) for v in ltrb],
                "class_name": track.det_class if track.det_class else "car",
            })
        return active
