"""
Accumulates per-frame track data and builds trajectory records.

Speed estimation is pixel-displacement-based (uncalibrated).
Week 3 / calibration pass will replace pixels_per_meter with a proper homography.
"""

import math
import uuid


class TrajectoryBuilder:
    def __init__(self, fps: float, pixels_per_meter: float = 30.0):
        """
        fps              : video frame rate (used to convert frame displacement → m/s)
        pixels_per_meter : rough calibration constant — 30 is a typical dashcam estimate.
                           Change this via --pixels-per-meter in week1_test.py once you
                           know your camera's approximate scale.
        """
        self.fps = fps
        self.pixels_per_meter = pixels_per_meter
        self._tracks: dict[int, dict] = {}

    # ------------------------------------------------------------------
    def update(self, frame_num: int, active_tracks: list[dict]) -> None:
        """Call once per frame with the confirmed tracks from VehicleTracker."""
        for track in active_tracks:
            tid = track["track_id"]
            l, t, r, b = track["bbox_ltrb"]
            cx = (l + r) // 2
            cy = (t + b) // 2
            ts_ms = int((frame_num / self.fps) * 1000)

            if tid not in self._tracks:
                self._tracks[tid] = {
                    "trajectory_id": str(uuid.uuid4()),
                    "vehicle_id": tid,
                    "vehicle_class": track["class_name"],
                    "first_frame": frame_num,
                    "last_frame": frame_num,
                    "frames": [],
                    "_prev_center": None,
                    "_prev_frame": None,
                }

            entry = self._tracks[tid]
            entry["last_frame"] = frame_num

            # Speed: pixel displacement between consecutive centres → km/h.
            # Zero-out if:
            #   (a) no previous position exists (first appearance), or
            #   (b) the vehicle was absent last frame (frame gap = tracking glitch), or
            #   (c) computed speed exceeds 200 km/h (impossible for road traffic).
            speed = 0.0
            consecutive = entry["_prev_frame"] is not None and entry["_prev_frame"] == frame_num - 1
            if entry["_prev_center"] is not None and consecutive:
                px, py = entry["_prev_center"]
                pixel_dist = math.hypot(cx - px, cy - py)
                meters_per_frame = pixel_dist / self.pixels_per_meter
                raw_speed = meters_per_frame * self.fps * 3.6   # m/s → km/h
                speed = round(raw_speed, 1) if raw_speed <= 200.0 else 0.0

            entry["frames"].append({
                "frame_num": frame_num,
                "timestamp_ms": ts_ms,
                "bbox": [l, t, r, b],
                "center": [cx, cy],
                "speed_estimate": speed,
            })
            entry["_prev_center"] = [cx, cy]
            entry["_prev_frame"] = frame_num

    # ------------------------------------------------------------------
    def get_trajectories(self, video_id: str) -> list[dict]:
        """Return trajectory records matching the project JSON spec."""
        result = []
        for entry in self._tracks.values():
            traj = {k: v for k, v in entry.items() if not k.startswith("_")}
            traj["video_id"] = video_id
            traj["frame_count"] = len(traj["frames"])
            result.append(traj)
        return result
