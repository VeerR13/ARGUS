"""
Accumulates per-frame track data and builds trajectory records.

Position/velocity uses a 4-state Kalman [cx, cy, vx, vy] — eliminates
phantom speed spikes from YOLO centroid jitter.

Depth uses a separate 2-state Kalman [d, vd] fed by Depth-Anything-V2 bbox
estimates. vd (depth rate of change per frame) is stored alongside pixel
speed and gives a second, geometry-independent closing-rate signal for TTC
— particularly reliable when a vehicle already fills most of the frame
and pixel-gap convergence saturates.

Both Kalmans are optional at runtime; if no depth_map is passed the module
degrades silently to 2D-only mode with depth fields set to None.
"""

import math
import uuid

import cv2
import numpy as np


class _KalmanTrack:
    """4-state constant-velocity Kalman: [cx, cy, vx, vy].

    Tracks pixel centroid position and velocity.
    """

    def __init__(self, cx: float, cy: float) -> None:
        self.kf = cv2.KalmanFilter(4, 2)
        dt = 1.0
        self.kf.transitionMatrix = np.array([
            [1, 0, dt,  0],
            [0, 1,  0, dt],
            [0, 0,  1,  0],
            [0, 0,  0,  1],
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)
        # Process noise: pos ±2 px, velocity ±4 px/frame
        self.kf.processNoiseCov = np.diag([4.0, 4.0, 16.0, 16.0]).astype(np.float32)
        # Measurement noise: YOLO centroid jitter ~10 px std → variance 100
        self.kf.measurementNoiseCov = np.diag([100.0, 100.0]).astype(np.float32)
        self.kf.statePost    = np.array([[cx], [cy], [0.0], [0.0]], dtype=np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 10.0

    def update(self, cx: float, cy: float) -> tuple[float, float, float, float]:
        """Predict then correct. Returns (smooth_cx, smooth_cy, vx, vy)."""
        self.kf.predict()
        self.kf.correct(np.array([[cx], [cy]], dtype=np.float32))
        s = self.kf.statePost.flatten()
        return float(s[0]), float(s[1]), float(s[2]), float(s[3])


class _DepthKalman:
    """2-state Kalman for Depth-Anything-V2 relative depth: [d, vd].

    d  ∈ [0, 1]: 0 = closest to camera, 1 = furthest.
    vd = depth rate of change per frame.
      vd < 0 → vehicle approaching (depth decreasing toward 0).
      vd > 0 → vehicle receding.

    Process noise is tight (0.0004 / 0.004) because depth changes smoothly
    between frames. Measurement noise (0.01) reflects DAv2-Small jitter.
    """

    def __init__(self, d0: float) -> None:
        self.kf = cv2.KalmanFilter(2, 1)
        self.kf.transitionMatrix  = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        self.kf.measurementMatrix = np.array([[1.0, 0.0]], dtype=np.float32)
        self.kf.processNoiseCov   = np.array([[4e-4, 0.0], [0.0, 4e-3]], dtype=np.float32)
        self.kf.measurementNoiseCov = np.array([[0.01]], dtype=np.float32)
        self.kf.statePost    = np.array([[d0], [0.0]], dtype=np.float32)
        self.kf.errorCovPost = np.eye(2, dtype=np.float32) * 0.1

    def update(self, d: float) -> tuple[float, float]:
        """Returns (smooth_depth, depth_velocity_per_frame)."""
        self.kf.predict()
        self.kf.correct(np.array([[d]], dtype=np.float32))
        s = self.kf.statePost.flatten()
        return float(s[0]), float(s[1])

    def predict_only(self) -> tuple[float, float]:
        """Use when depth reading is unavailable this frame (occlusion etc.)."""
        self.kf.predict()
        s = self.kf.statePost.flatten()
        return float(s[0]), float(s[1])


class TrajectoryBuilder:
    def __init__(self, fps: float, pixels_per_meter: float = 30.0):
        """
        fps              : video frame rate
        pixels_per_meter : calibration constant for absolute speed (km/h).
                           Relative speeds used for TTC are calibration-free.
        """
        self.fps = fps
        self.pixels_per_meter = pixels_per_meter
        self._tracks: dict[int, dict] = {}

    def update(
        self,
        frame_num: int,
        active_tracks: list[dict],
        depth_map: "np.ndarray | None" = None,
    ) -> None:
        """Call once per frame with confirmed tracks from VehicleTracker.

        depth_map : H×W float32 from DepthEstimator.estimate(), or None.
                    When provided each track gets depth_estimate and
                    depth_velocity populated; otherwise both are None.
        """
        for track in active_tracks:
            tid = track["track_id"]
            l, t, r, b = track["bbox_ltrb"]
            raw_cx = (l + r) / 2.0
            raw_cy = (t + b) / 2.0
            ts_ms  = int((frame_num / self.fps) * 1000)

            if tid not in self._tracks:
                kf  = _KalmanTrack(raw_cx, raw_cy)
                dkf = None
                if depth_map is not None:
                    from .depth import DepthEstimator  # lazy import — avoids heavy load
                    d0 = _median_bbox_depth(depth_map, l, t, r, b)
                    dkf = _DepthKalman(d0)

                self._tracks[tid] = {
                    "trajectory_id": str(uuid.uuid4()),
                    "vehicle_id":    tid,
                    "vehicle_class": track["class_name"],
                    "first_frame":   frame_num,
                    "last_frame":    frame_num,
                    "frames":        [],
                    "_kf":           kf,
                    "_dkf":          dkf,
                }

            entry = self._tracks[tid]
            entry["last_frame"] = frame_num

            cx, cy, vx, vy = entry["_kf"].update(raw_cx, raw_cy)

            pixel_speed = math.hypot(vx, vy)
            raw_speed   = pixel_speed / self.pixels_per_meter * self.fps * 3.6
            speed       = round(raw_speed, 1) if raw_speed <= 200.0 else 0.0

            depth_est = None
            depth_vel = None
            if depth_map is not None:
                raw_d = _median_bbox_depth(depth_map, l, t, r, b)
                if entry["_dkf"] is None:
                    entry["_dkf"] = _DepthKalman(raw_d)
                d_smooth, vd = entry["_dkf"].update(raw_d)
                depth_est = round(d_smooth, 4)
                depth_vel = round(vd, 5)
            elif entry["_dkf"] is not None:
                d_smooth, vd = entry["_dkf"].predict_only()
                depth_est = round(d_smooth, 4)
                depth_vel = round(vd, 5)

            entry["frames"].append({
                "frame_num":       frame_num,
                "timestamp_ms":    ts_ms,
                "bbox":            [l, t, r, b],
                "center":          [int(cx), int(cy)],
                "speed_estimate":  speed,
                "depth_estimate":  depth_est,
                "depth_velocity":  depth_vel,
            })

    def get_trajectories(self, video_id: str) -> list[dict]:
        """Return trajectory records matching the project JSON spec."""
        result = []
        for entry in self._tracks.values():
            traj = {k: v for k, v in entry.items() if not k.startswith("_")}
            traj["video_id"]    = video_id
            traj["frame_count"] = len(traj["frames"])
            result.append(traj)
        return result


def _median_bbox_depth(
    depth_map: "np.ndarray",
    l: int, t: int, r: int, b: int,
) -> float:
    """Median depth in the central 50% of a bounding box crop."""
    h, w = b - t, r - l
    cy, cx = (t + b) // 2, (l + r) // 2
    hh = max(h // 4, 1)
    hw = max(w // 4, 1)
    H, W = depth_map.shape[:2]
    y0, y1 = max(cy - hh, 0), min(cy + hh, H)
    x0, x1 = max(cx - hw, 0), min(cx + hw, W)
    crop = depth_map[y0:y1, x0:x1]
    if crop.size == 0:
        return float(depth_map[max(cy, 0), max(cx, 0)])
    return float(np.median(crop))
