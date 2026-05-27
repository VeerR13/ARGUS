"""
ml_pipeline — ARGUS core module.

Single-function public interface:

    from ml.ml_pipeline import analyze_video
    result = analyze_video("path/to/dashcam.mp4")
    # {"trajectories": [...], "incidents": [...]}

For depth-enabled dual-signal TTC (adds ~25 ms/frame on GPU):
    result = analyze_video("dashcam.mp4", use_depth=True)

Individual components are importable directly:
    from ml.ml_pipeline import VehicleDetector, VehicleTracker, TrajectoryBuilder
    from ml.ml_pipeline import detect_incidents
    from ml.ml_pipeline.depth import DepthEstimator  # heavy transformers dep — import on demand
"""

from __future__ import annotations

import uuid
import cv2

from .detection import VehicleDetector
from .tracking import VehicleTracker
from .trajectory import TrajectoryBuilder
from .interaction import detect_incidents

__all__ = [
    "analyze_video",
    "VehicleDetector",
    "VehicleTracker",
    "TrajectoryBuilder",
    "detect_incidents",
]


def analyze_video(
    video_path: str,
    progress_callback=None,
    model_path: str = "yolo12x.pt",
    confidence: float = 0.40,
    pixels_per_meter: float = 30.0,
    use_depth: bool = False,
) -> dict:
    """
    Analyse a dashcam or CCTV video end-to-end.

    Args:
        video_path        : Path to video file (MP4, AVI, MOV)
        progress_callback : Optional function(percent: int) called each frame
        model_path        : YOLO12x weights — pretrained yolo12x.pt or finetuned
                            argus_s3.pt for best dashcam/truck performance
        confidence        : Detection confidence threshold (0.35–0.45 for dashcam)
        pixels_per_meter  : Scale factor for absolute speed estimates (must be > 0).
                            TTC is calibration-free; only km/h display is affected.
        use_depth         : Enable Depth-Anything-V2-Small monocular depth for
                            dual-signal TTC (pixel convergence + depth closing rate).
                            Adds ~25 ms/frame on GPU; requires transformers>=4.38.

    Returns:
        {
            "trajectories": [list of trajectory dicts],
            "incidents":    [list of incident dicts]
        }
    """
    if pixels_per_meter <= 0:
        raise ValueError(f"pixels_per_meter must be > 0, got {pixels_per_meter}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_id     = str(uuid.uuid4())

    detector        = VehicleDetector(model_path=model_path, confidence=confidence)
    tracker         = VehicleTracker()
    traj_builder    = TrajectoryBuilder(fps=fps, pixels_per_meter=pixels_per_meter)
    depth_estimator = None

    if use_depth:
        from .depth import DepthEstimator
        depth_estimator = DepthEstimator()

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections    = detector.detect(frame)
        active_tracks = tracker.update(detections, frame)

        depth_map = None
        if depth_estimator is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            depth_map = depth_estimator.estimate(frame_rgb)

        traj_builder.update(frame_num, active_tracks, depth_map=depth_map)

        frame_num += 1
        if progress_callback and total_frames > 0:
            progress_callback(int(frame_num / total_frames * 100))

    cap.release()

    trajectories = traj_builder.get_trajectories(video_id)
    incidents    = detect_incidents(
        trajectories=trajectories,
        fps=fps,
        pixels_per_meter=pixels_per_meter,
        video_id=video_id,
        frame_width=frame_width,
        frame_height=frame_height,
    )

    return {"trajectories": trajectories, "incidents": incidents}
