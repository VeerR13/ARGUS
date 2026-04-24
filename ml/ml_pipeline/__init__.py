"""
ml_pipeline — ARGUS core module.

Public interface (used by Videh's Celery worker):

    from ml_pipeline import analyze_video
    results = analyze_video(video_path, progress_callback=lambda pct: ...)
    # returns {"trajectories": [...], "incidents": [...]}
"""

import uuid
import cv2

from .detection import VehicleDetector
from .tracking import VehicleTracker
from .trajectory import TrajectoryBuilder
from .interaction import detect_incidents


def analyze_video(
    video_path: str,
    progress_callback=None,
    model_path: str = "rfdetr-base",  # "rfdetr-large" for higher AP, ~2× slower
    confidence: float = 0.40,
    pixels_per_meter: float = 30.0,
) -> dict:
    """
    Analyze a traffic video end-to-end.

    Args:
        video_path        : Path to video file (MP4, AVI, MOV)
        progress_callback : Optional function(percent: int) called each frame
        model_path        : YOLOv8 weights file
        confidence        : Detection confidence threshold
        pixels_per_meter  : Rough scale factor for speed estimation

    Returns:
        {
            "trajectories": [list of trajectory dicts],
            "incidents":    [list of incident dicts]   # populated Week 2+
        }
    """
    video_id = str(uuid.uuid4())

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = VehicleDetector(model_path=model_path, confidence=confidence)
    tracker = VehicleTracker()
    traj_builder = TrajectoryBuilder(fps=fps, pixels_per_meter=pixels_per_meter)

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        active_tracks = tracker.update(detections, frame)
        traj_builder.update(frame_num, active_tracks)

        frame_num += 1
        if progress_callback and total_frames > 0:
            progress_callback(int(frame_num / total_frames * 100))

    cap.release()

    trajectories = traj_builder.get_trajectories(video_id)
    incidents = detect_incidents(
        trajectories=trajectories,
        fps=fps,
        pixels_per_meter=pixels_per_meter,
        video_id=video_id,
        frame_width=frame_width,
        frame_height=frame_height,
    )

    return {
        "trajectories": trajectories,
        "incidents":    incidents,
    }
