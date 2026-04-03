"""
Week 1 Test Script — YOLOv8 Detection + DeepSORT Tracking + Trajectory Extraction

Usage:
    python week1_test.py --video sample_videos/your_clip.mp4

Outputs:
    output_annotated.mp4      — video with bounding boxes, track IDs, and class labels
    output_trajectories.json  — trajectory records in the project JSON spec format

Optional flags:
    --model yolov8s.pt        — model file (yolov8n.pt is faster, yolov8m.pt more accurate)
    --conf  0.4               — detection confidence threshold (lower = more detections)
    --pixels-per-meter 30     — rough scale factor; adjust if speed estimates look wrong
                                 (highway dashcam: ~15-25 | intersection top-down: ~40-60)
    --output-video  FILE
    --output-json   FILE
"""

import argparse
import json
import os
import sys
import uuid

import cv2

sys.path.insert(0, os.path.dirname(__file__))

from ml_pipeline.detection import VehicleDetector
from ml_pipeline.tracking import VehicleTracker
from ml_pipeline.trajectory import TrajectoryBuilder


# One distinct colour per track ID (cycles after 12)
_PALETTE = [
    (255,  56,  56), (255, 157,  51), ( 77, 255,  83), ( 56, 182, 255),
    (255,  51, 255), (  0, 204, 204), (255, 204,  51), (204,  51,  51),
    ( 51, 204,  51), ( 51,  51, 204), (204, 153,   0), (153,   0, 204),
]

def _color(track_id: int):
    return _PALETTE[track_id % len(_PALETTE)]


def draw_tracks(frame, active_tracks: list[dict]):
    for track in active_tracks:
        tid = track["track_id"]
        l, t, r, b = track["bbox_ltrb"]
        color = _color(tid)

        cv2.rectangle(frame, (l, t), (r, b), color, 2)

        label = f"ID:{tid}  {track['class_name']}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (l, t - th - baseline - 6), (l + tw + 4, t), color, -1)
        cv2.putText(frame, label, (l + 2, t - baseline - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def main():
    parser = argparse.ArgumentParser(description="Week 1: detection + tracking + trajectory extraction")
    parser.add_argument("--video",            required=True,             help="Input video path")
    parser.add_argument("--model",            default="yolov8s.pt",      help="YOLOv8 weights")
    parser.add_argument("--conf",             type=float, default=0.4,   help="Detection confidence")
    parser.add_argument("--pixels-per-meter", type=float, default=30.0,  help="Scale for speed estimates")
    parser.add_argument("--output-video",     default="output_annotated.mp4")
    parser.add_argument("--output-json",      default="output_trajectories.json")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"ERROR: video not found: {args.video}")
        sys.exit(1)

    # ── Open video ────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: cannot open video: {args.video}")
        sys.exit(1)

    fps           = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width         = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n{'='*60}")
    print(f"  ARGUS — Week 1 Test")
    print(f"{'='*60}")
    print(f"  Video   : {args.video}")
    print(f"  Size    : {width}x{height} @ {fps:.1f} fps  ({total_frames} frames)")
    print(f"  Model   : {args.model}  (conf={args.conf})")
    print(f"  px/m    : {args.pixels_per_meter}  (speed scale factor)")
    print(f"{'='*60}\n")

    # ── Writer ────────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output_video, fourcc, fps, (width, height))

    # ── Pipeline components ───────────────────────────────────────────
    detector     = VehicleDetector(model_path=args.model, confidence=args.conf)
    tracker      = VehicleTracker()
    traj_builder = TrajectoryBuilder(fps=fps, pixels_per_meter=args.pixels_per_meter)
    video_id     = str(uuid.uuid4())

    # ── Main loop ─────────────────────────────────────────────────────
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections   = detector.detect(frame)
        active_tracks = tracker.update(detections, frame)
        traj_builder.update(frame_num, active_tracks)

        # Annotate
        annotated = draw_tracks(frame.copy(), active_tracks)
        progress  = int(frame_num / total_frames * 100) if total_frames > 0 else 0
        cv2.putText(annotated,
                    f"Frame {frame_num}  |  Vehicles: {len(active_tracks)}  |  {progress}%",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 80), 2, cv2.LINE_AA)
        writer.write(annotated)

        if frame_num % 100 == 0:
            print(f"  [{progress:3d}%]  frame {frame_num:5d}/{total_frames}  —  {len(active_tracks)} vehicles")

        frame_num += 1

    cap.release()
    writer.release()

    # ── Save JSON ─────────────────────────────────────────────────────
    trajectories = traj_builder.get_trajectories(video_id)
    output = {
        "video_id":       video_id,
        "video_path":     args.video,
        "fps":            fps,
        "total_frames":   frame_num,
        "trajectories":   trajectories,
        "incidents":      [],
    }
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"  Annotated video  → {args.output_video}")
    print(f"  Trajectories JSON → {args.output_json}")
    print(f"  Frames processed : {frame_num}")
    print(f"  Vehicles tracked : {len(trajectories)}")
    print(f"{'='*60}")

    print("\n  Vehicle breakdown:")
    by_class: dict[str, int] = {}
    for t in trajectories:
        by_class[t["vehicle_class"]] = by_class.get(t["vehicle_class"], 0) + 1
        print(f"    ID {t['vehicle_id']:3d}  ({t['vehicle_class']:12s})  "
              f"frames {t['first_frame']:5d}–{t['last_frame']:5d}  "
              f"({t['frame_count']} frames)")

    print(f"\n  Class totals: { {k: v for k, v in sorted(by_class.items())} }")
    print()


if __name__ == "__main__":
    main()
