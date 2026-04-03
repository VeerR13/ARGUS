"""
week3_annotate.py — Extract exactly 500 frames for CVAT human annotation.

Priority order:
  - night_road unseen clip  — every frame  (all ~89 frames)
  - highway unseen clip     — every 3rd frame (~90 frames)
  - accident clips unseen   — every 2nd frame (~112 each)
  - dashcam unseen          — every 5th frame (~54 frames)
  Fill to exactly 500 from remaining dashcam/highway frames if short.

Output:
  annotation_tasks/frames_to_annotate/  frame_00001.jpg → frame_00500.jpg
  annotation_tasks/cvat_import.zip      (frames + task.json)
  annotation_tasks/frame_index.json     original source → new name

Usage:
    python week3_annotate.py
    python week3_annotate.py --unseen unseen_test_videos/ --out annotation_tasks/
"""

import argparse
import json
import os
import shutil
import sys
import zipfile
from collections import OrderedDict

import cv2
import numpy as np

TARGET = 500

SAMPLING = {
    # (scenario_keyword, sample_every)
    "night": 1,      # every frame
    "highway": 3,
    "accident": 2,
    "dashcam": 5,
}

CLASS_LABELS = [
    {"name": "car",        "attributes": []},
    {"name": "truck",      "attributes": []},
    {"name": "motorcycle", "attributes": []},
    {"name": "bus",        "attributes": []},
    {"name": "vehicle",    "attributes": []},   # catch-all for ambiguous
]


def _scenario(fname: str) -> str:
    fl = fname.lower()
    if "night" in fl:  return "night"
    if "highway" in fl: return "highway"
    if "accident" in fl or "clip" in fl: return "accident"
    return "dashcam"


def _sample_video(video_path: str, step: int, max_frames: int = None) -> list:
    """Return list of (frame_index, frame_bgr) sampled every `step` frames."""
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    fn = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fn % step == 0:
            frames.append((fn, frame.copy()))
            if max_frames and len(frames) >= max_frames:
                break
        fn += 1
    cap.release()
    return frames


def extract_frames(unseen_dir: str, out_dir: str) -> dict:
    """
    Extract up to TARGET frames from unseen_test_videos/.
    Returns index mapping: output_name → {source_video, source_frame}
    """
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(unseen_dir):
        print(f"  WARNING: {unseen_dir} not found. Run week3_retrain.py first.")
        return {}

    clips = sorted(f for f in os.listdir(unseen_dir) if f.endswith(".mp4"))
    if not clips:
        print(f"  WARNING: no .mp4 files in {unseen_dir}")
        return {}

    # Sort clips by priority: night first, then accident, highway, dashcam
    priority = {"night": 0, "accident": 1, "highway": 2, "dashcam": 3}
    clips.sort(key=lambda f: priority.get(_scenario(f), 9))

    collected = []   # list of (src_video, src_frame, frame_bgr)
    remaining = TARGET

    print(f"\n  Extracting frames from {len(clips)} clips → target {TARGET}:")
    for clip in clips:
        if remaining <= 0:
            break
        cpath = os.path.join(unseen_dir, clip)
        scen  = _scenario(clip)
        step  = SAMPLING.get(scen, 5)

        frames = _sample_video(cpath, step, max_frames=remaining)
        for fn, frm in frames:
            collected.append((clip, fn, frm))
        remaining -= len(frames)
        print(f"    {clip:<40}  scenario={scen:<8}  step=1/{step}  frames={len(frames)}")

    # If still short, go back through dashcam/highway with step=1 to fill up
    if len(collected) < TARGET:
        for clip in clips:
            if len(collected) >= TARGET:
                break
            scen = _scenario(clip)
            if scen not in ("dashcam", "highway"):
                continue
            cpath = os.path.join(unseen_dir, clip)
            already_fn = {fn for (c, fn, _) in collected if c == clip}
            cap = cv2.VideoCapture(cpath)
            fn  = 0
            while len(collected) < TARGET:
                ret, frame = cap.read()
                if not ret:
                    break
                if fn not in already_fn:
                    collected.append((clip, fn, frame.copy()))
                fn += 1
            cap.release()

    # Trim to exactly TARGET
    collected = collected[:TARGET]

    # Write frames with sequential naming
    index = {}
    for seq, (src_video, src_fn, frame) in enumerate(collected, start=1):
        out_name = f"frame_{seq:05d}.jpg"
        out_path = os.path.join(out_dir, out_name)
        cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        index[out_name] = {
            "source_video": src_video,
            "source_frame": src_fn,
            "scenario":     _scenario(src_video),
            "seq":          seq,
        }

    print(f"\n  Saved {len(collected)} frames to {out_dir}/")
    return index


def write_cvat_package(frames_dir: str, index: dict, out_dir: str) -> str:
    """
    Create CVAT import package:
      - annotation_tasks/cvat_import.zip
        └── images/frame_00001.jpg ... frame_00500.jpg
        └── task.json

    task.json format is CVAT-compatible task creation metadata.
    """
    os.makedirs(out_dir, exist_ok=True)
    zip_path = os.path.join(out_dir, "cvat_import.zip")

    if os.path.exists(zip_path):
        print(f"  [CACHE] {zip_path} already exists — skipping zip creation.")
        return zip_path

    task_json = {
        "name":    "week2_vehicle_validation",
        "created": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "labels":  CLASS_LABELS,
        "frame_count": len(index),
        "frame_list":  sorted(index.keys()),
        "annotation_instructions": {
            "draw":    "Tight bounding box around every clearly visible vehicle",
            "include": ["cars", "trucks", "motorcycles", "buses", "vans"],
            "exclude": ["vehicles > 80% occluded", "vehicles in mirrors"],
            "class":   "Use 'vehicle' if unsure of exact type",
            "quality": "Boxes should touch vehicle edges within ~10px",
        },
        "export_format": "YOLO 1.1",
        "scenarios": {scen: sum(1 for v in index.values() if v["scenario"] == scen)
                      for scen in ("night", "highway", "accident", "dashcam")},
        "estimated_annotation_time_hours": round(len(index) / 180, 1),
    }

    print(f"  Creating {zip_path} ...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add task.json
        zf.writestr("task.json", json.dumps(task_json, indent=2))

        # Add all frames under images/
        frames = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
        for fname in frames:
            fpath = os.path.join(frames_dir, fname)
            zf.write(fpath, arcname=os.path.join("images", fname))
            if int(fname.split("_")[1].split(".")[0]) % 100 == 0:
                print(f"    zipped {fname}", flush=True)

    size_mb = os.path.getsize(zip_path) / 1e6
    print(f"  → {zip_path}  ({size_mb:.1f} MB  {len(frames)} frames)")
    return zip_path


def print_cvat_instructions(zip_path: str, n_frames: int, task_json_scenarios: dict):
    """Print step-by-step CVAT setup guide."""
    print("\n" + "=" * 65)
    print("  CVAT ANNOTATION SETUP INSTRUCTIONS")
    print("=" * 65)
    print(f"""
  Package ready: {zip_path}
  Frames:        {n_frames}
  Estimated time: ~{round(n_frames/180, 1)} hours  ({n_frames} frames ÷ ~3 min/10 frames)

  ── IMPORT STEPS ─────────────────────────────────────────────
  1. Open CVAT (app.cvat.ai or self-hosted instance)
  2. Click "Projects" → "Create new project"
     Name: "ARGUS Week 2 Validation"
  3. Add labels:
       car  truck  motorcycle  bus  vehicle
  4. Click "Tasks" → "Create new task"
     Name: "week2_vehicle_validation"
  5. Under "Select files" → drag in cvat_import.zip
     (CVAT will extract images automatically)
  6. Set annotation format: YOLO 1.1
  7. Click "Submit & Open"

  ── WHAT TO ANNOTATE ─────────────────────────────────────────
  - Draw TIGHT bounding boxes around every clearly visible vehicle
  - Include: cars, trucks, motorcycles, buses, vans
  - Exclude: vehicles >80% occluded, vehicles visible only in mirrors
  - If unsure of type → use "vehicle" class
  - Boxes should touch vehicle edges within ~10 pixels

  ── EXPORT ───────────────────────────────────────────────────
  After annotating all {n_frames} frames:
  1. Task → Actions → Export annotations
  2. Format: YOLO 1.1
  3. Save to: annotation_tasks/cvat_export/

  ── THEN RUN ─────────────────────────────────────────────────
  python eval_real.py \\
    --annotations annotation_tasks/cvat_export/ \\
    --footage unseen_test_videos/ \\
    --model models/week2_retrained.pt \\
    --output reports_validated/

  Scenarios in this batch:
""")
    for scen, cnt in task_json_scenarios.items():
        print(f"    {scen:<12}  {cnt} frames")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Extract 500 frames for CVAT annotation")
    parser.add_argument("--unseen", default="unseen_test_videos")
    parser.add_argument("--out",    default="annotation_tasks")
    args = parser.parse_args()

    frames_dir = os.path.join(args.out, "frames_to_annotate")

    print("\n" + "=" * 65)
    print("  Week 3 — Annotation Frame Extraction")
    print("=" * 65)

    # Check if frames already extracted
    existing = [f for f in os.listdir(frames_dir) if f.endswith(".jpg")] \
               if os.path.isdir(frames_dir) else []

    if len(existing) >= TARGET:
        print(f"  [CACHE] {len(existing)} frames already in {frames_dir}")
        index_path = os.path.join(args.out, "frame_index.json")
        index = json.load(open(index_path)) if os.path.exists(index_path) else {}
    else:
        index = extract_frames(args.unseen, frames_dir)
        index_path = os.path.join(args.out, "frame_index.json")
        os.makedirs(args.out, exist_ok=True)
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
        print(f"  saved → {index_path}")

    # Scenario counts
    scenarios = {}
    for v in index.values():
        scen = v.get("scenario", "unknown")
        scenarios[scen] = scenarios.get(scen, 0) + 1

    # Create CVAT zip
    zip_path = write_cvat_package(frames_dir, index, args.out)

    # Print instructions
    print_cvat_instructions(zip_path, len(index), scenarios)

    print("=" * 65)
    print(f"  {len(index)} frames ready for annotation")
    print(f"  {zip_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
