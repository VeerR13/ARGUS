"""
boost_classes.py — Extract frames from existing videos, pseudo-label with
COCO-pretrained YOLOv8s, and retrain week2_retrained.pt with balanced classes.

Usage:
    python boost_classes.py
    python boost_classes.py --epochs 30 --device mps
"""

import argparse
import os
import sys
import glob
import shutil
from pathlib import Path
from collections import Counter

import cv2
import numpy as np

# COCO class IDs → our class IDs
COCO_TO_OURS = {2: 0, 3: 1, 5: 2, 7: 3}   # car, motorcycle, bus, truck
CLASS_NAMES   = ["car", "motorcycle", "bus", "truck"]
MINORITY_CLASSES = {1, 2, 3}  # motorcycle, bus, truck

DATASET_DIR   = Path("dataset")
TRAIN_IMG_DIR = DATASET_DIR / "train" / "images"
TRAIN_LBL_DIR = DATASET_DIR / "train" / "labels"
VAL_IMG_DIR   = DATASET_DIR / "val"   / "images"
VAL_LBL_DIR   = DATASET_DIR / "val"   / "labels"

VIDEO_DIRS = ["sample_videos", "unseen_test_videos"]
COCO_MODEL = "yolov8s.pt"         # COCO-pretrained, knows all 4 classes
CONF_THRESH = 0.45
MINORITY_CONF = 0.35              # lower threshold for rare classes
FRAME_STEP = 4                    # sample every 4th frame
VAL_EVERY  = 8                    # every 8th accepted frame goes to val


def extract_and_label(video_path: str, model) -> tuple[int, int]:
    """Extract frames, run inference, write YOLO labels. Returns (added_train, added_val)."""
    from ultralytics import YOLO

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [skip] Cannot open {video_path}")
        return 0, 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_stem   = Path(video_path).stem
    added_train, added_val = 0, 0
    frame_idx = 0
    accepted  = 0

    print(f"  Processing {Path(video_path).name} ({total_frames} frames, sampling every {FRAME_STEP})")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % FRAME_STEP != 0:
            continue

        img_h, img_w = frame.shape[:2]
        results = model(frame, conf=MINORITY_CONF, iou=0.4, verbose=False)[0]

        if results.boxes is None or len(results.boxes) == 0:
            continue

        # Build YOLO labels, only for our 4 classes
        lines = []
        has_minority = False
        for box in results.boxes:
            coco_cls = int(box.cls[0].item())
            our_cls  = COCO_TO_OURS.get(coco_cls)
            if our_cls is None:
                continue
            conf = float(box.conf[0].item())
            # Apply stricter threshold for cars to keep balance
            if our_cls == 0 and conf < CONF_THRESH:
                continue
            if our_cls in MINORITY_CLASSES:
                has_minority = True

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2 / img_w
            cy = (y1 + y2) / 2 / img_h
            bw = (x2 - x1) / img_w
            bh = (y2 - y1) / img_h
            lines.append(f"{our_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if not lines:
            continue

        # Prefer frames with minority classes; skip pure-car frames occasionally
        if not has_minority and accepted % 3 != 0:
            continue

        stem = f"boost_{video_stem}_f{frame_idx:06d}"

        # Alternate between train and val
        if accepted % VAL_EVERY == 0:
            img_dir = VAL_IMG_DIR
            lbl_dir = VAL_LBL_DIR
            added_val += 1
        else:
            img_dir = TRAIN_IMG_DIR
            lbl_dir = TRAIN_LBL_DIR
            added_train += 1

        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(img_dir / f"{stem}.jpg"), frame)
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines))
        accepted += 1

    cap.release()
    return added_train, added_val


def count_labels(label_dir: Path) -> Counter:
    counts = Counter()
    for f in label_dir.glob("*.txt"):
        for line in f.read_text().splitlines():
            parts = line.split()
            if parts:
                counts[int(parts[0])] += 1
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",  type=int,   default=30)
    parser.add_argument("--batch",   type=int,   default=16)
    parser.add_argument("--device",  type=str,   default="mps")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip frame extraction, go straight to retrain")
    args = parser.parse_args()

    from ultralytics import YOLO

    # ── Step 1: Pseudo-label existing videos ──────────────────────────────────
    if not args.skip_extract:
        print("\n=== Step 1: Pseudo-labeling with COCO-pretrained YOLOv8s ===")
        coco_model = YOLO(COCO_MODEL)

        total_train, total_val = 0, 0
        for vdir in VIDEO_DIRS:
            videos = glob.glob(f"{vdir}/*.mp4") + glob.glob(f"{vdir}/*.avi")
            for vpath in sorted(videos):
                t, v = extract_and_label(vpath, coco_model)
                total_train += t
                total_val   += v
                print(f"    → +{t} train, +{v} val frames")

        print(f"\n  Added {total_train} train + {total_val} val frames total")

    # ── Step 2: Class distribution after augmentation ─────────────────────────
    print("\n=== Step 2: Updated class distribution ===")
    counts = count_labels(TRAIN_LBL_DIR)
    for cls_id, name in enumerate(CLASS_NAMES):
        print(f"  {name:12s}: {counts.get(cls_id, 0)} annotations")

    # ── Step 3: Compute class weights ─────────────────────────────────────────
    total = sum(counts.values()) or 1
    cls_weights = []
    for cls_id in range(4):
        n = counts.get(cls_id, 1)
        cls_weights.append(round(total / (4 * n), 3))
    print(f"\n  Class weights: {dict(zip(CLASS_NAMES, cls_weights))}")

    # ── Step 4: Retrain ────────────────────────────────────────────────────────
    print("\n=== Step 3: Retraining week2_retrained.pt ===")
    model = YOLO("models/week2_retrained.pt")
    results = model.train(
        data        = str(DATASET_DIR / "data.yaml"),
        epochs      = args.epochs,
        batch       = args.batch,
        imgsz       = 640,
        device      = args.device,
        lr0         = 3e-4,            # lower LR for fine-tune
        lrf         = 0.01,
        warmup_epochs = 3,
        cls         = 0.7,             # up classification loss weight
        patience    = 15,
        save        = True,
        project     = "runs/boost",
        name        = "week3_boosted",
        exist_ok    = True,
    )

    # ── Step 5: Save ──────────────────────────────────────────────────────────
    best_pt = Path("runs/boost/week3_boosted/weights/best.pt")
    if best_pt.exists():
        shutil.copy(best_pt, "models/week3_boosted.pt")
        print(f"\n  Saved: models/week3_boosted.pt")
    else:
        print("\n  WARNING: best.pt not found, check runs/boost/week3_boosted/")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
