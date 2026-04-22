"""
build_overhead_dataset.py — Build a clean overhead/CCTV-perspective dataset.

Sources:
  1. YouTube videos downloaded to yt_videos/ (overhead traffic cams)
  2. COCO images already in dataset/train/ (coco_* prefix)
  3. VisDrone annotations if available in overhead_data/VisDrone/

Clears dashcam frames from previous runs, keeps COCO, adds overhead pseudo-labels,
then retrains from yolov8s.pt (stronger base than week2_retrained which learned dashcam).

Usage:
    python build_overhead_dataset.py
    python build_overhead_dataset.py --epochs 50 --device mps --skip-extract
"""

import argparse
import glob
import os
import shutil
from pathlib import Path
from collections import Counter

import cv2
import numpy as np

# COCO pretrained class IDs → our IDs
COCO_TO_OURS = {2: 0, 3: 1, 5: 2, 7: 3}   # car, motorcycle, bus, truck
CLASS_NAMES   = ["car", "motorcycle", "bus", "truck"]

# VisDrone class IDs → our IDs (motor=9, car=3, van=4, truck=5, bus=8)
VISDRONE_TO_OURS = {3: 0, 4: 0, 5: 3, 8: 2, 9: 1}

DATASET_DIR   = Path("dataset")
TRAIN_IMG     = DATASET_DIR / "train" / "images"
TRAIN_LBL     = DATASET_DIR / "train" / "labels"
VAL_IMG       = DATASET_DIR / "val"   / "images"
VAL_LBL       = DATASET_DIR / "val"   / "labels"

CONF_CAR      = 0.50
CONF_MINORITY = 0.35
FRAME_STEP    = 5       # sample every 5th frame from video
VAL_EVERY     = 8


def purge_dashcam_frames():
    """Remove frames extracted from the old dashcam videos (boost_* prefix)."""
    removed = 0
    for d in [TRAIN_IMG, TRAIN_LBL, VAL_IMG, VAL_LBL]:
        for f in d.glob("boost_*"):
            f.unlink()
            removed += 1
    print(f"  Removed {removed} old dashcam-extracted files")


def load_visdrone(visdrone_root: Path, model) -> tuple[int, int]:
    """
    Import VisDrone detection annotations directly (ground truth, no pseudo-labeling).
    VisDrone annotation format: <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,
                                 <score>,<category>,<truncation>,<occlusion>
    """
    ann_dirs = list(visdrone_root.rglob("annotations")) + list(visdrone_root.rglob("Annotations"))
    if not ann_dirs:
        print("  No VisDrone annotation dirs found, skipping.")
        return 0, 0

    added_train = added_val = 0
    ann_dir = ann_dirs[0]
    img_dir  = ann_dir.parent / "images" if (ann_dir.parent / "images").exists() else ann_dir.parent / "sequences"

    for txt_path in sorted(ann_dir.glob("*.txt"))[:3000]:
        stem = txt_path.stem
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            cand = img_dir / (stem + ext)
            if cand.exists():
                img_path = cand
                break
        if img_path is None:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        lines = []
        for line in txt_path.read_text().splitlines():
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            try:
                x, y, bw, bh = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                cat = int(parts[5])
            except ValueError:
                continue
            our_cls = VISDRONE_TO_OURS.get(cat)
            if our_cls is None:
                continue
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            if nw <= 0 or nh <= 0:
                continue
            lines.append(f"{our_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        if not lines:
            continue

        is_val = (added_train + added_val) % VAL_EVERY == 0
        out_img = (VAL_IMG if is_val else TRAIN_IMG) / f"visdrone_{stem}.jpg"
        out_lbl = (VAL_LBL if is_val else TRAIN_LBL) / f"visdrone_{stem}.txt"

        for d in [TRAIN_IMG, TRAIN_LBL, VAL_IMG, VAL_LBL]:
            d.mkdir(parents=True, exist_ok=True)

        shutil.copy(img_path, out_img)
        out_lbl.write_text("\n".join(lines))

        if is_val:
            added_val += 1
        else:
            added_train += 1

    print(f"  VisDrone: +{added_train} train, +{added_val} val")
    return added_train, added_val


def extract_and_pseudolabel(video_path: str, model) -> tuple[int, int]:
    """Extract frames from overhead video and pseudo-label with COCO YOLOv8s."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, 0

    stem = Path(video_path).stem
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    # Only use first 8 minutes max to avoid bloat
    max_frames = int(fps * 60 * 8)

    added_train = added_val = 0
    frame_idx = 0
    accepted  = 0

    print(f"  {Path(video_path).name} ({total} frames, using up to {max_frames})")

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % FRAME_STEP != 0:
            continue

        img_h, img_w = frame.shape[:2]
        results = model(frame, conf=CONF_MINORITY, iou=0.4, verbose=False)[0]

        if results.boxes is None or len(results.boxes) == 0:
            continue

        lines = []
        has_minority = False
        for box in results.boxes:
            coco_cls = int(box.cls[0].item())
            our_cls  = COCO_TO_OURS.get(coco_cls)
            if our_cls is None:
                continue
            conf = float(box.conf[0].item())
            if our_cls == 0 and conf < CONF_CAR:
                continue
            if our_cls in {1, 2, 3}:
                has_minority = True

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2 / img_w
            cy = (y1 + y2) / 2 / img_h
            bw = (x2 - x1) / img_w
            bh = (y2 - y1) / img_h
            lines.append(f"{our_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if not lines:
            continue
        # Downsample pure-car frames
        if not has_minority and accepted % 4 != 0:
            continue

        is_val = accepted % VAL_EVERY == 0
        out_img = (VAL_IMG if is_val else TRAIN_IMG) / f"yt_{stem}_f{frame_idx:06d}.jpg"
        out_lbl = (VAL_LBL if is_val else TRAIN_LBL) / f"yt_{stem}_f{frame_idx:06d}.txt"

        for d in [TRAIN_IMG, TRAIN_LBL, VAL_IMG, VAL_LBL]:
            d.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(out_img), frame)
        out_lbl.write_text("\n".join(lines))
        accepted += 1
        if is_val:
            added_val += 1
        else:
            added_train += 1

    cap.release()
    return added_train, added_val


def count_labels(lbl_dir: Path) -> Counter:
    counts = Counter()
    for f in lbl_dir.glob("*.txt"):
        for line in f.read_text().splitlines():
            p = line.split()
            if p:
                counts[int(p[0])] += 1
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",       type=int, default=50)
    parser.add_argument("--batch",        type=int, default=16)
    parser.add_argument("--device",       type=str, default="mps")
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()

    from ultralytics import YOLO

    # ── 1. Remove old dashcam pseudo-labels ──────────────────────────────────
    print("\n=== Step 1: Purging old dashcam frames ===")
    purge_dashcam_frames()

    if not args.skip_extract:
        coco_model = YOLO("yolov8s.pt")   # COCO pretrained — knows all classes well

        # ── 2. VisDrone (if downloaded) ──────────────────────────────────────
        visdrone_root = Path("overhead_data/VisDrone")
        if visdrone_root.exists():
            print("\n=== Step 2: Importing VisDrone annotations ===")
            load_visdrone(visdrone_root, coco_model)
        else:
            print("\n=== Step 2: VisDrone not found, skipping ===")

        # ── 3. YouTube overhead videos ───────────────────────────────────────
        print("\n=== Step 3: Pseudo-labeling YouTube overhead videos ===")
        videos = sorted(glob.glob("yt_videos/*.mp4"))
        t_total = v_total = 0
        for vpath in videos:
            t, v = extract_and_pseudolabel(vpath, coco_model)
            t_total += t; v_total += v
            print(f"    → +{t} train, +{v} val")
        print(f"  Total from YouTube: +{t_total} train, +{v_total} val")

    # ── 4. Class distribution ────────────────────────────────────────────────
    print("\n=== Step 4: Dataset class distribution ===")
    counts = count_labels(TRAIN_LBL)
    total  = sum(counts.values()) or 1
    for cls_id, name in enumerate(CLASS_NAMES):
        n = counts.get(cls_id, 0)
        print(f"  {name:12s}: {n:4d} annotations  ({n/total*100:.1f}%)")

    # ── 5. Retrain from yolov8s.pt (NOT the dashcam-biased week2_retrained) ─
    print("\n=== Step 5: Training from yolov8s.pt (COCO pretrained) ===")
    model = YOLO("yolov8s.pt")
    model.train(
        data          = str(DATASET_DIR / "data.yaml"),
        epochs        = args.epochs,
        batch         = args.batch,
        imgsz         = 640,
        device        = args.device,
        lr0           = 1e-3,
        lrf           = 0.01,
        warmup_epochs = 5,
        cls           = 1.0,
        patience      = 20,
        save          = True,
        project       = "runs/overhead",
        name          = "week4_overhead",
        exist_ok      = True,
    )

    best = Path("runs/overhead/week4_overhead/weights/best.pt")
    if best.exists():
        shutil.copy(best, "models/week4_overhead.pt")
        print("\n  Saved: models/week4_overhead.pt")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
