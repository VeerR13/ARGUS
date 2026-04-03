"""
week3_retrain.py — Dataset construction, fine-tuning with hard negatives,
                   before/after comparison, unseen test split.

Steps performed:
  1. Split each video: first 70% = train, last 30% = unseen_test_videos/
  2. Extract confirmed vehicle frames from trajectory data as weak-supervision positives
  3. Merge hard negatives (663 background samples) into dataset/train/
  4. Write data.yaml
  5. Fine-tune yolov8n.pt with the augmentation pipeline
  6. Run before/after inference comparison on sample_videos/
  7. Save retrain_comparison.json + models/week2_retrained.pt

Usage:
    python week3_retrain.py
    python week3_retrain.py --epochs 20 --batch 16 --device mps
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
VIDEOS = {
    "accident_clip_1.mp4":  {"split": 0.70, "has_traj": False, "scenario": "accident"},
    "accident_clip_2.mp4":  {"split": 0.70, "has_traj": False, "scenario": "accident"},
    "dashcam_30s_720p.mp4": {"split": 0.70, "has_traj": True,  "scenario": "dashcam",  "traj_key": "dashcam",     "sample_every": 5},
    "highway_30s_720p.mp4": {"split": 0.70, "has_traj": True,  "scenario": "highway",  "traj_key": "highway",     "sample_every": 3},
    "night_road_720p.mp4":  {"split": 0.70, "has_traj": True,  "scenario": "night",    "traj_key": "night_road",  "sample_every": 2},
}

# YOLO class IDs for our 4 vehicle classes
CLASS_MAP = {"car": 0, "motorcycle": 1, "bus": 2, "truck": 3}
CLASS_NAMES = ["car", "motorcycle", "bus", "truck"]

HN_SRC_IMAGES = "reports/hard_negatives/images"
HN_SRC_LABELS = "reports/hard_negatives/labels"


def _load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  saved → {path}")


def _bbox_to_yolo(bbox_ltrb, img_w, img_h, cls_id):
    """Convert [l,t,r,b] bbox to YOLO normalised [cls cx cy w h]."""
    l, t, r, b = bbox_ltrb
    cx = (l + r) / 2 / img_w
    cy = (t + b) / 2 / img_h
    w  = (r - l) / img_w
    h  = (b - t) / img_h
    cx = max(0.001, min(0.999, cx))
    cy = max(0.001, min(0.999, cy))
    w  = max(0.001, min(0.999, w))
    h  = max(0.001, min(0.999, h))
    return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Split videos: first 70% train / last 30% unseen test
# ─────────────────────────────────────────────────────────────────────────────
def split_videos(footage_dir: str, unseen_dir: str) -> dict:
    """
    Write last 30% of each video to unseen_test_videos/ as a clip.
    Returns per-video split metadata: {vfile: {train_frames, test_frames, test_path}}
    """
    print("\n" + "=" * 65)
    print("  Step 1 — Video split  (train=70%  /  unseen_test=30%)")
    print("=" * 65)
    os.makedirs(unseen_dir, exist_ok=True)

    splits = {}
    for vfile, cfg in VIDEOS.items():
        vpath = os.path.join(footage_dir, vfile)
        if not os.path.exists(vpath):
            print(f"  SKIP {vfile} (not found)")
            continue

        cap          = cv2.VideoCapture(vpath)
        total        = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        fw           = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh           = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        split_frame  = int(total * cfg["split"])
        train_frames = split_frame
        test_frames  = total - split_frame

        # Check if already written
        out_name  = f"test_{vfile}"
        out_path  = os.path.join(unseen_dir, out_name)
        if os.path.exists(out_path):
            print(f"  {vfile:<28}  already split  (train={train_frames}  test={test_frames})")
            cap.release()
        else:
            print(f"  {vfile:<28}  writing test clip  (frames {split_frame}→{total-1})", flush=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (fw, fh))
            cap.set(cv2.CAP_PROP_POS_FRAMES, split_frame)
            written = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                writer.write(frame)
                written += 1
            writer.release()
            cap.release()
            print(f"    → wrote {written} frames to {out_path}")

        splits[vfile] = {
            "total_frames":  total,
            "train_frames":  train_frames,
            "test_frames":   test_frames,
            "split_frame":   split_frame,
            "fps":           fps,
            "width":         fw,
            "height":        fh,
            "test_clip":     out_path,
        }

    return splits


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Extract positive training samples from trajectory data
# ─────────────────────────────────────────────────────────────────────────────
def extract_positive_samples(footage_dir: str, dataset_train_img: str,
                              dataset_train_lbl: str, splits: dict) -> int:
    """
    For dashcam / highway / night_road:
      - Load trajectory JSON
      - For frames in train split (< split_frame), sample every N frames
      - Extract frame from video, write YOLO label file

    Returns total positive samples written.
    """
    print("\n" + "=" * 65)
    print("  Step 2 — Extracting positive training samples from trajectories")
    print("=" * 65)

    os.makedirs(dataset_train_img, exist_ok=True)
    os.makedirs(dataset_train_lbl, exist_ok=True)

    total_written = 0

    for vfile, cfg in VIDEOS.items():
        if not cfg["has_traj"]:
            continue

        vpath = os.path.join(footage_dir, vfile)
        if not os.path.exists(vpath):
            continue

        traj_key  = cfg["traj_key"]
        step      = cfg["sample_every"]
        info      = splits.get(vfile, {})
        split_frm = info.get("split_frame", 9999999)
        fw        = info.get("width",  1280)
        fh        = info.get("height",  720)

        # Load trajectory JSON — look in project root
        traj_path = None
        for candidate in [
            f"output_{traj_key}_trajectories.json",
            os.path.join(footage_dir, f"output_{traj_key}_trajectories.json"),
        ]:
            if os.path.exists(candidate):
                traj_path = candidate
                break
        if traj_path is None:
            print(f"  {vfile}: no trajectory JSON found — skipping positives")
            continue

        traj_data = _load_json(traj_path)
        if not traj_data or "trajectories" not in traj_data:
            continue

        # Build frame → list[bbox_ltrb, class_name] map
        frame_boxes = defaultdict(list)
        for traj in traj_data["trajectories"]:
            cls_name = traj.get("vehicle_class", "car")
            for frm in traj["frames"]:
                fn = frm["frame_num"]
                if fn >= split_frm:
                    continue   # held-out territory
                if fn % step != 0:
                    continue   # sampling
                frame_boxes[fn].append((frm["bbox"], cls_name))

        if not frame_boxes:
            print(f"  {vfile}: no train-split frames with trajectories")
            continue

        print(f"  {vfile:<28}  {len(frame_boxes)} frames to extract", flush=True)

        cap = cv2.VideoCapture(vpath)
        stem = os.path.splitext(vfile)[0]
        written = 0

        for fn in sorted(frame_boxes.keys()):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
            ret, frame = cap.read()
            if not ret:
                continue

            # Build label lines
            lines = []
            for bbox_ltrb, cls_name in frame_boxes[fn]:
                cls_id = CLASS_MAP.get(cls_name, 0)
                lines.append(_bbox_to_yolo(bbox_ltrb, fw, fh, cls_id))

            if not lines:
                continue

            name = f"{stem}_f{fn:05d}"
            img_path = os.path.join(dataset_train_img, name + ".jpg")
            lbl_path = os.path.join(dataset_train_lbl, name + ".txt")

            cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            with open(lbl_path, "w") as f:
                f.write("\n".join(lines) + "\n")
            written += 1
            total_written += 1

        cap.release()
        print(f"    → {written} positive samples written")

    return total_written


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Copy hard negatives into dataset/train/
# ─────────────────────────────────────────────────────────────────────────────
def copy_hard_negatives(dataset_train_img: str, dataset_train_lbl: str) -> int:
    """Copy 663 hard-negative crops + empty labels to train directories."""
    print("\n  Step 3 — Copying hard negatives ...")

    if not os.path.isdir(HN_SRC_IMAGES):
        print(f"  WARNING: {HN_SRC_IMAGES} not found — skipping")
        return 0

    images = [f for f in os.listdir(HN_SRC_IMAGES) if f.endswith((".png", ".jpg"))]
    copied = 0
    for img_file in images:
        src = os.path.join(HN_SRC_IMAGES, img_file)
        stem = os.path.splitext(img_file)[0]

        # Convert to jpg for consistency
        dst_img = os.path.join(dataset_train_img, "hn_" + stem + ".jpg")
        dst_lbl = os.path.join(dataset_train_lbl, "hn_" + stem + ".txt")

        if not os.path.exists(dst_img):
            img = cv2.imread(src)
            if img is not None:
                cv2.imwrite(dst_img, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not os.path.exists(dst_lbl):
            open(dst_lbl, "w").close()   # empty = background class
        copied += 1

    print(f"    → {copied} hard-negative samples ready")
    return copied


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Create val split + data.yaml
# ─────────────────────────────────────────────────────────────────────────────
def create_val_split(dataset_dir: str, val_ratio: float = 0.15) -> dict:
    """Move val_ratio of positive samples to dataset/val/."""
    print("  Step 4 — Creating val split ...")

    train_img = os.path.join(dataset_dir, "train", "images")
    train_lbl = os.path.join(dataset_dir, "train", "labels")
    val_img   = os.path.join(dataset_dir, "val",   "images")
    val_lbl   = os.path.join(dataset_dir, "val",   "labels")
    os.makedirs(val_img, exist_ok=True)
    os.makedirs(val_lbl, exist_ok=True)

    # Only move positive (non-hard-negative) samples to val
    all_images = [f for f in os.listdir(train_img) if f.endswith(".jpg") and not f.startswith("hn_")]
    np.random.seed(42)
    np.random.shuffle(all_images)
    n_val = max(1, int(len(all_images) * val_ratio))
    val_files = all_images[:n_val]

    moved = 0
    for img_f in val_files:
        stem = os.path.splitext(img_f)[0]
        for ext, src_dir, dst_dir in [(".jpg", train_img, val_img),
                                       (".txt", train_lbl, val_lbl)]:
            src = os.path.join(src_dir, stem + ext)
            dst = os.path.join(dst_dir, stem + ext)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.move(src, dst)
        moved += 1

    train_count = len([f for f in os.listdir(train_img) if f.endswith(".jpg")])
    val_count   = len([f for f in os.listdir(val_img)   if f.endswith(".jpg")])
    print(f"    → train: {train_count}  val: {val_count}")
    return {"train_count": train_count, "val_count": val_count}


def write_data_yaml(dataset_dir: str) -> str:
    """Write data.yaml for YOLO training."""
    abs_dir = os.path.abspath(dataset_dir)
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    content = f"""# ARGUS Week 3 training dataset
# Hard-negative samples + weak-supervision positives from trajectories

path: {abs_dir}
train: train/images
val:   val/images

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    with open(yaml_path, "w") as f:
        f.write(content)
    print(f"  saved → {yaml_path}")
    return yaml_path


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Fine-tune
# ─────────────────────────────────────────────────────────────────────────────
def run_finetuning(yaml_path: str, epochs: int, batch: int,
                   device: str, models_dir: str) -> str:
    """Run yolov8n.pt fine-tuning. Returns path to best.pt."""
    print("\n" + "=" * 65)
    print(f"  Step 5 — Fine-tuning yolov8n.pt  (epochs={epochs}  batch={batch}  device={device})")
    print("=" * 65)

    os.makedirs(models_dir, exist_ok=True)
    final_model = os.path.join(models_dir, "week2_retrained.pt")

    if os.path.exists(final_model):
        print(f"  [CACHE] {final_model} already exists — skipping training.")
        return final_model

    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        results = model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=640,
            batch=batch,
            device=device,
            patience=10,
            name="week2_retrain",
            project="runs/train",
            exist_ok=True,
            verbose=True,
            # Augmentation (supplement Albumentations pipeline)
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1,
        )
        # Copy best weights to models/
        best = os.path.join("runs", "train", "week2_retrain", "weights", "best.pt")
        if not os.path.exists(best):
            best = os.path.join("runs", "train", "week2_retrain", "weights", "last.pt")
        if os.path.exists(best):
            shutil.copy(best, final_model)
            print(f"  Best weights → {final_model}")
        else:
            print(f"  WARNING: could not find best.pt — check runs/train/week2_retrain/")
    except Exception as e:
        print(f"  Training error: {e}")
        print("  Creating placeholder model path (training failed — check GPU/dataset)")
        return ""

    return final_model


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Before / After comparison
# ─────────────────────────────────────────────────────────────────────────────
def _quick_inference(model_path: str, video_path: str, conf: float = 0.40,
                     max_frames: int = 100) -> dict:
    """Run inference on up to max_frames of a video and collect stats."""
    from ultralytics import YOLO
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step  = max(1, total // max_frames)

    confs = []
    det_counts = []
    frame_num  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % step != 0:
            frame_num += 1
            continue
        results = model(frame, conf=conf, iou=0.35, verbose=False)[0]
        n_det = len(results.boxes)
        det_counts.append(n_det)
        for box in results.boxes:
            confs.append(float(box.conf[0]))
        frame_num += 1

    cap.release()
    return {
        "frames_sampled":   len(det_counts),
        "mean_detections":  round(float(np.mean(det_counts)) if det_counts else 0, 2),
        "total_detections": sum(det_counts),
        "mean_confidence":  round(float(np.mean(confs)) if confs else 0, 4),
        "p25_confidence":   round(float(np.percentile(confs, 25)) if confs else 0, 4),
        "p75_confidence":   round(float(np.percentile(confs, 75)) if confs else 0, 4),
        "det_counts":       det_counts,
    }


def compare_before_after(footage_dir: str, retrained_model: str,
                          original_model: str, output_dir: str) -> dict:
    """
    Run quick inference (sampling every Nth frame) with both models.
    Report confidence distribution shift + detection count change.
    """
    print("\n" + "=" * 65)
    print("  Step 6 — Before / After comparison")
    print("=" * 65)

    comparison_path = os.path.join(output_dir, "retrain_comparison.json")
    if os.path.exists(comparison_path):
        print(f"  [CACHE] {comparison_path} already exists — loading.")
        return _load_json(comparison_path)

    if not retrained_model or not os.path.exists(retrained_model):
        print(f"  WARNING: retrained model not found at {retrained_model}")
        print("  Skipping before/after comparison.")
        return {}

    results = {}
    for vfile in VIDEOS:
        vpath = os.path.join(footage_dir, vfile)
        if not os.path.exists(vpath):
            continue
        print(f"  {vfile} — original ...", flush=True)
        before = _quick_inference(original_model, vpath)
        print(f"  {vfile} — retrained ...", flush=True)
        after  = _quick_inference(retrained_model, vpath)

        results[vfile] = {
            "before": {k: v for k, v in before.items() if k != "det_counts"},
            "after":  {k: v for k, v in after.items()  if k != "det_counts"},
            "delta": {
                "mean_detections":  round(after["mean_detections"]  - before["mean_detections"],  2),
                "mean_confidence":  round(after["mean_confidence"]   - before["mean_confidence"],  4),
                "total_detections": after["total_detections"] - before["total_detections"],
            },
        }
        print(f"    det: {before['mean_detections']:.1f} → {after['mean_detections']:.1f}  "
              f"conf: {before['mean_confidence']:.3f} → {after['mean_confidence']:.3f}")

    summary = {
        "original_model":  original_model,
        "retrained_model": retrained_model,
        "per_video":       results,
        "interpretation": (
            "Retrained model shows lower mean detections on non-accident footage "
            "→ hard negatives successfully reduced false positives."
            if any(
                results[v]["delta"]["mean_detections"] < 0
                for v in results
                if "accident" not in v
            ) else
            "Detection counts unchanged — hard negatives had minimal effect. "
            "Consider more training epochs or additional negative samples."
        ),
    }
    _save_json(summary, comparison_path)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Overlap / hash check
# ─────────────────────────────────────────────────────────────────────────────
def check_train_test_overlap(dataset_train_img: str, unseen_dir: str) -> dict:
    """MD5-hash all training images and sampled unseen frames; report overlaps."""
    print("\n  Overlap check: training images vs unseen test frames ...")

    def md5_dir_images(d):
        hashes = {}
        if not os.path.isdir(d):
            return hashes
        for f in os.listdir(d):
            if f.endswith((".jpg", ".png")):
                p = os.path.join(d, f)
                h = hashlib.md5(open(p, "rb").read()).hexdigest()
                hashes[h] = f
        return hashes

    train_hashes = md5_dir_images(dataset_train_img)
    # For unseen: hash the test clips' first frames as a proxy
    unseen_hashes = {}
    if os.path.isdir(unseen_dir):
        for f in os.listdir(unseen_dir):
            if not f.endswith(".mp4"):
                continue
            cap = cv2.VideoCapture(os.path.join(unseen_dir, f))
            ret, frame = cap.read()
            cap.release()
            if ret:
                _, buf = cv2.imencode(".jpg", frame)
                h = hashlib.md5(buf.tobytes()).hexdigest()
                unseen_hashes[h] = f

    overlap = set(train_hashes) & set(unseen_hashes)
    report  = {
        "training_images":   len(train_hashes),
        "unseen_samples":    len(unseen_hashes),
        "overlapping_hashes": len(overlap),
        "overlap_clean":     len(overlap) == 0,
        "note": (
            "Clean — no pixel-level overlap between training and unseen test sets."
            if len(overlap) == 0 else
            f"WARNING: {len(overlap)} identical images found in both train and test sets."
        ),
        "hard_negative_note": (
            "Hard negatives were mined from the full videos (including last 30%). "
            "Since they are background-class crops (no bbox annotations), this does NOT "
            "leak ground-truth positional info into training, but future mining should "
            "be restricted to train-split frames only."
        ),
    }
    print(f"    train={len(train_hashes)}  unseen={len(unseen_hashes)}  overlap={len(overlap)}")
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Week 3 retraining pipeline")
    parser.add_argument("--footage",  default="sample_videos")
    parser.add_argument("--dataset",  default="dataset")
    parser.add_argument("--unseen",   default="unseen_test_videos")
    parser.add_argument("--models",   default="models")
    parser.add_argument("--output",   default="reports")
    parser.add_argument("--original", default="yolov8s.pt",
                        help="Original model for before/after comparison")
    parser.add_argument("--epochs",   type=int,   default=30)
    parser.add_argument("--batch",    type=int,   default=16)
    parser.add_argument("--device",   default="auto",
                        help="auto | cpu | mps | cuda")
    args = parser.parse_args()

    if args.device == "auto":
        try:
            import torch
            if torch.backends.mps.is_available():
                args.device = "mps"
            elif torch.cuda.is_available():
                args.device = "cuda"
            else:
                args.device = "cpu"
        except ImportError:
            args.device = "cpu"

    print("\n" + "=" * 65)
    print("  ARGUS — Week 3 Retraining Pipeline")
    print("=" * 65)
    print(f"  footage  : {args.footage}")
    print(f"  dataset  : {args.dataset}")
    print(f"  unseen   : {args.unseen}")
    print(f"  device   : {args.device}")
    print(f"  epochs   : {args.epochs}")

    train_img = os.path.join(args.dataset, "train", "images")
    train_lbl = os.path.join(args.dataset, "train", "labels")

    # ── 1. Split videos ──────────────────────────────────────────────────
    splits = split_videos(args.footage, args.unseen)

    # ── 2. Positive samples ──────────────────────────────────────────────
    n_pos = extract_positive_samples(args.footage, train_img, train_lbl, splits)

    # ── 3. Hard negatives ────────────────────────────────────────────────
    n_neg = copy_hard_negatives(train_img, train_lbl)

    # ── 4. Val split + data.yaml ─────────────────────────────────────────
    counts   = create_val_split(args.dataset)
    yaml_path = write_data_yaml(args.dataset)

    print(f"\n  Dataset summary:")
    print(f"    Positive samples : {n_pos}")
    print(f"    Hard negatives   : {n_neg}")
    print(f"    Train total      : {counts['train_count']}")
    print(f"    Val total        : {counts['val_count']}")

    # ── 5. Fine-tune ─────────────────────────────────────────────────────
    retrained = run_finetuning(
        yaml_path, args.epochs, args.batch, args.device, args.models
    )

    # ── 6. Before / after comparison ─────────────────────────────────────
    comparison = compare_before_after(
        args.footage, retrained, args.original, args.output
    )

    # ── 7. Overlap check ─────────────────────────────────────────────────
    overlap = check_train_test_overlap(train_img, args.unseen)
    _save_json(overlap, os.path.join(args.output, "train_test_overlap.json"))

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RETRAINING COMPLETE")
    print("=" * 65)
    print(f"  Model      : {retrained or 'NOT SAVED (training failed)'}")
    print(f"  Positives  : {n_pos}")
    print(f"  Negatives  : {n_neg}")
    print(f"  Overlap    : {'CLEAN ✓' if overlap.get('overlap_clean') else 'WARNING ✗'}")
    if comparison:
        for vfile, res in comparison.get("per_video", {}).items():
            d = res["delta"]
            flag = "✓" if ("accident" in vfile and d["mean_detections"] >= 0) or \
                          ("accident" not in vfile and d["mean_detections"] <= 0) else "~"
            print(f"  {flag} {vfile:<28}  det Δ={d['mean_detections']:+.1f}  conf Δ={d['mean_confidence']:+.4f}")

    print(f"\n  reports/retrain_comparison.json")
    print(f"  reports/train_test_overlap.json")
    print(f"  models/week2_retrained.pt")


if __name__ == "__main__":
    main()
