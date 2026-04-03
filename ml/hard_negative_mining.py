"""
hard_negative_mining.py — Collect false-positive crops from your footage.

Strategy:
  - Run the full detection pipeline (with temporal confirmation buffer) on each video.
  - Detections that are rejected by the buffer (appeared in < CONFIRM_MIN_FRAMES
    consecutive frames) are likely phantoms: shadows, reflections, motion blur.
  - Crop those regions and save them as YOLO-format "background" (hard negative) samples.
  - Also collect confirmed detections for manual review / mixing.

Output layout:
    hard_negatives/
        images/   — cropped false-positive patches (PNG)
        labels/   — empty .txt files (YOLO background = no annotation)
        review/   — confirmed detections for human spot-check

Usage:
    python hard_negative_mining.py --videos sample_videos/ --out hard_negatives/
    python hard_negative_mining.py --videos sample_videos/ --conf 0.3  # lower conf → more candidates
"""

import argparse
import os
import sys
import uuid

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from ml_pipeline.detection import VehicleDetector, TemporalConfirmationBuffer, CONFIRM_MIN_FRAMES

# Minimum bbox area (px²) to bother saving — skip tiny specks
MIN_CROP_AREA = 400   # 20×20 px minimum
# Padding around bbox when saving crop
CROP_PAD_PX   = 8
# Maximum hard negatives per video (prevent massive dataset imbalance)
MAX_PER_VIDEO = 300


def _crop(frame: np.ndarray, bbox_xywh: list, pad: int = CROP_PAD_PX) -> np.ndarray:
    h, w = frame.shape[:2]
    x, y, bw, bh = bbox_xywh
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + bw + pad)
    y2 = min(h, y + bh + pad)
    return frame[y1:y2, x1:x2]


def mine_video(video_path: str, detector: VehicleDetector,
               out_dir: str, max_negs: int = MAX_PER_VIDEO) -> dict:
    """
    Run the detector on one video and collect hard negatives.

    Returns counts: {phantom_saved, confirmed_saved, total_frames}
    """
    img_dir     = os.path.join(out_dir, "images")
    lbl_dir     = os.path.join(out_dir, "labels")
    review_dir  = os.path.join(out_dir, "review")
    for d in (img_dir, lbl_dir, review_dir):
        os.makedirs(d, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: cannot open {video_path}")
        return {}

    vname         = os.path.splitext(os.path.basename(video_path))[0]
    total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    phantom_saved = 0
    confirmed_saved = 0
    frame_num     = 0

    # Fresh detector buffer for this video — don't share state across videos
    detector._buffer = TemporalConfirmationBuffer()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection (buffer updated internally)
        confirmed = detector.detect(frame)

        # Collect phantoms — detections rejected by the buffer this frame
        phantoms = detector.phantom_candidates

        # ── Save phantom crops as hard negatives ──────────────────────────
        for det in phantoms:
            if phantom_saved >= max_negs:
                break
            bw, bh = det["bbox_xywh"][2], det["bbox_xywh"][3]
            if bw * bh < MIN_CROP_AREA:
                continue
            crop = _crop(frame, det["bbox_xywh"])
            if crop.size == 0:
                continue
            name = f"{vname}_f{frame_num:05d}_{uuid.uuid4().hex[:6]}"
            cv2.imwrite(os.path.join(img_dir, name + ".png"), crop)
            # Empty label file = YOLO background sample
            open(os.path.join(lbl_dir, name + ".txt"), "w").close()
            phantom_saved += 1

        # ── Save a sample of confirmed detections for review ──────────────
        if frame_num % 30 == 0:   # every ~1 s
            for det in confirmed[:3]:
                crop = _crop(frame, det["bbox_xywh"])
                if crop.size == 0:
                    continue
                name = f"{vname}_conf_f{frame_num:05d}_{uuid.uuid4().hex[:4]}"
                cv2.imwrite(os.path.join(review_dir, name + ".png"), crop)
                confirmed_saved += 1

        if frame_num % 150 == 0:
            pct = int(frame_num / total_frames * 100) if total_frames else 0
            print(f"    [{pct:3d}%] frame {frame_num}  phantoms_so_far={phantom_saved}", flush=True)

        frame_num += 1

    cap.release()
    return {"phantom_saved": phantom_saved, "confirmed_saved": confirmed_saved,
            "total_frames": frame_num}


def main():
    parser = argparse.ArgumentParser(description="Hard negative mining from false-positive detections")
    parser.add_argument("--videos",  default="sample_videos",  help="Directory of input videos")
    parser.add_argument("--out",     default="hard_negatives", help="Output directory")
    parser.add_argument("--model",   default="yolov8s.pt")
    parser.add_argument("--conf",    type=float, default=0.35,  help="Use lower conf to catch more phantom candidates")
    parser.add_argument("--max",     type=int,   default=MAX_PER_VIDEO, help="Max negatives per video")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    videos = sorted(f for f in os.listdir(args.videos) if f.endswith(".mp4"))

    # Single detector instance (buffer reset per video inside mine_video)
    detector = VehicleDetector(
        model_path=args.model,
        confidence=args.conf,   # lower conf intentionally — we want candidate phantoms
        temporal_confirm=True,
    )

    total_phantoms = 0
    print(f"\nHard Negative Mining  ({len(videos)} videos, conf={args.conf})")
    print(f"Phantoms = detections appearing in < {CONFIRM_MIN_FRAMES} consecutive frames\n")

    for vname in videos:
        vpath = os.path.join(args.videos, vname)
        print(f"  Processing: {vname}")
        counts = mine_video(vpath, detector, args.out, args.max)
        p = counts.get("phantom_saved", 0)
        total_phantoms += p
        print(f"  → {p} phantom crops saved  ({counts.get('confirmed_saved',0)} confirmed review samples)\n")

    print(f"Done. Total hard negatives: {total_phantoms}")
    print(f"Output: {args.out}/images/  +  {args.out}/labels/")
    print()
    print("Next steps:")
    print("  1. Spot-check hard_negatives/review/ — remove any true positives that snuck in")
    print("  2. Add hard_negatives/ to your training dataset as a 'background' class")
    print("     (YOLO format: image + empty label file = no objects = background)")
    print("  3. Re-train: yolo train data=data.yaml model=yolov8s.pt epochs=50")
    print("  4. Repeat mining on the new model to progressively harden it")


if __name__ == "__main__":
    main()
