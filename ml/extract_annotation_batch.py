"""
extract_annotation_batch.py
Extract 500 diverse frames from ARGUS yt_videos for annotation in CVAT.
Samples evenly across all videos to maximize scene diversity.

Usage:
    python extract_annotation_batch.py

Output:
    annotation_batch_v3/   ← zip and upload to CVAT
"""

import cv2
import os
import math
from pathlib import Path

VIDEO_DIR   = Path(__file__).parent / 'yt_videos'
OUTPUT_DIR  = Path(__file__).parent / 'annotation_batch_v3'
TOTAL_FRAMES = 500

OUTPUT_DIR.mkdir(exist_ok=True)

# Get all videos and their frame counts
videos = []
for mp4 in sorted(VIDEO_DIR.glob('*.mp4')):
    cap = cv2.VideoCapture(str(mp4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if total > 30:  # skip corrupt/empty
        videos.append((mp4, total, fps))

print(f'Found {len(videos)} videos')

# Distribute 500 frames across videos proportionally (min 5 per video)
total_footage = sum(t for _, t, _ in videos)
# Cap at 50 per video, min 5, distribute remaining evenly
MAX_PER_VIDEO = 50
MIN_PER_VIDEO = 5

# First pass: proportional allocation with cap
allocations = []
for mp4, total, fps in videos:
    n = max(MIN_PER_VIDEO, min(MAX_PER_VIDEO, round(TOTAL_FRAMES * total / total_footage)))
    allocations.append([mp4, total, n])

# Distribute remaining frames to videos that have room under the cap
remaining = TOTAL_FRAMES - sum(n for _, _, n in allocations)
i = 0
while remaining > 0:
    mp4, total, n = allocations[i % len(allocations)]
    if n < MAX_PER_VIDEO:
        allocations[i % len(allocations)][2] += 1
        remaining -= 1
    i += 1
    if i > len(allocations) * MAX_PER_VIDEO:
        break  # safety

print(f'\nFrame allocation per video:')
for mp4, total, n in allocations:
    print(f'  {mp4.name}: {n} frames from {total} total')

# Extract frames
extracted = 0
for mp4, total, n in allocations:
    cap = cv2.VideoCapture(str(mp4))

    # Sample evenly, skip first/last 5% (often title cards or black frames)
    start = int(total * 0.05)
    end   = int(total * 0.95)
    indices = [int(start + i * (end - start) / (n - 1)) for i in range(n)]

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        fname = f'{mp4.stem}_f{idx:07d}.jpg'
        out_path = OUTPUT_DIR / fname
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        extracted += 1

    cap.release()
    print(f'  Extracted {n} frames from {mp4.name}')

print(f'\nDone. {extracted} frames saved to {OUTPUT_DIR}/')
print(f'\nNext steps:')
print(f'  1. zip -r annotation_batch_v3.zip annotation_batch_v3/')
print(f'  2. Upload to CVAT as a new task')
print(f'  3. Classes: car, motorcycle, bus, truck')
print(f'  4. Export as YOLO 1.1 format when done')
