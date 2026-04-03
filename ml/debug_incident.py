"""
Debug script: run pipeline on one video and log exactly which filter
rejects each pair, and which pairs got closest to triggering.
"""
import math, sys, os, uuid
from collections import defaultdict, deque
sys.path.insert(0, os.path.dirname(__file__))

import cv2
from ml_pipeline.detection import VehicleDetector
from ml_pipeline.tracking import VehicleTracker
from ml_pipeline.trajectory import TrajectoryBuilder
from ml_pipeline.interaction import (
    _bbox_edge_distance_px, _gap_based_ttc, _gap_trend_ok,
    MIN_BBOX_AREA_FRAC, MIN_CY_DIFF_FRAC, SAME_LANE_CX_FRAC,
    MIN_CLOSING_SPEED_KMH, MIN_SPEED_FLOOR_KMH, GAP_HISTORY_FRAMES,
    GAP_MONO_THRESHOLD, MIN_DIST_DANGER_M, TTC_NEAR_MISS_S,
    DENSE_TRAFFIC_MIN_VEHICLES, DENSE_ROLL_FRAMES, DENSE_TTC_S,
    DENSE_MIN_CLOSING_KMH, DENSE_MIN_SPEED_KMH,
    KMH_TO_MS,
)

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "sample_videos/accident_clip_1.mp4"
PPM   = float(sys.argv[2]) if len(sys.argv) > 2 else 22.0

cap = cv2.VideoCapture(VIDEO)
fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
fw    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fh    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

detector     = VehicleDetector(model_path="yolov8s.pt", confidence=0.4)
tracker      = VehicleTracker()
traj_builder = TrajectoryBuilder(fps=fps, pixels_per_meter=PPM)
video_id     = str(uuid.uuid4())

frame_num = 0
while True:
    ret, frame = cap.read()
    if not ret: break
    dets   = detector.detect(frame)
    tracks = tracker.update(dets, frame)
    traj_builder.update(frame_num, tracks)
    frame_num += 1
cap.release()

trajectories = traj_builder.get_trajectories(video_id)
print(f"\nVideo: {os.path.basename(VIDEO)}")
print(f"  {fw}x{fh} @ {fps}fps  {total}fr  |  {len(trajectories)} trajectories")

# Rebuild frame index
from collections import defaultdict
frame_index = defaultdict(list)
for traj in trajectories:
    for fd in traj["frames"]:
        frame_index[fd["frame_num"]].append({
            "vehicle_id":    traj["vehicle_id"],
            "vehicle_class": traj["vehicle_class"],
            "frame_data":    fd,
        })

# Per-pair rejection counters and "closest call" tracking
reject_counts = defaultdict(int)   # filter_name -> count
pair_best_ttc  = {}  # pair_key -> min TTC seen (regardless of filters)
pair_best_dist = {}  # pair_key -> min dist seen
pair_gap_hist  = {}
vehicle_count_roll = deque(maxlen=DENSE_ROLL_FRAMES)

frame_area = fw * fh
min_bbox_area    = MIN_BBOX_AREA_FRAC * frame_area
min_cy_diff_px   = MIN_CY_DIFF_FRAC * fh
min_closing_px_s = (MIN_CLOSING_SPEED_KMH * KMH_TO_MS) * PPM

pair_streak = {}  # pair_key -> consecutive danger frames

for frame_num in sorted(frame_index.keys()):
    vehicles = frame_index[frame_num]
    vehicle_count_roll.append(len(vehicles))
    avg_count = sum(vehicle_count_roll) / len(vehicle_count_roll)
    dense = avg_count >= DENSE_TRAFFIC_MIN_VEHICLES

    if dense:
        eff_ttc_thresh   = DENSE_TTC_S
        eff_closing_px_s = (DENSE_MIN_CLOSING_KMH * KMH_TO_MS) * PPM
        eff_speed_floor  = DENSE_MIN_SPEED_KMH
        eff_dist_gate    = False
    else:
        eff_ttc_thresh   = TTC_NEAR_MISS_S
        eff_closing_px_s = min_closing_px_s
        eff_speed_floor  = MIN_SPEED_FLOOR_KMH
        eff_dist_gate    = True

    if len(vehicles) < 2:
        for pk in list(pair_streak): del pair_streak[pk]
        continue

    widths = [max(0.0, v["frame_data"]["bbox"][2] - v["frame_data"]["bbox"][0]) for v in vehicles]
    avg_w  = sum(widths) / len(widths) if widths else 1.0
    max_cx = SAME_LANE_CX_FRAC * avg_w

    active = set()
    for i in range(len(vehicles)):
        for j in range(i+1, len(vehicles)):
            va, vb = vehicles[i], vehicles[j]
            aid, bid = va["vehicle_id"], vb["vehicle_id"]
            pk = (min(aid,bid), max(aid,bid))
            fd_a, fd_b = va["frame_data"], vb["frame_data"]

            area_a = (fd_a["bbox"][2]-fd_a["bbox"][0])*(fd_a["bbox"][3]-fd_a["bbox"][1])
            area_b = (fd_b["bbox"][2]-fd_b["bbox"][0])*(fd_b["bbox"][3]-fd_b["bbox"][1])
            if area_a < min_bbox_area and area_b < min_bbox_area:
                reject_counts["F1_tiny_bbox"] += 1; continue

            cy_diff = abs(fd_a["center"][1] - fd_b["center"][1])
            if cy_diff < min_cy_diff_px:
                reject_counts["F2_same_depth"] += 1; continue

            cx_diff = abs(fd_a["center"][0] - fd_b["center"][0])
            if cx_diff > max_cx:
                reject_counts["F6_diff_lane"] += 1; continue

            curr_gap = _bbox_edge_distance_px(fd_a["bbox"], fd_b["bbox"])
            if pk not in pair_gap_hist:
                pair_gap_hist[pk] = deque(maxlen=GAP_HISTORY_FRAMES)
            hist = pair_gap_hist[pk]
            prev_gap = hist[-1] if hist else None
            hist.append(curr_gap)

            ttc = None
            gap_closing = 0.0
            if prev_gap is not None:
                gap_closing = (prev_gap - curr_gap) * fps
                ttc = _gap_based_ttc(curr_gap, prev_gap, fps)

            # Track best metrics regardless of filters
            dist_m = curr_gap / PPM
            if ttc is not None:
                pair_best_ttc[pk]  = min(pair_best_ttc.get(pk, 999), ttc)
            pair_best_dist[pk] = min(pair_best_dist.get(pk, 999), dist_m)

            if abs(gap_closing) < eff_closing_px_s:
                reject_counts["F5_slow_closing"] += 1; continue

            speed_a = fd_a["speed_estimate"]
            speed_b = fd_b["speed_estimate"]
            if max(speed_a, speed_b) < eff_speed_floor:
                reject_counts["F7_speed_floor"] += 1; continue

            closing = gap_closing > 0
            if eff_dist_gate:
                dangerous = (dist_m < MIN_DIST_DANGER_M and closing) or \
                            (ttc is not None and ttc < eff_ttc_thresh)
            else:
                dangerous = ttc is not None and ttc < eff_ttc_thresh

            if not dangerous:
                reject_counts["F3_not_dangerous"] += 1; continue

            if not _gap_trend_ok(hist):
                reject_counts["F8_oscillating"] += 1; continue

            # Passed all per-frame filters — accumulate streak
            active.add(pk)
            pair_streak[pk] = pair_streak.get(pk, 0) + 1
            reject_counts["PASS_frame"] += 1

    for pk in list(pair_streak):
        if pk not in active:
            if pair_streak[pk] < 8:
                reject_counts[f"F4_streak_too_short_{pair_streak[pk]}fr"] += 1
            del pair_streak[pk]

print(f"\n── Filter rejection breakdown ──────────────────────────────")
for k, v in sorted(reject_counts.items(), key=lambda x: -x[1]):
    print(f"  {k:<45} {v:>6}")

print(f"\n── Dense mode stats ────────────────────────────────────────")
vehicle_count_roll2 = deque(maxlen=DENSE_ROLL_FRAMES)
dense_frames = normal_frames = 0
for fn in sorted(frame_index.keys()):
    vehicle_count_roll2.append(len(frame_index[fn]))
    avg = sum(vehicle_count_roll2)/len(vehicle_count_roll2)
    if avg >= DENSE_TRAFFIC_MIN_VEHICLES: dense_frames += 1
    else: normal_frames += 1
print(f"  Dense frames : {dense_frames}  ({100*dense_frames/(dense_frames+normal_frames+1):.0f}%)")
print(f"  Normal frames: {normal_frames}")

print(f"\n── Closest calls (min TTC seen, all pairs) ─────────────────")
top_ttc  = sorted(pair_best_ttc.items(),  key=lambda x: x[1])[:10]
top_dist = sorted(pair_best_dist.items(), key=lambda x: x[1])[:10]
print("  By TTC:")
for pk, ttc in top_ttc:
    dist = pair_best_dist.get(pk, 999)
    print(f"    pair {pk}  TTC={ttc:.3f}s  min_dist={dist:.2f}m")
print("  By distance:")
for pk, dist in top_dist:
    ttc = pair_best_ttc.get(pk, None)
    print(f"    pair {pk}  dist={dist:.2f}m  TTC={ttc:.3f}s" if ttc else f"    pair {pk}  dist={dist:.2f}m  TTC=n/a")
