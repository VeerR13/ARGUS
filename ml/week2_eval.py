"""
week2_eval.py — Run updated Week 2 pipeline on all videos.

For each video:
  - Runs analyze_video() with updated interaction thresholds
  - Saves a JSON results file
  - Extracts an annotated clip for every detected incident (padding seconds before/after)
  - Prints a summary table

Usage:
    python week2_eval.py [--pixels-per-meter 22] [--conf 0.4]
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
from ml_pipeline.interaction import detect_incidents

VIDEOS_DIR        = "sample_videos"
OUTPUT_CLIPS      = "output_clips"
CLIP_PAD_S        = 2.0    # seconds of padding before/after each incident in clips
DEDUP_TIME_WINDOW = 5.0    # seconds — incidents sharing a vehicle within this window → same event
DEDUP_SPACE_PX    = 80     # pixels — spatial centre proximity threshold for secondary dedup

_PALETTE = [
    (255,  56,  56), (255, 157,  51), ( 77, 255,  83), ( 56, 182, 255),
    (255,  51, 255), (  0, 204, 204), (255, 204,  51), (204,  51,  51),
    ( 51, 204,  51), ( 51,  51, 204), (204, 153,   0), (153,   0, 204),
]

def _color(tid):
    return _PALETTE[tid % len(_PALETTE)]

INC_COLORS = {"near_miss": (0, 80, 255), "accident": (0, 0, 220), "risky_interaction": (0, 200, 255)}
SEV_COLORS = {"high": (0, 0, 255), "medium": (0, 128, 255), "low": (0, 200, 200)}


def run_video(video_path, model_path, confidence, pixels_per_meter):
    """Full pipeline on one video. Returns (result_dict, all_frames_list)."""
    video_id = str(uuid.uuid4())
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw           = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh           = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector     = VehicleDetector(model_path=model_path, confidence=confidence)
    tracker      = VehicleTracker()
    traj_builder = TrajectoryBuilder(fps=fps, pixels_per_meter=pixels_per_meter)

    all_frames   = []
    active_by_frame = {}
    frame_num    = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detections    = detector.detect(frame)
        active_tracks = tracker.update(detections, frame)
        traj_builder.update(frame_num, active_tracks)
        active_by_frame[frame_num] = active_tracks
        all_frames.append(frame.copy())
        if frame_num % 150 == 0:
            pct = int(frame_num / total_frames * 100) if total_frames else 0
            print(f"    [{pct:3d}%] frame {frame_num}/{total_frames}  vehicles={len(active_tracks)}")
        frame_num += 1

    cap.release()

    trajectories = traj_builder.get_trajectories(video_id)
    incidents    = detect_incidents(
        trajectories     = trajectories,
        fps              = fps,
        pixels_per_meter = pixels_per_meter,
        video_id         = video_id,
        frame_width      = fw,
        frame_height     = fh,
    )

    return {
        "video_id":    video_id,
        "video_path":  video_path,
        "fps":         fps,
        "frame_width": fw,
        "frame_height":fh,
        "total_frames":frame_num,
        "trajectories":trajectories,
        "incidents":   incidents,
    }, all_frames, active_by_frame


def draw_frame(frame, active_tracks, incident_vehicles=None, overlay_text=""):
    out = frame.copy()
    inc_set = set(incident_vehicles) if incident_vehicles else set()

    for track in active_tracks:
        tid = track["track_id"]
        l, t, r, b = track["bbox_ltrb"]
        color = (0, 0, 220) if tid in inc_set else _color(tid)
        thick = 3 if tid in inc_set else 2
        cv2.rectangle(out, (l, t), (r, b), color, thick)
        label = f"ID:{tid} {track['class_name']}"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (l, t - th - bl - 6), (l + tw + 4, t), color, -1)
        cv2.putText(out, label, (l + 2, t - bl - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    if overlay_text:
        cv2.rectangle(out, (0, 0), (out.shape[1], 38), (0, 0, 0), -1)
        cv2.putText(out, overlay_text, (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 80), 2, cv2.LINE_AA)

    return out


def _incident_centre(incident, active_by_frame):
    """Approximate pixel centre of an incident from vehicle bbox midpoints at incident time."""
    vids = set(incident["vehicles_involved"])
    frame = incident["frame_start"]
    tracks = active_by_frame.get(frame, [])
    pts = []
    for t in tracks:
        if t["track_id"] in vids:
            l, top, r, b = t["bbox_ltrb"]
            pts.append(((l + r) / 2, (top + b) / 2))
    if pts:
        return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
    return (0, 0)


def dedup_incidents(incidents, active_by_frame):
    """
    Collapse multiple detections of the same real-world event into one clip.

    Two incidents are considered duplicates (same event) when they:
      1. Share at least one vehicle ID, AND
      2. Their time ranges are within DEDUP_TIME_WINDOW seconds of each other.

    A secondary spatial check (DEDUP_SPACE_PX) catches the rare case where
    unrelated vehicles happen to get the same ID across far-apart clips.

    Incidents are already sorted highest-severity first, so the first incident
    in each cluster (most severe) is the canonical representative.
    """
    kept = []
    for inc in incidents:
        t_s   = inc["timestamp_start_ms"] / 1000
        t_e   = inc["timestamp_end_ms"]   / 1000
        vids  = set(inc["vehicles_involved"])
        cx, cy = _incident_centre(inc, active_by_frame)

        duplicate = False
        for k in kept:
            k_s  = k["timestamp_start_ms"] / 1000
            k_e  = k["timestamp_end_ms"]   / 1000
            k_v  = set(k["vehicles_involved"])
            k_cx, k_cy = _incident_centre(k, active_by_frame)

            time_close    = not (t_e + DEDUP_TIME_WINDOW < k_s or
                                 k_e + DEDUP_TIME_WINDOW < t_s)
            vehicle_share = bool(vids & k_v)
            space_close   = ((cx - k_cx) ** 2 + (cy - k_cy) ** 2) ** 0.5 < DEDUP_SPACE_PX

            if time_close and (vehicle_share or space_close):
                duplicate = True
                break

        if not duplicate:
            kept.append(inc)

    return kept


def save_incident_clip(incident, result, all_frames, active_by_frame, out_dir, clip_idx):
    """Extract and save a padded clip around one incident."""
    fps       = result["fps"]
    fw, fh    = result["frame_width"], result["frame_height"]
    pad       = int(CLIP_PAD_S * fps)
    f_start   = max(0, incident["frame_start"] - pad)
    f_end     = min(len(all_frames) - 1, incident["frame_end"] + pad)
    inc_vids  = set(incident["vehicles_involved"])
    m         = incident["metrics"]
    inc_type  = incident["type"]
    sev       = incident["severity"]
    ts_s      = incident["timestamp_start_ms"] / 1000.0

    vname  = os.path.splitext(os.path.basename(result["video_path"]))[0]
    fname  = f"{vname}_incident{clip_idx:02d}_{inc_type}_{sev}.mp4"
    fpath  = os.path.join(out_dir, fname)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(fpath, fourcc, fps, (fw, fh))

    for fn in range(f_start, f_end + 1):
        active = active_by_frame.get(fn, [])
        in_incident = f_start + pad <= fn <= f_end - pad
        overlay = (
            f"[{inc_type.upper()}] sev={sev} | TTC={m['min_ttc']}s "
            f"dist={m['min_distance_meters']}m spd={m['relative_speed_kmh']}km/h | t={ts_s:.1f}s"
            if in_incident else
            f"  {'PRE' if fn < incident['frame_start'] else 'POST'}-INCIDENT padding  | frame {fn}"
        )
        frame = draw_frame(all_frames[fn], active,
                           incident_vehicles=inc_vids if in_incident else None,
                           overlay_text=overlay)
        writer.write(frame)

    writer.release()
    return fname


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",            default="yolov8s.pt")
    parser.add_argument("--conf",             type=float, default=0.4)
    parser.add_argument("--pixels-per-meter", type=float, default=22.0)
    args = parser.parse_args()

    os.makedirs(OUTPUT_CLIPS, exist_ok=True)

    videos = sorted(f for f in os.listdir(VIDEOS_DIR) if f.endswith(".mp4"))

    all_results = []
    clip_files  = []

    for vname in videos:
        vpath = os.path.join(VIDEOS_DIR, vname)
        print(f"\n{'='*60}")
        print(f"  Processing: {vname}")
        print(f"{'='*60}")

        result, all_frames, active_by_frame = run_video(
            vpath, args.model, args.conf, args.pixels_per_meter
        )

        # Save JSON
        json_path = f"output_{os.path.splitext(vname)[0]}_w2.json"
        with open(json_path, "w") as f:
            # trajectories can be large — omit frame details from saved JSON for brevity
            slim = {k: v for k, v in result.items() if k not in ("trajectories",)}
            slim["trajectory_count"] = len(result["trajectories"])
            json.dump(slim, f, indent=2)

        # Deduplicate incidents (same crash → one clip) then save
        raw_count   = len(result["incidents"])
        deduped     = dedup_incidents(result["incidents"], active_by_frame)
        if raw_count != len(deduped):
            print(f"  ℹ  Deduped {raw_count} incidents → {len(deduped)} unique events")

        video_clips = []
        for idx, inc in enumerate(deduped, 1):
            fname = save_incident_clip(inc, result, all_frames, active_by_frame,
                                       OUTPUT_CLIPS, idx)
            video_clips.append(fname)
            clip_files.append(fname)
            print(f"  → clip saved: output_clips/{fname}")

        all_results.append((vname, result))

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  WEEK 2 EVALUATION RESULTS  (pixels_per_meter={args.pixels_per_meter})")
    print(f"{'='*80}")

    expected = {
        "accident_clip_1.mp4":  "≥1 (accident/near_miss)",
        "accident_clip_2.mp4":  "≥1 (accident/near_miss)",
        "dashcam_30s_720p.mp4": "0  (crowded city — should suppress)",
        "highway_30s_720p.mp4": "0  (sparse follow-cam)",
        "night_road_720p.mp4":  "0  (crowded night — should suppress)",
    }

    header = f"{'Video':<28} {'Expected':<30} {'Actual':>6}  {'Types & Severity'}"
    print(f"\n  {header}")
    print(f"  {'-'*90}")

    for vname, result in all_results:
        incs = result["incidents"]
        n    = len(incs)
        exp  = expected.get(vname, "?")

        if n == 0:
            detail = "—"
        else:
            parts = []
            for inc in incs:
                parts.append(f"{inc['type']}({inc['severity']},"
                             f"TTC={inc['metrics']['min_ttc']}s,"
                             f"d={inc['metrics']['min_distance_meters']}m,"
                             f"spd={inc['metrics']['relative_speed_kmh']}kmh)")
            detail = " | ".join(parts)

        ok = ""
        if "accident" in vname and n >= 1:
            ok = " ✓"
        elif "accident" not in vname and n == 0:
            ok = " ✓"
        elif n > 0:
            ok = " ✗ (false positive)"

        print(f"  {vname:<28} {exp:<30} {n:>6}{ok}")
        if detail != "—":
            for inc in incs:
                m = inc["metrics"]
                ts = inc["timestamp_start_ms"] / 1000
                te = inc["timestamp_end_ms"]   / 1000
                print(f"  {'':28}   {inc['type']:<20} sev={inc['severity']:<6} "
                      f"conf={inc['confidence']}  "
                      f"TTC={m['min_ttc']}s  dist={m['min_distance_meters']}m  "
                      f"spd={m['relative_speed_kmh']}km/h  "
                      f"t={ts:.1f}s–{te:.1f}s  "
                      f"vehicles={inc['vehicles_involved']}")

    print(f"\n  {'─'*90}")
    print(f"\n  Saved clips ({len(clip_files)} total):")
    for cf in clip_files:
        print(f"    output_clips/{cf}")
    print()


if __name__ == "__main__":
    main()
