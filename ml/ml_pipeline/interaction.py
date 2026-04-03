"""
ml_pipeline/interaction.py — Week 2: Interaction Metrics & Near-Miss Detection

Distinguishes genuine near-misses / accidents from crowded-traffic false positives
using the original six filters (F1–F6) plus three new ones:

    F7 — speed floor: at least one vehicle must be moving meaningfully.
         Suppresses slow-creep traffic where max possible relative speed is tiny.

    F8 — gap monotonicity: the gap must be trending steadily downward (not
         oscillating). Stop-and-go traffic produces oscillating gaps; genuine
         approach produces a clear, sustained decreasing trend.

    F9 — dense-traffic mode: when the rolling average vehicle count in frame
         reaches DENSE_TRAFFIC_MIN_VEHICLES, apply stricter thresholds AND
         disable the distance gate entirely.  In congestion, proximity is
         normal — only a meaningful TTC with serious closing speed counts.

NOTE — F6 (same-lane cx gate) was removed after diagnostic showed it blocked
6 580 intersection-approach pair-frames, preventing all accident detection.
Adjacent-lane false positives are now suppressed by F2 (same-depth), F7
(speed floor), and F8 (gap monotonicity) instead.

Calibration-independent metrics:
    TTC   — derived from frame-to-frame gap closing rate (px/s); ppm cancels out.
    Decel — relative % speed drop over DECEL_WINDOW_FRAMES; not absolute km/h.

Key threshold changes vs. second pass (eval-driven):
    F6 removed entirely          (was blocking all intersection accidents)
    MIN_CONSECUTIVE_FRAMES 8→4   (accidents last 1-3 frames; 8 was too strict)
    DENSE_MIN_CONSECUTIVE 12→8   (same reason in dense mode)
    DENSE_TTC_S           0.5→0.7 (TTC 0.5-0.7s in busy traffic is dangerous)
    DENSE_TRAFFIC_MIN_VEHICLES 5→6 (5 vehicles is modest, not truly congested)
"""

import math
import uuid
from collections import defaultdict, deque
from typing import Optional


# ── Base thresholds ───────────────────────────────────────────────────────────
TTC_NEAR_MISS_S        = 1.0    # TTC < this → potential near_miss (normal mode)
TTC_ACCIDENT_S         = 0.5    # TTC < this AND bbox overlap → accident
MIN_DIST_DANGER_M      = 2.5    # metres edge-to-edge; distance gate (normal mode only)
DECEL_DANGER_MS2       = 6.0
DECEL_WINDOW_FRAMES    = 10
DECEL_SPEED_DROP_FRAC  = 0.35
PET_MAX_FRAMES         = 150
GRID_SIZE_PX           = 80
MIN_CONSECUTIVE_FRAMES = 4      # lowered: accidents are fast (1-3 frames); 8 missed them
MIN_BBOX_AREA_FRAC     = 0.05
MIN_CY_DIFF_FRAC       = 0.15
# SAME_LANE_CX_FRAC removed — F6 deleted; see note in module docstring
MIN_CLOSING_SPEED_KMH  = 15.0   # raised from 10; routine fluctuations < 10 km/h

KMH_TO_MS = 1.0 / 3.6

# ── F7: absolute speed floor ──────────────────────────────────────────────────
# At least one vehicle in the pair must exceed this to be flaggable.
# Uses uncalibrated speed_estimate (km/h), so the value is intentionally
# conservative — the true threshold is proportionally higher once calibrated.
MIN_SPEED_FLOOR_KMH    = 15.0

# ── F8: gap monotonicity ───────────────────────────────────────────────────────
# Track the last GAP_HISTORY_FRAMES edge-to-edge gaps per pair.
# Require at least GAP_MONO_THRESHOLD fraction of consecutive transitions to
# be closing (gap shrinking) before counting any frame as dangerous.
GAP_HISTORY_FRAMES     = 15
GAP_MONO_THRESHOLD     = 0.65   # ≥ 65 % of transitions must be closing

# ── F9: dense-traffic mode ────────────────────────────────────────────────────
DENSE_TRAFFIC_MIN_VEHICLES = 6   # raised from 5; 5 vehicles is not truly congested
DENSE_ROLL_FRAMES          = 30  # window size for rolling vehicle count
# Stricter thresholds applied in dense mode:
DENSE_TTC_S                = 0.7   # raised from 0.5; TTC 0.5-0.7s IS dangerous
DENSE_MIN_CLOSING_KMH      = 25.0  # serious relative speed required
DENSE_MIN_CONSECUTIVE      = 8     # lowered from 12; accidents are fast
DENSE_MIN_SPEED_KMH        = 20.0  # higher speed floor
# Distance gate is DISABLED in dense mode (proximity is normal in congestion).


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _bbox_edge_distance_px(b1: list, b2: list) -> float:
    """Pixel distance between the closest edges of two axis-aligned bboxes."""
    x1_min, y1_min, x1_max, y1_max = b1
    x2_min, y2_min, x2_max, y2_max = b2
    dx = max(0.0, max(x2_min - x1_max, x1_min - x2_max))
    dy = max(0.0, max(y2_min - y1_max, y1_min - y2_max))
    return math.sqrt(dx * dx + dy * dy)


def _boxes_overlap(b1: list, b2: list) -> bool:
    x1_min, y1_min, x1_max, y1_max = b1
    x2_min, y2_min, x2_max, y2_max = b2
    return not (x1_max < x2_min or x2_max < x1_min or
                y1_max < y2_min or y2_max < y1_min)


def _gap_based_ttc(curr_gap_px: float, prev_gap_px: float, fps: float) -> Optional[float]:
    """
    TTC in seconds from frame-to-frame gap change.
    pixels_per_meter cancels out — calibration-independent.
    Returns None if gap is stable or widening.
    """
    if curr_gap_px <= 0.0:
        return 0.0
    closing_rate = (prev_gap_px - curr_gap_px) * fps   # px/s; positive = approaching
    if closing_rate <= 0.0:
        return None
    return curr_gap_px / closing_rate


def _gap_trend_ok(history: deque) -> bool:
    """
    F8: return True if the gap history shows a sustained downward trend.

    Requires at least GAP_MONO_THRESHOLD of consecutive-frame transitions to
    be closing (gap[i+1] < gap[i]).  This distinguishes a genuine approach
    (monotonically shrinking gap) from stop-and-go oscillation, where the gap
    alternately closes and opens.

    Returns True (don't filter) if history is too short to judge.
    """
    if len(history) < 3:
        return True
    gaps = list(history)
    closing = sum(1 for a, b in zip(gaps, gaps[1:]) if b < a)
    return closing / (len(gaps) - 1) >= GAP_MONO_THRESHOLD


# ── Index builder ─────────────────────────────────────────────────────────────

def _build_frame_index(trajectories: list) -> dict:
    """frame_num -> list of {vehicle_id, vehicle_class, frame_data}."""
    index: dict = defaultdict(list)
    for traj in trajectories:
        for fd in traj["frames"]:
            index[fd["frame_num"]].append({
                "vehicle_id":    traj["vehicle_id"],
                "vehicle_class": traj["vehicle_class"],
                "frame_data":    fd,
            })
    return index


# ── Deceleration detection (relative, calibration-independent) ────────────────

def _detect_decelerations(trajectories: list, fps: float) -> list:
    """
    Per-vehicle sudden-deceleration events using relative speed drop.

    Flags if speed drops by more than DECEL_SPEED_DROP_FRAC (35 %) over
    DECEL_WINDOW_FRAMES — calibration-independent.  The reported
    deceleration_ms2 is approximate (uncalibrated speed), but detection
    relies only on the relative drop.
    """
    events = []
    dt = DECEL_WINDOW_FRAMES / fps
    for traj in trajectories:
        vid = traj["vehicle_id"]
        frames = traj["frames"]
        for i in range(DECEL_WINDOW_FRAMES, len(frames)):
            prev_fd = frames[i - DECEL_WINDOW_FRAMES]
            curr_fd = frames[i]
            v_before = prev_fd["speed_estimate"]
            v_after  = curr_fd["speed_estimate"]
            if v_before <= 0:
                continue
            drop_frac = (v_before - v_after) / v_before
            if drop_frac < DECEL_SPEED_DROP_FRAC:
                continue
            decel_ms2 = (v_before - v_after) * KMH_TO_MS / dt
            events.append({
                "vehicle_id":         vid,
                "frame_start":        prev_fd["frame_num"],
                "frame_end":          curr_fd["frame_num"],
                "timestamp_start_ms": prev_fd["timestamp_ms"],
                "timestamp_end_ms":   curr_fd["timestamp_ms"],
                "deceleration_ms2":   round(decel_ms2, 2),
            })
    return events


# ── PET ───────────────────────────────────────────────────────────────────────

def _compute_pet(trajectories: list) -> list:
    """Post-Encroachment Time between vehicle pairs sharing a spatial zone."""
    zone_events: dict = defaultdict(list)
    for traj in trajectories:
        vid = traj["vehicle_id"]
        frames = traj["frames"]
        prev_zones: set = set()
        for fd in frames:
            cx, cy = fd["center"]
            zone = (cx // GRID_SIZE_PX, cy // GRID_SIZE_PX)
            curr_zones = {zone}
            for z in curr_zones - prev_zones:
                zone_events[z].append((fd["frame_num"], vid, "enter"))
            for z in prev_zones - curr_zones:
                zone_events[z].append((fd["frame_num"], vid, "exit"))
            prev_zones = curr_zones
        for z in prev_zones:
            zone_events[z].append((frames[-1]["frame_num"] + 1, vid, "exit"))

    pet_list = []
    for zone, events in zone_events.items():
        events.sort()
        exits  = [(f, v) for f, v, e in events if e == "exit"]
        enters = [(f, v) for f, v, e in events if e == "enter"]
        for ex_frame, ex_vid in exits:
            for en_frame, en_vid in enters:
                if en_vid != ex_vid and 0 < en_frame - ex_frame <= PET_MAX_FRAMES:
                    pet_list.append({
                        "vehicle_a":     ex_vid,
                        "vehicle_b":     en_vid,
                        "zone":          zone,
                        "pet_frames":    en_frame - ex_frame,
                        "frame_a_exit":  ex_frame,
                        "frame_b_enter": en_frame,
                    })
    return pet_list


# ── Severity / confidence ─────────────────────────────────────────────────────

def _severity(ttc: Optional[float], dist_m: float, decel_ms2: float) -> str:
    score = 0
    if ttc is not None:
        if ttc < 0.5:   score += 3
        elif ttc < 1.0: score += 2
        else:           score += 1
    if dist_m < 1.0:                   score += 3
    elif dist_m < 1.5:                 score += 2
    elif dist_m < MIN_DIST_DANGER_M:   score += 1
    if decel_ms2 > 10.0:               score += 2
    elif decel_ms2 > DECEL_DANGER_MS2: score += 1
    if score >= 5: return "high"
    if score >= 3: return "medium"
    return "low"


def _confidence(ttc: Optional[float], dist_m: float, decel_ms2: float) -> float:
    signals = sum([
        ttc is not None and ttc < TTC_NEAR_MISS_S,
        dist_m < MIN_DIST_DANGER_M,
        decel_ms2 > DECEL_DANGER_MS2,
    ])
    return round(0.5 + signals * 0.15, 2)


# ── Public API ────────────────────────────────────────────────────────────────

def detect_incidents(
    trajectories: list,
    fps: float,
    pixels_per_meter: float,
    video_id: str,
    frame_width: int = 1280,
    frame_height: int = 720,
) -> list:
    """
    Analyse vehicle trajectories and return spec-compliant incident records.

    Logic layers (in evaluation order per pair per frame):
        F1  Skip both-tiny bboxes (far/small vehicles).
        F2  Skip same-depth pairs (similar cy → likely side-by-side).
        F5  Skip below minimum closing speed (mode-dependent threshold).
        F7  Skip if neither vehicle is moving meaningfully (speed floor).
        F3  Skip if gap is not actually dangerous (mode-dependent TTC / dist gate).
        F8  Skip if gap is oscillating rather than monotonically decreasing.
        F4  Only emit incident if danger persists for enough consecutive frames.
        F9  Dense-traffic mode: stricter on all thresholds, distance gate off.
        (F6 removed — was blocking intersection accidents; see module docstring)
    """
    if not trajectories:
        return []

    frame_area       = frame_width * frame_height
    min_bbox_area    = MIN_BBOX_AREA_FRAC * frame_area
    min_cy_diff_px   = MIN_CY_DIFF_FRAC * frame_height
    # Normal-mode closing-speed gate (px/s)
    min_closing_px_s = (MIN_CLOSING_SPEED_KMH * KMH_TO_MS) * pixels_per_meter

    frame_index  = _build_frame_index(trajectories)
    decel_events = _detect_decelerations(trajectories, fps)

    decel_by_vehicle: dict = defaultdict(list)
    for ev in decel_events:
        decel_by_vehicle[ev["vehicle_id"]].append(ev)

    # Per-pair state
    pair_streak:      dict = {}   # pair_key → {count, req_consecutive, metrics}
    pair_best:        dict = {}   # pair_key → best qualifying metrics
    pair_gap_history: dict = {}   # pair_key → deque[float] of last GAP_HISTORY_FRAMES gaps

    # F9: rolling vehicle count for dense-mode detection
    vehicle_count_roll: deque = deque(maxlen=DENSE_ROLL_FRAMES)

    def _merge_into_best(pk: tuple, m: dict) -> None:
        if pk not in pair_best:
            pair_best[pk] = dict(m)
            return
        pb = pair_best[pk]
        if m["min_ttc"] is not None and (pb["min_ttc"] is None or m["min_ttc"] < pb["min_ttc"]):
            pb["min_ttc"] = m["min_ttc"]
        if m["min_dist_m"]    < pb["min_dist_m"]:    pb["min_dist_m"]    = m["min_dist_m"]
        if m["max_decel_ms2"] > pb["max_decel_ms2"]: pb["max_decel_ms2"] = m["max_decel_ms2"]
        if m["max_rel_kmh"]   > pb["max_rel_kmh"]:   pb["max_rel_kmh"]   = m["max_rel_kmh"]
        pb["frame_end"]   = max(pb["frame_end"],   m["frame_end"])
        pb["ts_end"]      = max(pb["ts_end"],      m["ts_end"])
        pb["has_overlap"] = pb["has_overlap"] or m["has_overlap"]

    for frame_num in sorted(frame_index.keys()):
        vehicles = frame_index[frame_num]

        # ── F9: compute dense mode for this frame ─────────────────────────
        vehicle_count_roll.append(len(vehicles))
        avg_count = sum(vehicle_count_roll) / len(vehicle_count_roll)
        dense = avg_count >= DENSE_TRAFFIC_MIN_VEHICLES

        if dense:
            eff_ttc_thresh   = DENSE_TTC_S
            eff_closing_px_s = (DENSE_MIN_CLOSING_KMH * KMH_TO_MS) * pixels_per_meter
            eff_consecutive  = DENSE_MIN_CONSECUTIVE
            eff_speed_floor  = DENSE_MIN_SPEED_KMH
            eff_dist_gate    = False   # proximity is normal in congestion
        else:
            eff_ttc_thresh   = TTC_NEAR_MISS_S
            eff_closing_px_s = min_closing_px_s
            eff_consecutive  = MIN_CONSECUTIVE_FRAMES
            eff_speed_floor  = MIN_SPEED_FLOOR_KMH
            eff_dist_gate    = True

        if len(vehicles) < 2:
            # Flush all active streaks — no pairs possible this frame
            for pk in list(pair_streak.keys()):
                if pair_streak[pk]["count"] >= pair_streak[pk]["req_consecutive"]:
                    _merge_into_best(pk, pair_streak[pk]["metrics"])
                del pair_streak[pk]
            continue

        active_this_frame: set = set()

        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                va = vehicles[i]
                vb = vehicles[j]
                aid = va["vehicle_id"]
                bid = vb["vehicle_id"]
                pair_key = (min(aid, bid), max(aid, bid))

                fd_a = va["frame_data"]
                fd_b = vb["frame_data"]

                # ── F1: skip both-tiny bboxes ────────────────────────────────
                area_a = (fd_a["bbox"][2] - fd_a["bbox"][0]) * (fd_a["bbox"][3] - fd_a["bbox"][1])
                area_b = (fd_b["bbox"][2] - fd_b["bbox"][0]) * (fd_b["bbox"][3] - fd_b["bbox"][1])
                if area_a < min_bbox_area and area_b < min_bbox_area:
                    continue

                # ── F2: skip same-depth pairs (side-by-side in same lane) ────
                cy_diff = abs(fd_a["center"][1] - fd_b["center"][1])
                if cy_diff < min_cy_diff_px:
                    continue

                # ── Gap computation + gap history ────────────────────────────
                curr_gap_px = _bbox_edge_distance_px(fd_a["bbox"], fd_b["bbox"])

                if pair_key not in pair_gap_history:
                    pair_gap_history[pair_key] = deque(maxlen=GAP_HISTORY_FRAMES)
                hist = pair_gap_history[pair_key]
                prev_gap_px = hist[-1] if hist else None
                hist.append(curr_gap_px)

                ttc = None
                gap_closing_px_s = 0.0
                if prev_gap_px is not None:
                    gap_closing_px_s = (prev_gap_px - curr_gap_px) * fps
                    ttc = _gap_based_ttc(curr_gap_px, prev_gap_px, fps)

                # ── F5: minimum closing speed (mode-dependent) ────────────────
                if abs(gap_closing_px_s) < eff_closing_px_s:
                    continue

                # ── F7: absolute speed floor ──────────────────────────────────
                # At least one vehicle must be moving meaningfully.
                # Eliminates slow-creep traffic where even a 100% relative
                # closing speed represents only a few km/h of actual motion.
                speed_a = fd_a["speed_estimate"]
                speed_b = fd_b["speed_estimate"]
                if max(speed_a, speed_b) < eff_speed_floor:
                    continue

                dist_m  = curr_gap_px / pixels_per_meter
                closing = gap_closing_px_s > 0

                # ── F3: danger check (mode-dependent) ────────────────────────
                if eff_dist_gate:
                    # Normal mode: either genuinely close AND closing, or TTC alarm
                    dangerous = (
                        (dist_m < MIN_DIST_DANGER_M and closing) or
                        (ttc is not None and ttc < eff_ttc_thresh)
                    )
                else:
                    # Dense mode: distance gate off; only a very tight TTC counts
                    dangerous = ttc is not None and ttc < eff_ttc_thresh

                if not dangerous:
                    continue

                # ── F8: gap monotonicity ──────────────────────────────────────
                # Reject oscillating stop-and-go gaps; require sustained closing.
                if not _gap_trend_ok(hist):
                    continue

                # ── Max deceleration near this frame ─────────────────────────
                max_decel = 0.0
                for vid in (aid, bid):
                    for dev in decel_by_vehicle.get(vid, []):
                        if dev["frame_start"] <= frame_num <= dev["frame_end"] + DECEL_WINDOW_FRAMES:
                            max_decel = max(max_decel, dev["deceleration_ms2"])

                overlapping   = _boxes_overlap(fd_a["bbox"], fd_b["bbox"])
                rel_speed_kmh = (abs(gap_closing_px_s) / pixels_per_meter) * 3.6

                active_this_frame.add(pair_key)

                # ── F4: streak tracking ───────────────────────────────────────
                frame_metrics = {
                    "min_ttc":       ttc,
                    "min_dist_m":    dist_m,
                    "max_decel_ms2": max_decel,
                    "max_rel_kmh":   rel_speed_kmh,
                    "frame_start":   frame_num,
                    "frame_end":     frame_num,
                    "ts_start":      fd_a["timestamp_ms"],
                    "ts_end":        fd_a["timestamp_ms"],
                    "has_overlap":   overlapping,
                    "vehicles":      [aid, bid],
                }

                if pair_key not in pair_streak:
                    pair_streak[pair_key] = {
                        "count":           1,
                        "req_consecutive": eff_consecutive,
                        "metrics":         frame_metrics,
                    }
                else:
                    run = pair_streak[pair_key]
                    run["count"] += 1
                    # Update to current mode's requirement (most recent frame wins)
                    run["req_consecutive"] = eff_consecutive
                    m = run["metrics"]
                    if ttc is not None and (m["min_ttc"] is None or ttc < m["min_ttc"]):
                        m["min_ttc"] = ttc
                    if dist_m        < m["min_dist_m"]:    m["min_dist_m"]    = dist_m
                    if max_decel     > m["max_decel_ms2"]: m["max_decel_ms2"] = max_decel
                    if rel_speed_kmh > m["max_rel_kmh"]:   m["max_rel_kmh"]   = rel_speed_kmh
                    m["frame_end"]   = frame_num
                    m["ts_end"]      = fd_a["timestamp_ms"]
                    m["has_overlap"] = m["has_overlap"] or overlapping

        # Close out streaks for pairs not dangerous this frame
        for pk in list(pair_streak.keys()):
            if pk not in active_this_frame:
                if pair_streak[pk]["count"] >= pair_streak[pk]["req_consecutive"]:
                    _merge_into_best(pk, pair_streak[pk]["metrics"])
                del pair_streak[pk]

    # Flush still-active streaks at end of video
    for pk, run in pair_streak.items():
        if run["count"] >= run["req_consecutive"]:
            _merge_into_best(pk, run["metrics"])

    # ── Build incident records ─────────────────────────────────────────────────
    incidents = []
    for pm in pair_best.values():
        ttc    = pm["min_ttc"]
        dist_m = pm["min_dist_m"]
        decel  = pm["max_decel_ms2"]

        if pm["has_overlap"] and ttc is not None and ttc < TTC_ACCIDENT_S:
            inc_type = "accident"
        elif (ttc is not None and ttc < TTC_NEAR_MISS_S) or dist_m < 1.0:
            inc_type = "near_miss"
        else:
            inc_type = "risky_interaction"

        incidents.append({
            "incident_id":        str(uuid.uuid4()),
            "video_id":           video_id,
            "type":               inc_type,
            "severity":           _severity(ttc, dist_m, decel),
            "timestamp_start_ms": pm["ts_start"],
            "timestamp_end_ms":   pm["ts_end"],
            "frame_start":        pm["frame_start"],
            "frame_end":          pm["frame_end"],
            "vehicles_involved":  sorted(pm["vehicles"]),
            "metrics": {
                "min_ttc":              round(ttc, 3) if ttc is not None else None,
                "min_distance_meters":  round(dist_m, 3),
                "max_deceleration_ms2": round(decel, 3),
                "relative_speed_kmh":   round(pm["max_rel_kmh"], 2),
            },
            "causal_factors": [],
            "confidence":         _confidence(ttc, dist_m, decel),
        })

    sev_order = {"high": 0, "medium": 1, "low": 2}
    incidents.sort(key=lambda x: (sev_order[x["severity"]], x["timestamp_start_ms"]))
    return incidents
