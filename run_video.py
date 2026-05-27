#!/usr/bin/env python3
"""
ARGUS — analyse a dashcam or CCTV video for traffic incidents.

Usage:
    python run_video.py VIDEO [options]

Examples:
    python run_video.py dashcam.mp4
    python run_video.py footage.mp4 --model argus_s3.pt --depth
    python run_video.py footage.mp4 --model argus_s3.pt --out report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ARGUS traffic incident analyser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("video", help="Path to video file (MP4, AVI, MOV)")
    p.add_argument(
        "--model", default="yolo12x.pt",
        help="YOLO weights file (default: yolo12x.pt; use argus_s3.pt for finetuned)",
    )
    p.add_argument(
        "--confidence", type=float, default=0.40,
        help="Detection confidence threshold (default: 0.40)",
    )
    p.add_argument(
        "--pixels-per-meter", type=float, default=30.0, dest="ppm",
        help="Pixels-per-metre calibration constant (default: 30.0)",
    )
    p.add_argument(
        "--depth", action="store_true",
        help="Enable Depth-Anything-V2-Small for dual-signal TTC (+~25 ms/frame on GPU)",
    )
    p.add_argument(
        "--out", metavar="FILE",
        help="Write JSON report to FILE instead of stdout",
    )
    p.add_argument("--quiet", action="store_true", help="Suppress progress output")
    return p.parse_args()


def main() -> None:
    args = _parse()

    if not Path(args.video).exists():
        sys.exit(f"error: video not found: {args.video}")

    from ml.ml_pipeline import analyze_video

    def _progress(pct: int) -> None:
        if not args.quiet:
            print(f"\r  Analysing: {pct:3d}%", end="", flush=True)

    if not args.quiet:
        depth_note = " + Depth-Anything-V2" if args.depth else ""
        print(f"ARGUS  {Path(args.video).name}{depth_note}")
        print(f"  model={args.model}  conf={args.confidence}  ppm={args.ppm}")

    result = analyze_video(
        args.video,
        progress_callback=_progress,
        model_path=args.model,
        confidence=args.confidence,
        pixels_per_meter=args.ppm,
        use_depth=args.depth,
    )

    if not args.quiet:
        print()
        n_inc = len(result["incidents"])
        n_trj = len(result["trajectories"])
        print(f"  {n_trj} trajectories  ·  {n_inc} incident(s) detected")
        sev_order = {"high": 0, "medium": 1, "low": 2}
        for inc in sorted(result["incidents"], key=lambda x: sev_order[x["severity"]]):
            sev  = inc["severity"].upper()
            ts   = inc["timestamp_start_ms"] / 1000
            kind = inc["type"].replace("_", " ")
            ttc  = inc["metrics"].get("min_ttc")
            ttc_s = f"  TTC={ttc:.2f}s" if ttc else ""
            print(f"    [{sev}] {ts:6.1f}s  {kind}{ttc_s}")

    payload = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(payload)
        if not args.quiet:
            print(f"  report → {args.out}")
    else:
        print(payload)


if __name__ == "__main__":
    main()
