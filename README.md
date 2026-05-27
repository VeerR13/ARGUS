# ARGUS — Autonomous Road Guard Unified Surveillance

End-to-end traffic anomaly detection from dashcam or CCTV footage. No GPS, no LiDAR, no labeled incident data.

**Live demo:** [argus-platform.vercel.app](https://argus-platform.vercel.app) — frontend prototype, mock data.

---

## What It Does

ARGUS processes raw video and automatically identifies dangerous events — near-misses, sudden braking, unsafe lane changes — using a pure computer-vision pipeline.

**Core research question:** Can a purely CV system, with no GPS or road-map data, reliably detect traffic incidents from raw video in real time?

That requires solving three problems simultaneously:
1. Detect and classify every vehicle under real-world conditions (occlusion, lighting, motion blur)
2. Maintain consistent vehicle identity across frames under partial occlusion
3. Infer intent and danger from raw pixel trajectories — no labeled incident data, no depth sensor

---

## Dashcam Report Product

The product direction: **upload dashcam footage → receive a driving safety report with timestamped incident analysis.**

```
"At 00:02:34 you looked away from the road for 1.8 s.
 During this window a motorcycle was entering the intersection
 on your right with a TTC of 2.1 s — near-miss."
```

### Architecture

```
Forward Camera                          Driver Camera (optional)
      │                                        │
      ▼                                        ▼
VehicleDetector                         MediaPipe Face Mesh
  YOLO12x → ByteTrack                     468-landmark head pose
  TrajectoryBuilder (Kalman)             PERCLOS drowsiness metric
  InteractionScorer (9 filters)            (>80% closure for >0.5 s)
  DepthEstimator (dual-signal TTC)       Gaze deviation (>30° = away)
      │                                        │
      ▼                                        ▼
  forward_incidents[]                   driver_events[]
  [{ts, type, ttc, severity}]           [{start_ts, end_ts, type}]
              │                                │
              └──────────┬─────────────────────┘
                         ▼
              Temporal Sync Engine
                GPS metadata → audio cross-correlation → manual offset
                         │
                         ▼
              Correlation Engine
                match driver inattention windows ↔ forward-cam incidents
                classify: causal / near-miss / isolated
                         │
                         ▼
              Report Generator
                Claude API (sonnet-4-6) → natural language narrative
                per-incident: what, when, severity, driver contribution
                trip summary: safety score, top 3 risks
```

### Components

| Component | Status | Notes |
|---|---|---|
| Forward cam pipeline | ✅ Done | Full ARGUS pipeline — detection → tracking → TTC → incident scoring |
| Driver cam pipeline | ⏳ Planned | MediaPipe Face Mesh + PERCLOS, ~50 lines of Python, no custom training needed |
| Temporal sync | ⏳ Planned | GPS MP4 metadata → FFT audio cross-correlation → manual fallback |
| Correlation engine | ⏳ Planned | Match gaze-away windows to forward-cam incident timestamps |
| Report generation | ⏳ Planned | Claude API converts JSON incidents to causal narrative |

### Why No Custom Training for Driver Cam

MediaPipe Face Mesh + PERCLOS covers 80% of the consumer use case (daytime, no sunglasses, forward-facing driver). The novel contribution is the **correlation engine** — matching driver attention failures to forward-cam incident events. No drowsiness model training required for an MVP.

---

## Forward Cam Pipeline

```
Video Input
    │
    ▼
VehicleDetector                   ml/ml_pipeline/detection.py
  YOLO12x (ultralytics 8.4.x)    ─ finetuned on BDD100K + IDD + KITTI (S3)
  confidence=0.35                 ─ auto device: CUDA → MPS → CPU
  TemporalConfirmation            ─ 2-frame streak suppresses phantom detections
  Auto class-map resolution       ─ COCO vs finetuned weights detected at load time
    │
    ▼
VehicleTracker                    ml/ml_pipeline/tracking.py
  ByteTrack (supervision)         ─ IoU-only, no appearance model
  track_thresh=0.25               ─ recovers low-confidence detections
  match_thresh=0.80               ─ handles occlusion and frame drops
  max_age=30                      ─ re-ID across ~1 s of occlusion
    │
    ▼
DepthEstimator                    ml/ml_pipeline/depth.py
  Depth-Anything-V2-Small         ─ ~25 ms/frame on GPU (opt-in via use_depth=True)
  bbox_depth()                    ─ median depth of central 50% of each bbox
  Closing-rate signal             ─ geometry-independent TTC supplement
    │
    ▼
TrajectoryBuilder                 ml/ml_pipeline/trajectory.py
  4-state Kalman [cx, cy, vx, vy] ─ eliminates centroid jitter from YOLO noise
  2-state Kalman [d, vd]          ─ depth rate-of-change per frame
  Dual-signal TTC                 ─ pixel convergence + depth closing rate
  Graceful degradation            ─ depth_map=None → 2D-only mode
    │
    ▼
InteractionScorer                 ml/ml_pipeline/interaction.py
  9-filter anomaly logic:
    TTC  (Time-to-Collision)       ─ sub-2 s threshold triggers near-miss
    PET  (Post-Encroachment Time)  ─ conflict point crossing detection
    Deceleration spikes            ─ emergency braking detection
    Gap monotonicity               ─ closing gap without intent to stop
    Lateral proximity              ─ unsafe lane change / sideswipe risk
    Speed differential             ─ dangerous overtake detection
    Trajectory divergence          ─ evasive manoeuvre signature
    Stationarity in traffic        ─ stalled vehicle / accident remnant
    Cluster density change         ─ sudden density spike → pile-up risk
    │
    ▼
Incident Report                   { trajectories, incidents, timestamps }
```

### Design Decisions

**Why not LiDAR or GPS?** ARGUS targets dashcam and fixed CCTV — neither carries depth or map data. The pipeline runs on 2D pixel coordinates scaled by a calibrated `pixels_per_meter` constant, with optional monocular depth from Depth-Anything-V2.

**Why ByteTrack over DeepSORT?** DeepSORT's appearance model was trained on pedestrian re-ID datasets and produced 65%+ false positives on vehicles. ByteTrack's IoU-only association runs on CPU alongside YOLO on GPU and handles occlusion better in dense traffic.

**Why 9 heuristic filters over a learned incident classifier?** No large-scale labeled dashcam incident dataset exists. The filter bank encodes physics (TTC, PET) and traffic law (gap, lateral rules) — it generalises without overfitting to a single camera or road type.

**Why add monocular depth?** Pixel-gap TTC saturates when a vehicle already fills most of the frame. Depth-Anything-V2's closing rate `vd` gives a second signal that remains informative even at close range.

---

## ML Stack

### Detector — YOLO12x

| | RF-DETR Base | YOLO11x | **YOLO12x** |
|---|---|---|---|
| COCO mAP50-95 | 53.3 | 59.0 | **59.1** |
| MPS inference | ~24 fps | ~32 fps | **~38 fps** |
| Jetson (TRT) | ~12 fps | ~55 fps | **~70 fps** |

YOLO12x introduces Area Attention — attention-based feature fusion without full transformer overhead. Better than YOLO11x on small occluded objects at no inference speed cost.

### Class Map

```python
# ml/ml_pipeline/constants.py
COCO_VEHICLE_CLASSES      = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
FINETUNED_VEHICLE_CLASSES = {0: "car", 1: "motorcycle", 2: "bus", 3: "truck", 4: "bicycle"}
```

Auto-detected from `model.names` at load time — no manual flag required.

---

## Training

**Model:** YOLO12x (ultralytics 8.4.x) · **Platform:** Kaggle T4 × 2

### Run 1 — Completed (baseline, bugs present)

| Metric | Value |
|---|---|
| Epochs | 50 |
| Final mAP50 | 0.498 |
| Motorcycle AP50 | **0.000** |

**Root cause:** BDD100K class IDs passed without remapping (motorcycle absent); small dataset subset (~7.6k images).

### S2 — Completed (bugs fixed)

Both bugs fixed: full BDD100K (~70k images) + correct class remapping (`{2:0, 3:2, 4:3, 5:4, 6:1}`). Checkpoint: `argus_s2.pt`.

### S2.5 — Completed (hard-example mining, 3 phases)

Fine-tuned from `argus_s2.pt` on a 50k proximity-selected hard-example subset.

| Phase | Epochs | mAP50 | mAP50-95 | Notes |
|---|---|---|---|---|
| POL (polish) | 3 | **0.994** | **0.920** | Full hard set, warm-start |
| HM-A | 2 | 0.986 | 0.871 | Hardest 20% subset |
| HM-B | 2 | 0.992 | 0.904 | Hardest 20% subset |

*Metrics on hard-mining validation set (4k images). HM-A dip is expected — harder examples lower apparent metrics but improve generalisation.*

Checkpoint: `runs/s25_hmb/argus_s25_hmb/weights/best.pt`

### S3 — In Progress (KITTI integration, drop UAVDT)

UAVDT overhead UAV data removed — it contaminated dashcam feature representations (truck mAP50=0.138 on UAVDT val). KITTI dashcam dataset added with 3× truck oversample.

| Config | Value |
|---|---|
| Base checkpoint | argus_s25_hmb/best.pt |
| Datasets | BDD100K (~70k) + IDD (~7k) + KITTI (~6.8k train, ×3 truck) |
| Resolution | 1280px |
| Epochs | 45 · lr0=5e-5 (fine-tune) |
| Target | Truck mAP50 0.138 → **0.45+** |

Notebook: `notebooks/argus_s3.ipynb`

---

## Research Prospects

**1. Unsupervised Incident Detection Without Labels**  
The 9-filter InteractionScorer detects anomalies using physics (TTC, PET) and traffic law with zero labeled incident data. Benchmark against supervised approaches on DoTA, DADA-2000.

**2. Cross-Domain Vehicle Detection: Western to Indian Traffic**  
BDD100K (US/diverse) + IDD (Indian) exposes domain shift — different vehicle morphology (autorickshaws, two-wheelers), density, and road structure. Class remapping + resampling pipeline is a reproducible contribution.

**3. Class Imbalance in Multi-Source Traffic Datasets**  
Motorcycle underrepresentation + label remapping bug is a strong motivating case. Run 1 → S2 diagnosis is a clean ablation.

**4. Benchmarking YOLO12x vs RF-DETR for Edge Traffic Surveillance**  
Systematic comparison on dashcam/CCTV conditions — not COCO benchmarks. Latency, mAP, deployment on Jetson.

**5. Real-Time Near-Miss Detection Using Monocular TTC Estimation**  
TTC from monocular video is ill-posed without depth. ARGUS combines 2D bounding-box convergence + Depth-Anything-V2 closing rate. Characterise the error model and when 2D+monocular depth TTC is sufficient for safety alerts.

**6. Driver Attention × Forward-Cam Incident Correlation**  
Novel end-to-end causal pipeline: gaze-away windows (PERCLOS + head pose) matched to forward-cam incident timestamps. No open-source system does this end-to-end.

---

## Setup

```bash
pip install -r requirements.txt
```

**CLI:**
```bash
# Pretrained YOLO12x (downloads automatically on first run)
python run_video.py dashcam.mp4

# S3 finetuned weights — best dashcam/truck accuracy
python run_video.py dashcam.mp4 --model argus_s3.pt

# With monocular depth for dual-signal TTC
python run_video.py dashcam.mp4 --model argus_s3.pt --depth --out report.json
```

**Python API:**
```python
from ml.ml_pipeline import analyze_video

result = analyze_video("dashcam.mp4", model_path="argus_s3.pt")
result = analyze_video("dashcam.mp4", model_path="argus_s3.pt", use_depth=True)
print(result["incidents"])
```

**Individual components:**
```python
from ml.ml_pipeline import VehicleDetector, VehicleTracker, TrajectoryBuilder
from ml.ml_pipeline import detect_incidents
from ml.ml_pipeline.depth import DepthEstimator
```

---

## Repo Structure

```
ARGUS/
├── ml/
│   ├── ml_pipeline/
│   │   ├── constants.py       # Shared class ID maps (COCO + finetuned)
│   │   ├── detection.py       # VehicleDetector — YOLO12x, auto class-map
│   │   ├── tracking.py        # VehicleTracker — ByteTrack via supervision
│   │   ├── trajectory.py      # TrajectoryBuilder — 4-state Kalman + depth Kalman
│   │   ├── depth.py           # DepthEstimator — Depth-Anything-V2-Small
│   │   └── interaction.py     # InteractionScorer — 9-filter anomaly logic
│   └── eval_real.py           # Eval against CVAT YOLO annotations
├── notebooks/
│   ├── argus_s2.ipynb         # S2 — full BDD100K + IDD, fixed class remapping
│   ├── argus_s2_5.ipynb       # S2.5 — hard-example mining (POL + HM-A + HM-B)
│   ├── argus_s3.ipynb         # S3 — KITTI integration, drop UAVDT (in progress)
│   └── argus_final.ipynb      # Eval — UAVDT + VisDrone held-out metrics
├── ui/                        # Frontend — deployed to Vercel
├── run_video.py               # CLI — analyse any video from the command line
├── requirements.txt
└── README.md
```

---

*BITS Pilani · Semester II 2025–26 · Veer Raghuvanshi*
