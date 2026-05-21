# ARGUS — Autonomous Road Guard Unified Surveillance

> **Live demo:** [argus-platform.vercel.app](https://argus-platform.vercel.app) — frontend prototype, mock data.

---

## What Is ARGUS?

ARGUS is an end-to-end traffic anomaly detection system that processes dashcam or CCTV footage and automatically identifies dangerous events — accidents, near-misses, sudden braking, unsafe lane changes — with no human review.

Core research question: **can a purely computer-vision system, with no GPS or road-map data, reliably detect traffic incidents in real time from raw video?**

That requires solving three problems simultaneously:
1. Detecting and classifying every vehicle in every frame under real-world conditions (occlusion, lighting, shadows, motion blur)
2. Maintaining consistent vehicle identity across frames under partial occlusion
3. Inferring intent and danger from raw pixel trajectories — no labeled incident data, no LiDAR, no depth sensor

---

## System Architecture

```
Video Input
    │
    ▼
VehicleDetector                   ml_pipeline/detection.py
  YOLO12x (ultralytics 8.4.x)    ─ finetuned on BDD100K + IDD
  confidence=0.35                 ─ auto device: CUDA → MPS → CPU
  TemporalConfirmation            ─ 2-frame streak suppresses phantom detections
  Auto class-map resolution       ─ COCO vs finetuned weights, detected at load time
    │
    ▼
VehicleTracker                    ml_pipeline/tracking.py
  ByteTrack (supervision)         ─ IoU-only, no appearance model
  track_thresh=0.25               ─ recovers low-confidence detections
  match_thresh=0.80               ─ handles occlusion and frame drops
  max_age=30                      ─ re-ID across ~1 second of occlusion
    │
    ▼
DepthEstimator                    ml_pipeline/depth.py
  Depth-Anything-V2-Small         ─ ~25 ms/frame on GPU
  bbox_depth()                    ─ median depth of central 50% of each bbox
  Closing-rate signal             ─ geometry-independent TTC supplement
    │
    ▼
TrajectoryBuilder                 ml_pipeline/trajectory.py
  4-state Kalman [cx, cy, vx, vy] ─ eliminates centroid jitter from YOLO noise
  2-state Kalman [d, vd]          ─ depth rate-of-change per frame (Depth-Anything)
  Dual-signal TTC                 ─ pixel convergence + depth closing rate
  Graceful degradation            ─ depth_map=None → 2D-only mode
    │
    ▼
InteractionScorer                 ml_pipeline/interaction.py
  9-filter anomaly logic:
    TTC  (Time-to-Collision)       ─ sub-2s threshold triggers near-miss
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

**Why not LiDAR or GPS?** ARGUS targets dashcam and fixed CCTV — neither carries depth or map data. The entire pipeline runs on 2D pixel coordinates scaled by a calibrated `pixels_per_meter` constant, with optional monocular depth from Depth-Anything-V2.

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
| Multi-GPU training | Complex | Native | **Native** |

YOLO12x introduces Area Attention — attention-based feature fusion without full transformer overhead. Better than YOLO11x on small occluded objects at no inference speed cost.

### Class Map — Pretrained vs Finetuned

`detection.py` auto-detects which scheme is active from `model.names` at load time:

```python
# ml_pipeline/constants.py
COCO_VEHICLE_CLASSES      = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
FINETUNED_VEHICLE_CLASSES = {0: "car", 1: "motorcycle", 2: "bus", 3: "truck", 4: "bicycle"}
```

### Depth — Depth-Anything-V2-Small

```python
from ml.ml_pipeline.depth import DepthEstimator
estimator  = DepthEstimator()                   # loads once, ~25 ms/frame on GPU
depth_map  = estimator.estimate(frame_rgb)       # H×W float32 [0=near, 1=far]
d          = estimator.bbox_depth(depth_map, bbox_ltrb)  # scalar closing rate
```

### Tracker — ByteTrack

ByteTrack via `supervision` (IoU-only, no appearance model):
- No GPU-heavy embedding — runs on CPU while YOLO occupies the GPU
- Low-confidence detection recycling recovers partially-occluded vehicles
- Track age 30 frames re-IDs across ~1 s of full occlusion (e.g. behind a truck)

---

## Training

**Model:** YOLO12x (ultralytics 8.4.x)  
**Datasets:** BDD100K (~70k train images) + IDD Indian Driving Dataset  
**Target classes:** car · motorcycle · bus · truck · bicycle  
**Training platform:** Kaggle (T4 × 2 GPU)

### Run 1 — Completed (baseline, bugs present)

| Metric | Value |
|---|---|
| Epochs | 50 |
| Final mAP50 | 0.498 |
| Final mAP50-95 | 0.325 |
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

*Metrics on hard-mining validation set (4k images). HM-A dip is expected — harder examples lower apparent metrics but improve generalisation. HM-B recovery to 0.992 confirms successful mining.*

Final checkpoint: `runs/s25_hmb/argus_s25_hmb/weights/best.pt`

### S3 — Planned

Notebook: `notebooks/argus_s3.ipynb`. Full evaluation on held-out general set + optional distillation to nano model.

---

## Project Status

| Stage | Task | Status |
|---|---|---|
| Week 1 | Pipeline scaffold, YOLOv8 baseline | ✅ Done |
| Week 2 | ByteTrack integration, anomaly scoring | ✅ Done |
| Week 3 | RF-DETR → YOLO12x migration | ✅ Done |
| Week 3 | ML pipeline refactor — constants, class-map auto-detection | ✅ Done |
| Week 3 | Finetuning Run 1 — completed, root cause diagnosed | ✅ Done |
| Week 3 | Frontend — dark editorial theme, ambient sound, Vercel deploy | ✅ Done |
| Week 4 | S2 — full BDD100K + IDD, fixed class remapping | ✅ Done |
| Week 4 | S2.5 — hard-example mining (POL + HM-A + HM-B) | ✅ Done |
| Week 4 | Depth-Anything-V2 integration (`depth.py`) | ✅ Done |
| Week 4 | Trajectory Kalman filter + dual-signal TTC | ✅ Done |
| Week 5 | S3 — general eval + distillation | ⏳ Planned |
| Week 5 | Upload page + live backend API | ⏳ Planned |
| Week 5 | Jetson TRT export + latency benchmarking | ⏳ Planned |

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
TTC from monocular video is ill-posed without depth. ARGUS combines 2D bounding-box convergence + Depth-Anything-V2 closing rate. A paper could characterise the error model and when 2D+monocular depth TTC is sufficient for safety alerts.

**6. Temporal Confirmation as a Precision Filter**
The 2-frame `TemporalConfirmation` streak filter significantly cuts phantom detections at low latency. Can be generalised and benchmarked as a lightweight post-processing stage for any detector.

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
│   ├── argus_s2.ipynb         # S2 training — full BDD100K + IDD, fixed remapping
│   ├── argus_s2_cont.ipynb    # S2 continuation
│   ├── argus_s2_5.ipynb       # S2.5 — hard-example mining (POL + HM-A + HM-B)
│   └── argus_s3.ipynb         # S3 — general eval + distillation (planned)
├── ui/                        # Frontend — deployed to Vercel
│   ├── index.html
│   ├── dashboard.html
│   └── js/
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt

python - <<'EOF'
from ml.ml_pipeline import analyze_video
result = analyze_video("path/to/video.mp4")
print(result["incidents"])
EOF
```

Pretrained YOLO12x weights download automatically on first run. For finetuned weights, place the S2.5 checkpoint in the project root:

```python
from ml.ml_pipeline import analyze_video
result = analyze_video("video.mp4", model_path="argus_s25_best.pt")
```

---

*BITS Pilani · Semester II 2025–26 · Veer Raghuvanshi*
