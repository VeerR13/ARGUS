# ARGUS — Autonomous Road Guard Unified Surveillance

> **Live demo:** [argus-platform.vercel.app](https://argus-platform.vercel.app) — frontend prototype, mock data. Run 2 finetuning in progress.

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
TrajectoryBuilder                 ml_pipeline/trajectory.py
  Per-track position history      ─ center positions, pixel-displacement speed
  Speed smoothing (EMA)           ─ suppresses jitter from detection jitter
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

**Why not LiDAR or GPS?** ARGUS targets dashcam and fixed CCTV — neither carries depth or map data. The entire pipeline runs on 2D pixel coordinates scaled by a calibrated `pixels_per_meter` constant.

**Why ByteTrack over DeepSORT?** DeepSORT's appearance model was trained on pedestrian re-ID datasets and produced 65%+ false positives on vehicles. ByteTrack's IoU-only association runs on CPU alongside YOLO on GPU and handles occlusion better in dense traffic.

**Why 9 heuristic filters over a learned incident classifier?** No large-scale labeled dashcam incident dataset exists. The filter bank encodes physics (TTC, PET) and traffic law (gap, lateral rules) — it generalises without overfitting to a single camera or road type.

---

## ML Stack

### Detector — YOLO12x (migrated from RF-DETR)

The project began with RF-DETR (transformer-based detector). After Week 2, migrated to **YOLO12x** for three reasons:

| | RF-DETR Base | YOLO11x | **YOLO12x** |
|---|---|---|---|
| COCO mAP50-95 | 53.3 | 59.0 | **59.1** |
| MPS inference | ~24 fps | ~32 fps | **~38 fps** |
| Jetson (TRT) | ~12 fps | ~55 fps | **~70 fps** |
| Training convergence | Slow (50+ epochs) | Fast | Fast |
| Edge export | Manual | ONNX/TRT | **ONNX/TRT** |
| Multi-GPU training | Complex | Native | **Native** |

YOLO12x introduces attention-based feature fusion (Area Attention) without full transformer overhead — better than YOLO11x on small occluded objects at no inference speed cost.

### Class Map — Pretrained vs Finetuned

`detection.py` auto-detects which scheme is active from `model.names` at load time:

```python
# ml_pipeline/constants.py
COCO_VEHICLE_CLASSES      = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
FINETUNED_VEHICLE_CLASSES = {0: "car", 1: "motorcycle", 2: "bus", 3: "truck", 4: "bicycle"}
```

### Tracker — ByteTrack

ByteTrack via `supervision` (IoU-only, no appearance model):
- No GPU-heavy embedding — runs on CPU while YOLO occupies the GPU
- Low-confidence detection recycling recovers partially-occluded vehicles
- Track age 30 frames re-IDs across ~1 s of full occlusion (e.g. behind a truck)

---

## Training

**Model:** YOLO12x (ultralytics 8.4.x)  
**Datasets:** BDD100K (~15k train images capped) + IDD Indian Driving Dataset (~10k)  
**Target classes:** car · motorcycle · bus · truck · bicycle  
**Training platform:** Lightning AI (T4 GPU, persistent studio)

### Run 1 — Completed

| Metric | Value |
|---|---|
| Epochs | 50 |
| Batch | 16 (T4 × 2) |
| imgsz | 640 |
| Final mAP50 | 0.498 |
| Final mAP50-95 | 0.325 |
| Motorcycle AP50 | **0.000** |

**Root cause of plateau:** Two compounding bugs —
1. BDD100K labels were copied raw without class ID remapping. BDD's `car=2, motorcycle=6, bus=4` were passed directly to YOLO as `class 2 = bus, class 6 = (dropped, out of range)`. Motorcycle was entirely absent from training.
2. The `a7madmostafa/bdd100k-yolo` Kaggle dataset contains ~7.6k training images — a small subset of full BDD100K (70k images).

### Run 2 — In Progress (Lightning AI)

Both bugs fixed:

- **Class remapping:** `argus_yolo12x_run2.ipynb` and `argus_lightning_train.py` read BDD's `data.yaml`, auto-detect the class ordering, and remap every label file on the fly before training. Zero out-of-range class IDs guaranteed by sanity check.
- **Dataset:** Full `a7madmostafa/bdd100k-yolo` dataset (~70k images) + IDD 20k, running on Lightning AI with persistent background execution.

| Config | Value |
|---|---|
| Epochs | 60 |
| Batch | 8 (T4 single) |
| imgsz | 640 |
| Motorcycle resampling | ×2 (corrects class imbalance) |
| Warm-start | `yolo12x.pt` pretrained COCO |
| Training script | `ml/training/argus_lightning_train.py` |

### Expected Results (Run 2)

| Metric | Pretrained COCO | Run 1 | **Run 2 target** |
|---|---|---|---|
| mAP50 | ~65–68% | 49.8% | **~78–82%** |
| mAP50-95 | ~42–45% | 32.5% | **~52–56%** |
| Car AP50 | ~80% | ~62% | **~88%** |
| Motorcycle AP50 | ~55% | 0.0% | **~67%** |
| Bus AP50 | ~68% | ~58% | **~79%** |
| Truck AP50 | ~62% | ~54% | **~74%** |
| Bicycle AP50 | ~50% | ~41% | **~63%** |

---

## Project Status

| Week | Task | Status |
|---|---|---|
| Week 1 | Pipeline scaffold, YOLOv8 baseline | ✅ Done |
| Week 2 | ByteTrack integration, anomaly scoring | ✅ Done |
| Week 3 | RF-DETR → YOLO12x migration | ✅ Done |
| Week 3 | ML pipeline refactor — constants, class-map auto-detection | ✅ Done |
| Week 3 | Finetuning Run 1 — completed, root cause diagnosed | ✅ Done |
| Week 3 | Frontend — dark editorial theme, ambient sound, Vercel deploy | ✅ Done |
| Week 4 | Finetuning Run 2 — class remapping fixed, Lightning AI | ⏳ Running |
| Week 4 | Upload page + live backend API | Planned |
| Week 4 | Jetson TRT export + latency benchmarking | Planned |

---

## Research Prospects

ARGUS touches several open problems in traffic CV. Potential paper directions:

**1. Unsupervised Incident Detection Without Labels**
The 9-filter InteractionScorer detects anomalies using physics (TTC, PET) and traffic law with zero labeled incident data. A paper could formalise this as a knowledge-driven anomaly detection framework and benchmark it against supervised approaches on public incident datasets (DoTA, DADA-2000).

**2. Cross-Domain Vehicle Detection: Western to Indian Traffic**
BDD100K (US/diverse) + IDD (Indian sub-continent) exposes sharp domain shift — different vehicle morphology (autorickshaws, two-wheelers), density, and road structure. The class remapping + resampling pipeline is a reproducible contribution here.

**3. Class Imbalance in Multi-Source Traffic Datasets**
Motorcycle and bicycle underrepresentation in BDD100K combined with the label remapping bug makes a strong motivating case for a paper on multi-source dataset curation for safety-critical detection. The Run 1 → Run 2 diagnosis is a clean ablation.

**4. Benchmarking YOLO12x vs RF-DETR for Edge Traffic Surveillance**
Systematic comparison of transformer-based (RF-DETR) vs attention-augmented CNN (YOLO12x) detectors specifically on dashcam and CCTV conditions — not COCO benchmarks. Latency, mAP, and deployment story on Jetson.

**5. Real-Time Near-Miss Detection Using Monocular TTC Estimation**
TTC from monocular video is an ill-posed problem without depth. ARGUS approximates it from 2D bounding box convergence rate + calibrated `pixels_per_meter`. A paper could characterise the error model and when 2D TTC is sufficient for safety alerts.

**6. Temporal Confirmation as a Precision Filter for Detection Pipelines**
ARGUS's 2-frame `TemporalConfirmation` class is a simple streak-based filter that significantly cuts phantom detections at low additional latency. Could be generalised and benchmarked as a lightweight post-processing stage for any detector.

---

## Repo Structure

```
ARGUS/
├── ml/
│   ├── ml_pipeline/
│   │   ├── constants.py               # Shared class ID maps (COCO + finetuned)
│   │   ├── detection.py               # VehicleDetector — YOLO12x, auto class-map
│   │   ├── tracking.py                # VehicleTracker — ByteTrack via supervision
│   │   ├── trajectory.py              # TrajectoryBuilder
│   │   └── interaction.py             # InteractionScorer (9-filter anomaly logic)
│   ├── training/
│   │   ├── argus_yolo12x_run2.ipynb   # Kaggle notebook (Run 2, bug-fixed)
│   │   ├── argus_lightning_train.py   # Lightning AI script (current run)
│   │   └── argus_yolo12x_kaggle.ipynb # Kaggle notebook (Run 1, archived)
│   ├── eval_real.py                   # Eval against CVAT YOLO annotations
│   └── week3_annotate.py              # Frame extraction for human annotation
├── ui/                                # Frontend — deployed to Vercel
│   ├── index.html                     # Landing page (dark editorial, ambient sound)
│   ├── dashboard.html                 # Analysis results
│   ├── js/
│   │   ├── landing.js                 # Three.js globe + upload flow
│   │   ├── dashboard.js
│   │   └── api.js
│   └── assets/
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

Pretrained YOLO12x weights download automatically on first run. For finetuned weights, place `argus_yolo12x_best.pt` in the project root:

```python
from ml.ml_pipeline import analyze_video
result = analyze_video("video.mp4", model_path="argus_yolo12x_best.pt")
```

---

*BITS Pilani · Semester II 2025–26 · Veer Raghuvanshi*
