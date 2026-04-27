# ARGUS — Autonomous Road Guard Unified Surveillance

> **Live demo:** [argus-platform.vercel.app](https://argus-platform.vercel.app) — frontend prototype, mock data. Backend training in progress.

---

## What Is ARGUS?

ARGUS is an end-to-end traffic anomaly detection system that processes dashcam or CCTV footage and automatically identifies dangerous events — accidents, near-misses, sudden braking, unsafe lane changes — with no human review.

Core research question: **can a purely computer-vision system, with no GPS or road-map data, reliably detect traffic incidents in real time from raw video?**

That requires solving three problems simultaneously:
1. Detecting and classifying every vehicle in every frame under real-world conditions (occlusion, lighting changes, shadows, motion blur)
2. Maintaining consistent vehicle identity across frames
3. Inferring intent and danger from raw pixel trajectories — no labeled incident data, no LiDAR, no depth sensor

---

## System Architecture

```
Video Input
    │
    ▼
VehicleDetector          ml_pipeline/detection.py
  YOLO12x                ─ 59.1 mAP50-95 on COCO (flagship, 2025)
  confidence 0.35        ─ auto device: MPS → CUDA → CPU
  TemporalConfirmation   ─ 2-frame streak to suppress phantom detections
    │
    ▼
VehicleTracker           ml_pipeline/tracking.py
  ByteTrack (sv)         ─ IoU-only association, no appearance model
  track_thresh=0.25, match_thresh=0.80, max_age=30
    │
    ▼
TrajectoryBuilder        ml_pipeline/trajectory.py
  Per-track history      ─ center positions, pixel-displacement speed
    │
    ▼
InteractionScorer        ml_pipeline/interaction.py
  9-filter anomaly logic ─ TTC, PET, deceleration, gap monotonicity
    │
    ▼
Incident Report          { trajectories, incidents }
```

---

## ML Stack

### Detector — YOLO12x

Migrated from RF-DETR to **YOLO12x** (ultralytics 8.4.x):

| | RF-DETR Base | YOLO11x | **YOLO12x** |
|---|---|---|---|
| COCO mAP50-95 | 53.3 | 59.0 | **59.1** |
| MPS inference | ~24 fps | ~32 fps | **~38 fps** |
| Jetson (TRT) | ~12 fps | ~55 fps | **~70 fps** |
| Training speed | Slow | Fast | Fast |
| Edge export | Manual | ONNX/TRT | **ONNX/TRT** |

### Tracker — ByteTrack

ByteTrack via `supervision` (IoU-only, no appearance model):
- No GPU-heavy embedding network — runs on CPU alongside YOLO on GPU
- Low-confidence detection recycling recovers partially-occluded vehicles
- Replaced DeepSORT which had 65%+ false-positive rate on dashcam footage due to pedestrian-trained embeddings

---

## Training

**Datasets:** BDD100K (70k images, diverse conditions) + IDD (10k images, Indian traffic)
**Classes:** car · motorcycle · bus · truck · bicycle
**Notebook:** `ml/training/argus_yolo12x_kaggle.ipynb` — runs on Kaggle T4 x2

### Expected results after finetuning

| Metric | Pretrained only | After finetuning |
|--------|----------------|-----------------|
| mAP50 | ~65–68% | **~78–82%** |
| mAP50-95 | ~42–45% | **~52–56%** |
| Car AP50 | ~80% | **~88%** |
| Truck AP50 | ~62% | **~74%** |
| Bus AP50 | ~68% | **~79%** |
| Motorcycle AP50 | ~55% | **~67%** |
| Bicycle AP50 | ~50% | **~63%** |
| Inference (MPS) | ~38 fps | ~38 fps |
| Inference (Jetson Orin TRT) | ~70 fps | ~70 fps |

Finetuning gains come from BDD100K night/rain/fog conditions and IDD Indian traffic density and vehicle distributions underrepresented in COCO.

---

## Project Status

| Week | Task | Status |
|------|------|--------|
| Week 1 | Pipeline scaffold, YOLOv8 baseline | ✅ Done |
| Week 2 | ByteTrack integration, anomaly scoring | ✅ Done |
| Week 3 | Model migration RF-DETR → YOLO12x, Kaggle training setup | ✅ Done |
| Week 3 | BDD100K + IDD finetuning run | ⏳ In progress |
| Week 4 | Connect finetuned model to live backend | Planned |

---

## Repo Structure

```
ARGUS/
├── ml/
│   ├── ml_pipeline/
│   │   ├── detection.py      # VehicleDetector (YOLO12x)
│   │   ├── tracking.py       # VehicleTracker (ByteTrack)
│   │   ├── trajectory.py     # TrajectoryBuilder
│   │   └── interaction.py    # InteractionScorer
│   ├── training/
│   │   ├── argus_yolo12x_kaggle.ipynb   # Kaggle training notebook
│   │   └── kaggle_yolo12x_finetune.py   # Training script
│   ├── eval_real.py          # Evaluation on real footage
│   └── week3_annotate.py     # Frame extraction for annotation
├── ui/                       # Frontend (Vercel)
│   ├── index.html
│   ├── js/landing.js
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

---

## Kaggle Training

1. Open `ml/training/argus_yolo12x_kaggle.ipynb` on Kaggle
2. Attach: `a7madmostafa/bdd100k-yolo` and `abhishekprajapat/idd-20k`
3. Set accelerator to T4 x2
4. Run all cells — trains 50 epochs (~3–4 hours), saves `argus_yolo12x_best.pt`

---

*BITS Pilani · Semester II 2025–26 · Veer Raghuvanshi*
