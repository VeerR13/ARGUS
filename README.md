# ARGUS — Autonomous Road Guard Unified Surveillance

> **Live demo:** [argus-platform.vercel.app](https://argus-platform.vercel.app) — frontend prototype, mock data. Backend finetuning in progress.

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
  YOLO12x                ─ flagship ultralytics 8.4.x (2025)
  confidence 0.35        ─ auto device: CUDA → MPS → CPU
  TemporalConfirmation   ─ 2-frame streak suppresses phantom detections
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

Migrated from RF-DETR to **YOLO12x** (ultralytics 8.4.x) after RF-DETR showed slow convergence and poor edge deployment story.

| | RF-DETR Base | YOLO11x | **YOLO12x** |
|---|---|---|---|
| COCO mAP50-95 | 53.3 | 59.0 | **59.1** |
| MPS inference | ~24 fps | ~32 fps | **~38 fps** |
| Jetson (TRT) | ~12 fps | ~55 fps | **~70 fps** |
| Training speed | Slow | Fast | Fast |
| Edge export | Manual | ONNX/TRT | **ONNX/TRT** |

Why YOLO12x over YOLO11x: marginally better mAP50-95, same inference speed, same training pipeline. The delta is small but it's the current flagship and future ultralytics updates will target it.

### Class Map — Pretrained vs Finetuned

YOLO12x uses different class IDs depending on weights. `detection.py` auto-detects which scheme is active from `model.names` at load time — no manual flag needed.

```python
# ml_pipeline/constants.py
COCO_VEHICLE_CLASSES     = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
FINETUNED_VEHICLE_CLASSES = {0: "car", 1: "motorcycle", 2: "bus", 3: "truck", 4: "bicycle"}
```

### Tracker — ByteTrack

ByteTrack via `supervision` (IoU-only, no appearance model):
- No GPU-heavy embedding network — runs on CPU alongside YOLO on GPU
- Low-confidence detection recycling recovers partially-occluded vehicles
- Replaced DeepSORT which had 65%+ false-positive rate on dashcam footage due to pedestrian-trained embeddings

---

## Training

**Datasets:** BDD100K + IDD (Indian Driving Dataset)  
**Classes:** car · motorcycle · bus · truck · bicycle  
**Notebook:** `ml/training/argus_yolo12x_kaggle.ipynb` — Kaggle T4 x2

### Run 1 (in progress)

| | Value |
|---|---|
| Epochs | 50 |
| Batch | 16 (T4 x2) |
| imgsz | 640 |
| Time/epoch | ~9.4 min |
| Current mAP50 (ep 45) | 0.498 |
| Current mAP50-95 | 0.325 |

Run 1 converged early at ~50% mAP50. Root cause under investigation — the Kaggle `bdd100k-yolo` dataset source appears to contain far fewer images (~7.6k training) than the full BDD100K (70k). Run 2 will use the correct full-scale dataset.

### Expected results (Run 2, full dataset)

| Metric | Pretrained | After finetuning |
|---|---|---|
| mAP50 | ~65–68% | **~78–82%** |
| mAP50-95 | ~42–45% | **~52–56%** |
| Car AP50 | ~80% | **~88%** |
| Truck AP50 | ~62% | **~74%** |
| Bus AP50 | ~68% | **~79%** |
| Motorcycle AP50 | ~55% | **~67%** |
| Bicycle AP50 | ~50% | **~63%** |
| Inference MPS | 38 fps | 38 fps |
| Inference Jetson TRT | 70 fps | 70 fps |

Run 2 will warm-start from Run 1's `best.pt` — the ~50% mAP50 baseline is preserved as a starting point.

---

## Project Status

| Week | Task | Status |
|---|---|---|
| Week 1 | Pipeline scaffold, YOLOv8 baseline | ✅ Done |
| Week 2 | ByteTrack integration, anomaly scoring | ✅ Done |
| Week 3 | RF-DETR → YOLO12x migration | ✅ Done |
| Week 3 | ML pipeline refactor — constants, class-map auto-detection, clean imports | ✅ Done |
| Week 3 | Kaggle finetuning Run 1 (BDD100K + IDD) | ⏳ In progress |
| Week 3 | Frontend — dark editorial theme, ambient sound, deployed to Vercel | ✅ Done |
| Week 4 | Finetuning Run 2 (full dataset, warm-start from Run 1) | Planned |
| Week 4 | Upload page + live backend API | Planned |

---

## Repo Structure

```
ARGUS/
├── ml/
│   ├── ml_pipeline/
│   │   ├── constants.py      # Shared class ID maps (COCO + finetuned)
│   │   ├── detection.py      # VehicleDetector — YOLO12x, auto class-map
│   │   ├── tracking.py       # VehicleTracker — ByteTrack via supervision
│   │   ├── trajectory.py     # TrajectoryBuilder
│   │   └── interaction.py    # InteractionScorer (9-filter anomaly logic)
│   ├── training/
│   │   ├── argus_yolo12x_kaggle.ipynb   # Kaggle training notebook
│   │   └── kaggle_yolo12x_finetune.py   # Script version
│   ├── eval_real.py          # Eval against CVAT YOLO annotations
│   └── week3_annotate.py     # Frame extraction for human annotation
├── ui/                       # Frontend — deployed to Vercel
│   ├── index.html            # Landing page (dark editorial, ambient sound)
│   ├── dashboard.html        # Analysis results
│   ├── js/
│   │   ├── landing.js        # Three.js globe + upload flow
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

Pretrained YOLO12x weights download automatically on first run. For finetuned weights, place `argus_yolo12x_best.pt` in the project root and pass `model_path="argus_yolo12x_best.pt"`.

---

## Kaggle Training

1. Open `ml/training/argus_yolo12x_kaggle.ipynb` on Kaggle
2. Attach datasets: `a7madmostafa/bdd100k-yolo` and `abhishekprajapat/idd-20k`
3. Set accelerator → **GPU T4 x2** (required — do not run on CPU)
4. Run the explore cell first to verify dataset paths
5. Run all cells — 50 epochs at ~9.4 min/epoch, saves `argus_yolo12x_best.pt`

---

*BITS Pilani · Semester II 2025–26 · Veer Raghuvanshi*
