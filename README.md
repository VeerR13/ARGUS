# ARGUS — Autonomous Road Guard Unified Surveillance

> Active development — pipeline, model, and UI are all evolving. Current training results are a baseline we are actively working to improve.

Real-time traffic anomaly detection system. Detects vehicles, tracks trajectories, and flags accidents, near-misses, and dangerous interactions from dashcam footage.

Live demo → **[argus-platform.vercel.app](https://argus-platform.vercel.app)**

### Team

| Role | Responsibilities |
|------|-----------------|
| ML Generalist | Pipeline architecture, detection/tracking, anomaly scoring, evaluation |
| ML Engineer | Dataset construction, model training, hard-negative mining |
| Frontend Engineer | Dashboard UI, Three.js landing, charts, persona system, AI chat |

---

## Repository Structure

```
ARGUS/
├── ml/
│   ├── ml_pipeline/            # Core detection + tracking + anomaly scoring
│   │   ├── detection.py        # YOLOv8 detector + TemporalConfirmationBuffer
│   │   ├── tracking.py         # DeepSORT wrapper (Kalman-filtered bboxes)
│   │   ├── trajectory.py       # Per-track trajectory + speed estimation
│   │   └── interaction.py      # 9-filter anomaly scorer (TTC, PET, decel)
│   ├── training/
│   │   ├── argus-1.ipynb       # Kaggle notebook — BDD100K fine-tune run
│   │   └── results.csv         # Per-epoch training metrics from that run
│   ├── week1_test.py           # Week 1: baseline detection + tracking eval
│   ├── week2_eval_complete.py  # Week 2: full pipeline eval with all metrics
│   ├── week3_retrain.py        # Week 3: dataset build + YOLOv8n fine-tuning
│   ├── boost_classes.py        # Minority class balancing (motorcycle/bus/truck)
│   ├── hard_negative_mining.py # Background FP mining from TemporalBuffer
│   ├── train_augmentation.py   # Augmentation pipeline for retraining
│   ├── eval_real.py            # Evaluation against human-annotated ground truth
│   └── dataset/data.yaml       # YOLO dataset config
└── ui/
    ├── index.html              # Landing page (Three.js Earth globe)
    ├── dashboard.html          # Analysis dashboard
    ├── incident.html           # Per-incident detail view
    ├── report.html             # Printable report
    ├── css/main.css
    └── js/
        ├── landing.js          # Three.js scene + upload flow
        ├── dashboard.js        # Charts, stats, persona switcher, AI panel
        ├── api.js              # Data fetching (mock + real API stubs)
        └── utils.js
```

---

## Current Training Results

> These are our current numbers. We are actively working to improve them — see [What We Are Working On](#what-we-are-working-on) below.

**Model:** YOLOv8l fine-tuned on BDD100K (20,000 dashcam images, 4 vehicle classes)
**Training run:** `ml/training/argus-1.ipynb` · Full per-epoch log: `ml/training/results.csv`

### Overall (BDD100K validation set, 2,000 images)

| Metric | Value |
|--------|-------|
| mAP50 | **62.3%** |
| mAP50-95 | 40.9% |
| Precision | 69.0% |
| Recall | 55.6% |

### Per-Class Breakdown

| Class | mAP50 | Precision | Recall | Training samples |
|-------|-------|-----------|--------|-----------------|
| Car | **80.4%** | 80.5% | 73.9% | 205,514 labels |
| Truck | 64.9% | 67.2% | 59.6% | 3,350 labels |
| Bus | 61.1% | 66.4% | 54.5% | 8,630 labels |
| Motorcycle | 42.8% | 61.8% | 34.3% | 840 labels |

The class imbalance is the main problem. Motorcycle is 0.4% of all training labels — the model barely learned it. Car dominates at 94% of labels, which is why it performs well.

### Week 2 Pipeline Eval (synthetic, pre-annotation)

| Metric | Score |
|--------|-------|
| Phantom detection rate | 100% |
| IoU consistency | 93.7% |
| Bbox lag | 91.1% |
| Dedup accuracy | 100% |
| **Aggregate** | **96.1 / 100** |

---

## What We Are Working On

| Priority | Task |
|----------|------|
| 1 | Fix class imbalance — motorcycle/bus/truck are severely underrepresented in BDD100K. `boost_classes.py` adds pseudo-labeled minority class frames to the training set. |
| 2 | Run more epochs — training was cut at 30, still improving. Target is 80%+ mAP50 overall. |
| 3 | Backend API — wire up `/api/upload` and `/api/jobs/{id}/status` to replace the UI mock. |
| 4 | Speed calibration — replace pixel-based speed estimate with homography. |

---

## ML Pipeline

### Stack
Python · YOLOv8 (Ultralytics) · DeepSORT · OpenCV · NumPy

### Detection — `ml_pipeline/detection.py`
- **Model:** YOLOv8l (BDD100K fine-tuned) or fallback to COCO `yolov8s.pt`
- **Classes:** car, motorcycle, bus, truck
- **TemporalConfirmationBuffer:** detection must appear in 3 consecutive frames (IoU ≥ 0.40) before reaching the tracker — eliminates shadow/reflection phantom fires

### Tracking — `ml_pipeline/tracking.py`
- DeepSORT with MobileNet embedder
- `n_init=2` · `max_age=30` frames
- Returns post-Kalman `to_ltrb()` positions — reduces bbox lag on fast vehicles

### Anomaly Scoring — `ml_pipeline/interaction.py`
Physics-based pairwise scorer with 9 filters:

| Filter | What it checks |
|--------|---------------|
| F1 | TTC < 1.0 s (near-miss) or < 0.5 s (accident) |
| F2 | Same-depth gate — suppresses laterally-separated vehicles |
| F3 | Relative deceleration ≥ 6 m/s² over 10 frames |
| F4 | Minimum 4 consecutive danger frames |
| F5 | PET = 0 + bbox overlap → accident |
| F7 | Speed floor — at least one vehicle must exceed 15 km/h |
| F8 | Gap monotonicity — gap must be steadily closing (≥ 65% of transitions) |
| F9 | Dense-traffic mode — when ≥ 6 vehicles in frame, stricter thresholds |

---

## UI

### Stack
HTML · CSS · Vanilla JS (ES modules) · Three.js · Vercel

### Pages
- **`index.html`** — Three.js Earth globe with city-to-city animated arcs, upload modal, progress flow
- **`dashboard.html`** — Anomaly timeline, confidence charts, speed distribution, congestion heatmap, persona switcher (Insurance / City Operations / Emergency Services), Claude AI chat
- **`incident.html`** — Per-incident detail with vehicle trajectory and causal factor breakdown
- **`report.html`** — Printable full report

---

## Setup

### ML Pipeline

```bash
cd ml
pip install ultralytics deep-sort-realtime opencv-python numpy
python eval_real.py --source path/to/video.mp4
```

### Training (Kaggle)

See `ml/training/argus-1.ipynb`. Requires:
1. Kaggle account with GPU P100 enabled
2. BDD100K YOLO dataset added as input (`a7madmostafa/bdd100k-yolo`)

### UI (local)

```bash
cd ui
python -m http.server 8080
# open http://localhost:8080
```

---

## Development Timeline

| Phase | Status | Summary |
|-------|--------|---------|
| Week 1 | Done | Detection + DeepSORT tracking + trajectory builder |
| Week 2 | Done | Hard negative mining · pseudo-GT · 96.1/100 pipeline eval |
| Week 3 | Done | YOLOv8n fine-tune attempt · CVAT annotation pipeline |
| BDD100K training | Done | YOLOv8l on 20K dashcam images · 62.3% mAP50 · results in `ml/training/` |
| Week 4 | Active | Class balancing · more training epochs · target 80%+ mAP50 |
| Upcoming | Planned | Backend API · homography speed calibration · live stream support |

---

## Dataset

- BDD100K: 20,000 dashcam images (train), 2,000 (val) — 4 vehicle classes remapped from BDD's 10 classes
- Hard negatives: 663 background patches mined from TemporalConfirmationBuffer rejects
- Large image sets not included in repo — see `ml/dataset/data.yaml` for directory structure
