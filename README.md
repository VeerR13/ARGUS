# ARGUS — Autonomous Road Guard Unified Surveillance

> **Active development** — pipeline, UI, and dataset are all evolving. See [Development Timeline](#development-timeline) for current status and next steps.

Real-time traffic anomaly detection system. Detects vehicles, tracks trajectories, and flags accidents, near-misses, and dangerous interactions from dashcam and overhead CCTV footage.

Live demo → **[argus-platform.vercel.app](https://argus-platform.vercel.app)**

### Team

| Role | Responsibilities |
|------|-----------------|
| ML Generalist | Pipeline architecture, detection/tracking, anomaly scoring, evaluation |
| ML Engineer | Dataset construction, model fine-tuning, hard-negative mining, annotation pipeline |
| Frontend Engineer | Dashboard UI, Three.js landing, charts, persona system, AI chat integration |

---

## What It Does

Upload a dashcam or CCTV clip → ARGUS detects every vehicle frame-by-frame, builds per-track trajectories, and applies a physics-based interaction scorer to flag:

- **Accidents** — bounding-box overlap + PET ≈ 0 s + TTC < 0.5 s
- **Near-misses** — TTC < 1.0 s with confirmed closing speed ≥ 15 km/h
- **Dangerous braking** — sudden deceleration ≥ 6 m/s² over 10 frames
- **Speed violations** — per-vehicle speed relative to ambient traffic
- **Dense-traffic anomalies** — stricter thresholds kick in automatically when ≥ 6 vehicles fill the frame

The UI presents the analysis as a multi-persona report (insurance, city operations, emergency services) with an incident timeline, heatmap, confidence metrics, and an embedded AI chat.

---

## Repository Structure

```
ARGUS/
├── ml/                         # Computer vision + anomaly detection pipeline
│   ├── ml_pipeline/
│   │   ├── detection.py        # YOLOv8 detector + TemporalConfirmationBuffer
│   │   ├── tracking.py         # DeepSORT wrapper (Kalman-filtered bboxes)
│   │   ├── trajectory.py       # Per-track trajectory + speed estimation
│   │   └── interaction.py      # 9-filter anomaly scorer (TTC, PET, decel, F7-F9)
│   ├── week1_test.py           # Week 1: baseline detection + tracking eval
│   ├── week2_eval.py           # Week 2: evaluation metrics
│   ├── week2_eval_complete.py  # Week 2: full pipeline eval with all metrics
│   ├── week3_retrain.py        # Week 3: dataset build + YOLOv8n fine-tuning
│   ├── week3_annotate.py       # Week 3: CVAT annotation frame extractor (500 frames)
│   ├── boost_classes.py        # Minority class balancing (motorcycle/bus/truck)
│   ├── build_overhead_dataset.py  # Overhead/CCTV dataset from YouTube + COCO + VisDrone
│   ├── download_coco_vehicles.py  # COCO vehicle subset downloader
│   ├── extract_annotation_batch.py # Export frames to CVAT for human annotation
│   ├── hard_negative_mining.py # Background false-positive mining from TemporalBuffer
│   ├── train_augmentation.py   # Augmentation pipeline for retraining
│   ├── eval_real.py            # Real-footage evaluation against human annotations
│   ├── score_projection.py     # Score projection utilities
│   ├── debug_incident.py       # Incident visualisation debug tool
│   ├── dataset/
│   │   └── data.yaml           # YOLO dataset config
│   ├── annotation_tasks/       # CVAT metadata + frame index (frames excluded)
│   ├── annotations_remapped/   # Remapped YOLO-format annotation labels
│   └── reports_validated*/     # Eval reports against human-annotated ground truth
└── ui/                         # Analytics dashboard
    ├── index.html              # Landing page (Three.js Earth globe)
    ├── dashboard.html          # Main analysis dashboard
    ├── incident.html           # Per-incident detail view
    ├── report.html             # Printable report
    ├── css/main.css            # Global design system (dark, monospace aesthetic)
    ├── js/
    │   ├── landing.js          # Three.js scene + upload flow
    │   ├── dashboard.js        # Charts, stats, persona switcher, AI panel
    │   ├── incident.js         # Incident detail renderer
    │   ├── report.js           # Report builder
    │   ├── api.js              # Data fetching (mock + real stubs)
    │   └── utils.js            # Shared helpers
    └── assets/
        └── mock_analysis.json  # Demo analysis payload
```

---

## ML Pipeline

### Stack
Python · YOLOv8 (Ultralytics) · DeepSORT · OpenCV · NumPy

### Detection — `ml_pipeline/detection.py`
- **Model:** YOLOv8s (COCO pretrained) or fine-tuned `week2_retrained.pt`
- **Classes:** car, motorcycle, bus, truck
- **Confidence:** 0.50 · **NMS IoU:** 0.35
- **TemporalConfirmationBuffer:** a detection must appear in 3 consecutive frames (IoU ≥ 0.40) before reaching the tracker — eliminates shadow/reflection phantom fires

### Tracking — `ml_pipeline/tracking.py`
- DeepSORT with MobileNet embedder
- `n_init=2` (track confirmed after 2 frames) · `max_age=30` frames
- Returns **post-Kalman** `to_ltrb()` positions — reduces bbox lag on fast vehicles

### Trajectory — `ml_pipeline/trajectory.py`
- Per-track center history + pixel-displacement speed estimate (km/h)
- Speed capped at 200 km/h; zeroed on non-consecutive frames

### Anomaly Scoring — `ml_pipeline/interaction.py`
Physics-based pairwise interaction scorer with 9 filters:

| Filter | Description |
|--------|-------------|
| F1 | TTC < threshold (1.0 s near-miss / 0.5 s accident) |
| F2 | Same-depth check — suppress laterally-separated vehicles |
| F3 | Relative deceleration ≥ 6 m/s² over 10 frames |
| F4 | Minimum consecutive danger frames = 4 (fast accidents fire in 1-3 frames) |
| F5 | PET = 0 + bbox overlap → accident |
| F7 | Speed floor — at least one vehicle must exceed 15 km/h |
| F8 | Gap monotonicity — gap must be steadily closing (≥ 65% of transitions) |
| F9 | Dense-traffic mode — when ≥ 6 vehicles in frame, stricter TTC/speed gates; distance gate disabled |

---

## Model Performance

### Week 2 Pipeline Evaluation (synthetic metrics)
| Metric | Score | Weight |
|--------|-------|--------|
| Phantom detection rate | 100 % | 0.25 |
| IoU consistency | 93.7 % | 0.25 |
| Bbox lag | 91.1 % | 0.20 |
| Dedup accuracy | 100 % | 0.15 |
| Inter-frame consistency | 96.0 % | 0.15 |
| **Aggregate** | **96.1 / 100 — Grade A** | |

### Week 3 Real-World Evaluation (human-annotated ground truth)
| Metric | Value |
|--------|-------|
| Mean IoU | 70.3 % |
| Recall | 30.0 % |
| Precision | 34.4 % |
| Phantom FP rate | 34.4 % |
| Bbox lag | 75.5 % |
| **Aggregate** | **48.6 / 100 — Grade D** |

> **Gap analysis:** The pipeline was built and evaluated on dashcam footage; the human-annotated dataset came from overhead CCTV at a different perspective, scale, and angle. This caused most of the recall/precision drop. `build_overhead_dataset.py` and `boost_classes.py` directly address this by sourcing overhead training data from YouTube traffic cams, COCO, and VisDrone.

### YOLOv8n Fine-tune (Week 3)
| Metric | Value |
|--------|-------|
| Base model | yolov8n.pt |
| Training samples | 438 positives + 663 hard negatives |
| Best epoch | 27 / 30 |
| mAP@50 | 78.2 % |
| mAP@50-95 | 57.1 % |
| Precision | 81.1 % |
| Recall | 74.4 % |

---

## UI

### Stack
HTML · CSS · Vanilla JS (ES modules) · Three.js · Vercel

### Pages

**`index.html` — Landing**
- Three.js Earth globe with city-to-city animated arcs (21 global pairs)
- Radar sweep animation
- Video upload modal → progress ramp → redirect to dashboard

**`dashboard.html` — Analysis Dashboard**
- File info bar, detection confidence chips, summary stats
- Persona switcher: **Insurance** / **City Operations** / **Emergency Services** — each re-renders the AI analysis section with role-specific language
- Anomaly timeline log with severity badges and causal-factor accordion
- Confidence bar charts, speed distribution, congestion heatmap
- Embedded Claude AI chat for natural-language queries on the report
- Export to CSV / PDF

**`incident.html` — Incident Detail**
Per-incident view with vehicle trajectory playback, causal factor breakdown, and confidence scoring.

**`report.html` — Report View**
Printable full analysis report.

---

## Development Timeline

| Phase | Status | Work |
|-------|--------|------|
| Week 1 | Done | YOLOv8 detector · DeepSORT integration · trajectory builder · `week1_test.py` |
| Week 2 | Done | Hard negative mining · pseudo-GT labelling · confidence/track-length histograms · **96.1/100** pipeline eval |
| Week 3 | Done | YOLOv8n fine-tune (mAP50=78 %) · CVAT annotation pipeline (500 frames) · real-world eval vs human annotations (**48.6/100** — perspective gap identified) |
| Week 4 | Active | Overhead dataset builder from YouTube + COCO + VisDrone · minority class boosting (motorcycle/bus/truck) · retrain on overhead data to close the perspective gap |
| Upcoming | Planned | Backend API (`/api/upload`, `/api/jobs/{id}/status`) to replace UI mocks · homography-based speed calibration · live RTSP stream support |

---

## Setup

### ML Pipeline

```bash
cd ml
pip install ultralytics deep-sort-realtime opencv-python numpy

# Run on a video
python eval_real.py --source path/to/video.mp4

# Or step through the weekly scripts
python week1_test.py
python week2_eval_complete.py
python week3_retrain.py --epochs 30 --device cpu
```

### Overhead Dataset (Week 4 prep)

```bash
cd ml
# Download COCO vehicle subset
python download_coco_vehicles.py

# Download YouTube overhead traffic clips (requires yt-dlp)
pip install yt-dlp
bash download_yt_traffic.sh

# Build overhead training set + retrain
python build_overhead_dataset.py --epochs 50 --device cpu
```

### Class Balancing

```bash
cd ml
python boost_classes.py --epochs 30 --device cpu
```

### CVAT Annotation Export

```bash
cd ml
python extract_annotation_batch.py   # outputs annotation_batch_v3/ (500 frames)
```

### UI (local)

```bash
cd ui
# Open index.html in a browser — no build step required
# ES modules require a local server:
python -m http.server 8080
# then open http://localhost:8080
```

---

## Dataset

- Detection model pretrained on COCO, fine-tuned on dashcam footage
- 438 confirmed-vehicle positives + 663 hard negatives (background patches from TemporalBuffer rejects)
- Human-annotated ground truth: overhead CCTV frames labelled in CVAT
- Large image sets not included in repo (see `ml/dataset/data.yaml` for directory structure)

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Detection | YOLOv8s / YOLOv8n (Ultralytics) |
| Tracking | DeepSORT (deep-sort-realtime) |
| Anomaly scoring | Custom Python — TTC, PET, deceleration, gap monotonicity |
| Training | Ultralytics YOLO fine-tune · augmentation pipeline |
| Annotation | CVAT (Cloud) |
| Frontend | HTML5 · CSS3 · Vanilla JS (ES modules) · Three.js |
| Deployment (UI) | Vercel |
