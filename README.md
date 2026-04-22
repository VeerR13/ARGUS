# ARGUS — Autonomous Road Guard Unified Surveillance

> **Status:** Active research project. The live demo at [argus-platform.vercel.app](https://argus-platform.vercel.app) is a frontend prototype — no model is connected. All analysis shown is mock data. We are building toward a real backend.

---

## What Is ARGUS?

ARGUS is an end-to-end traffic anomaly detection system designed to process dashcam or CCTV footage and automatically identify dangerous events — accidents, near-misses, sudden braking, and unsafe lane changes — without any human review.

The core research question: **can a purely computer-vision-based system, with no GPS or road-map data, reliably detect traffic incidents in real time using only raw video?**

The answer involves solving three subproblems simultaneously:
1. Detecting and classifying every vehicle in every frame under real-world conditions (occlusion, lighting changes, shadows, motion blur)
2. Maintaining consistent vehicle identity across frames despite those same conditions
3. Inferring *intent and danger* from raw pixel trajectories — no labeled incident data, no LiDAR, no depth sensor

---

## System Architecture

```
Video Input
    │
    ▼
VehicleDetector          detection.py
  YOLOv8l (BDD100K)      ─ confidence 0.50, NMS IoU 0.35
  TemporalConfirmation   ─ 3-frame streak required before passing to tracker
    │
    ▼
VehicleTracker           tracking.py
  DeepSORT + MobileNet   ─ Kalman-filtered bbox positions
  n_init=2, max_age=30
    │
    ▼
TrajectoryBuilder        trajectory.py
  Per-track history      ─ center positions, pixel-displacement speed (km/h)
    │
    ▼
InteractionScorer        interaction.py
  9-filter anomaly logic ─ TTC, PET, deceleration, gap monotonicity, dense-traffic mode
    │
    ▼
Incident Report          { trajectories, incidents }
```

The pipeline is exposed as a single function `analyze_video()` in `ml_pipeline/__init__.py`, which returns trajectories and detected incidents ready for the UI or API.

---

## ML Pipeline — Technical Detail

### Detection (`ml_pipeline/detection.py`)

Uses YOLOv8l fine-tuned on BDD100K. A **TemporalConfirmationBuffer** sits between the detector and tracker: a detection must appear in 3 consecutive frames (IoU ≥ 0.40) before being forwarded. This kills phantom detections from shadows and reflections — single-frame fires that plagued the early baseline.

### Tracking (`ml_pipeline/tracking.py`)

DeepSORT with MobileNet embedder. Returns **post-Kalman** `to_ltrb()` positions rather than raw YOLO bboxes — the Kalman prediction leads the bbox on fast-moving vehicles instead of lagging behind.

### Anomaly Scoring (`ml_pipeline/interaction.py`)

Physics-based pairwise scorer. For every pair of confirmed tracks per frame, nine filters are evaluated:

| Filter | Logic |
|--------|-------|
| F1 | TTC < 1.0 s → near-miss; TTC < 0.5 s → accident |
| F2 | Same-depth check — suppresses laterally-separated vehicles |
| F3 | Relative deceleration ≥ 6 m/s² over 10 frames |
| F4 | Min 4 consecutive danger frames (accidents are fast — 8 was too strict) |
| F5 | PET = 0 + bbox overlap → collision confirmed |
| F7 | Speed floor — at least one vehicle must exceed 15 km/h |
| F8 | Gap monotonicity — gap must be steadily closing (≥ 65% of transitions) |
| F9 | Dense-traffic mode — when ≥ 6 vehicles in frame, stricter TTC/speed gates, distance gate disabled |

TTC and deceleration are calibration-independent: TTC uses frame-to-frame gap closing rate (pixels/second cancel out), and deceleration uses relative percentage speed drop. This means the scorer works across different camera setups without manual calibration.

---

## Current Model — Where We Are

We are here. These are real numbers from `ml/training/results.csv`.

**Model:** YOLOv8l fine-tuned on BDD100K (20,000 dashcam images, 30 epochs, Kaggle P100)
**Notebook:** `ml/training/argus-1.ipynb`

### Overall Performance (BDD100K val, 2,000 images)

| Metric | Value |
|--------|-------|
| mAP50 | **62.3%** |
| mAP50-95 | 40.9% |
| Precision | 69.0% |
| Recall | 55.6% |

### Per-Class Breakdown

| Class | mAP50 | Training labels | Note |
|-------|-------|----------------|------|
| Car | **80.4%** | 205,514 | Strong — data-rich |
| Truck | 64.9% | 3,350 | Acceptable |
| Bus | 61.1% | 8,630 | Acceptable |
| Motorcycle | 42.8% | 840 | Weak — severely underrepresented |

**The problem is clear:** motorcycle is 0.4% of all training labels. The model barely learned it. Car dominates at 94% of the training data. We want to improve the overall number to 80%+ mAP50 before connecting the model to the live system.

### Week 2 Pipeline Eval (synthetic — pre-real-world testing)

| Metric | Score |
|--------|-------|
| Phantom detection rate | 100% |
| IoU consistency | 93.7% |
| Bbox lag | 91.1% |
| Dedup accuracy | 100% |
| **Aggregate** | **96.1 / 100** |

---

## What We Are Trying to Improve and Why

### 1. Class Imbalance (highest priority)

Motorcycle recall at 34% is unacceptable for a safety system. A motorcycle that the model misses is a motorcycle involved in an undetected near-miss. `boost_classes.py` addresses this by pseudo-labeling minority class frames from existing videos and rebalancing the training set before the next fine-tune.

### 2. More Training Epochs

The BDD100K run was cut at 30 epochs — validation mAP50 was still climbing (62.0% at epoch 30, up from 61.7% at epoch 23). The model had not plateaued. A 60-epoch run on the same data would likely push past 65%.

### 3. Additional Dashcam Data

BDD100K is good but US/highway-biased. Adding dashcam footage from different geographies and traffic patterns (dense urban, night, rain) would improve generalization. This is a future data collection effort.

### 4. Speed Calibration

Current speed estimates are pixel-displacement-based with a fixed `pixels_per_meter = 30.0` constant. This is wrong for any camera that isn't the exact one used during development. A homography-based calibration (mapping road markings to real-world distances) is planned for the next phase.

---

## Research Prospects

The anomaly scoring system has properties that make it interesting beyond the immediate application:

**Calibration independence.** The TTC and deceleration metrics are designed to be camera-agnostic. Any system that requires physical calibration per deployment is impractical at scale. The current approach works without it, and the tradeoffs are quantified.

**Temporal confirmation as a prior.** The TemporalConfirmationBuffer is essentially a learned prior over detection reliability — it suppresses low-confidence, short-lived detections without requiring a trained classifier. The buffer's false-negative rate (legitimate detections suppressed) vs false-positive suppression rate is a tunable parameter with a measurable tradeoff curve.

**Physics-based scoring without ground truth.** The anomaly scorer uses no labeled incident data. It derives danger from physics (TTC, PET, closing speed) applied to raw trajectories. This means it can be evaluated on new footage types without retraining — only the detection model needs domain adaptation.

**Dense-traffic mode.** F9 (automatic mode switch when ≥ 6 vehicles in frame) is an early form of scene-context awareness. In congested traffic, proximity is normal; only serious closing speed and tight TTC should fire. This is a precursor to more sophisticated context modeling.

---

## The Demo (UI)

**[argus-platform.vercel.app](https://argus-platform.vercel.app)** — a frontend prototype only.

No model is running behind it. All analysis data shown on the dashboard is hardcoded mock data (`ui/js/api.js` → `MOCK_DATA`). The upload flow simulates a 5-second processing ramp then loads a fixed demo result.

The real API endpoints are stubbed in `api.js` but commented out:
- `POST /api/upload` — video upload
- `GET /api/jobs/{id}/status` — polling for analysis progress
- `POST /api/ai/analyze` — Claude streaming analysis
- `POST /api/ai/ask` — AI chat on the report

The UI will be wired to a real backend once the model hits acceptable accuracy. Building the frontend first let us validate the UX and data contract before committing to a backend architecture.

**UI features (all functional with mock data):**
- Three.js Earth globe landing with city-arc animations
- Full analysis dashboard: incident timeline, confidence charts, speed distribution, heatmap
- Persona switcher — Insurance / City Operations / Emergency Services (each reframes the AI analysis)
- Per-incident detail view with causal factor breakdown
- CSV and PDF export
- Embedded Claude AI chat for natural-language queries on the report

---

## Development Timeline

| Phase | Status | What was done |
|-------|--------|--------------|
| Week 1 | Done | YOLOv8 detector, DeepSORT tracking, trajectory builder, basic eval |
| Week 2 | Done | Hard negative mining, pseudo-GT labelling, TemporalConfirmationBuffer, 96.1/100 pipeline eval |
| Week 3 | Done | YOLOv8n fine-tune attempt, CVAT annotation pipeline |
| BDD100K training | Done | YOLOv8l on 20K dashcam images — 62.3% mAP50. Results in `ml/training/` |
| Now | **Active** | Fix class imbalance, push toward 80%+ mAP50 before connecting model to UI |
| Next | Planned | Backend API, speed calibration, live stream support |

---

## Setup

### Run the pipeline

```bash
cd ml
pip install ultralytics deep-sort-realtime opencv-python numpy
python -c "from ml_pipeline import analyze_video; print(analyze_video('your_video.mp4'))"
```

### Training (Kaggle)

Open `ml/training/argus-1.ipynb` on Kaggle with:
- GPU: P100
- Dataset: `a7madmostafa/bdd100k-yolo`

### UI (local)

```bash
cd ui
python -m http.server 8080
# http://localhost:8080
```

---

## Team

| Role | Responsibilities |
|------|-----------------|
| ML Generalist | Pipeline architecture, anomaly scoring, evaluation |
| ML Engineer | Dataset construction, model training, class balancing |
| Frontend Engineer | Dashboard, Three.js landing, persona system, AI chat |
