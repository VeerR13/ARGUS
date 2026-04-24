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
  RF-DETR Base           ─ COCO AP 53.3 (vs YOLOv8s 48.0)
  confidence 0.40        ─ lower threshold safe: RF-DETR scores better-calibrated
  TemporalConfirmation   ─ 2-frame streak required (down from 3: transformer
                           suppresses single-frame phantoms natively)
    │
    ▼
VehicleTracker           tracking.py
  ByteTrack              ─ IoU-only association, no appearance model
  track_thresh=0.40, match_thresh=0.70, max_age=30
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

## ML Stack — Current

### Why RF-DETR?

We migrated from YOLOv8 to **RF-DETR** (Roboflow Detection Transformer) as the primary detector:

| Metric | YOLOv8s | RF-DETR Base | RF-DETR Large |
|--------|---------|-------------|--------------|
| COCO AP | 48.0 | **53.3** | 56.3 |
| Architecture | CNN grid | Transformer | Transformer |
| Phantom rate | Higher (shadows, reflections) | Lower (attention sees whole scene) | Lower |

Transformer attention captures whole-scene context, which directly addresses the phantom detection problem that plagued the YOLOv8 baseline — single-frame fires from shadows and road markings that fooled the anchor-based grid classifier.

### Why ByteTrack?

We replaced DeepSORT + MobileNet appearance embedder with **ByteTrack** (IoU-only):

- DeepSORT's appearance model added ~12 ms/frame overhead with no measurable gain on traffic footage where vehicles look similar across cameras
- ByteTrack's low-confidence detection recycling recovers partially-occluded vehicles that DeepSORT would lose
- Zero dependency on a GPU-heavy embedding network — ByteTrack runs on CPU alongside RF-DETR on GPU

---

## ML Pipeline — Technical Detail

### Detection (`ml_pipeline/detection.py`)

RF-DETR Base loaded via `RFDETRBase()`. A **TemporalConfirmationBuffer** sits between the detector and tracker: a detection must appear in 2 consecutive frames (IoU ≥ 0.40) before being forwarded. This kills phantom detections from shadows and reflections. The minimum was reduced from 3 → 2 because RF-DETR's transformer attention already suppresses most single-frame phantoms — the 3-frame latency was cutting legitimate fast-approach detections.

Accepts an optional `pretrain_weights` path to load a fine-tuned checkpoint instead of COCO weights.

### Tracking (`ml_pipeline/tracking.py`)

ByteTrack with IoU-only association. Returns post-Kalman `tlbr` positions — the Kalman prediction leads the bbox on fast-moving vehicles instead of lagging behind. No appearance model dependency.

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

## Model Status

### Currently Training

**RF-DETR Base fine-tuned on BDD100K + IDD** — running on Kaggle T4 GPU.

- Dataset: 20,000 BDD100K dashcam images + IDD-Detection (Indian mixed traffic, Pascal VOC XML)
- Classes: car · motorcycle · bus · truck
- Notebook: `ml/training/argus_rfdetr_kaggle.ipynb`
- Target: >65% mAP50, better motorcycle recall than the YOLOv8 baseline

### Previous Baseline (YOLOv8l on BDD100K)

These numbers are from the old stack (`ml/training/argus-1.ipynb`) and serve as the benchmark to beat.

| Metric | Value |
|--------|-------|
| mAP50 | 62.3% |
| mAP50-95 | 40.9% |
| Precision | 69.0% |
| Recall | 55.6% |

Per-class breakdown:

| Class | mAP50 | Training labels | Note |
|-------|-------|----------------|------|
| Car | 80.4% | 205,514 | Strong — data-rich |
| Truck | 64.9% | 3,350 | Acceptable |
| Bus | 61.1% | 8,630 | Acceptable |
| Motorcycle | 42.8% | 840 | Weak — severely underrepresented |

Motorcycle at 42.8% mAP50 (34% recall) is the main failure mode. A motorcycle the model misses is a motorcycle in an undetected near-miss. The RF-DETR fine-tune on IDD (which has more two-wheeler diversity) directly targets this.

### Week 2 Pipeline Eval

| Metric | Score |
|--------|-------|
| Phantom detection rate | 100% |
| IoU consistency | 93.7% |
| Bbox lag | 91.1% |
| Dedup accuracy | 100% |
| **Aggregate** | **96.1 / 100** |

---

## What We Are Improving and Why

### 1. Detector Architecture (done)
RF-DETR replaces YOLOv8. The transformer attention mechanism directly addresses phantom detections from shadows and road markings — the primary false-positive source in the old baseline.

### 2. Tracker (done)
ByteTrack replaces DeepSORT. Removes the appearance model overhead, improves occlusion recovery, same IoU-based association logic.

### 3. Class Imbalance (in progress)
Motorcycle recall at 34% is unacceptable for a safety system. The BDD100K + IDD fine-tune adds Indian two-wheeler diversity (scooters, mopeds) that BDD100K lacks. `boost_classes.py` handles pseudo-label rebalancing for the next local fine-tune pass (`week3_retrain.py`).

### 4. Speed Calibration (planned)
Current speed estimates use a fixed `pixels_per_meter = 30.0` constant. A homography-based calibration mapping road markings to real-world distances is planned for the next phase.

---

## Research Prospects

**Calibration independence.** The TTC and deceleration metrics are camera-agnostic by design. Any system that requires per-deployment physical calibration is impractical at scale. The current approach avoids this.

**Temporal confirmation as a prior.** The TemporalConfirmationBuffer is a learned prior over detection reliability — it suppresses low-confidence, short-lived detections without a trained classifier. The false-negative rate (legitimate detections suppressed) vs false-positive suppression rate is a tunable parameter with a measurable tradeoff curve.

**Physics-based scoring without ground truth.** The anomaly scorer uses no labeled incident data. It derives danger from physics (TTC, PET, closing speed) applied to raw trajectories — it can be evaluated on new footage types without retraining.

**Dense-traffic mode.** F9 (automatic mode switch when ≥ 6 vehicles in frame) is scene-context awareness. In congested traffic, proximity is normal; only serious closing speed and tight TTC should fire. This is a precursor to more sophisticated context modeling.

---

## The Demo (UI)

**[argus-platform.vercel.app](https://argus-platform.vercel.app)** — a frontend prototype only.

No model is running behind it. All analysis data shown on the dashboard is hardcoded mock data (`ui/js/api.js` → `MOCK_DATA`). The upload flow simulates a 5-second processing ramp then loads a fixed demo result.

**UI features (all functional with mock data):**
- Three.js Earth globe landing with city-arc animations
- Full analysis dashboard: incident timeline, confidence charts, speed distribution, heatmap
- Persona switcher — Insurance / City Operations / Emergency Services
- Per-incident detail view with causal factor breakdown
- CSV and PDF export
- Embedded Claude AI chat for natural-language queries on the report

---

## Development Timeline

| Phase | Status | What was done |
|-------|--------|--------------|
| Week 1 | Done | YOLOv8 detector, DeepSORT tracking, trajectory builder, basic eval |
| Week 2 | Done | Hard negative mining, pseudo-GT labelling, TemporalConfirmationBuffer, 96.1/100 pipeline eval |
| Week 3 | Done | Dataset construction, CVAT annotation pipeline, fine-tune scaffolding |
| BDD100K baseline | Done | YOLOv8l on 20K images → 62.3% mAP50 |
| Stack migration | Done | RF-DETR + ByteTrack replaces YOLOv8 + DeepSORT across all pipeline files |
| BDD100K + IDD fine-tune | **In progress** | RF-DETR training on Kaggle T4 (`argus_rfdetr_kaggle.ipynb`) |
| Next | Planned | Load fine-tuned weights → local week3 retrain on dashcam data → backend API |

---

## Setup

### Run the pipeline

```bash
cd ml
pip install rfdetr supervision opencv-python numpy
python -c "from ml_pipeline import analyze_video; print(analyze_video('your_video.mp4'))"
```

To use a fine-tuned checkpoint instead of COCO weights:
```python
from ml_pipeline import analyze_video
result = analyze_video('your_video.mp4', detector_weights='path/to/argus_rfdetr_finetuned.pth')
```

### Training on Kaggle

Open `ml/training/argus_rfdetr_kaggle.ipynb` on Kaggle with:
- Accelerator: GPU T4
- Internet: On
- Dataset: `a7madmostafa/bdd100k-yolo` (public)
- Dataset: `Blank_0013/idd-detection` (optional, adds Indian traffic)

### Local fine-tune (after downloading Kaggle checkpoint)

```bash
cd ml
python week3_retrain.py --epochs 30 --batch 4 --device mps
```

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
