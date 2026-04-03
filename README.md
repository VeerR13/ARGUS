# ARGUS — Real-Time Traffic Anomaly Detection

End-to-end system that detects vehicles, tracks trajectories, and flags anomalies (accidents, sudden stops, wrong-way driving) from dashcam/CCTV footage in real time.

## Repository Structure

```
ARGUS/
├── ml/     # Computer vision pipeline (YOLOv8 + tracking + anomaly scoring)
└── ui/     # Analytics dashboard (vanilla JS, deployed on Vercel)
```

## ARGUS-ML

**Stack:** Python · YOLOv8 · ByteTrack · OpenCV

- Multi-class vehicle detection (car, motorcycle, bus, truck)
- ByteTrack multi-object tracking with trajectory smoothing
- Interaction scoring: proximity, velocity delta, lane deviation
- Hard-negative mining and weak-supervision pseudo-labelling
- Weekly evaluation pipeline with confidence/track-length reports

## ARGUS-UI

**Stack:** HTML · CSS · Vanilla JS · Vercel

Live demo → [argus-platform.vercel.app](https://argus-platform.vercel.app)

- Upload video → fake progress → mock anomaly report
- Incident timeline, heatmap overlay, CSV export
- Works fully client-side (no backend required for demo)

## Setup

### ML Pipeline
```bash
cd ml
pip install ultralytics opencv-python numpy
python eval_real.py --source <video_path>
```

### UI (local)
```bash
cd ui
# Open index.html in a browser — no build step needed
```

## Dataset

Detection model trained on COCO-pretrained YOLOv8n fine-tuned on dashcam footage.
Dataset not included in repo (too large). See `ml/dataset/data.yaml` for structure.
