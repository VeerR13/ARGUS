---
title: ARGUS Traffic Safety API
emoji: 🚦
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.0.0"
app_file: app.py
pinned: false
license: mit
---

# ARGUS Backend

FastAPI backend for [ARGUS](https://argus-platform.vercel.app) — autonomous road guard unified surveillance.

Uses **ZeroGPU** (free shared A10G) for fast YOLO12x inference (~15-30s per 30s clip).

## API

```
POST /api/upload                     — upload video, start processing
GET  /api/jobs/{video_id}/status     — poll progress
GET  /api/videos/{video_id}/analysis — full results JSON
GET  /api/incidents/{incident_id}    — single incident detail
POST /api/ai/analyze                 — SSE: Claude narrative (optional)
POST /api/ai/ask                     — SSE: Claude Q&A (optional)
```
