---
title: ARGUS Traffic Safety API
emoji: 🚦
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
---

# ARGUS Backend

FastAPI backend for [ARGUS](https://argus-platform.vercel.app) — autonomous road guard unified surveillance.

Detects vehicles, builds trajectories, and scores traffic incidents (near-misses, sudden braking, unsafe lane changes) from raw dashcam/CCTV footage using YOLO12x fine-tuned on BDD100K + IDD.

## API

```
POST /api/upload          — upload video, start processing
GET  /api/jobs/{id}/status — poll progress
GET  /api/videos/{id}/analysis — full results JSON
GET  /api/incidents/{id}  — single incident detail
POST /api/ai/analyze      — SSE: Claude narrative (requires ANTHROPIC_API_KEY secret)
POST /api/ai/ask          — SSE: Claude Q&A
```

## Notes

- CPU deployment — processes every 5th frame at 320px for speed
- Videos capped at 3 minutes
- Claude AI endpoints are optional; all detection/tracking works without an API key
