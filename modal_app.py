"""
ARGUS Modal deployment — serverless GPU backend.

Free $30/month credit on Modal covers hundreds of demo videos.
GPU: T4 (15 GB VRAM) — YOLO12x inference ~5-15s per 30s clip.

Deploy:
    modal token new            # one-time browser login
    modal deploy modal_app.py  # → prints public URL

Upload model weights (one-time):
    modal volume create argus-data
    modal volume put argus-data /path/to/argus_s25.pt argus_s25.pt

Set Claude key (optional, for AI narrative):
    modal secret create argus-secrets ANTHROPIC_API_KEY=sk-ant-...
"""

from __future__ import annotations
import os
import modal

# ── Container image ───────────────────────────────────────────────────────────

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "libgomp1", "ffmpeg")
    .pip_install(
        "fastapi>=0.111.0",
        "uvicorn>=0.30.0",
        "python-multipart>=0.0.9",
        "anthropic>=0.20.0",
        "ultralytics>=8.4.0",
        "supervision>=0.27.0",
        "opencv-python-headless>=4.8.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
    )
    .add_local_dir("ml", remote_path="/argus/ml")
    .add_local_file("server.py", remote_path="/argus/server.py")
)

# ── Persistent volume: model weights ──────────────────────────────────────────

volume = modal.Volume.from_name("argus-data", create_if_missing=True)

# ── App ───────────────────────────────────────────────────────────────────────

app = modal.App("argus", image=image)

@app.function(
    gpu="T4",
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("argus-secrets", required=False)],
    timeout=600,
    allow_concurrent_inputs=5,
    scaledown_window=300,   # keep container warm for 5 min between requests
)
@modal.asgi_app()
def web():
    import sys
    sys.path.insert(0, "/argus")

    # Point server to the model file in the persistent volume
    model_candidates = ["/data/argus_s25.pt", "/data/argus_yolo12x_best.pt"]
    for p in model_candidates:
        if os.path.exists(p):
            os.environ["ARGUS_MODEL"] = p
            break

    import server
    return server.app
