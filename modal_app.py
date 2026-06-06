"""
ARGUS Modal deployment — serverless GPU backend.

Free $30/month credit on Modal covers hundreds of demo videos.
GPU: T4 (15 GB VRAM) — YOLO12x inference ~5-15s per 30s clip.

Deploy:
    modal token new            # one-time browser login
    modal deploy modal_app.py  # → prints public URL

Model weights are auto-downloaded from HF Hub on first run and cached
in the Modal volume. No manual upload needed.

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
        "huggingface_hub>=0.20.0",
    )
    .add_local_dir("ml", remote_path="/argus/ml")
    .add_local_file("server.py", remote_path="/argus/server.py")
)

# ── Persistent volume: model weights cache ────────────────────────────────────

volume = modal.Volume.from_name("argus-data", create_if_missing=True)

# ── App ───────────────────────────────────────────────────────────────────────

app = modal.App("argus", image=image)

def _resolve_model() -> str:
    """Download model from HF Hub on first run, cache in volume."""
    cached = "/data/argus_s25.pt"
    if os.path.exists(cached):
        return cached
    print("Downloading argus_s25.pt from HF Hub…")
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id="VeerR13/argus-weights", filename="argus_s25.pt",
                           local_dir="/data")
    volume.commit()   # persist to volume so next cold start skips download
    return path


@app.function(
    gpu="T4",
    volumes={"/data": volume},
    secrets=[],  # add modal.Secret.from_name("argus-secrets") after: modal secret create argus-secrets ANTHROPIC_API_KEY=sk-ant-...
    timeout=600,
    scaledown_window=300,
)
@modal.concurrent(max_inputs=5)
@modal.asgi_app()
def web():
    import sys
    sys.path.insert(0, "/argus")
    os.environ["ARGUS_MODEL"] = _resolve_model()
    import server
    return server.app
