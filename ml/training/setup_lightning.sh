#!/bin/bash
# Run this once when your Lightning AI Studio is ready.
# Paste these commands in the Studio terminal.

set -e

# ── 1. Install deps ───────────────────────────────────────────────────────────
pip install -q ultralytics>=8.4.0 kaggle opencv-python pyyaml

# ── 2. Kaggle credentials ─────────────────────────────────────────────────────
# Option A: paste your kaggle.json (from kaggle.com → Settings → API → Create New Token)
mkdir -p ~/.kaggle
# cat > ~/.kaggle/kaggle.json << 'EOF'
# {"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}
# EOF
chmod 600 ~/.kaggle/kaggle.json

# ── 3. Download datasets ──────────────────────────────────────────────────────
mkdir -p /teamspace/studios/this_studio/data

# BDD100K YOLO format (~1.5 GB)
kaggle datasets download -d a7madmostafa/bdd100k-yolo \
    -p /teamspace/studios/this_studio/data/bdd100k_zip --unzip
mv /teamspace/studios/this_studio/data/bdd100k_zip/* \
   /teamspace/studios/this_studio/data/bdd100k/ 2>/dev/null || true

# IDD Detection (~2 GB)
kaggle datasets download -d abhishekprajapat/idd-20k \
    -p /teamspace/studios/this_studio/data/idd_zip --unzip
mv /teamspace/studios/this_studio/data/idd_zip/* \
   /teamspace/studios/this_studio/data/idd/ 2>/dev/null || true

echo "Data downloaded. Run: tmux new -s argus && python argus_lightning_train.py"
