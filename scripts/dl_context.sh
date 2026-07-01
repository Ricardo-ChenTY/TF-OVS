#!/usr/bin/env bash
# Download PASCAL VOC 2010 images + PASCAL Context annotations.
# Produces data for context59_val and context459_val.
# Usage: bash scripts/dl_context.sh [DATA_ROOT]
#
# Requires:  pip install detail pillow numpy
#   'detail' is the official PASCAL Context Python API for .mat → PNG conversion.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="${1:-$ROOT/data/raw}"
mkdir -p "$RAW"

# ── 1. VOC 2010 images ──────────────────────────────────────────────────────
VOC10="$RAW/VOCdevkit/VOC2010"
if [ -d "$VOC10" ]; then
    echo "[context] VOC 2010 already at $VOC10 — skipping images."
else
    echo "[context] Downloading PASCAL VOC 2010 (~1.3 GB)..."
    wget -c --show-progress \
        "http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar" \
        -O "$RAW/voc2010.tar"
    echo "[context] Extracting VOC 2010..."
    tar -xf "$RAW/voc2010.tar" -C "$RAW"
    rm "$RAW/voc2010.tar"
    echo "[context] VOC 2010 done → $VOC10"
fi

# ── 2. PASCAL Context annotations (.mat) ────────────────────────────────────
CTX_DIR="$RAW/pascal_context"
mkdir -p "$CTX_DIR"
if [ -f "$CTX_DIR/trainval.json" ]; then
    echo "[context] Annotations already at $CTX_DIR/trainval.json — skipping."
else
    echo "[context] Downloading PASCAL Context annotations (~280 MB)..."
    wget -c --show-progress \
        "https://cs.stanford.edu/~roozbeh/pascal-context/trainval.tar.gz" \
        -O "$CTX_DIR/trainval.tar.gz"
    echo "[context] Extracting annotations..."
    tar -xzf "$CTX_DIR/trainval.tar.gz" -C "$CTX_DIR"
    rm "$CTX_DIR/trainval.tar.gz"
    echo "[context] Annotations done → $CTX_DIR"
fi

echo ""
echo "[context] Raw data ready.  Now run:"
echo "  python scripts/prepare_manifests.py --dataset context59"
echo "  python scripts/prepare_manifests.py --dataset context459"
