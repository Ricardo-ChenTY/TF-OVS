#!/usr/bin/env bash
# Download PASCAL VOC 2012 for the voc20_val split.
# Usage: bash scripts/dl_voc20.sh [DATA_ROOT]
#   DATA_ROOT defaults to ./data/raw
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="${1:-$ROOT/data/raw}"
mkdir -p "$RAW"

TARGET="$RAW/VOCdevkit/VOC2012"
if [ -d "$TARGET" ]; then
    echo "[voc20] Already at $TARGET — skipping."
    exit 0
fi

echo "[voc20] Downloading PASCAL VOC 2012 (~2 GB)..."
wget -c --show-progress \
    "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar" \
    -O "$RAW/voc2012.tar"

echo "[voc20] Extracting..."
tar -xf "$RAW/voc2012.tar" -C "$RAW"
rm "$RAW/voc2012.tar"

echo "[voc20] Done → $TARGET"
echo "[voc20] Next: run  python scripts/prepare_manifests.py --dataset voc20"
