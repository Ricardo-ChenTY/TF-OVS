#!/usr/bin/env bash
# Download COCO 2017 val images + COCO-Stuff stuffthingmaps for coco_stuff171_val.
# Usage: bash scripts/dl_coco_stuff.sh [DATA_ROOT]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="${1:-$ROOT/data/raw}"
COCO="$RAW/coco_stuff171"
mkdir -p "$COCO"

# ── 1. COCO 2017 val images (~778 MB) ───────────────────────────────────────
if [ -d "$COCO/val2017" ]; then
    echo "[coco] val2017 images already present — skipping."
else
    echo "[coco] Downloading COCO 2017 val images (~778 MB)..."
    wget -c --show-progress \
        "http://images.cocodataset.org/zips/val2017.zip" \
        -O "$COCO/val2017.zip"
    echo "[coco] Extracting images..."
    unzip -q "$COCO/val2017.zip" -d "$COCO"
    rm "$COCO/val2017.zip"
    echo "[coco] Images done → $COCO/val2017"
fi

# ── 2. COCO-Stuff stuffthingmaps annotations (~500 MB) ──────────────────────
if [ -d "$COCO/annotations" ]; then
    echo "[coco] Stuff annotations already present — skipping."
else
    echo "[coco] Downloading COCO-Stuff stuffthingmaps (~500 MB)..."
    wget -c --show-progress \
        "http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip" \
        -O "$COCO/stuffthingmaps.zip" || {
        # Fallback: try the GitHub releases
        echo "[coco] Calvin server failed, trying GitHub releases..."
        wget -c --show-progress \
            "https://github.com/nightrome/cocostuff/releases/download/v1.1/stuffthingmaps_trainval2017.zip" \
            -O "$COCO/stuffthingmaps.zip"
    }
    echo "[coco] Extracting annotations..."
    unzip -q "$COCO/stuffthingmaps.zip" -d "$COCO/annotations_raw"
    rm "$COCO/stuffthingmaps.zip"
    # stuffthingmaps extracts to train2017/ and val2017/ — keep only val
    mkdir -p "$COCO/annotations"
    mv "$COCO/annotations_raw/val2017" "$COCO/annotations/val2017" 2>/dev/null || \
    mv "$COCO/annotations_raw/"*/val2017 "$COCO/annotations/val2017"
    rm -rf "$COCO/annotations_raw"
    echo "[coco] Annotations done → $COCO/annotations/val2017"
fi

echo ""
echo "[coco] Done.  Now run:"
echo "  python scripts/prepare_manifests.py --dataset coco_stuff171"
