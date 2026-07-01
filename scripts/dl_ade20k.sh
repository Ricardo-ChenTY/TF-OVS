#!/usr/bin/env bash
# Download ADE20K datasets.
#   --150   ADE20K-150  (MIT ADEChallengeData2016, ~923 MB)  [default: yes]
#   --847   ADE20K-847  (full ADE20K, ~8 GB, requires MIT form acceptance)
# Usage: bash scripts/dl_ade20k.sh [DATA_ROOT] [--150] [--847]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="${1:-$ROOT/data/raw}"
DO_150=true
DO_847=false

for arg in "$@"; do
    case "$arg" in
        --150) DO_150=true ;;
        --847) DO_847=true ;;
        --no-150) DO_150=false ;;
    esac
done
mkdir -p "$RAW"

# ── ADE20K-150 (ADEChallengeData2016) ───────────────────────────────────────
if $DO_150; then
    TARGET="$RAW/ADEChallengeData2016"
    if [ -d "$TARGET" ]; then
        echo "[ade150] Already at $TARGET — skipping."
    else
        echo "[ade150] Downloading ADE20K-150 (~923 MB)..."
        wget -c --show-progress \
            "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip" \
            -O "$RAW/ADEChallengeData2016.zip"
        echo "[ade150] Extracting..."
        unzip -q "$RAW/ADEChallengeData2016.zip" -d "$RAW"
        rm "$RAW/ADEChallengeData2016.zip"
        echo "[ade150] Done → $TARGET"
    fi
fi

# ── ADE20K-847 (full ADE20K) ─────────────────────────────────────────────────
if $DO_847; then
    TARGET847="$RAW/ADE20K_2021_17_01"
    if [ -d "$TARGET847" ]; then
        echo "[ade847] Already at $TARGET847 — skipping."
    else
        echo "[ade847] Attempting direct download of full ADE20K (~8 GB)..."
        echo "[ade847] NOTE: MIT requires form acceptance at:"
        echo "         http://groups.csail.mit.edu/vision/datasets/ADE20K/request_data.php"
        echo "[ade847] Trying direct URL (may fail if not accepted)..."
        wget -c --show-progress \
            "http://groups.csail.mit.edu/vision/datasets/ADE20K/ADE20K_2021_17_01.zip" \
            -O "$RAW/ADE20K_2021_17_01.zip" || {
            echo ""
            echo "[ade847] Direct download failed. Manual steps:"
            echo "  1. Go to http://groups.csail.mit.edu/vision/datasets/ADE20K/request_data.php"
            echo "  2. Accept the terms and download ADE20K_2021_17_01.zip"
            echo "  3. Place it at $RAW/ADE20K_2021_17_01.zip"
            echo "  4. Re-run this script with --847"
            exit 1
        }
        echo "[ade847] Extracting (~8 GB, this takes a while)..."
        unzip -q "$RAW/ADE20K_2021_17_01.zip" -d "$RAW"
        rm "$RAW/ADE20K_2021_17_01.zip"
        echo "[ade847] Done → $TARGET847"
    fi
fi

echo ""
echo "[ade20k] Done.  Now run:"
$DO_150 && echo "  python scripts/prepare_manifests.py --dataset ade20k150"
$DO_847 && echo "  python scripts/prepare_manifests.py --dataset ade20k847"
true
