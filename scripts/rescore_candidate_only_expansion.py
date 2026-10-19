#!/usr/bin/env python
"""E-11: candidate-only vocabulary expansion rescoring.

Addresses the review finding that E2 (Context-59->459, ADE-150->847)
confounds candidate-vocabulary growth with a change in GT taxonomy/
annotation scheme, since the compact and large splits use different label
files, not just a longer candidate list scored against the same GT.

This script takes each method's already-computed LARGE-vocabulary dense
predictions (context459 / ade847), remaps them into the COMPACT class
space via a name-normalized crosswalk (any predicted class with no match
in the compact vocabulary is treated as "off-vocabulary", i.e. wrong for
whatever pixel it covers), and rescores against the ORIGINAL compact GT
(context59 / ade20k150). This is a descriptive endpoint comparison using saved artifacts. It does not
strictly isolate candidate competition: the base strings/groupings may differ
and the crosswalk may not preserve every compact class identity.

Usage:
    python scripts/rescore_candidate_only_expansion.py
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from PIL import Image

from tf_ovos.data import load_manifest, read_vocab
from tf_ovos.metrics import load_label_map

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "runs" / "artifacts" / "official_predictions"
OUT_DIR = ROOT / "runs" / "analysis"

PAIRS = [
    {
        "name": "context",
        "compact_dataset": "context59",
        "large_dataset": "context459",
        "compact_manifest": ROOT / "data" / "manifests" / "context59_val.jsonl",
        "large_manifest": ROOT / "data" / "manifests" / "context459_val.jsonl",
        "compact_vocab": ROOT / "configs" / "vocab" / "context_59.txt",
        "large_vocab": ROOT / "configs" / "vocab" / "context_459.txt",
        "compact_num_classes": 59,
        "compact_void_label": 255,
        "large_num_classes": 459,
        "large_void_label": 65535,
    },
    {
        "name": "ade",
        "compact_dataset": "ade20k",
        "large_dataset": "ade847",
        "compact_manifest": ROOT / "data" / "manifests" / "ade20k150_val.jsonl",
        "large_manifest": ROOT / "data" / "manifests" / "ade20k847_val.jsonl",
        "compact_vocab": ROOT / "configs" / "vocab" / "ade20k_150.txt",
        "large_vocab": ROOT / "configs" / "vocab" / "ade20k_847.txt",
        "compact_num_classes": 150,
        "compact_void_label": 255,
        "large_num_classes": 847,
        "large_void_label": 65535,
    },
]




def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


# Known singular/plural naming variants between compact and large vocab
# files that survive punctuation/case normalization but are still the same
# class (checked by hand against configs/vocab/*.txt; add here only after
# confirming it's a naming variant, not a real distinct class). Both sides
# of each pair canonicalize to the same key regardless of which vocab file
# uses which spelling.
_CANONICAL_ALIASES = {
    _norm("people"): "person",
    _norm("person"): "person",
}


def _canon(key: str) -> str:
    return _CANONICAL_ALIASES.get(key, key)


def _base_id(image_id: str) -> str:
    """Strip any leading path-like prefix (e.g. ADE-847's scene-category
    directories) down to the bare image id used as the artifact filename."""
    return Path(image_id).name


def _build_crosswalk(compact_names: list[str], large_names: list[str]) -> dict[int, int]:
    """large_index -> compact_index for every large class whose name (or any
    comma-separated synonym) normalizes to match a compact class name."""
    compact_lookup: dict[str, int] = {}
    for idx, name in enumerate(compact_names):
        for alias in name.split(","):
            compact_lookup.setdefault(_canon(_norm(alias)), idx)

    crosswalk: dict[int, int] = {}
    for large_idx, name in enumerate(large_names):
        for alias in name.split(","):
            key = _canon(_norm(alias))
            if key in compact_lookup:
                crosswalk[large_idx] = compact_lookup[key]
                break
    return crosswalk


def _resize_like(pred: np.ndarray, gt_shape: tuple[int, int]) -> np.ndarray:
    if pred.shape == gt_shape:
        return pred
    image = Image.fromarray(pred.astype(np.uint16))
    image = image.resize((gt_shape[1], gt_shape[0]), resample=Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.int32)


def _rescore_method_pair(method: str, pair: dict) -> dict | None:
    pred_dir = ARTIFACT_ROOT / method / pair["large_dataset"]
    if not pred_dir.is_dir():
        return None

    compact_names = read_vocab(pair["compact_vocab"])
    large_names = read_vocab(pair["large_vocab"])
    crosswalk = _build_crosswalk(compact_names, large_names)
    unmatched = pair["large_num_classes"] - len(crosswalk)

    # LUT: large index -> compact index, or -1 (never matches any real class)
    lut_size = max(pair["large_num_classes"] + 1, 65536)
    lut = np.full(lut_size, -1, dtype=np.int32)
    for large_idx, compact_idx in crosswalk.items():
        lut[large_idx] = compact_idx

    compact_samples = load_manifest(pair["compact_manifest"])
    n = pair["compact_num_classes"]
    confusion = np.zeros((n, n), dtype=np.int64)
    fn_only = np.zeros(n, dtype=np.int64)  # GT pixels with an out-of-vocab prediction
    processed = 0
    missing = 0

    for sample in compact_samples:
        base = _base_id(sample.image_id)
        pred_path = pred_dir / f"{base}.png"
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing prediction: {pred_path}")

        gt = load_label_map(str(sample.mask_path), pair["compact_void_label"])
        raw_pred = np.asarray(Image.open(pred_path), dtype=np.int32)
        raw_pred = _resize_like(raw_pred, gt.shape)

        valid_pred_mask = (raw_pred >= 0) & (raw_pred < pair["large_num_classes"])
        if not valid_pred_mask.all():
            raise ValueError(f"Unexpected prediction IDs in {pred_path}")
        compact_pred = np.where(valid_pred_mask, lut[np.clip(raw_pred, 0, lut_size - 1)], -1)

        valid_gt = (gt >= 0) & (gt != pair["compact_void_label"]) & (gt < n)
        gt_flat = gt[valid_gt].astype(np.int64)
        pred_flat = compact_pred[valid_gt].astype(np.int64)

        good = pred_flat >= 0
        np.add.at(
            confusion,
            (gt_flat[good], pred_flat[good]),
            1,
        )
        bad_gt, bad_counts = np.unique(gt_flat[~good], return_counts=True)
        fn_only[bad_gt] += bad_counts
        processed += 1

    tp = np.diag(confusion)
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp + fn_only
    gt_support = confusion.sum(axis=1) + fn_only
    valid_classes = gt_support > 0
    denom = tp + fp + fn
    per_class_iou = np.divide(tp, denom, out=np.zeros_like(denom, dtype=np.float64), where=denom > 0)
    miou = float(per_class_iou[valid_classes].mean()) if valid_classes.any() else None

    return {
        "method": method,
        "pair": pair["name"],
        "processed_images": processed,
        "missing_images": missing,
        "large_vocab_size": pair["large_num_classes"],
        "compact_vocab_size": n,
        "crosswalk_matched": len(crosswalk),
        "compact_classes_retained": len(set(crosswalk.values())),
        "interpretation": "historical_endpoint_not_causal_decomposition",
        "crosswalk_unmatched": unmatched,
        "candidate_only_rescored_mIoU": round(miou, 4) if miou is not None else None,
    }


def main() -> None:
    global ARTIFACT_ROOT, OUT_DIR
    parser = argparse.ArgumentParser(description="Historical endpoint rescoring; not a strict causal decomposition")
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--pairs", nargs="+", choices=["context", "ade"], default=["context", "ade"])
    parser.add_argument("--methods", nargs="+")
    args = parser.parse_args()
    ARTIFACT_ROOT, OUT_DIR = args.artifact_root, args.out_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    methods = args.methods or sorted(p.name for p in ARTIFACT_ROOT.iterdir() if p.is_dir())
    if "cliptrase" in methods:
        raise ValueError("CLIPtrase saved maps have center-crop coordinates; cannot score them on full-image compact GT")
    rows = []
    for pair in PAIRS:
        if pair["name"] not in args.pairs:
            continue
        for method in methods:
            row = _rescore_method_pair(method, pair)
            if row is not None:
                rows.append(row)
                print(f"{row['pair']}/{method}: candidate_only_rescored_mIoU={row['candidate_only_rescored_mIoU']}", flush=True)

    out_path = OUT_DIR / "e11_candidate_only_rescoring.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
