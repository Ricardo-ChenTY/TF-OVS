#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from tf_ovos.data import load_manifest, read_vocab
from tf_ovos.metrics import boundary_iou, load_label_map


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "runs" / "artifacts" / "official_predictions"
OUT_DIR = ROOT / "runs" / "analysis"

DATASETS = {
    "voc20": {
        "manifest": ROOT / "data" / "manifests" / "voc20_val.jsonl",
        "vocab": ROOT / "configs" / "vocab" / "voc20.txt",
        "num_classes": 20,
        "void_label": 255,
        "prediction_label_offset": -1,
    },
    "context59": {
        "manifest": ROOT / "data" / "manifests" / "context59_val.jsonl",
        "vocab": ROOT / "configs" / "vocab" / "context_59.txt",
        "num_classes": 59,
        "void_label": 255,
        "prediction_label_offset": -1,
    },
    "ade20k": {
        "manifest": ROOT / "data" / "manifests" / "ade20k150_val.jsonl",
        "vocab": ROOT / "configs" / "vocab" / "ade20k_150.txt",
        "num_classes": 150,
        "void_label": 255,
        "prediction_label_offset": -1,
    },
    "coco_stuff164k": {
        "manifest": ROOT / "data" / "manifests" / "coco_stuff171_val.jsonl",
        "vocab": ROOT / "configs" / "vocab" / "coco_stuff_171.txt",
        "num_classes": 171,
        "void_label": 255,
        "prediction_label_offset": -1,
    },
    "context459": {
        "manifest": ROOT / "data" / "manifests" / "context459_val.jsonl",
        "vocab": ROOT / "configs" / "vocab" / "context_459.txt",
        "num_classes": 459,
        "void_label": 65535,
        "prediction_label_offset": 0,
    },
    "ade847": {
        "manifest": ROOT / "data" / "manifests" / "ade20k847_val.jsonl",
        "vocab": ROOT / "configs" / "vocab" / "ade20k_847.txt",
        "num_classes": 847,
        "void_label": 65535,
        "prediction_label_offset": 0,
    },
}


@dataclass
class DatasetSpec:
    name: str
    manifest: Path
    vocab: Path
    num_classes: int
    void_label: int
    prediction_label_offset: int


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _resize_like(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if pred.shape == gt.shape:
        return pred
    image = Image.fromarray(pred.astype(np.uint16))
    image = image.resize((gt.shape[1], gt.shape[0]), resample=Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.int32)


def _entropy_from_counts(counts: np.ndarray) -> float:
    total = int(counts.sum())
    if total == 0:
        return 0.0
    probs = counts[counts > 0].astype(np.float64) / total
    return float(-(probs * np.log2(probs)).sum())


def _semantic_biou(pred: np.ndarray, gt: np.ndarray, spec: DatasetSpec, radius: int) -> float | None:
    valid_gt = (gt >= 0) & (gt != spec.void_label) & (gt < spec.num_classes)
    if not valid_gt.any():
        return None

    pred_valid = valid_gt & (pred >= 0) & (pred < spec.num_classes) & (pred != spec.void_label)
    labels = np.union1d(gt[valid_gt], pred[pred_valid]).astype(np.int32)
    scores: list[float] = []
    for label in labels:
        gt_mask = valid_gt & (gt == label)
        pred_mask = valid_gt & (pred == label)
        if gt_mask.any() or pred_mask.any():
            scores.append(boundary_iou(pred_mask, gt_mask, radius=radius))
    return float(np.mean(scores)) if scores else None


def _prediction_path(pred_dir: Path, image_id: str) -> Path:
    path = pred_dir / f"{image_id}.png"
    if path.exists():
        return path
    return pred_dir / f"{Path(image_id).name}.png"


def _analyse_one(
    method: str,
    pred_dir: Path,
    spec: DatasetSpec,
    labels: list[str],
    include_biou: bool,
    biou_radius: int,
    max_images: int | None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    samples = load_manifest(spec.manifest)
    if max_images is not None:
        samples = samples[:max_images]

    conf = np.zeros((spec.num_classes, spec.num_classes), dtype=np.int64)
    pred_counts = np.zeros(spec.num_classes, dtype=np.int64)
    valid_pixels = 0
    valid_pred_pixels = 0
    invalid_pred_pixels = 0
    empty_images = 0
    entropy_sum = 0.0
    entropy_images = 0
    biou_sum = 0.0
    biou_images = 0
    processed = 0
    missing = 0
    shape_mismatch = 0
    raw_pred_min: int | None = None
    raw_pred_max: int | None = None
    pred_modes: set[str] = set()

    for sample in samples:
        pred_path = _prediction_path(pred_dir, sample.image_id)
        if not pred_path.exists():
            missing += 1
            continue

        gt = load_label_map(str(sample.mask_path), spec.void_label)
        pred_image = Image.open(pred_path)
        pred_modes.add(pred_image.mode)
        pred = np.asarray(pred_image, dtype=np.int32)
        if raw_pred_min is None:
            raw_pred_min = int(pred.min())
            raw_pred_max = int(pred.max())
        else:
            raw_pred_min = min(raw_pred_min, int(pred.min()))
            raw_pred_max = max(raw_pred_max or int(pred.max()), int(pred.max()))
        if spec.prediction_label_offset:
            pred = pred + spec.prediction_label_offset
        if pred.shape != gt.shape:
            shape_mismatch += 1
            pred = _resize_like(pred, gt)

        valid_gt = (gt >= 0) & (gt != spec.void_label) & (gt < spec.num_classes)
        if not valid_gt.any():
            processed += 1
            empty_images += 1
            continue

        valid_pred = valid_gt & (pred >= 0) & (pred < spec.num_classes) & (pred != spec.void_label)
        bad_pred = valid_gt & ~valid_pred
        valid_pixels += int(valid_gt.sum())
        valid_pred_pixels += int(valid_pred.sum())
        invalid_pred_pixels += int(bad_pred.sum())
        if not valid_pred.any():
            empty_images += 1
        else:
            image_counts = np.bincount(pred[valid_pred].astype(np.int64), minlength=spec.num_classes)
            pred_counts += image_counts
            entropy_sum += _entropy_from_counts(image_counts)
            entropy_images += 1

            pair_ids = spec.num_classes * gt[valid_pred].astype(np.int64) + pred[valid_pred].astype(np.int64)
            conf += np.bincount(pair_ids, minlength=spec.num_classes ** 2).reshape(
                spec.num_classes, spec.num_classes
            )

        if include_biou:
            score = _semantic_biou(pred, gt, spec, radius=biou_radius)
            if score is not None:
                biou_sum += score
                biou_images += 1

        processed += 1

    off_diag = conf.copy()
    np.fill_diagonal(off_diag, 0)
    top_pairs: list[dict[str, object]] = []
    if off_diag.sum() > 0:
        flat = off_diag.ravel()
        top_indices = np.argpartition(flat, -min(50, flat.size))[-min(50, flat.size):]
        top_indices = top_indices[np.argsort(flat[top_indices])[::-1]]
        for idx in top_indices:
            count = int(flat[idx])
            if count <= 0:
                continue
            gt_id = int(idx // spec.num_classes)
            pred_id = int(idx % spec.num_classes)
            gt_total = int(conf[gt_id].sum())
            top_pairs.append(
                {
                    "method": method,
                    "dataset": spec.name,
                    "gt_class_id": gt_id,
                    "gt_class": labels[gt_id] if gt_id < len(labels) else str(gt_id),
                    "pred_class_id": pred_id,
                    "pred_class": labels[pred_id] if pred_id < len(labels) else str(pred_id),
                    "pixels": count,
                    "fraction_of_valid_pixels": count / max(valid_pixels, 1),
                    "fraction_of_gt_class_pixels": count / max(gt_total, 1),
                }
            )

    aggregate_entropy = _entropy_from_counts(pred_counts)
    max_entropy = math.log2(spec.num_classes) if spec.num_classes > 1 else 1.0
    row: dict[str, object] = {
        "method": method,
        "dataset": spec.name,
        "artifact_dir": str(pred_dir.relative_to(ROOT)),
        "processed_images": processed,
        "expected_images": len(samples),
        "missing_images": missing,
        "artifact_completeness": processed / max(len(samples), 1),
        "shape_mismatch_images": shape_mismatch,
        "prediction_label_offset": spec.prediction_label_offset,
        "raw_prediction_min_label": raw_pred_min,
        "raw_prediction_max_label": raw_pred_max,
        "raw_prediction_modes": ";".join(sorted(pred_modes)),
        "possible_8bit_large_vocab_artifact": bool(
            spec.num_classes > 256
            and pred_modes
            and all(mode in {"1", "L", "P"} for mode in pred_modes)
            and (raw_pred_max is None or raw_pred_max <= 255)
        ),
        "valid_pixels": valid_pixels,
        "prediction_coverage": valid_pred_pixels / max(valid_pixels, 1),
        "invalid_prediction_rate": invalid_pred_pixels / max(valid_pixels, 1),
        "empty_prediction_image_rate": empty_images / max(processed, 1),
        "aggregate_prediction_entropy_bits": aggregate_entropy,
        "aggregate_prediction_entropy_normalized": aggregate_entropy / max_entropy,
        "mean_image_prediction_entropy_bits": entropy_sum / max(entropy_images, 1),
        "unique_predicted_classes": int((pred_counts > 0).sum()),
        "predicted_class_coverage": int((pred_counts > 0).sum()) / spec.num_classes,
        "include_biou": include_biou,
    }
    if include_biou:
        row["mean_semantic_biou"] = biou_sum / max(biou_images, 1)
        row["biou_images"] = biou_images
        row["biou_radius"] = biou_radius
    return row, top_pairs


def _dataset_spec(name: str) -> DatasetSpec:
    data = DATASETS[name]
    return DatasetSpec(
        name=name,
        manifest=Path(data["manifest"]),
        vocab=Path(data["vocab"]),
        num_classes=int(data["num_classes"]),
        void_label=int(data["void_label"]),
        prediction_label_offset=int(data["prediction_label_offset"]),
    )


def _analyse_task(
    task: tuple[str, Path, str, bool, int, int | None],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    method, pred_dir, dataset, include_biou, biou_radius, max_images = task
    spec = _dataset_spec(dataset)
    labels = read_vocab(spec.vocab)
    return _analyse_one(
        method=method,
        pred_dir=pred_dir,
        spec=spec,
        labels=labels,
        include_biou=include_biou,
        biou_radius=biou_radius,
        max_images=max_images,
    )


def _print_progress(row: dict[str, object]) -> None:
    print(
        f"{row['method']}/{row['dataset']}: processed={row['processed_images']} "
        f"coverage={float(row['prediction_coverage']):.4f} "
        f"entropy_norm={float(row['aggregate_prediction_entropy_normalized']):.4f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-process saved official prediction label maps into novelty/failure-mode metrics."
    )
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--methods", nargs="*", help="Subset of method directory names to process.")
    parser.add_argument("--datasets", nargs="*", choices=sorted(DATASETS), help="Subset of datasets to process.")
    parser.add_argument("--include-biou", action="store_true", help="Also compute semantic boundary IoU.")
    parser.add_argument("--biou-radius", type=int, default=2)
    parser.add_argument("--max-images", type=int, help="Debug/smoke-test on the first N manifest rows.")
    parser.add_argument("--jobs", type=int, default=1, help="Number of method/dataset workers to run in parallel.")
    args = parser.parse_args()

    methods = args.methods or sorted(path.name for path in args.artifact_root.iterdir() if path.is_dir())
    datasets = args.datasets or sorted(DATASETS)

    tasks: list[tuple[str, Path, str, bool, int, int | None]] = []
    for method in methods:
        for dataset in datasets:
            pred_dir = args.artifact_root / method / dataset
            if pred_dir.is_dir():
                tasks.append((method, pred_dir, dataset, args.include_biou, args.biou_radius, args.max_images))

    summary_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    if args.jobs <= 1 or len(tasks) <= 1:
        for task in tasks:
            row, pairs = _analyse_task(task)
            summary_rows.append(row)
            confusion_rows.extend(pairs)
            _print_progress(row)
    else:
        results: list[tuple[dict[str, object], list[dict[str, object]]] | None] = [None] * len(tasks)
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            future_to_index = {executor.submit(_analyse_task, task): idx for idx, task in enumerate(tasks)}
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                row, pairs = future.result()
                results[idx] = (row, pairs)
                _print_progress(row)
        for result in results:
            if result is None:
                continue
            row, pairs = result
            summary_rows.append(row)
            confusion_rows.extend(pairs)

    suffix = "_sample" if args.max_images is not None else ""
    if args.include_biou:
        suffix += "_biou"
    _write_csv(args.out_dir / f"official_prediction_map_metrics{suffix}.csv", summary_rows)
    _write_csv(args.out_dir / f"official_confusion_pairs{suffix}.csv", confusion_rows)


if __name__ == "__main__":
    main()
