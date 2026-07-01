#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
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


METHOD_GROUPS = {
    "sclip": "Dense-map TF methods",
    "scclip": "Dense-map TF methods",
    "cliptrase": "Dense-map TF methods",
    "naclip": "Dense-map TF methods",
    "resclip": "Dense-map TF methods",
    "proxyclip": "Proposal+naming TF methods",
    "corrclip": "Proposal+naming TF methods",
    "trident": "CLIP + VFM TF methods",
    "cass": "CLIP + VFM TF methods",
    "freeda": "Diffusion/reference TF methods",
    "san": "Trained references",
}


@dataclass(frozen=True)
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


def _write_md(path: Path, title: str, rows: list[dict[str, object]], keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    if not rows:
        lines.append("_No rows generated._")
    else:
        lines.append("| " + " | ".join(keys) + " |")
        lines.append("| " + " | ".join(["---"] * len(keys)) + " |")
        for row in rows:
            vals = [_format_md(row.get(key, "")) for key in keys]
            lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_md(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    text = str(value)
    return text.replace("|", "\\|")


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


def _prediction_path(pred_dir: Path, image_id: str) -> Path:
    path = pred_dir / f"{image_id}.png"
    if path.exists():
        return path
    return pred_dir / f"{Path(image_id).name}.png"


def _resize_like(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if pred.shape == gt.shape:
        return pred
    image = Image.fromarray(pred.astype(np.uint16))
    image = image.resize((gt.shape[1], gt.shape[0]), resample=Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.int32)


def _load_prediction(path: Path, spec: DatasetSpec) -> np.ndarray:
    image = Image.open(path)
    pred = np.asarray(image, dtype=np.int32)
    if spec.prediction_label_offset:
        pred = pred + spec.prediction_label_offset
    return pred


def _safe_mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _analyse_method_dataset(
    method: str,
    pred_dir: Path,
    spec: DatasetSpec,
    labels: list[str],
    include_biou: bool,
    max_images: int | None,
    max_biou_regions: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    samples = load_manifest(spec.manifest)
    if max_images is not None:
        samples = samples[:max_images]

    total_regions = 0
    naming_top1_hits = 0
    localized_05 = localized_075 = 0
    matched_05 = matched_075 = 0
    gt_text_ious: list[float] = []
    gt_text_bious: list[float] = []
    oracle_ious: list[float] = []
    top1_confusions: Counter[tuple[int, int]] = Counter()
    oracle_confusions: Counter[tuple[int, int]] = Counter()
    processed = missing = 0
    biou_regions = 0

    for sample in samples:
        pred_path = _prediction_path(pred_dir, sample.image_id)
        if not pred_path.exists():
            missing += 1
            continue

        gt = load_label_map(str(sample.mask_path), spec.void_label)
        pred = _resize_like(_load_prediction(pred_path, spec), gt)
        valid = (gt >= 0) & (gt != spec.void_label) & (gt < spec.num_classes)
        if not valid.any():
            processed += 1
            continue

        pred_valid = np.where((pred >= 0) & (pred < spec.num_classes), pred, -1)
        gt_flat = gt[valid].astype(np.int64)
        pred_flat = pred_valid[valid].astype(np.int64)
        good_pred = pred_flat >= 0
        gt_flat_good = gt_flat[good_pred]
        pred_flat_good = pred_flat[good_pred]

        gt_counts = np.bincount(gt_flat, minlength=spec.num_classes)
        pred_counts = np.bincount(pred_flat_good, minlength=spec.num_classes)
        pair_counts = np.bincount(
            spec.num_classes * gt_flat_good + pred_flat_good,
            minlength=spec.num_classes * spec.num_classes,
        ).reshape(spec.num_classes, spec.num_classes)

        for gt_id in np.flatnonzero(gt_counts):
            gt_total = int(gt_counts[gt_id])
            if gt_total <= 0:
                continue
            total_regions += 1
            row = pair_counts[gt_id]
            top1_pred = int(np.argmax(row))
            if top1_pred == gt_id:
                naming_top1_hits += 1
            else:
                top1_confusions[(int(gt_id), top1_pred)] += 1

            same_inter = int(row[gt_id])
            same_union = gt_total + int(pred_counts[gt_id]) - same_inter
            same_iou = same_inter / same_union if same_union > 0 else 0.0
            gt_text_ious.append(float(same_iou))

            inter = row.astype(np.float64)
            union = gt_total + pred_counts.astype(np.float64) - inter
            ious = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
            oracle_pred = int(np.argmax(ious))
            oracle_iou = float(ious[oracle_pred])
            oracle_ious.append(oracle_iou)
            if oracle_pred != gt_id:
                oracle_confusions[(int(gt_id), oracle_pred)] += 1

            if oracle_iou >= 0.5:
                localized_05 += 1
                if oracle_pred == gt_id:
                    matched_05 += 1
            if oracle_iou >= 0.75:
                localized_075 += 1
                if oracle_pred == gt_id:
                    matched_075 += 1

            if include_biou and (max_biou_regions <= 0 or biou_regions < max_biou_regions):
                gt_mask = valid & (gt == gt_id)
                pred_mask = valid & (pred_valid == gt_id)
                gt_text_bious.append(boundary_iou(pred_mask, gt_mask, radius=2))
                biou_regions += 1

        processed += 1

    def _mcmr(localized: int, matched: int) -> float | None:
        if localized == 0:
            return None
        return float((localized - matched) / localized)

    summary = {
        "method": method,
        "method_group": METHOD_GROUPS.get(method, "other"),
        "dataset": spec.name,
        "processed_images": processed,
        "expected_images": len(samples),
        "missing_images": missing,
        "gt_class_regions": total_regions,
        "gt_region_naming_top1": naming_top1_hits / max(total_regions, 1),
        "gt_region_naming_top5": None,
        "gt_text_localization_iou": _safe_mean(gt_text_ious),
        "gt_text_localization_biou": _safe_mean(gt_text_bious),
        "proposal_oracle_iou": _safe_mean(oracle_ious),
        "proposal_recall_at_05": localized_05 / max(total_regions, 1),
        "proposal_recall_at_075": localized_075 / max(total_regions, 1),
        "mcmr_at_05": _mcmr(localized_05, matched_05),
        "mcmr_at_075": _mcmr(localized_075, matched_075),
        "biou_regions": biou_regions,
        "diagnostic_source": "proxy_from_dense_label_map",
        "artifact_gap": (
            "Top-5 naming, true proposal recall, and MCC before/after need saved "
            "region scores/proposal masks/MCC rerank artifacts."
        ),
    }

    mismatch_rows: list[dict[str, object]] = []
    for (gt_id, pred_id), count in top1_confusions.most_common(20):
        mismatch_rows.append(
            {
                "method": method,
                "dataset": spec.name,
                "kind": "gt_region_top1_confusion",
                "gt_class_id": gt_id,
                "gt_class": labels[gt_id] if gt_id < len(labels) else str(gt_id),
                "pred_class_id": pred_id,
                "pred_class": labels[pred_id] if pred_id < len(labels) else str(pred_id),
                "count": count,
            }
        )
    for (gt_id, pred_id), count in oracle_confusions.most_common(20):
        mismatch_rows.append(
            {
                "method": method,
                "dataset": spec.name,
                "kind": "proposal_oracle_confusion",
                "gt_class_id": gt_id,
                "gt_class": labels[gt_id] if gt_id < len(labels) else str(gt_id),
                "pred_class_id": pred_id,
                "pred_class": labels[pred_id] if pred_id < len(labels) else str(pred_id),
                "count": count,
            }
        )
    return summary, mismatch_rows


def _table11_rows(summaries: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in summaries:
        grouped[str(row["method_group"])].append(row)

    desired = [
        "Dense-map TF methods",
        "Detector+SAM TF methods",
        "Proposal+naming TF methods",
        "CLIP + VFM TF methods",
        "Diffusion/reference TF methods",
        "MCC consistency diagnostics",
        "Trained references",
    ]
    rows: list[dict[str, object]] = []
    for group in desired:
        vals = grouped.get(group, [])
        if vals:
            localized = [float(v["proposal_recall_at_05"]) for v in vals if v["proposal_recall_at_05"] is not None]
            mcmr = [float(v["mcmr_at_05"]) for v in vals if v["mcmr_at_05"] is not None]
            top1 = [float(v["gt_region_naming_top1"]) for v in vals if v["gt_region_naming_top1"] is not None]
            rows.append(
                {
                    "method_group": group,
                    "localized_pairs": int(sum(int(v["gt_class_regions"]) for v in vals)),
                    "mismatch_pairs_proxy": int(
                        sum(
                            round(float(v["mcmr_at_05"] or 0.0) * int(v["gt_class_regions"]))
                            for v in vals
                        )
                    ),
                    "MCMR@0.5_proxy": _safe_mean(mcmr),
                    "GT_region_top1_proxy": _safe_mean(top1),
                    "proposal_recall_at_0.5_proxy": _safe_mean(localized),
                    "example_mismatch_pairs_or_communities": "see table11_e2_mismatch_pairs.csv",
                    "status": "computed_proxy_from_dense_label_maps",
                }
            )
        else:
            rows.append(
                {
                    "method_group": group,
                    "localized_pairs": "",
                    "mismatch_pairs_proxy": "",
                    "MCMR@0.5_proxy": "",
                    "GT_region_top1_proxy": "",
                    "proposal_recall_at_0.5_proxy": "",
                    "example_mismatch_pairs_or_communities": "",
                    "status": "artifact_gap",
                }
            )
    return rows


def _table12_rows(summaries: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "readout": "Per-class behavior",
            "what_to_record": "Per-class IoU, pixel accuracy, mean class accuracy.",
            "current_output": "official_log_metrics.csv; proposal_novelty_metrics.csv",
            "status": "computed_from_official_logs",
        },
        {
            "readout": "Vocabulary ambiguity",
            "what_to_record": "MCMR@0.75, top mismatch pairs, confusion communities.",
            "current_output": "table11_e2_mismatch_summary.csv; table11_e2_mismatch_pairs.csv; official_confusion_pairs.csv",
            "status": "computed_proxy_from_dense_label_maps",
        },
        {
            "readout": "Proposal quality",
            "what_to_record": "BestIoU, Recall@0.5, Recall@0.7, number of proposals.",
            "current_output": "table9_diagnostic_probes.csv proposal_oracle_* proxy columns",
            "status": "proxy_from_dense_label_maps; true proposal masks still needed",
        },
        {
            "readout": "GT oracle diagnostics",
            "what_to_record": "GT-region naming Top-1/Top-5 and GT-text localization IoU/BIoU.",
            "current_output": "table9_diagnostic_probes.csv",
            "status": "Top-1 and GT-text IoU computed; Top-5 needs region score artifacts",
        },
        {
            "readout": "MCC consistency",
            "what_to_record": "Before/after mIoU, before/after MCMR@0.5, qualitative examples.",
            "current_output": "",
            "status": "artifact_gap; needs MCC rerank outputs",
        },
        {
            "readout": "Efficiency details",
            "what_to_record": "Offline time, per-image inference time, memory, model calls, input resolution.",
            "current_output": "proposal_novelty_metrics.csv; e4_isolated queue outputs",
            "status": "partial; isolated queue running, peak-memory sampler/model-call instrumentation still needed",
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Table 9/11/12 diagnostic readouts from saved predictions.")
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--methods", nargs="*", help="Subset of methods.")
    parser.add_argument("--datasets", nargs="*", choices=sorted(DATASETS), help="Subset of datasets.")
    parser.add_argument("--include-biou", action="store_true")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--max-biou-regions", type=int, default=5000)
    args = parser.parse_args()

    methods = args.methods or sorted(path.name for path in args.artifact_root.iterdir() if path.is_dir())
    datasets = args.datasets or sorted(DATASETS)

    summaries: list[dict[str, object]] = []
    mismatch_rows: list[dict[str, object]] = []
    for method in methods:
        for dataset in datasets:
            pred_dir = args.artifact_root / method / dataset
            if not pred_dir.is_dir():
                continue
            spec = _dataset_spec(dataset)
            labels = read_vocab(spec.vocab)
            summary, mismatches = _analyse_method_dataset(
                method=method,
                pred_dir=pred_dir,
                spec=spec,
                labels=labels,
                include_biou=args.include_biou,
                max_images=args.max_images,
                max_biou_regions=args.max_biou_regions,
            )
            summaries.append(summary)
            mismatch_rows.extend(mismatches)
            print(
                f"{method}/{dataset}: regions={summary['gt_class_regions']} "
                f"top1={float(summary['gt_region_naming_top1']):.4f} "
                f"recall05={float(summary['proposal_recall_at_05']):.4f}",
                flush=True,
            )

    suffix = "_sample" if args.max_images is not None else ""
    if args.include_biou:
        suffix += "_biou"

    table9 = args.out_dir / f"table9_diagnostic_probes{suffix}.csv"
    table11 = args.out_dir / f"table11_e2_mismatch_summary{suffix}.csv"
    table11_pairs = args.out_dir / f"table11_e2_mismatch_pairs{suffix}.csv"
    table12 = args.out_dir / f"table12_supporting_readouts{suffix}.csv"

    _write_csv(table9, summaries)
    _write_csv(table11, _table11_rows(summaries))
    _write_csv(table11_pairs, mismatch_rows)
    _write_csv(table12, _table12_rows(summaries))

    _write_md(
        args.out_dir / f"table9_diagnostic_probes{suffix}.md",
        "Table 9 Diagnostic Probes",
        summaries,
        [
            "method",
            "dataset",
            "gt_region_naming_top1",
            "gt_text_localization_iou",
            "gt_text_localization_biou",
            "proposal_oracle_iou",
            "proposal_recall_at_05",
            "mcmr_at_05",
            "diagnostic_source",
            "artifact_gap",
        ],
    )
    _write_md(
        args.out_dir / f"table11_e2_mismatch_summary{suffix}.md",
        "Table 11 E2 Mismatch Summary",
        _table11_rows(summaries),
        [
            "method_group",
            "localized_pairs",
            "mismatch_pairs_proxy",
            "MCMR@0.5_proxy",
            "GT_region_top1_proxy",
            "proposal_recall_at_0.5_proxy",
            "status",
        ],
    )
    _write_md(
        args.out_dir / f"table12_supporting_readouts{suffix}.md",
        "Table 12 Spreadsheet Supporting Readouts",
        _table12_rows(summaries),
        ["readout", "what_to_record", "current_output", "status"],
    )
    print(json.dumps({"table9": str(table9), "table11": str(table11), "table12": str(table12)}, indent=2))


if __name__ == "__main__":
    main()
