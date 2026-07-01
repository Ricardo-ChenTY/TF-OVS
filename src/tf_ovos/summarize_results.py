"""Summarise TF-OVOS E1/E2/E3/E4 result files.

Standard-tier columns (mIoU) come first in every table.
Exploratory-tier columns (MCMR, delta_vocab, per-class, …) follow and are
clearly separated — they do not change method rankings.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

# Standard-tier datasets used for E1/E2/E3 ranking.
E1_DATASETS = ["voc20_val", "context59_val", "ade20k150_val", "coco_stuff171_val"]
# E2 large-vocabulary counterparts (paired with compact above).
E2_LARGE = {"context59_val": "context459_val", "ade20k150_val": "ade20k847_val"}
# Appendix hard-domain datasets (mask-only / class-aware).
APPENDIX_DATASETS = ["ovcamo_te", "camo_te", "cod10k_te_camo", "nc4k"]


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _flatten(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, dict):
            flat.update(_flatten(f"{prefix}{key}.", value))
        elif isinstance(value, list):
            flat[f"{prefix}{key}"] = json.dumps(value, ensure_ascii=False)
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
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


def _runtime_summary(method_root: Path, dataset_name: str) -> dict[str, Any]:
    runtime_paths = sorted((method_root / dataset_name / "shards").glob("part-*/runtime.json"))
    runtimes = [_load_json(path) for path in runtime_paths]
    runtimes = [row for row in runtimes if row is not None]
    total_samples = sum(int(row.get("num_samples", 0)) for row in runtimes)
    total_time = sum(float(row.get("wall_time_sec", 0.0)) for row in runtimes)
    peak_memory_values = [
        float(row["peak_memory_mb"])
        for row in runtimes
        if row.get("peak_memory_mb") is not None
    ]
    model_call_values = [
        int(row["model_calls"])
        for row in runtimes
        if row.get("model_calls") is not None
    ]
    return {
        "runtime_shards": len(runtimes),
        "runtime_num_samples": total_samples,
        "runtime_wall_time_sec": total_time,
        "runtime_sec_per_image": total_time / total_samples if total_samples else None,
        "runtime_peak_memory_mb": max(peak_memory_values) if peak_memory_values else None,
        "runtime_model_calls": sum(model_call_values) if model_call_values else None,
    }


def collect_method(method: str, cfg: dict[str, Any], run_root: Path) -> dict[str, Any]:
    datasets = cfg.get("datasets", {})
    method_root = run_root / method

    # --- per-dataset rows (all tasks) ---
    dataset_rows: list[dict[str, Any]] = []
    # metric dicts keyed by dataset name for downstream E2/E3 assembly
    metrics_by_dataset: dict[str, dict[str, Any]] = {}

    for dataset_name, dataset in datasets.items():
        metrics_path = method_root / dataset_name / "metrics.json"
        metrics = _load_json(metrics_path)
        if metrics is None:
            continue
        task = dataset.get("task", metrics.get("task", "mask-only"))
        row: dict[str, Any] = {
            "method": method,
            "dataset": dataset_name,
            "task": task,
            "metrics_path": str(metrics_path),
        }
        # Exclude large per_class_iou list from flat CSV (keep in JSON only).
        flat_metrics = {k: v for k, v in metrics.items() if k not in ("ambiguity", "per_class_iou")}
        row.update(_flatten("", flat_metrics))
        row.update(_runtime_summary(method_root, dataset_name))
        dataset_rows.append(row)
        metrics_by_dataset[dataset_name] = metrics

    # --- E1: standard mIoU leaderboard ---
    e1_row: dict[str, Any] = {"method": method}
    e1_miou_values: list[float] = []
    for ds in E1_DATASETS:
        m = metrics_by_dataset.get(ds, {})
        val = m.get("miou")
        e1_row[f"{ds}_mIoU"] = val
        if val is not None:
            e1_miou_values.append(float(val))
    e1_row["avg_mIoU"] = sum(e1_miou_values) / len(e1_miou_values) if e1_miou_values else None

    # --- E2: vocabulary robustness ---
    e2_row: dict[str, Any] = {"method": method}
    delta_parts: list[float] = []
    for compact_ds, large_ds in E2_LARGE.items():
        compact_miou = (metrics_by_dataset.get(compact_ds) or {}).get("miou")
        large_miou = (metrics_by_dataset.get(large_ds) or {}).get("miou")
        e2_row[f"{compact_ds}_mIoU"] = compact_miou
        e2_row[f"{large_ds}_mIoU"] = large_miou
        if compact_miou is not None and large_miou is not None:
            delta = float(compact_miou) - float(large_miou)
            e2_row[f"delta_{compact_ds.split('_')[0]}"] = delta
            delta_parts.append(delta)
    e2_row["delta_vocab"] = sum(delta_parts) / len(delta_parts) if delta_parts else None
    # Exploratory: MCMR from compact-vocab datasets
    mcmr_values: list[float] = []
    for ds in E2_LARGE:
        m = metrics_by_dataset.get(ds, {})
        v = m.get("mcmr_05")
        if v is not None:
            mcmr_values.append(float(v))
    e2_row["mcmr_05"] = sum(mcmr_values) / len(mcmr_values) if mcmr_values else None

    # Appendix E2 ambiguity (class-aware targets)
    e2_ambiguity_rows: list[dict[str, Any]] = []
    for dataset_name, metrics in metrics_by_dataset.items():
        if "ambiguity" in metrics:
            row = {"method": method, "dataset": dataset_name}
            row.update(_flatten("", metrics["ambiguity"]))
            e2_ambiguity_rows.append(row)

    # --- E3: cross-dataset generalization ---
    e3_row: dict[str, Any] = {"method": method}
    e3_miou_values: list[float] = []
    for ds in E1_DATASETS:
        val = (metrics_by_dataset.get(ds) or {}).get("miou")
        e3_row[f"{ds}_mIoU"] = val
        if val is not None:
            e3_miou_values.append(float(val))
    e3_row["avg_mIoU"] = sum(e3_miou_values) / len(e3_miou_values) if e3_miou_values else None
    e3_row["worst_mIoU"] = min(e3_miou_values) if e3_miou_values else None

    # --- E4: test-time cost / efficiency ---
    e4_rows: list[dict[str, Any]] = []
    for row in dataset_rows:
        quality_metric = "mIoU" if row.get("miou") is not None else "IoU"
        quality_value = row.get("miou")
        if quality_value is None:
            quality_value = row.get("mask_metrics.IoU")
        sec_per_image = row.get("runtime_sec_per_image")
        e4_rows.append(
            {
                "method": method,
                "dataset": row["dataset"],
                "task": row["task"],
                "quality_metric": quality_metric,
                "quality_value": quality_value,
                "num_samples": row.get("num_samples"),
                "runtime_shards": row.get("runtime_shards"),
                "runtime_num_samples": row.get("runtime_num_samples"),
                "runtime_wall_time_sec": row.get("runtime_wall_time_sec"),
                "runtime_sec_per_image": sec_per_image,
                "runtime_peak_memory_mb": row.get("runtime_peak_memory_mb"),
                "runtime_model_calls": row.get("runtime_model_calls"),
                "quality_per_second": (
                    float(quality_value) / float(sec_per_image)
                    if quality_value is not None and sec_per_image
                    else None
                ),
                "note": "Runtime is collected from shard runtime.json; use isolated reruns for final paper-grade hardware numbers.",
            }
        )

    # Appendix: hard-domain mask-only targets
    appendix_row: dict[str, Any] = {"method": method}
    for ds in APPENDIX_DATASETS:
        m = metrics_by_dataset.get(ds, {})
        mm = m.get("mask_metrics", {})
        appendix_row[f"{ds}_IoU"] = mm.get("IoU")
        cam = m.get("class_aware_metrics", {})
        if cam:
            appendix_row[f"{ds}_cIoU"] = cam.get("cIoU")

    return {
        "datasets": dataset_rows,
        "e1": e1_row,
        "e2": e2_row,
        "e2_ambiguity": e2_ambiguity_rows,
        "e3": e3_row,
        "e4": e4_rows,
        "appendix": appendix_row,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize TF-OVOS E1/E2/E3/E4 result files.")
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark.yaml"))
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--method", action="append", help="Method folder under run-root. Omit to scan all.")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/tables"))
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    methods = args.method or sorted(path.name for path in args.run_root.iterdir() if path.is_dir())

    dataset_rows: list[dict[str, Any]] = []
    e1_rows: list[dict[str, Any]] = []
    e2_rows: list[dict[str, Any]] = []
    e2_ambiguity_rows: list[dict[str, Any]] = []
    e3_rows: list[dict[str, Any]] = []
    e4_rows: list[dict[str, Any]] = []
    appendix_rows: list[dict[str, Any]] = []

    for method in methods:
        collected = collect_method(method, cfg, args.run_root)
        dataset_rows.extend(collected["datasets"])
        e1_rows.append(collected["e1"])
        e2_rows.append(collected["e2"])
        e2_ambiguity_rows.extend(collected["e2_ambiguity"])
        e3_rows.append(collected["e3"])
        e4_rows.extend(collected["e4"])
        appendix_rows.append(collected["appendix"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(
            {
                "datasets": dataset_rows,
                "e1": e1_rows,
                "e2": e2_rows,
                "e2_ambiguity": e2_ambiguity_rows,
                "e3": e3_rows,
                "e4": e4_rows,
                "appendix": appendix_rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(dataset_rows, args.out_dir / "dataset_metrics.csv")
    _write_csv(e1_rows, args.out_dir / "e1_standard.csv")
    _write_csv(e2_rows, args.out_dir / "e2_vocab_robustness.csv")
    _write_csv(e2_ambiguity_rows, args.out_dir / "e2_ambiguity.csv")
    _write_csv(e3_rows, args.out_dir / "e3_generalization.csv")
    _write_csv(e4_rows, args.out_dir / "e4_cost.csv")
    _write_csv(appendix_rows, args.out_dir / "appendix_hard_domain.csv")
    print(f"Wrote summaries to {args.out_dir}")


if __name__ == "__main__":
    main()
