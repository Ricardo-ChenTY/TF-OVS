#!/usr/bin/env python
from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "runs" / "analysis"
LOGS = ROOT / "runs" / "logs"
PRED = ROOT / "runs" / "artifacts" / "official_predictions"
VOCAB = ROOT / "configs" / "vocab"
OUT_CSV = ANALYSIS / "alias_collapse_decomposition.csv"

DATASETS = {
    "context459": {
        "vocab": VOCAB / "context_459.txt",
        "compact": "context59",
    },
    "ade847": {
        "vocab": VOCAB / "ade20k_847.txt",
        "compact": "ade20k",
    },
}

METHODS = ("sclip", "naclip", "proxyclip", "corrclip", "san")


def read_vocab(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def alias_groups(labels: list[str]) -> list[list[int]]:
    alias_to_ids: dict[str, set[int]] = defaultdict(set)
    for idx, label in enumerate(labels):
        for alias in label.split(","):
            norm = re.sub(r"[^a-z0-9]+", " ", alias.lower()).strip()
            if norm:
                alias_to_ids[norm].add(idx)
    return [sorted(ids) for ids in alias_to_ids.values() if len(ids) > 1]


def parse_pipe_log(path: Path) -> list[float]:
    values: list[float] = []
    row_re = re.compile(r"^\|\s*(?P<class>[^|]+?)\s*\|\s*(?P<iou>nan|[-+0-9.]+)\s*\|\s*(?P<acc>nan|[-+0-9.]+)\s*\|")
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = row_re.match(line)
        if not match:
            continue
        raw = match.group("iou")
        values.append(float("nan") if raw == "nan" else float(raw))
    return values


def parse_san_blocks(path: Path) -> dict[str, list[float]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    blocks: dict[str, list[float]] = {}
    for match in re.finditer(r"'mIoU':\s*([-+0-9.]+).*?(?=\n\[05/18|\Z)", text, flags=re.S):
        block = match.group(0)
        miou = float(match.group(1))
        values = []
        for item in re.finditer(r"'IoU-([^']+)':\s*(nan|[-+0-9.eE]+)", block):
            raw = item.group(2)
            values.append(float("nan") if raw == "nan" else float(raw))
        if len(values) > 800 or abs(miou - 10.2428) < 0.1:
            blocks["ade847"] = values
        elif len(values) > 300 or abs(miou - 12.75) < 0.5:
            blocks["context459"] = values
    return blocks


def optimistic_alias_ziou(values: list[float], groups: list[list[int]]) -> tuple[float, int]:
    class_to_group = list(range(len(values)))
    next_group = len(values)
    for ids in groups:
        gid = next_group
        next_group += 1
        for idx in ids:
            if idx < len(class_to_group):
                class_to_group[idx] = gid
    grouped: dict[int, list[float]] = defaultdict(list)
    for idx, value in enumerate(values):
        grouped[class_to_group[idx]].append(value)
    valid = 0
    zero = 0
    for vals in grouped.values():
        finite = [v for v in vals if not math.isnan(v)]
        if not finite:
            continue
        valid += 1
        if all(v == 0.0 for v in finite):
            zero += 1
    return (100.0 * zero / valid if valid else float("nan")), valid


def zero_support_percent(method: str, dataset: str, zero_indices: set[int]) -> str:
    pred_dir = PRED / method / dataset
    if not pred_dir.is_dir():
        return "--"
    modes = set()
    support: set[int] = set()
    for path in pred_dir.glob("*.png"):
        image = Image.open(path)
        modes.add(image.mode)
        arr = np.asarray(image)
        support.update(int(v) for v in np.unique(arr) if int(v) >= 0)
    if dataset in {"context459", "ade847"} and "I;16" not in modes:
        return "--"
    if not zero_indices:
        return "--"
    unsupported = len([idx for idx in zero_indices if idx not in support])
    return f"{100.0 * unsupported / len(zero_indices):.1f}"


def main() -> None:
    with (ANALYSIS / "official_log_metrics.csv").open(encoding="utf-8") as handle:
        log_rows = list(csv.DictReader(handle))
    official = {}
    log_path = {}
    for row in log_rows:
        if row["method"] in METHODS and row["dataset"] in {"context59", "context459", "ade20k", "ade847"} and row["status"] == "complete":
            key = (row["method"], row["dataset"])
            if key not in official or row["variant"] in {"datafix", "single"}:
                official[key] = row
                log_path[key] = RESULT_ROOT / row["log_path"]

    with (ANALYSIS / "exploratory_taxonomy_light.csv").open(encoding="utf-8") as handle:
        taxonomy = list(csv.DictReader(handle))
    alias_gain = {
        (row["method"], row["dataset"]): float(row["delta_vs_baseline"]) * 100.0
        for row in taxonomy
        if row["metric"] == "synonym_alias_collapsed_mIoU"
    }

    san_blocks = parse_san_blocks(LOGS / "san_official_all.log")
    san_table = {
        ("san", "context59"): {"miou": "52.43"},
        ("san", "context459"): {"miou": "12.75", "zero_class_iou_rate": str(109 / 345), "zero_class_iou_count": "109", "num_classes_logged": "345"},
        ("san", "ade20k"): {"miou": "27.56"},
        ("san", "ade847"): {"miou": "10.24", "zero_class_iou_rate": str(434 / 846), "zero_class_iou_count": "434", "num_classes_logged": "846"},
    }

    rows: list[dict[str, str]] = []
    for method in METHODS:
        for dataset, ds_info in DATASETS.items():
            compact = ds_info["compact"]
            if method == "san":
                values = san_blocks[dataset]
                large = san_table[(method, dataset)]
                compact_miou = float(san_table[(method, compact)]["miou"])
                large_miou = float(large["miou"])
                official_ziou = 100.0 * float(large["zero_class_iou_rate"])
                official_zero = int(large["zero_class_iou_count"])
                valid_classes = int(large["num_classes_logged"])
                recovered = None
            else:
                values = parse_pipe_log(log_path[(method, dataset)])
                large = official[(method, dataset)]
                compact_miou = float(official[(method, compact)]["miou"])
                large_miou = float(large["miou"])
                official_ziou = 100.0 * float(large["zero_class_iou_rate"])
                official_zero = int(large["zero_class_iou_count"])
                valid_classes = int(large["num_classes_logged"])
                recovered = alias_gain.get((method, dataset), 0.0)

            labels = read_vocab(ds_info["vocab"])
            groups = alias_groups(labels)
            alias_ziou, alias_valid = optimistic_alias_ziou(values, groups)
            zero_indices = {idx for idx, value in enumerate(values) if not math.isnan(value) and value == 0.0}
            support = zero_support_percent(method, dataset, zero_indices)
            collapse = compact_miou - large_miou
            if recovered is None:
                recovered_text = "--"
                pct_text = "--"
            else:
                recovered_text = f"+{recovered:.2f}"
                pct_text = f"{(100.0 * recovered / collapse):.2f}" if collapse > 0 else "--"

            rows.append({
                "method": method,
                "dataset": dataset,
                "official_ziou_pct": f"{official_ziou:.1f}",
                "official_zero_classes": str(official_zero),
                "valid_classes": str(valid_classes),
                "alias_groups": str(len(groups)),
                "alias_merged_ziou_pct": f"{alias_ziou:.1f}",
                "alias_valid_groups": str(alias_valid),
                "zero_iou_without_prediction_support_pct": support,
                "miou_recovered_points": recovered_text,
                "collapse_explained_pct": pct_text,
            })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(OUT_CSV)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
