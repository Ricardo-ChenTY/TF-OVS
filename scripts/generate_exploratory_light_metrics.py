#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from tf_ovos.data import load_manifest, read_vocab
from tf_ovos.metrics import load_label_map, miou_from_confusion

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

@dataclass(frozen=True)
class DatasetSpec:
    name: str
    manifest: Path
    vocab: Path
    num_classes: int
    void_label: int
    prediction_label_offset: int


def _spec(name: str) -> DatasetSpec:
    d = DATASETS[name]
    return DatasetSpec(name, Path(d["manifest"]), Path(d["vocab"]), int(d["num_classes"]), int(d["void_label"]), int(d["prediction_label_offset"]))


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
    lines = [f"# {title}", ""]
    if not rows:
        lines.append("_No rows generated._")
    else:
        lines.append("| " + " | ".join(keys) + " |")
        lines.append("| " + " | ".join(["---"] * len(keys)) + " |")
        for row in rows:
            vals = []
            for key in keys:
                val = row.get(key, "")
                if isinstance(val, float):
                    vals.append(f"{val:.4f}")
                else:
                    vals.append(str(val).replace("|", "\\|"))
            lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prediction_path(pred_dir: Path, image_id: str) -> Path:
    path = pred_dir / f"{image_id}.png"
    if path.exists():
        return path
    return pred_dir / f"{Path(image_id).name}.png"


def _load_prediction(path: Path, spec: DatasetSpec) -> np.ndarray:
    pred = np.asarray(Image.open(path), dtype=np.int32)
    if spec.prediction_label_offset:
        pred = pred + spec.prediction_label_offset
    return pred


def _resize_like(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if pred.shape == gt.shape:
        return pred
    image = Image.fromarray(pred.astype(np.uint16))
    image = image.resize((gt.shape[1], gt.shape[0]), resample=Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.int32)


def _confusion_for_method_dataset(method: str, dataset: str, max_images: int | None) -> tuple[np.ndarray, int, int]:
    spec = _spec(dataset)
    pred_dir = ARTIFACT_ROOT / method / dataset
    samples = load_manifest(spec.manifest)
    if max_images is not None:
        samples = samples[:max_images]
    conf = np.zeros((spec.num_classes, spec.num_classes), dtype=np.int64)
    processed = missing = 0
    for sample in samples:
        pred_path = _prediction_path(pred_dir, sample.image_id)
        if not pred_path.exists():
            missing += 1
            continue
        gt = load_label_map(str(sample.mask_path), spec.void_label)
        pred = _resize_like(_load_prediction(pred_path, spec), gt)
        valid = (gt >= 0) & (gt != spec.void_label) & (gt < spec.num_classes)
        pred_valid = np.where((pred >= 0) & (pred < spec.num_classes), pred, -1)
        good = valid & (pred_valid >= 0)
        if good.any():
            gt_flat = gt[good].astype(np.int64)
            pred_flat = pred_valid[good].astype(np.int64)
            conf += np.bincount(spec.num_classes * gt_flat + pred_flat, minlength=spec.num_classes * spec.num_classes).reshape(spec.num_classes, spec.num_classes)
        processed += 1
    return conf, processed, missing


def _taxonomy_rows(methods: list[str], datasets: list[str], max_images: int | None) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    synonym_rows: list[dict[str, object]] = []
    conf_cache: dict[tuple[str, str], np.ndarray] = {}

    for dataset in datasets:
        spec = _spec(dataset)
        labels = read_vocab(spec.vocab)
        alias_to_ids: dict[str, set[int]] = defaultdict(set)
        for i, label in enumerate(labels):
            for alias in label.split(","):
                norm = re.sub(r"[^a-z0-9]+", " ", alias.lower()).strip()
                if norm:
                    alias_to_ids[norm].add(i)
        groups = {alias: sorted(ids) for alias, ids in alias_to_ids.items() if len(ids) > 1}
        for alias, ids in sorted(groups.items()):
            synonym_rows.append({
                "dataset": dataset,
                "alias": alias,
                "class_ids": ";".join(map(str, ids)),
                "classes": "; ".join(labels[i] for i in ids),
                "n_classes": len(ids),
                "status": "alias_collision_found",
            })

        for method in methods:
            pred_dir = ARTIFACT_ROOT / method / dataset
            if not pred_dir.is_dir():
                continue
            conf, processed, missing = _confusion_for_method_dataset(method, dataset, max_images)
            conf_cache[(method, dataset)] = conf
            per_class, miou = miou_from_confusion(conf)

            if dataset == "coco_stuff164k":
                thing_ids = list(range(80))
                stuff_ids = list(range(80, min(171, spec.num_classes)))
                for split_name, ids in [("thing", thing_ids), ("stuff", stuff_ids)]:
                    vals = [float(per_class[i]) for i in ids if i < len(per_class) and not np.isnan(per_class[i])]
                    rows.append({
                        "metric": "thing_stuff_mIoU",
                        "method": method,
                        "dataset": dataset,
                        "split": split_name,
                        "value": float(np.mean(vals)) if vals else None,
                        "n_classes": len(vals),
                        "processed_images": processed,
                        "missing_images": missing,
                        "status": "computed_from_coco_vocab_order",
                    })

            if groups:
                # Collapse only classes sharing a normalized alias. Other classes remain singleton groups.
                class_to_group = list(range(spec.num_classes))
                next_group = spec.num_classes
                for ids in groups.values():
                    gid = next_group
                    next_group += 1
                    for cid in ids:
                        class_to_group[cid] = gid
                unique = {gid: j for j, gid in enumerate(sorted(set(class_to_group)))}
                mapped = [unique[gid] for gid in class_to_group]
                collapsed = np.zeros((len(unique), len(unique)), dtype=np.int64)
                for gi in range(spec.num_classes):
                    for pi in range(spec.num_classes):
                        collapsed[mapped[gi], mapped[pi]] += conf[gi, pi]
                _, collapsed_miou = miou_from_confusion(collapsed)
                rows.append({
                    "metric": "synonym_alias_collapsed_mIoU",
                    "method": method,
                    "dataset": dataset,
                    "split": "alias_collision_groups",
                    "value": collapsed_miou,
                    "baseline_mIoU": miou,
                    "delta_vs_baseline": collapsed_miou - miou,
                    "n_alias_groups": len(groups),
                    "status": "computed_alias_collision_collapse",
                })
            else:
                rows.append({
                    "metric": "synonym_alias_collapsed_mIoU",
                    "method": method,
                    "dataset": dataset,
                    "split": "alias_collision_groups",
                    "value": miou,
                    "baseline_mIoU": miou,
                    "delta_vs_baseline": 0.0,
                    "n_alias_groups": 0,
                    "status": "no_duplicate_aliases_in_vocab",
                })
    return rows, synonym_rows


def _communities(confusion_csv: Path, max_edges: int) -> list[dict[str, object]]:
    if not confusion_csv.exists():
        return []
    with confusion_csv.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_key: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("gt_class_id") == row.get("pred_class_id"):
            continue
        by_key[(row.get("method", ""), row.get("dataset", ""))].append(row)

    out: list[dict[str, object]] = []
    for (method, dataset), vals in by_key.items():
        def weight(row: dict[str, str]) -> float:
            for key in ("pixels", "count", "fraction_of_valid_pixels"):
                if row.get(key) not in (None, ""):
                    return float(row[key])
            return 0.0
        vals = sorted(vals, key=weight, reverse=True)[:max_edges]
        graph: dict[str, set[str]] = defaultdict(set)
        edge_weight: Counter[tuple[str, str]] = Counter()
        names: dict[str, str] = {}
        for row in vals:
            g = row["gt_class_id"]
            p = row["pred_class_id"]
            graph[g].add(p)
            graph[p].add(g)
            a, b = sorted((g, p))
            edge_weight[(a, b)] += weight(row)
            names[g] = row.get("gt_class", g)
            names[p] = row.get("pred_class", p)
        seen: set[str] = set()
        comps: list[list[str]] = []
        for node in graph:
            if node in seen:
                continue
            q = deque([node])
            seen.add(node)
            comp = []
            while q:
                cur = q.popleft()
                comp.append(cur)
                for nxt in graph[cur]:
                    if nxt not in seen:
                        seen.add(nxt)
                        q.append(nxt)
            comps.append(comp)
        comps.sort(key=lambda c: (-len(c), c[0]))
        for idx, comp in enumerate(comps, start=1):
            comp_set = set(comp)
            total = sum(w for (a, b), w in edge_weight.items() if a in comp_set and b in comp_set)
            out.append({
                "method": method,
                "dataset": dataset,
                "community_id": idx,
                "n_classes": len(comp),
                "classes": "; ".join(names[c] for c in sorted(comp, key=lambda x: int(x))),
                "class_ids": ";".join(sorted(comp, key=lambda x: int(x))),
                "edge_weight": float(total),
                "source_edges": len(vals),
                "status": "computed_from_top_confusion_pairs",
            })
    return out


def _component_labels(mask: np.ndarray) -> list[np.ndarray]:
    # Compact pure-numpy/Python connected components for binary masks. It is used on
    # per-class masks, so the queue stays small enough for exploratory summaries.
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if seen[sy, sx]:
            continue
        q = [(sy, sx)]
        seen[sy, sx] = True
        coords_y: list[int] = []
        coords_x: list[int] = []
        while q:
            y, x = q.pop()
            coords_y.append(y)
            coords_x.append(x)
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        comp = np.zeros(mask.shape, dtype=bool)
        comp[coords_y, coords_x] = True
        comps.append(comp)
    return comps


def _scale_bin(area: int) -> str:
    if area < 32 * 32:
        return "small"
    if area < 96 * 96:
        return "medium"
    return "large"


def _scale_component_rows(methods: list[str], datasets: list[str], max_images: int | None, min_component_area: int) -> list[dict[str, object]]:
    acc: dict[tuple[str, str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for dataset in datasets:
        spec = _spec(dataset)
        samples = load_manifest(spec.manifest)
        if max_images is not None:
            samples = samples[:max_images]
        for method in methods:
            pred_dir = ARTIFACT_ROOT / method / dataset
            if not pred_dir.is_dir():
                continue
            for sample in samples:
                pred_path = _prediction_path(pred_dir, sample.image_id)
                if not pred_path.exists():
                    continue
                gt = load_label_map(str(sample.mask_path), spec.void_label)
                pred = _resize_like(_load_prediction(pred_path, spec), gt)
                valid = (gt >= 0) & (gt != spec.void_label) & (gt < spec.num_classes)
                pred_valid = np.where((pred >= 0) & (pred < spec.num_classes), pred, -1)
                for cid in np.flatnonzero(np.bincount(gt[valid].astype(np.int64), minlength=spec.num_classes)):
                    gt_mask = valid & (gt == cid)
                    for comp in _component_labels(gt_mask):
                        area = int(comp.sum())
                        if area < min_component_area:
                            continue
                        pred_mask = valid & (pred_valid == cid)
                        inter = int((comp & pred_mask).sum())
                        union = int((comp | pred_mask).sum())
                        iou = inter / union if union else 0.0
                        key = (method, dataset, _scale_bin(area))
                        acc[key]["n_components"] += 1
                        acc[key]["sum_iou"] += iou
                        acc[key]["recall05"] += 1 if iou >= 0.5 else 0
                        acc[key]["recall075"] += 1 if iou >= 0.75 else 0
                        acc[key]["sum_area"] += area
            print(f"scale-components {method}/{dataset} done", flush=True)
    rows: list[dict[str, object]] = []
    for (method, dataset, scale), v in sorted(acc.items()):
        n = max(v["n_components"], 1)
        rows.append({
            "method": method,
            "dataset": dataset,
            "scale_bin": scale,
            "n_components": int(v["n_components"]),
            "mean_component_iou": v["sum_iou"] / n,
            "component_recall_at_05": v["recall05"] / n,
            "component_recall_at_075": v["recall075"] / n,
            "mean_component_area_px": v["sum_area"] / n,
            "status": "computed_from_gt_connected_components_and_prediction_maps",
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate low-cost exploratory metrics from saved prediction maps.")
    parser.add_argument("--methods", nargs="*")
    parser.add_argument("--datasets", nargs="*", choices=sorted(DATASETS))
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--max-confusion-edges", type=int, default=80)
    parser.add_argument("--min-component-area", type=int, default=64)
    parser.add_argument("--skip-components", action="store_true")
    args = parser.parse_args()

    methods = args.methods or sorted(path.name for path in ARTIFACT_ROOT.iterdir() if path.is_dir())
    datasets = args.datasets or sorted(DATASETS)
    suffix = "_sample" if args.max_images is not None else ""

    taxonomy_rows, synonym_rows = _taxonomy_rows(methods, datasets, args.max_images)
    community_rows = _communities(OUT_DIR / "official_confusion_pairs.csv", args.max_confusion_edges)
    mismatch_community_rows = _communities(OUT_DIR / "table11_e2_mismatch_pairs.csv", args.max_confusion_edges)
    component_rows = [] if args.skip_components else _scale_component_rows(methods, datasets, args.max_images, args.min_component_area)

    outputs = {
        "taxonomy": OUT_DIR / f"exploratory_taxonomy_light{suffix}.csv",
        "synonyms": OUT_DIR / f"exploratory_synonym_alias_groups{suffix}.csv",
        "confusion_communities": OUT_DIR / f"exploratory_confusion_communities{suffix}.csv",
        "mismatch_communities": OUT_DIR / f"exploratory_mismatch_communities{suffix}.csv",
        "scale_components": OUT_DIR / f"exploratory_scale_components{suffix}.csv",
    }
    _write_csv(outputs["taxonomy"], taxonomy_rows)
    _write_csv(outputs["synonyms"], synonym_rows)
    _write_csv(outputs["confusion_communities"], community_rows)
    _write_csv(outputs["mismatch_communities"], mismatch_community_rows)
    _write_csv(outputs["scale_components"], component_rows)

    _write_md(OUT_DIR / f"exploratory_taxonomy_light{suffix}.md", "Exploratory Taxonomy-Light Metrics", taxonomy_rows, ["metric", "method", "dataset", "split", "value", "delta_vs_baseline", "status"])
    _write_md(OUT_DIR / f"exploratory_confusion_communities{suffix}.md", "Exploratory Confusion Communities", community_rows[:100], ["method", "dataset", "community_id", "n_classes", "classes", "edge_weight", "status"])
    _write_md(OUT_DIR / f"exploratory_scale_components{suffix}.md", "Exploratory Scale/Component Metrics", component_rows, ["method", "dataset", "scale_bin", "n_components", "mean_component_iou", "component_recall_at_05", "component_recall_at_075", "status"])
    print(json.dumps({k: str(v) for k, v in outputs.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
