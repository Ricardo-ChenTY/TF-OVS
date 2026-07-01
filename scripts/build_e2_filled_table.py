#!/usr/bin/env python
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image

from tf_ovos.data import load_manifest
from tf_ovos.metrics import load_label_map


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "analysis"
ARTIFACT_DIR = ROOT / "runs" / "artifacts" / "official_predictions"


DATASETS = {
    "context59": ("context59_val", ROOT / "data/manifests/context59_val.jsonl", 59, 255, -1),
    "context459": ("context459_val", ROOT / "data/manifests/context459_val.jsonl", 459, 65535, 0),
    "ade20k": ("ade20k150_val", ROOT / "data/manifests/ade20k150_val.jsonl", 150, 255, -1),
    "ade847": ("ade20k847_val", ROOT / "data/manifests/ade20k847_val.jsonl", 847, 65535, 0),
}


METHODS = [
    ("maskclip", "MaskCLIP", "local_json"),
    ("maskclip_attn", "MaskCLIP-Attn", "local_json"),
    ("maskclip_attn_slide", "MaskCLIP-Attn-Slide", "local_json"),
    ("sclip", "SCLIP", "official_best"),
    ("naclip", "NACLIP", "official_best"),
    ("cliptrase", "CLIPtrase", "cliptrase"),
    ("scclip", "SC-CLIP", "mmengine"),
    ("resclip", "ResCLIP", "official_best"),
    ("proxyclip", "ProxyCLIP", "official_best"),
    ("corrclip", "CorrCLIP", "official_best"),
    ("trident", "Trident", "mmengine"),
    ("cass", "CASS", "mmengine"),
    ("freeda", "FreeDA", "freeda"),
    ("sam_amg_clip", "SAM-AMG + CLIP", "local_json"),
    ("sam_amg_siglip", "SAM-AMG + SigLIP", "local_json"),
    ("dinov2_sam_clip", "DINOv2 + SAM + CLIP", "local_json"),
    ("dinov2_sam_siglip", "DINOv2 + SAM + SigLIP", "local_json"),
    ("san", "SAN (ref.)", "san"),
]


MMENGINE_LOGS = {
    ("scclip", "context59"): "runs/logs/official_scclip_context59.log",
    ("scclip", "context459"): "runs/logs/official_scclip_context459.log",
    ("scclip", "ade20k"): "runs/logs/official_scclip_ade20k.log",
    ("scclip", "ade847"): "runs/logs/official_scclip_ade847.log",
    ("trident", "context59"): "runs/logs/official_trident_context59.log",
    ("trident", "context459"): "runs/logs/official_trident_context459.log",
    ("trident", "ade20k"): "runs/logs/official_trident_ade20k.log",
    ("trident", "ade847"): "runs/logs/official_trident_ade847.log",
    ("cass", "context59"): "runs/logs/official_cass_context59.log",
    ("cass", "context459"): "runs/logs/official_cass_context459.log",
    ("cass", "ade20k"): "runs/logs/official_cass_ade20k.log",
    ("cass", "ade847"): "runs/logs/official_cass_ade847.log",
}


FREEDA_LOGS = {
    "context59": "runs/freeda/context59/log.txt",
    "context459": "runs/freeda/context459/log.txt",
    "ade20k": "runs/freeda/ade20k150/log.txt",
    "ade847": "runs/freeda/ade847/log.txt",
}


CLIPTRASE_LOGS = {
    "context59": "runs/logs/official_cliptrase_PC59_e1.log",
    "context459": "runs/logs/official_cliptrase_PC459.log",
    "ade20k": "runs/logs/official_cliptrase_ADE150_e1.log",
    "ade847": "runs/logs/official_cliptrase_ADEfull.log",
}


SAN_DATASET_ORDER = ["coco_stuff164k", "voc20", "context59", "ade20k", "context459", "ade847"]


def read_official_best() -> dict[tuple[str, str], dict[str, str]]:
    path = OUT_DIR / "official_best_metrics.csv"
    with path.open(newline="") as handle:
        return {(r["method"], r["dataset"]): r for r in csv.DictReader(handle)}


def read_table9_mcmr() -> dict[str, float]:
    vals: dict[str, list[float]] = {}
    for path in [OUT_DIR / "table9_diagnostic_probes.csv", OUT_DIR / "san_diag" / "table9_diagnostic_probes.csv"]:
        if not path.exists():
            continue
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row["dataset"] not in {"context459", "ade847"}:
                    continue
                if row.get("mcmr_at_05"):
                    vals.setdefault(row["method"], []).append(float(row["mcmr_at_05"]))
    return {method: float(np.mean(v)) for method, v in vals.items()}


def finite_stats(values: list[float]) -> tuple[int, int, float]:
    finite = [v for v in values if not math.isnan(v)]
    zero = sum(1 for v in finite if abs(v) < 1e-12)
    return len(finite), zero, 100.0 * zero / len(finite)


def local_json_metrics(method: str, dataset: str) -> tuple[float, tuple[int, int, float], float | None]:
    run_dir, _, _, _, _ = DATASETS[dataset]
    data = json.loads((ROOT / "runs" / method / run_dir / "metrics.json").read_text())
    return 100.0 * float(data["miou"]), finite_stats([float(v) for v in data["per_class_iou"]]), data.get("mcmr_05")


def parse_mmengine_log(path: Path) -> tuple[float, tuple[int, int, float]]:
    text = path.read_text(errors="ignore")
    miou = float(re.findall(r"mIoU:\s*([0-9.]+)", text)[-1])
    vals: list[float] = []
    in_table = False
    for line in text.splitlines():
        if "per class results" in line:
            in_table = True
            continue
        if in_table and ("Iter(test)" in line or "Summary" in line):
            if vals:
                break
        if in_table and "|" in line:
            nums = []
            for cell in line.split("|"):
                try:
                    nums.append(float(cell.strip()))
                except ValueError:
                    pass
            if nums:
                vals.append(nums[0])
    return miou, finite_stats(vals)


def parse_freeda_log(path: Path, dataset: str) -> tuple[float, tuple[int, int, float]]:
    text = path.read_text(errors="ignore")
    miou_match = re.search(rf"\[{re.escape(dataset)}\] mIoU .*?:\s*([0-9.]+)%", text)
    if not miou_match and dataset == "ade20k":
        miou_match = re.search(r"\[ade20k\] mIoU .*?:\s*([0-9.]+)%", text)
    vals = [float(m.group(1)) for m in re.finditer(r"'IoU\.[^']+': ([0-9.eE+-]+|nan)", text) if m.group(1) != "nan"]
    return float(miou_match.group(1)), finite_stats(vals)


def parse_cliptrase_miou(path: Path) -> float:
    text = path.read_text(errors="ignore")
    vals = [float(x) for x in re.findall(r"'mIoU': np\.float64\(([0-9.]+)\)", text)]
    return vals[-1]


def pred_path(pred_dir: Path, image_id: str) -> Path:
    path = pred_dir / f"{image_id}.png"
    if path.exists():
        return path
    return pred_dir / f"{Path(image_id).name}.png"


def compute_map_ziou(method: str, dataset: str) -> tuple[int, int, float]:
    _, manifest, num_classes, void_label, offset = DATASETS[dataset]
    inter = np.zeros(num_classes, dtype=np.float64)
    union = np.zeros(num_classes, dtype=np.float64)
    pred_dir = ARTIFACT_DIR / method / dataset
    for sample in load_manifest(manifest):
        path = pred_path(pred_dir, sample.image_id)
        if not path.exists():
            continue
        gt = load_label_map(str(sample.mask_path), void_label)
        pred = np.asarray(Image.open(path), dtype=np.int32) + offset
        if pred.shape != gt.shape:
            pred = np.asarray(
                Image.fromarray(pred.astype(np.uint16)).resize((gt.shape[1], gt.shape[0]), Image.Resampling.NEAREST),
                dtype=np.int32,
            )
        valid = (gt >= 0) & (gt != void_label) & (gt < num_classes)
        pred_valid = valid & (pred >= 0) & (pred < num_classes)
        gt_flat = gt[valid].astype(np.int64)
        pred_flat = pred[pred_valid].astype(np.int64)
        gt_pred_flat = gt[pred_valid].astype(np.int64)
        gt_counts = np.bincount(gt_flat, minlength=num_classes)
        pred_counts = np.bincount(pred_flat, minlength=num_classes)
        same = pred_flat == gt_pred_flat
        same_counts = np.bincount(gt_pred_flat[same], minlength=num_classes)
        inter += same_counts
        union += gt_counts + pred_counts - same_counts
    finite = union > 0
    iou = np.divide(inter[finite], union[finite], out=np.zeros_like(inter[finite]), where=union[finite] > 0)
    return finite_stats([float(v) for v in iou])


def parse_san_log() -> dict[str, tuple[float, tuple[int, int, float]]]:
    text = (ROOT / "runs/logs/san_official_all.log").read_text(errors="ignore")
    chunks = text.split("OrderedDict([('sem_seg', {")[1:]
    parsed: dict[str, tuple[float, tuple[int, int, float]]] = {}
    for dataset, chunk in zip(SAN_DATASET_ORDER, chunks):
        body = chunk.split("'mACC':", 1)[0]
        miou = float(re.search(r"'mIoU': ([0-9.]+)", body).group(1))
        vals = [float(x) for x in re.findall(r"'IoU-[^']+': ([0-9.eE+-]+)", body)]
        parsed[dataset] = (miou, finite_stats(vals))
    return parsed


def fmt(value: float | None, digits: int) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def make_latex(rows: list[dict[str, object]]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{E2 vocabulary and category-ambiguity robustness with completed diagnostic readouts. The first four numeric columns report target mIoU. ZIoU is the zero-IoU class-collapse rate over finite per-class IoU entries in each large vocabulary; lower is better. MCMR@0.5 is averaged over Context-459 and ADE-847 when available.}",
        r"\label{tab:e2-vocab-robustness-filled}",
        r"\setlength{\tabcolsep}{2.5pt}",
        "\\begin{adjustbox}{max width=\\textwidth}\n\\begin{tabular}{lrrrrrrrrp{0.24\\textwidth}}",
        r"\toprule",
        r"Method & Ctx-59$\uparrow$ & Ctx-459$\uparrow$ & Ctx459 ZIoU$\downarrow$ & ADE-150$\uparrow$ & ADE-847$\uparrow$ & ADE847 ZIoU$\downarrow$ & $\Delta_{\mathrm{vocab}}\downarrow$ & MCMR@0.5$\downarrow$ & Note \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['method']} & {row['context59_miou']} & {row['context459_miou']} & {row['context459_ziou']} & "
            f"{row['ade150_miou']} & {row['ade847_miou']} & {row['ade847_ziou']} & {row['delta_vocab']} & "
            f"{row['mcmr_05_e2']} & {row['note']} \\\\"
        )
    lines.extend([r"\bottomrule", "\\end{tabular}\n\\end{adjustbox}", r"\end{table}", ""])
    return "\n".join(lines)


def main() -> None:
    best = read_official_best()
    table9_mcmr = read_table9_mcmr()
    san = parse_san_log()
    rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []

    for method, display, source in METHODS:
        miou: dict[str, float | None] = {}
        ziou: dict[str, float | None] = {}
        finite_counts: dict[str, str] = {}
        local_mcmr: dict[str, float] = {}

        for dataset in ["context59", "context459", "ade20k", "ade847"]:
            if source == "local_json":
                m, stats, mcmr = local_json_metrics(method, dataset)
                miou[dataset] = m
                ziou[dataset] = stats[2]
                finite_counts[dataset] = f"{stats[1]}/{stats[0]}"
                if mcmr is not None:
                    local_mcmr[dataset] = float(mcmr)
            elif source == "official_best":
                row = best[(method, {"ade20k": "ade20k"}.get(dataset, dataset))]
                miou[dataset] = float(row["miou"])
                if row.get("zero_class_iou_rate"):
                    ziou[dataset] = 100.0 * float(row["zero_class_iou_rate"])
                    finite_counts[dataset] = f"{row['zero_class_iou_count']}/{row['num_classes_logged']}"
            elif source == "mmengine":
                m, stats = parse_mmengine_log(ROOT / MMENGINE_LOGS[(method, dataset)])
                miou[dataset] = m
                ziou[dataset] = stats[2]
                finite_counts[dataset] = f"{stats[1]}/{stats[0]}"
            elif source == "freeda":
                m, stats = parse_freeda_log(ROOT / FREEDA_LOGS[dataset], dataset)
                miou[dataset] = m
                ziou[dataset] = stats[2]
                finite_counts[dataset] = f"{stats[1]}/{stats[0]}"
            elif source == "cliptrase":
                miou[dataset] = parse_cliptrase_miou(ROOT / CLIPTRASE_LOGS[dataset])
                if dataset in {"context459", "ade847"}:
                    stats = compute_map_ziou(method, dataset)
                    ziou[dataset] = stats[2]
                    finite_counts[dataset] = f"{stats[1]}/{stats[0]}"
            elif source == "san":
                m, stats = san[dataset]
                miou[dataset] = m
                ziou[dataset] = stats[2]
                finite_counts[dataset] = f"{stats[1]}/{stats[0]}"

        mcmr_vals = []
        if method in table9_mcmr:
            mcmr = table9_mcmr[method]
        else:
            mcmr_vals = [local_mcmr[d] for d in ["context459", "ade847"] if d in local_mcmr]
            mcmr = float(np.mean(mcmr_vals)) if mcmr_vals else None

        delta_context = miou["context59"] - miou["context459"]
        delta_ade = miou["ade20k"] - miou["ade847"]
        delta_vocab = (delta_context + delta_ade) / 2.0
        note = "filled from existing artifacts"
        if method == "san":
            note = "ZIoU from log; MCMR from converted SAN maps"
        elif method == "cliptrase":
            note = "ZIoU map-proxy; MCMR from dense-map diagnostics"

        out = {
            "method": display,
            "context59_miou": fmt(miou["context59"], 2),
            "context459_miou": fmt(miou["context459"], 2),
            "context459_ziou": fmt(ziou.get("context459"), 1),
            "ade150_miou": fmt(miou["ade20k"], 2),
            "ade847_miou": fmt(miou["ade847"], 2),
            "ade847_ziou": fmt(ziou.get("ade847"), 1),
            "delta_vocab": fmt(delta_vocab, 2),
            "mcmr_05_e2": fmt(mcmr, 3),
            "note": note,
        }
        rows.append(out)
        source_rows.append(
            {
                **out,
                "context459_zero_classes": finite_counts.get("context459", ""),
                "ade847_zero_classes": finite_counts.get("ade847", ""),
                "source": source,
            }
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "e2_vocab_robustness_filled.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (OUT_DIR / "e2_vocab_robustness_filled_sources.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)

    md_lines = ["# E2 Vocabulary Robustness Filled", ""]
    headers = list(rows[0])
    md_lines.append("| " + " | ".join(headers) + " |")
    md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        md_lines.append("| " + " | ".join(str(row[h]).replace("|", "\\|") for h in headers) + " |")
    (OUT_DIR / "e2_vocab_robustness_filled.md").write_text("\n".join(md_lines) + "\n")
    (OUT_DIR / "e2_vocab_robustness_filled.tex").write_text(make_latex(rows))


if __name__ == "__main__":
    main()
