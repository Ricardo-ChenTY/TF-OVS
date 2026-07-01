#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "runs" / "logs"
OUT_DIR = ROOT / "runs" / "analysis"
PRED_ARTIFACT_DIR = ROOT / "runs" / "artifacts" / "official_predictions"

E1_DATASETS = ("voc20", "context59", "ade20k", "coco_stuff164k")
E2_PAIRS = (("context59", "context459"), ("ade20k", "ade847"))
DATASET_TOTALS = {
    "voc20": 1449,
    "context59": 5105,
    "ade20k": 2000,
    "coco_stuff164k": 5000,
    "context459": 5105,
    "ade847": 2000,
}
METHOD_FAMILY = {
    "sclip": "clip_dense",
    "scclip": "clip_dense",
    "naclip": "clip_dense",
    "resclip": "clip_dense",
    "proxyclip": "clip_vfm",
    "corrclip": "clip_vfm",
    "freeda": "diffusion_reference",
    "ovdiff": "diffusion_reference",
    "diffsegmenter": "diffusion_reference",
    "cliptrase": "clip_dense",
    "trident": "clip_vfm",
    "cass": "clip_vfm",
    "ovseg": "trained_reference",
    "san": "trained_reference",
    "odise": "trained_reference",
}

METRIC_RE = re.compile(
    r"Iter\(test\)\s+\[\s*(?P<iter>\d+)\s*/\s*(?P<total>\d+)\]\s+"
    r".*?aAcc:\s*(?P<aacc>[-+0-9.]+)\s+"
    r"mIoU:\s*(?P<miou>[-+0-9.]+)\s+"
    r"mAcc:\s*(?P<macc>[-+0-9.]+)"
    r".*?\stime:\s*(?P<time>[-+0-9.]+)"
)
PROGRESS_RE = re.compile(
    r"Iter\(test\)\s+\[\s*(?P<iter>\d+)\s*/\s*(?P<total>\d+)\]"
    r".*?(?:eta:\s*(?P<eta>[0-9:]+))?"
    r".*?\stime:\s*(?P<time>[-+0-9.]+)"
    r".*?(?:memory:\s*(?P<memory>\d+))?"
)
DATASET_MARK_RE = re.compile(r"\[(?P<method>[A-Za-z]+)\]\s+dataset=(?P<dataset>[A-Za-z0-9_]+)\s+start")
CONFIG_RE = re.compile(r"--config\s+configs/cfg_tfovos_(?P<dataset>[A-Za-z0-9_]+)\.py")
WORKDIR_RE = re.compile(r"--work-dir\s+(?P<work>\S+)")
CLASS_ROW_RE = re.compile(
    r"^\|\s*(?P<class>[^|]+?)\s*\|\s*(?P<iou>nan|[-+0-9.]+)\s*\|\s*(?P<acc>nan|[-+0-9.]+)\s*\|"
)
FREEDA_RESULT_RE = re.compile(
    r"\[(?P<dataset>[A-Za-z0-9_]+)\]\s+mIoU of (?P<total>\d+) test images:\s*(?P<miou>[-+0-9.]+)%"
)
SUMMARY_ROW_RE = re.compile(
    r"^\|\s*(?P<aacc>[-+0-9.]+)\s*\|\s*(?P<miou>[-+0-9.]+)\s*\|\s*(?P<macc>[-+0-9.]+)\s*\|"
)
OVDIFF_IOU_RE = re.compile(r"^(?P<class>[A-Za-z_][A-Za-z0-9_ ]*):\s*(?P<iou>0?\.\d+)")
OVDIFF_MIOU_RE = re.compile(r"^mIoU\s+(?P<miou>[-+0-9.]+)")

PROPOSAL_NOVELTY_METRICS = [
    {
        "block": "E1",
        "metric": "per-class IoU summary",
        "proposal_group": "Group (i); cheap",
        "status": "computed_when_class_table_present",
        "result_inputs": "official mmseg per-class table",
        "note": "Includes worst-class IoU and zero-IoU class collapse rate when the official log prints the class table.",
    },
    {
        "block": "E1",
        "metric": "thing/stuff mIoU",
        "proposal_group": "Group (i); cheap",
        "status": "requires_taxonomy",
        "result_inputs": "per-class IoU plus dataset thing/stuff taxonomy",
        "note": "Can be computed from log class tables after adding dataset taxonomy maps.",
    },
    {
        "block": "E1",
        "metric": "pixel accuracy",
        "proposal_group": "Group (i); cheap",
        "status": "computed_from_official_log",
        "result_inputs": "official mmseg aAcc",
        "note": "Parsed from the official evaluator summary.",
    },
    {
        "block": "E1",
        "metric": "mean class accuracy",
        "proposal_group": "Group (i); cheap",
        "status": "computed_from_official_log",
        "result_inputs": "official mmseg mAcc",
        "note": "Parsed from the official evaluator summary.",
    },
    {
        "block": "E1",
        "metric": "frequency-stratified mIoU",
        "proposal_group": "Group (i); cheap",
        "status": "requires_taxonomy",
        "result_inputs": "per-class IoU plus train-frequency bins",
        "note": "Head/mid/tail tertiles need dataset frequency metadata.",
    },
    {
        "block": "E1",
        "metric": "small/medium/large object mIoU",
        "proposal_group": "Group (i); cheap",
        "status": "requires_prediction_artifacts",
        "result_inputs": "prediction label maps plus GT instance/region areas",
        "note": "Semantic mIoU logs do not include object area bins.",
    },
    {
        "block": "E1",
        "metric": "BIoU",
        "proposal_group": "Group (i); cheap",
        "status": "requires_prediction_artifacts",
        "result_inputs": "prediction label maps plus GT boundary maps",
        "note": "Boundary IoU needs saved masks, not only aggregate logs.",
    },
    {
        "block": "E1",
        "metric": "proposal Recall@0.5/0.7",
        "proposal_group": "Group (i); cheap",
        "status": "requires_prediction_artifacts",
        "result_inputs": "class-agnostic proposal masks plus GT masks",
        "note": "Available only for methods exposing proposal masks.",
    },
    {
        "block": "E1",
        "metric": "confidence calibration ECE",
        "proposal_group": "Group (i); cheap",
        "status": "requires_score_artifacts",
        "result_inputs": "per-pixel or per-region confidence and correctness",
        "note": "Needs confidence tensors/scores saved during official inference.",
    },
    {
        "block": "E1",
        "metric": "confidence calibration Brier",
        "proposal_group": "Group (i); cheap",
        "status": "requires_score_artifacts",
        "result_inputs": "per-pixel or per-region confidence and correctness",
        "note": "Needs confidence tensors/scores saved during official inference.",
    },
    {
        "block": "E1",
        "metric": "empty-prediction rate",
        "proposal_group": "Group (i); cheap",
        "status": "requires_prediction_artifacts",
        "result_inputs": "prediction label maps",
        "note": "Counts images or regions with no valid foreground / no non-ignore prediction.",
    },
    {
        "block": "E1",
        "metric": "over-segmentation rate",
        "proposal_group": "Group (i); cheap",
        "status": "requires_prediction_artifacts",
        "result_inputs": "prediction label maps or proposal masks",
        "note": "Needs connected-component or proposal-count statistics.",
    },
    {
        "block": "E1",
        "metric": "prediction coverage",
        "proposal_group": "Group (i); cheap",
        "status": "requires_prediction_artifacts",
        "result_inputs": "prediction label maps",
        "note": "Fraction of valid pixels receiving non-ignore predictions.",
    },
    {
        "block": "E2",
        "metric": "Delta_vocab",
        "proposal_group": "Group (ii); novelty-driving",
        "status": "computed_when_pairs_complete",
        "result_inputs": "compact-vocab mIoU and large-vocab mIoU",
        "note": "mIoU drop from Context59 to Context459 and ADE150 to ADE847.",
    },
    {
        "block": "E2",
        "metric": "MCMR@0.5",
        "proposal_group": "Group (ii); novelty-driving",
        "status": "requires_region_artifacts",
        "result_inputs": "matched predicted regions, GT regions, predicted labels",
        "note": "Primary mask-category mismatch rate.",
    },
    {
        "block": "E2",
        "metric": "MCMR@0.75",
        "proposal_group": "Group (ii); novelty-driving",
        "status": "requires_region_artifacts",
        "result_inputs": "matched predicted regions, GT regions, predicted labels",
        "note": "Strict mask-category mismatch rate.",
    },
    {
        "block": "E2",
        "metric": "semantic-distance-weighted mismatch",
        "proposal_group": "Group (ii); novelty-driving",
        "status": "requires_region_artifacts",
        "result_inputs": "mismatch pairs plus CLIP/WordNet label distances",
        "note": "Separates near-misses from wild semantic errors.",
    },
    {
        "block": "E2",
        "metric": "hierarchical IoU",
        "proposal_group": "Group (ii); novelty-driving",
        "status": "requires_taxonomy",
        "result_inputs": "prediction/GT labels plus WordNet or dataset hierarchy",
        "note": "Gives partial credit for hypernym/hyponym predictions.",
    },
    {
        "block": "E2",
        "metric": "synonym-collapsed mIoU",
        "proposal_group": "Group (ii); novelty-driving",
        "status": "requires_taxonomy",
        "result_inputs": "prediction/GT labels plus synonym mapping",
        "note": "Tests whether large-vocab drop is true ambiguity or synonym competition.",
    },
    {
        "block": "E2",
        "metric": "prompt-sensitivity variance",
        "proposal_group": "Group (ii); novelty-driving",
        "status": "requires_controlled_runs",
        "result_inputs": "fixed prompt-template reruns per method",
        "note": "mIoU standard deviation across templates.",
    },
    {
        "block": "E2",
        "metric": "vocabulary-size scaling curve",
        "proposal_group": "Group (ii); novelty-driving",
        "status": "requires_controlled_runs",
        "result_inputs": "subsampled Context/ADE vocabulary reruns",
        "note": "Curve shape can become a finding: linear, saturating, or cliff-like.",
    },
    {
        "block": "E2",
        "metric": "region-level Top-1 naming accuracy",
        "proposal_group": "Group (iii); probe",
        "status": "requires_score_artifacts",
        "result_inputs": "top-k class scores for best-matched regions",
        "note": "Separates absent knowledge from wrong rank ordering.",
    },
    {
        "block": "E2",
        "metric": "region-level Top-3 naming accuracy",
        "proposal_group": "Group (iii); probe",
        "status": "requires_score_artifacts",
        "result_inputs": "top-k class scores for best-matched regions",
        "note": "Separates absent knowledge from wrong rank ordering.",
    },
    {
        "block": "E2",
        "metric": "region-level Top-5 naming accuracy",
        "proposal_group": "Group (iii); probe",
        "status": "requires_score_artifacts",
        "result_inputs": "top-k class scores for best-matched regions",
        "note": "Separates absent knowledge from wrong rank ordering.",
    },
    {
        "block": "E2",
        "metric": "GT-region naming Top-1",
        "proposal_group": "Group (iii); probe",
        "status": "requires_probe_outputs",
        "result_inputs": "GT-crop naming probe outputs",
        "note": "Naming ability with localization removed.",
    },
    {
        "block": "E2",
        "metric": "GT-region naming Top-5",
        "proposal_group": "Group (iii); probe",
        "status": "requires_probe_outputs",
        "result_inputs": "GT-crop naming probe outputs",
        "note": "Naming ability with localization removed.",
    },
    {
        "block": "E2",
        "metric": "GT-text localization IoU",
        "proposal_group": "Group (iii); probe",
        "status": "requires_probe_outputs",
        "result_inputs": "GT-class prompt localization outputs",
        "note": "Localization ability with naming removed.",
    },
    {
        "block": "E2",
        "metric": "GT-text localization BIoU",
        "proposal_group": "Group (iii); probe",
        "status": "requires_probe_outputs",
        "result_inputs": "GT-class prompt localization outputs",
        "note": "Boundary version of GT-text localization.",
    },
    {
        "block": "E2",
        "metric": "proposal-recall oracle BestIoU",
        "proposal_group": "Group (iii); probe",
        "status": "requires_region_artifacts",
        "result_inputs": "class-agnostic proposals plus GT regions",
        "note": "Upper bound for proposal-based pipelines.",
    },
    {
        "block": "E2",
        "metric": "MCC consistency before/after",
        "proposal_group": "Group (iii); probe",
        "status": "requires_controlled_runs",
        "result_inputs": "baseline and post-hoc mask-category re-ranking outputs",
        "note": "Compares mIoU and MCMR before/after MCC.",
    },
    {
        "block": "E2",
        "metric": "top mismatch pairs",
        "proposal_group": "Qualitative",
        "status": "requires_prediction_artifacts",
        "result_inputs": "confusion pairs from predicted and GT labels",
        "note": "Qualitative evidence for ambiguity failure modes.",
    },
    {
        "block": "E2",
        "metric": "confusion communities",
        "proposal_group": "Qualitative",
        "status": "requires_prediction_artifacts",
        "result_inputs": "class confusion graph",
        "note": "Finds clusters of systematic label confusion.",
    },
    {
        "block": "E2",
        "metric": "novel-vs-base class split",
        "proposal_group": "Qualitative",
        "status": "requires_taxonomy",
        "result_inputs": "per-class IoU plus base/novel mapping",
        "note": "Checks whether failures concentrate on novel classes.",
    },
    {
        "block": "E3",
        "metric": "Kendall tau rank correlation",
        "proposal_group": "Group (ii); novelty-driving",
        "status": "computed_when_enough_methods_complete",
        "result_inputs": "method rankings across targets",
        "note": "Hard evidence for cross-dataset rank instability.",
    },
    {
        "block": "E3",
        "metric": "Spearman rank correlation",
        "proposal_group": "Group (ii); novelty-driving",
        "status": "computed_when_enough_methods_complete",
        "result_inputs": "method rankings across targets",
        "note": "Hard evidence for cross-dataset rank instability.",
    },
    {
        "block": "E3",
        "metric": "cross-target rank stability",
        "proposal_group": "Group (ii); novelty-driving",
        "status": "computed_when_enough_methods_complete",
        "result_inputs": "method rankings across targets",
        "note": "One-minus-rank-variance style summaries can be added after the table is dense.",
    },
    {
        "block": "E3",
        "metric": "worst-target mIoU",
        "proposal_group": "Group (i)",
        "status": "computed_when_multiple_targets_complete",
        "result_inputs": "completed target mIoU values",
        "note": "Catches methods that average well but collapse on one target.",
    },
    {
        "block": "E3",
        "metric": "worst-class within target",
        "proposal_group": "Group (i)",
        "status": "computed_when_class_table_present",
        "result_inputs": "official mmseg per-class table",
        "note": "Lowest per-class IoU within each target.",
    },
    {
        "block": "E3",
        "metric": "source/target gap",
        "proposal_group": "Group (i)",
        "status": "requires_trained_reference_scores",
        "result_inputs": "trained-reference source and target mIoU",
        "note": "Contextualizes trained-method generalization claims.",
    },
    {
        "block": "E3",
        "metric": "per-target Delta vs E1",
        "proposal_group": "Group (ii)",
        "status": "requires_transfer_protocol",
        "result_inputs": "E1 target scores and E3 transfer target scores",
        "note": "Measures transfer drop relative to the matched E1 target.",
    },
    {
        "block": "E3",
        "metric": "per-target Delta MCMR",
        "proposal_group": "Group (ii)",
        "status": "requires_region_artifacts",
        "result_inputs": "MCMR@0.5 on E1/E2 and E3 targets",
        "note": "Separates localization transfer failure from naming transfer failure.",
    },
    {
        "block": "E3",
        "metric": "class-overlap-stratified mIoU",
        "proposal_group": "Group (ii); novelty-driving",
        "status": "requires_taxonomy",
        "result_inputs": "per-class IoU plus source-vocabulary overlap mapping",
        "note": "Splits target classes into source-overlapping vs source-novel.",
    },
    {
        "block": "E3",
        "metric": "domain-shift sensitivity",
        "proposal_group": "Group (ii); novelty-driving",
        "status": "requires_embedding_artifacts",
        "result_inputs": "target mIoU drops plus CLIP/DINO dataset embedding distances",
        "note": "Tests whether drops track source-to-target embedding distance.",
    },
    {
        "block": "E3",
        "metric": "transferability of MCMR@0.5",
        "proposal_group": "Group (ii)",
        "status": "requires_region_artifacts",
        "result_inputs": "MCMR@0.5 across E1/E2/E3 targets",
        "note": "Checks whether mask-category mismatch patterns transfer.",
    },
    {
        "block": "E4",
        "metric": "training time",
        "proposal_group": "standard cost reporting",
        "status": "not_applicable_for_training_free_or_requires_refs",
        "result_inputs": "trained-reference training logs",
        "note": "Training-free methods should be zero or N/A; trained refs need official training logs.",
    },
    {
        "block": "E4",
        "metric": "offline preprocessing cost",
        "proposal_group": "standard cost reporting",
        "status": "requires_method_stage_logs",
        "result_inputs": "offline stage timing logs",
        "note": "Needed for methods with prototypes, retrieval, or diffusion preprocessing.",
    },
    {
        "block": "E4",
        "metric": "inference time",
        "proposal_group": "standard cost reporting",
        "status": "computed_partial_from_official_log",
        "result_inputs": "official mmseg per-iteration time",
        "note": "Current value is useful for monitoring; isolated E4 reruns are still needed for paper numbers.",
    },
    {
        "block": "E4",
        "metric": "peak memory",
        "proposal_group": "standard cost reporting",
        "status": "computed_when_log_reports_memory",
        "result_inputs": "official mmseg memory field",
        "note": "Some logs do not print memory, so nvidia-smi sampling may be needed.",
    },
    {
        "block": "E4",
        "metric": "trainable parameters",
        "proposal_group": "standard cost reporting",
        "status": "requires_model_introspection",
        "result_inputs": "official model parameter count",
        "note": "Training-free rows should expose zero trainable params when true.",
    },
    {
        "block": "E4",
        "metric": "model calls",
        "proposal_group": "standard cost reporting",
        "status": "requires_method_stage_logs",
        "result_inputs": "instrumented method call counters",
        "note": "Counts CLIP, VFM, proposal, diffusion, and post-processing calls.",
    },
    {
        "block": "E4",
        "metric": "mIoU per GFLOP",
        "proposal_group": "Group (ii); novelty-supporting",
        "status": "requires_flops",
        "result_inputs": "mIoU plus FLOPs",
        "note": "For Pareto-frontier plots.",
    },
    {
        "block": "E4",
        "metric": "mIoU per second",
        "proposal_group": "Group (ii); novelty-supporting",
        "status": "computed_partial_from_official_log",
        "result_inputs": "mIoU plus official log time",
        "note": "Monitoring value; isolated E4 reruns are needed for final paper values.",
    },
    {
        "block": "E4",
        "metric": "offline amortization curve",
        "proposal_group": "Group (ii); novelty-supporting",
        "status": "requires_method_stage_logs",
        "result_inputs": "offline cost, online cost, evaluation set size N",
        "note": "Fair treatment of methods with high one-time preprocessing.",
    },
    {
        "block": "E4",
        "metric": "per-stage memory/latency breakdown",
        "proposal_group": "Group (i)",
        "status": "requires_method_stage_logs",
        "result_inputs": "encoder/proposal/naming/post-processing stage timings",
        "note": "Identifies bottleneck stage per method.",
    },
]

EXTRA_DISCOVERY_METRICS = [
    {
        "block": "extra",
        "metric": "class-IoU dispersion",
        "proposal_group": "extra discovery",
        "status": "computed_when_class_table_present",
        "result_inputs": "official mmseg per-class table",
        "note": "Std/quantile/Gini summary of per-class IoU; catches methods with the same mIoU but very different failure distribution.",
    },
    {
        "block": "extra",
        "metric": "nonzero-class coverage",
        "proposal_group": "extra discovery",
        "status": "computed_when_class_table_present",
        "result_inputs": "official mmseg per-class table",
        "note": "Fraction of classes with IoU > 0; a simple collapse detector for large vocabularies.",
    },
    {
        "block": "extra",
        "metric": "prediction label entropy",
        "proposal_group": "extra discovery",
        "status": "requires_prediction_artifacts",
        "result_inputs": "prediction label maps",
        "note": "Detects single-label dominance or overly diffuse predictions from saved label maps.",
    },
    {
        "block": "extra",
        "metric": "rare-class rescue score",
        "proposal_group": "extra discovery",
        "status": "requires_taxonomy",
        "result_inputs": "per-class IoU plus train-frequency bins",
        "note": "Tail-vs-head relative gain; useful if a method is not best on mIoU but uniquely helps rare classes.",
    },
    {
        "block": "extra",
        "metric": "near-tie robustness",
        "proposal_group": "extra discovery",
        "status": "requires_score_artifacts",
        "result_inputs": "top-2 class scores or logits",
        "note": "Fraction of pixels/regions whose top-1 and top-2 labels are close; flags ambiguous predictions that mIoU treats as hard errors.",
    },
    {
        "block": "extra",
        "metric": "Pareto dominance flag",
        "proposal_group": "extra discovery",
        "status": "computed_when_timing_available",
        "result_inputs": "mIoU, inference time, and memory",
        "note": "Identifies methods dominated by another method with higher mIoU and lower cost on the same dataset.",
    },
    {
        "block": "extra",
        "metric": "method-family gap",
        "proposal_group": "extra discovery",
        "status": "computed_when_families_complete",
        "result_inputs": "canonical mIoU grouped by method family",
        "note": "Compares CLIP-dense vs CLIP+VFM families to see whether gains come from method class rather than individual implementation.",
    },
    {
        "block": "extra",
        "metric": "artifact completeness",
        "proposal_group": "artifact readiness",
        "status": "computed",
        "result_inputs": "prediction artifact directory",
        "note": "Tracks whether each official run has enough saved prediction maps for artifact-based novelty metrics.",
    },
]


@dataclass
class RunRecord:
    method: str
    dataset: str
    variant: str
    log_path: str
    status: str
    iter: int | None = None
    total: int | None = None
    progress_pct: float | None = None
    aacc: float | None = None
    miou: float | None = None
    macc: float | None = None
    sec_per_iter: float | None = None
    peak_mem_mb: int | None = None
    worst_class_iou: float | None = None
    num_classes_logged: int | None = None
    zero_class_iou_count: int | None = None
    zero_class_iou_rate: float | None = None
    class_iou_std: float | None = None
    class_iou_p10: float | None = None
    class_iou_p50: float | None = None
    class_iou_p90: float | None = None
    class_iou_gini: float | None = None
    eta: str | None = None
    family: str | None = None


def _method_from_name(path: Path) -> str | None:
    name = path.name.lower()
    for method in METHOD_FAMILY:
        if f"_{method}_" in name or name.startswith(f"{method}_"):
            return method
    return None


def _dataset_from_text(path: Path, text: str) -> str | None:
    config = CONFIG_RE.search(text)
    if config:
        return config.group("dataset")
    lower = path.name.lower()
    for dataset in ("coco_stuff164k", "context459", "context59", "ade847", "ade20k", "voc20"):
        if dataset in lower:
            return dataset
    return None


def _variant_from_path(path: Path, work_dir: str | None) -> str:
    lower = path.name.lower()
    if "officialnames" in lower or (work_dir and "officialnames" in work_dir):
        return "officialnames"
    if "datafix" in lower or (work_dir and "datafix" in work_dir):
        return "datafix"
    if "officialscale" in lower or (work_dir and "officialscale" in work_dir):
        return "officialscale"
    if "officialmap" in lower or (work_dir and "officialmap" in work_dir):
        return "officialmap_highres"
    if lower.endswith("_e2.log") or (work_dir and "e2" in work_dir):
        return "phase2_e2"
    if lower.endswith("_e1.log"):
        return "phase1_e1"
    if "fixed" in lower or "fix" in lower:
        return "fix_or_debug"
    return "single"


def _class_iou_stats(class_ious: list[float]) -> dict[str, float | int | None]:
    zero_count = sum(1 for value in class_ious if value == 0.0)
    return {
        "worst_class_iou": min(class_ious) if class_ious else None,
        "num_classes_logged": len(class_ious) if class_ious else None,
        "zero_class_iou_count": zero_count if class_ious else None,
        "zero_class_iou_rate": (zero_count / len(class_ious)) if class_ious else None,
        "class_iou_std": _std(class_ious),
        "class_iou_p10": _quantile(class_ious, 0.10),
        "class_iou_p50": _quantile(class_ious, 0.50),
        "class_iou_p90": _quantile(class_ious, 0.90),
        "class_iou_gini": _gini(class_ious),
    }


def _records_from_freeda_log(path: Path) -> list[RunRecord]:
    text = path.read_text(encoding="utf-8", errors="replace")
    dataset_map = {
        "voc20": "voc20",
        "context59": "context59",
        "ade20k": "ade20k",
        "coco_stuff": "coco_stuff164k",
    }
    class_ious: list[float] = []
    summary: tuple[float, float, float] | None = None
    in_summary = False

    for line in text.splitlines():
        if "Summary:" in line:
            in_summary = True
            continue
        if in_summary:
            row = SUMMARY_ROW_RE.search(line)
            if row:
                summary = (float(row.group("aacc")), float(row.group("miou")), float(row.group("macc")))
                in_summary = False
                continue

        class_row = CLASS_ROW_RE.search(line)
        if class_row:
            value = class_row.group("iou")
            if value != "nan":
                try:
                    class_ious.append(float(value))
                except ValueError:
                    pass
            continue

        result = FREEDA_RESULT_RE.search(line)
        if result:
            dataset = dataset_map.get(result.group("dataset"), result.group("dataset"))
            total = int(result.group("total"))
            stats = _class_iou_stats(class_ious)
            aacc, _, macc = summary if summary else (None, None, None)
            return [
                RunRecord(
                    method="freeda",
                    dataset=dataset,
                    variant="official_repo",
                    log_path=str(path.relative_to(ROOT)),
                    status="complete",
                    iter=total,
                    total=total,
                    progress_pct=100.0,
                    aacc=aacc,
                    miou=float(result.group("miou")),
                    macc=macc,
                    family=METHOD_FAMILY.get("freeda"),
                    **stats,
                )
            ]
    return []


def _records_from_ovdiff_log(path: Path) -> list[RunRecord]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if "per class results" not in text or "mIoU" not in text:
        return []

    class_ious: list[float] = []
    printed_miou: float | None = None
    for line in text.splitlines():
        metric = OVDIFF_MIOU_RE.search(line.strip())
        if metric:
            printed_miou = float(metric.group("miou"))
            continue
        iou = OVDIFF_IOU_RE.search(line.strip())
        if not iou:
            continue
        value = float(iou.group("iou")) * 100.0
        if iou.group("class") != "__background__":
            class_ious.append(value)

    if not class_ious and printed_miou is None:
        return []

    # OVDiff prints VOC mIoU with background. Proposal tables use foreground VOC20.
    miou = sum(class_ious) / len(class_ious) if class_ious else printed_miou
    stats = _class_iou_stats(class_ious)
    return [
        RunRecord(
            method="ovdiff",
            dataset="voc20",
            variant="official_repo_foreground",
            log_path=str(path.relative_to(ROOT)),
            status="complete",
            iter=1449,
            total=1449,
            progress_pct=100.0,
            miou=miou,
            aacc=None,
            macc=None,
            family=METHOD_FAMILY.get("ovdiff"),
            **stats,
        )
    ]


def _records_from_log(path: Path) -> list[RunRecord]:
    if "runs/freeda" in str(path):
        return _records_from_freeda_log(path)
    if "ovdiff" in path.name.lower():
        return _records_from_ovdiff_log(path)

    text = path.read_text(encoding="utf-8", errors="replace")
    method = _method_from_name(path)
    if method is None:
        return []

    records: list[RunRecord] = []
    current_dataset = _dataset_from_text(path, text)
    work_dir_match = WORKDIR_RE.search(text)
    variant = _variant_from_path(path, work_dir_match.group("work") if work_dir_match else None)
    latest_progress: dict[str, tuple[int, int, str | None, float | None, int | None]] = {}
    current_class_ious: list[float] = []
    had_traceback = "Traceback (most recent call last)" in text
    interrupted = "KeyboardInterrupt" in text

    for line in text.splitlines():
        mark = DATASET_MARK_RE.search(line)
        if mark:
            current_dataset = mark.group("dataset")
            current_class_ious = []
            continue

        class_row = CLASS_ROW_RE.search(line)
        if class_row:
            value = class_row.group("iou")
            if value != "nan":
                try:
                    current_class_ious.append(float(value))
                except ValueError:
                    pass

        progress = PROGRESS_RE.search(line)
        if progress and current_dataset:
            iteration = int(progress.group("iter"))
            total = int(progress.group("total"))
            memory = int(progress.group("memory")) if progress.group("memory") else None
            sec = float(progress.group("time")) if progress.group("time") else None
            latest_progress[current_dataset] = (iteration, total, progress.group("eta"), sec, memory)

        metric = METRIC_RE.search(line)
        if metric and current_dataset:
            iteration = int(metric.group("iter"))
            total = int(metric.group("total"))
            memory = None
            latest = latest_progress.get(current_dataset)
            if latest:
                memory = latest[4]
            class_ious = current_class_ious[:]
            zero_count = sum(1 for value in class_ious if value == 0.0)
            records.append(
                RunRecord(
                    method=method,
                    dataset=current_dataset,
                    variant=variant,
                    log_path=str(path.relative_to(ROOT)),
                    status="complete" if iteration == total else "partial",
                    iter=iteration,
                    total=total,
                    progress_pct=100.0 * iteration / total if total else None,
                    aacc=float(metric.group("aacc")),
                    miou=float(metric.group("miou")),
                    macc=float(metric.group("macc")),
                    sec_per_iter=float(metric.group("time")),
                    peak_mem_mb=memory,
                    worst_class_iou=min(class_ious) if class_ious else None,
                    num_classes_logged=len(class_ious) if class_ious else None,
                    zero_class_iou_count=zero_count if class_ious else None,
                    zero_class_iou_rate=(zero_count / len(class_ious)) if class_ious else None,
                    class_iou_std=_std(class_ious),
                    class_iou_p10=_quantile(class_ious, 0.10),
                    class_iou_p50=_quantile(class_ious, 0.50),
                    class_iou_p90=_quantile(class_ious, 0.90),
                    class_iou_gini=_gini(class_ious),
                    family=METHOD_FAMILY.get(method),
                )
            )
            current_class_ious = []

    if records:
        return records

    if current_dataset:
        latest = latest_progress.get(current_dataset)
        if latest:
            iteration, total, eta, sec, memory = latest
            status = "interrupted" if interrupted else "failed" if had_traceback else "running_or_partial"
            return [
                RunRecord(
                    method=method,
                    dataset=current_dataset,
                    variant=variant,
                    log_path=str(path.relative_to(ROOT)),
                    status=status,
                    iter=iteration,
                    total=total,
                    progress_pct=100.0 * iteration / total if total else None,
                    sec_per_iter=sec,
                    peak_mem_mb=memory,
                    eta=eta,
                    family=METHOD_FAMILY.get(method),
                )
            ]

    return []


def _canonical_rank(records: list[RunRecord]) -> int:
    variant_rank = {
        "officialnames": 0,
        "datafix": 1,
        "officialscale": 2,
        "phase1_e1": 3,
        "phase2_e2": 4,
        "single": 5,
        "officialmap_highres": 6,
        "fix_or_debug": 7,
    }
    return variant_rank.get(records[0].variant, 9)


def _best_records(records: list[RunRecord]) -> list[RunRecord]:
    grouped: dict[tuple[str, str], list[RunRecord]] = {}
    for record in records:
        if record.status != "complete" or record.miou is None:
            continue
        grouped.setdefault((record.method, record.dataset), []).append(record)

    best: list[RunRecord] = []
    for rows in grouped.values():
        rows.sort(key=lambda r: (_canonical_rank([r]), -float(r.miou or -1)))
        best.append(rows[0])
    return sorted(best, key=lambda r: (r.method, r.dataset))


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


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.2f}"


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def _std(values: list[float]) -> float | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def _gini(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(max(value, 0.0) for value in values)
    total = sum(ordered)
    if total == 0:
        return 0.0
    n = len(ordered)
    weighted = sum((idx + 1) * value for idx, value in enumerate(ordered))
    return (2.0 * weighted) / (n * total) - (n + 1.0) / n


def _spearman(rows: list[tuple[str, float, float]]) -> float | None:
    if len(rows) < 3:
        return None
    left = sorted(rows, key=lambda x: x[1])
    right = sorted(rows, key=lambda x: x[2])
    rank_left = {method: rank for rank, (method, _, _) in enumerate(left, start=1)}
    rank_right = {method: rank for rank, (method, _, _) in enumerate(right, start=1)}
    n = len(rows)
    d2 = sum((rank_left[m] - rank_right[m]) ** 2 for m, _, _ in rows)
    return 1.0 - (6.0 * d2) / (n * (n * n - 1))


def _kendall_tau(rows: list[tuple[str, float, float]]) -> float | None:
    if len(rows) < 3:
        return None
    concordant = 0
    discordant = 0
    for i, (_, left_i, right_i) in enumerate(rows):
        for _, left_j, right_j in rows[i + 1 :]:
            left_delta = left_i - left_j
            right_delta = right_i - right_j
            if left_delta == 0 or right_delta == 0:
                continue
            if left_delta * right_delta > 0:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return None
    return (concordant - discordant) / total


def _proposal_manifest_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in PROPOSAL_NOVELTY_METRICS:
        row = dict(item)
        row["tier"] = "exploratory"
        rows.append(row)
    for item in EXTRA_DISCOVERY_METRICS:
        row = dict(item)
        row["tier"] = "extra_discovery"
        rows.append(row)
    rows.append(
        {
            "block": "E1/E2/E3",
            "metric": "mIoU",
            "proposal_group": "standard tier; ranking",
            "status": "computed_from_official_log",
            "result_inputs": "official evaluator summary",
            "note": "Primary comparable ranking metric; kept separate from novelty/exploratory readouts.",
            "tier": "standard",
        }
    )
    return rows


def _pending_metric_row(spec: dict[str, object]) -> dict[str, object]:
    return {
        "block": spec["block"],
        "metric": spec["metric"],
        "proposal_group": spec["proposal_group"],
        "status": spec["status"],
        "method": "all",
        "dataset": "",
        "value": "",
        "artifact_needed": spec["result_inputs"],
        "note": spec["note"],
    }


def _prediction_artifact_status(record: RunRecord) -> dict[str, object]:
    artifact_dir = PRED_ARTIFACT_DIR / record.method / record.dataset
    count = len(list(artifact_dir.glob("*.png"))) if artifact_dir.exists() else 0
    expected = DATASET_TOTALS.get(record.dataset)
    if expected and count >= expected:
        status = "ready"
    elif count:
        status = "partial"
    else:
        status = "missing"
    pct = (100.0 * count / expected) if expected else None
    return {
        "block": "extra",
        "metric": "artifact completeness",
        "proposal_group": "artifact readiness",
        "status": status,
        "method": record.method,
        "dataset": record.dataset,
        "value": pct,
        "artifact_count": count,
        "artifact_expected": expected,
        "artifact_needed": str(artifact_dir),
        "note": "Readiness check for proposal metrics requiring prediction label maps.",
    }


def _proposal_metric_rows(records: list[RunRecord]) -> tuple[list[dict[str, object]], list[str]]:
    complete = [r for r in records if r.status == "complete" and r.miou is not None]
    canonical = _best_records(complete)
    by_method_dataset = {(r.method, r.dataset): r for r in canonical}
    rows: list[dict[str, object]] = []
    computed_metric_names: set[str] = set()
    computed_extra_metric_names: set[str] = set()

    # E1 novelty/supporting diagnostics that can be read from official logs.
    for record in canonical:
        per_class_status = "computed_from_official_log" if record.worst_class_iou is not None else "pending_class_table"
        rows.extend(
            [
                {
                    "block": "E1",
                    "metric": "per-class IoU summary",
                    "proposal_group": "Group (i); cheap",
                    "status": per_class_status,
                    "method": record.method,
                    "dataset": record.dataset,
                    "value": record.worst_class_iou,
                    "worst_class_iou": record.worst_class_iou,
                    "num_classes_logged": record.num_classes_logged,
                    "zero_class_iou_count": record.zero_class_iou_count,
                    "zero_class_iou_rate": record.zero_class_iou_rate,
                    "artifact_needed": "" if record.worst_class_iou is not None else "official mmseg per-class table",
                    "note": "Proposal E1 per-class readout; value column is worst-class IoU.",
                },
                {
                    "block": "E1",
                    "metric": "pixel accuracy",
                    "proposal_group": "Group (i); cheap",
                    "status": "computed_from_official_log" if record.aacc is not None else "pending_log_summary",
                    "method": record.method,
                    "dataset": record.dataset,
                    "value": record.aacc,
                    "artifact_needed": "" if record.aacc is not None else "official mmseg aAcc",
                    "note": "Proposal E1 supporting readout parsed as aAcc.",
                },
                {
                    "block": "E1",
                    "metric": "mean class accuracy",
                    "proposal_group": "Group (i); cheap",
                    "status": "computed_from_official_log" if record.macc is not None else "pending_log_summary",
                    "method": record.method,
                    "dataset": record.dataset,
                    "value": record.macc,
                    "artifact_needed": "" if record.macc is not None else "official mmseg mAcc",
                    "note": "Proposal E1 supporting readout parsed as mAcc.",
                },
                {
                    "block": "E3",
                    "metric": "worst-class within target",
                    "proposal_group": "Group (i)",
                    "status": per_class_status,
                    "method": record.method,
                    "dataset": record.dataset,
                    "value": record.worst_class_iou,
                    "artifact_needed": "" if record.worst_class_iou is not None else "official mmseg per-class table",
                    "note": "Proposal E3 worst-class diagnostic within each completed target.",
                },
                {
                    "block": "E4",
                    "metric": "inference time",
                    "proposal_group": "standard cost reporting",
                    "status": "computed_partial_from_official_log" if record.sec_per_iter is not None else "pending_e4_log",
                    "method": record.method,
                    "dataset": record.dataset,
                    "value": record.sec_per_iter,
                    "artifact_needed": "" if record.sec_per_iter is not None else "official timing log",
                    "note": "Monitoring value from official log; final E4 still needs isolated reruns.",
                },
                {
                    "block": "E4",
                    "metric": "peak memory",
                    "proposal_group": "standard cost reporting",
                    "status": "computed_from_official_log" if record.peak_mem_mb is not None else "pending_memory_log",
                    "method": record.method,
                    "dataset": record.dataset,
                    "value": record.peak_mem_mb,
                    "artifact_needed": "" if record.peak_mem_mb is not None else "official memory field or nvidia-smi sampler",
                    "note": "Peak memory field when present in official logs.",
                },
                {
                    "block": "E4",
                    "metric": "mIoU per second",
                    "proposal_group": "Group (ii); novelty-supporting",
                    "status": "computed_partial_from_official_log" if record.miou is not None and record.sec_per_iter else "pending_e4_log",
                    "method": record.method,
                    "dataset": record.dataset,
                    "value": (record.miou / record.sec_per_iter) if record.miou is not None and record.sec_per_iter else None,
                    "artifact_needed": "" if record.miou is not None and record.sec_per_iter else "official timing log",
                    "note": "Proposal E4 normalized efficiency proxy; final paper value should use isolated E4.",
                },
                {
                    "block": "extra",
                    "metric": "zero-IoU class collapse rate",
                    "proposal_group": "extra discovery",
                    "status": per_class_status,
                    "method": record.method,
                    "dataset": record.dataset,
                    "value": record.zero_class_iou_rate,
                    "zero_class_iou_count": record.zero_class_iou_count,
                    "num_classes_logged": record.num_classes_logged,
                    "artifact_needed": "" if record.zero_class_iou_rate is not None else "official mmseg per-class table",
                    "note": "Extra metric: fraction of classes with exactly zero IoU; useful for spotting hidden collapse.",
                },
                {
                    "block": "extra",
                    "metric": "class-IoU dispersion",
                    "proposal_group": "extra discovery",
                    "status": per_class_status,
                    "method": record.method,
                    "dataset": record.dataset,
                    "value": record.class_iou_std,
                    "class_iou_p10": record.class_iou_p10,
                    "class_iou_p50": record.class_iou_p50,
                    "class_iou_p90": record.class_iou_p90,
                    "class_iou_gini": record.class_iou_gini,
                    "artifact_needed": "" if record.class_iou_std is not None else "official mmseg per-class table",
                    "note": "Extra metric: per-class IoU spread; value column is standard deviation.",
                },
                {
                    "block": "extra",
                    "metric": "nonzero-class coverage",
                    "proposal_group": "extra discovery",
                    "status": per_class_status,
                    "method": record.method,
                    "dataset": record.dataset,
                    "value": (1.0 - record.zero_class_iou_rate) if record.zero_class_iou_rate is not None else None,
                    "zero_class_iou_count": record.zero_class_iou_count,
                    "num_classes_logged": record.num_classes_logged,
                    "artifact_needed": "" if record.zero_class_iou_rate is not None else "official mmseg per-class table",
                    "note": "Extra metric: fraction of classes with nonzero IoU.",
                },
                _prediction_artifact_status(record),
            ]
        )
        computed_metric_names.update(
            {
                "per-class IoU summary",
                "pixel accuracy",
                "mean class accuracy",
                "worst-class within target",
                "inference time",
                "peak memory",
                "mIoU per second",
            }
        )
        computed_extra_metric_names.update(
            {
                "class-IoU dispersion",
                "nonzero-class coverage",
                "artifact completeness",
            }
        )

    # E2 delta_vocab, once compact/large pairs exist.
    for method in sorted(METHOD_FAMILY):
        deltas: list[float] = []
        for compact, large in E2_PAIRS:
            compact_record = by_method_dataset.get((method, compact))
            large_record = by_method_dataset.get((method, large))
            if compact_record and large_record and compact_record.miou is not None and large_record.miou is not None:
                delta = float(compact_record.miou) - float(large_record.miou)
                deltas.append(delta)
                rows.append(
                    {
                        "block": "E2",
                        "metric": "Delta_vocab_pair",
                        "proposal_group": "Group (ii); novelty-driving",
                        "status": "computed",
                        "method": method,
                        "dataset": f"{compact}->{large}",
                        "value": delta,
                        "artifact_needed": "",
                        "note": "mIoU drop from compact to large vocabulary.",
                    }
                )
        if deltas:
            computed_metric_names.add("Delta_vocab")
            rows.append(
                {
                    "block": "E2",
                    "metric": "Delta_vocab",
                    "proposal_group": "Group (ii); novelty-driving",
                    "status": "computed",
                    "method": method,
                    "dataset": "context+ade",
                    "value": sum(deltas) / len(deltas),
                    "artifact_needed": "",
                    "note": "Average compact-to-large vocabulary drop across available pairs.",
                }
            )

    # E3-style rank stability across completed E1 targets. This uses the same targets
    # until a separate E3 protocol is materialized.
    for i, left_ds in enumerate(E1_DATASETS):
        for right_ds in E1_DATASETS[i + 1 :]:
            paired: list[tuple[str, float, float]] = []
            for method in METHOD_FAMILY:
                left_record = by_method_dataset.get((method, left_ds))
                right_record = by_method_dataset.get((method, right_ds))
                if left_record and right_record and left_record.miou is not None and right_record.miou is not None:
                    paired.append((method, float(left_record.miou), float(right_record.miou)))
            rho = _spearman(paired)
            tau = _kendall_tau(paired)
            rows.append(
                {
                    "block": "E3",
                    "metric": "Spearman rank correlation",
                    "proposal_group": "Group (ii); novelty-driving",
                    "status": "computed" if rho is not None else "pending_more_methods",
                    "method": "all",
                    "dataset": f"{left_ds}<->{right_ds}",
                    "value": rho,
                    "n_methods": len(paired),
                    "artifact_needed": "",
                    "note": "Low rank correlation is a candidate generalization-fragility finding.",
                }
            )
            rows.append(
                {
                    "block": "E3",
                    "metric": "Kendall tau rank correlation",
                    "proposal_group": "Group (ii); novelty-driving",
                    "status": "computed" if tau is not None else "pending_more_methods",
                    "method": "all",
                    "dataset": f"{left_ds}<->{right_ds}",
                    "value": tau,
                    "n_methods": len(paired),
                    "artifact_needed": "",
                    "note": "Low rank correlation is a candidate generalization-fragility finding.",
                }
            )
            if rho is not None:
                computed_metric_names.add("Spearman rank correlation")
            if tau is not None:
                computed_metric_names.add("Kendall tau rank correlation")

    for method in sorted(METHOD_FAMILY):
        method_targets = [
            r
            for ds in E1_DATASETS
            if (r := by_method_dataset.get((method, ds))) is not None and r.miou is not None
        ]
        if method_targets:
            worst = min(method_targets, key=lambda r: float(r.miou or 0))
            rows.append(
                {
                    "block": "E3",
                    "metric": "worst-target mIoU",
                    "proposal_group": "Group (i)",
                    "status": "computed" if len(method_targets) >= 2 else "partial_one_target",
                    "method": method,
                    "dataset": worst.dataset,
                    "value": worst.miou,
                    "n_targets": len(method_targets),
                    "artifact_needed": "",
                    "note": "Proposal E3 target-collapse diagnostic.",
                }
            )
            if len(method_targets) >= 2:
                computed_metric_names.add("worst-target mIoU")

    # Protocol sensitivity is not in the core proposal list, but it is useful for
    # deciding whether official-scale vs shared-scale differences deserve an appendix.
    variants: dict[tuple[str, str], list[RunRecord]] = {}
    for record in complete:
        variants.setdefault((record.method, record.dataset), []).append(record)
    for (method, dataset), method_rows in variants.items():
        if len(method_rows) < 2:
            continue
        method_rows = sorted(method_rows, key=lambda r: float(r.miou or 0))
        delta = float(method_rows[-1].miou or 0) - float(method_rows[0].miou or 0)
        rows.append(
            {
                "block": "extra",
                "metric": "protocol / scale sensitivity",
                "proposal_group": "extra discovery",
                "status": "computed",
                "method": method,
                "dataset": dataset,
                "value": delta,
                "low_variant": method_rows[0].variant,
                "high_variant": method_rows[-1].variant,
                "artifact_needed": "",
                "note": "Extra diagnostic: flags whether resize/protocol changes dominate method comparisons.",
            }
        )

    for dataset in E1_DATASETS:
        dataset_rows = [
            r
            for r in canonical
            if r.dataset == dataset and r.miou is not None and r.sec_per_iter is not None
        ]
        if len(dataset_rows) < 2:
            continue
        for row in dataset_rows:
            dominators = [
                other.method
                for other in dataset_rows
                if other.method != row.method
                and float(other.miou or 0) >= float(row.miou or 0)
                and float(other.sec_per_iter or 0) <= float(row.sec_per_iter or 0)
                and (
                    float(other.miou or 0) > float(row.miou or 0)
                    or float(other.sec_per_iter or 0) < float(row.sec_per_iter or 0)
                )
            ]
            rows.append(
                {
                    "block": "extra",
                    "metric": "Pareto dominance flag",
                    "proposal_group": "extra discovery",
                    "status": "computed_latency_only",
                    "method": row.method,
                    "dataset": dataset,
                    "value": 1 if dominators else 0,
                    "dominators": ",".join(dominators),
                    "miou": row.miou,
                    "sec_per_image": row.sec_per_iter,
                    "artifact_needed": "",
                    "note": "1 means another completed method has at least as high mIoU and no slower official-log latency.",
                }
            )
        computed_extra_metric_names.add("Pareto dominance flag")

    for dataset in E1_DATASETS:
        by_family: dict[str, list[RunRecord]] = {}
        for row in canonical:
            if row.dataset == dataset and row.miou is not None and row.family:
                by_family.setdefault(row.family, []).append(row)
        if len(by_family) < 2:
            continue
        family_avgs = {family: sum(float(r.miou or 0) for r in rows_) / len(rows_) for family, rows_ in by_family.items()}
        best_family = max(family_avgs, key=family_avgs.get)
        worst_family = min(family_avgs, key=family_avgs.get)
        rows.append(
            {
                "block": "extra",
                "metric": "method-family gap",
                "proposal_group": "extra discovery",
                "status": "computed",
                "method": "family_average",
                "dataset": dataset,
                "value": family_avgs[best_family] - family_avgs[worst_family],
                "best_family": best_family,
                "worst_family": worst_family,
                "artifact_needed": "",
                "note": "Average mIoU gap between completed method families on the same dataset.",
            }
        )
        computed_extra_metric_names.add("method-family gap")

    # Explicitly materialize every proposal novelty metric that cannot be computed
    # from current official logs, so it stays visible in the nightly report.
    for spec in PROPOSAL_NOVELTY_METRICS:
        metric = str(spec["metric"])
        if metric in computed_metric_names:
            continue
        rows.append(_pending_metric_row(spec))
    for spec in EXTRA_DISCOVERY_METRICS:
        metric = str(spec["metric"])
        if metric in computed_extra_metric_names:
            continue
        rows.append(_pending_metric_row(spec))

    md_lines = [
        "",
        "## Proposal Novelty Metrics",
        "",
        "Rows below are the proposal-listed novelty/exploratory metrics plus a few extra diagnostics.",
        "`computed` rows are derived from current official outputs; `requires_*` rows identify the exact result artifact future official runs must save.",
        "",
        "| block | metric | group | status | method | dataset | value | artifact needed |",
        "|---|---|---|---|---|---|---:|---|",
    ]
    for row in rows:
        value = row.get("value")
        if isinstance(value, float):
            value_text = f"{value:.4f}"
        elif value is None:
            value_text = ""
        else:
            value_text = str(value)
        md_lines.append(
            f"| {row.get('block', '')} | {row.get('metric', '')} | {row.get('proposal_group', '')} | {row.get('status', '')} | "
            f"{row.get('method', '')} | {row.get('dataset', '')} | {value_text} | {row.get('artifact_needed', '')} |"
        )
    return rows, md_lines


def _make_signals(records: list[RunRecord]) -> tuple[list[dict[str, object]], str]:
    complete = [r for r in records if r.status == "complete" and r.miou is not None]
    canonical = _best_records(complete)
    signals: list[dict[str, object]] = []
    lines: list[str] = [
        "# Official Result Watch",
        "",
        "This file is auto-generated from `runs/logs/official*.log`.",
        "Use `officialscale` rows for COCO main-table candidates; `officialmap_highres` rows are diagnostics.",
        "",
        "## E1 Progress",
        "",
        "| method | VOC20 | Context59 | ADE150 | COCO171 | completed | avg mIoU |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    by_method_dataset = {(r.method, r.dataset): r for r in canonical}
    for method in sorted(METHOD_FAMILY):
        vals = [by_method_dataset.get((method, ds)) for ds in E1_DATASETS]
        miou_vals = [float(v.miou) for v in vals if v and v.miou is not None]
        completed = len(miou_vals)
        avg = sum(miou_vals) / completed if completed else None
        lines.append(
            "| "
            + " | ".join(
                [
                    method,
                    _fmt(vals[0].miou if vals[0] else None),
                    _fmt(vals[1].miou if vals[1] else None),
                    _fmt(vals[2].miou if vals[2] else None),
                    _fmt(vals[3].miou if vals[3] else None),
                    str(completed),
                    _fmt(avg),
                ]
            )
            + " |"
        )
        if completed >= 2:
            signals.append(
                {
                    "type": "e1_partial_average",
                    "method": method,
                    "datasets": completed,
                    "value": round(avg or 0.0, 4),
                    "note": "Partial E1 average; useful once at least two datasets are complete.",
                }
            )

    lines.extend(["", "## Candidate Signals", ""])

    # Dataset-specific winners among completed canonical rows.
    for dataset in E1_DATASETS:
        rows = [r for r in canonical if r.dataset == dataset and r.miou is not None]
        if len(rows) < 2:
            continue
        rows.sort(key=lambda r: float(r.miou or -1), reverse=True)
        top = rows[0]
        runner_up = rows[1]
        margin = float(top.miou or 0) - float(runner_up.miou or 0)
        if margin >= 1.0:
            signal = {
                "type": "dataset_leader",
                "dataset": dataset,
                "method": top.method,
                "value": top.miou,
                "margin": round(margin, 4),
                "note": f"{top.method} leads {dataset} by {margin:.2f} mIoU among completed canonical runs.",
            }
            signals.append(signal)

    # Scale sensitivity / protocol sensitivity.
    grouped_variants: dict[tuple[str, str], list[RunRecord]] = {}
    for row in complete:
        grouped_variants.setdefault((row.method, row.dataset), []).append(row)
    for (method, dataset), rows in grouped_variants.items():
        if len(rows) < 2:
            continue
        rows = sorted(rows, key=lambda r: float(r.miou or 0))
        delta = float(rows[-1].miou or 0) - float(rows[0].miou or 0)
        if delta >= 1.0:
            signals.append(
                {
                    "type": "protocol_sensitivity",
                    "method": method,
                    "dataset": dataset,
                    "value": round(delta, 4),
                    "low_variant": rows[0].variant,
                    "high_variant": rows[-1].variant,
                    "note": f"{method}/{dataset} changes by {delta:.2f} mIoU across logged protocols.",
                }
            )

    if signals:
        for signal in signals:
            lines.append(f"- **{signal['type']}**: {signal['note']}")
    else:
        lines.append("- No promoted candidate yet; waiting for more completed runs.")

    _, proposal_metric_md = _proposal_metric_rows(records)
    lines.extend(proposal_metric_md)

    lines.extend(["", "## Recent Running Or Failed Logs", ""])
    active = [r for r in records if r.status != "complete"]
    if active:
        for row in sorted(active, key=lambda r: r.log_path):
            pct = "" if row.progress_pct is None else f" {row.progress_pct:.1f}%"
            lines.append(f"- `{row.status}` {row.method}/{row.dataset}/{row.variant}{pct}: `{row.log_path}`")
    else:
        lines.append("- None.")

    return signals, "\n".join(lines) + "\n"


def analyze_once(log_dir: Path, out_dir: Path) -> dict[str, object]:
    records: list[RunRecord] = []
    for path in sorted(log_dir.glob("official*.log")):
        records.extend(_records_from_log(path))
    extra_paths = [
        ROOT / "runs" / "logs" / "ovdiff_voc_predict.log",
        *sorted((ROOT / "runs" / "freeda").glob("*/log.txt")),
    ]
    for path in extra_paths:
        if path.exists():
            records.extend(_records_from_log(path))

    rows = [asdict(r) for r in sorted(records, key=lambda r: (r.method, r.dataset, r.variant, r.log_path))]
    best = [asdict(r) for r in _best_records(records)]
    manifest_rows = _proposal_manifest_rows()
    proposal_metric_rows, _ = _proposal_metric_rows(records)
    signals, markdown = _make_signals(records)

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "official_log_metrics.csv", rows)
    _write_csv(out_dir / "official_best_metrics.csv", best)
    _write_csv(out_dir / "proposal_metric_manifest.csv", manifest_rows)
    _write_csv(out_dir / "proposal_novelty_metrics.csv", proposal_metric_rows)
    _write_csv(out_dir / "candidate_signals.csv", signals)
    (out_dir / "candidate_signals.md").write_text(markdown, encoding="utf-8")
    (out_dir / "candidate_signals.json").write_text(
        json.dumps(
            {
                "records": rows,
                "best": best,
                "proposal_metric_manifest": manifest_rows,
                "proposal_novelty_metrics": proposal_metric_rows,
                "signals": signals,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"records": len(rows), "best": len(best), "proposal_metrics": len(proposal_metric_rows), "signals": len(signals)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch official logs and surface candidate novelty signals.")
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval-sec", type=int, default=300)
    args = parser.parse_args()

    while True:
        summary = analyze_once(args.log_dir, args.out_dir)
        print(
            f"[analyze-official] records={summary['records']} best={summary['best']} "
            f"proposal_metrics={summary['proposal_metrics']} signals={summary['signals']} out={args.out_dir}",
            flush=True,
        )
        if not args.watch:
            break
        time.sleep(args.interval_sec)


if __name__ == "__main__":
    main()
