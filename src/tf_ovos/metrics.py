"""Two-tier metrics for TF-OVOS.

Standard tier  : mIoU — used for E1/E2/E3 ranking on semantic targets.
Exploratory tier: all other readouts — recorded in the same pass, never used
                  for leaderboard ranking, intended to surface failure modes.
Appendix tier  : binary-mask metrics (IoU, Sm, Fm, Em, MAE, BIoU) — for
                 hard-domain camouflage targets only.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageFilter


EPS = 1e-8

# ---------------------------------------------------------------------------
# Label-map helpers (standard + exploratory tiers)
# ---------------------------------------------------------------------------

def load_label_map(path: str, void_label: int = 255) -> np.ndarray:
    """Load a label-map PNG as an int32 array.  Supports 8-bit and 16-bit."""
    img = Image.open(path)
    if img.mode in ("I;16", "I"):
        arr = np.array(img, dtype=np.int32)
    else:
        arr = np.array(img.convert("L"), dtype=np.int32)
    return arr


def accumulate_confusion(
    pred: np.ndarray,
    gt: np.ndarray,
    num_classes: int,
    void_label: int = 255,
) -> np.ndarray:
    """Accumulate an N×N confusion matrix for one image pair."""
    pred = pred.flatten().astype(np.int64)
    gt = gt.flatten().astype(np.int64)
    valid = (gt >= 0) & (gt != void_label) & (gt < num_classes)
    pred = pred[valid]
    gt = gt[valid]
    # Clamp pred to valid range so out-of-vocab predictions don't crash.
    pred = np.clip(pred, 0, num_classes - 1)
    conf = np.bincount(num_classes * gt + pred, minlength=num_classes ** 2)
    return conf.reshape(num_classes, num_classes)


def miou_from_confusion(conf: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (per_class_iou, mean_iou) from an accumulated confusion matrix."""
    tp = np.diag(conf)
    fp = conf.sum(axis=0) - tp
    fn = conf.sum(axis=1) - tp
    denom = tp + fp + fn
    present = denom > 0
    per_class = np.where(present, tp / np.maximum(denom, 1), np.nan)
    miou = float(np.nanmean(per_class)) if present.any() else 0.0
    return per_class, miou


# ---------------------------------------------------------------------------
# Standard-tier evaluation (semantic tasks, E1/E2/E3)
# ---------------------------------------------------------------------------

@dataclass
class SemanticResult:
    """Holds results of semantic evaluation for one dataset run."""
    # Standard tier
    miou: float = 0.0
    per_class_iou: list[float] = field(default_factory=list)   # length = num_classes
    num_classes: int = 0
    num_images: int = 0

    # Exploratory tier – E1
    pixel_accuracy: float = 0.0
    mean_class_accuracy: float = 0.0
    per_class_present: list[bool] = field(default_factory=list)

    # Exploratory tier – E2 (MCMR)
    mcmr_05: float = 0.0     # primary: τ = 0.5
    mcmr_075: float = 0.0    # strict:  τ = 0.75

    # E2 vocabulary-robustness drop (filled by summarize_results, not eval)
    delta_vocab: float | None = None


def evaluate_semantic(
    gt_maps: list[np.ndarray],
    pred_maps: list[np.ndarray],
    num_classes: int,
    void_label: int = 255,
) -> SemanticResult:
    """Evaluate a list of (gt, pred) label-map pairs.

    All exploratory readouts are computed in the same pass as mIoU.
    """
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    mcmr_05_localized = 0
    mcmr_05_matched = 0
    mcmr_075_localized = 0
    mcmr_075_matched = 0
    total_correct_pixels = 0
    total_valid_pixels = 0

    for gt, pred in zip(gt_maps, pred_maps):
        # Resize pred to gt shape if needed.
        if pred.shape != gt.shape:
            img = Image.fromarray(pred.astype(np.uint16))
            img = img.resize((gt.shape[1], gt.shape[0]), resample=Image.Resampling.NEAREST)
            pred = np.array(img, dtype=np.int32)

        conf += accumulate_confusion(pred, gt, num_classes, void_label)

        # Pixel accuracy (exploratory)
        valid = (gt >= 0) & (gt != void_label) & (gt < num_classes)
        if valid.any():
            total_correct_pixels += int((pred[valid] == gt[valid]).sum())
            total_valid_pixels += int(valid.sum())

        # MCMR (exploratory)
        m05, m075 = _mcmr_one_image(pred, gt, num_classes, void_label)
        mcmr_05_localized += m05[0]
        mcmr_05_matched += m05[1]
        mcmr_075_localized += m075[0]
        mcmr_075_matched += m075[1]

    per_class_iou, miou = miou_from_confusion(conf)

    # Per-class accuracy (exploratory)
    tp_per_class = np.diag(conf)
    gt_per_class = conf.sum(axis=1)
    present = gt_per_class > 0
    per_class_acc = np.where(present, tp_per_class / np.maximum(gt_per_class, 1), np.nan)
    mean_class_acc = float(np.nanmean(per_class_acc)) if present.any() else 0.0

    pixel_acc = total_correct_pixels / max(total_valid_pixels, 1)

    mcmr_05 = (mcmr_05_localized - mcmr_05_matched) / max(mcmr_05_localized, 1)
    mcmr_075 = (mcmr_075_localized - mcmr_075_matched) / max(mcmr_075_localized, 1)

    return SemanticResult(
        miou=miou,
        per_class_iou=per_class_iou.tolist(),
        num_classes=num_classes,
        num_images=len(gt_maps),
        pixel_accuracy=float(pixel_acc),
        mean_class_accuracy=mean_class_acc,
        per_class_present=present.tolist(),
        mcmr_05=float(mcmr_05),
        mcmr_075=float(mcmr_075),
    )


def _mcmr_one_image(
    pred: np.ndarray,
    gt: np.ndarray,
    num_classes: int,
    void_label: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return (localized_05, matched_05), (localized_075, matched_075) for one image."""
    loc05 = loc075 = matched05 = matched075 = 0
    for c in range(num_classes):
        gt_mask = (gt == c)
        if not gt_mask.any():
            continue
        # Find predicted class with most overlap on GT region.
        pred_on_gt = pred[gt_mask & (gt != void_label)]
        if pred_on_gt.size == 0:
            continue
        pred_on_gt = np.clip(pred_on_gt, 0, num_classes - 1)
        counts = np.bincount(pred_on_gt, minlength=num_classes)
        best_p = int(np.argmax(counts))
        pred_mask = (pred == best_p)
        inter = int((pred_mask & gt_mask).sum())
        union = int((pred_mask | gt_mask).sum())
        best_iou = inter / union if union > 0 else 0.0
        if best_iou >= 0.5:
            loc05 += 1
            if best_p == c:
                matched05 += 1
        if best_iou >= 0.75:
            loc075 += 1
            if best_p == c:
                matched075 += 1
    return (loc05, matched05), (loc075, matched075)


def delta_vocab(miou_compact: float, miou_large: float) -> float:
    """Vocabulary-robustness drop: positive means degradation."""
    return miou_compact - miou_large


# ---------------------------------------------------------------------------
# Binary-mask helpers (appendix tier: mask-only and class-aware tasks)
# ---------------------------------------------------------------------------

def load_binary_mask(path: str, threshold: float = 0.5) -> np.ndarray:
    image = Image.open(path).convert("L")
    arr = np.asarray(image, dtype=np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr >= threshold


def resize_mask_to(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    image = image.resize((shape[1], shape[0]), resample=Image.Resampling.NEAREST)
    return np.asarray(image) > 0


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, gt).sum() / union)


def mae(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.abs(pred.astype(np.float32) - gt.astype(np.float32)).mean())


def f_beta(pred: np.ndarray, gt: np.ndarray, beta2: float = 0.3) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    precision = tp / max(pred.sum(), EPS)
    recall = tp / max(gt.sum(), EPS)
    return float((1 + beta2) * precision * recall / max(beta2 * precision + recall, EPS))


def e_measure(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_f = pred.astype(np.float32)
    gt_f = gt.astype(np.float32)
    pred_centered = pred_f - pred_f.mean()
    gt_centered = gt_f - gt_f.mean()
    align = 2 * pred_centered * gt_centered / (pred_centered**2 + gt_centered**2 + EPS)
    enhanced = ((align + 1) ** 2) / 4
    return float(enhanced.mean())


def boundary(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    dilated = np.asarray(image.filter(ImageFilter.MaxFilter(radius * 2 + 1))) > 0
    eroded = np.asarray(image.filter(ImageFilter.MinFilter(radius * 2 + 1))) > 0
    return np.logical_xor(dilated, eroded)


def boundary_iou(pred: np.ndarray, gt: np.ndarray, radius: int = 2) -> float:
    return iou(boundary(pred, radius), boundary(gt, radius))


@dataclass(frozen=True)
class MaskMetrics:
    iou: float
    f_beta: float
    e_measure: float
    mae: float
    boundary_iou: float


def compute_mask_metrics(pred: np.ndarray, gt: np.ndarray) -> MaskMetrics:
    pred = resize_mask_to(pred, gt.shape)
    return MaskMetrics(
        iou=iou(pred, gt),
        f_beta=f_beta(pred, gt),
        e_measure=e_measure(pred, gt),
        mae=mae(pred, gt),
        boundary_iou=boundary_iou(pred, gt),
    )


def class_aware(metrics: MaskMetrics, pred_label: str | None, gt_label: str | None) -> dict[str, float]:
    exact = pred_label == gt_label and gt_label is not None
    gate = 1.0 if exact else 0.0
    return {
        "cIoU": metrics.iou * gate,
        "cF_beta": metrics.f_beta * gate,
        "cE_m": metrics.e_measure * gate,
        "cBIoU": metrics.boundary_iou * gate,
        "cMAE": metrics.mae if exact else 1.0,
        "Exact": gate,
    }


def ambiguity_rows(
    rows: list[tuple[float, str | None, str | None]],
    loc_threshold: float = 0.5,
) -> dict[str, object]:
    """Compute MCMR-style ambiguity stats for binary-mask / class-aware targets."""
    localized = [(gt, pred) for loc_iou, gt, pred in rows if loc_iou >= loc_threshold]
    exact = [(gt, pred) for gt, pred in localized if gt == pred and gt is not None]
    confusions = Counter(
        (gt, pred)
        for gt, pred in localized
        if gt != pred and gt is not None and pred is not None
    )
    loc_at = len(localized) / max(len(rows), 1)
    exact_at = len(exact) / max(len(rows), 1)
    return {
        "Loc@0.5": loc_at,
        "Exact@0.5": exact_at,
        "ClsErr@Loc": 1.0 - exact_at / max(loc_at, EPS),
        "num_edges": len(confusions),
        "top_confusions": [
            {"gt": gt, "pred": pred, "count": count}
            for (gt, pred), count in confusions.most_common(20)
        ],
    }
