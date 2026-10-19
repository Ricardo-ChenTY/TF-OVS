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
    """Preserve integer indices, including palette PNGs and 16-bit labels."""
    with Image.open(path) as img:
        if img.mode not in ("L", "P", "I", "I;16", "I;16L", "I;16B"):
            raise ValueError(f"Expected an indexed label map, got {img.mode}: {path}")
        return np.array(img, dtype=np.int32)


def resize_label_map(pred: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if pred.ndim != 2 or not np.issubdtype(pred.dtype, np.integer):
        raise ValueError("Label maps must be two-dimensional integer arrays")
    if pred.shape == shape:
        return pred
    # Signed I mode preserves explicit negative rejection IDs during resizing.
    image = Image.fromarray(pred.astype(np.int32))
    return np.array(image.resize((shape[1], shape[0]), Image.Resampling.NEAREST), dtype=np.int32)


def accumulate_confusion(
    pred: np.ndarray,
    gt: np.ndarray,
    num_classes: int,
    void_label: int = 255,
    *,
    num_candidates: int | None = None,
    rejection_label: int | None = None,
) -> np.ndarray:
    """GT rows × candidate columns, plus an optional rejection column.

    Evaluation classes occupy the candidate prefix [0, num_classes). Extra
    candidates and declared rejection pixels remain in GT row totals (FN).
    Undeclared prediction IDs raise; they are never clipped or dropped.
    """
    k = num_classes if num_candidates is None else num_candidates
    if num_classes <= 0 or k < num_classes:
        raise ValueError("Require num_candidates >= num_classes > 0")
    if rejection_label is not None and 0 <= rejection_label < k:
        raise ValueError("rejection_label must be outside the candidate range")
    pred, gt = np.asarray(pred), np.asarray(gt)
    if pred.shape != gt.shape or pred.ndim != 2:
        raise ValueError("Prediction and GT must be equally shaped 2-D label maps")
    if not all(np.issubdtype(a.dtype, np.integer) for a in (pred, gt)):
        raise ValueError("Label maps must contain integer IDs")
    allowed = (pred >= 0) & (pred < k)
    if rejection_label is not None:
        allowed |= pred == rejection_label
    if not allowed.all():
        raise ValueError(f"Unexpected prediction IDs: {np.unique(pred[~allowed])[:10].tolist()}")
    valid = (gt >= 0) & (gt != void_label) & (gt < num_classes)
    p, g = pred[valid].astype(np.int64), gt[valid].astype(np.int64)
    columns = k + int(rejection_label is not None)
    if rejection_label is not None:
        p = np.where(p == rejection_label, k, p)
    return np.bincount(columns * g + p, minlength=num_classes * columns).reshape(num_classes, columns)


def miou_from_confusion(conf: np.ndarray) -> tuple[np.ndarray, float]:
    """Union-supported mIoU; extra columns contribute FN but no extra classes."""
    n = conf.shape[0]
    if conf.ndim != 2 or conf.shape[1] < n:
        raise ValueError("Confusion matrix must have at least as many columns as GT rows")
    tp = np.diag(conf[:, :n])
    denom = conf.sum(axis=1) + conf[:, :n].sum(axis=0) - tp
    present = denom > 0
    per_class = np.where(present, tp / np.maximum(denom, 1), np.nan)
    return per_class, float(np.nanmean(per_class)) if present.any() else 0.0


@dataclass
class DenseRegionStatistics:
    """One region per image–GT-class, matched by maximum IoU on valid GT pixels.

    Each candidate has one whole-image mask; this is not instance/proposal
    matching. Ties choose the lowest candidate ID. Rejection is not a mask.
    """
    confusion: np.ndarray
    gt_counts: np.ndarray
    pred_counts: np.ndarray
    gt_ids: np.ndarray
    best_ids: np.ndarray
    best_ious: np.ndarray

    def counts(self, threshold: float) -> tuple[int, int]:
        localized = self.best_ious >= threshold
        return int(localized.sum()), int((localized & (self.best_ids == self.gt_ids)).sum())


def dense_region_statistics(
    pred: np.ndarray, gt: np.ndarray, num_classes: int, void_label: int = 255,
    *, num_candidates: int | None = None, rejection_label: int | None = None,
) -> DenseRegionStatistics:
    k = num_classes if num_candidates is None else num_candidates
    conf = accumulate_confusion(pred, gt, num_classes, void_label,
                                num_candidates=k, rejection_label=rejection_label)
    gt_counts = conf.sum(axis=1)
    pred_counts = conf[:, :k].sum(axis=0)
    gt_ids = np.flatnonzero(gt_counts)
    inter = conf[gt_ids, :k].astype(np.float64)
    union = gt_counts[gt_ids, None] + pred_counts[None, :] - inter
    ious = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    best_ids = np.argmax(ious, axis=1)
    best_ious = ious[np.arange(len(gt_ids)), best_ids]
    best_ids = np.where(best_ious > 0, best_ids, -1)
    return DenseRegionStatistics(conf, gt_counts, pred_counts, gt_ids, best_ids, best_ious)


@dataclass
class SemanticResult:
    miou: float = 0.0
    per_class_iou: list[float] = field(default_factory=list)
    num_classes: int = 0
    num_candidates: int = 0
    num_images: int = 0
    pixel_accuracy: float = 0.0
    mean_class_accuracy: float = 0.0
    per_class_present: list[bool] = field(default_factory=list)
    mcmr_05: float | None = None
    mcmr_075: float | None = None
    gt_class_regions: int = 0
    localized_pairs_at_05: int = 0
    mismatch_pairs_at_05: int = 0
    localized_pairs_at_075: int = 0
    mismatch_pairs_at_075: int = 0
    coverage_at_05: float | None = None
    coverage_at_075: float | None = None
    ziou_gt_supported: float | None = None
    ziou_union_supported: float | None = None
    never_predicted_class_rate: float | None = None
    predicted_but_zero_iou_class_rate: float | None = None
    metric_version: str = "dense_class_region_max_iou_v2"
    miou_denominator: str = "union_supported_eval_classes"
    delta_vocab: float | None = None


def evaluate_semantic(
    gt_maps: list[np.ndarray], pred_maps: list[np.ndarray], num_classes: int,
    void_label: int = 255, *, num_candidates: int | None = None,
    rejection_label: int | None = None,
) -> SemanticResult:
    if len(gt_maps) != len(pred_maps):
        raise ValueError("GT and prediction counts differ")
    k = num_classes if num_candidates is None else num_candidates
    if num_classes <= 0 or k < num_classes:
        raise ValueError("Require num_candidates >= num_classes > 0")
    conf = np.zeros((num_classes, k + int(rejection_label is not None)), dtype=np.int64)
    loc05 = match05 = loc075 = match075 = regions = 0
    for gt, pred in zip(gt_maps, pred_maps):
        pred = resize_label_map(pred, gt.shape)
        stats = dense_region_statistics(pred, gt, num_classes, void_label,
                                       num_candidates=k, rejection_label=rejection_label)
        conf += stats.confusion
        a, b = stats.counts(0.5)
        c, d = stats.counts(0.75)
        loc05 += a
        match05 += b
        loc075 += c
        match075 += d
        regions += len(stats.gt_ids)
    per_class_iou, miou = miou_from_confusion(conf)
    tp = np.diag(conf[:, :num_classes])
    gt_counts = conf.sum(axis=1)
    pred_counts = conf[:, :num_classes].sum(axis=0)
    present = gt_counts > 0
    union_present = gt_counts + pred_counts - tp > 0
    def rate(mask: np.ndarray, support: np.ndarray) -> float | None:
        return float((mask & support).sum() / support.sum()) if support.any() else None
    return SemanticResult(
        miou=miou, per_class_iou=per_class_iou.tolist(), num_classes=num_classes,
        num_candidates=k, num_images=len(gt_maps),
        pixel_accuracy=float(tp.sum() / max(gt_counts.sum(), 1)),
        mean_class_accuracy=float((tp[present] / gt_counts[present]).mean()) if present.any() else 0.0,
        per_class_present=present.tolist(), gt_class_regions=regions,
        localized_pairs_at_05=loc05, mismatch_pairs_at_05=loc05-match05,
        localized_pairs_at_075=loc075, mismatch_pairs_at_075=loc075-match075,
        mcmr_05=(loc05-match05)/loc05 if loc05 else None,
        mcmr_075=(loc075-match075)/loc075 if loc075 else None,
        coverage_at_05=loc05/regions if regions else None,
        coverage_at_075=loc075/regions if regions else None,
        ziou_gt_supported=rate(tp == 0, present),
        ziou_union_supported=rate(tp == 0, union_present),
        never_predicted_class_rate=rate(pred_counts == 0, present),
        predicted_but_zero_iou_class_rate=rate((pred_counts > 0) & (tp == 0), present),
    )


def _mcmr_one_image(
    pred: np.ndarray, gt: np.ndarray, num_classes: int, void_label: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    stats = dense_region_statistics(pred, gt, num_classes, void_label)
    return stats.counts(0.5), stats.counts(0.75)


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
