import numpy as np
import pytest

from tf_ovos.metrics import (
    ambiguity_rows,
    class_aware,
    compute_mask_metrics,
    delta_vocab,
    evaluate_semantic,
    miou_from_confusion,
    accumulate_confusion,
)


# ── Binary-mask metrics (appendix tier) ────────────────────────────────────

def test_perfect_mask_metrics():
    mask = np.array([[1, 0], [0, 1]], dtype=bool)
    metrics = compute_mask_metrics(mask, mask)
    assert metrics.iou == 1.0
    assert metrics.mae == 0.0


def test_class_gate_zeroes_overlap_scores_on_wrong_label():
    mask = np.array([[1, 0], [0, 1]], dtype=bool)
    metrics = compute_mask_metrics(mask, mask)
    gated = class_aware(metrics, "frog", "fish")
    assert gated["cIoU"] == 0.0
    assert gated["cMAE"] == 1.0


def test_ambiguity_counts_localized_wrong_label():
    result = ambiguity_rows([(0.8, "frog", "toad"), (0.4, "fish", "fish")])
    assert result["Loc@0.5"] == 0.5
    assert result["Exact@0.5"] == 0.0
    assert result["num_edges"] == 1


# ── Standard-tier: semantic mIoU ───────────────────────────────────────────

def test_perfect_semantic_miou():
    # 2×2 label map, 3 classes, all correct
    gt = np.array([[0, 1], [2, 0]], dtype=np.int32)
    pred = np.array([[0, 1], [2, 0]], dtype=np.int32)
    result = evaluate_semantic([gt], [pred], num_classes=3)
    assert result.miou == pytest.approx(1.0)


def test_all_wrong_semantic_miou():
    # Everything predicted as class 0 when GT has class 1 everywhere
    gt = np.ones((2, 2), dtype=np.int32)
    pred = np.zeros((2, 2), dtype=np.int32)
    result = evaluate_semantic([gt], [pred], num_classes=2)
    # class 0: FP=4, FN=0, TP=0 → IoU=0; class 1: FP=0, FN=4, TP=0 → IoU=0
    assert result.miou == pytest.approx(0.0)


def test_void_label_ignored():
    gt = np.array([[0, 255], [1, 255]], dtype=np.int32)
    pred = np.array([[0, 0], [1, 0]], dtype=np.int32)
    result = evaluate_semantic([gt], [pred], num_classes=2, void_label=255)
    # Only pixels (0,0)→class0 and (1,0)→class1 are valid
    assert result.miou == pytest.approx(1.0)


def test_confusion_matrix_accumulation():
    gt = np.array([[0, 1]], dtype=np.int32)
    pred = np.array([[0, 0]], dtype=np.int32)  # class 1 predicted as 0
    conf = accumulate_confusion(pred, gt, num_classes=2)
    # row=GT, col=pred: GT=0,pred=0 → conf[0,0]=1; GT=1,pred=0 → conf[1,0]=1
    assert conf[0, 0] == 1
    assert conf[1, 0] == 1
    per_class, miou = miou_from_confusion(conf)
    # class 0: TP=1, FP=1, FN=0 → IoU=0.5; class 1: TP=0, FP=0, FN=1 → IoU=0
    assert per_class[0] == pytest.approx(0.5)
    assert per_class[1] == pytest.approx(0.0)
    assert miou == pytest.approx(0.25)


# ── Exploratory tier: delta_vocab ──────────────────────────────────────────

def test_delta_vocab():
    assert delta_vocab(0.40, 0.25) == pytest.approx(0.15)
    assert delta_vocab(0.30, 0.30) == pytest.approx(0.0)


# ── Exploratory tier: MCMR ─────────────────────────────────────────────────

def test_mcmr_perfect():
    # All predictions correct → MCMR should be 0
    gt = np.array([[0, 1], [1, 0]], dtype=np.int32)
    pred = np.array([[0, 1], [1, 0]], dtype=np.int32)
    result = evaluate_semantic([gt], [pred], num_classes=2)
    assert result.mcmr_05 == pytest.approx(0.0)


def test_mcmr_all_wrong_class():
    # Swap classes: GT=0 pred=1, GT=1 pred=0 — well localised but wrong class
    gt = np.array([[0, 0], [1, 1]], dtype=np.int32)
    pred = np.array([[1, 1], [0, 0]], dtype=np.int32)
    result = evaluate_semantic([gt], [pred], num_classes=2)
    # Both classes are localized (IoU≥0.5) but wrong → MCMR = 1.0
    assert result.mcmr_05 == pytest.approx(1.0)
