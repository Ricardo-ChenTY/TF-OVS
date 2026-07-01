from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from tf_ovos.data import iter_missing_predictions, load_manifest, load_predictions
from tf_ovos.metrics import (
    SemanticResult,
    accumulate_confusion,
    ambiguity_rows,
    class_aware,
    compute_mask_metrics,
    evaluate_semantic,
    load_binary_mask,
    load_label_map,
    miou_from_confusion,
)


def mean_dict(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def evaluate(
    manifest: Path,
    predictions_path: Path,
    task: str,
    threshold: float,
    num_classes: int | None = None,
    void_label: int = 255,
    prediction_label_offset: int = 0,
) -> dict[str, object]:
    samples = load_manifest(manifest)
    predictions = load_predictions(predictions_path)
    missing = iter_missing_predictions(samples, predictions)
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"Missing {len(missing)} predictions. First missing ids: {preview}")

    if task == "semantic":
        if num_classes is None:
            raise ValueError("num_classes is required for task='semantic'")
        gt_maps = [load_label_map(str(s.mask_path), void_label) for s in samples]
        pred_maps = [load_label_map(str(predictions[s.image_id].mask_path), void_label) for s in samples]
        if prediction_label_offset:
            pred_maps = [
                np.clip(pred.astype(np.int32) + prediction_label_offset, 0, num_classes - 1)
                for pred in pred_maps
            ]
        result_obj: SemanticResult = evaluate_semantic(gt_maps, pred_maps, num_classes, void_label)
        result: dict[str, object] = {
            "num_samples": len(samples),
            "task": task,
            "num_classes": num_classes,
            # Standard tier
            "miou": result_obj.miou,
            # Exploratory tier
            "pixel_accuracy": result_obj.pixel_accuracy,
            "mean_class_accuracy": result_obj.mean_class_accuracy,
            "mcmr_05": result_obj.mcmr_05,
            "mcmr_075": result_obj.mcmr_075,
            "per_class_iou": result_obj.per_class_iou,
            "prediction_label_offset": prediction_label_offset,
        }
        return result

    # mask-only and class-aware (appendix hard-domain targets)
    mask_rows: list[dict[str, float]] = []
    class_rows: list[dict[str, float]] = []
    ambiguity_input: list[tuple[float, str | None, str | None]] = []

    for sample in samples:
        pred = predictions[sample.image_id]
        gt_mask = load_binary_mask(str(sample.mask_path), threshold=threshold)
        pred_mask = load_binary_mask(str(pred.mask_path), threshold=threshold)
        metrics = compute_mask_metrics(pred_mask, gt_mask)
        mask_rows.append(
            {
                "IoU": metrics.iou,
                "F_beta": metrics.f_beta,
                "E_m": metrics.e_measure,
                "MAE": metrics.mae,
                "BIoU": metrics.boundary_iou,
            }
        )
        ambiguity_input.append((metrics.iou, sample.label, pred.label))
        if task == "class-aware":
            class_rows.append(class_aware(metrics, pred.label, sample.label))

    result = {
        "num_samples": len(samples),
        "task": task,
        "mask_metrics": mean_dict(mask_rows),
    }
    if task == "class-aware":
        result["class_aware_metrics"] = mean_dict(class_rows)
        result["ambiguity"] = ambiguity_rows(ambiguity_input)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TF-OVOS predictions.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--task", choices=["semantic", "mask-only", "class-aware"], required=True)
    parser.add_argument("--num-classes", type=int, help="Required for --task semantic")
    parser.add_argument("--void-label", type=int, default=255)
    parser.add_argument("--prediction-label-offset", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5, help="Binary mask threshold (mask-only/class-aware)")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    result = evaluate(
        args.manifest,
        args.predictions,
        args.task,
        args.threshold,
        num_classes=args.num_classes,
        void_label=args.void_label,
        prediction_label_offset=args.prediction_label_offset,
    )
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
