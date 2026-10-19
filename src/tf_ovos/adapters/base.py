from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

from tf_ovos.data import Prediction, Sample


class MethodAdapter(ABC):
    name: str
    runnable: bool = True
    setup_hint: str = ""

    def configure(self, **kwargs: object) -> None:
        """Receive optional runner context such as task or num_classes."""

    def resolved_config(self) -> dict[str, object]:
        """Export actual adapter settings, including environment overrides."""
        names = ("task", "num_classes", "sam_checkpoint", "clip_model", "clip_pretrained",
                 "siglip_model", "siglip_pretrained", "prompt_template", "device_name",
                 "points_per_side", "pred_iou_thresh", "stability_score_thresh",
                 "min_mask_region_area", "cache_dir", "dinov2_model_id", "dinov2_weight")
        result = {name: getattr(self, name) for name in names if hasattr(self, name)}
        result["adapter"] = type(self).__name__
        for name in ("clip_preprocess", "siglip_preprocess", "dinov2_processor"):
            if hasattr(self, name):
                result[name] = repr(getattr(self, name))
        if hasattr(self, "sam_checkpoint"):
            result["sam_backbone"] = "vit_h"
            result["uncovered_pixel_policy"] = "candidate_0_legacy_compositing"
        return result

    @abstractmethod
    def predict_one(self, sample: Sample, vocabulary: list[str], output_dir: Path) -> Prediction:
        """Return exactly one prediction for one image.

        For semantic tasks the prediction mask_path should point to a label-map
        PNG (H×W, pixel = class index).  For mask-only / class-aware tasks it
        should point to a binary mask PNG.
        """

    def predict_many(self, samples: Iterable[Sample], vocabulary: list[str], output_dir: Path) -> list[Prediction]:
        output_dir.mkdir(parents=True, exist_ok=True)
        return [self.predict_one(sample, vocabulary, output_dir) for sample in samples]


def write_predictions(predictions: Iterable[Prediction], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for pred in predictions:
            row = {
                "image_id": pred.image_id,
                "mask_path": str(pred.mask_path),
                "label": pred.label,
            }
            if pred.score is not None:
                row["score"] = pred.score
            if pred.metadata is not None:
                row["metadata"] = pred.metadata
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
