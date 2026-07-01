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
