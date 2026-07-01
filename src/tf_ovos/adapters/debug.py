from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image

from tf_ovos.adapters.base import MethodAdapter
from tf_ovos.data import Prediction, Sample, as_output_path


class CopyGroundTruthAdapter(MethodAdapter):
    """Copies the GT mask/label-map as the prediction.  Perfect score on any task."""
    name = "debug_copy_gt"
    runnable = True

    def predict_one(self, sample: Sample, vocabulary: list[str], output_dir: Path) -> Prediction:
        pred_dir = output_dir / "pred_masks"
        pred_dir.mkdir(parents=True, exist_ok=True)
        out_mask = pred_dir / f"{sample.image_id}.png"
        shutil.copyfile(sample.mask_path, out_mask)
        return Prediction(
            image_id=sample.image_id,
            mask_path=Path(as_output_path(out_mask, output_dir)),
            label=sample.label,
            score=1.0,
            metadata={"debug": "copied ground-truth mask; not a benchmark method"},
        )


class EmptyMaskAdapter(MethodAdapter):
    """Outputs an all-zero mask/label-map.  Zero score on any task."""
    name = "debug_empty"
    runnable = True

    def predict_one(self, sample: Sample, vocabulary: list[str], output_dir: Path) -> Prediction:
        pred_dir = output_dir / "pred_masks"
        pred_dir.mkdir(parents=True, exist_ok=True)
        out_mask = pred_dir / f"{sample.image_id}.png"
        with Image.open(sample.mask_path) as gt:
            Image.new("L", gt.size, 0).save(out_mask)
        label = vocabulary[0] if vocabulary else sample.label
        return Prediction(
            image_id=sample.image_id,
            mask_path=Path(as_output_path(out_mask, output_dir)),
            label=label,
            score=0.0,
            metadata={"debug": "empty mask; not a benchmark method"},
        )
