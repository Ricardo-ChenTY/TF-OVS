from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from tf_ovos.adapters.sam_amg_clip import SamAmgClipAdapter
from tf_ovos.adapters.sam_amg_siglip import SamAmgSiglipAdapter
from tf_ovos.data import Prediction, Sample, as_output_path

_SAFE_CHARS = str.maketrans("/\\", "__")


def _mask_name(image_id: str) -> str:
    return image_id.translate(_SAFE_CHARS)


class _Dinov2GuidanceMixin:
    """Adds a DINOv2 objectness prior to the existing SAM proposal ranking."""

    dinov2_model_name = "facebook/dinov2-base"

    def _init_dinov2_config(self) -> None:
        self.dinov2_model_id = os.environ.get("TF_OVOS_DINOV2_MODEL", self.dinov2_model_name)
        self.dinov2_weight = float(os.environ.get("TF_OVOS_DINOV2_WEIGHT", "0.35"))
        self.dinov2_cache_dir = os.environ.get(
            "TF_OVOS_DINOV2_CACHE",
            str(Path(__file__).resolve().parents[3] / ".cache" / "huggingface"),
        )

    def _load_dinov2(self) -> None:
        from transformers import AutoImageProcessor, AutoModel

        Path(self.dinov2_cache_dir).mkdir(parents=True, exist_ok=True)
        self.dinov2_processor = AutoImageProcessor.from_pretrained(
            self.dinov2_model_id,
            cache_dir=self.dinov2_cache_dir,
        )
        self.dinov2_model = AutoModel.from_pretrained(
            self.dinov2_model_id,
            cache_dir=self.dinov2_cache_dir,
        )
        self.dinov2_model.eval().to(self.device)

    def _dinov2_mask_scores(self, image_rgb: Image.Image, masks: list[dict]) -> np.ndarray:
        torch = self.torch
        if not masks:
            return np.zeros(0, dtype=np.float32)

        inputs = self.dinov2_processor(images=image_rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            output = self.dinov2_model(**inputs)
            tokens = output.last_hidden_state[:, 1:, :].float()[0]
            tokens = tokens / tokens.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        n_tokens = int(tokens.shape[0])
        grid_h, grid_w = self._infer_patch_grid(n_tokens, inputs)
        tokens = tokens[: grid_h * grid_w].reshape(grid_h, grid_w, -1)
        global_feat = tokens.reshape(-1, tokens.shape[-1]).mean(dim=0)
        global_feat = global_feat / global_feat.norm().clamp_min(1e-6)

        raw_scores: list[float] = []
        for mask in masks:
            seg = Image.fromarray(mask["segmentation"].astype(np.uint8) * 255)
            seg_small = seg.resize((grid_w, grid_h), Image.Resampling.NEAREST)
            seg_np = np.asarray(seg_small) > 0
            if not np.any(seg_np):
                raw_scores.append(0.0)
                continue
            selected = tokens[torch.as_tensor(seg_np, device=self.device)]
            feat = selected.mean(dim=0)
            feat = feat / feat.norm().clamp_min(1e-6)
            distinctiveness = float((1.0 - torch.dot(feat, global_feat)).clamp(min=0.0).item())
            area_frac = float(np.mean(seg_np))
            area_prior = math.sqrt(max(area_frac, 1e-6)) * (1.0 - math.sqrt(min(area_frac, 1.0)))
            raw_scores.append(distinctiveness * (0.5 + area_prior))

        scores = np.asarray(raw_scores, dtype=np.float32)
        if scores.size and float(scores.max()) > float(scores.min()):
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            scores = np.zeros_like(scores)
        return scores

    def _infer_patch_grid(self, n_tokens: int, inputs: dict) -> tuple[int, int]:
        side = int(math.sqrt(n_tokens))
        if side * side == n_tokens:
            return side, side

        pixel_values = inputs.get("pixel_values")
        if pixel_values is not None:
            height = int(pixel_values.shape[-2])
            width = int(pixel_values.shape[-1])
            patch = int(getattr(getattr(self.dinov2_model, "config", object()), "patch_size", 14))
            grid_h = max(1, height // patch)
            grid_w = max(1, width // patch)
            if grid_h * grid_w <= n_tokens:
                return grid_h, grid_w

        return side, max(1, n_tokens // max(side, 1))

    def _predict_one_guided(
        self,
        sample: Sample,
        vocabulary: list[str],
        text_features,
        output_dir: Path,
        adapter_name: str,
    ) -> Prediction:
        with Image.open(sample.image_path) as img:
            image_rgb = img.convert("RGB")
        width, height = image_rgb.size
        image_np = np.asarray(image_rgb)
        masks = self.mask_generator.generate(image_np)
        num_classes = self.num_classes or len(vocabulary)

        if not masks:
            label_map = np.zeros((height, width), dtype=np.uint8 if num_classes <= 256 else np.uint16)
            crop_feat = self._encode_crops([image_rgb])
            sims = (crop_feat @ text_features.T)[0].float().cpu().numpy()
            label_map[:] = int(np.argmax(sims))
            num_masks = 0
        else:
            masks_sorted = sorted(masks, key=lambda m: m["area"], reverse=True)
            crops: list[Image.Image] = []
            for mask in masks_sorted:
                x, y, w, h = [int(v) for v in mask["bbox"]]
                x2, y2 = min(x + w, width), min(y + h, height)
                crop = image_rgb.crop((x, y, x2, y2))
                crops.append(crop if crop.size[0] >= 1 and crop.size[1] >= 1 else image_rgb)

            crop_features = self._encode_crops(crops)
            sims = (crop_features @ text_features.T).float().cpu().numpy()
            best_class = np.argmax(sims, axis=1)
            dino_scores = self._dinov2_mask_scores(image_rgb, masks_sorted)

            label_map = np.zeros((height, width), dtype=np.int32)
            confidence_map = np.full((height, width), -1.0, dtype=np.float32)
            for i, mask in enumerate(masks_sorted):
                seg = mask["segmentation"]
                sam_conf = float(mask["predicted_iou"]) * float(mask["stability_score"])
                guided_conf = sam_conf * (1.0 + self.dinov2_weight * float(dino_scores[i]))
                cls = int(best_class[i])
                update = seg & (guided_conf > confidence_map)
                label_map[update] = cls
                confidence_map[update] = guided_conf

            label_map = np.clip(label_map, 0, num_classes - 1)
            num_masks = len(masks_sorted)

        pred_dir = output_dir / "pred_masks"
        pred_dir.mkdir(parents=True, exist_ok=True)
        out_path = pred_dir / f"{_mask_name(sample.image_id)}.png"
        if num_classes > 256:
            Image.fromarray(label_map.astype(np.uint16), mode="I;16").save(out_path)
        else:
            Image.fromarray(label_map.astype(np.uint8), mode="L").save(out_path)

        return Prediction(
            image_id=sample.image_id,
            mask_path=Path(as_output_path(out_path, output_dir)),
            label=None,
            metadata={
                "adapter": adapter_name,
                "num_masks": num_masks,
                "dinov2_model": self.dinov2_model_id,
                "dinov2_weight": self.dinov2_weight,
            },
        )


class Dinov2SamClipAdapter(_Dinov2GuidanceMixin, SamAmgClipAdapter):
    """SAM proposals named by CLIP, with DINOv2-guided proposal ranking."""

    name = "dinov2_sam_clip"
    runnable = True
    setup_hint = (
        "Requires SAM ViT-H weights plus Hugging Face transformers DINOv2 weights "
        "(default facebook/dinov2-base, configurable via TF_OVOS_DINOV2_MODEL)."
    )

    def __init__(self) -> None:
        super().__init__()
        self._init_dinov2_config()

    def _load(self) -> None:
        if self._loaded:
            return
        super()._load()
        self._load_dinov2()

    def predict_many(
        self, samples: Iterable[Sample], vocabulary: list[str], output_dir: Path
    ) -> list[Prediction]:
        samples = list(samples)
        if not vocabulary:
            raise ValueError("dinov2_sam_clip requires a non-empty vocabulary")
        self._load()
        text_features = self._encode_text(vocabulary)
        output_dir.mkdir(parents=True, exist_ok=True)
        return [
            self._predict_one_guided(s, vocabulary, text_features, output_dir, self.name)
            for s in samples
        ]


class Dinov2SamSiglipAdapter(_Dinov2GuidanceMixin, SamAmgSiglipAdapter):
    """SAM proposals named by SigLIP, with DINOv2-guided proposal ranking."""

    name = "dinov2_sam_siglip"
    runnable = True
    setup_hint = (
        "Requires SAM ViT-H weights plus Hugging Face transformers DINOv2 weights "
        "(default facebook/dinov2-base, configurable via TF_OVOS_DINOV2_MODEL)."
    )

    def __init__(self) -> None:
        super().__init__()
        self._init_dinov2_config()

    def _load(self) -> None:
        if self._loaded:
            return
        super()._load()
        self._load_dinov2()

    def predict_many(
        self, samples: Iterable[Sample], vocabulary: list[str], output_dir: Path
    ) -> list[Prediction]:
        samples = list(samples)
        if not vocabulary:
            raise ValueError("dinov2_sam_siglip requires a non-empty vocabulary")
        self._load()
        text_features = self._encode_text(vocabulary)
        output_dir.mkdir(parents=True, exist_ok=True)
        return [
            self._predict_one_guided(s, vocabulary, text_features, output_dir, self.name)
            for s in samples
        ]
