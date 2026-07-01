from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from tf_ovos.adapters.base import MethodAdapter
from tf_ovos.data import Prediction, Sample, as_output_path


def _safe_mask_name(image_id: str) -> str:
    return image_id.replace("/", "__").replace("\\", "__")


class MaskClipAdapter(MethodAdapter):
    """CLIP patch-token dense-map baseline.

    This is a lightweight MaskCLIP-style adapter for validating the real-model
    pipeline. It uses frozen CLIP ViT patch tokens and text embeddings; it does
    not implement the full MaskCLIP attention surgery.
    """

    name = "maskclip"
    runnable = True

    def __init__(self) -> None:
        self.task: str | None = None
        self.num_classes: int | None = None
        self.model_name = os.environ.get("TF_OVOS_MASKCLIP_MODEL", "ViT-B-16")
        self.pretrained = os.environ.get("TF_OVOS_MASKCLIP_PRETRAINED", "openai")
        self.prompt_template = os.environ.get("TF_OVOS_MASKCLIP_PROMPT", "a photo of a {}")
        self.image_size = int(os.environ.get("TF_OVOS_MASKCLIP_IMAGE_SIZE", "224"))
        self.mask_threshold = float(os.environ.get("TF_OVOS_MASKCLIP_THRESHOLD", "0.5"))
        self.device_name = os.environ.get("TF_OVOS_MASKCLIP_DEVICE", "cuda")
        self._loaded = False

    def configure(self, **kwargs: object) -> None:
        self.task = kwargs.get("task") if isinstance(kwargs.get("task"), str) else self.task
        value = kwargs.get("num_classes")
        self.num_classes = int(value) if value is not None else self.num_classes

    def predict_many(self, samples: Iterable[Sample], vocabulary: list[str], output_dir: Path) -> list[Prediction]:
        samples = list(samples)
        if not vocabulary:
            raise ValueError("maskclip requires a non-empty vocabulary")
        self._load()
        text_features = self._encode_text(vocabulary)
        output_dir.mkdir(parents=True, exist_ok=True)
        return [self._predict_with_text(sample, vocabulary, text_features, output_dir) for sample in samples]

    def predict_one(self, sample: Sample, vocabulary: list[str], output_dir: Path) -> Prediction:
        return self.predict_many([sample], vocabulary, output_dir)[0]

    def _load(self) -> None:
        if self._loaded:
            return
        import open_clip
        import torch

        self.torch = torch
        self.open_clip = open_clip
        if self.device_name == "cuda" and not torch.cuda.is_available():
            self.device_name = "cpu"
        self.device = torch.device(self.device_name)
        self.model, _, _ = open_clip.create_model_and_transforms(self.model_name, pretrained=self.pretrained)
        self.model.eval().to(self.device)
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self._loaded = True

    def _preprocess(self, image: Image.Image):
        torch = self.torch
        image = image.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        arr = np.asarray(image).astype("float32") / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])[:, None, None]
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711])[:, None, None]
        return ((tensor - mean) / std).unsqueeze(0).to(self.device)

    def _encode_text(self, vocabulary: list[str]):
        torch = self.torch
        prompts = [self.prompt_template.format(label) for label in vocabulary]
        tokens = self.tokenizer(prompts).to(self.device)
        with torch.inference_mode():
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return features

    def _encode_patch_tokens(self, image_tensor):
        torch = self.torch
        visual = self.model.visual
        old_output_tokens = getattr(visual, "output_tokens", False)
        visual.output_tokens = True
        try:
            with torch.inference_mode():
                _, tokens = visual(image_tensor)
                if getattr(visual, "proj", None) is not None:
                    tokens = tokens @ visual.proj
                tokens = tokens / tokens.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        finally:
            visual.output_tokens = old_output_tokens
        return tokens

    def _score_maps(self, sample: Sample, text_features):
        torch = self.torch
        with Image.open(sample.image_path) as image:
            original_size = image.size
            image_tensor = self._preprocess(image)
        tokens = self._encode_patch_tokens(image_tensor)
        grid = int(tokens.shape[1] ** 0.5)
        scores = tokens[0] @ text_features.T
        scores = scores.T.reshape(1, -1, grid, grid)
        scores = torch.nn.functional.interpolate(
            scores,
            size=(original_size[1], original_size[0]),
            mode="bilinear",
            align_corners=False,
        )[0]
        return scores.float().cpu().numpy(), original_size

    def _predict_with_text(
        self,
        sample: Sample,
        vocabulary: list[str],
        text_features,
        output_dir: Path,
    ) -> Prediction:
        score_maps, _ = self._score_maps(sample, text_features)
        pred_dir = output_dir / "pred_masks"
        pred_dir.mkdir(parents=True, exist_ok=True)

        if self.task == "semantic":
            label_map = np.argmax(score_maps, axis=0)
            max_class = self.num_classes or len(vocabulary)
            label_map = np.clip(label_map, 0, max_class - 1)
            out_path = pred_dir / f"{_safe_mask_name(sample.image_id)}.png"
            if max_class > 256:
                Image.fromarray(label_map.astype(np.uint16), mode="I;16").save(out_path)
            else:
                Image.fromarray(label_map.astype(np.uint8), mode="L").save(out_path)
            return Prediction(
                image_id=sample.image_id,
                mask_path=Path(as_output_path(out_path, output_dir)),
                label=None,
                metadata={"adapter": "maskclip", "variant": "clip_patch_dense_lite"},
            )

        class_scores = score_maps.reshape(score_maps.shape[0], -1).max(axis=1)
        class_idx = int(np.argmax(class_scores))
        class_map = score_maps[class_idx]
        lo, hi = float(class_map.min()), float(class_map.max())
        norm = (class_map - lo) / max(hi - lo, 1e-6)
        binary = (norm >= self.mask_threshold).astype(np.uint8) * 255
        out_path = pred_dir / f"{_safe_mask_name(sample.image_id)}.png"
        Image.fromarray(binary, mode="L").save(out_path)
        return Prediction(
            image_id=sample.image_id,
            mask_path=Path(as_output_path(out_path, output_dir)),
            label=vocabulary[class_idx],
            score=float(class_scores[class_idx]),
            metadata={"adapter": "maskclip", "variant": "clip_patch_dense_lite"},
        )
