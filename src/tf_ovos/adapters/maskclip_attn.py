from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from tf_ovos.adapters.base import MethodAdapter
from tf_ovos.data import Prediction, Sample, as_output_path

_SAFE_CHARS = str.maketrans("/\\", "__")


def _mask_name(image_id: str) -> str:
    return image_id.translate(_SAFE_CHARS)


# 80 CLIP prompt templates (same as MaskCLIP paper).
_PROMPT_TEMPLATES = [
    "a bad photo of a {}.", "a photo of many {}.", "a sculpture of a {}.",
    "a photo of the hard to see {}.", "a low resolution photo of the {}.",
    "a rendering of a {}.", "graffiti of a {}.", "a bad photo of the {}.",
    "a cropped photo of the {}.", "a tattoo of a {}.", "the embroidered {}.",
    "a photo of a hard to see {}.", "a bright photo of a {}.",
    "a photo of a clean {}.", "a photo of a dirty {}.", "a dark photo of the {}.",
    "a drawing of a {}.", "a photo of my {}.", "the plastic {}.",
    "a photo of the cool {}.", "a close-up photo of a {}.",
    "a black and white photo of the {}.", "a painting of the {}.",
    "a painting of a {}.", "a pixelated photo of the {}.",
    "a sculpture of the {}.", "a bright photo of the {}.",
    "a cropped photo of a {}.", "a plastic {}.", "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.", "a blurry photo of the {}.",
    "a photo of the {}.", "a good photo of the {}.", "a rendering of the {}.",
    "a {} in a video game.", "a photo of one {}.", "a doodle of a {}.",
    "a close-up photo of the {}.", "a photo of a {}.", "the origami {}.",
    "the {} in a video game.", "a sketch of a {}.", "a doodle of the {}.",
    "a origami {}.", "a low resolution photo of a {}.", "the toy {}.",
    "a rendition of the {}.", "a photo of the clean {}.",
    "a photo of a large {}.", "a rendition of a {}.", "a photo of a nice {}.",
    "a photo of a weird {}.", "a blurry photo of a {}.", "a cartoon {}.",
    "art of a {}.", "a sketch of the {}.", "a embroidered {}.",
    "a pixelated photo of a {}.", "itap of the {}.",
    "a jpeg corrupted photo of the {}.", "a good photo of a {}.",
    "a plushie {}.", "a photo of the nice {}.", "a photo of the small {}.",
    "a photo of the weird {}.", "the cartoon {}.", "art of the {}.",
    "a drawing of the {}.", "a photo of the large {}.",
    "a black and white photo of a {}.", "the plushie {}.",
    "a dark photo of a {}.", "itap of a {}.", "graffiti of the {}.",
    "a toy {}.", "itap of my {}.", "a photo of a cool {}.",
    "a photo of a small {}.", "a tattoo of the {}.",
    "there is a {} in the scene.", "there is the {} in the scene.",
    "this is a {} in the scene.", "this is the {} in the scene.",
    "this is one {} in the scene.",
]


class MaskClipAttnAdapter(MethodAdapter):
    """MaskCLIP-style dense CLIP features via all-patch evaluation.

    Implements zero-shot OVSS using CLIP ViT-B/16 dense features:
    - Hooks the last transformer block output to capture all patch tokens.
    - Applies ln_post + visual.proj (same as CLIP CLS path) to each patch.
    - Assigns each spatial location the highest-similarity class label.
    - Uses 80-template prompt ensemble (same as MaskCLIP paper).
    """

    name = "maskclip_attn"
    runnable = True
    setup_hint = "Requires `pip install git+https://github.com/openai/CLIP.git` (the `clip` package)."

    def __init__(self) -> None:
        self.task: str | None = None
        self.num_classes: int | None = None
        self.image_size = int(os.environ.get("TF_OVOS_MASKCLIP_IMGSIZE", "224"))
        self.device_name = os.environ.get("TF_OVOS_MASKCLIP_DEVICE", "cuda")
        self._loaded = False

    def configure(self, **kwargs: object) -> None:
        self.task = kwargs.get("task") if isinstance(kwargs.get("task"), str) else self.task
        v = kwargs.get("num_classes")
        self.num_classes = int(v) if v is not None else self.num_classes

    def predict_many(
        self, samples: Iterable[Sample], vocabulary: list[str], output_dir: Path
    ) -> list[Prediction]:
        samples = list(samples)
        if not vocabulary:
            raise ValueError("maskclip_attn requires a non-empty vocabulary")
        self._load()
        text_features = self._encode_text(vocabulary)
        output_dir.mkdir(parents=True, exist_ok=True)
        return [self._predict_one(s, vocabulary, text_features, output_dir) for s in samples]

    def predict_one(self, sample: Sample, vocabulary: list[str], output_dir: Path) -> Prediction:
        return self.predict_many([sample], vocabulary, output_dir)[0]

    # ── model loading ───────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._loaded:
            return
        import torch
        import clip as openai_clip

        self.torch = torch
        if self.device_name == "cuda" and not torch.cuda.is_available():
            self.device_name = "cpu"
        self.device = torch.device(self.device_name)

        self.clip_model, self.clip_preprocess = openai_clip.load("ViT-B/16", device=self.device)
        self.clip_model.eval()
        self.tokenizer = openai_clip.tokenize

        self.visual_proj = self.clip_model.visual.proj  # (768, 512)
        self.ln_post = self.clip_model.visual.ln_post

        # Hook last transformer block output to capture all patch tokens.
        # Capturing full block output (post-attn, post-MLP) then applying
        # ln_post + proj gives the same projection space as the CLS token,
        # enabling cosine similarity with text features.
        self._token_cache: dict = {}
        last_block = self.clip_model.visual.transformer.resblocks[-1]
        self._hook = last_block.register_forward_hook(self._block_hook)

        self._loaded = True

    def _block_hook(self, module, inputs, output):
        """Capture last transformer block output: (seq_len, batch, D)."""
        self._token_cache["tokens"] = output.permute(1, 0, 2).detach()  # (batch, seq_len, D)

    def __del__(self):
        if hasattr(self, "_hook"):
            try:
                self._hook.remove()
            except Exception:
                pass

    # ── text encoding ───────────────────────────────────────────────────────

    def _encode_text(self, vocabulary: list[str]):
        torch = self.torch
        weights = []
        for cls in vocabulary:
            tokens = self.tokenizer(
                [t.format(cls) for t in _PROMPT_TEMPLATES], truncate=True
            ).to(self.device)
            with torch.inference_mode():
                feats = self.clip_model.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                feat = feats.mean(dim=0)
                feat = feat / feat.norm()
            weights.append(feat)
        return torch.stack(weights, dim=0)  # (C, D_text)

    # ── prediction core ─────────────────────────────────────────────────────

    def _preprocess(self, image: Image.Image):
        torch = self.torch
        image = image.convert("RGB").resize(
            (self.image_size, self.image_size), Image.Resampling.BICUBIC
        )
        arr = np.asarray(image).astype("float32") / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device)[:, None, None]
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device)[:, None, None]
        return ((tensor.to(self.device) - mean) / std).unsqueeze(0)

    def _predict_one(
        self,
        sample: Sample,
        vocabulary: list[str],
        text_features,
        output_dir: Path,
    ) -> Prediction:
        torch = self.torch
        with Image.open(sample.image_path) as img:
            image_rgb = img.convert("RGB")
        W_orig, H_orig = image_rgb.size
        sims_spatial, grid = self._dense_similarity(image_rgb, text_features)

        # Upsample to original image size
        sims_up = torch.nn.functional.interpolate(
            sims_spatial, size=(H_orig, W_orig), mode="bilinear", align_corners=False
        )[0]  # (C, H, W)

        label_map = sims_up.argmax(dim=0).cpu().numpy()  # (H, W)

        num_classes = self.num_classes or len(vocabulary)
        label_map = np.clip(label_map, 0, num_classes - 1)

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
            metadata={"adapter": "maskclip_attn", "grid_size": grid},
        )

    def _dense_similarity(self, image: Image.Image, text_features):
        torch = self.torch
        image_tensor = self._preprocess(image)

        # Run CLIP visual encoder; hook captures last block output.
        with torch.no_grad():
            _ = self.clip_model.visual(image_tensor.type(self.clip_model.dtype))
        tokens = self._token_cache.get("tokens")  # (1, seq_len, D)

        if tokens is None:
            raise RuntimeError("Block output hook did not fire.")

        patch_tokens = tokens[0, 1:]

        with torch.no_grad():
            patch_feat = self.ln_post(patch_tokens)
        feat = patch_feat @ self.visual_proj
        feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        sims = (feat @ text_features.T).float()

        grid = int(patch_tokens.shape[0] ** 0.5)
        sims_spatial = sims.T.reshape(1, -1, grid, grid)
        return sims_spatial, grid


class MaskClipAttnSlideAdapter(MaskClipAttnAdapter):
    """Sliding-window dense CLIP variant for higher effective resolution."""

    name = "maskclip_attn_slide"

    def __init__(self) -> None:
        super().__init__()
        self.crop_size = int(os.environ.get("TF_OVOS_MASKCLIP_CROP", "224"))
        self.stride = int(os.environ.get("TF_OVOS_MASKCLIP_STRIDE", "112"))

    @staticmethod
    def _starts(length: int, crop_size: int, stride: int) -> list[int]:
        if length <= crop_size:
            return [0]
        starts = list(range(0, max(length - crop_size, 0) + 1, stride))
        last = length - crop_size
        if starts[-1] != last:
            starts.append(last)
        return starts

    def _predict_one(
        self,
        sample: Sample,
        vocabulary: list[str],
        text_features,
        output_dir: Path,
    ) -> Prediction:
        torch = self.torch
        with Image.open(sample.image_path) as img:
            image_rgb = img.convert("RGB")
        W_orig, H_orig = image_rgb.size

        num_classes = self.num_classes or len(vocabulary)
        best_score = torch.full((H_orig, W_orig), -float("inf"), device=self.device)
        label_map_t = torch.zeros((H_orig, W_orig), dtype=torch.long, device=self.device)
        grid_size = None
        crop_count = 0

        for top in self._starts(H_orig, self.crop_size, self.stride):
            crop_h = min(self.crop_size, H_orig)
            bottom = top + crop_h
            for left in self._starts(W_orig, self.crop_size, self.stride):
                crop_w = min(self.crop_size, W_orig)
                right = left + crop_w
                crop = image_rgb.crop((left, top, right, bottom))
                sims_spatial, grid = self._dense_similarity(crop, text_features)
                grid_size = grid
                sims_up = torch.nn.functional.interpolate(
                    sims_spatial,
                    size=(crop_h, crop_w),
                    mode="bilinear",
                    align_corners=False,
                )[0]
                crop_score, crop_label = sims_up.max(dim=0)
                weight = self._crop_weight(crop_h, crop_w, crop_score.device)
                crop_score = crop_score + weight.log()
                target_score = best_score[top:bottom, left:right]
                target_label = label_map_t[top:bottom, left:right]
                update = crop_score > target_score
                target_score[update] = crop_score[update]
                target_label[update] = crop_label[update]
                crop_count += 1

        label_map = label_map_t.cpu().numpy()
        label_map = np.clip(label_map, 0, num_classes - 1)

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
                "adapter": "maskclip_attn_slide",
                "grid_size": grid_size,
                "crop_size": self.crop_size,
                "stride": self.stride,
                "crop_count": crop_count,
            },
        )

    def _crop_weight(self, crop_h: int, crop_w: int, device):
        torch = self.torch
        if crop_h <= 1 or crop_w <= 1:
            return torch.ones((crop_h, crop_w), device=device)
        wy = torch.hann_window(crop_h, periodic=False, device=device).clamp_min(0.25)
        wx = torch.hann_window(crop_w, periodic=False, device=device).clamp_min(0.25)
        weight = wy[:, None] * wx[None, :]
        return weight / weight.max().clamp_min(1e-6)
