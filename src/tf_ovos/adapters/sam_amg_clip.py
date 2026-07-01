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


class SamAmgClipAdapter(MethodAdapter):
    """SAM Automatic Mask Generator + CLIP naming.

    Each image is segmented with SAM AMG (class-agnostic), then every
    proposed mask region is scored against the vocabulary with CLIP.
    Each pixel is assigned to the class of the highest-scoring mask that
    covers it; ties are broken by mask confidence.
    """

    name = "sam_amg_clip"
    runnable = True
    setup_hint = (
        "Requires `pip install segment-anything` and SAM ViT-H weights at "
        "weights/sam_vit_h_4b8939.pth (or TF_OVOS_SAM_CHECKPOINT env var)."
    )

    def __init__(self) -> None:
        self.task: str | None = None
        self.num_classes: int | None = None
        self.sam_checkpoint = os.environ.get(
            "TF_OVOS_SAM_CHECKPOINT",
            str(Path(__file__).resolve().parents[3] / "weights" / "sam_vit_h_4b8939.pth"),
        )
        self.clip_model = os.environ.get("TF_OVOS_SAM_CLIP_MODEL", "ViT-L-14")
        self.clip_pretrained = os.environ.get("TF_OVOS_SAM_CLIP_PRETRAINED", "openai")
        self.prompt_template = os.environ.get("TF_OVOS_SAM_CLIP_PROMPT", "a photo of a {}")
        self.device_name = os.environ.get("TF_OVOS_SAM_DEVICE", "cuda")
        # AMG parameters (defaults match SAM paper)
        self.points_per_side = int(os.environ.get("TF_OVOS_SAM_PPS", "32"))
        self.pred_iou_thresh = float(os.environ.get("TF_OVOS_SAM_IOU_THR", "0.86"))
        self.stability_score_thresh = float(os.environ.get("TF_OVOS_SAM_STAB_THR", "0.92"))
        self.min_mask_region_area = int(os.environ.get("TF_OVOS_SAM_MIN_AREA", "400"))
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
            raise ValueError("sam_amg_clip requires a non-empty vocabulary")
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
        import open_clip
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

        self.torch = torch
        if self.device_name == "cuda" and not torch.cuda.is_available():
            self.device_name = "cpu"
        self.device = torch.device(self.device_name)

        # SAM
        ckpt = Path(self.sam_checkpoint)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"SAM checkpoint not found: {ckpt}\n"
                "Download: wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P weights/"
            )
        sam = sam_model_registry["vit_h"](checkpoint=str(ckpt))
        sam.to(self.device).eval()
        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=self.points_per_side,
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
            min_mask_region_area=self.min_mask_region_area,
        )

        # CLIP
        self.clip_model_obj, self.clip_preprocess, _ = open_clip.create_model_and_transforms(
            self.clip_model, pretrained=self.clip_pretrained
        )
        self.clip_model_obj.eval().to(self.device)
        self.tokenizer = open_clip.get_tokenizer(self.clip_model)
        self._loaded = True

    # ── text encoding ───────────────────────────────────────────────────────

    def _encode_text(self, vocabulary: list[str]):
        prompts = [self.prompt_template.format(cls) for cls in vocabulary]
        tokens = self.tokenizer(prompts).to(self.device)
        with self.torch.inference_mode():
            feats = self.clip_model_obj.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return feats  # (C, D)

    # ── image encoding ──────────────────────────────────────────────────────

    def _encode_crops(self, crops: list[Image.Image]):
        torch = self.torch
        if not crops:
            return torch.zeros(0, device=self.device)
        tensors = torch.stack([self.clip_preprocess(c) for c in crops]).to(self.device)
        with torch.inference_mode():
            feats = self.clip_model_obj.encode_image(tensors)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return feats  # (N, D)

    # ── prediction core ─────────────────────────────────────────────────────

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
        W, H = image_rgb.size
        image_np = np.asarray(image_rgb)

        # Generate proposals
        masks = self.mask_generator.generate(image_np)
        # masks: list of dicts with keys: segmentation (H×W bool), bbox ([x,y,w,h]),
        #        predicted_iou, stability_score, area

        num_classes = self.num_classes or len(vocabulary)

        if not masks:
            # fallback: entire image as one region
            label_map = np.zeros((H, W), dtype=np.uint8 if num_classes <= 256 else np.uint16)
            crop_feat = self._encode_crops([image_rgb])
            sims = (crop_feat @ text_features.T)[0].float().cpu().numpy()
            label_map[:] = int(np.argmax(sims))
        else:
            # Sort by area descending so large masks fill background first
            masks_sorted = sorted(masks, key=lambda m: m["area"], reverse=True)

            # Crop each proposal's bounding box
            crops: list[Image.Image] = []
            for m in masks_sorted:
                x, y, w, h = [int(v) for v in m["bbox"]]
                x2, y2 = min(x + w, W), min(y + h, H)
                crop = image_rgb.crop((x, y, x2, y2))
                if crop.size[0] < 1 or crop.size[1] < 1:
                    crop = image_rgb
                crops.append(crop)

            crop_features = self._encode_crops(crops)  # (N, D)
            sims = (crop_features @ text_features.T).float().cpu().numpy()  # (N, C)
            best_class = np.argmax(sims, axis=1)  # (N,)

            # Build label map: paint from large→small so small masks win
            label_map = np.zeros((H, W), dtype=np.int32)
            confidence_map = np.full((H, W), -1.0, dtype=np.float32)
            for i, m in enumerate(masks_sorted):
                seg = m["segmentation"]  # (H, W) bool
                conf = float(m["predicted_iou"]) * float(m["stability_score"])
                cls = int(best_class[i])
                update = seg & (conf > confidence_map)
                label_map[update] = cls
                confidence_map[update] = conf

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
            metadata={"adapter": "sam_amg_clip", "num_masks": len(masks) if masks else 0},
        )
