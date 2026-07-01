from __future__ import annotations

from pathlib import Path

from tf_ovos.adapters.base import MethodAdapter
from tf_ovos.data import Prediction, Sample


class PlannedAdapter(MethodAdapter):
    runnable = False
    setup_hint = "This adapter is planned but not implemented yet."

    def predict_one(self, sample: Sample, vocabulary: list[str], output_dir: Path) -> Prediction:
        raise NotImplementedError(self.setup_hint)


def planned_adapter(name: str, setup_hint: str) -> type[PlannedAdapter]:
    class_name = "".join(part.capitalize() for part in name.replace("-", "_").split("_")) + "Adapter"
    return type(class_name, (PlannedAdapter,), {"name": name, "setup_hint": setup_hint})


# ── CLIP-only / attention-edited OVSS ──────────────────────────────────────
MaskClipAdapter = planned_adapter(
    "maskclip",
    "MaskCLIP needs its external repo or dense CLIP feature extraction code.",
)
ClipDiyAdapter = planned_adapter(
    "clip_diy",
    "CLIP-DIY needs dense CLIP inference and localization-prior code.",
)
SclipAdapter = planned_adapter(
    "sclip",
    "SCLIP needs the external SCLIP repo, CUDA PyTorch, weights, and dense-map conversion code.",
)
NaclipAdapter = planned_adapter(
    "naclip",
    "NACLIP needs the external NACLIP implementation and dense-map conversion code.",
)
CliptraseAdapter = planned_adapter(
    "cliptrase",
    "CLIPtrase is a backup row and needs a stable public implementation check.",
)
ScClipAdapter = planned_adapter(
    "sc_clip",
    "SC-CLIP is a backup row and needs a stable public implementation check.",
)
ResClipAdapter = planned_adapter(
    "resclip",
    "ResCLIP needs the external ResCLIP implementation and dense-map conversion code.",
)

# ── CLIP + VFM OVSS ────────────────────────────────────────────────────────
ProxyClipAdapter = planned_adapter(
    "proxyclip",
    "ProxyCLIP needs the external ProxyCLIP repo, CUDA PyTorch, weights, and dense-map conversion code.",
)
CorrClipAdapter = planned_adapter(
    "corrclip",
    "CorrCLIP needs a public implementation check and patch-correlation reconstruction code.",
)
TridentAdapter = planned_adapter(
    "trident",
    "Trident needs CLIP, DINO, SAM, and the public high-resolution OVSS pipeline code.",
)
CassAdapter = planned_adapter(
    "cass",
    "CASS is a backup row and needs a stable public implementation check.",
)

# ── Diffusion / reference-based OVSS ───────────────────────────────────────
OvDiffAdapter = planned_adapter(
    "ovdiff",
    "OVDiff needs diffusion reference generation and runtime feasibility checks.",
)
FreeDaAdapter = planned_adapter(
    "freeda",
    "FreeDA needs offline diffusion-augmented prototype generation code.",
)

# ── OV detection + SAM ─────────────────────────────────────────────────────
GroundingDinoSamAdapter = planned_adapter(
    "groundingdino_sam",
    "GroundingDINO+SAM needs GroundingDINO weights, SAM weights, CUDA PyTorch, and model-loading code.",
)
GroundingDinoSam2Adapter = planned_adapter(
    "groundingdino_sam2",
    "GroundingDINO+SAM2 needs GroundingDINO weights, SAM2 weights, CUDA PyTorch, and model-loading code.",
)

# ── Class-agnostic proposal + VLM naming ───────────────────────────────────
# SamAmgClipAdapter and SamAmgSiglipAdapter are implemented in sam_amg_clip.py / sam_amg_siglip.py
Dinov2SamClipAdapter = planned_adapter(
    "dinov2_sam_clip",
    "DINOv2+SAM+CLIP needs DINOv2 spatial priors, SAM proposals, CLIP naming, and ranking code.",
)
Dinov2SamSiglipAdapter = planned_adapter(
    "dinov2_sam_siglip",
    "DINOv2+SAM+SigLIP needs DINOv2 spatial priors, SAM proposals, SigLIP naming, and ranking code.",
)
# MCC is a post-hoc diagnostic probe, not a primary leaderboard method.
Dinov2SamClipMccAdapter = planned_adapter(
    "dinov2_sam_clip_mcc",
    "DINOv2+SAM+CLIP+MCC: post-hoc mask-category consistency diagnostic probe (not a ranked method).",
)

# ── Compact trained references (not strict TF) ─────────────────────────────
OvsegRefAdapter = planned_adapter(
    "ovseg_ref",
    "OVSeg is a compact trained reference; needs official weights and mask-adapted CLIP inference code.",
)
SanRefAdapter = planned_adapter(
    "san_ref",
    "SAN is a compact trained reference; needs official weights and side-adapter inference code.",
)
OdiseRefAdapter = planned_adapter(
    "odise_ref",
    "ODISE is a compact trained reference; needs official weights and diffusion-based panoptic inference code.",
)


PROPOSAL_PLANNED_ADAPTERS: tuple[type[PlannedAdapter], ...] = (
    MaskClipAdapter,
    ClipDiyAdapter,
    SclipAdapter,
    NaclipAdapter,
    CliptraseAdapter,
    ScClipAdapter,
    ResClipAdapter,
    ProxyClipAdapter,
    CorrClipAdapter,
    TridentAdapter,
    CassAdapter,
    OvDiffAdapter,
    FreeDaAdapter,
    GroundingDinoSamAdapter,
    GroundingDinoSam2Adapter,
    Dinov2SamClipAdapter,
    Dinov2SamSiglipAdapter,
    Dinov2SamClipMccAdapter,
    OvsegRefAdapter,
    SanRefAdapter,
    OdiseRefAdapter,
)
