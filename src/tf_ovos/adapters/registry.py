from __future__ import annotations

from dataclasses import dataclass

from tf_ovos.adapters.base import MethodAdapter
from tf_ovos.adapters.debug import CopyGroundTruthAdapter, EmptyMaskAdapter
from tf_ovos.adapters.dinov2_sam import Dinov2SamClipAdapter, Dinov2SamSiglipAdapter
from tf_ovos.adapters.maskclip import MaskClipAdapter
from tf_ovos.adapters.maskclip_attn import MaskClipAttnAdapter, MaskClipAttnSlideAdapter
from tf_ovos.adapters.sam_amg_clip import SamAmgClipAdapter
from tf_ovos.adapters.sam_amg_siglip import SamAmgSiglipAdapter
from tf_ovos.adapters.planned import PROPOSAL_PLANNED_ADAPTERS


@dataclass(frozen=True)
class AdapterInfo:
    name: str
    adapter_cls: type[MethodAdapter]
    runnable: bool
    setup_hint: str = ""


_ADAPTER_CLASSES: tuple[type[MethodAdapter], ...] = (
    CopyGroundTruthAdapter,
    EmptyMaskAdapter,
    *PROPOSAL_PLANNED_ADAPTERS,
    MaskClipAdapter,
    MaskClipAttnAdapter,
    MaskClipAttnSlideAdapter,
    SamAmgClipAdapter,
    SamAmgSiglipAdapter,
    Dinov2SamClipAdapter,
    Dinov2SamSiglipAdapter,
)

ADAPTERS: dict[str, type[MethodAdapter]] = {adapter.name: adapter for adapter in _ADAPTER_CLASSES}


def adapter_info() -> list[AdapterInfo]:
    rows: list[AdapterInfo] = []
    for name in sorted(ADAPTERS):
        adapter_cls = ADAPTERS[name]
        rows.append(
            AdapterInfo(
                name=name,
                adapter_cls=adapter_cls,
                runnable=getattr(adapter_cls, "runnable", True),
                setup_hint=getattr(adapter_cls, "setup_hint", ""),
            )
        )
    return rows
