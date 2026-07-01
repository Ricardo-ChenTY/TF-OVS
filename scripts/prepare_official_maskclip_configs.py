#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "third_party/official_methods/maskclip"
CFG_DIR = REPO / "configs/tfovos"
DATA = ROOT / "data/official_mmseg"
ADE = ROOT / "data/raw/ADEChallengeData2016"

COMMON = """
# Official MaskCLIP config generated for TF-OVOS data paths.
# Uses the upstream MaskCLIP model and prompt embeddings; only data roots and
# evaluation splits are redirected to this workspace.
""".lstrip()


def write(name: str, body: str) -> None:
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    path = CFG_DIR / name
    path.write_text(COMMON + body, encoding="utf-8")
    print(f"[maskclip-config] wrote {path}")


write("maskclip_vit16_tfovos_voc20.py", f"""
_base_ = ['../maskclip/maskclip_vit16_512x512_voc12aug_20.py']

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    val=dict(data_root={str(DATA / 'voc20')!r}, img_dir='JPEGImages', ann_dir='SegmentationClass', split='ImageSets/Segmentation/val.txt'),
    test=dict(data_root={str(DATA / 'voc20')!r}, img_dir='JPEGImages', ann_dir='SegmentationClass', split='ImageSets/Segmentation/val.txt'))
""")

write("maskclip_vit16_tfovos_context59.py", f"""
_base_ = ['../maskclip/maskclip_vit16_520x520_pascal_context_59.py']

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    val=dict(data_root={str(DATA / 'context59')!r}, img_dir='JPEGImages', ann_dir='SegmentationClassContext', split='ImageSets/SegmentationContext/val.txt'),
    test=dict(data_root={str(DATA / 'context59')!r}, img_dir='JPEGImages', ann_dir='SegmentationClassContext', split='ImageSets/SegmentationContext/val.txt'))
""")

write("maskclip_vit16_tfovos_ade150.py", f"""
_base_ = ['../_base_/models/maskclip_vit16.py', '../_base_/datasets/ade20k.py', '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py']

model = dict(
    decode_head=dict(
        num_classes=150,
        text_categories=150,
        text_channels=512,
        text_embeddings_path='pretrain/ade_ViT16_clip_text.pth',
        visual_projs_path='pretrain/ViT16_clip_weights.pth'))

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    val=dict(data_root={str(ADE)!r}, img_dir='images/validation', ann_dir='annotations/validation'),
    test=dict(data_root={str(ADE)!r}, img_dir='images/validation', ann_dir='annotations/validation'))
""")

write("maskclip_vit16_tfovos_coco171.py", f"""
_base_ = ['../maskclip/maskclip_vit16_512x512_coco-stuff164k.py']

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    val=dict(data_root={str(DATA / 'coco_stuff171')!r}, img_dir='images/val2017', ann_dir='annotations/val2017'),
    test=dict(data_root={str(DATA / 'coco_stuff171')!r}, img_dir='images/val2017', ann_dir='annotations/val2017'))
""")
