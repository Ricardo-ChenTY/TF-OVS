#!/usr/bin/env python
from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SC_CLIP = ROOT / "third_party" / "official_methods" / "SC-CLIP"
CONFIGS = SC_CLIP / "configs"
ARTIFACT_ROOT = ROOT / "runs" / "artifacts" / "official_predictions"


DATA_ROOTS = {
    "voc20": DATA / "raw" / "VOCdevkit" / "VOC2012",
    "context59": DATA / "official_mmseg" / "context59",
    "ade20k": DATA / "raw" / "ADEChallengeData2016",
    "coco_stuff164k": DATA / "official_mmseg" / "coco_stuff171",
    "context459": DATA / "raw" / "VOCdevkit" / "VOC2010",
    "ade847": DATA / "official_mmseg" / "ade847",
}


def evaluator_block(dataset: str) -> str:
    out = ARTIFACT_ROOT / "scclip" / dataset
    return f"\ntest_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'], output_dir={str(out)!r})\n"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path}")


def copy_class_files() -> None:
    for name in ("cls_context459.txt", "cls_ade20k847.txt"):
        src = ROOT / "third_party" / "official_methods" / "Trident" / "configs" / name
        dst = CONFIGS / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            print(f"copied {src} -> {dst}")


def patch_datasets() -> None:
    path = SC_CLIP / "custom_datasets.py"
    text = path.read_text(encoding="utf-8")
    marker = "# Training-Free Open-Vocabulary Segmentation E2 dataset extensions"
    if marker in text:
        return
    extra = f'''

{marker}
import warnings
import numpy as np
import mmcv
from mmcv.transforms import LoadAnnotations as MMCV_LoadAnnotations
from mmseg.registry import TRANSFORMS


def _tfovos_classes(name):
    cls_path = osp.join(osp.dirname(__file__), "configs", name)
    with open(cls_path, "r", encoding="utf-8") as handle:
        return tuple(line.strip() for line in handle if line.strip())


@DATASETS.register_module()
class ADE20K847Dataset(BaseSegDataset):
    METAINFO = dict(classes=_tfovos_classes("cls_ade20k847.txt"))

    def __init__(self, ann_file, img_suffix=".jpg", seg_map_suffix=".tif", reduce_zero_label=False, **kwargs):
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            ann_file=ann_file,
            **kwargs,
        )
        assert fileio.exists(self.data_prefix["img_path"], self.backend_args) and osp.isfile(self.ann_file)


@DATASETS.register_module()
class PascalContext459Dataset(BaseSegDataset):
    METAINFO = dict(classes=_tfovos_classes("cls_context459.txt"))

    def __init__(self, ann_file, img_suffix=".jpg", seg_map_suffix=".tif", reduce_zero_label=False, **kwargs):
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ann_file=ann_file,
            reduce_zero_label=reduce_zero_label,
            **kwargs,
        )


@TRANSFORMS.register_module()
class MyLoadAnnotations(MMCV_LoadAnnotations):
    def __init__(self, reduce_zero_label=None, backend_args=None, imdecode_backend="pillow") -> None:
        super().__init__(
            with_bbox=False,
            with_label=False,
            with_seg=True,
            with_keypoints=False,
            imdecode_backend=imdecode_backend,
            backend_args=backend_args,
        )
        self.reduce_zero_label = reduce_zero_label
        self.imdecode_backend = imdecode_backend
        if self.reduce_zero_label is not None:
            warnings.warn("`reduce_zero_label` will be deprecated; set it on the dataset instead")

    def _load_seg_map(self, results: dict) -> None:
        img_bytes = fileio.get(results["seg_map_path"], backend_args=self.backend_args)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag="unchanged", backend=self.imdecode_backend
        ).squeeze().astype(np.uint16)
        if self.reduce_zero_label is None:
            self.reduce_zero_label = results["reduce_zero_label"]
        if self.reduce_zero_label:
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        if results.get("label_map", None) is not None:
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results["label_map"].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        results["gt_seg_map"] = gt_semantic_seg
        results["seg_fields"].append("gt_seg_map")
'''
    path.write_text(text + extra, encoding="utf-8")
    print(f"patched {path}")


def write_configs() -> None:
    write(
        CONFIGS / "cfg_tfovos_voc20.py",
        f'''_base_ = "./base_config.py"

model = dict(name_path="./configs/cls_voc20.txt")

dataset_type = "PascalVOC20Dataset"
data_root = "{DATA_ROOTS["voc20"]}"

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(2048, 336), keep_ratio=True),
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path="JPEGImages", seg_map_path="SegmentationClass"),
        ann_file="ImageSets/Segmentation/val.txt",
        pipeline=test_pipeline,
    ),
)
{evaluator_block("voc20")}
''',
    )
    write(
        CONFIGS / "cfg_tfovos_context59.py",
        f'''_base_ = "./base_config.py"

model = dict(name_path="./configs/cls_context59.txt")

dataset_type = "PascalContext59Dataset"
data_root = "{DATA_ROOTS["context59"]}"

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(2048, 336), keep_ratio=True),
    dict(type="LoadAnnotations", reduce_zero_label=True),
    dict(type="PackSegInputs"),
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path="JPEGImages",
            seg_map_path="SegmentationClassContext",
        ),
        ann_file="ImageSets/SegmentationContext/val.txt",
        reduce_zero_label=True,
        pipeline=test_pipeline,
    ),
)
{evaluator_block("context59")}
''',
    )
    write(
        CONFIGS / "cfg_tfovos_ade20k.py",
        f'''_base_ = "./base_config.py"

model = dict(name_path="./configs/cls_ade20k.txt")

dataset_type = "ADE20KDataset"
data_root = "{DATA_ROOTS["ade20k"]}"

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(2048, 336), keep_ratio=True),
    dict(type="LoadAnnotations", reduce_zero_label=True),
    dict(type="PackSegInputs"),
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path="images/validation", seg_map_path="annotations/validation"),
        pipeline=test_pipeline,
    ),
)
{evaluator_block("ade20k")}
''',
    )
    write(
        CONFIGS / "cfg_tfovos_coco_stuff164k.py",
        f'''_base_ = "./base_config.py"

model = dict(name_path="./configs/cls_coco_stuff.txt")

dataset_type = "COCOStuffDataset"
data_root = "{DATA_ROOTS["coco_stuff164k"]}"

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(2048, 336), keep_ratio=True),
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path="images/val2017", seg_map_path="annotations/val2017"),
        pipeline=test_pipeline,
    ),
)
{evaluator_block("coco_stuff164k")}
''',
    )
    write(
        CONFIGS / "cfg_tfovos_context459.py",
        f'''_base_ = "./base_config.py"

model = dict(name_path="./configs/cls_context459.txt")

dataset_type = "PascalContext459Dataset"
data_root = "{DATA_ROOTS["context459"]}"

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(2048, 336), keep_ratio=True),
    dict(type="MyLoadAnnotations", reduce_zero_label=False),
    dict(type="PackSegInputs"),
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path="JPEGImages",
            seg_map_path="{DATA / "official_mmseg" / "context459" / "annotations_detectron2" / "pc459_val"}",
        ),
        ann_file="ImageSets/SegmentationContext/val.txt",
        pipeline=test_pipeline,
    ),
)
{evaluator_block("context459")}
''',
    )
    write(
        CONFIGS / "cfg_tfovos_ade847.py",
        f'''_base_ = "./base_config.py"

model = dict(name_path="./configs/cls_ade20k847.txt")

dataset_type = "ADE20K847Dataset"
data_root = "{DATA_ROOTS["ade847"]}"

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(2048, 336), keep_ratio=True),
    dict(type="MyLoadAnnotations", reduce_zero_label=False),
    dict(type="PackSegInputs"),
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path="images_detectron2/validation", seg_map_path="annotations_detectron2/validation"),
        ann_file="validation.txt",
        pipeline=test_pipeline,
    ),
)
{evaluator_block("ade847")}
''',
    )


def main() -> None:
    copy_class_files()
    patch_datasets()
    write_configs()


if __name__ == "__main__":
    main()
