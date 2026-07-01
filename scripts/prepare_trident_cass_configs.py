#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"
DATA = ROOT / "data"
THIRD_PARTY = ROOT / "third_party" / "official_methods"
ARTIFACT_ROOT = ROOT / "runs" / "artifacts" / "official_predictions"


DATA_ROOTS = {
    "voc20": DATA / "raw" / "VOCdevkit" / "VOC2012",
    "context59": DATA / "raw" / "VOCdevkit" / "VOC2010",
    "ade20k": DATA / "raw" / "ADEChallengeData2016",
    "coco_stuff164k": DATA / "official_mmseg" / "coco_stuff171",
    "context459": DATA / "raw" / "VOCdevkit" / "VOC2010",
    "ade847": DATA / "official_mmseg" / "ade847",
}


TRIDENT_BASE = {
    "voc20": "cfg_voc20.py",
    "context59": "cfg_context59.py",
    "ade20k": "cfg_ade20k.py",
    "coco_stuff164k": "cfg_coco_stuff164k.py",
    "context459": "cfg_context459.py",
    "ade847": "cfg_ade847.py",
}

CASS_BASE = {
    "voc20": "cfg_voc20.py",
    "context59": "cfg_context59.py",
    "ade20k": "cfg_ade20k.py",
    "coco_stuff164k": "cfg_coco_stuff164k.py",
    "context459": "cfg_context459.py",
    "ade847": "cfg_ade847.py",
}


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path}")


def evaluator_block(method: str, dataset: str) -> str:
    out = ARTIFACT_ROOT / method / dataset
    return f"\ntest_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'], output_dir={str(out)!r})\n"


def ensure_links() -> None:
    ctx_root = DATA / "raw" / "VOCdevkit" / "VOC2010"
    ann_dir = ctx_root / "annotations_detectron2"
    ann_dir.mkdir(parents=True, exist_ok=True)
    target = DATA / "official_mmseg" / "context459" / "annotations_detectron2" / "pc459_val"
    link = ann_dir / "pc459_val"
    if not link.exists():
        link.symlink_to(target)
        print(f"linked {link} -> {target}")


def patch_cass_e2_support() -> None:
    path = THIRD_PARTY / "CASS" / "custom_datasets.py"
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
        if self.reduce_zero_label is not None:
            warnings.warn("`reduce_zero_label` will be deprecated; set it on the dataset instead")
        self.imdecode_backend = imdecode_backend

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


def write_trident_configs() -> None:
    cfg_dir = THIRD_PARTY / "Trident" / "configs"
    sam_ckpt = WEIGHTS / "sam_vit_b_01ec64.pth"
    for dataset, base in TRIDENT_BASE.items():
        write(
            cfg_dir / f"cfg_tfovos_{dataset}.py",
            f'''_base_ = "./{base}"

model = dict(
    sam_ckpt="{sam_ckpt}",
    sam_model_type="vit_b",
)

data_root = "{DATA_ROOTS[dataset]}"
test_dataloader = dict(dataset=dict(data_root=data_root))
{evaluator_block("trident", dataset)}
''',
        )


def write_cass_configs() -> None:
    cfg_dir = THIRD_PARTY / "CASS" / "configs"
    for dataset, base in CASS_BASE.items():
        if dataset in {"context459", "ade847"}:
            source = {
                "context459": '''_base_ = "./base_config.py"

model = dict(
    name_path="./configs/cls_context459.txt",
    global_semantics_weight=0.25,
    mean_vector_weight=0.04,
    h_threshold=0.09,
)

dataset_type = "PascalContext459Dataset"
data_root = "{root}"

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
        data_prefix=dict(img_path="JPEGImages", seg_map_path="annotations_detectron2/pc459_val"),
        ann_file="ImageSets/SegmentationContext/val.txt",
        pipeline=test_pipeline,
    ),
)
''',
                "ade847": '''_base_ = "./base_config.py"

model = dict(
    name_path="./configs/cls_ade20k847.txt",
    global_semantics_weight=0.3,
    mean_vector_weight=0.05,
    h_threshold=0.06,
)

dataset_type = "ADE20K847Dataset"
data_root = "{root}"

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
''',
            }[dataset]
            write(cfg_dir / f"cfg_tfovos_{dataset}.py", source.format(root=DATA_ROOTS[dataset]) + evaluator_block("cass", dataset))
        else:
            write(
                cfg_dir / f"cfg_tfovos_{dataset}.py",
                f'''_base_ = "./{base}"

data_root = "{DATA_ROOTS[dataset]}"
test_dataloader = dict(dataset=dict(data_root=data_root))
{evaluator_block("cass", dataset)}
''',
            )


def main() -> None:
    ensure_links()
    patch_cass_e2_support()
    write_trident_configs()
    write_cass_configs()


if __name__ == "__main__":
    main()
