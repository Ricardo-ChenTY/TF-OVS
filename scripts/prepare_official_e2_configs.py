from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "third_party" / "official_methods"
DATA_ROOT = ROOT / "data" / "official_mmseg"
ARTIFACT_ROOT = ROOT / "runs" / "artifacts" / "official_predictions"

SCLIP = OFFICIAL / "SCLIP"
NACLIP = OFFICIAL / "NACLIP"
RESCLIP = OFFICIAL / "ResCLIP"
PROXY = OFFICIAL / "ProxyCLIP"
CORR = OFFICIAL / "CorrCLIP"


def _rel(target: Path, link_parent: Path) -> str:
    return str(target.resolve())


def _symlink(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        link.unlink()
    elif link.exists():
        return
    link.symlink_to(_rel(target, link.parent))


def _manifest_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_ids(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(row["image_id"] for row in rows) + "\n", encoding="utf-8")


def _prepare_context459() -> Path:
    root = DATA_ROOT / "context459"
    rows = _manifest_rows(ROOT / "data" / "manifests" / "context459_val.jsonl")

    _symlink(ROOT / "data/raw/VOCdevkit/VOC2010/JPEGImages", root / "JPEGImages")
    _write_ids(rows, root / "ImageSets/SegmentationContext/val.txt")

    labels = root / "annotations_detectron2" / "pc459_val"
    labels.mkdir(parents=True, exist_ok=True)
    for row in rows:
        source = (ROOT / "data" / "raw" / "context459_labels" / f"{row['image_id']}.png").resolve()
        _symlink(source, labels / f"{row['image_id']}.tif")

    return root


def _ade_label_name(image_id: str) -> str:
    return image_id.replace("/", "__") + ".png"


def _prepare_ade847() -> Path:
    root = DATA_ROOT / "ade847"
    rows = _manifest_rows(ROOT / "data" / "manifests" / "ade20k847_val.jsonl")

    _symlink(
        ROOT / "data/raw/ADE20K_2021_17_01/ADE20K_2021_17_01/images/ADE/validation",
        root / "images_detectron2" / "validation",
    )
    _write_ids(rows, root / "validation.txt")

    labels = root / "annotations_detectron2" / "validation"
    for row in rows:
        source = (ROOT / "data" / "raw" / "ade20k847_labels" / _ade_label_name(row["image_id"])).resolve()
        _symlink(source, labels / f"{row['image_id']}.tif")

    return root


def _copy_proxy_e2_class_files(dst_repo: Path) -> None:
    for name in ("cls_context459.txt", "cls_ade20k847.txt"):
        source = PROXY / "configs" / name
        target = dst_repo / "configs" / name
        if target.exists() and target.read_text(encoding="utf-8", errors="replace") == source.read_text(
            encoding="utf-8", errors="replace"
        ):
            continue
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def _patch_custom_datasets(repo: Path) -> None:
    path = repo / "custom_datasets.py"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    marker = "# Training-Free Open-Vocabulary Segmentation E2 dataset extensions"
    if marker in text:
        return

    block = '''

# Training-Free Open-Vocabulary Segmentation E2 dataset extensions
import warnings

import mmcv
import numpy as np
from mmcv.transforms import LoadAnnotations as MMCVLoadAnnotations
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
            reduce_zero_label=reduce_zero_label,
            ann_file=ann_file,
            **kwargs,
        )


@TRANSFORMS.register_module()
class MyLoadAnnotations(MMCVLoadAnnotations):
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
    path.write_text(text + block, encoding="utf-8")


def _config_text(
    class_file: str,
    dataset_type: str,
    data_root: Path,
    img_path: str,
    seg_path: str,
    ann_file: str,
    output_dir: Path,
    extra_model: dict[str, str] | None = None,
) -> str:
    extra_model_lines = ""
    if extra_model:
        extra_model_lines = "".join(f"\n    {key}={value!r}," for key, value in extra_model.items())

    return f"""_base_ = './base_config.py'

model = dict(name_path='./configs/{class_file}',{extra_model_lines}
)

dataset_type = {dataset_type!r}
data_root = {str(data_root)!r}

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 448), keep_ratio=True),
    dict(type='MyLoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path={img_path!r}, seg_map_path={seg_path!r}),
        ann_file={ann_file!r},
        pipeline=test_pipeline))

test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'], output_dir={str(output_dir)!r})
"""


def _write_repo_configs(repo: Path, roots: dict[str, Path]) -> None:
    if not (repo / "configs").exists():
        return

    _copy_proxy_e2_class_files(repo)
    _patch_custom_datasets(repo)
    lower = repo.name.lower()
    extras = {}
    if repo.name == "CorrCLIP":
        extras = {
            "context459": {"instance_mask_path": "data/region_masks/context"},
            "ade847": {"instance_mask_path": "data/region_masks/ade"},
        }
    elif repo.name == "ResCLIP":
        # ResCLIP's base config enables RCS/SFR, but the E2 configs need the
        # same mixing weights used by the corresponding official small-vocab
        # datasets; otherwise the upstream code raises NotImplementedError.
        extras = {
            "context459": {
                "temp_thd": 0.25,
                "delete_same_entity": True,
                "attn_rcs_weights": [2.0, 0.4],
                "attn_sfr_weights": [1.8, 0.7],
            },
            "ade847": {
                "temp_thd": 0.25,
                "delete_same_entity": True,
                "attn_rcs_weights": [2.0, 0.3],
                "attn_sfr_weights": [2.1, 0.7],
            },
        }

    configs = {
        "tfovos_context459_e2": _config_text(
            "cls_context459.txt",
            "PascalContext459Dataset",
            roots["context459"],
            "JPEGImages",
            "annotations_detectron2/pc459_val",
            "ImageSets/SegmentationContext/val.txt",
            ARTIFACT_ROOT / lower / "context459",
            extras.get("context459"),
        ),
        "tfovos_ade847_e2": _config_text(
            "cls_ade20k847.txt",
            "ADE20K847Dataset",
            roots["ade847"],
            "images_detectron2/validation",
            "annotations_detectron2/validation",
            "validation.txt",
            ARTIFACT_ROOT / lower / "ade847",
            extras.get("ade847"),
        ),
    }

    for name, text in configs.items():
        (repo / "configs" / f"cfg_{name}.py").write_text(text, encoding="utf-8")


def main() -> None:
    roots = {
        "context459": _prepare_context459(),
        "ade847": _prepare_ade847(),
    }
    repos = (SCLIP, NACLIP, RESCLIP, PROXY, CORR)
    for repo in repos:
        _write_repo_configs(repo, roots)
    print("Prepared official E2 configs under " + ", ".join(str(repo) for repo in repos))


if __name__ == "__main__":
    main()
