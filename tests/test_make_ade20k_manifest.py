from pathlib import Path

import numpy as np
from PIL import Image

from tf_ovos.make_ade20k_manifest import build_ade20k_manifest, convert_label_map, decode_ade_segmentation


def _ade_rgb(raw_id: int) -> tuple[int, int, int]:
    return ((raw_id // 256) * 10, raw_id % 256, 7)


def test_decode_and_convert_label_map(tmp_path: Path) -> None:
    raw_ids = np.array([[2978, 2420], [0, 6500]], dtype=np.int32)
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    for y in range(raw_ids.shape[0]):
        for x in range(raw_ids.shape[1]):
            rgb[y, x] = _ade_rgb(int(raw_ids[y, x]))

    tmp = tmp_path / "ade_rgb_test.png"
    Image.fromarray(rgb, mode="RGB").save(tmp)
    decoded = decode_ade_segmentation(tmp)
    np.testing.assert_array_equal(decoded, raw_ids)

    converted = convert_label_map(decoded, {2978: 0, 2420: 2})
    np.testing.assert_array_equal(converted, np.array([[0, 2], [255, 255]], dtype=np.uint8))


def test_build_ade20k_manifest_writes_relative_paths(tmp_path: Path) -> None:
    root = tmp_path / "ADE20K_2021_17_01"
    image_root = root / "images" / "ADE" / "validation" / "scene"
    image_root.mkdir(parents=True)
    (root / "objects.txt").write_text(
        "Wordnet name \t Name index \t is object counts \t is part counts \t ADE names \t Attributes \t Has parts \t Is part of \n"
        "wall\t2978\t1\t0\twall\t\t\t\n"
        "sky\t2420\t1\t0\tsky\t\t\t\n",
        encoding="utf-8",
    )
    vocab = tmp_path / "ade150.txt"
    vocab.write_text("wall\nbuilding\nsky\n", encoding="utf-8")

    image_path = image_root / "ADE_val_00000001.jpg"
    Image.new("RGB", (2, 2), (10, 20, 30)).save(image_path)
    seg = np.array(
        [
            [_ade_rgb(2978), _ade_rgb(2420)],
            [_ade_rgb(0), _ade_rgb(2978)],
        ],
        dtype=np.uint8,
    )
    Image.fromarray(seg, mode="RGB").save(image_root / "ADE_val_00000001_seg.png")

    out = tmp_path / "data" / "manifests" / "ade20k150_val.jsonl"
    rows = build_ade20k_manifest(root, "validation", vocab, out, tmp_path / "data" / "processed" / "ade20k150")

    assert rows == [
        {
            "image_id": "scene/ADE_val_00000001",
            "image_path": "../../ADE20K_2021_17_01/images/ADE/validation/scene/ADE_val_00000001.jpg",
            "mask_path": "../processed/ade20k150/scene__ADE_val_00000001.png",
            "label": None,
        }
    ]
    mask = np.asarray(Image.open(tmp_path / "data" / "processed" / "ade20k150" / "scene__ADE_val_00000001.png"))
    np.testing.assert_array_equal(mask, np.array([[0, 2], [255, 0]], dtype=np.uint8))
