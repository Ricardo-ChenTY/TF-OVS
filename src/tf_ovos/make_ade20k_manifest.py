from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from tf_ovos.data import as_output_path, read_vocab, write_jsonl

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
VOID_LABEL = 255
ADE_IMAGE_RE = re.compile(r"^ADE_(?:train|val)_\d+$")


def _normalize_label(label: str) -> str:
    label = label.strip().lower().replace("_", " ").replace("\'", "")
    label = re.sub(r"\s+", " ", label)
    return label


def _label_aliases(label: str) -> set[str]:
    aliases = {_normalize_label(label)}
    aliases.update(_normalize_label(part) for part in label.split(","))
    return {alias for alias in aliases if alias}


def _read_objects(objects_path: Path) -> dict[int, set[str]]:
    raw_to_names: dict[int, set[str]] = {}
    with objects_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.rstrip("\n")
            if not line or line_no == 1:
                continue
            columns = line.split("\t")
            if len(columns) < 5:
                continue
            try:
                raw_id = int(columns[1])
            except ValueError:
                continue
            names = set()
            names.update(_label_aliases(columns[0]))
            names.update(_label_aliases(columns[4]))
            raw_to_names[raw_id] = names
    return raw_to_names


def build_raw_to_vocab_map(objects_path: Path, vocab_path: Path) -> dict[int, int]:
    vocab = read_vocab(vocab_path)
    alias_to_index: dict[str, int] = {}
    for index, label in enumerate(vocab):
        for alias in _label_aliases(label):
            alias_to_index.setdefault(alias, index)

    raw_to_vocab: dict[int, int] = {}
    for raw_id, aliases in _read_objects(objects_path).items():
        for alias in aliases:
            if alias in alias_to_index:
                raw_to_vocab[raw_id] = alias_to_index[alias]
                break
    return raw_to_vocab


def decode_ade_segmentation(seg_path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(seg_path).convert("RGB"))
    return (rgb[..., 0].astype(np.int32) // 10) * 256 + rgb[..., 1].astype(np.int32)


def convert_label_map(raw_map: np.ndarray, raw_to_vocab: dict[int, int], void_label: int = VOID_LABEL) -> np.ndarray:
    max_vocab_idx = max(raw_to_vocab.values()) if raw_to_vocab else 0
    dtype = np.uint16 if (max_vocab_idx > 254 or void_label > 255) else np.uint8
    label_map = np.full(raw_map.shape, void_label, dtype=dtype)
    for raw_id in np.unique(raw_map):
        if raw_id == 0:
            continue
        vocab_index = raw_to_vocab.get(int(raw_id))
        if vocab_index is not None:
            label_map[raw_map == raw_id] = vocab_index
    return label_map


def image_id_from_path(image_path: Path, image_root: Path) -> str:
    return image_path.relative_to(image_root).with_suffix("").as_posix()


def build_ade20k_manifest(
    root: Path,
    split: str,
    vocab: Path,
    out: Path,
    mask_dir: Path,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    image_root = root / "images" / "ADE" / split
    objects_path = root / "objects.txt"
    if not image_root.exists():
        raise FileNotFoundError(f"Missing ADE split image directory: {image_root}")
    if not objects_path.exists():
        raise FileNotFoundError(f"Missing ADE objects file: {objects_path}")

    raw_to_vocab = build_raw_to_vocab_map(objects_path, vocab)
    if not raw_to_vocab:
        raise ValueError(f"No ADE object names matched vocab labels: {vocab}")

    image_paths = sorted(
        path
        for path in image_root.glob("**/*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTS
        and ADE_IMAGE_RE.match(path.stem)
        and not path.name.endswith("_seg.png")
        and "_parts_" not in path.name
    )
    if limit is not None:
        image_paths = image_paths[:limit]

    rows: list[dict[str, Any]] = []
    missing_seg: list[str] = []
    mask_dir.mkdir(parents=True, exist_ok=True)
    for image_path in image_paths:
        image_id = image_id_from_path(image_path, image_root)
        seg_path = image_path.with_name(f"{image_path.stem}_seg.png")
        if not seg_path.exists():
            missing_seg.append(image_id)
            continue

        label_map = convert_label_map(decode_ade_segmentation(seg_path), raw_to_vocab)
        mask_path = mask_dir / f"{image_id.replace('/', '__')}.png"
        if label_map.dtype == np.uint16:
            Image.fromarray(label_map, mode="I;16").save(mask_path)
        else:
            Image.fromarray(label_map, mode="L").save(mask_path)
        rows.append(
            {
                "image_id": image_id,
                "image_path": as_output_path(image_path, out.parent),
                "mask_path": as_output_path(mask_path, out.parent),
                "label": None,
            }
        )

    if missing_seg:
        preview = ", ".join(missing_seg[:10])
        raise ValueError(f"Missing {len(missing_seg)} ADE segmentation files. First ids: {preview}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a TF-OVOS semantic manifest from raw ADE20K files.")
    parser.add_argument("--root", required=True, type=Path, help="ADE20K_2021_17_01 dataset root.")
    parser.add_argument("--split", default="validation", choices=["training", "validation"])
    parser.add_argument("--vocab", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--mask-dir", required=True, type=Path, help="Output dir for converted label-map PNGs.")
    parser.add_argument("--limit", type=int, help="Only convert the first N images, useful for debugging.")
    args = parser.parse_args()

    rows = build_ade20k_manifest(
        root=args.root,
        split=args.split,
        vocab=args.vocab,
        out=args.out,
        mask_dir=args.mask_dir,
        limit=args.limit,
    )
    write_jsonl(rows, args.out)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
