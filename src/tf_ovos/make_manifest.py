from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from tf_ovos.data import as_output_path, read_jsonl, write_jsonl

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MASK_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def collect_files(root: Path, exts: tuple[str, ...], recursive: bool) -> dict[str, Path]:
    pattern = "**/*" if recursive else "*"
    files: dict[str, Path] = {}
    for path in root.glob(pattern):
        if path.is_file() and path.suffix.lower() in exts:
            key = path.stem
            if key in files:
                raise ValueError(f"Duplicate stem {key!r}: {files[key]} and {path}")
            files[key] = path
    return files


def load_label_map_files(root: Path | None, recursive: bool) -> dict[str, Path]:
    if root is None:
        return {}
    return collect_files(root, MASK_EXTS, recursive)


def load_label_map_source(path: Path | None, id_field: str, label_field: str) -> dict[str, str]:
    if path is None:
        return {}
    if path.suffix.lower() == ".jsonl":
        return {str(row[id_field]): str(row[label_field]) for row in read_jsonl(path)}
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(key): str(value) for key, value in data.items()}
        if isinstance(data, list):
            return {str(row[id_field]): str(row[label_field]) for row in data}
        raise ValueError(f"Unsupported JSON label file shape: {path}")
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return {str(row[id_field]): str(row[label_field]) for row in csv.DictReader(handle)}
    if path.suffix.lower() == ".txt":
        labels: dict[str, str] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    raise ValueError(f"Expected '<image_id> <label>' at {path}:{line_no}")
                labels[parts[0]] = " ".join(parts[1:])
        return labels
    raise ValueError(f"Unsupported label source: {path}")


def build_manifest(
    image_dir: Path,
    mask_dir: Path,
    out: Path,
    label_source: Path | None,
    recursive: bool,
    require_labels: bool,
    id_field: str,
    label_field: str,
) -> list[dict[str, Any]]:
    images = collect_files(image_dir, IMAGE_EXTS, recursive)
    masks = collect_files(mask_dir, MASK_EXTS, recursive)
    labels = load_label_map_source(label_source, id_field, label_field)

    rows: list[dict[str, Any]] = []
    missing_masks: list[str] = []
    missing_labels: list[str] = []
    for image_id in sorted(images):
        if image_id not in masks:
            missing_masks.append(image_id)
            continue
        label = labels.get(image_id)
        if require_labels and not label:
            missing_labels.append(image_id)
            continue
        rows.append(
            {
                "image_id": image_id,
                "image_path": as_output_path(images[image_id], out.parent),
                "mask_path": as_output_path(masks[image_id], out.parent),
                "label": label,
            }
        )

    if missing_masks:
        preview = ", ".join(missing_masks[:10])
        raise ValueError(f"Missing {len(missing_masks)} masks. First ids: {preview}")
    if missing_labels:
        preview = ", ".join(missing_labels[:10])
        raise ValueError(f"Missing {len(missing_labels)} labels. First ids: {preview}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a TF-OVOS dataset manifest.")
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--mask-dir", required=True, type=Path,
                        help="Binary mask dir (mask-only/class-aware) or label-map dir (semantic).")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--label-source", type=Path,
                        help="Per-image class label file (skip for semantic task).")
    parser.add_argument("--id-field", default="image_id")
    parser.add_argument("--label-field", default="label")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--require-labels", action="store_true")
    args = parser.parse_args()

    rows = build_manifest(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        out=args.out,
        label_source=args.label_source,
        recursive=args.recursive,
        require_labels=args.require_labels,
        id_field=args.id_field,
        label_field=args.label_field,
    )
    write_jsonl(rows, args.out)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
