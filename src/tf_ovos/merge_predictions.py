from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from tf_ovos.data import _resolve_path, as_output_path, load_manifest, read_jsonl, read_vocab, write_jsonl


def find_prediction_files(shards_dir: Path) -> list[Path]:
    return sorted(shards_dir.glob("part-*/predictions.jsonl")) + sorted(shards_dir.glob("part-*.jsonl"))


def rewrite_prediction_row(row: dict[str, Any], source_base: Path, target_base: Path) -> dict[str, Any]:
    rewritten = dict(row)
    rewritten["mask_path"] = as_output_path(_resolve_path(row["mask_path"], source_base), target_base)
    return rewritten


def merge_predictions(
    manifest: Path,
    shards_dir: Path,
    out: Path,
    vocab: Path | None = None,
    allow_missing: bool = False,
) -> list[dict[str, Any]]:
    samples = load_manifest(manifest)
    expected_ids = [sample.image_id for sample in samples]
    expected_set = set(expected_ids)
    allowed_labels = set(read_vocab(vocab)) if vocab else None

    rows_by_id: dict[str, dict[str, Any]] = {}
    shard_files = find_prediction_files(shards_dir)
    if not shard_files:
        raise ValueError(f"No shard prediction files found under {shards_dir}")

    for shard_file in shard_files:
        for row in read_jsonl(shard_file):
            image_id = str(row["image_id"])
            if image_id in rows_by_id:
                raise ValueError(f"Duplicate prediction for image_id={image_id!r}")
            if image_id not in expected_set:
                raise ValueError(f"Unexpected prediction image_id={image_id!r}")
            label = row.get("label")
            if allowed_labels is not None and label is not None and label not in allowed_labels:
                raise ValueError(f"Label {label!r} for image_id={image_id!r} is not in vocab")
            rows_by_id[image_id] = rewrite_prediction_row(row, shard_file.parent, out.parent)

    missing = [image_id for image_id in expected_ids if image_id not in rows_by_id]
    if missing and not allow_missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"Missing {len(missing)} predictions. First ids: {preview}")

    return [rows_by_id[image_id] for image_id in expected_ids if image_id in rows_by_id]


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge prediction shards into one predictions.jsonl.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--shards-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--vocab", type=Path)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    rows = merge_predictions(args.manifest, args.shards_dir, args.out, args.vocab, args.allow_missing)
    write_jsonl(rows, args.out)
    print(f"Wrote {len(rows)} merged predictions to {args.out}")


if __name__ == "__main__":
    main()
