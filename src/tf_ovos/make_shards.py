from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from tf_ovos.data import _resolve_path, as_output_path, read_jsonl, write_jsonl


def rewrite_manifest_row(row: dict[str, Any], source_base: Path, target_base: Path) -> dict[str, Any]:
    rewritten = dict(row)
    rewritten["image_path"] = as_output_path(_resolve_path(row["image_path"], source_base), target_base)
    rewritten["mask_path"] = as_output_path(_resolve_path(row["mask_path"], source_base), target_base)
    return rewritten


def make_shards(manifest: Path, num_shards: int, out_dir: Path, strategy: str) -> list[Path]:
    rows = read_jsonl(manifest)
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    shards: list[list[dict[str, Any]]] = [[] for _ in range(num_shards)]

    if strategy == "round-robin":
        for idx, row in enumerate(rows):
            shards[idx % num_shards].append(row)
    elif strategy == "contiguous":
        per_shard = (len(rows) + num_shards - 1) // num_shards
        for shard_idx in range(num_shards):
            start = shard_idx * per_shard
            shards[shard_idx].extend(rows[start : start + per_shard])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    out_paths: list[Path] = []
    for shard_idx, shard_rows in enumerate(shards):
        shard_path = out_dir / f"part-{shard_idx:03d}.jsonl"
        rewritten = [rewrite_manifest_row(row, manifest.parent, shard_path.parent) for row in shard_rows]
        write_jsonl(rewritten, shard_path)
        out_paths.append(shard_path)
    return out_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a manifest into resumable shards.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--num-shards", required=True, type=int)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--strategy", choices=["round-robin", "contiguous"], default="round-robin")
    args = parser.parse_args()

    paths = make_shards(args.manifest, args.num_shards, args.out_dir, args.strategy)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
