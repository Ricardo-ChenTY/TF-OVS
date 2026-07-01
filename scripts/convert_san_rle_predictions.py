#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Detectron2/SAN sem_seg_predictions.json RLEs to dense PNG label maps.")
    parser.add_argument("--pred-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--void-label", type=int, default=65535)
    parser.add_argument("--suffix", default=".png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    entries = json.loads(args.pred_json.read_text())
    by_file: dict[str, list[dict[str, object]]] = defaultdict(list)
    for entry in entries:
        by_file[str(entry["file_name"])].append(entry)

    for idx, (file_name, file_entries) in enumerate(sorted(by_file.items()), start=1):
        first_seg = file_entries[0]["segmentation"]
        height, width = first_seg["size"]
        dense = np.full((height, width), args.void_label, dtype=np.uint16)
        for entry in file_entries:
            category_id = int(entry["category_id"])
            rle = entry["segmentation"]
            if isinstance(rle.get("counts"), str):
                rle = {"size": rle["size"], "counts": rle["counts"].encode("utf-8")}
            mask = mask_utils.decode(rle).astype(bool)
            dense[mask] = category_id
        out_name = Path(file_name).stem + args.suffix
        Image.fromarray(dense, mode="I;16").save(args.out_dir / out_name)
        if idx % 500 == 0:
            print(f"converted {idx}/{len(by_file)}", flush=True)
    print(f"converted {len(by_file)} files to {args.out_dir}")


if __name__ == "__main__":
    main()
