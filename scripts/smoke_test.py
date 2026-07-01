from __future__ import annotations

import json
import tempfile
from pathlib import Path

from PIL import Image

from tf_ovos.data import write_jsonl
from tf_ovos.eval import evaluate
from tf_ovos.make_manifest import build_manifest
from tf_ovos.make_shards import make_shards
from tf_ovos.merge_predictions import merge_predictions
from tf_ovos.run_method import run_method


def _write_mask(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (8, 8), value).save(path)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="tf_ovs_smoke_") as tmp:
        root = Path(tmp)
        image_dir = root / "data" / "raw" / "toy" / "images"
        mask_dir = root / "data" / "raw" / "toy" / "masks"
        image_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)

        labels = {"0001": "frog", "0002": "fish", "0003": "bird"}
        for image_id in labels:
            Image.new("RGB", (8, 8), (128, 64, 32)).save(image_dir / f"{image_id}.jpg")
            _write_mask(mask_dir / f"{image_id}.png", 255)

        labels_path = root / "data" / "raw" / "toy" / "labels.json"
        labels_path.write_text(json.dumps(labels), encoding="utf-8")

        manifest = root / "data" / "manifests" / "toy.jsonl"
        rows = build_manifest(
            image_dir=image_dir,
            mask_dir=mask_dir,
            out=manifest,
            label_source=labels_path,
            recursive=False,
            require_labels=True,
            id_field="image_id",
            label_field="label",
        )
        write_jsonl(rows, manifest)

        shard_dir = root / "data" / "manifests" / "shards" / "toy"
        shard_paths = make_shards(manifest, num_shards=2, out_dir=shard_dir, strategy="round-robin")

        run_root = root / "runs" / "debug_copy_gt" / "toy"
        for shard_path in shard_paths:
            run_method(
                method="debug_copy_gt",
                manifest=shard_path,
                vocab=None,
                out_dir=run_root / "shards" / shard_path.stem,
                skip_existing=False,
            )

        predictions = run_root / "predictions.jsonl"
        merged = merge_predictions(manifest, run_root / "shards", predictions)
        write_jsonl(merged, predictions)
        result = evaluate(manifest, predictions, task="class-aware", threshold=0.5)

        if result["num_samples"] != 3:
            raise RuntimeError(f"Unexpected smoke-test sample count: {result['num_samples']}")
        print("TF-OVS smoke test passed.")


if __name__ == "__main__":
    main()
