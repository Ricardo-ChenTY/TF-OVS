import json
from pathlib import Path

from PIL import Image

from tf_ovos.make_manifest import build_manifest
from tf_ovos.make_shards import make_shards
from tf_ovos.merge_predictions import merge_predictions
from tf_ovos.run_benchmark import run_dataset
from tf_ovos.run_method import run_method
from tf_ovos.summarize_results import collect_method


def write_mask(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (4, 4), value).save(path)


def test_manifest_shard_run_merge_eval_paths(tmp_path):
    image_dir = tmp_path / "raw" / "images"
    mask_dir = tmp_path / "raw" / "masks"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    for image_id in ["a", "b", "c"]:
        Image.new("RGB", (4, 4), (255, 0, 0)).save(image_dir / f"{image_id}.jpg")
        write_mask(mask_dir / f"{image_id}.png", 255)

    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps({"a": "frog", "b": "fish", "c": "bird"}), encoding="utf-8")
    manifest = tmp_path / "data" / "manifests" / "toy.jsonl"
    rows = build_manifest(image_dir, mask_dir, manifest, labels_path, False, True, "image_id", "label")
    assert len(rows) == 3
    from tf_ovos.data import write_jsonl

    write_jsonl(rows, manifest)

    shard_dir = tmp_path / "data" / "manifests" / "shards" / "toy"
    shard_paths = make_shards(manifest, 2, shard_dir, "round-robin")
    assert len(shard_paths) == 2

    for shard_path in shard_paths:
        out_dir = tmp_path / "runs" / "debug_copy_gt" / shard_path.stem
        run_method("debug_copy_gt", shard_path, None, out_dir, False)
        target_dir = tmp_path / "runs" / "debug_copy_gt" / "toy" / "shards" / shard_path.stem
        target_dir.mkdir(parents=True)
        (out_dir / "predictions.jsonl").replace(target_dir / "predictions.jsonl")
        pred_masks = target_dir / "pred_masks"
        pred_masks.mkdir()
        for mask in (out_dir / "pred_masks").glob("*.png"):
            mask.replace(pred_masks / mask.name)

    merged = merge_predictions(
        manifest,
        tmp_path / "runs" / "debug_copy_gt" / "toy" / "shards",
        tmp_path / "runs" / "debug_copy_gt" / "toy" / "predictions.jsonl",
    )
    assert [row["image_id"] for row in merged] == ["a", "b", "c"]


def test_run_dataset_writes_runtime_metrics_and_summary(tmp_path):
    image_dir = tmp_path / "raw" / "images"
    mask_dir = tmp_path / "raw" / "masks"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    for image_id in ["a", "b"]:
        Image.new("RGB", (4, 4), (255, 0, 0)).save(image_dir / f"{image_id}.jpg")
        write_mask(mask_dir / f"{image_id}.png", 255)

    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps({"a": "frog", "b": "fish"}), encoding="utf-8")
    manifest = tmp_path / "data" / "manifests" / "toy.jsonl"
    rows = build_manifest(image_dir, mask_dir, manifest, labels_path, False, True, "image_id", "label")
    from tf_ovos.data import write_jsonl

    write_jsonl(rows, manifest)
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("frog\nfish\n", encoding="utf-8")

    outputs = run_dataset(
        method="debug_copy_gt",
        dataset_name="toy",
        manifest=manifest,
        task="class-aware",
        vocab=vocab,
        run_root=tmp_path / "runs",
        num_shards=2,
        shard_strategy="round-robin",
        skip_existing=False,
        evaluate_predictions=True,
    )

    assert outputs["predictions"].exists()
    assert outputs["metrics"].exists()
    assert (tmp_path / "runs" / "debug_copy_gt" / "toy" / "shards" / "part-000" / "runtime.json").exists()

    cfg = {"datasets": {"toy": {"manifest": str(manifest), "task": "class-aware"}}}
    summary = collect_method("debug_copy_gt", cfg, tmp_path / "runs")
    assert summary["datasets"][0]["class_aware_metrics.cIoU"] == 1.0
    assert summary["e2_ambiguity"][0]["Loc@0.5"] == 1.0
