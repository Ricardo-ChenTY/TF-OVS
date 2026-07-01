from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from tf_ovos.data import read_jsonl, write_jsonl
from tf_ovos.eval import evaluate
from tf_ovos.make_shards import make_shards, rewrite_manifest_row
from tf_ovos.merge_predictions import merge_predictions
from tf_ovos.run_method import run_method


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _dataset_vocab(cfg: dict[str, Any], dataset: dict[str, Any]) -> Path | None:
    """Return vocab path: dataset-level 'vocab' key takes priority."""
    if "vocab" in dataset:
        return Path(dataset["vocab"])
    # Legacy fallback for mask-only / class-aware appendix datasets.
    vocabularies = cfg.get("vocabularies", {})
    task = dataset.get("task", "mask-only")
    if task == "class-aware":
        value = vocabularies.get("ovcamo_61_unseen")
    else:
        value = vocabularies.get("ovcamo_75")
    return Path(value) if value else None


def _write_limited_manifest(source: Path, out: Path, limit: int) -> Path:
    rows = read_jsonl(source)[:limit]
    if not rows:
        raise ValueError(f"Cannot create a limited manifest from an empty file: {source}")
    rewritten = [rewrite_manifest_row(row, source.parent, out.parent) for row in rows]
    write_jsonl(rewritten, out)
    return out


def run_dataset(
    method: str,
    dataset_name: str,
    manifest: Path,
    task: str,
    vocab: Path | None,
    run_root: Path,
    num_shards: int,
    shard_strategy: str,
    skip_existing: bool,
    evaluate_predictions: bool,
    num_classes: int | None = None,
    void_label: int = 255,
    prediction_label_offset: int = 0,
) -> dict[str, Path]:
    dataset_root = run_root / method / dataset_name
    manifest_shards_dir = dataset_root / "manifest_shards"
    shards_dir = dataset_root / "shards"
    shard_paths = make_shards(manifest, num_shards, manifest_shards_dir, shard_strategy)

    for shard_path in shard_paths:
        run_method(
            method=method,
            manifest=shard_path,
            vocab=vocab,
            out_dir=shards_dir / shard_path.stem,
            skip_existing=skip_existing,
            task=task,
            num_classes=num_classes,
        )

    predictions_path = dataset_root / "predictions.jsonl"
    rows = merge_predictions(
        manifest=manifest,
        shards_dir=shards_dir,
        out=predictions_path,
        vocab=vocab,
        allow_missing=False,
    )
    write_jsonl(rows, predictions_path)
    print(f"Wrote {len(rows)} merged predictions to {predictions_path}")

    metrics_path = dataset_root / "metrics.json"
    if evaluate_predictions:
        result = evaluate(
            manifest,
            predictions_path,
            task=task,
            threshold=0.5,
            num_classes=num_classes,
            void_label=void_label,
            prediction_label_offset=prediction_label_offset,
        )
        metrics_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote metrics to {metrics_path}")

    return {
        "dataset_root": dataset_root,
        "shards_dir": shards_dir,
        "predictions": predictions_path,
        "metrics": metrics_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one TF-OVOS method across benchmark datasets.")
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark.yaml"))
    parser.add_argument("--method", required=True)
    parser.add_argument("--dataset", action="append", help="Dataset key from benchmark.yaml. Repeat or omit for all.")
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-strategy", choices=["round-robin", "contiguous"], default="round-robin")
    parser.add_argument("--vocab", type=Path, help="Override vocab path for every dataset.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--limit", type=int, help="Run only the first N manifest rows for smoke tests.")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    datasets = cfg.get("datasets", {})
    selected = args.dataset or list(datasets)
    if not selected:
        raise ValueError(f"No datasets found in {args.config}")

    for dataset_name in selected:
        if dataset_name not in datasets:
            available = ", ".join(sorted(datasets))
            raise ValueError(f"Unknown dataset {dataset_name!r}. Available: {available}")
        dataset = datasets[dataset_name]
        task = dataset.get("task", "mask-only")
        vocab = args.vocab or _dataset_vocab(cfg, dataset)
        num_classes = dataset.get("num_classes")
        void_label = dataset.get("void_label", 255)
        prediction_label_offset = int(dataset.get("prediction_label_offset", 0))
        manifest = Path(dataset["manifest"])
        if args.limit is not None:
            manifest = _write_limited_manifest(
                manifest,
                args.run_root / args.method / dataset_name / "limited_manifest.jsonl",
                args.limit,
            )
        run_dataset(
            method=args.method,
            dataset_name=dataset_name,
            manifest=manifest,
            task=task,
            vocab=vocab,
            run_root=args.run_root,
            num_shards=args.num_shards,
            shard_strategy=args.shard_strategy,
            skip_existing=args.skip_existing,
            evaluate_predictions=not args.no_eval,
            num_classes=num_classes,
            void_label=void_label,
            prediction_label_offset=prediction_label_offset,
        )


if __name__ == "__main__":
    main()
