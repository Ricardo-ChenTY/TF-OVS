from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tf_ovos.adapters.base import write_predictions
from tf_ovos.adapters.registry import ADAPTERS, adapter_info
from tf_ovos.data import load_manifest, read_vocab


def run_method(
    method: str,
    manifest: Path,
    vocab: Path | None,
    out_dir: Path,
    skip_existing: bool,
    task: str | None = None,
    num_classes: int | None = None,
) -> Path:
    if method not in ADAPTERS:
        available = ", ".join(sorted(ADAPTERS))
        raise ValueError(f"Unknown method {method!r}. Available adapters: {available}")

    adapter_cls = ADAPTERS[method]
    if not getattr(adapter_cls, "runnable", True):
        hint = getattr(adapter_cls, "setup_hint", "")
        raise NotImplementedError(f"Adapter {method!r} is not implemented yet. {hint}")

    predictions_path = out_dir / "predictions.jsonl"
    if skip_existing and predictions_path.exists():
        print(f"Skipping existing predictions: {predictions_path}")
        return predictions_path

    samples = load_manifest(manifest)
    vocabulary = read_vocab(vocab) if vocab else []
    adapter = adapter_cls()
    adapter.configure(task=task, num_classes=num_classes)
    start = time.perf_counter()
    predictions = adapter.predict_many(samples, vocabulary, out_dir)
    elapsed = time.perf_counter() - start
    write_predictions(predictions, predictions_path)
    runtime = {
        "method": method,
        "manifest": str(manifest),
        "vocab": str(vocab) if vocab else None,
        "num_samples": len(samples),
        "task": task,
        "num_classes": num_classes,
        "wall_time_sec": elapsed,
        "sec_per_image": elapsed / max(len(samples), 1),
        "model_calls": None,
        "peak_memory_mb": None,
        "adapter": adapter_cls.__name__,
    }
    (out_dir / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(predictions)} predictions to {predictions_path}")
    return predictions_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a TF-OVOS method adapter.")
    parser.add_argument("--method")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--vocab", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--task", choices=["semantic", "mask-only", "class-aware"])
    parser.add_argument("--num-classes", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--list-methods", action="store_true")
    args = parser.parse_args()

    if args.list_methods:
        for info in adapter_info():
            status = "runnable" if info.runnable else "planned"
            print(f"{info.name}\t{status}")
        return

    if not args.method or not args.manifest or not args.out:
        parser.error("--method, --manifest, and --out are required unless --list-methods is used")
    run_method(args.method, args.manifest, args.vocab, args.out, args.skip_existing, args.task, args.num_classes)


if __name__ == "__main__":
    main()
