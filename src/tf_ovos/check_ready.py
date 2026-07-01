from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tf_ovos.adapters.registry import ADAPTERS, adapter_info
from tf_ovos.data import _resolve_path, read_jsonl, read_vocab

# Expected class counts per vocabulary name.  Warn if mismatch.
EXPECTED_VOCAB_COUNTS: dict[str, int] = {
    "voc20": 20,
    "context_59": 59,
    "context_459": 459,
    "ade20k_150": 150,
    "ade20k_847": 847,
    "coco_stuff_171": 171,
    # Appendix camouflage targets
    "ovcamo_75": 75,
    "ovcamo_61_unseen": 61,
}


@dataclass(frozen=True)
class CheckRow:
    level: str
    item: str
    message: str


def _add(rows: list[CheckRow], level: str, item: str, message: str) -> None:
    rows.append(CheckRow(level=level, item=item, message=message))


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data or {}


def _check_vocab(rows: list[CheckRow], name: str, path: Path, expected_count: int | None) -> None:
    if not path.exists():
        _add(rows, "FAIL", f"vocab:{name}", f"Missing vocab file: {path}")
        return
    labels = read_vocab(path)
    if not labels:
        _add(rows, "FAIL", f"vocab:{name}", f"No labels found in {path}")
        return
    if any("fill with" in label.lower() for label in labels):
        _add(rows, "FAIL", f"vocab:{name}", f"Vocab still looks like a placeholder: {path}")
        return
    if expected_count is not None and len(labels) != expected_count:
        _add(rows, "WARN", f"vocab:{name}", f"Expected {expected_count} labels, found {len(labels)}")
        return
    _add(rows, "OK", f"vocab:{name}", f"{len(labels)} labels")


def _check_manifest(
    rows: list[CheckRow],
    name: str,
    path: Path,
    task: str,
    vocab_labels: set[str] | None,
) -> None:
    if not path.exists():
        _add(rows, "WARN", f"manifest:{name}", f"Missing manifest: {path}")
        return
    try:
        records = read_jsonl(path)
    except Exception as exc:
        _add(rows, "FAIL", f"manifest:{name}", str(exc))
        return
    if not records:
        _add(rows, "FAIL", f"manifest:{name}", f"Manifest has no rows: {path}")
        return

    missing_paths: list[str] = []
    missing_labels: list[str] = []
    out_of_vocab: list[str] = []
    base_dir = path.parent
    for row in records:
        image_id = str(row.get("image_id", "<missing>"))
        for key in ("image_path", "mask_path"):
            value = row.get(key)
            if value is None or not _resolve_path(value, base_dir).exists():
                missing_paths.append(f"{image_id}:{key}")
        label = row.get("label")
        # class-aware tasks require per-sample label strings.
        if task == "class-aware" and not label:
            missing_labels.append(image_id)
        # semantic tasks encode labels in the map — no per-sample label required.
        if vocab_labels is not None and label and label not in vocab_labels:
            out_of_vocab.append(f"{image_id}:{label}")

    if missing_paths:
        preview = ", ".join(missing_paths[:5])
        _add(rows, "FAIL", f"manifest:{name}", f"Missing image/mask paths: {preview}")
    if missing_labels:
        preview = ", ".join(missing_labels[:5])
        _add(rows, "FAIL", f"manifest:{name}", f"Missing class-aware labels: {preview}")
    if out_of_vocab:
        preview = ", ".join(out_of_vocab[:5])
        _add(rows, "FAIL", f"manifest:{name}", f"Labels not found in vocab: {preview}")
    if not missing_paths and not missing_labels and not out_of_vocab:
        _add(rows, "OK", f"manifest:{name}", f"{len(records)} rows")


def _check_method_configs(rows: list[CheckRow], root: Path) -> None:
    method_dir = root / "configs" / "methods"
    if not method_dir.exists():
        _add(rows, "FAIL", "methods", f"Missing method config directory: {method_dir}")
        return
    configured = {path.stem for path in method_dir.glob("*.yaml")}
    registered = set(ADAPTERS)
    missing_configs = sorted(registered - configured)
    missing_adapters = sorted(configured - registered)
    for name in missing_configs:
        _add(rows, "WARN", f"method:{name}", "Adapter is registered but has no YAML config")
    for name in missing_adapters:
        _add(rows, "WARN", f"method:{name}", "YAML config exists but adapter is not registered")
    for info in adapter_info():
        status = "runnable" if info.runnable else "planned"
        _add(rows, "OK" if info.runnable else "WARN", f"adapter:{info.name}", status)


def check_ready(root: Path, benchmark_config: Path) -> list[CheckRow]:
    rows: list[CheckRow] = []
    cfg = _load_yaml(benchmark_config)
    if not cfg:
        _add(rows, "FAIL", "config", f"Could not load benchmark config: {benchmark_config}")
        return rows

    vocab_sets: dict[str, set[str]] = {}
    for name, rel_path in cfg.get("vocabularies", {}).items():
        path = root / rel_path
        expected = EXPECTED_VOCAB_COUNTS.get(name)
        _check_vocab(rows, name, path, expected)
        if path.exists():
            labels = set(read_vocab(path))
            if labels and all("fill with" not in label.lower() for label in labels):
                vocab_sets[name] = labels

    for name, dataset in cfg.get("datasets", {}).items():
        path = root / dataset["manifest"]
        task = dataset.get("task", "mask-only")
        # For class-aware tasks, validate labels against the dataset's own vocab.
        class_vocab: set[str] | None = None
        if task == "class-aware":
            vocab_path = dataset.get("vocab")
            if vocab_path:
                vname = Path(vocab_path).stem
                class_vocab = vocab_sets.get(vname)
        _check_manifest(rows, name, path, task, class_vocab)

    # Also check dataset-level vocab files.
    for name, dataset in cfg.get("datasets", {}).items():
        vocab_path = dataset.get("vocab")
        if vocab_path:
            vname = Path(vocab_path).stem
            path = root / vocab_path
            expected = EXPECTED_VOCAB_COUNTS.get(vname)
            _check_vocab(rows, f"dataset:{name}:vocab", path, expected)

    _check_method_configs(rows, root)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Check TF-OVOS readiness before moving to a GPU server.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark.yaml"))
    parser.add_argument("--strict", action="store_true", help="Exit nonzero on WARN or FAIL rows.")
    args = parser.parse_args()

    root = args.root.resolve()
    config = args.config if args.config.is_absolute() else root / args.config
    rows = check_ready(root, config)
    for row in rows:
        print(f"{row.level}\t{row.item}\t{row.message}")

    if args.strict and any(row.level in {"FAIL", "WARN"} for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
