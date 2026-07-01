from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Sample:
    image_id: str
    image_path: Path
    mask_path: Path       # binary mask (mask-only/class-aware) or label map (semantic)
    label: str | None = None


@dataclass(frozen=True)
class Prediction:
    image_id: str
    mask_path: Path       # binary mask (mask-only/class-aware) or label map (semantic)
    label: str | None
    score: float | None = None
    metadata: dict[str, Any] | None = None


def _resolve_path(path: str | Path, base_dir: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return base_dir / candidate


def as_output_path(path: str | Path, base_dir: str | Path) -> str:
    path = Path(path).resolve()
    base_dir = Path(base_dir).resolve()
    try:
        return Path(os.path.relpath(path, base_dir)).as_posix()
    except ValueError:
        return str(path)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_manifest(path: str | Path) -> list[Sample]:
    path = Path(path)
    base_dir = path.parent
    samples: list[Sample] = []
    for row in read_jsonl(path):
        samples.append(
            Sample(
                image_id=str(row["image_id"]),
                image_path=_resolve_path(row["image_path"], base_dir),
                mask_path=_resolve_path(row["mask_path"], base_dir),
                label=row.get("label"),
            )
        )
    return samples


def load_predictions(path: str | Path) -> dict[str, Prediction]:
    path = Path(path)
    base_dir = path.parent
    predictions: dict[str, Prediction] = {}
    for row in read_jsonl(path):
        pred = Prediction(
            image_id=str(row["image_id"]),
            mask_path=_resolve_path(row["mask_path"], base_dir),
            label=row.get("label"),
            score=row.get("score"),
            metadata=row.get("metadata"),
        )
        if pred.image_id in predictions:
            raise ValueError(f"Duplicate prediction for image_id={pred.image_id!r}")
        predictions[pred.image_id] = pred
    return predictions


def read_vocab(path: str | Path) -> list[str]:
    labels: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                labels.append(line)
    return labels


def iter_missing_predictions(samples: Iterable[Sample], predictions: dict[str, Prediction]) -> list[str]:
    return [sample.image_id for sample in samples if sample.image_id not in predictions]
