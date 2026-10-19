"""Strict source selection for diagnostic tables; never average duplicate runs."""
from __future__ import annotations

import csv
from pathlib import Path


def read_unique_diagnostics(paths: list[Path]) -> list[dict[str, str]]:
    rows: dict[tuple[str, str, str], dict[str, str]] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (row["method"], row["dataset"], row.get("run_id") or "legacy_unspecified")
                if key in rows:
                    old = rows[key]
                    fields = set(old) & set(row) - {"source_path"}
                    if any(old[f] != row[f] for f in fields):
                        raise ValueError(f"Conflicting diagnostic rows for {key}: {old['source_path']} vs {path}")
                    continue
                rows[key] = {**row, "source_path": str(path)}
    # Multiple distinct runs need explicit selection, not an implicit mean.
    cells: set[tuple[str, str]] = set()
    for method, dataset, run_id in rows:
        if (method, dataset) in cells:
            raise ValueError(f"Multiple runs for {(method, dataset)}; select a run explicitly")
        cells.add((method, dataset))
    return list(rows.values())
