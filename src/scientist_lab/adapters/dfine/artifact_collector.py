"""Collect expected artifacts from a run directory."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def collect_artifacts(
    output_dir: Path | str,
    expected: list[str] | None = None,
) -> dict[str, Any]:
    root = Path(output_dir)
    expected = list(expected or ["metrics.json", "checkpoint_selection.json"])
    present: list[str] = []
    missing: list[str] = []
    for name in expected:
        path = root / name
        if path.is_file():
            present.append(name)
        else:
            missing.append(name)
    extra = []
    if root.is_dir():
        extra = sorted(
            p.name
            for p in root.iterdir()
            if p.is_file() and p.name not in expected
        )
    return {
        "paths": present,
        "missing_expected": missing,
        "extra": extra,
        "output_dir": str(root),
    }
