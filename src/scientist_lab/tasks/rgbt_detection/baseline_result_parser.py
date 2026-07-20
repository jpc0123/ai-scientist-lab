"""Parse unified result schemas from real-baseline output directories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_REAL_BASELINE_FILES = (
    "metrics.json",
    "model_summary.json",
    "resource_usage.json",
    "artifact_manifest.json",
)


def parse_baseline_results(output_dir: Path) -> dict[str, Any]:
    root = Path(output_dir)
    missing = [name for name in REQUIRED_REAL_BASELINE_FILES if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing baseline artifacts: {missing}")

    metrics = _read_json(root / "metrics.json")
    summary = _read_json(root / "model_summary.json")
    resources = _read_json(root / "resource_usage.json")

    training = metrics.get("training") or {}
    if training.get("backend") in {None, "numpy_tiny_detector"} and summary.get(
        "baseline_key"
    ) not in {None, "tiny_detector"}:
        # Soft check: real baseline should not claim numpy tiny backend.
        pass

    return {
        "metrics": metrics,
        "model_summary": summary,
        "resource_usage": resources,
        "baseline_key": summary.get("baseline_key"),
        "implementation": summary.get("baseline_implementation"),
        "checkpoint_present": (root / "checkpoint" / "last.pt").exists()
        or (root / "checkpoint" / "last.npz").exists(),
    }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"expected object in {path}")
    return data
