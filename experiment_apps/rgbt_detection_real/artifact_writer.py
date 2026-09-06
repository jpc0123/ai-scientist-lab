from __future__ import annotations

from typing import Any

import json
from pathlib import Path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def write_manifest(output_dir: Path, *, mode: str) -> dict[str, Any]:
    needs_train_curve = mode in {"smoke_train", "fast_eval"}
    # Formal / stand-in real baseline uses .pt; keep training_history for smoke.
    if mode == "fast_eval" and not (output_dir / "training_history.csv").exists():
        needs_train_curve = False
    payload = {
        "schema_version": "1.0",
        "baseline_key": "dfine_s",
        "files": [
            {"path": "metrics.json", "artifact_type": "metrics"},
            {"path": "dataset_report.json", "artifact_type": "dataset_report"},
            {"path": "execution.json", "artifact_type": "execution"},
            {"path": "model_summary.json", "artifact_type": "summary"},
            {"path": "resource_usage.json", "artifact_type": "resource_usage"},
            {"path": "combined.log", "artifact_type": "log"},
        ],
        "artifacts": [
            {"path": "metrics.json", "type": "metrics", "required": True},
            {"path": "dataset_report.json", "type": "dataset_report", "required": True},
            {
                "path": "training_history.csv",
                "type": "training_curve",
                "required": needs_train_curve and mode == "smoke_train",
            },
            {
                "path": "checkpoint/last.pt",
                "type": "checkpoint",
                "required": mode in {"smoke_train", "fast_eval"},
            },
            {"path": "model_summary.json", "type": "summary", "required": True},
            {
                "path": "resource_usage.json",
                "type": "resource_usage",
                "required": True,
            },
            {
                "path": "sample_predictions.json",
                "type": "predictions",
                "required": mode in {"smoke_train", "fast_eval"},
            },
            {"path": "combined.log", "type": "log", "required": False},
        ],
    }
    if needs_train_curve and mode == "smoke_train":
        payload["files"].append(
            {"path": "training_history.csv", "artifact_type": "training_curve"}
        )
    payload["files"].append(
        {"path": "checkpoint/last.pt", "artifact_type": "checkpoint"}
    )
    write_json(output_dir / "artifact_manifest.json", payload)
    return payload
