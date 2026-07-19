from __future__ import annotations

from typing import Any

from validate_dataset import write_json


def write_manifest(output_dir, *, mode: str) -> dict[str, Any]:
    payload = {
        "schema_version": "1.0",
        "files": [
            {"path": "metrics.json", "artifact_type": "metrics"},
            {"path": "dataset_report.json", "artifact_type": "dataset_report"},
            {"path": "execution.json", "artifact_type": "execution"},
            {"path": "combined.log", "artifact_type": "log"},
        ],
        "artifacts": [
            {"path": "metrics.json", "type": "metrics", "required": True},
            {"path": "dataset_report.json", "type": "dataset_report", "required": True},
            {
                "path": "training_history.csv",
                "type": "training_curve",
                "required": mode != "validate_data",
            },
            {
                "path": "checkpoint/last.npz",
                "type": "checkpoint",
                "required": mode != "validate_data",
            },
            {
                "path": "checkpoint/last_smoke.txt",
                "type": "checkpoint_meta",
                "required": False,
            },
            {"path": "model_summary.json", "type": "summary", "required": False},
            {
                "path": "sample_predictions.json",
                "type": "predictions",
                "required": False,
            },
            {"path": "combined.log", "type": "log", "required": False},
        ],
    }
    if mode != "validate_data":
        payload["files"].extend(
            [
                {"path": "training_history.csv", "artifact_type": "training_curve"},
                {"path": "checkpoint/last.npz", "artifact_type": "checkpoint"},
                {"path": "model_summary.json", "artifact_type": "summary"},
            ]
        )
    write_json(output_dir / "artifact_manifest.json", payload)
    return payload
