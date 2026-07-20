"""Normalize execution results for Local / Remote schema comparisons."""

from __future__ import annotations

from typing import Any


def normalize_execution_result(result: dict[str, Any]) -> dict[str, Any]:
    """Drop runner-specific absolute paths / ids; keep research schema fields."""
    metrics = dict(result.get("metrics") or {})
    training = dict(metrics.get("training") or {})
    # Keep comparable fields only.
    normalized_metrics = {
        "schema_version": metrics.get("schema_version"),
        "project_id": metrics.get("project_id"),
        "node_id": metrics.get("node_id"),
        "task_type": metrics.get("task_type"),
        "primary_metric": metrics.get("primary_metric"),
        "metrics": {
            key: value
            for key, value in dict(metrics.get("metrics") or {}).items()
            if key
            not in {
                "duration_seconds",
                "peak_gpu_memory_mb",
            }
        },
        "training": {
            key: value
            for key, value in training.items()
            if key
            not in {
                "duration_seconds",
                "device",
            }
        },
        "evaluation_scope": metrics.get("evaluation_scope"),
        "claim_level": metrics.get("claim_level"),
        "execution_mode": metrics.get("execution_mode"),
        "status": metrics.get("status"),
    }
    artifact_names = sorted(
        {
            item.get("relative_path") or item.get("path")
            for item in (result.get("artifacts") or [])
            if isinstance(item, dict)
        }
        - {None}
    )
    return {
        "status": result.get("status"),
        "metrics": normalized_metrics,
        "artifact_names": artifact_names,
        "error_type": (result.get("error") or {}).get("error_type")
        if isinstance(result.get("error"), dict)
        else None,
    }
