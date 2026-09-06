"""Bridge smoke_train completions into multi-seed fast_eval contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scientist_lab.tasks.rgbt_detection.decision_rules import is_smoke_detection_context

DEFAULT_FAST_EVAL_SEEDS = [42, 43, 44]


def build_fast_eval_contract(
    smoke_contract: dict[str, Any],
    *,
    checkpoint_source: str,
    node_id: str | None = None,
    parent_node_id: str | None = None,
) -> dict[str, Any]:
    """Derive a fast_eval contract from a completed smoke_train contract."""
    if not checkpoint_source or not str(checkpoint_source).strip():
        raise ValueError("checkpoint_source is required")

    base = dict(smoke_contract or {})
    task_type = base.get("task_type")
    mode = base.get("execution_mode")
    claim = (base.get("task_config") or {}).get("claim_level")
    if not is_smoke_detection_context(
        task_type=task_type, execution_mode=mode, claim_level=claim
    ):
        raise ValueError(
            "prepare-fast-eval expects an rgbt_detection smoke/debug source contract"
        )

    source_node = str(base.get("node_id") or "rgbt_node")
    out_node = node_id or f"{source_node}_fast_eval"
    parameters = dict(base.get("parameters") or {})
    parameters["epochs"] = 0
    parameters["learning_rate"] = 0.0
    parameters["checkpoint_source"] = str(checkpoint_source)
    parameters.setdefault("max_val_images", 8)
    parameters.setdefault("eval_split", "val")

    task_config = dict(base.get("task_config") or {})
    task_config["claim_level"] = "pipeline_validation_only"
    task_config.setdefault("evaluation_scope", "debug_subset")

    expected = [
        name
        for name in list(
            base.get("expected_outputs")
            or [
                "dataset_report.json",
                "metrics.json",
                "execution.json",
                "artifact_manifest.json",
                "model_summary.json",
                "sample_predictions.json",
                "checkpoint/last.npz",
                "combined.log",
            ]
        )
        if name != "training_history.csv"
    ]
    for name in (
        "model_summary.json",
        "sample_predictions.json",
        "checkpoint/last.npz",
    ):
        if name not in expected:
            expected.append(name)

    return {
        "schema_version": base.get("schema_version", "1.1"),
        "project_id": base.get("project_id", "project_rgbt_001"),
        "node_id": out_node,
        "parent_node_id": parent_node_id or source_node,
        "title": f"{base.get('title', source_node)} (fast_eval)",
        "research_goal": "Evaluate a fixed smoke checkpoint without retraining",
        "hypothesis": (
            "Multi-seed fast_eval can load a smoke checkpoint and produce "
            "stable pipeline-validation metrics on the debug val split."
        ),
        "task_type": "rgbt_detection",
        "runner_profile": base.get("runner_profile", "local"),
        "environment_key": base.get("environment_key", "rgbt-detection-v1"),
        "code_reference": base.get("code_reference", "local:rgbt_detector"),
        "dataset_reference": base.get("dataset_reference", "dataset:rgbt_debug_v1"),
        "entrypoint": base.get("entrypoint", "run_detection_experiment.py"),
        "execution_mode": "fast_eval",
        "parameters": parameters,
        "task_config": task_config,
        "seed": int(base.get("seed") or DEFAULT_FAST_EVAL_SEEDS[0]),
        "resources": dict(base.get("resources") or {}),
        "expected_outputs": expected,
    }


def relative_checkpoint_source(
    *,
    project_id: str,
    execution_id: str,
    checkpoint_name: str = "last.npz",
) -> str:
    return f"{project_id}/{execution_id}/checkpoint/{checkpoint_name}"


def resolve_checkpoint_path(outputs_root: Path, checkpoint_source: str) -> Path:
    src = Path(checkpoint_source)
    if src.is_absolute():
        return src
    root = Path(outputs_root).resolve()
    candidates = [
        root / src,
        root.parent / src,
        root.parent / "outputs" / src,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"checkpoint_source not found: {checkpoint_source}")
