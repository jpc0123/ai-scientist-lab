from __future__ import annotations

import re
from typing import Any

from scientist_lab.domain.feedback import FeedbackReport


def _next_node_id(candidate_node_id: str, existing_ids: set[str]) -> str:
    match = re.fullmatch(r"(node_)(\d+)", candidate_node_id)
    if match:
        width = max(3, len(match.group(2)))
        number = int(match.group(2)) + 1
        while True:
            candidate = f"node_{number:0{width}d}"
            if candidate not in existing_ids:
                return candidate
            number += 1
    match = re.fullmatch(r"([A-Za-z0-9]+_node_)(\d+)", candidate_node_id)
    if match:
        width = max(3, len(match.group(2)))
        number = int(match.group(2)) + 1
        while True:
            candidate = f"{match.group(1)}{number:0{width}d}"
            if candidate not in existing_ids:
                return candidate
            number += 1
    suffix = 1
    while True:
        candidate = f"{candidate_node_id}_next_{suffix}"
        if candidate not in existing_ids:
            return candidate
        suffix += 1


def select_proposal_recommendation(report: FeedbackReport) -> dict[str, Any] | None:
    for item in report.recommendations:
        if not item.parameter_changes:
            continue
        if item.status in {"deferred", "already_evaluated"}:
            continue
        return item.model_dump()
    return None


def build_next_contract(
    *,
    baseline_contract: dict[str, Any],
    candidate_contract: dict[str, Any],
    feedback: FeedbackReport,
    existing_node_ids: set[str],
) -> dict[str, Any] | None:
    recommendation = select_proposal_recommendation(feedback)
    if recommendation is None:
        return None

    param_changes = recommendation.get("parameter_changes") or {}
    if not param_changes:
        return None

    base = dict(candidate_contract or baseline_contract)
    if not base:
        return None

    parent_id = feedback.candidate_node_id
    new_node_id = _next_node_id(parent_id, existing_node_ids)

    parameters = dict(base.get("parameters") or {})
    parameters.update(param_changes)

    title = base.get("title") or "Proposed next experiment"
    if "hidden_units" in param_changes:
        title = f"Digits MLP hidden units {param_changes['hidden_units']}"
    elif "epochs" in param_changes and base.get("task_type") == "rgbt_detection":
        title = (
            f"RGB-T smoke epochs={param_changes['epochs']} "
            f"({parameters.get('input_mode', 'rgb')})"
        )

    hypothesis = recommendation.get("rationale") or (
        "Test recommended parameter changes from Feedback Analyzer."
    )

    contract = {
        "schema_version": base.get("schema_version", "1.0"),
        "project_id": base.get("project_id", "project_001"),
        "node_id": new_node_id,
        "parent_node_id": parent_id,
        "title": title,
        "research_goal": base.get("research_goal")
        or "Follow-up experiment from Feedback Analyzer",
        "hypothesis": hypothesis,
        "task_type": base.get("task_type"),
        "task_config": dict(base.get("task_config") or {}),
        "runner_profile": base.get("runner_profile", "local"),
        "environment_key": base.get("environment_key", "digits-mlp-v1"),
        "code_reference": base.get("code_reference", "local:experiment_app"),
        "dataset_reference": base.get("dataset_reference", "sklearn:digits"),
        "entrypoint": base.get("entrypoint", "run_experiment.py"),
        "execution_mode": base.get("execution_mode", "fast_eval"),
        "parameters": parameters,
        "seed": base.get("seed", 42),
        "resources": dict(base.get("resources") or {}),
        "expected_outputs": list(
            base.get("expected_outputs")
            or [
                "metrics.json",
                "model.joblib",
                "training_curve.csv",
                "confusion_matrix.csv",
                "execution.json",
                "artifact_manifest.json",
                "combined.log",
            ]
        ),
    }
    return {
        "contract": contract,
        "recommendation": recommendation,
        "source_baseline_node_id": feedback.baseline_node_id,
        "source_candidate_node_id": feedback.candidate_node_id,
    }
