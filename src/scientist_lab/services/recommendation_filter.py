from __future__ import annotations

import hashlib
import json
from typing import Any

from scientist_lab.domain.feedback import ExperimentRecommendation


CRITICAL_PARAM_KEYS = (
    "learning_rate",
    "epochs",
    "hidden_units",
    "batch_size",
    "test_size",
)


def canonicalize_parameters(parameters: dict[str, Any] | None) -> str:
    params = dict(parameters or {})
    narrowed = {
        key: params.get(key)
        for key in CRITICAL_PARAM_KEYS
        if key in params
    }
    return json.dumps(narrowed, sort_keys=True, default=str)


def parameter_fingerprint(
    environment_key: str,
    dataset_reference: str,
    code_reference: str,
    parameters: dict[str, Any] | None,
) -> str:
    """Stable SHA256 over env/dataset/code + critical parameters."""
    payload = {
        "environment_key": environment_key or "",
        "dataset_reference": dataset_reference or "",
        "code_reference": code_reference or "",
        "parameters": {
            key: (parameters or {}).get(key)
            for key in CRITICAL_PARAM_KEYS
            if key in (parameters or {})
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def apply_parameter_changes(
    source_parameters: dict[str, Any],
    changes: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(source_parameters or {})
    merged.update(changes or {})
    return merged


def collect_tested_parameter_index(
    nodes: list[dict[str, Any]],
) -> dict[str, str]:
    """Map parameter signature / fingerprint -> first node_id that owns it."""
    index: dict[str, str] = {}
    for node in nodes:
        node_id = str(node.get("node_id") or "")
        params = node.get("parameters") or {}
        if not node_id or not isinstance(params, dict):
            continue
        # Keep legacy canonicalize key for backward-compatible matching.
        index.setdefault(canonicalize_parameters(params), node_id)
        index.setdefault(
            parameter_fingerprint(
                str(node.get("environment_key") or ""),
                str(node.get("dataset_reference") or ""),
                str(node.get("code_reference") or ""),
                params,
            ),
            node_id,
        )
    return index


def _values_by_param(nodes: list[dict[str, Any]]) -> dict[str, set[Any]]:
    values_by_key: dict[str, set[Any]] = {}
    for node in nodes:
        params = node.get("parameters") or {}
        if not isinstance(params, dict):
            continue
        for key in CRITICAL_PARAM_KEYS:
            if key in params:
                values_by_key.setdefault(key, set()).add(params[key])
    return values_by_key


def _should_defer_probe(
    *,
    recommendation_type: str,
    changes: dict[str, Any],
    values_by_key: dict[str, set[Any]],
) -> bool:
    """Defer denser grid search once endpoints and an interior point already exist."""
    if recommendation_type not in {
        "intermediate_value",
        "expand_range",
        "efficiency_tradeoff",
    }:
        return False
    for key, value in changes.items():
        known = values_by_key.get(key) or set()
        if value in known:
            continue
        try:
            nums = sorted({float(v) for v in known})
            target = float(value)
        except (TypeError, ValueError):
            continue
        if len(nums) < 2:
            continue
        if not (nums[0] < target < nums[-1]):
            continue
        interior = [v for v in nums if nums[0] < v < nums[-1]]
        # e.g. already have 64,96,128 and now probing 80 or 112.
        if interior or len(nums) >= 3:
            return True
    return False


def annotate_recommendations_against_tested(
    recommendations: list[ExperimentRecommendation],
    *,
    source_parameters: dict[str, Any],
    tested_index: dict[str, str],
    tested_nodes: list[dict[str, Any]] | None = None,
    defer_intermediate: bool = False,
    source_environment_key: str = "",
    source_dataset_reference: str = "",
    source_code_reference: str = "",
) -> list[ExperimentRecommendation]:
    _ = defer_intermediate  # kept for API compatibility; defer uses axis coverage.
    tested_nodes = tested_nodes or []
    values_by_key = _values_by_param(tested_nodes)

    annotated: list[ExperimentRecommendation] = []
    for item in recommendations:
        data = item.model_dump()
        changes = dict(data.get("parameter_changes") or {})
        if not changes:
            data.setdefault("status", "active")
            annotated.append(ExperimentRecommendation.model_validate(data))
            continue

        resulting = apply_parameter_changes(source_parameters, changes)
        signature = canonicalize_parameters(resulting)
        fingerprint = parameter_fingerprint(
            source_environment_key,
            source_dataset_reference,
            source_code_reference,
            resulting,
        )
        existing_node = tested_index.get(fingerprint) or tested_index.get(signature)
        if existing_node is not None:
            data["status"] = "already_evaluated"
            data["already_evaluated_node_id"] = existing_node
            data["priority"] = min(float(data.get("priority") or 0.0), 0.35)
            data["rationale"] = (
                f"{data.get('rationale', '').rstrip()} "
                f"[already_evaluated by {existing_node}; do not propose a duplicate node]"
            ).strip()
            annotated.append(ExperimentRecommendation.model_validate(data))
            continue

        if _should_defer_probe(
            recommendation_type=str(data.get("recommendation_type")),
            changes=changes,
            values_by_key=values_by_key,
        ):
            data["status"] = "deferred"
            data["priority"] = min(float(data.get("priority") or 0.0), 0.45)
            data["rationale"] = (
                "Potential future efficiency exploration; not required for the "
                "current decision. "
                f"Original rationale: {data.get('rationale', '')}"
            ).strip()
            annotated.append(ExperimentRecommendation.model_validate(data))
            continue

        data.setdefault("status", "active")
        annotated.append(ExperimentRecommendation.model_validate(data))
    return annotated
