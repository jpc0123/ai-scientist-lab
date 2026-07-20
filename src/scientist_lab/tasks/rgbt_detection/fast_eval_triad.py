"""Formal Fast Eval triad: RGB / Thermal / Early-Fusion comparison helpers."""

from __future__ import annotations

from typing import Any

from scientist_lab.tasks.rgbt_detection.decision_rules import (
    is_exploratory_fast_eval_context,
)
from scientist_lab.tasks.rgbt_detection.feedback_rules import apply_detection_claim_gate


TRIAD_ROLES = ("rgb", "thermal", "fusion")

DEFAULT_TRIAD_NODES = {
    "rgb": "rgbt_fast_node_001",
    "thermal": "rgbt_fast_node_002",
    "fusion": "rgbt_fast_node_003",
}

SHARED_FIELDS = (
    "dataset_reference",
    "code_reference",
    "environment_key",
    "entrypoint",
    "execution_mode",
    "seed",
)

SHARED_PARAMETERS = (
    "baseline",
    "epochs",
    "batch_size",
    "learning_rate",
    "image_width",
    "image_height",
    "max_train_images",
    "max_val_images",
)

EXPECTED_MODES = {
    "rgb": {"input_mode": "rgb", "fusion_method": "none"},
    "thermal": {"input_mode": "thermal", "fusion_method": "none"},
    "fusion": {"input_mode": "rgbt", "fusion_method": "early_concat"},
}

PRIMARY_METRICS = ("mAP50_95", "mAP50", "AP_small", "precision", "recall")

CAVEATS = [
    "Comparison is limited to the frozen Fast Eval subset and budget.",
    "Evidence strength remains weak; do not treat ranking as a full-benchmark result.",
    "SOTA / significance / cross-dataset generalization claims are forbidden.",
    "Resource cost differences must be reported alongside metric deltas.",
]


def _task_config(contract: dict[str, Any]) -> dict[str, Any]:
    return dict(contract.get("task_config") or {})


def _parameters(contract: dict[str, Any]) -> dict[str, Any]:
    return dict(contract.get("parameters") or {})


def validate_triad_contracts(
    contracts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Validate that three contracts form a fair exploratory Fast Eval triad."""
    issues: list[str] = []
    warnings: list[str] = []

    missing_roles = [role for role in TRIAD_ROLES if role not in contracts]
    if missing_roles:
        return {
            "ok": False,
            "issues": [f"missing roles: {', '.join(missing_roles)}"],
            "warnings": warnings,
        }

    for role, contract in contracts.items():
        if contract.get("execution_mode") != "fast_eval":
            issues.append(f"{role}: execution_mode must be fast_eval")
        task_config = _task_config(contract)
        claim = str(task_config.get("claim_level") or "")
        scope = str(task_config.get("evaluation_scope") or "")
        baseline = str(_parameters(contract).get("baseline") or "")
        if not is_exploratory_fast_eval_context(
            task_type=str(contract.get("task_type") or ""),
            execution_mode=str(contract.get("execution_mode") or ""),
            evaluation_scope=scope,
            claim_level=claim,
        ):
            issues.append(
                f"{role}: not exploratory Fast Eval "
                f"(claim_level={claim!r}, scope={scope!r})"
            )
        expected = EXPECTED_MODES[role]
        params = _parameters(contract)
        for key, value in expected.items():
            if str(params.get(key) or "") != value:
                issues.append(
                    f"{role}: parameters.{key} expected {value!r}, "
                    f"got {params.get(key)!r}"
                )
        if baseline in {"", "tiny_detector"}:
            warnings.append(
                f"{role}: baseline={baseline!r} is not a formal real baseline"
            )

    reference = contracts["rgb"]
    for role in ("thermal", "fusion"):
        other = contracts[role]
        for field in SHARED_FIELDS:
            if reference.get(field) != other.get(field):
                issues.append(
                    f"{role}: {field} differs from rgb "
                    f"({reference.get(field)!r} vs {other.get(field)!r})"
                )
        ref_params = _parameters(reference)
        other_params = _parameters(other)
        for key in SHARED_PARAMETERS:
            if ref_params.get(key) != other_params.get(key):
                issues.append(
                    f"{role}: parameters.{key} differs from rgb "
                    f"({ref_params.get(key)!r} vs {other_params.get(key)!r})"
                )
        ref_claim = _task_config(reference).get("claim_level")
        other_claim = _task_config(other).get("claim_level")
        if ref_claim != other_claim:
            issues.append(f"{role}: claim_level differs from rgb")
        ref_scope = _task_config(reference).get("evaluation_scope")
        other_scope = _task_config(other).get("evaluation_scope")
        if ref_scope != other_scope:
            issues.append(f"{role}: evaluation_scope differs from rgb")

    return {"ok": not issues, "issues": issues, "warnings": warnings}


def _metric_value(metrics: dict[str, Any], key: str) -> float | None:
    raw = metrics.get(key)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def build_triad_comparison(
    *,
    contracts: dict[str, dict[str, Any]],
    metrics_by_role: dict[str, dict[str, Any]],
    execution_ids: dict[str, str] | None = None,
    resource_by_role: dict[str, dict[str, Any]] | None = None,
    primary_metric: str = "mAP50_95",
) -> dict[str, Any]:
    fairness = validate_triad_contracts(contracts)
    resource_by_role = resource_by_role or {}
    execution_ids = execution_ids or {}

    rows: list[dict[str, Any]] = []
    for role in TRIAD_ROLES:
        metrics = dict(metrics_by_role.get(role) or {})
        row = {
            "role": role,
            "node_id": (contracts.get(role) or {}).get("node_id"),
            "execution_id": execution_ids.get(role),
            "input_mode": _parameters(contracts.get(role) or {}).get("input_mode"),
            "fusion_method": _parameters(contracts.get(role) or {}).get(
                "fusion_method"
            ),
            "metrics": {
                key: _metric_value(metrics, key) for key in PRIMARY_METRICS
            },
            "resources": dict(resource_by_role.get(role) or {}),
        }
        rows.append(row)

    ranking = sorted(
        (
            {
                "role": row["role"],
                "node_id": row["node_id"],
                "value": row["metrics"].get(primary_metric),
            }
            for row in rows
            if row["metrics"].get(primary_metric) is not None
        ),
        key=lambda item: item["value"],
        reverse=True,
    )

    pairwise: dict[str, Any] = {}
    pairs = (("rgb", "thermal"), ("rgb", "fusion"), ("thermal", "fusion"))
    for left, right in pairs:
        left_metrics = dict(metrics_by_role.get(left) or {})
        right_metrics = dict(metrics_by_role.get(right) or {})
        deltas = {}
        for key in PRIMARY_METRICS:
            lv = _metric_value(left_metrics, key)
            rv = _metric_value(right_metrics, key)
            if lv is None or rv is None:
                deltas[key] = None
            else:
                deltas[key] = round(rv - lv, 6)
        pairwise[f"{left}_vs_{right}"] = {
            "baseline_role": left,
            "candidate_role": right,
            "deltas_candidate_minus_baseline": deltas,
        }

    sample_contract = contracts.get("fusion") or contracts.get("rgb") or {}
    gate = apply_detection_claim_gate(
        execution_mode=str(sample_contract.get("execution_mode") or "fast_eval"),
        evaluation_scope=str(
            _task_config(sample_contract).get("evaluation_scope") or ""
        ),
        claim_level=str(_task_config(sample_contract).get("claim_level") or ""),
        baseline_key=str(_parameters(sample_contract).get("baseline") or ""),
    )

    return {
        "schema_version": "1.0",
        "comparison_kind": "fast_eval_triad",
        "claim_level": gate.get("claim_level", "exploratory_comparison"),
        "evidence_strength": gate.get("max_evidence_strength", "weak"),
        "primary_metric": primary_metric,
        "fairness": fairness,
        "nodes": rows,
        "ranking_by_primary_metric": ranking,
        "pairwise": pairwise,
        "caveats": list(CAVEATS),
        "forbidden_claims": [
            item.get("claim") if isinstance(item, dict) else item
            for item in (gate.get("blocked_claims") or [])
        ],
        "allowed_conclusions": list(gate.get("allowed_claims") or []),
        "decision_hint": gate.get("suggested_decision_type")
        if fairness.get("ok")
        else "fix_fairness_budget",
    }
