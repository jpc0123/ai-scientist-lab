"""Deterministic DecisionRubric: rules first, Reviewer explains after.

Does not emit KEEP/DISCARD. That remains review_decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class RubricResult:
    objective_check: dict[str, dict[str, float | None]]
    constraint_check: dict[str, str]
    primary_delta: float | None
    constraints_ok: bool
    suggest_validate: bool
    suggest_discard_threshold: bool
    keep_threshold_ok: bool


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def evaluate_rubric(
    protocol: Mapping[str, Any],
    *,
    current_metrics: Mapping[str, Any],
    baseline_metrics: Mapping[str, Any] | None = None,
) -> RubricResult:
    objective = protocol.get("objective") or {}
    primary = objective.get("primary") or {}
    metric = primary.get("metric")
    direction = primary.get("direction", "maximize")
    policy = protocol.get("decision_policy") or {}
    constraints = protocol.get("constraints") or {}

    baseline_metrics = baseline_metrics or {}
    current = _num(current_metrics.get(metric)) if metric else None
    baseline = _num(baseline_metrics.get(metric)) if metric else None
    if current is not None and baseline is not None:
        raw_delta = current - baseline
        delta = raw_delta if direction == "maximize" else -raw_delta
    else:
        delta = None

    objective_check: dict[str, dict[str, float | None]] = {}
    if metric:
        objective_check[metric] = {
            "baseline": baseline,
            "current": current,
            "delta": (current - baseline) if current is not None and baseline is not None else None,
        }

    constraint_check: dict[str, str] = {}
    constraints_ok = True
    for key in ("params_m", "flops_g", "gpu_memory_gb"):
        bound = constraints.get(key)
        if not isinstance(bound, dict):
            constraint_check[key] = "NOT_CHECKED"
            continue
        observed = _num(current_metrics.get(key))
        if observed is None:
            constraint_check[key] = "NOT_CHECKED"
            continue
        max_v = bound.get("max")
        min_v = bound.get("min")
        ok = True
        if max_v is not None and observed > float(max_v):
            ok = False
        if min_v is not None and observed < float(min_v):
            ok = False
        constraint_check[key] = "PASS" if ok else "FAIL"
        constraints_ok = constraints_ok and ok

    keep_floor = (policy.get("keep_requires") or {}).get("primary_not_worse_than")
    validate_at = (policy.get("validate_if") or {}).get("primary_gain_at_least")
    discard_below = (policy.get("discard_if") or {}).get("primary_worse_than")

    keep_ok = True if delta is None or keep_floor is None else delta >= float(keep_floor)
    suggest_validate = False if delta is None or validate_at is None else delta >= float(validate_at)
    suggest_discard = False if delta is None or discard_below is None else delta < float(discard_below)

    return RubricResult(
        objective_check=objective_check,
        constraint_check=constraint_check,
        primary_delta=delta,
        constraints_ok=constraints_ok,
        suggest_validate=suggest_validate,
        suggest_discard_threshold=suggest_discard,
        keep_threshold_ok=keep_ok,
    )
