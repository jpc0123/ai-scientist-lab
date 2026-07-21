"""Detection-oriented multi-metric / resource group comparison (v0.9.3)."""

from __future__ import annotations

from typing import Any

# Performance metrics (higher is better) for RGB-T detection Fast Eval.
DETECTION_METRICS = (
    "mAP50_95",
    "mAP50",
    "AP_small",
    "precision",
    "recall",
)

# Resource metrics (lower is better) for cost analysis.
RESOURCE_METRICS = (
    "duration_seconds",
    "peak_gpu_memory_mb",
    "parameter_count",
)

DEFAULT_METRIC_EPS = 1e-6
DEFAULT_RESOURCE_REL_EPS = 0.05  # 5% relative change for resources
DEFAULT_RESOURCE_ABS_EPS = 1e-9


def _mean_of(aggregate: dict[str, Any], key: str) -> float | None:
    blob = (aggregate.get("aggregate_metrics") or {}).get(key) or {}
    value = blob.get("mean")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def classify_higher_is_better(
    baseline_mean: float | None,
    candidate_mean: float | None,
    *,
    eps: float = DEFAULT_METRIC_EPS,
) -> str:
    if baseline_mean is None or candidate_mean is None:
        return "missing"
    delta = candidate_mean - baseline_mean
    if abs(delta) <= eps:
        return "equivalent"
    return "candidate_better" if delta > 0 else "baseline_better"


def classify_lower_is_better(
    baseline_mean: float | None,
    candidate_mean: float | None,
    *,
    rel_eps: float = DEFAULT_RESOURCE_REL_EPS,
    abs_eps: float = DEFAULT_RESOURCE_ABS_EPS,
) -> str:
    """Return candidate_better / candidate_worse / equivalent / missing."""
    if baseline_mean is None or candidate_mean is None:
        return "missing"
    delta = candidate_mean - baseline_mean
    scale = max(abs(baseline_mean), abs_eps)
    if abs(delta) <= max(abs_eps, scale * rel_eps):
        return "equivalent"
    # Lower resource cost is better for the candidate.
    return "candidate_better" if delta < 0 else "candidate_worse"


def build_metric_relations(
    baseline_aggregate: dict[str, Any],
    candidate_aggregate: dict[str, Any],
    *,
    metrics: tuple[str, ...] = DETECTION_METRICS,
    eps: float = DEFAULT_METRIC_EPS,
) -> dict[str, str]:
    relations: dict[str, str] = {}
    for key in metrics:
        relations[key] = classify_higher_is_better(
            _mean_of(baseline_aggregate, key),
            _mean_of(candidate_aggregate, key),
            eps=eps,
        )
    return relations


def build_resource_relations(
    baseline_aggregate: dict[str, Any],
    candidate_aggregate: dict[str, Any],
    *,
    metrics: tuple[str, ...] = RESOURCE_METRICS,
    rel_eps: float = DEFAULT_RESOURCE_REL_EPS,
) -> dict[str, str]:
    relations: dict[str, str] = {}
    for key in metrics:
        relations[key] = classify_lower_is_better(
            _mean_of(baseline_aggregate, key),
            _mean_of(candidate_aggregate, key),
            rel_eps=rel_eps,
        )
    return relations


def resource_missing_warnings(
    baseline_aggregate: dict[str, Any],
    candidate_aggregate: dict[str, Any],
    *,
    metrics: tuple[str, ...] = RESOURCE_METRICS,
) -> list[str]:
    warnings: list[str] = []
    for key in metrics:
        b = _mean_of(baseline_aggregate, key)
        c = _mean_of(candidate_aggregate, key)
        if b is None and c is None:
            warnings.append(f"resource metric missing on both nodes: {key}")
        elif b is None:
            warnings.append(f"resource metric missing on baseline: {key}")
        elif c is None:
            warnings.append(f"resource metric missing on candidate: {key}")
    return warnings


def extract_claim_level(
    baseline_aggregate: dict[str, Any] | None = None,
    candidate_aggregate: dict[str, Any] | None = None,
    *,
    sample_contract: dict[str, Any] | None = None,
) -> str | None:
    for source in (sample_contract,):
        if not source:
            continue
        task_config = source.get("task_config") or {}
        claim = str(task_config.get("claim_level") or "").strip()
        if claim:
            return claim
        claim = str(source.get("claim_level") or "").strip()
        if claim:
            return claim
    for aggregate in (candidate_aggregate, baseline_aggregate):
        if not aggregate:
            continue
        for item in aggregate.get("executions") or []:
            # claim may live on nested metrics snapshot if present later
            metrics = item.get("metrics") or {}
            claim = str(metrics.get("claim_level") or "").strip()
            if claim:
                return claim
    return None


def enrich_group_comparison(
    result: dict[str, Any],
    *,
    sample_contract: dict[str, Any] | None = None,
    detection_metrics: tuple[str, ...] = DETECTION_METRICS,
    resource_metrics: tuple[str, ...] = RESOURCE_METRICS,
) -> dict[str, Any]:
    """Attach v0.9.3 detection multi-metric / resource relation fields."""
    baseline = result.get("baseline_aggregate") or {}
    candidate = result.get("candidate_aggregate") or {}

    metric_relations = build_metric_relations(
        baseline, candidate, metrics=detection_metrics
    )
    resource_relations = build_resource_relations(
        baseline, candidate, metrics=resource_metrics
    )

    verification = dict(result.get("verification") or {})
    warnings = list(verification.get("warnings") or [])
    for warning in resource_missing_warnings(
        baseline, candidate, metrics=resource_metrics
    ):
        if warning not in warnings:
            warnings.append(warning)
    verification["warnings"] = warnings

    claim_level = extract_claim_level(
        baseline, candidate, sample_contract=sample_contract
    )
    # Digits / non-detection runs still get relations for whatever overlap exists.
    # Prefer protocol exploratory default only when detection metrics are present.
    has_detection = any(
        metric_relations.get(key) != "missing" for key in detection_metrics
    )
    if claim_level is None and has_detection:
        claim_level = "exploratory_comparison"

    metric_deltas: dict[str, float | None] = {}
    for key in detection_metrics:
        b = _mean_of(baseline, key)
        c = _mean_of(candidate, key)
        metric_deltas[key] = None if b is None or c is None else round(c - b, 6)

    resource_deltas: dict[str, float | None] = {}
    for key in resource_metrics:
        b = _mean_of(baseline, key)
        c = _mean_of(candidate, key)
        resource_deltas[key] = None if b is None or c is None else round(c - b, 6)

    result = dict(result)
    result["verification"] = verification
    result["paired_win_count"] = result.get("candidate_win_count")
    result["paired_loss_count"] = result.get("baseline_win_count")
    result["metric_relations"] = metric_relations
    result["resource_relations"] = resource_relations
    result["metric_mean_deltas"] = metric_deltas
    result["resource_mean_deltas"] = resource_deltas
    if claim_level is not None:
        result["claim_level"] = claim_level
    result["comparison_schema"] = "detection_group_v0_9_3"
    return result
