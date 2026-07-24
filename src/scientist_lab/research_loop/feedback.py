"""Build and gate RoundFeedbackSummary for multi-round planning (v2.1.2)."""

from __future__ import annotations

from typing import Any, Literal

from scientist_lab.agents.models import PlanningContext, RoundFeedbackSummary
from scientist_lab.research_loop.errors import RealLoopValidationError


OutcomeLabel = Literal[
    "improved",
    "degraded",
    "comparable",
    "unstable",
    "costlier",
    "uncertain",
    "failed",
]


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _delta_maps_from_comparison(
    comparison: dict[str, Any] | None,
) -> tuple[
    dict[str, float | None],
    dict[str, float | None],
    dict[str, float | None],
    OutcomeLabel,
]:
    comparison = dict(comparison or {})
    metric_deltas: dict[str, float | None] = {}
    stability_deltas: dict[str, float | None] = {}
    resource_deltas: dict[str, float | None] = {}

    # Common comparison shapes used across Scientist Lab.
    for key, target in (
        ("metric_deltas", metric_deltas),
        ("stability_deltas", stability_deltas),
        ("resource_deltas", resource_deltas),
    ):
        raw = comparison.get(key)
        if isinstance(raw, dict):
            for mk, mv in raw.items():
                target[str(mk)] = _as_float(mv)

    primary = comparison.get("primary_metric") or comparison.get("metric")
    delta = comparison.get("primary_metric_delta")
    if delta is None:
        delta = comparison.get("delta")
    if delta is None:
        delta = comparison.get("mean_delta")
    if primary and delta is not None and str(primary) not in metric_deltas:
        metric_deltas[str(primary)] = _as_float(delta)
    elif delta is not None and "accuracy" not in metric_deltas and not primary:
        # Digits / node-group comparisons often only expose mean_delta.
        metric_deltas["accuracy"] = _as_float(delta)
    elif primary and delta is not None:
        metric_deltas.setdefault(str(primary), _as_float(delta))

    # Per-seed deltas list → optional stability proxy.
    seed_deltas = comparison.get("deltas") or comparison.get("per_seed_deltas")
    if isinstance(seed_deltas, list) and seed_deltas and "seed_std" not in stability_deltas:
        vals = [_as_float(x) for x in seed_deltas]
        nums = [v for v in vals if v is not None]
        if len(nums) >= 2:
            mean = sum(nums) / len(nums)
            var = sum((v - mean) ** 2 for v in nums) / len(nums)
            stability_deltas["seed_std"] = var**0.5

    metrics = comparison.get("metrics")
    if isinstance(metrics, dict):
        for mk, mv in metrics.items():
            if isinstance(mv, dict) and "delta" in mv:
                metric_deltas.setdefault(str(mk), _as_float(mv.get("delta")))
            elif mk not in metric_deltas:
                metric_deltas[str(mk)] = _as_float(mv)

    for rk in ("duration_seconds", "gpu_hours", "param_count", "parameters"):
        if rk in comparison and rk not in resource_deltas:
            resource_deltas[rk] = _as_float(comparison.get(rk))
        nested = comparison.get("resources")
        if isinstance(nested, dict) and rk in nested:
            resource_deltas.setdefault(rk, _as_float(nested.get(rk)))

    for sk in ("std", "seed_std", "accuracy_std"):
        if sk in comparison:
            stability_deltas.setdefault(sk, _as_float(comparison.get(sk)))

    label = _infer_outcome_label(
        comparison=comparison,
        metric_deltas=metric_deltas,
        stability_deltas=stability_deltas,
        resource_deltas=resource_deltas,
    )
    return metric_deltas, stability_deltas, resource_deltas, label


def _infer_outcome_label(
    *,
    comparison: dict[str, Any],
    metric_deltas: dict[str, float | None],
    stability_deltas: dict[str, float | None],
    resource_deltas: dict[str, float | None],
) -> OutcomeLabel:
    explicit = str(comparison.get("outcome_label") or comparison.get("conclusion") or "")
    explicit_l = explicit.lower()
    if comparison.get("failed") or explicit_l in {"failed", "failure"}:
        return "failed"
    if "unstable" in explicit_l or "不稳定" in explicit:
        return "unstable"
    if "cost" in explicit_l or "成本" in explicit:
        return "costlier"
    if explicit_l in {"improved", "better", "win"} or "提升" in explicit:
        return "improved"
    if explicit_l in {"degraded", "worse", "loss"} or "下降" in explicit:
        return "degraded"
    if explicit_l in {"comparable", "tie", "equivalent"} or "相当" in explicit:
        return "comparable"

    primary_delta = None
    for key in ("accuracy", "f1_macro", "ap", "map"):
        if key in metric_deltas and metric_deltas[key] is not None:
            primary_delta = metric_deltas[key]
            break
    if primary_delta is None and metric_deltas:
        primary_delta = next(
            (v for v in metric_deltas.values() if v is not None), None
        )

    std_delta = next((v for v in stability_deltas.values() if v is not None), None)
    cost_delta = next((v for v in resource_deltas.values() if v is not None), None)

    if std_delta is not None and std_delta > 0:
        return "unstable"
    if primary_delta is not None:
        if primary_delta > 1e-6:
            if cost_delta is not None and cost_delta > 0:
                return "costlier"
            return "improved"
        if primary_delta < -1e-6:
            return "degraded"
        return "comparable"
    if cost_delta is not None and cost_delta > 0:
        return "costlier"
    return "uncertain"


def build_round_feedback_summary(
    *,
    source_round: int,
    parent_node_id: str,
    executed_node_id: str,
    comparison: dict[str, Any] | None = None,
    evidence_ids: list[str] | None = None,
    claims_changed: list[str] | None = None,
    comparison_ids: list[str] | None = None,
    success_criteria_met: list[str] | None = None,
    failure_criteria_met: list[str] | None = None,
    unresolved_gaps: list[str] | None = None,
    limitations: list[str] | None = None,
    previous_hypothesis: str | None = None,
    executed_parameters: dict[str, Any] | None = None,
    failure: dict[str, Any] | None = None,
    outcome_label: OutcomeLabel | None = None,
) -> RoundFeedbackSummary:
    metric_deltas, stability_deltas, resource_deltas, inferred = (
        _delta_maps_from_comparison(comparison)
    )
    failure = dict(failure or {})
    label: OutcomeLabel = outcome_label or inferred
    if failure.get("error_type") or failure.get("message"):
        label = "failed"

    return RoundFeedbackSummary(
        source_round=int(source_round),
        parent_node_id=parent_node_id,
        executed_node_id=executed_node_id,
        metric_deltas=metric_deltas,
        stability_deltas=stability_deltas,
        resource_deltas=resource_deltas,
        evidence_added=[str(x) for x in (evidence_ids or []) if str(x).strip()],
        claims_changed=[str(x) for x in (claims_changed or []) if str(x).strip()],
        comparison_ids=[str(x) for x in (comparison_ids or []) if str(x).strip()],
        success_criteria_met=list(success_criteria_met or []),
        failure_criteria_met=list(failure_criteria_met or []),
        unresolved_gaps=list(unresolved_gaps or []),
        limitations=list(limitations or []),
        outcome_label=label,
        failure_category=(
            str(failure.get("error_type") or failure.get("category") or "") or None
        ),
        error_summary=(str(failure.get("message") or failure.get("summary") or "") or None),
        failed_stage=(str(failure.get("stage") or "") or None),
        recoverability=(str(failure.get("recoverability") or "") or None),
        previous_hypothesis=previous_hypothesis,
        executed_parameters=dict(executed_parameters or {}),
    )


def planning_context_has_round_feedback(context: PlanningContext) -> bool:
    summary = context.round_feedback_summary
    if summary is None:
        return False
    if not summary.parent_node_id or not summary.executed_node_id:
        return False
    return True


def require_round_feedback_for_planning(
    context: PlanningContext,
    *,
    round_number: int,
) -> None:
    """Block round >= 2 planning when feedback summary is missing."""
    if int(round_number) < 2:
        return
    if planning_context_has_round_feedback(context):
        # Ensure the summary points at a prior round.
        summary = context.round_feedback_summary
        assert summary is not None
        if int(summary.source_round) != int(round_number) - 1:
            raise RealLoopValidationError(
                "round_feedback_summary.source_round must be "
                f"{int(round_number) - 1} for round {round_number} planning "
                f"(got {summary.source_round})"
            )
        return
    raise RealLoopValidationError(
        "round_feedback_summary is required before real Planner "
        f"for round {round_number}; complete record-feedback first"
    )


def normalize_comparison_for_feedback(
    comparison: dict[str, Any] | None,
) -> dict[str, Any]:
    """Map node-group comparison fields into RoundFeedbackSummary-friendly shape."""
    raw = dict(comparison or {})
    out = dict(raw)
    primary = raw.get("primary_metric") or raw.get("metric") or "accuracy"
    out.setdefault("primary_metric", primary)
    if out.get("primary_metric_delta") is None and raw.get("mean_delta") is not None:
        out["primary_metric_delta"] = raw.get("mean_delta")
    if out.get("delta") is None and raw.get("mean_delta") is not None:
        out["delta"] = raw.get("mean_delta")
    if not out.get("outcome_label") and raw.get("decision"):
        decision = str(raw.get("decision") or "").lower()
        if "candidate" in decision and ("better" in decision or "win" in decision):
            out["outcome_label"] = "improved"
        elif "baseline" in decision and ("better" in decision or "win" in decision):
            out["outcome_label"] = "degraded"
        elif "equivalent" in decision or "tie" in decision:
            out["outcome_label"] = "comparable"
    return out
