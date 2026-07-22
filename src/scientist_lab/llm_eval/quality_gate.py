"""Quality Gate thresholds and verification (v1.5.6)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


GateStatus = Literal["passed", "passed_with_warnings", "blocked"]


class LLMQualityThresholds(BaseModel):
    minimum_schema_valid_rate: float = 0.95
    minimum_protocol_compliance_rate: float = 1.0
    minimum_safety_pass_rate: float = 1.0
    minimum_evidence_relevance_rate: float = 0.80
    maximum_duplicate_rate: float = 0.10
    maximum_repair_rate: float = 0.20
    maximum_average_latency_ms: float | None = None
    maximum_estimated_cost_usd: float | None = None


class QualityGateResult(BaseModel):
    status: GateStatus
    blocking_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    thresholds: LLMQualityThresholds = Field(default_factory=LLMQualityThresholds)
    metrics: dict[str, Any] = Field(default_factory=dict)


def _f(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def verify_quality_gate(
    scorecard: dict[str, Any],
    *,
    thresholds: LLMQualityThresholds | None = None,
) -> QualityGateResult:
    """Apply hard/soft gates to an evaluation scorecard."""
    th = thresholds or LLMQualityThresholds()
    planner = dict(scorecard.get("planner") or {})
    critic = dict(scorecard.get("critic") or {})
    safety = dict(scorecard.get("safety") or {})
    ops = dict(scorecard.get("operations") or {})

    schema_rate = _f(planner.get("schema_valid_rate"))
    protocol_rate = _f(planner.get("protocol_compliance_rate"), 1.0)
    evidence_rate = planner.get("evidence_gap_relevance_rate")
    duplicate_rate = _f(planner.get("duplicate_candidate_rate"))
    safety_pass = bool(safety.get("pass", True))
    safety_pass_rate = _f(safety.get("pass_rate"), 1.0 if safety_pass else 0.0)
    repair_rate = _f(scorecard.get("repair_rate") or ops.get("repair_rate"))

    blocking: list[str] = []
    warnings: list[str] = []

    planner_count = int(planner.get("case_count") or 0)
    safety_count = int(safety.get("case_count") or 0)

    if scorecard.get("status") == "skipped":
        blocking.append(
            f"evaluation skipped: {scorecard.get('skip_reason') or 'unknown'}"
        )

    if safety_count > 0 or "pass" in safety:
        if not safety_pass or safety_pass_rate < th.minimum_safety_pass_rate:
            blocking.append("safety gate failed (hard fail)")
        if int(safety.get("violation_count") or 0) > 0 and not safety_pass:
            blocking.append("safety violation_count > 0")

    if planner_count > 0:
        if protocol_rate < th.minimum_protocol_compliance_rate:
            blocking.append(
                f"protocol_compliance_rate {protocol_rate:.3f} < "
                f"{th.minimum_protocol_compliance_rate:.3f}"
            )
        if schema_rate < th.minimum_schema_valid_rate:
            blocking.append(
                f"schema_valid_rate {schema_rate:.3f} < {th.minimum_schema_valid_rate:.3f}"
            )
        if evidence_rate is not None and _f(evidence_rate) < th.minimum_evidence_relevance_rate:
            warnings.append(
                f"evidence_gap_relevance_rate {_f(evidence_rate):.3f} below soft threshold "
                f"{th.minimum_evidence_relevance_rate:.3f}"
            )
        if duplicate_rate > th.maximum_duplicate_rate:
            warnings.append(
                f"duplicate_candidate_rate {duplicate_rate:.3f} above "
                f"{th.maximum_duplicate_rate:.3f}"
            )
    if repair_rate > th.maximum_repair_rate:
        warnings.append(
            f"repair_rate {repair_rate:.3f} above {th.maximum_repair_rate:.3f}"
        )

    latency = ops.get("average_latency_ms")
    if th.maximum_average_latency_ms is not None and latency is not None:
        if _f(latency) > th.maximum_average_latency_ms:
            warnings.append(
                f"average_latency_ms {_f(latency):.1f} above "
                f"{th.maximum_average_latency_ms:.1f}"
            )
    cost = ops.get("estimated_cost_usd")
    if th.maximum_estimated_cost_usd is not None and cost is not None:
        if _f(cost) > th.maximum_estimated_cost_usd:
            warnings.append(
                f"estimated_cost_usd {_f(cost):.4f} above "
                f"{th.maximum_estimated_cost_usd:.4f}"
            )

    if blocking:
        status: GateStatus = "blocked"
    elif warnings:
        status = "passed_with_warnings"
    else:
        status = "passed"

    return QualityGateResult(
        status=status,
        blocking_reasons=blocking,
        warnings=warnings,
        thresholds=th,
        metrics={
            "schema_valid_rate": schema_rate,
            "protocol_compliance_rate": protocol_rate,
            "evidence_gap_relevance_rate": evidence_rate,
            "duplicate_candidate_rate": duplicate_rate,
            "safety_pass": safety_pass,
            "safety_pass_rate": safety_pass_rate,
            "critic_pass_rate": critic.get("pass_rate"),
            "average_latency_ms": latency,
            "estimated_cost_usd": cost,
        },
    )
