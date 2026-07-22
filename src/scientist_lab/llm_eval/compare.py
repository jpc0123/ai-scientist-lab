"""Compare two evaluation scorecards for Prompt/model regression (v1.5.7)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from scientist_lab.llm_eval.quality_gate import LLMQualityThresholds, _f


class RegressionThresholds(BaseModel):
    """Maximum allowed negative deltas (candidate - baseline)."""

    max_schema_valid_drop: float = 0.02
    max_protocol_compliance_drop: float = 0.0
    max_evidence_relevance_drop: float = 0.05
    max_duplicate_rate_rise: float = 0.05
    max_cost_rise_usd: float | None = None
    max_latency_rise_ms: float | None = None
    require_candidate_gate_not_blocked: bool = True


class EvaluationCompareResult(BaseModel):
    baseline_evaluation_id: str
    candidate_evaluation_id: str
    schema_valid_delta: float | None = None
    protocol_compliance_delta: float | None = None
    evidence_relevance_delta: float | None = None
    duplicate_rate_delta: float | None = None
    average_cost_delta: float | None = None
    average_latency_delta: float | None = None
    safety_pass_baseline: bool | None = None
    safety_pass_candidate: bool | None = None
    regression_detected: bool = False
    blocking_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


def _delta(cand: Any, base: Any) -> float | None:
    if cand is None or base is None:
        return None
    return _f(cand) - _f(base)


def compare_evaluations(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    baseline_id: str,
    candidate_id: str,
    thresholds: RegressionThresholds | None = None,
    quality_thresholds: LLMQualityThresholds | None = None,
) -> EvaluationCompareResult:
    """Compare candidate vs baseline scorecards.

    Negative quality deltas beyond thresholds → regression_detected + blocking_reasons.
    """
    th = thresholds or RegressionThresholds()
    base_planner = dict(baseline.get("planner") or {})
    cand_planner = dict(candidate.get("planner") or {})
    base_ops = dict(baseline.get("operations") or {})
    cand_ops = dict(candidate.get("operations") or {})
    base_safety = dict(baseline.get("safety") or {})
    cand_safety = dict(candidate.get("safety") or {})

    schema_d = _delta(
        cand_planner.get("schema_valid_rate"), base_planner.get("schema_valid_rate")
    )
    protocol_d = _delta(
        cand_planner.get("protocol_compliance_rate"),
        base_planner.get("protocol_compliance_rate"),
    )
    evidence_d = _delta(
        cand_planner.get("evidence_gap_relevance_rate"),
        base_planner.get("evidence_gap_relevance_rate"),
    )
    dup_d = _delta(
        cand_planner.get("duplicate_candidate_rate"),
        base_planner.get("duplicate_candidate_rate"),
    )
    cost_d = _delta(
        cand_ops.get("estimated_cost_usd"), base_ops.get("estimated_cost_usd")
    )
    latency_d = _delta(
        cand_ops.get("average_latency_ms"), base_ops.get("average_latency_ms")
    )

    blocking: list[str] = []
    warnings: list[str] = []

    if th.require_candidate_gate_not_blocked:
        from scientist_lab.llm_eval.quality_gate import verify_quality_gate

        gate = verify_quality_gate(candidate, thresholds=quality_thresholds)
        if gate.status == "blocked":
            blocking.append(
                "Candidate evaluation is blocked by Quality Gate: "
                + "; ".join(gate.blocking_reasons or ["unknown"])
            )

    if not bool(cand_safety.get("pass", True)):
        blocking.append("Candidate safety gate failed.")

    if schema_d is not None and schema_d < -th.max_schema_valid_drop:
        blocking.append(
            f"Schema valid rate decreased beyond the allowed threshold "
            f"(delta={schema_d:.3f})."
        )
    if protocol_d is not None and protocol_d < -th.max_protocol_compliance_drop:
        blocking.append(
            f"Protocol compliance decreased beyond the allowed threshold "
            f"(delta={protocol_d:.3f})."
        )
    if evidence_d is not None and evidence_d < -th.max_evidence_relevance_drop:
        blocking.append(
            "Evidence-gap relevance decreased beyond the allowed threshold."
        )
    if dup_d is not None and dup_d > th.max_duplicate_rate_rise:
        blocking.append(
            f"Duplicate candidate rate increased beyond the allowed threshold "
            f"(delta={dup_d:.3f})."
        )

    if cost_d is not None and th.max_cost_rise_usd is not None:
        if cost_d > th.max_cost_rise_usd:
            warnings.append(f"average cost rose by {cost_d:.4f} USD")
    elif cost_d is not None and cost_d > 0.05:
        warnings.append(f"average cost rose by {cost_d:.4f} USD")

    if latency_d is not None and th.max_latency_rise_ms is not None:
        if latency_d > th.max_latency_rise_ms:
            warnings.append(f"average latency rose by {latency_d:.1f} ms")
    elif latency_d is not None and latency_d > 2000:
        warnings.append(f"average latency rose by {latency_d:.1f} ms")

    return EvaluationCompareResult(
        baseline_evaluation_id=baseline_id,
        candidate_evaluation_id=candidate_id,
        schema_valid_delta=schema_d,
        protocol_compliance_delta=protocol_d,
        evidence_relevance_delta=evidence_d,
        duplicate_rate_delta=dup_d,
        average_cost_delta=cost_d,
        average_latency_delta=latency_d,
        safety_pass_baseline=bool(base_safety.get("pass", True)),
        safety_pass_candidate=bool(cand_safety.get("pass", True)),
        regression_detected=bool(blocking),
        blocking_reasons=blocking,
        warnings=warnings,
        metrics={
            "baseline_profile_id": baseline.get("profile_id"),
            "candidate_profile_id": candidate.get("profile_id"),
            "baseline_suite": baseline.get("suite_version"),
            "candidate_suite": candidate.get("suite_version"),
        },
    )
