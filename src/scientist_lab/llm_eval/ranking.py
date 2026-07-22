"""Static LLM profile ranking among Quality-Gate-qualified profiles (v1.5.8)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from scientist_lab.llm_eval.quality_gate import (
    LLMQualityThresholds,
    _f,
    verify_quality_gate,
)


class RankWeights(BaseModel):
    planner_quality: float = 0.35
    critic_quality: float = 0.25
    safety_score: float = 0.20
    cost_efficiency: float = 0.10
    latency_score: float = 0.10


class ProfileRankEntry(BaseModel):
    profile_id: str
    evaluation_id: str | None = None
    qualified: bool
    profile_score: float | None = None
    disqualify_reasons: list[str] = Field(default_factory=list)
    components: dict[str, float] = Field(default_factory=dict)
    scorecard_summary: dict[str, Any] = Field(default_factory=dict)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _planner_quality(planner: dict[str, Any]) -> float:
    schema = _f(planner.get("schema_valid_rate"))
    protocol = _f(planner.get("protocol_compliance_rate"), 1.0)
    evidence = planner.get("evidence_gap_relevance_rate")
    evidence_v = _f(evidence, schema)
    dup = _f(planner.get("duplicate_candidate_rate"))
    return _clamp01(0.4 * schema + 0.35 * protocol + 0.25 * evidence_v - 0.2 * dup)


def _critic_quality(critic: dict[str, Any]) -> float:
    accept = critic.get("valid_acceptance_rate")
    reject = critic.get("invalid_rejection_rate")
    overreach = critic.get("claim_overreach_detection_rate")
    parts = [_f(x) for x in (accept, reject, overreach) if x is not None]
    if not parts:
        return _f(critic.get("pass_rate"), 0.0)
    return _clamp01(sum(parts) / len(parts))


def _safety_score(safety: dict[str, Any]) -> float:
    if not bool(safety.get("pass", True)):
        return 0.0
    return _clamp01(_f(safety.get("pass_rate"), 1.0))


def _cost_efficiency(ops: dict[str, Any], *, ref_cost: float = 1.0) -> float:
    cost = ops.get("estimated_cost_usd")
    if cost is None:
        return 0.5
    c = _f(cost)
    if c <= 0:
        return 1.0
    return _clamp01(ref_cost / (ref_cost + c))


def _latency_score(ops: dict[str, Any], *, ref_ms: float = 2000.0) -> float:
    latency = ops.get("average_latency_ms")
    if latency is None:
        return 0.5
    return _clamp01(ref_ms / (ref_ms + max(0.0, _f(latency))))


def score_profile(
    scorecard: dict[str, Any],
    *,
    weights: RankWeights | None = None,
) -> tuple[float, dict[str, float]]:
    w = weights or RankWeights()
    planner = dict(scorecard.get("planner") or {})
    critic = dict(scorecard.get("critic") or {})
    safety = dict(scorecard.get("safety") or {})
    ops = dict(scorecard.get("operations") or {})
    components = {
        "planner_quality": _planner_quality(planner),
        "critic_quality": _critic_quality(critic),
        "safety_score": _safety_score(safety),
        "cost_efficiency": _cost_efficiency(ops),
        "latency_score": _latency_score(ops),
    }
    total = (
        w.planner_quality * components["planner_quality"]
        + w.critic_quality * components["critic_quality"]
        + w.safety_score * components["safety_score"]
        + w.cost_efficiency * components["cost_efficiency"]
        + w.latency_score * components["latency_score"]
    )
    return total, components


def rank_profiles(
    entries: list[dict[str, Any]],
    *,
    thresholds: LLMQualityThresholds | None = None,
    weights: RankWeights | None = None,
) -> list[ProfileRankEntry]:
    """Rank profiles. Unqualified (blocked gate / safety fail) are excluded from score order.

    ``entries`` items: ``{profile_id, evaluation_id?, scorecard}``
    """
    ranked: list[ProfileRankEntry] = []
    for item in entries:
        profile_id = str(item.get("profile_id") or "")
        scorecard = dict(item.get("scorecard") or {})
        gate = verify_quality_gate(scorecard, thresholds=thresholds)
        reasons: list[str] = []
        if gate.status == "blocked":
            reasons.extend(gate.blocking_reasons)
        safety = dict(scorecard.get("safety") or {})
        if not bool(safety.get("pass", True)):
            reasons.append("safety not passed")
        planner = dict(scorecard.get("planner") or {})
        if int(planner.get("case_count") or 0) > 0:
            if _f(planner.get("protocol_compliance_rate"), 1.0) < 1.0:
                reasons.append("protocol compliance < 100%")

        qualified = not reasons
        score = None
        components: dict[str, float] = {}
        if qualified:
            score, components = score_profile(scorecard, weights=weights)
        ranked.append(
            ProfileRankEntry(
                profile_id=profile_id,
                evaluation_id=item.get("evaluation_id"),
                qualified=qualified,
                profile_score=score,
                disqualify_reasons=reasons,
                components=components,
                scorecard_summary={
                    "planner_pass_rate": planner.get("pass_rate"),
                    "critic_pass_rate": (scorecard.get("critic") or {}).get("pass_rate"),
                    "safety_pass": safety.get("pass"),
                    "average_latency_ms": (scorecard.get("operations") or {}).get(
                        "average_latency_ms"
                    ),
                    "estimated_cost_usd": (scorecard.get("operations") or {}).get(
                        "estimated_cost_usd"
                    ),
                },
            )
        )

    qualified = sorted(
        [r for r in ranked if r.qualified and r.profile_score is not None],
        key=lambda r: float(r.profile_score or 0.0),
        reverse=True,
    )
    unqualified = [r for r in ranked if not r.qualified]
    return qualified + unqualified
