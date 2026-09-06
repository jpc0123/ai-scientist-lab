"""Rule-based candidate ranking (LLM priority is only one signal)."""

from __future__ import annotations

from typing import Any

from scientist_lab.agents.critic import CriticReview
from scientist_lab.agents.models import CandidateExperiment


def score_candidate(
    candidate: CandidateExperiment,
    *,
    critic: CriticReview | None = None,
    evidence_gap_coverage: float | None = None,
    protocol_compliance: float = 1.0,
) -> dict[str, Any]:
    planner_priority = float(candidate.priority)
    info_gain = float(critic.expected_information_gain) if critic else 0.5
    cost_eff = float(critic.cost_effectiveness) if critic else 0.5
    gap = (
        evidence_gap_coverage
        if evidence_gap_coverage is not None
        else (0.8 if candidate.evidence_gap_addressed else 0.2)
    )
    compliance = float(protocol_compliance)

    final = (
        0.30 * planner_priority
        + 0.25 * info_gain
        + 0.20 * gap
        + 0.15 * cost_eff
        + 0.10 * compliance
    )

    # Penalties
    if critic and critic.novelty_status == "duplicate":
        final -= 0.40
    if critic and critic.risk_level == "high":
        final -= 0.20
    if critic and critic.recommendation == "reject":
        final -= 0.50
    if len(candidate.parameter_changes or {}) > 2:
        final -= 0.10
    if not candidate.evidence_gap_addressed:
        final -= 0.10

    final = max(0.0, min(1.0, final))
    return {
        "candidate_id": candidate.candidate_id,
        "planner_priority": planner_priority,
        "critic_information_gain": info_gain,
        "cost_effectiveness": cost_eff,
        "evidence_gap_coverage": gap,
        "protocol_compliance": compliance,
        "final_score": round(final, 4),
    }


def rank_candidates(
    items: list[tuple[CandidateExperiment, CriticReview | None]],
    *,
    max_keep: int = 3,
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for candidate, critic in items:
        if critic and critic.recommendation == "reject":
            continue
        row = score_candidate(candidate, critic=critic)
        scored.append(row)
    scored.sort(key=lambda item: item["final_score"], reverse=True)
    kept = scored[: max(1, max_keep)]
    for index, item in enumerate(kept, start=1):
        item["rank"] = index
    return kept
