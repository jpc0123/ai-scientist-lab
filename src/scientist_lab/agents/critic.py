"""Critic reviews Planner candidates (Mock first; LLM optional later)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, Field

from scientist_lab.agents.models import CandidateExperiment, PlanningContext
from scientist_lab.planning.candidate_verifier import parameter_fingerprint


class CriticReview(BaseModel):
    candidate_id: str

    scientific_validity: Literal["valid", "weak", "invalid"] = "valid"
    novelty_status: Literal["new", "partially_redundant", "duplicate"] = "new"

    expected_information_gain: float = Field(default=0.5, ge=0.0, le=1.0)
    cost_effectiveness: float = Field(default=0.5, ge=0.0, le=1.0)
    risk_level: Literal["low", "medium", "high"] = "low"

    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    required_revisions: list[str] = Field(default_factory=list)

    recommendation: Literal["accept", "revise", "reject"] = "accept"


class Critic(ABC):
    @abstractmethod
    def review(
        self,
        candidate: CandidateExperiment,
        context: PlanningContext,
    ) -> CriticReview:
        raise NotImplementedError


class MockCritic(Critic):
    def review(
        self,
        candidate: CandidateExperiment,
        context: PlanningContext,
    ) -> CriticReview:
        changes = dict(candidate.parameter_changes or {})
        fp = parameter_fingerprint(changes)
        tested = set(context.tested_parameter_fingerprints or [])

        strengths: list[str] = []
        weaknesses: list[str] = []
        revisions: list[str] = []

        novelty: Literal["new", "partially_redundant", "duplicate"] = "new"
        if fp in tested:
            novelty = "duplicate"
            weaknesses.append("Parameter fingerprint already tested.")
        elif len(changes) > 2:
            novelty = "partially_redundant"
            weaknesses.append("Multiple variables changed; confounding risk.")
            revisions.append("Reduce to a single controlled variable when possible.")

        if candidate.experiment_type == "ablation":
            strengths.append("Ablation isolates module contribution under matched protocol.")
        if candidate.evidence_gap_addressed:
            strengths.append("Explicitly addresses documented evidence gaps.")
        else:
            weaknesses.append("No explicit evidence gap listed.")
            revisions.append("Link candidate to Claim Matrix / Evidence limitations.")

        cost = float((candidate.estimated_cost or {}).get("gpu_hours") or 1.0)
        cost_score = max(0.0, min(1.0, 1.0 - (cost / 10.0)))
        info_gain = 0.82 if candidate.experiment_type == "ablation" else 0.55
        if novelty == "duplicate":
            info_gain = 0.1
            recommendation: Literal["accept", "revise", "reject"] = "reject"
            validity: Literal["valid", "weak", "invalid"] = "invalid"
            risk: Literal["low", "medium", "high"] = "high"
        elif novelty == "partially_redundant" or len(changes) > 2:
            recommendation = "revise"
            validity = "weak"
            risk = "medium"
        else:
            recommendation = "accept"
            validity = "valid"
            risk = "low"

        # Exaggeration
        blob = f"{candidate.hypothesis} {candidate.rationale}".lower()
        if any(p in blob for p in ("sota", "state-of-the-art", "proven")):
            weaknesses.append("Hypothesis language may overclaim scientific strength.")
            recommendation = "revise"
            validity = "weak"
            revisions.append("Remove SOTA / proven wording; keep exploratory claims.")

        return CriticReview(
            candidate_id=candidate.candidate_id,
            scientific_validity=validity,
            novelty_status=novelty,
            expected_information_gain=info_gain,
            cost_effectiveness=cost_score,
            risk_level=risk,
            strengths=strengths,
            weaknesses=weaknesses,
            required_revisions=revisions,
            recommendation=recommendation,
        )
