from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ExperimentRecommendation(BaseModel):
    recommendation_type: Literal[
        "intermediate_value",
        "expand_range",
        "repeat_seeds",
        "efficiency_tradeoff",
        "ablation",
        "stop",
    ]
    priority: float = Field(ge=0.0, le=1.0)
    rationale: str
    parameter_changes: dict[str, Any] = Field(default_factory=dict)
    status: Literal["active", "deferred", "already_evaluated"] = "active"
    already_evaluated_node_id: str | None = None


class FeedbackInput(BaseModel):
    baseline_node_id: str
    candidate_node_id: str
    comparison: dict[str, Any]
    baseline_contract: dict[str, Any]
    candidate_contract: dict[str, Any]
    research_goal: str = ""
    existing_node_parameters: list[dict[str, Any]] = Field(default_factory=list)


class FeedbackReport(BaseModel):
    baseline_node_id: str
    candidate_node_id: str

    hypothesis_status: Literal[
        "supported_with_repeated_evidence",
        "rejected_with_repeated_evidence",
        "inconclusive",
        "invalid_comparison",
    ]

    evidence_strength: Literal["weak", "moderate", "strong"]

    positive_findings: list[str] = Field(default_factory=list)
    negative_findings: list[str] = Field(default_factory=list)
    tradeoffs: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)

    recommendations: list[ExperimentRecommendation] = Field(default_factory=list)
    recommended_action: Literal[
        "continue",
        "verify",
        "revise",
        "stop",
        "compare_fast_eval_nodes",
        "prepare_fast_eval",
    ]

    comparison_summary: dict[str, Any] = Field(default_factory=dict)
