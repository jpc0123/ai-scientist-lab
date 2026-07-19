from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class HypothesisStatus(StrEnum):
    SUPPORTED = "supported"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"
    SUPPORTED_WITH_REPEATED_EVIDENCE = "supported_with_repeated_evidence"
    REJECTED_WITH_REPEATED_EVIDENCE = "rejected_with_repeated_evidence"
    INVALID_COMPARISON = "invalid_comparison"


class VerificationReport(BaseModel):
    valid: bool
    warnings: list[str] = Field(default_factory=list)
    blocking_issues: list[str] = Field(default_factory=list)


class ComparisonResult(BaseModel):
    baseline_node_id: str
    candidate_node_id: str
    baseline_execution_id: str
    candidate_execution_id: str

    parameter_changes: dict[str, Any] = Field(default_factory=dict)
    metric_changes: dict[str, Any] = Field(default_factory=dict)

    primary_metric: str | None = None
    primary_metric_delta: float | None = None
    relative_improvement_percent: float | None = None

    verification: VerificationReport
    experiment_valid: bool
    hypothesis_status: HypothesisStatus
    conclusion: str

    # Raw attempts for UI / debugging (not part of the scientific summary alone)
    execution_a: dict[str, Any] | None = None
    execution_b: dict[str, Any] | None = None
