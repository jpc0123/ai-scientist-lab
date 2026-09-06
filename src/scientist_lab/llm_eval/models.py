"""LLM evaluation models for v1.5 quality governance."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


TaskType = Literal["planner", "critic", "safety"]


class ExpectedProperties(BaseModel):
    """Structured expectations — not verbatim gold answers."""

    experiment_type: list[str] = Field(default_factory=list)
    must_address_gaps: list[str] = Field(default_factory=list)
    forbidden_parameter_changes: list[str] = Field(default_factory=list)
    allowed_parameter_changes: list[str] = Field(default_factory=list)
    stop_expected: bool | None = None
    expected_recommendation: Literal["accept", "revise", "reject"] | None = None
    must_detect: list[str] = Field(default_factory=list)
    must_not_accept: bool = False
    safety_violation_expected: bool = False
    safety_kinds: list[str] = Field(default_factory=list)
    max_parameter_changes: int | None = None
    require_success_criteria: bool = False
    require_failure_criteria: bool = False
    require_claim_limitations: bool = False
    budget_gpu_hours_max: float | None = None
    notes: str = ""


class EvaluationCase(BaseModel):
    case_id: str
    task_type: TaskType
    title: str = ""
    context: dict[str, Any] = Field(default_factory=dict)
    candidate: dict[str, Any] | None = None
    output_fixture: dict[str, Any] | None = None
    expected_properties: ExpectedProperties = Field(default_factory=ExpectedProperties)
    tags: list[str] = Field(default_factory=list)


class MetricScore(BaseModel):
    name: str
    passed: bool
    detail: str = ""
    hard_fail: bool = False


class CaseGrade(BaseModel):
    case_id: str
    task_type: TaskType
    passed: bool
    hard_fail: bool = False
    scores: list[MetricScore] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def score_map(self) -> dict[str, bool]:
        return {item.name: item.passed for item in self.scores}


class EvaluationSuiteManifest(BaseModel):
    suite_version: str
    title: str = ""
    description: str = ""
    grader_version: str = "v1"
    schema_version: str = "v1"
    case_files: list[str] = Field(default_factory=list)
    expected_files: list[str] = Field(default_factory=list)
    case_counts: dict[str, int] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationSuite(BaseModel):
    manifest: EvaluationSuiteManifest
    cases: list[EvaluationCase] = Field(default_factory=list)

    def by_task(self, task_type: TaskType) -> list[EvaluationCase]:
        return [c for c in self.cases if c.task_type == task_type]
