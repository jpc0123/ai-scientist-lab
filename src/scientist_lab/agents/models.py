from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


ExperimentType = Literal[
    "improve",
    "ablation",
    "debug",
    "robustness",
    "efficiency",
    "replication",
]

PlanStatus = Literal[
    "generated",
    "verified",
    "reviewed",
    "ranked",
    "approved",
    "rejected",
    "contract_generated",
    "executed",
    "planner_failed",
]

CandidateStatus = Literal[
    "generated",
    "verified",
    "reviewed",
    "ranked",
    "approved",
    "rejected",
    "contract_generated",
    "executed",
]


class ExpectedOutcome(BaseModel):
    metric: str
    direction: Literal["increase", "decrease", "maintain"]
    rationale: str


class CandidateExperiment(BaseModel):
    candidate_id: str
    parent_node_id: str

    title: str
    hypothesis: str
    experiment_type: ExperimentType

    parameter_changes: dict[str, Any] = Field(default_factory=dict)

    expected_outcomes: list[ExpectedOutcome] = Field(default_factory=list)
    evidence_gap_addressed: list[str] = Field(default_factory=list)

    success_criteria: dict[str, Any] = Field(default_factory=dict)
    failure_criteria: dict[str, Any] = Field(default_factory=dict)

    estimated_cost: dict[str, Any] = Field(default_factory=dict)
    priority: float = Field(default=0.5, ge=0.0, le=1.0)

    rationale: str = ""
    claim_limitations: list[str] = Field(default_factory=list)

    @field_validator("candidate_id", "parent_node_id", "title", "hypothesis")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        text = (value or "").strip()
        if not text:
            raise ValueError("field must be non-empty")
        return text


class PlannerOutput(BaseModel):
    project_id: str
    reasoning_summary: str
    candidates: list[CandidateExperiment] = Field(default_factory=list)
    stop_recommended: bool = False
    stop_reason: str | None = None


class RoundFeedbackSummary(BaseModel):
    """Deterministic summary of a completed research-loop round (v2.1.2).

    Round 2+ PlanningContext must include this object; without it, real Planner
    calls for subsequent rounds are blocked.
    """

    source_round: int

    parent_node_id: str
    executed_node_id: str

    metric_deltas: dict[str, float | None] = Field(default_factory=dict)
    stability_deltas: dict[str, float | None] = Field(default_factory=dict)
    resource_deltas: dict[str, float | None] = Field(default_factory=dict)

    evidence_added: list[str] = Field(default_factory=list)
    claims_changed: list[str] = Field(default_factory=list)
    comparison_ids: list[str] = Field(default_factory=list)

    success_criteria_met: list[str] = Field(default_factory=list)
    failure_criteria_met: list[str] = Field(default_factory=list)

    unresolved_gaps: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    outcome_label: Literal[
        "improved",
        "degraded",
        "comparable",
        "unstable",
        "costlier",
        "uncertain",
        "failed",
    ] = "uncertain"

    failure_category: str | None = None
    error_summary: str | None = None
    failed_stage: str | None = None
    recoverability: str | None = None

    previous_hypothesis: str | None = None
    executed_parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("parent_node_id", "executed_node_id")
    @classmethod
    def _non_empty_ids(cls, value: str) -> str:
        text = (value or "").strip()
        if not text:
            raise ValueError("node id must be non-empty")
        return text


class PlanningContext(BaseModel):
    project_id: str
    research_goal: str

    protocol: dict[str, Any] = Field(default_factory=dict)
    current_best_node_id: str | None = None

    nodes: list[dict[str, Any]] = Field(default_factory=list)
    comparisons: list[dict[str, Any]] = Field(default_factory=list)
    evidence_records: list[dict[str, Any]] = Field(default_factory=list)
    claim_support_matrix: dict[str, Any] = Field(default_factory=dict)

    tested_parameter_fingerprints: list[str] = Field(default_factory=list)
    remaining_budget: dict[str, Any] = Field(default_factory=dict)

    allowed_parameter_changes: list[str] = Field(default_factory=list)
    blocked_actions: list[str] = Field(default_factory=list)

    # --- v2.1.2: multi-round feedback fields ---
    loop_session_id: str | None = None
    loop_round_number: int | None = None
    round_feedback_summary: RoundFeedbackSummary | None = None
    recent_execution_summary: dict[str, Any] = Field(default_factory=dict)
    previous_planner_hypothesis: str | None = None
    previous_parameter_changes: dict[str, Any] = Field(default_factory=dict)

    # Audit helpers (not for LLM reasoning)
    context_sha256: str | None = None
    protocol_id: str | None = None


class CandidateVerification(BaseModel):
    candidate_id: str
    valid: bool
    blocking_issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    parameter_fingerprint: str | None = None
    estimated_cost: dict[str, Any] = Field(default_factory=dict)


class ExperimentPlanRecord(BaseModel):
    plan_id: str
    project_id: str
    status: PlanStatus = "generated"

    context_json: dict[str, Any] = Field(default_factory=dict)
    planner_output_json: dict[str, Any] = Field(default_factory=dict)

    model_provider: str | None = "mock"
    model_name: str | None = "mock-planner-v1"
    prompt_version: str | None = "mock_v1"

    context_sha256: str | None = None
    output_sha256: str | None = None

    created_at: datetime | None = None
    updated_at: datetime | None = None


class ExperimentCandidateRecord(BaseModel):
    candidate_id: str
    plan_id: str
    candidate_json: dict[str, Any] = Field(default_factory=dict)
    verification_json: dict[str, Any] | None = None
    critic_review_json: dict[str, Any] | None = None
    final_score: float | None = None
    rank: int | None = None
    status: CandidateStatus = "generated"
    created_at: datetime | None = None


# Fields that candidates must never touch (safety).
BLOCKED_PARAMETER_KEYS = frozenset(
    {
        "entrypoint",
        "environment_key",
        "dataset_reference",
        "code_reference",
        "code_version",
        "host_path",
        "image",
        "docker_image",
        "network",
        "command",
        "shell",
        "script",
    }
)

DEFAULT_BLOCKED_ACTIONS = [
    "run_docker",
    "emit_shell",
    "change_environment_key",
    "change_dataset_reference",
    "change_code_reference",
    "edit_database",
    "claim_verified_scientific_result",
]
