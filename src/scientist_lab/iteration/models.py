from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


IterationStatus = Literal[
    "created",
    "feedback_ready",
    "proposal_ready",
    "waiting_approval",
    "approved",
    "running",
    "comparing",
    "waiting_decision",
    "completed",
    "rejected",
    "failed",
    "cancelled",
    "stopped_no_recommendation",
]


class IterationSession(BaseModel):
    iteration_id: str
    project_id: str

    source_baseline_node_id: str
    source_candidate_node_id: str
    proposed_node_id: str | None = None

    status: IterationStatus

    seeds: list[int] = Field(default_factory=list)

    feedback_path: str | None = None
    proposal_path: str | None = None

    proposal_sha256: str | None = None
    approved_sha256: str | None = None

    execution_ids: list[str] = Field(default_factory=list)
    comparison_paths: list[str] = Field(default_factory=list)

    selected_node_id: str | None = None
    decision_id: str | None = None

    error_type: str | None = None
    error_message: str | None = None

    created_at: str
    updated_at: str


class ApprovalRecord(BaseModel):
    approval_id: str
    iteration_id: str

    decision: Literal["approved", "rejected"]
    reason: str | None = None

    contract_path: str
    contract_sha256: str

    created_at: str
