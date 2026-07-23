from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from scientist_lab.domain import (
    ErrorType,
    JobStatus,
    NodeStage,
    NodeStatus,
    NodeType,
    ProjectStatus,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


class ResearchProject(BaseModel):
    """Research project with workbench lifecycle fields (v2.0.1)."""

    project_id: str = Field(default_factory=lambda: new_id("project"))
    title: str
    research_goal: str = ""
    description: str = ""
    research_question: str = ""
    task_type: str = "general_ml"
    status: ProjectStatus = ProjectStatus.DRAFT
    dataset_keys: list[str] = Field(default_factory=list)
    protocol_ids: list[str] = Field(default_factory=list)
    runner_profile_keys: list[str] = Field(default_factory=list)
    default_llm_profile_id: str | None = None
    default_tree_id: str | None = None
    expected_metrics: dict[str, Any] = Field(default_factory=dict)
    constraints: dict[str, Any] = Field(default_factory=dict)
    protocol_draft: dict[str, Any] = Field(default_factory=dict)
    wizard_completed: bool = False
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)

    def touch(self) -> None:
        self.updated_at = utc_now_iso()

    def normalized_status(self) -> ProjectStatus:
        if self.status == ProjectStatus.ACTIVE:
            return ProjectStatus.READY
        return self.status


class ExperimentNode(BaseModel):
    node_id: str
    project_id: str
    parent_node_id: str | None = None
    node_type: NodeType = NodeType.SMOKE
    stage: NodeStage = NodeStage.INTAKE
    hypothesis: str | None = None
    status: NodeStatus = NodeStatus.PLANNED
    depth: int = 0
    contract_json: dict[str, Any] = Field(default_factory=dict)
    feedback_json: dict[str, Any] | None = None
    score: float | None = None
    created_at: str
    updated_at: str


class ExecutionAttempt(BaseModel):
    execution_id: str
    node_id: str
    attempt_index: int
    runner_profile: str
    status: JobStatus = JobStatus.CREATED
    container_id: str | None = None
    image_reference: str
    code_version: str | None = None
    dataset_version: str | None = None
    result_json: dict[str, Any] | None = None
    error_json: dict[str, Any] | None = None
    started_at: str | None = None
    completed_at: str | None = None
    created_at: str


class ExperimentArtifact(BaseModel):
    artifact_id: str
    execution_id: str
    artifact_type: str
    relative_path: str
    size_bytes: int | None = None
    sha256: str | None = None
    metadata_json: dict[str, Any] | None = None
    created_at: str


class ExecutionError(BaseModel):
    error_type: ErrorType
    stage: str
    message: str
    retryable: bool = False
    suggested_action: str | None = None
    log_artifact: str | None = None


class ExperimentDecision(BaseModel):
    """Node-selection decision with optional evidence / protocol linkage (v0.9.7)."""

    decision_id: str
    selected_node_id: str
    decision_type: str
    alternatives: list[str] = Field(default_factory=list)
    reason: str
    evidence_strength: str = "weak"
    baseline_node_id: str | None = None
    candidate_node_id: str | None = None
    claim_level: str | None = None

    supporting_evidence_ids: list[str] = Field(default_factory=list)
    claim_matrix_path: str | None = None
    protocol_id: str | None = None
    project_id: str | None = None

    recorded_at: str
    decision_path: str | None = None
