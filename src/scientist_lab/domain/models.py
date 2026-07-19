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
    project_id: str
    title: str
    research_goal: str
    status: ProjectStatus = ProjectStatus.ACTIVE
    created_at: str
    updated_at: str


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
