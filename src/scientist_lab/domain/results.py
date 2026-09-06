from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from scientist_lab.domain import JobStatus
from scientist_lab.domain.models import ExecutionError, ExperimentArtifact


class SubmissionResult(BaseModel):
    execution_id: str
    status: JobStatus
    runner_profile: str


class ExecutionStatus(BaseModel):
    execution_id: str
    status: JobStatus
    progress: float | None = None
    message: str | None = None
    container_id: str | None = None


class ArtifactManifestItem(BaseModel):
    type: str
    path: str
    required: bool = True


class ArtifactManifest(BaseModel):
    schema_version: str = "1.0"
    artifacts: list[ArtifactManifestItem] = Field(default_factory=list)


class ExecutionResult(BaseModel):
    execution_id: str
    status: JobStatus
    return_code: int | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[ExperimentArtifact] = Field(default_factory=list)
    error: ExecutionError | None = None
    output_directory: str | None = None
