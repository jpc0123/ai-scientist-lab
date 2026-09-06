from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class RunnerSubmission(BaseModel):
    execution_id: str
    runner_profile: str
    status: str
    submitted_at: datetime | str


class RunnerStatus(BaseModel):
    execution_id: str
    status: str
    progress: float | None = None
    stage: str | None = None
    updated_at: datetime | str | None = None


class RunnerLogChunk(BaseModel):
    execution_id: str
    cursor: str | None = None
    next_cursor: str | None = None
    content: str = ""
    complete: bool = False


class RemoteJobBinding(BaseModel):
    execution_id: str
    request_id: str
    job_id: str
    runner_profile: str
    endpoint: str
    project_id: str
    node_id: str
    output_directory: str
    contract: dict[str, Any] = Field(default_factory=dict)
    submitted_at: str
    status: str = "queued"
