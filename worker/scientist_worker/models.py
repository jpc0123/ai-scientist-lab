from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


JobStatus = Literal[
    "received",
    "validating",
    "queued",
    "preparing",
    "running",
    "collecting",
    "completed",
    "validation_failed",
    "prepare_failed",
    "failed",
    "timed_out",
    "cancelled",
    "cancel_requested",
    "artifact_failed",
]


WORKER_TRANSITIONS: dict[str, set[str]] = {
    "received": {"validating", "cancelled"},
    "validating": {"queued", "validation_failed", "cancelled"},
    "queued": {"preparing", "cancelled"},
    "preparing": {"running", "prepare_failed", "cancelled"},
    "running": {"collecting", "failed", "timed_out", "cancelled", "cancel_requested"},
    "cancel_requested": {"cancelled", "collecting"},
    "collecting": {"completed", "artifact_failed"},
}


class JobSubmitRequest(BaseModel):
    request_id: str
    execution_id: str
    contract: dict[str, Any]
    environment: dict[str, Any] = Field(default_factory=dict)
    dataset_mounts: list[dict[str, Any]] = Field(default_factory=list)
    code_bundle: dict[str, Any] = Field(default_factory=dict)
    artifact_policy: dict[str, Any] = Field(default_factory=dict)


class JobSubmitResponse(BaseModel):
    job_id: str
    execution_id: str
    status: str
    submitted_at: str
    request_id: str | None = None


class JobStatusResponse(BaseModel):
    job_id: str
    request_id: str
    execution_id: str
    status: str
    progress: float | None = None
    stage: str | None = None
    environment_key: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    submitted_at: str
    started_at: str | None = None
    finished_at: str | None = None
    updated_at: str


class LogChunkResponse(BaseModel):
    execution_id: str
    job_id: str
    cursor: str | None
    next_cursor: str | None
    content: str
    complete: bool


class JobRecord(BaseModel):
    job_id: str
    request_id: str
    execution_id: str
    status: str
    contract_json: dict[str, Any]
    environment_key: str
    container_id: str | None = None
    output_path: str
    log_path: str
    error_type: str | None = None
    error_message: str | None = None
    progress: float | None = None
    stage: str | None = None
    submitted_at: str
    started_at: str | None = None
    finished_at: str | None = None
    updated_at: str
    result_json: dict[str, Any] | None = None
