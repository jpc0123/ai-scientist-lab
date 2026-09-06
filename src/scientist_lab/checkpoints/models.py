from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


CheckpointRole = Literal["best", "last", "intermediate", "other"]
CheckpointFormat = Literal["pt", "npz", "bin", "safetensors", "txt", "other"]


class CheckpointRecord(BaseModel):
    checkpoint_id: str
    project_id: str
    execution_id: str
    node_id: str | None = None
    relative_path: str
    role: CheckpointRole = "other"
    format: CheckpointFormat = "other"
    size_bytes: int
    sha256: str
    baseline_key: str | None = None
    verified: bool = False
    verification_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str

    @field_validator("relative_path")
    @classmethod
    def _path_ok(cls, value: str) -> str:
        text = (value or "").replace("\\", "/").strip().lstrip("/")
        if not text:
            raise ValueError("relative_path required")
        if text.startswith("/") or ".." in text.split("/"):
            raise ValueError(f"unsafe relative_path: {value}")
        return text
