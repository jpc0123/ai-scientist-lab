"""LLM provider request/response models (v1.3)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


LLMPurpose = Literal["planner", "critic", "reviewer", "other"]


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMRequest(BaseModel):
    purpose: LLMPurpose = "other"
    messages: list[dict[str, Any]] = Field(default_factory=list)
    response_schema: dict[str, Any] | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("messages")
    @classmethod
    def _messages_ok(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not isinstance(value, list) or not value:
            raise ValueError("messages must be a non-empty list")
        return value


class LLMResponse(BaseModel):
    request_id: str
    content: str
    parsed_json: dict[str, Any] | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: float = 0.0
    provider: str
    model: str
    request_fingerprint: str
    schema_valid: bool = True
    schema_errors: list[str] = Field(default_factory=list)
    created_at: datetime | None = None
    provider_request_id: str | None = None
    usage_unknown: bool = False
    api_mode: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMCallRecord(BaseModel):
    """Persisted audit record for one provider call."""

    request_id: str
    request_fingerprint: str
    purpose: LLMPurpose
    provider: str
    model: str

    request: dict[str, Any] = Field(default_factory=dict)
    response: dict[str, Any] = Field(default_factory=dict)

    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: float = 0.0
    schema_valid: bool = True
    schema_errors: list[str] = Field(default_factory=list)

    created_at: datetime | None = None
    project_id: str | None = None
    path: str | None = None
