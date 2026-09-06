from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator


class RunnerProfile(BaseModel):
    profile_key: str
    runner_type: Literal["local_docker", "remote_docker"]
    enabled: bool = True
    endpoint: str | None = None
    auth_token_env: str | None = None
    allowed_environment_keys: list[str] = Field(default_factory=list)
    default_timeout_seconds: int = 7200
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | str
    updated_at: datetime | str

    @field_validator("profile_key")
    @classmethod
    def _key_ok(cls, value: str) -> str:
        text = (value or "").strip()
        if not text:
            raise ValueError("profile_key required")
        if any(ch.isspace() for ch in text):
            raise ValueError("profile_key must not contain whitespace")
        return text

    @field_validator("endpoint")
    @classmethod
    def _endpoint_ok(cls, value: str | None) -> str | None:
        if value is None or value == "":
            return None
        text = value.strip().rstrip("/")
        if not (text.startswith("http://") or text.startswith("https://")):
            raise ValueError("endpoint must start with http:// or https://")
        # Validate roughly via HttpUrl
        HttpUrl(text)
        return text
