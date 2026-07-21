from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class AblationVariant(BaseModel):
    variant_id: str
    title: str
    parameter_changes: dict[str, Any] = Field(default_factory=dict)
    expected_effect: str = ""
    # Optional stable node id; planner may assign if missing.
    node_id: str | None = None

    @field_validator("variant_id", "title")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        text = (value or "").strip()
        if not text:
            raise ValueError("field must be non-empty")
        return text


class AblationPlan(BaseModel):
    ablation_id: str
    project_id: str
    reference_node_id: str
    protocol_id: str

    controlled_variables: list[str]
    variants: list[AblationVariant]

    title: str | None = None
    created_at: datetime | None = None

    @field_validator("ablation_id", "project_id", "reference_node_id", "protocol_id")
    @classmethod
    def _required_ids(cls, value: str) -> str:
        text = (value or "").strip()
        if not text:
            raise ValueError("field must be non-empty")
        return text

    @field_validator("controlled_variables")
    @classmethod
    def _vars_ok(cls, value: list[str]) -> list[str]:
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        if not cleaned:
            raise ValueError("controlled_variables must be non-empty")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("controlled_variables must be unique")
        return cleaned

    @field_validator("variants")
    @classmethod
    def _variants_ok(cls, value: list[AblationVariant]) -> list[AblationVariant]:
        if not value:
            raise ValueError("variants must be non-empty")
        return value


class AblationVerificationReport(BaseModel):
    valid: bool
    warnings: list[str] = Field(default_factory=list)
    blocking_issues: list[str] = Field(default_factory=list)
    ablation_id: str | None = None
    deduplicated_variant_ids: list[str] = Field(default_factory=list)
