from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


ExecutionMode = Literal["fast_eval", "full_train"]
ClaimLevel = Literal["exploratory_comparison", "benchmark_evidence"]


class ExperimentProtocol(BaseModel):
    protocol_id: str
    project_id: str
    title: str

    task_type: str
    dataset_reference: str
    dataset_version: str
    split_reference: str

    environment_key: str
    code_reference: str
    code_version: str

    execution_mode: ExecutionMode

    seeds: list[int]

    primary_metric: str
    secondary_metrics: list[str] = Field(default_factory=list)

    fixed_parameters: dict[str, Any] = Field(default_factory=dict)
    allowed_variables: list[str] = Field(default_factory=list)

    resource_metrics: list[str] = Field(default_factory=list)

    claim_level: ClaimLevel

    created_at: datetime

    @field_validator("protocol_id", "project_id", "title")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        text = (value or "").strip()
        if not text:
            raise ValueError("field must be non-empty")
        return text

    @field_validator("seeds")
    @classmethod
    def _seeds_ok(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("seeds must be non-empty")
        if len(set(value)) != len(value):
            raise ValueError("seeds must be unique")
        return list(value)


class ProtocolVerificationReport(BaseModel):
    valid: bool
    warnings: list[str] = Field(default_factory=list)
    blocking_issues: list[str] = Field(default_factory=list)
    protocol_id: str | None = None
    contract_node_id: str | None = None
