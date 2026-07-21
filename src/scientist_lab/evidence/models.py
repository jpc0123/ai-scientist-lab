from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


EvidenceType = Literal[
    "single_execution",
    "repeated_experiment",
    "paired_comparison",
    "ablation",
    "resource_comparison",
    "failure_analysis",
]

EvidenceStrength = Literal["weak", "moderate", "strong"]

ClaimSupportStatus = Literal[
    "supported",
    "partially_supported",
    "unsupported",
    "blocked",
]


class EvidenceRecord(BaseModel):
    evidence_id: str
    project_id: str

    evidence_type: EvidenceType

    source_node_ids: list[str] = Field(default_factory=list)
    source_execution_ids: list[str] = Field(default_factory=list)
    source_artifact_ids: list[str] = Field(default_factory=list)

    protocol_id: str | None = None
    metric_summary: dict[str, Any] = Field(default_factory=dict)

    evidence_strength: EvidenceStrength = "weak"
    # Explicit dual tags when scientific strength is capped below engineering signal.
    engineering_evidence_level: str | None = None
    scientific_evidence_level: str | None = None

    limitations: list[str] = Field(default_factory=list)
    valid: bool = True

    claim_level: str | None = None
    comparison_path: str | None = None
    created_at: datetime | None = None

    @field_validator("evidence_id", "project_id")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        text = (value or "").strip()
        if not text:
            raise ValueError("field must be non-empty")
        return text


class ScientificClaim(BaseModel):
    claim_id: str
    project_id: str
    claim_text: str

    claim_type: str
    required_evidence_types: list[str] = Field(default_factory=list)

    support_status: ClaimSupportStatus = "unsupported"

    supporting_evidence_ids: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    reason: str | None = None

    @field_validator("claim_id", "project_id", "claim_text", "claim_type")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        text = (value or "").strip()
        if not text:
            raise ValueError("field must be non-empty")
        return text


class ClaimSupportMatrix(BaseModel):
    project_id: str
    protocol_id: str | None = None
    claims: list[ScientificClaim] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    created_at: datetime | None = None
    matrix_path: str | None = None

    @field_validator("project_id")
    @classmethod
    def _project_ok(cls, value: str) -> str:
        text = (value or "").strip()
        if not text:
            raise ValueError("project_id must be non-empty")
        return text
