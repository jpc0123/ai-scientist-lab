"""Research evidence reporting models (v1.2)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


ReportStatus = Literal[
    "context_built",
    "tables_ready",
    "draft",
    "verified",
    "exported",
    "failed",
    "build_interrupted",
]

ClaimSupportStatus = Literal[
    "supported",
    "partially_supported",
    "unsupported",
    "blocked",
]


class ReportContext(BaseModel):
    """Deterministic snapshot used to generate ResearchReport / Audit Bundle."""

    project_id: str
    research_goal: str = ""

    protocol_id: str | None = None
    protocol: dict[str, Any] = Field(default_factory=dict)

    tree_id: str | None = None
    tree: dict[str, Any] | None = None
    key_path: list[dict[str, Any]] = Field(default_factory=list)

    nodes: list[dict[str, Any]] = Field(default_factory=list)
    executions: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)

    evidence_records: list[dict[str, Any]] = Field(default_factory=list)
    claim_support_matrix: dict[str, Any] = Field(default_factory=dict)
    decisions: list[dict[str, Any]] = Field(default_factory=list)
    comparisons: list[dict[str, Any]] = Field(default_factory=list)
    ablations: list[dict[str, Any]] = Field(default_factory=list)
    failures: list[dict[str, Any]] = Field(default_factory=list)

    open_evidence_gaps: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    recommended_next_experiments: list[str] = Field(default_factory=list)

    dataset_references: list[str] = Field(default_factory=list)
    environment_keys: list[str] = Field(default_factory=list)
    image_references: list[str] = Field(default_factory=list)

    remaining_budget: dict[str, Any] = Field(default_factory=dict)

    context_sha256: str | None = None
    built_at: datetime | None = None
    builder_version: str = "v1.2.1"

    @field_validator("project_id")
    @classmethod
    def _project_ok(cls, value: str) -> str:
        text = (value or "").strip()
        if not text:
            raise ValueError("project_id must be non-empty")
        return text


class ResultTables(BaseModel):
    key_path: list[dict[str, Any]] = Field(default_factory=list)
    metrics: list[dict[str, Any]] = Field(default_factory=list)
    stability: list[dict[str, Any]] = Field(default_factory=list)
    resources: list[dict[str, Any]] = Field(default_factory=list)
    ablations: list[dict[str, Any]] = Field(default_factory=list)
    failures: list[dict[str, Any]] = Field(default_factory=list)
    claims: list[dict[str, Any]] = Field(default_factory=list)
    evidence_summary: list[dict[str, Any]] = Field(default_factory=list)


class ReportConclusion(BaseModel):
    conclusion_id: str
    text: str
    claim_id: str | None = None
    support_status: ClaimSupportStatus | str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    strength: str | None = None


class ResearchReport(BaseModel):
    report_id: str
    project_id: str
    status: ReportStatus = "draft"

    research_goal: str = ""
    protocol_id: str | None = None
    protocol_summary: dict[str, Any] = Field(default_factory=dict)

    dataset_references: list[str] = Field(default_factory=list)
    environment_keys: list[str] = Field(default_factory=list)
    image_references: list[str] = Field(default_factory=list)

    tree_id: str | None = None
    tree_summary: dict[str, Any] = Field(default_factory=dict)
    key_path: list[dict[str, Any]] = Field(default_factory=list)
    key_nodes: list[dict[str, Any]] = Field(default_factory=list)

    metrics_table: list[dict[str, Any]] = Field(default_factory=list)
    stability_table: list[dict[str, Any]] = Field(default_factory=list)
    resource_table: list[dict[str, Any]] = Field(default_factory=list)
    ablation_table: list[dict[str, Any]] = Field(default_factory=list)
    failure_table: list[dict[str, Any]] = Field(default_factory=list)

    evidence_records: list[dict[str, Any]] = Field(default_factory=list)
    supported_claims: list[dict[str, Any]] = Field(default_factory=list)
    blocked_claims: list[dict[str, Any]] = Field(default_factory=list)
    conclusions: list[ReportConclusion] = Field(default_factory=list)

    limitations: list[str] = Field(default_factory=list)
    open_evidence_gaps: list[str] = Field(default_factory=list)
    recommended_next_experiments: list[str] = Field(default_factory=list)

    context_sha256: str | None = None
    verification: dict[str, Any] = Field(default_factory=dict)
    markdown_path: str | None = None
    json_path: str | None = None
    summary_path: str | None = None

    created_at: datetime | None = None
    generator_version: str = "v1.2.4"

    @field_validator("report_id", "project_id")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        text = (value or "").strip()
        if not text:
            raise ValueError("field must be non-empty")
        return text


class ReproducibilityManifest(BaseModel):
    project_id: str
    report_id: str | None = None
    bundle_id: str | None = None

    git_commit: str | None = None
    git_tags: list[str] = Field(default_factory=list)
    api_versions: dict[str, str] = Field(default_factory=dict)

    protocol_id: str | None = None
    protocol_sha256: str | None = None
    dataset_versions: list[str] = Field(default_factory=list)
    environment_keys: list[str] = Field(default_factory=list)
    image_references: list[str] = Field(default_factory=list)

    contracts: list[dict[str, Any]] = Field(default_factory=list)
    executions: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    checkpoints: list[dict[str, Any]] = Field(default_factory=list)

    acceptance: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None


class ArtifactManifest(BaseModel):
    project_id: str
    bundle_id: str | None = None
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime | None = None


class AuditBundle(BaseModel):
    bundle_id: str
    project_id: str
    report_id: str | None = None
    tree_id: str | None = None
    root_dir: str | None = None
    status: str = "built"
    verification: dict[str, Any] = Field(default_factory=dict)
    paths: dict[str, str] = Field(default_factory=dict)
    created_at: datetime | None = None
