"""Request bodies for controlled write endpoints (v1.7.1)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ReasonBody(BaseModel):
    reason: str = ""


class PatchTestBody(BaseModel):
    profile: Literal["smoke", "syntax", "mock_experiment"] = "smoke"


class PatchApplySandboxBody(BaseModel):
    force: bool = False


class PatchRecordEvidenceBody(BaseModel):
    require_tests: bool = False


class PatchDecideMergeBody(BaseModel):
    decision: Literal["merge", "discard"]
    reason: str = ""


class IterationFinalizeBody(BaseModel):
    selected_node_id: str
    reason: str
    decision_type: str = "efficiency_tradeoff"
    evidence_strength: str = "moderate"


class CandidateRejectBody(BaseModel):
    reason: str | None = Field(default=None)


class CompareExecutionsBody(BaseModel):
    execution_id_a: str
    execution_id_b: str


class CompareNodesBody(BaseModel):
    node_id_a: str
    node_id_b: str


class CompareTriadBody(BaseModel):
    rgb_node_id: str = "rgbt_fast_node_001"
    thermal_node_id: str = "rgbt_fast_node_002"
    fusion_node_id: str = "rgbt_fast_node_003"
    write_report: bool = True


class ReportBuildBody(BaseModel):
    project_id: str
    tree_id: str | None = None
    protocol_id: str | None = None


class AuditBuildBody(BaseModel):
    project_id: str
    tree_id: str | None = None
    protocol_id: str | None = None
    report_id: str | None = None


class AuditExportBody(BaseModel):
    output_dir: str
    release_id: str | None = None


class ReleaseCreateBody(BaseModel):
    project_id: str
    title: str = ""
    tree_id: str | None = None
    report_id: str | None = None
    audit_bundle_id: str | None = None
    patch_ids: list[str] = Field(default_factory=list)


class ReleaseFreezeBody(BaseModel):
    notes: str = ""


class ReleaseDiscardBody(BaseModel):
    reason: str = ""


class MergePrepareBody(BaseModel):
    patch_id: str
    target_branch: str | None = None


class MergeTestBody(BaseModel):
    profile: Literal[
        "syntax", "unit", "smoke", "full_regression", "acceptance"
    ] = "smoke"


class MergeApproveBody(BaseModel):
    reason: str = ""
    approved_by: str = "human"


class MergeRejectBody(BaseModel):
    reason: str = ""
    approved_by: str = "human"


class MergeFinalizeBody(BaseModel):
    post_merge_profile: (
        Literal["syntax", "unit", "smoke", "full_regression", "acceptance"] | None
    ) = "syntax"
    auto_rollback_on_failure: bool = True


class MergeRollbackBody(BaseModel):
    reason: str = ""
    trigger: str = "human"


class ReleaseCandidateCreateBody(BaseModel):
    version: str
    project_id: str = ""
    base_tag: str = ""
    merge_candidate_ids: list[str] = Field(default_factory=list)
    notes: str = ""


class ProjectCreateBody(BaseModel):
    title: str
    research_question: str = ""
    research_goal: str = ""
    description: str = ""
    task_type: str = "general_ml"
    dataset_keys: list[str] = Field(default_factory=list)
    protocol_ids: list[str] = Field(default_factory=list)
    runner_profile_keys: list[str] = Field(default_factory=list)
    default_llm_profile_id: str | None = None
    expected_metrics: dict[str, Any] = Field(default_factory=dict)
    constraints: dict[str, Any] = Field(default_factory=dict)
    protocol_draft: dict[str, Any] = Field(default_factory=dict)
    project_id: str | None = None
    mark_ready: bool = True
