"""Request bodies for controlled write endpoints (v1.7.1)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ReasonBody(BaseModel):
    reason: str = ""


class PatchTestBody(BaseModel):
    profile: Literal["smoke", "syntax", "unit", "mock_experiment"] = "smoke"


class PatchApplySandboxBody(BaseModel):
    force: bool = False


class PatchRecordEvidenceBody(BaseModel):
    require_tests: bool = False


class PatchDecideMergeBody(BaseModel):
    decision: Literal["merge", "discard"]
    reason: str = ""


class CodeContextBuildBody(BaseModel):
    """Build a restricted CodeContextBundle (v2.2.1; no provider call)."""

    digits_demo: bool = True
    project_id: str | None = None
    persist: bool = True
    bundle_id: str | None = None


class PatchProposeRealBody(BaseModel):
    """Propose a restricted Unified Diff via real provider (v2.2.2/4)."""

    bundle_id: str
    allow_network: bool = False
    provider: str = "openai-compatible"
    real_only: bool = True


class PatchExportReplayBody(BaseModel):
    """Export a redacted Patch Replay Bundle under outputs/ (v2.2.8)."""

    output_dir: str | None = None
    label: str = "patch_replay"


class PatchCheckSealBody(BaseModel):
    persist: bool = True


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


class PlanNextBody(BaseModel):
    protocol_id: str | None = None
    current_best_node_id: str | None = None
    max_new_nodes: int = 3
    max_gpu_hours: float = 12.0
    provider: str = "mock"
    allow_network: bool = False
    model_profile: str | None = None
    require_quality_gate: bool = False


class TreeCreateBody(BaseModel):
    project_id: str
    root_node_id: str
    protocol_id: str
    max_depth: int = 3
    max_nodes: int = 8
    max_children: int = 3
    tree_id: str | None = None


class TreePlanNextBody(BaseModel):
    rescore: bool = True
    max_gpu_hours: float = 12.0
    provider: str = "mock"
    allow_network: bool = False
    model_profile: str | None = None
    require_quality_gate: bool = False


class TreeApproveBody(BaseModel):
    candidate_id: str
    seeds: list[int] = Field(default_factory=list)


class TreeAdvanceBody(BaseModel):
    tree_node_id: str | None = None


class TreeStopBody(BaseModel):
    reason: str


class ClaimMatrixBuildBody(BaseModel):
    project_id: str
    protocol_id: str | None = None


class RecoverBody(BaseModel):
    dry_run: bool = True


class DemoCreateBody(BaseModel):
    kind: Literal["digits", "rgbt-debug"]
    force: bool = False


class ProjectExportBody(BaseModel):
    output_dir: str


class ProjectImportBody(BaseModel):
    path: str
    force: bool = False


class RealLoopCreateBody(BaseModel):
    project_id: str
    profile_id: str
    protocol_id: str
    rounds: int = 2
    baseline_node_ids: list[str] = Field(default_factory=list)
    tree_id: str | None = None


class RealLoopPlanBody(BaseModel):
    round_number: int | None = None
    allow_network: bool = False
    provider: str = "openai-compatible"


class RealLoopReviewBody(BaseModel):
    round_number: int | None = None
    allow_network: bool = False
    provider: str = "openai-compatible"


class RealLoopApproveBody(BaseModel):
    candidate_id: str
    round_number: int | None = None
    seeds: list[int] = Field(default_factory=list)


class RealLoopRejectBody(BaseModel):
    candidate_id: str | None = None
    round_number: int | None = None
    reason: str | None = None


class RealLoopExecuteBody(BaseModel):
    round_number: int | None = None
    seeds: list[int] = Field(default_factory=list)
    wait: bool = True


class RealLoopExportBody(BaseModel):
    output_dir: str | None = None
    allow_incomplete: bool = False


class RealLoopVerifyBody(BaseModel):
    round_number: int = 2
    plan_id: str | None = None
    persist: bool = True


class LlmConfigUpdateBody(BaseModel):
    """Update process-local LLM connection settings (secrets stay in runtime/)."""

    provider: str | None = None
    base_url: str | None = None
    model: str | None = None
    timeout_seconds: float | None = Field(default=None, gt=0)
    api_key: str | None = None
    allow_network: bool | None = None
    clear_api_key: bool = False


class LlmProfileRegisterBody(BaseModel):
    profile_id: str
    provider: str = "mock"
    model: str = "mock-planner-v1"
    api_mode: str = "chat_completions"
    planner_prompt_version: str = "mock_v1"
    critic_prompt_version: str = "mock_v1"
    temperature: float = 0.0
    max_output_tokens: int = 2048
    enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class LocalRunActionBody(BaseModel):
    """Demo UI: replay / dry-run. --live / --execute stay off unless confirmed."""

    action: Literal["llm_plan_replay", "llm_review_replay", "manager_run"]
    live: bool = False
    execute: bool = False
    confirm_live: bool = False
    confirm_execute: bool = False
    provider: str = "mock"


class ConsoleChatBody(BaseModel):
    """Command-center chat. GPU stays off. Keys never echo."""

    message: str = Field(min_length=1, max_length=8000)
    history: list[dict[str, Any]] = Field(default_factory=list)
    live: bool = False
    agent: str = "planner"
    session_id: str | None = None


class ConsoleSessionCreateBody(BaseModel):
    title: str | None = None
    workspace_id: str | None = None


class ConsoleSessionPatchBody(BaseModel):
    title: str | None = None
    pinned: bool | None = None
    active_agent: str | None = None
    workspace_id: str | None = None


class ConsoleMemoryBody(BaseModel):
    text: str = Field(min_length=1, max_length=500)


class ConsoleWorkspaceCreateBody(BaseModel):
    title: str | None = None
    kind: str | None = "project"
    project_id: str | None = None
    pack_id: str | None = None


class ConsoleWorkspacePatchBody(BaseModel):
    title: str | None = None
    kind: str | None = None
    project_id: str | None = None
    pack_id: str | None = None
    active: bool | None = None
