"""Workspace, release package, and merge candidate models (v1.8 / v1.9)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from scientist_lab.domain.models import new_id


WorkspaceKind = Literal["main", "sandbox", "frozen"]

ReleaseStatus = Literal["draft", "frozen", "exported", "discarded"]

MergeCandidateStatus = Literal[
    "created",
    "preparing",
    "testing",
    "waiting_approval",
    "approved",
    "committing",
    "merged",
    "failed",
    "rolled_back",
    "rejected",
    "merge_conflict",
]


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class WorkspaceView(BaseModel):
    """Read-only view of a workspace layer (never mutates main)."""

    kind: WorkspaceKind
    path: str | None = None
    writable: bool = False
    main_workspace_modified: bool = False
    notes: str = ""


class WorkspaceSnapshot(BaseModel):
    """Freeze-time metadata; does not copy the main source tree."""

    snapshot_id: str = Field(default_factory=lambda: new_id("ws_snap"))
    project_id: str
    git_commit: str | None = None
    git_branch: str | None = None
    main_workspace_modified: bool = False
    sandbox_paths: list[str] = Field(default_factory=list)
    captured_at: str = Field(default_factory=_now)
    notes: str = ""


class ReleasePackage(BaseModel):
    """Release candidate binding report/audit/merge-intent patches."""

    release_id: str = Field(default_factory=lambda: new_id("rel"))
    project_id: str
    status: ReleaseStatus = "draft"
    title: str = ""
    tree_id: str | None = None
    report_id: str | None = None
    audit_bundle_id: str | None = None
    patch_ids: list[str] = Field(default_factory=list)
    snapshot: WorkspaceSnapshot | None = None
    export_path: str | None = None
    archive_path: str | None = None
    archive_sha256: str | None = None
    manifest: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)
    frozen_at: str | None = None
    exported_at: str | None = None
    discarded_at: str | None = None

    def touch(self) -> None:
        self.updated_at = _now()


class MergeCandidate(BaseModel):
    """Isolated merge attempt for an evidenced patch (v1.9)."""

    merge_candidate_id: str = Field(default_factory=lambda: new_id("mc"))
    project_id: str
    patch_id: str
    patch_evidence_id: str
    source_commit: str
    target_branch: str
    workspace_path: str
    patch_sha256: str
    diff_sha256: str
    status: MergeCandidateStatus = "created"
    test_run_id: str | None = None
    commit_sha: str | None = None
    apply_check_ok: bool | None = None
    apply_check_detail: str = ""
    reverify: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)
    error: str | None = None

    def touch(self) -> None:
        self.updated_at = _now()


class MergeApproval(BaseModel):
    approval_id: str = Field(default_factory=lambda: new_id("ma"))
    merge_candidate_id: str
    decision: Literal["approve", "reject", "request_changes"]
    reason: str = ""
    approved_by: str = "human"
    approved_at: str = Field(default_factory=_now)


class RollbackRecord(BaseModel):
    rollback_id: str = Field(default_factory=lambda: new_id("rb"))
    merge_candidate_id: str
    original_commit_sha: str
    rollback_commit_sha: str | None = None
    trigger: str = ""
    reason: str = ""
    status: str = "pending"
    created_at: str = Field(default_factory=_now)
