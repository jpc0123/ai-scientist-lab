"""Workspace and release package models (v1.8.1)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from scientist_lab.domain.models import new_id


WorkspaceKind = Literal["main", "sandbox", "frozen"]

ReleaseStatus = Literal["draft", "frozen", "exported", "discarded"]


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
