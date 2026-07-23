"""Release freeze helpers (v1.8.1); packaging tar.gz lands in v1.8.3."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from scientist_lab.release.models import WorkspaceSnapshot, WorkspaceView


def describe_workspaces(
    *,
    project_root: Path,
    sandbox_root: Path,
) -> list[WorkspaceView]:
    return [
        WorkspaceView(
            kind="main",
            path=str(project_root),
            writable=False,
            main_workspace_modified=False,
            notes="Main source tree is read-only for releases; never mutated by ReleaseService.",
        ),
        WorkspaceView(
            kind="sandbox",
            path=str(sandbox_root),
            writable=True,
            main_workspace_modified=False,
            notes="Patch sandboxes under outputs/_patch_sandboxes.",
        ),
        WorkspaceView(
            kind="frozen",
            path=None,
            writable=False,
            main_workspace_modified=False,
            notes="Frozen snapshots are metadata-only; no full tree copy.",
        ),
    ]


def read_git_identity(project_root: Path) -> dict[str, str | None]:
    commit: str | None = None
    branch: str | None = None
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(project_root),
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            or None
        )
    except Exception:  # noqa: BLE001
        commit = None
    try:
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=str(project_root),
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            or None
        )
    except Exception:  # noqa: BLE001
        branch = None
    return {"git_commit": commit, "git_branch": branch}


def build_workspace_snapshot(
    *,
    project_id: str,
    project_root: Path,
    sandbox_paths: list[str] | None = None,
    notes: str = "",
) -> WorkspaceSnapshot:
    identity = read_git_identity(project_root)
    return WorkspaceSnapshot(
        project_id=project_id,
        git_commit=identity.get("git_commit"),
        git_branch=identity.get("git_branch"),
        main_workspace_modified=False,
        sandbox_paths=list(sandbox_paths or []),
        notes=notes
        or "Freeze records metadata only; main workspace was not modified.",
    )


def release_manifest_stub(package_dump: dict[str, Any]) -> dict[str, Any]:
    """Minimal manifest used at freeze time; archive hashing is v1.8.3."""
    return {
        "schema_version": "1.8.1",
        "release_id": package_dump.get("release_id"),
        "project_id": package_dump.get("project_id"),
        "status": package_dump.get("status"),
        "tree_id": package_dump.get("tree_id"),
        "report_id": package_dump.get("report_id"),
        "audit_bundle_id": package_dump.get("audit_bundle_id"),
        "patch_ids": package_dump.get("patch_ids") or [],
        "main_workspace_modified": False,
        "snapshot": package_dump.get("snapshot"),
    }
