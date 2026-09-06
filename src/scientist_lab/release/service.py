"""Release and workspace management service (v1.8.1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy.orm import sessionmaker

from scientist_lab.domain.models import utc_now_iso
from scientist_lab.release.models import ReleasePackage
from scientist_lab.release.package import (
    build_workspace_snapshot,
    describe_workspaces,
    release_manifest_stub,
)
from scientist_lab.release.repository import ReleaseRepository


class ReleaseService:
    """Create / freeze / discard release packages without touching main workspace."""

    def __init__(
        self,
        session_factory: sessionmaker,
        *,
        project_root: Path,
        outputs_root: Path,
        sandbox_root: Path | None = None,
    ) -> None:
        self._repo = ReleaseRepository(session_factory)
        self.project_root = Path(project_root)
        self.outputs_root = Path(outputs_root)
        self.sandbox_root = Path(sandbox_root or (self.outputs_root / "_patch_sandboxes"))

    def workspace_summary(self) -> dict[str, Any]:
        views = describe_workspaces(
            project_root=self.project_root,
            sandbox_root=self.sandbox_root,
        )
        return {
            "workspaces": [v.model_dump(mode="json") for v in views],
            "main_workspace_modified": False,
            "can_write_main": False,
            "notes": (
                "Releases never mutate the main workspace; "
                "patch apply remains sandbox-only."
            ),
        }

    def list_releases(
        self, *, project_id: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        return [
            item.model_dump(mode="json")
            for item in self._repo.list_all(project_id=project_id, limit=limit)
        ]

    def show(self, release_id: str) -> dict[str, Any]:
        return self._enrich(self._repo.require(release_id))

    def create(
        self,
        *,
        project_id: str,
        title: str = "",
        tree_id: str | None = None,
        report_id: str | None = None,
        audit_bundle_id: str | None = None,
        patch_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        if not (project_id or "").strip():
            raise ValueError("project_id is required")
        package = ReleasePackage(
            project_id=project_id.strip(),
            title=(title or "").strip() or f"Release for {project_id}",
            tree_id=tree_id,
            report_id=report_id,
            audit_bundle_id=audit_bundle_id,
            patch_ids=list(patch_ids or []),
            status="draft",
        )
        self._repo.upsert(package)
        return self._enrich(package)

    def freeze(self, release_id: str, *, notes: str = "") -> dict[str, Any]:
        package = self._repo.require(release_id)
        if package.status == "discarded":
            raise ValueError("cannot freeze a discarded release")
        if package.status == "exported":
            raise ValueError("release already exported; create a new release to freeze")
        sandbox_paths: list[str] = []
        if self.sandbox_root.is_dir():
            for patch_id in package.patch_ids:
                candidate = self.sandbox_root / patch_id
                if candidate.is_dir():
                    sandbox_paths.append(str(candidate))
        snapshot = build_workspace_snapshot(
            project_id=package.project_id,
            project_root=self.project_root,
            sandbox_paths=sandbox_paths,
            notes=notes,
        )
        package.snapshot = snapshot
        package.status = "frozen"
        package.frozen_at = snapshot.captured_at
        package.manifest = release_manifest_stub(package.model_dump(mode="json"))
        package.manifest["main_workspace_modified"] = False
        self._repo.upsert(package)
        return self._enrich(package)

    def discard(self, release_id: str, *, reason: str = "") -> dict[str, Any]:
        package = self._repo.require(release_id)
        if package.status == "discarded":
            return self._enrich(package)
        package.status = "discarded"
        package.discarded_at = utc_now_iso()
        if reason:
            package.manifest = {
                **(package.manifest or {}),
                "discard_reason": reason,
                "main_workspace_modified": False,
            }
        self._repo.upsert(package)
        return self._enrich(package)

    def mark_exported(
        self,
        release_id: str,
        *,
        export_path: str,
        archive_path: str | None = None,
        archive_sha256: str | None = None,
    ) -> dict[str, Any]:
        """Record export path after audit/directory export (tar.gz in v1.8.3)."""
        package = self._repo.require(release_id)
        if package.status == "discarded":
            raise ValueError("cannot export a discarded release")
        if package.status == "draft":
            raise ValueError("freeze the release before export")
        package.export_path = export_path
        package.archive_path = archive_path
        package.archive_sha256 = archive_sha256
        package.status = "exported"
        package.exported_at = utc_now_iso()
        package.manifest = {
            **(package.manifest or {}),
            "export_path": export_path,
            "archive_path": archive_path,
            "archive_sha256": archive_sha256,
            "main_workspace_modified": False,
        }
        self._repo.upsert(package)
        return self._enrich(package)

    def _enrich(self, package: ReleasePackage) -> dict[str, Any]:
        data = package.model_dump(mode="json")
        data["can_freeze"] = package.status == "draft"
        data["can_export"] = package.status in {"frozen", "exported"}
        data["can_discard"] = package.status in {"draft", "frozen"}
        data["can_write_main"] = False
        data["main_workspace_modified"] = False
        return data
