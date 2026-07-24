"""Release Candidate create / verify / show (v1.9.8). Local only — no remote publish."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy.orm import sessionmaker

from scientist_lab.domain.models import utc_now_iso
from scientist_lab.release.audit import append_audit_event
from scientist_lab.release.git_adapter import GitAdapter
from scientist_lab.release.manifest import (
    verify_release_manifest,
    write_release_manifest,
)
from scientist_lab.release.models import ReleaseCandidate
from scientist_lab.release.repository import (
    MergeCandidateRepository,
    ReleaseCandidateRepository,
)


class ReleaseCandidateService:
    def __init__(
        self,
        session_factory: sessionmaker,
        *,
        project_root: Path,
        outputs_root: Path,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.outputs_root = Path(outputs_root).resolve()
        self._repo = ReleaseCandidateRepository(session_factory)
        self._merges = MergeCandidateRepository(session_factory)
        self._git: GitAdapter | None = None

    @property
    def git(self) -> GitAdapter:
        if self._git is None:
            self._git = GitAdapter(self.project_root)
        return self._git

    def show(self, release_candidate_id: str) -> dict[str, Any]:
        return self._view(self._repo.require(release_candidate_id))

    def list_all(
        self, *, project_id: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        return [
            self._view(item)
            for item in self._repo.list_all(project_id=project_id, limit=limit)
        ]

    def create(
        self,
        *,
        version: str,
        project_id: str = "",
        base_tag: str = "",
        merge_candidate_ids: list[str] | None = None,
        notes: str = "",
    ) -> dict[str, Any]:
        version = (version or "").strip()
        if not version:
            raise ValueError("version is required")

        ids = list(merge_candidate_ids or [])
        patch_ids: list[str] = []
        regression: dict[str, Any] = {"items": []}
        acceptance: dict[str, Any] = {"items": []}
        resolved_project = (project_id or "").strip()

        for mc_id in ids:
            mc = self._merges.require(mc_id)
            if mc.status != "merged":
                raise ValueError(
                    f"merge candidate {mc_id} must be merged; got {mc.status!r}"
                )
            if not resolved_project:
                resolved_project = mc.project_id
            elif mc.project_id != resolved_project:
                raise ValueError("all merge candidates must share the same project_id")
            patch_ids.append(mc.patch_id)
            meta = dict(mc.metadata or {})
            regression["items"].append(
                {
                    "merge_candidate_id": mc_id,
                    "last_test": meta.get("last_test"),
                    "post_merge_check": meta.get("post_merge_check"),
                }
            )
            acceptance["items"].append(
                {
                    "merge_candidate_id": mc_id,
                    "patch_id": mc.patch_id,
                    "patch_evidence_id": mc.patch_evidence_id,
                    "finalize": meta.get("finalize"),
                }
            )

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_patches: list[str] = []
        for pid in patch_ids:
            if pid not in seen:
                seen.add(pid)
                unique_patches.append(pid)

        commit_sha = self.git.rev_parse("HEAD")
        if not base_tag:
            # Prefer nearest annotated/lightweight tag pointing at HEAD when present.
            base_tag = self._tag_at_head() or "untagged"

        candidate = ReleaseCandidate(
            version=version,
            project_id=resolved_project or project_id or "_release",
            base_tag=base_tag,
            commit_sha=commit_sha,
            included_patch_ids=unique_patches,
            included_merge_candidate_ids=ids,
            regression_test_summary=regression,
            acceptance_summary=acceptance,
            status="draft",
            notes=notes
            or "Local Release Candidate only; remote publish is out of scope for v1.9.8.",
        )
        path, manifest = write_release_manifest(
            candidate, outputs_root=self.outputs_root
        )
        candidate.manifest_path = str(path)
        candidate.manifest = manifest
        self._repo.upsert(candidate)
        append_audit_event(
            self.outputs_root,
            project_id=candidate.project_id,
            event_type="release_candidate_created",
            payload={
                "release_candidate_id": candidate.release_candidate_id,
                "version": candidate.version,
                "commit_sha": candidate.commit_sha,
            },
        )
        return self._view(candidate)

    def verify(self, release_candidate_id: str) -> dict[str, Any]:
        candidate = self._repo.require(release_candidate_id)
        head = self.git.rev_parse("HEAD")
        result = verify_release_manifest(candidate, expected_commit_sha=None)
        # Soft check: warn if HEAD drifted (RC still valid as historical snapshot).
        if candidate.commit_sha and candidate.commit_sha != head:
            warnings = list(result.get("warnings") or [])
            warnings.append(
                f"HEAD is {head}; RC snapshot commit is {candidate.commit_sha}"
            )
            result["warnings"] = warnings

        candidate.verification = result
        if result.get("valid"):
            candidate.status = "verified"
            candidate.verified_at = utc_now_iso()
        self._repo.upsert(candidate)
        append_audit_event(
            self.outputs_root,
            project_id=candidate.project_id,
            event_type="release_candidate_verified",
            payload={
                "release_candidate_id": candidate.release_candidate_id,
                "valid": result.get("valid"),
                "blocking_issues": result.get("blocking_issues"),
            },
        )
        view = self._view(candidate)
        view["verification"] = result
        return view

    def _tag_at_head(self) -> str | None:
        return None

    def _view(self, candidate: ReleaseCandidate) -> dict[str, Any]:
        data = candidate.model_dump(mode="json")
        data["can_verify"] = True
        data["can_publish_remote"] = False
        data["remote_published"] = False
        data["main_workspace_modified"] = False
        return data
