"""Controlled merge prepare / show (v1.9.1). Commit/merge land in later subversions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy.orm import sessionmaker

from scientist_lab.domain.models import new_id
from scientist_lab.patching.models import PatchProposal
from scientist_lab.patching.repository import PatchRepository
from scientist_lab.release.git_adapter import GitAdapter, GitAdapterError
from scientist_lab.release.models import MergeCandidate
from scientist_lab.release.repository import MergeCandidateRepository
from scientist_lab.release.verifier import reverify_patch, sha256_text
from scientist_lab.release.workspace_service import WorkspaceRegistry


class MergeService:
    def __init__(
        self,
        session_factory: sessionmaker,
        *,
        project_root: Path,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self._patches = PatchRepository(session_factory)
        self._repo = MergeCandidateRepository(session_factory)
        self.git = GitAdapter(self.project_root)
        self.workspaces = WorkspaceRegistry(self.project_root, git=self.git)

    def show(self, merge_candidate_id: str) -> dict[str, Any]:
        return self._view(self._repo.require(merge_candidate_id))

    def list_candidates(
        self,
        *,
        project_id: str | None = None,
        patch_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        return [
            self._view(item)
            for item in self._repo.list_all(
                project_id=project_id, patch_id=patch_id, limit=limit
            )
        ]

    def prepare(
        self,
        patch_id: str,
        *,
        target_branch: str | None = None,
    ) -> dict[str, Any]:
        """Create isolated worktree + MergeCandidate. Does not commit or merge."""
        proposal = self._patches.require(patch_id)
        self._require_ready_for_prepare(proposal)

        evidence = dict((proposal.metadata or {}).get("patch_evidence") or {})
        evidence_id = str(
            evidence.get("evidence_id")
            or (proposal.metadata or {}).get("patch_evidence_id")
            or ""
        )
        if not evidence_id:
            raise ValueError("PatchEvidence required before merge-prepare")

        reverify = reverify_patch(proposal)
        if not reverify.get("ok"):
            raise ValueError(
                "patch re-verification failed; refuse MergeCandidate creation: "
                f"{reverify.get('issues')}"
            )

        branch = (target_branch or self.git.current_branch()).strip()
        if not branch or branch == "HEAD":
            branch = self.git.current_branch()
        source_commit = self.git.rev_parse(branch if target_branch else "HEAD")

        merge_candidate_id = new_id("mc")
        # Pre-create row in preparing state only after worktree succeeds;
        # allocate id first for path naming.
        try:
            ws = self.workspaces.create(
                merge_candidate_id=merge_candidate_id,
                source_commit=source_commit,
                patch_id=patch_id,
                target_branch=branch,
            )
        except GitAdapterError as exc:
            raise ValueError(f"failed to create isolated worktree: {exc}") from exc

        workspace_path = Path(ws["path"])
        patch_file = workspace_path / "PENDING_PATCH.diff"
        patch_file.write_text(proposal.unified_diff, encoding="utf-8")
        check = self.git.apply_check(patch_file, cwd=workspace_path)
        apply_ok = check.ok
        detail = check.stderr or check.stdout or ""

        status: str = "created"
        error: str | None = None
        if not apply_ok:
            status = "merge_conflict"
            error = detail or "git apply --check failed"

        candidate = MergeCandidate(
            merge_candidate_id=merge_candidate_id,
            project_id=proposal.project_id,
            patch_id=proposal.patch_id,
            patch_evidence_id=evidence_id,
            source_commit=source_commit,
            target_branch=branch,
            workspace_path=str(workspace_path),
            patch_sha256=str(
                reverify.get("fingerprint_sha256") or proposal.fingerprint_sha256
            ),
            diff_sha256=str(reverify.get("diff_sha256") or sha256_text(proposal.unified_diff)),
            status=status,  # type: ignore[arg-type]
            apply_check_ok=apply_ok,
            apply_check_detail=detail[:2000],
            reverify=reverify,
            metadata={
                "main_workspace_modified": False,
                "can_commit": False,
                "can_merge": False,
                "v19_stage": "1.9.1-prepare-only",
                "merge_decision": (proposal.metadata or {}).get("merge_decision"),
            },
            error=error,
        )
        self._repo.upsert(candidate)
        return self._view(candidate)

    def _require_ready_for_prepare(self, proposal: PatchProposal) -> None:
        meta = dict(proposal.metadata or {})
        evidence = meta.get("patch_evidence")
        if not evidence:
            raise ValueError("PatchEvidence missing; run patch-record-evidence first")
        decision = dict(meta.get("merge_decision") or {})
        if decision.get("decision") != "merge":
            raise ValueError(
                "merge decision must be 'merge' (run patch-decide-merge --decision merge)"
            )
        if proposal.status not in {"merged", "evidence_recorded"}:
            raise ValueError(
                "merge-prepare requires evidenced patch with merge intent; "
                f"got status={proposal.status!r}"
            )
        if meta.get("applied_main") is True:
            raise ValueError("patch already applied to main; refuse new MergeCandidate")

    def _view(self, candidate: MergeCandidate) -> dict[str, Any]:
        data = candidate.model_dump(mode="json")
        data["can_commit"] = False
        data["can_merge"] = False
        data["can_approve"] = False
        data["main_workspace_modified"] = False
        data["workspace_exists"] = Path(candidate.workspace_path).exists()
        return data
