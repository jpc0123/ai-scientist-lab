"""Patch proposal service — propose/verify/approve/reject (no apply in v1.6.1–3)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import sessionmaker

from scientist_lab.domain.models import new_id
from scientist_lab.patching.fingerprint import fingerprint_diff
from scientist_lab.patching.models import (
    PatchApproval,
    PatchProposal,
    PatchVerification,
)
from scientist_lab.patching.path_policy import PathPolicy
from scientist_lab.patching.repository import PatchRepository
from scientist_lab.patching.verifier import PatchVerifier


def build_mock_unified_diff(
    *,
    relative_path: str = (
        "src/scientist_lab/tasks/rgbt_detection/feedback_rules.py"
    ),
) -> str:
    """Deterministic mock Unified Diff within the allow-list."""
    return (
        f"diff --git a/{relative_path} b/{relative_path}\n"
        f"--- a/{relative_path}\n"
        f"+++ b/{relative_path}\n"
        "@@ -1,3 +1,6 @@\n"
        " # feedback rules\n"
        "+\n"
        "+# Mock patch: document an evidence-gap oriented comment.\n"
        "+# Addresses missing controlled fusion ablation rationale.\n"
        " def placeholder():\n"
        "     return True\n"
    )


class PatchingService:
    """Lifecycle for restricted PatchProposal objects.

    Important: this stage never applies patches to the main workspace and
    exposes no apply/commit/push APIs.
    """

    def __init__(
        self,
        session_factory: sessionmaker,
        *,
        policy: PathPolicy | None = None,
    ) -> None:
        self._repo = PatchRepository(session_factory)
        self.policy = policy or PathPolicy()
        self.verifier = PatchVerifier(self.policy)

    def propose_mock(
        self,
        project_id: str,
        *,
        title: str | None = None,
        rationale: str | None = None,
        evidence_gap_ids: list[str] | None = None,
        unified_diff: str | None = None,
        auto_verify: bool = True,
    ) -> dict[str, Any]:
        diff = unified_diff or build_mock_unified_diff()
        fp = fingerprint_diff(diff)
        duplicate = self._repo.find_by_fingerprint(fp)
        proposal = PatchProposal(
            patch_id=new_id("patch"),
            project_id=project_id,
            status="proposed",
            title=title
            or "Mock patch: annotate fusion ablation feedback rule",
            rationale=rationale
            or (
                "MockProvider proposes a documentation-level change addressing "
                "a missing controlled fusion ablation evidence gap."
            ),
            evidence_gap_ids=list(
                evidence_gap_ids
                or ["missing controlled fusion ablation"]
            ),
            unified_diff=diff,
            fingerprint_sha256=fp,
            provider="mock",
            metadata={"applied": False, "apply_available": False},
        )
        if auto_verify:
            verification = self.verifier.verify(
                diff,
                duplicate_of=duplicate.patch_id if duplicate else None,
            )
            proposal.verification = verification
            proposal.files_touched = list(verification.files_touched)
            proposal.fingerprint_sha256 = verification.fingerprint_sha256
            proposal.status = (
                "verified" if verification.ok else "rejected_by_verifier"
            )
        self._repo.upsert(proposal)
        return self._view(proposal)

    def show(self, patch_id: str) -> dict[str, Any]:
        return self._view(self._repo.require(patch_id))

    def verify(self, patch_id: str) -> dict[str, Any]:
        proposal = self._repo.require(patch_id)
        if proposal.status in {"approved", "merged"}:
            # Re-verify is allowed for audit, but does not apply.
            pass
        duplicate = self._repo.find_by_fingerprint(
            fingerprint_diff(proposal.unified_diff),
            exclude_patch_id=proposal.patch_id,
        )
        verification = self.verifier.verify(
            proposal.unified_diff,
            duplicate_of=duplicate.patch_id if duplicate else None,
        )
        proposal.verification = verification
        proposal.files_touched = list(verification.files_touched)
        proposal.fingerprint_sha256 = verification.fingerprint_sha256
        if proposal.status not in {"approved", "rejected", "merged", "discarded"}:
            proposal.status = (
                "verified" if verification.ok else "rejected_by_verifier"
            )
        self._repo.upsert(proposal)
        return self._view(proposal)

    def approve(self, patch_id: str, *, reason: str = "") -> dict[str, Any]:
        proposal = self._repo.require(patch_id)
        if proposal.status == "rejected_by_verifier":
            raise ValueError("cannot approve a patch rejected by verifier")
        if proposal.status == "rejected":
            raise ValueError("cannot approve a human-rejected patch")
        # Ensure verification exists and passed.
        if proposal.verification is None or not proposal.verification.ok:
            self.verify(patch_id)
            proposal = self._repo.require(patch_id)
            if proposal.verification is None or not proposal.verification.ok:
                raise ValueError("patch must pass verification before approval")
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        proposal.approval = PatchApproval(
            decision="approved",
            reason=reason or "human approved for future sandbox stage",
            decided_at=now,
            decided_by="human",
        )
        proposal.status = "approved"
        proposal.metadata = {
            **dict(proposal.metadata or {}),
            "applied": False,
            "apply_available": False,
            "note": "Approval does not apply the patch in v1.6.1–1.6.3",
        }
        self._repo.upsert(proposal)
        return self._view(proposal)

    def reject(self, patch_id: str, *, reason: str = "") -> dict[str, Any]:
        proposal = self._repo.require(patch_id)
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        proposal.approval = PatchApproval(
            decision="rejected",
            reason=reason or "human rejected",
            decided_at=now,
            decided_by="human",
        )
        proposal.status = "rejected"
        proposal.metadata = {
            **dict(proposal.metadata or {}),
            "applied": False,
            "apply_available": False,
        }
        self._repo.upsert(proposal)
        return self._view(proposal)

    def list_patches(self, project_id: str) -> list[dict[str, Any]]:
        return [self._view(item) for item in self._repo.list_for_project(project_id)]

    def _view(self, proposal: PatchProposal) -> dict[str, Any]:
        data = proposal.model_dump(mode="json")
        data["can_apply"] = False
        data["applied"] = False
        return data
