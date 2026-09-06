"""Approval content seals — bind fingerprints so tampering voids approval (v2.2.5)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from scientist_lab.patching.fingerprint import fingerprint_diff
from scientist_lab.patching.models import (
    ApprovalContentSeal,
    PatchApproval,
    PatchProposal,
)


class ApprovalSealError(ValueError):
    """Raised when an approval seal is missing or no longer matches content."""


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def proposal_content_sha256(proposal: PatchProposal) -> str:
    """Stable hash of proposal body (excludes approval / volatile timestamps)."""
    payload = {
        "patch_id": proposal.patch_id,
        "project_id": proposal.project_id,
        "title": proposal.title,
        "rationale": proposal.rationale,
        "evidence_gap_ids": list(proposal.evidence_gap_ids or []),
        "unified_diff": proposal.unified_diff,
        "files_touched": list(proposal.files_touched or []),
        "fingerprint_sha256": proposal.fingerprint_sha256
        or fingerprint_diff(proposal.unified_diff),
        "provider": proposal.provider,
        "bundle_id": (proposal.metadata or {}).get("bundle_id"),
        "context_sha256": (proposal.metadata or {}).get("context_sha256"),
        "source_commit": (proposal.metadata or {}).get("source_commit"),
    }
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_approval_seal(proposal: PatchProposal) -> ApprovalContentSeal:
    meta = dict(proposal.metadata or {})
    patch_sha = proposal.fingerprint_sha256 or fingerprint_diff(proposal.unified_diff)
    return ApprovalContentSeal(
        source_commit=str(meta.get("source_commit") or ""),
        context_sha256=str(meta.get("context_sha256") or ""),
        patch_sha256=patch_sha,
        proposal_sha256=proposal_content_sha256(proposal),
        sealed_at=_now(),
        seal_version="v2.2.5",
    )


def current_seal_material(proposal: PatchProposal) -> dict[str, str]:
    meta = dict(proposal.metadata or {})
    return {
        "source_commit": str(meta.get("source_commit") or ""),
        "context_sha256": str(meta.get("context_sha256") or ""),
        "patch_sha256": proposal.fingerprint_sha256
        or fingerprint_diff(proposal.unified_diff),
        "proposal_sha256": proposal_content_sha256(proposal),
    }


def compare_approval_seal(
    proposal: PatchProposal,
    *,
    seal: ApprovalContentSeal | None = None,
) -> dict[str, Any]:
    """Return a seal validity report without mutating the proposal."""
    approval = proposal.approval
    sealed = seal
    if sealed is None and approval is not None:
        sealed = approval.content_seal
    if sealed is None:
        return {
            "ok": False,
            "code": "seal_missing",
            "message": "approval content seal is missing; re-approve required",
            "mismatches": ["seal_missing"],
            "expected": None,
            "observed": current_seal_material(proposal),
        }

    observed = current_seal_material(proposal)
    expected = {
        "source_commit": sealed.source_commit,
        "context_sha256": sealed.context_sha256,
        "patch_sha256": sealed.patch_sha256,
        "proposal_sha256": sealed.proposal_sha256,
    }
    mismatches: list[str] = []
    for key, want in expected.items():
        got = observed.get(key, "")
        if key in {"source_commit", "context_sha256"} and not want and not got:
            continue
        if want != got:
            mismatches.append(key)

    ok = not mismatches
    return {
        "ok": ok,
        "code": "ok" if ok else "seal_mismatch",
        "message": (
            "approval seal matches current content"
            if ok
            else "approval seal mismatch: " + ", ".join(mismatches)
        ),
        "mismatches": mismatches,
        "expected": expected,
        "observed": observed,
        "seal_version": sealed.seal_version,
    }


def attach_approval_seal(
    proposal: PatchProposal,
    *,
    decision: str,
    reason: str,
    decided_by: str = "human",
) -> PatchApproval:
    seal = build_approval_seal(proposal)
    return PatchApproval(
        decision=decision,  # type: ignore[arg-type]
        reason=reason,
        decided_at=_now(),
        decided_by=decided_by,
        content_seal=seal,
        seal_valid=True,
        invalidated=False,
        invalidate_reason="",
    )


def invalidate_approval(
    proposal: PatchProposal, *, reason: str
) -> PatchProposal:
    """Clear approved status when content no longer matches the seal."""
    if proposal.approval is not None:
        proposal.approval.seal_valid = False
        proposal.approval.invalidated = True
        proposal.approval.invalidate_reason = reason
    if proposal.status == "approved":
        proposal.status = "verified"
    meta = dict(proposal.metadata or {})
    meta["sandbox_apply_available"] = False
    meta["approval_invalidated"] = True
    meta["approval_invalidate_reason"] = reason
    proposal.metadata = meta
    return proposal
