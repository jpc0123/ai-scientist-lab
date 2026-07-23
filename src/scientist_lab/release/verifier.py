"""Fingerprint / gate helpers for controlled merge (v1.9.1)."""

from __future__ import annotations

import hashlib
from typing import Any

from scientist_lab.patching.models import PatchProposal
from scientist_lab.patching.path_policy import PathPolicy
from scientist_lab.patching.verifier import PatchVerifier


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def reverify_patch(proposal: PatchProposal) -> dict[str, Any]:
    """Re-run path policy + static verifier; compare fingerprint."""
    policy = PathPolicy()
    verification = PatchVerifier(policy).verify(proposal.unified_diff)
    fingerprint = proposal.fingerprint_sha256 or ""
    diff_sha = sha256_text(proposal.unified_diff)
    ok = bool(verification.ok) and (
        not fingerprint or fingerprint == verification.fingerprint_sha256
    )
    return {
        "ok": ok,
        "verification_ok": verification.ok,
        "fingerprint_sha256": fingerprint or verification.fingerprint_sha256,
        "diff_sha256": diff_sha,
        "issues": [i.model_dump(mode="json") for i in verification.issues],
        "files_touched": list(verification.files_touched),
    }
