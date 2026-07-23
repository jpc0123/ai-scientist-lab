"""Release Candidate manifest builder / verifier (v1.9.8)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scientist_lab.domain.models import utc_now_iso
from scientist_lab.release.models import ReleaseCandidate


MANIFEST_SCHEMA = "1.9.8"


def build_release_manifest(candidate: ReleaseCandidate) -> dict[str, Any]:
    payload = {
        "schema_version": MANIFEST_SCHEMA,
        "release_candidate_id": candidate.release_candidate_id,
        "version": candidate.version,
        "project_id": candidate.project_id,
        "base_tag": candidate.base_tag,
        "commit_sha": candidate.commit_sha,
        "included_patch_ids": list(candidate.included_patch_ids),
        "included_merge_candidate_ids": list(candidate.included_merge_candidate_ids),
        "regression_test_summary": dict(candidate.regression_test_summary or {}),
        "acceptance_summary": dict(candidate.acceptance_summary or {}),
        "status": candidate.status,
        "created_at": candidate.created_at,
        "notes": candidate.notes,
        "main_workspace_modified": False,
        "remote_published": False,
        "built_at": utc_now_iso(),
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    payload["manifest_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def write_release_manifest(
    candidate: ReleaseCandidate,
    *,
    outputs_root: Path,
) -> tuple[Path, dict[str, Any]]:
    manifest = build_release_manifest(candidate)
    out_dir = (
        Path(outputs_root)
        / (candidate.project_id or "_release")
        / "release_candidates"
        / candidate.release_candidate_id
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "release_manifest.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path, manifest


def verify_release_manifest(
    candidate: ReleaseCandidate,
    *,
    expected_commit_sha: str | None = None,
) -> dict[str, Any]:
    """Validate RC consistency; does not publish remotely."""
    issues: list[str] = []
    warnings: list[str] = []
    if not (candidate.version or "").strip():
        issues.append("version missing")
    if not (candidate.commit_sha or "").strip():
        issues.append("commit_sha missing")
    if not candidate.included_merge_candidate_ids:
        warnings.append("no merge candidates linked")
    if not candidate.included_patch_ids:
        warnings.append("no patch ids linked")
    if expected_commit_sha and candidate.commit_sha != expected_commit_sha:
        issues.append(
            f"commit_sha mismatch expected={expected_commit_sha} "
            f"actual={candidate.commit_sha}"
        )
    manifest = candidate.manifest or {}
    if manifest:
        rebuilt = build_release_manifest(candidate)
        # Compare identity fields (ignore built_at / hash drift from timestamps).
        for key in (
            "release_candidate_id",
            "version",
            "commit_sha",
            "base_tag",
            "project_id",
        ):
            if manifest.get(key) != rebuilt.get(key):
                issues.append(f"manifest field mismatch: {key}")
        if manifest.get("remote_published") is True:
            issues.append("remote_published must be false in v1.9.8")
        if manifest.get("main_workspace_modified") is True:
            issues.append("main_workspace_modified must be false")
    elif candidate.status in {"verified", "approved", "released"}:
        issues.append("manifest missing for non-draft RC")

    valid = not issues
    return {
        "valid": valid,
        "blocking_issues": issues,
        "warnings": warnings,
        "release_candidate_id": candidate.release_candidate_id,
        "checked_at": utc_now_iso(),
    }
