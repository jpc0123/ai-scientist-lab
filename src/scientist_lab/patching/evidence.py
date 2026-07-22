"""PatchEvidence builder and merge-decision records (v1.6.6)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from scientist_lab.domain.models import new_id
from scientist_lab.patching.models import PatchProposal
from scientist_lab.storage.artifact_store import write_json


class PatchEvidence(BaseModel):
    """Evidence produced by sandbox apply + optional sandbox tests.

    Does not imply the patch was merged into the main working tree.
    """

    evidence_id: str
    patch_id: str
    project_id: str
    title: str = ""
    fingerprint_sha256: str = ""
    files_touched: list[str] = Field(default_factory=list)
    provider: str = "mock"
    sandbox_dir: str = ""
    sandbox_files_written: list[str] = Field(default_factory=list)
    sandbox_tests: dict[str, Any] | None = None
    sandbox_tests_ok: bool | None = None
    main_workspace_modified: bool = False
    evidence_strength: Literal["weak", "moderate", "strong"] = "weak"
    limitations: list[str] = Field(default_factory=list)
    recorded_at: str = ""
    artifact_path: str | None = None


class PatchMergeDecision(BaseModel):
    """Human decision about whether a patch may be merged later.

    ``merge`` records intent only — v1.6 never writes the main workspace.
    """

    decision: Literal["merge", "discard"]
    reason: str = ""
    decided_at: str = ""
    decided_by: str = "human"
    applied_main: bool = False


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def build_patch_evidence(
    proposal: PatchProposal,
    *,
    outputs_root: Path,
) -> PatchEvidence:
    meta = dict(proposal.metadata or {})
    limitations: list[str] = [
        "Sandbox-only validation; main workspace was not modified.",
        "Human merge still required; v1.6 does not auto-apply to main tree.",
    ]
    tests = meta.get("sandbox_tests")
    tests_ok = meta.get("sandbox_tests_ok")
    if tests is None:
        limitations.append("Sandbox tests were not run before recording evidence.")
        strength: Literal["weak", "moderate", "strong"] = "weak"
    elif tests_ok is True:
        strength = "moderate"
    else:
        limitations.append("Sandbox tests reported failures.")
        strength = "weak"

    evidence_id = new_id("patch_ev")
    evidence = PatchEvidence(
        evidence_id=evidence_id,
        patch_id=proposal.patch_id,
        project_id=proposal.project_id,
        title=proposal.title,
        fingerprint_sha256=proposal.fingerprint_sha256,
        files_touched=list(proposal.files_touched),
        provider=proposal.provider,
        sandbox_dir=str(meta.get("sandbox_dir") or ""),
        sandbox_files_written=list(meta.get("sandbox_files_written") or []),
        sandbox_tests=dict(tests) if isinstance(tests, dict) else None,
        sandbox_tests_ok=bool(tests_ok) if tests_ok is not None else None,
        main_workspace_modified=False,
        evidence_strength=strength,
        limitations=limitations,
        recorded_at=_now(),
    )

    out_dir = Path(outputs_root) / proposal.project_id / "patch_evidence"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{evidence_id}.json"
    payload = evidence.model_dump(mode="json")
    payload["artifact_path"] = str(path)
    write_json(path, payload)
    evidence.artifact_path = str(path)

    # Also mirror into the sandbox when present.
    sandbox_dir = meta.get("sandbox_dir")
    if sandbox_dir:
        sandbox_path = Path(sandbox_dir) / "PATCH_EVIDENCE.json"
        try:
            write_json(sandbox_path, payload)
        except OSError:
            pass
    return evidence
