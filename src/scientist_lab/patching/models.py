"""Restricted source patching models (v1.6)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


PatchStatus = Literal[
    "draft",
    "proposed",
    "verified",
    "rejected_by_verifier",
    "approved",
    "rejected",
    "applied_sandbox",
    "failed_sandbox",
    "evidence_recorded",
    "merged",
    "discarded",
]


class DiffHunk(BaseModel):
    header: str
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str] = Field(default_factory=list)


class DiffFile(BaseModel):
    old_path: str | None = None
    new_path: str | None = None
    is_new_file: bool = False
    is_deleted_file: bool = False
    is_binary: bool = False
    hunks: list[DiffHunk] = Field(default_factory=list)

    @property
    def path(self) -> str:
        return self.new_path or self.old_path or ""


class ParsedDiff(BaseModel):
    files: list[DiffFile] = Field(default_factory=list)
    raw: str = ""

    @property
    def paths(self) -> list[str]:
        return [f.path for f in self.files if f.path]


class VerificationIssue(BaseModel):
    code: str
    message: str
    path: str | None = None
    blocking: bool = True


class PatchVerification(BaseModel):
    ok: bool
    issues: list[VerificationIssue] = Field(default_factory=list)
    files_touched: list[str] = Field(default_factory=list)
    fingerprint_sha256: str = ""
    duplicate_of: str | None = None
    checked_at: str = ""


class ApprovalContentSeal(BaseModel):
    """Fingerprints captured at human approval time (v2.2.5)."""

    source_commit: str = ""
    context_sha256: str = ""
    patch_sha256: str = ""
    proposal_sha256: str = ""
    sealed_at: str = ""
    seal_version: str = "v2.2.5"


class PatchApproval(BaseModel):
    decision: Literal["approved", "rejected"]
    reason: str = ""
    decided_at: str = ""
    decided_by: str = "human"
    content_seal: ApprovalContentSeal | None = None
    seal_valid: bool | None = None
    invalidated: bool = False
    invalidate_reason: str = ""


class PatchProposal(BaseModel):
    patch_id: str
    project_id: str
    status: PatchStatus = "proposed"
    title: str
    rationale: str = ""
    evidence_gap_ids: list[str] = Field(default_factory=list)
    unified_diff: str
    files_touched: list[str] = Field(default_factory=list)
    fingerprint_sha256: str = ""
    provider: str = "mock"
    verification: PatchVerification | None = None
    approval: PatchApproval | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    def touch(self) -> None:
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        if not self.created_at:
            self.created_at = now
        self.updated_at = now
