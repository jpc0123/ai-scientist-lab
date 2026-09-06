"""Code context models for real-provider restricted Diff (v2.2.1)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


SourceFileRole = Literal[
    "entrypoint",
    "module",
    "config",
    "test",
    "interface",
    "other",
]


class AllowedSourceFile(BaseModel):
    """A path the model may read / propose edits for."""

    path: str
    reason: str = ""
    role: SourceFileRole = "module"
    allow_create: bool = False


class SourceSnapshot(BaseModel):
    """Frozen file content for a single allowed path."""

    path: str
    content: str
    content_sha256: str
    size_bytes: int
    encoding: str = "utf-8"
    truncated: bool = False


class ContextSizeBudget(BaseModel):
    """Hard limits for code context packing (no LLM call)."""

    max_files: int = 20
    max_bytes_total: int = 200_000
    max_bytes_per_file: int = 80_000


class PatchRequest(BaseModel):
    """Inputs for building a restricted CodeContextBundle (no provider call).

    Also known as PatchGoal in the v2.2 product vocabulary.
    """

    request_id: str
    project_id: str
    goal: str
    failure_summary: str = ""
    test_errors: list[str] = Field(default_factory=list)
    evidence_gap_ids: list[str] = Field(default_factory=list)
    patch_target: str = ""
    allowed_files: list[AllowedSourceFile] = Field(default_factory=list)
    interface_notes: list[str] = Field(default_factory=list)
    source_commit: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# Product alias used in v2.2 plan docs.
PatchGoal = PatchRequest


class CodeContextBundle(BaseModel):
    """Stable, auditable code context for a future RealPatchPlanner."""

    bundle_id: str
    request_id: str
    project_id: str
    source_commit: str = ""
    goal: str = ""
    failure_summary: str = ""
    test_errors: list[str] = Field(default_factory=list)
    evidence_gap_ids: list[str] = Field(default_factory=list)
    patch_target: str = ""
    interface_notes: list[str] = Field(default_factory=list)
    allowed_files: list[AllowedSourceFile] = Field(default_factory=list)
    snapshots: list[SourceSnapshot] = Field(default_factory=list)
    excluded_paths: list[dict[str, str]] = Field(default_factory=list)
    budget: ContextSizeBudget = Field(default_factory=ContextSizeBudget)
    total_bytes: int = 0
    context_sha256: str = ""
    created_at: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    def touch(self) -> None:
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        if not self.created_at:
            self.created_at = now
