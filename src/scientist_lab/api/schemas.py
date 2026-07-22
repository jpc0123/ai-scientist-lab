"""Request bodies for controlled write endpoints (v1.7.1)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ReasonBody(BaseModel):
    reason: str = ""


class PatchTestBody(BaseModel):
    profile: Literal["smoke", "syntax", "mock_experiment"] = "smoke"


class PatchApplySandboxBody(BaseModel):
    force: bool = False


class PatchRecordEvidenceBody(BaseModel):
    require_tests: bool = False


class PatchDecideMergeBody(BaseModel):
    decision: Literal["merge", "discard"]
    reason: str = ""


class IterationFinalizeBody(BaseModel):
    selected_node_id: str
    reason: str
    decision_type: str = "efficiency_tradeoff"
    evidence_strength: str = "moderate"


class CandidateRejectBody(BaseModel):
    reason: str | None = Field(default=None)


class CompareExecutionsBody(BaseModel):
    execution_id_a: str
    execution_id_b: str


class CompareNodesBody(BaseModel):
    node_id_a: str
    node_id_b: str
