"""Structured PatchProposal schema for RealPatchPlanner (v2.2.2)."""

from __future__ import annotations

from typing import Any


PATCH_PROPOSAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["title", "rationale", "unified_diff"],
    "additionalProperties": True,
    "properties": {
        "title": {"type": "string"},
        "rationale": {"type": "string"},
        "unified_diff": {"type": "string"},
        "risks": {"type": "array", "items": {"type": "string"}},
        "expected_tests": {"type": "array", "items": {"type": "string"}},
        "expected_impact": {"type": "string"},
        "evidence_gap_ids": {"type": "array", "items": {"type": "string"}},
        "files_touched": {"type": "array", "items": {"type": "string"}},
    },
}


REAL_PATCH_SYSTEM_PROMPT = (
    "You are a restricted source-code patch planner for Scientist Lab. "
    "You may ONLY propose a Unified Diff that modifies files listed in the "
    "allowed snapshots. Never invent paths outside the allow-list. "
    "Never include secrets, shell commands, or dependency/CI/Docker changes. "
    "Return JSON matching the provided schema only."
)
