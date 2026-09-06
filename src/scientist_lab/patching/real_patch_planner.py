"""RealPatchPlanner — CodeContextBundle → PatchProposal via real provider (v2.2.2).

Offline tests may inject OpenAICompatibleProvider + MockTransport.
When real_only=True, silent fallback to mock/fake/replay is forbidden.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from scientist_lab.domain.models import new_id
from scientist_lab.llm.models import LLMRequest
from scientist_lab.llm.provider import LLMProvider
from scientist_lab.patching.context_models import CodeContextBundle
from scientist_lab.patching.fingerprint import fingerprint_diff
from scientist_lab.patching.models import PatchProposal
from scientist_lab.patching.patch_schema import (
    PATCH_PROPOSAL_SCHEMA,
    REAL_PATCH_SYSTEM_PROMPT,
)
from scientist_lab.patching.path_policy import PathPolicy
from scientist_lab.patching.verifier import PatchVerifier
from scientist_lab.research_loop.provider_gate import (
    assert_no_fallback,
    assert_real_only_provider,
    provider_audit,
    unwrap_provider_label,
)


class RealPatchPlannerError(RuntimeError):
    """Raised when real patch planning fails (never silently mock)."""


class RealPatchPlannerOutput(BaseModel):
    """Validated structured LLM output before PatchProposal persistence."""

    title: str
    rationale: str
    unified_diff: str
    risks: list[str] = Field(default_factory=list)
    expected_tests: list[str] = Field(default_factory=list)
    expected_impact: str = ""
    evidence_gap_ids: list[str] = Field(default_factory=list)
    files_touched: list[str] = Field(default_factory=list)


def code_context_to_patch_request(
    bundle: CodeContextBundle,
    *,
    system_prompt: str = REAL_PATCH_SYSTEM_PROMPT,
) -> LLMRequest:
    """Build an LLMRequest from a restricted CodeContextBundle only."""
    snapshots = [
        {
            "path": snap.path,
            "content_sha256": snap.content_sha256,
            "size_bytes": snap.size_bytes,
            "truncated": snap.truncated,
            "content": snap.content,
        }
        for snap in bundle.snapshots
    ]
    payload = {
        "bundle_id": bundle.bundle_id,
        "request_id": bundle.request_id,
        "project_id": bundle.project_id,
        "source_commit": bundle.source_commit,
        "context_sha256": bundle.context_sha256,
        "goal": bundle.goal,
        "failure_summary": bundle.failure_summary,
        "test_errors": list(bundle.test_errors),
        "evidence_gap_ids": list(bundle.evidence_gap_ids),
        "patch_target": bundle.patch_target,
        "interface_notes": list(bundle.interface_notes),
        "allowed_files": [f.model_dump(mode="json") for f in bundle.allowed_files],
        "excluded_paths": list(bundle.excluded_paths),
        "snapshots": snapshots,
        "constraints": {
            "unified_diff_only": True,
            "no_shell": True,
            "no_secrets": True,
            "whitelist_only": True,
            "max_files": bundle.budget.max_files,
        },
    }
    return LLMRequest(
        purpose="other",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
            },
        ],
        response_schema=PATCH_PROPOSAL_SCHEMA,
        temperature=0.0,
        metadata={
            "project_id": bundle.project_id,
            "bundle_id": bundle.bundle_id,
            "context_sha256": bundle.context_sha256,
            "purpose_detail": "real_patch_planner",
        },
    )


def _parse_planner_output(parsed: dict[str, Any] | None) -> RealPatchPlannerOutput:
    if not isinstance(parsed, dict):
        raise RealPatchPlannerError("provider returned empty/invalid parsed_json")
    try:
        return RealPatchPlannerOutput.model_validate(parsed)
    except Exception as exc:  # noqa: BLE001
        raise RealPatchPlannerError(
            f"PatchProposal schema mapping failed: {exc}"
        ) from exc


class RealPatchPlanner:
    """Generate a restricted PatchProposal from CodeContextBundle."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        requested_provider: str = "openai-compatible",
        real_only: bool = True,
        verifier: PatchVerifier | None = None,
        policy: PathPolicy | None = None,
    ) -> None:
        self.provider = provider
        self.requested_provider = requested_provider
        self.real_only = bool(real_only)
        path_policy = policy or PathPolicy.for_code_context()
        self.verifier = verifier or PatchVerifier(path_policy)

    def propose(
        self,
        bundle: CodeContextBundle,
        *,
        auto_verify: bool = True,
        patch_id: str | None = None,
        system_prompt: str | None = None,
    ) -> tuple[PatchProposal, dict[str, Any]]:
        """Call provider and build PatchProposal.

        Returns ``(proposal, audit)``. On failure raises RealPatchPlannerError
        (or provider gate errors) — never invents a mock patch.
        """
        requested = assert_real_only_provider(
            self.requested_provider, real_only=self.real_only
        )
        actual_name = unwrap_provider_label(getattr(self.provider, "name", None))
        assert_no_fallback(
            requested_provider=requested,
            actual_provider=actual_name,
            fallback_allowed=False,
            fallback_used=False,
        )

        request = code_context_to_patch_request(
            bundle, system_prompt=system_prompt or REAL_PATCH_SYSTEM_PROMPT
        )
        try:
            response = self.provider.complete(request)
        except Exception as exc:  # noqa: BLE001 — surface, do not mock-fallback
            raise RealPatchPlannerError(
                f"real patch provider call failed: {exc}"
            ) from exc

        actual_after = unwrap_provider_label(
            getattr(response, "provider", None) or actual_name
        )
        assert_no_fallback(
            requested_provider=requested,
            actual_provider=actual_after,
            fallback_allowed=False,
            fallback_used=False,
        )

        if not getattr(response, "schema_valid", True):
            errors = list(getattr(response, "schema_errors", None) or [])
            raise RealPatchPlannerError(
                "structured PatchProposal schema invalid: "
                + ("; ".join(errors) or "unknown")
            )

        output = _parse_planner_output(response.parsed_json)
        if not str(output.unified_diff or "").strip():
            raise RealPatchPlannerError("unified_diff is empty")

        fp = fingerprint_diff(output.unified_diff)
        gap_ids = list(output.evidence_gap_ids) or list(bundle.evidence_gap_ids)
        proposal = PatchProposal(
            patch_id=patch_id or new_id("patch"),
            project_id=bundle.project_id,
            status="proposed",
            title=output.title.strip() or "Restricted real patch",
            rationale=output.rationale.strip(),
            evidence_gap_ids=gap_ids,
            unified_diff=output.unified_diff,
            files_touched=list(output.files_touched),
            fingerprint_sha256=fp,
            provider=str(actual_after or requested),
            metadata={
                "applied_main": False,
                "applied_sandbox": False,
                "apply_main_available": False,
                "provider_call": True,
                "real_only": self.real_only,
                "fallback_used": False,
                "bundle_id": bundle.bundle_id,
                "context_sha256": bundle.context_sha256,
                "source_commit": bundle.source_commit,
                "request_id": bundle.request_id,
                "risks": list(output.risks),
                "expected_tests": list(output.expected_tests),
                "expected_impact": output.expected_impact,
                "llm_model": getattr(response, "model", None),
                "request_fingerprint": getattr(
                    response, "request_fingerprint", None
                ),
            },
        )

        audit = provider_audit(
            requested_provider=requested,
            actual_provider=actual_after,
            fallback_allowed=False,
            fallback_used=False,
            extra={
                "bundle_id": bundle.bundle_id,
                "context_sha256": bundle.context_sha256,
                "patch_id": proposal.patch_id,
                "schema_valid": True,
            },
        )

        if auto_verify:
            verification = self.verifier.verify(output.unified_diff)
            proposal.verification = verification
            proposal.files_touched = list(verification.files_touched)
            proposal.fingerprint_sha256 = verification.fingerprint_sha256
            proposal.status = (
                "verified" if verification.ok else "rejected_by_verifier"
            )
            audit["verification_ok"] = verification.ok

        return proposal, audit
