"""Gate real planning behind a qualified LLM evaluation (v1.5.9)."""

from __future__ import annotations

from typing import Any

from scientist_lab.llm.errors import LLMError
from scientist_lab.llm_eval.quality_gate import (
    LLMQualityThresholds,
    verify_quality_gate,
)


class LLMProfileNotQualifiedError(LLMError):
    """Raised when a profile lacks a passing evaluation for formal planning."""


def assert_profile_qualified_for_planning(
    *,
    profile: dict[str, Any] | Any,
    evaluation: dict[str, Any] | None,
    suite_version: str | None = None,
    require_quality_gate: bool = True,
    allow_unqualified: bool = False,
    thresholds: LLMQualityThresholds | None = None,
) -> dict[str, Any]:
    """Validate profile + latest evaluation before real planning.

    Returns a gate audit dict to attach to plan metadata.
    """
    profile_id = getattr(profile, "profile_id", None) or (profile or {}).get(
        "profile_id"
    )
    planner_prompt = getattr(profile, "planner_prompt_version", None) or (
        profile or {}
    ).get("planner_prompt_version")
    critic_prompt = getattr(profile, "critic_prompt_version", None) or (
        profile or {}
    ).get("critic_prompt_version")

    audit: dict[str, Any] = {
        "profile_id": profile_id,
        "require_quality_gate": require_quality_gate,
        "quality_gate_bypassed": False,
        "warning": None,
    }

    if not require_quality_gate:
        return audit

    reasons: list[str] = []
    if evaluation is None:
        reasons.append("no evaluation found for profile")
    else:
        result = dict(evaluation.get("result") or evaluation)
        eval_suite = evaluation.get("suite_version") or result.get("suite_version")
        if suite_version and eval_suite and str(eval_suite) != str(suite_version):
            reasons.append(
                f"suite_version mismatch: evaluation={eval_suite} required={suite_version}"
            )
        meta = dict(result.get("metadata") or {})
        eval_planner_prompt = meta.get("planner_prompt_version")
        eval_critic_prompt = meta.get("critic_prompt_version")
        if planner_prompt and eval_planner_prompt and eval_planner_prompt != planner_prompt:
            reasons.append("planner_prompt_version mismatch")
        if critic_prompt and eval_critic_prompt and eval_critic_prompt != critic_prompt:
            reasons.append("critic_prompt_version mismatch")
        if result.get("status") == "skipped":
            reasons.append("evaluation was skipped")
        gate = verify_quality_gate(result, thresholds=thresholds)
        audit["gate_status"] = gate.status
        if gate.status == "blocked":
            reasons.extend(gate.blocking_reasons or ["quality gate blocked"])

    if reasons:
        if allow_unqualified:
            audit["quality_gate_bypassed"] = True
            audit["warning"] = (
                "Unqualified LLM profile was used for development testing."
            )
            audit["bypass_reasons"] = reasons
            return audit
        raise LLMProfileNotQualifiedError(
            f"LLM profile {profile_id!r} is not qualified for formal planning: "
            + "; ".join(reasons)
        )

    audit["gate_status"] = audit.get("gate_status") or "passed"
    return audit
