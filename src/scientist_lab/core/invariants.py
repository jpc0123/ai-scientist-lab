"""Cross-document invariants that JSON Schema cannot express alone."""

from __future__ import annotations

from typing import Any, Mapping


class InvariantError(ValueError):
    pass


def assert_plan_memory_policy(plan: Mapping[str, Any]) -> None:
    """Round >= 1 requires memory_refs + evidence_runs unless bootstrap."""
    round_index = int(plan.get("round_index") or 0)
    bootstrap = bool(plan.get("bootstrap"))
    refs = plan.get("memory_refs") or {}
    lesson_ids = list(refs.get("lesson_ids") or [])
    strategy_ids = list(refs.get("strategy_ids") or [])
    evidence_runs = list(plan.get("evidence_runs") or [])

    if round_index <= 0 or bootstrap:
        return
    if not lesson_ids and not strategy_ids:
        raise InvariantError("Round>=1 Plan must cite lesson_ids or strategy_ids")
    if not evidence_runs:
        raise InvariantError("Round>=1 Plan must cite evidence_runs")


def assert_lesson_has_evidence(lesson: Mapping[str, Any]) -> None:
    created = list(lesson.get("created_from") or [])
    evidence = list(lesson.get("evidence") or [])
    if not created:
        raise InvariantError("Lesson requires created_from")
    if not evidence:
        raise InvariantError("Lesson requires evidence")


def assert_memory_refs_resolvable(
    plan: Mapping[str, Any],
    *,
    lesson_ids: Mapping[str, Any] | set[str] | list[str],
    strategy_ids: Mapping[str, Any] | set[str] | list[str],
) -> None:
    """Round>=1 memory_refs must resolve to stored lessons/strategies."""
    round_index = int(plan.get("round_index") or 0)
    if round_index <= 0 or bool(plan.get("bootstrap")):
        return
    known_lessons = set(lesson_ids)
    known_strategies = set(strategy_ids)
    refs = plan.get("memory_refs") or {}
    missing_lessons = [i for i in (refs.get("lesson_ids") or []) if i not in known_lessons]
    missing_strategies = [
        i for i in (refs.get("strategy_ids") or []) if i not in known_strategies
    ]
    if missing_lessons or missing_strategies:
        raise InvariantError(
            f"Plan memory_refs not resolvable: lessons={missing_lessons} "
            f"strategies={missing_strategies}"
        )


def assert_keep_is_not_claim(claim_result: Mapping[str, Any]) -> None:
    """KEEP answers next-round action; it must not be the ClaimGate support reason."""
    status = str(claim_result.get("status") or "")
    reason = str(claim_result.get("reason") or "")
    if claim_result.get("keep_is_not_claim") is not True:
        raise InvariantError("ClaimGate result must set keep_is_not_claim=true")
    if status == "SUPPORTED" and "KEEP" in reason and "did not decide" not in reason.lower():
        raise InvariantError(
            "KEEP must not be used as ClaimGate support; "
            "SUPPORTED requires independent formal evidence"
        )


def assert_discard_is_not_module_ineffective(claim_result: Mapping[str, Any]) -> None:
    """DISCARD is not a scientific claim that a module is ineffective."""
    if str(claim_result.get("review_decision") or "") != "DISCARD":
        return
    text = str(claim_result.get("claim_text") or "").lower()
    if "ineffective" not in text and "无效" not in text:
        return
    if str(claim_result.get("status") or "") == "SUPPORTED":
        raise InvariantError(
            "DISCARD must not be treated as a supported 'module ineffective' claim"
        )


def evaluate_stop_rules(
    stop_rules: Mapping[str, Any] | None,
    *,
    round_index: int,
    consecutive_discards: int = 0,
    execution_failures: int = 0,
    rounds_without_improvement: int = 0,
    duplicate_plan_rejects: int = 0,
) -> str | None:
    """Return orchestration action id if a hard stop rule fires, else None."""
    rules = stop_rules or {}
    if rules.get("max_rounds") is not None and round_index >= int(rules["max_rounds"]):
        return "STOP"
    if rules.get("max_consecutive_discards") is not None and consecutive_discards >= int(
        rules["max_consecutive_discards"]
    ):
        return "STOP"
    if rules.get("max_execution_failures") is not None and execution_failures >= int(
        rules["max_execution_failures"]
    ):
        return "STOP"
    if rules.get("stop_if_no_improvement_for") is not None and rounds_without_improvement >= int(
        rules["stop_if_no_improvement_for"]
    ):
        return "STOP"
    if rules.get("max_duplicate_plan_rejects") is not None and duplicate_plan_rejects >= int(
        rules["max_duplicate_plan_rejects"]
    ):
        return "NOVELTY_EXHAUSTED"
    return None
