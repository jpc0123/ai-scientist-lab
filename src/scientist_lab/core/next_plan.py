"""Deterministic N→N+1 Plan construction. Not a Planner.

Planner (WHAT/WHY) may use rules or an explicit LLM backend; this helper
only proves that Round N MemoryWriter output is citable by a following
ExperimentPlan. Adapter still does HOW and must not invent modules.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from scientist_lab.core.gate_engine import GateEngine, GateStatus, GateVerdict
from scientist_lab.core.invariants import (
    InvariantError,
    assert_memory_refs_resolvable,
    assert_plan_memory_policy,
)
from scientist_lab.core.memory_writer import MemoryWriter
from scientist_lab.core.schema_registry import validate_named


def _as_catalog(
    memory: MemoryWriter | Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if isinstance(memory, MemoryWriter):
        return memory.load_lessons(), memory.load_strategies()
    lessons = memory.get("lessons") or memory.get("lesson_ids") or {}
    strategies = memory.get("strategies") or memory.get("strategy_ids") or {}
    if isinstance(lessons, Sequence) and not isinstance(lessons, (str, bytes)):
        lessons = {str(i): {"lesson_id": str(i)} for i in lessons}
    if isinstance(strategies, Sequence) and not isinstance(strategies, (str, bytes)):
        strategies = {str(i): {"strategy_id": str(i)} for i in strategies}
    return dict(lessons), dict(strategies)


def memory_refs_from_run(
    lessons: Mapping[str, Mapping[str, Any]],
    strategies: Mapping[str, Mapping[str, Any]],
    *,
    parent_run_id: str,
) -> dict[str, list[str]]:
    """Cite lessons/strategies written from run N. Negative evidence remains citable."""
    lesson_ids = [
        lid
        for lid, row in lessons.items()
        if parent_run_id in list(row.get("created_from") or [])
    ]
    if not lesson_ids:
        lesson_ids = [str(lid) for lid in lessons]
    cited = set(lesson_ids)
    strategy_ids = [
        sid
        for sid, row in strategies.items()
        if cited.intersection(row.get("reason_lesson_ids") or [])
    ]
    if not strategy_ids:
        strategy_ids = [str(sid) for sid in strategies]
    return {"lesson_ids": lesson_ids, "strategy_ids": strategy_ids}


def build_candidate_next_plan(
    previous_plan: Mapping[str, Any],
    memory: MemoryWriter | Mapping[str, Any],
    *,
    parent_run_id: str,
    proposed_changes: Sequence[Mapping[str, Any]] | None = None,
    modification_scope: Sequence[str] | None = None,
    hypothesis: str | None = None,
    observation: str | None = None,
    expected_effect: Mapping[str, Any] | None = None,
    evaluation: Mapping[str, Any] | None = None,
    budget_class: str | None = None,
    risk_level: str | None = None,
    rationale: str | None = None,
    decision_summary: Mapping[str, Any] | None = None,
    memory_refs: Mapping[str, Sequence[str]] | None = None,
    evidence_runs: Sequence[str] | None = None,
    controlled_variables: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build Round N+1 ExperimentPlan that must cite written memory.

    Copies previous scientific fields unless overlays are provided (Planner WHAT/WHY).
    Does not invent modules (no FDPN-from-slogan). proposed_changes remain required.
    """
    lessons, strategies = _as_catalog(memory)
    refs = (
        {
            "lesson_ids": [str(i) for i in (memory_refs.get("lesson_ids") or [])],
            "strategy_ids": [str(i) for i in (memory_refs.get("strategy_ids") or [])],
        }
        if memory_refs is not None
        else memory_refs_from_run(lessons, strategies, parent_run_id=str(parent_run_id))
    )
    if not refs["lesson_ids"] and not refs["strategy_ids"]:
        raise InvariantError("Cannot build Round N+1 Plan: no written lessons/strategies to cite")

    changes = [
        dict(row)
        for row in (
            proposed_changes
            if proposed_changes is not None
            else list(previous_plan.get("proposed_changes") or [])
        )
    ]
    if not changes or not all(row.get("target") and row.get("summary") for row in changes):
        raise InvariantError(
            "Round N+1 Plan requires proposed_changes with target+summary; "
            "helper/Adapter will not invent a module"
        )

    scope = list(
        modification_scope
        if modification_scope is not None
        else previous_plan.get("modification_scope") or []
    )
    if not scope:
        raise InvariantError("Round N+1 Plan requires modification_scope from the previous Plan")

    prev_round = int(previous_plan.get("round_index") or 0)
    round_index = prev_round + 1
    plan_id = f"plan_round{round_index}_from_{parent_run_id}"
    first_lesson = lessons.get(refs["lesson_ids"][0]) if refs["lesson_ids"] else {}
    lesson_type = str((first_lesson or {}).get("type") or "written_memory")
    statement = str((first_lesson or {}).get("statement") or "").strip()
    observation_text = observation or (
        statement
        if statement
        else (
            f"Run {parent_run_id} wrote {lesson_type} memory; "
            "Round N+1 must cite it."
        )
    )
    runs = [str(i) for i in (evidence_runs if evidence_runs is not None else [parent_run_id])]
    if not runs:
        runs = [str(parent_run_id)]

    plan: dict[str, Any] = {
        "schema_version": "1.0.0",
        "plan_id": plan_id,
        "project_id": previous_plan["project_id"],
        "protocol_id": previous_plan["protocol_id"],
        "protocol_version": previous_plan["protocol_version"],
        "parent_run_id": str(parent_run_id),
        "round_index": round_index,
        "observation": observation_text,
        "hypothesis": hypothesis if hypothesis is not None else str(previous_plan.get("hypothesis") or ""),
        "modification_scope": scope,
        "proposed_changes": changes,
        "controlled_variables": list(
            controlled_variables
            if controlled_variables is not None
            else previous_plan.get("controlled_variables") or ["evaluator"]
        ),
        "expected_effect": dict(
            expected_effect
            if expected_effect is not None
            else previous_plan.get("expected_effect") or {}
        ),
        "evaluation": dict(
            evaluation
            if evaluation is not None
            else previous_plan.get("evaluation") or {"method": "fast_eval"}
        ),
        "budget_class": budget_class or previous_plan.get("budget_class", "probe"),
        "risk_level": risk_level or previous_plan.get("risk_level", "auto"),
        "memory_refs": refs,
        "evidence_runs": runs,
        "bootstrap": False,
        "rationale": rationale
        or (
            f"Deterministic N+1 candidate citing {refs['lesson_ids']} / "
            f"{refs['strategy_ids']} from {parent_run_id}."
        ),
    }
    if decision_summary is not None:
        plan["decision_summary"] = dict(decision_summary)
    validate_named("experiment_plan", plan)
    assert_plan_memory_policy(plan)
    assert_memory_refs_resolvable(plan, lesson_ids=lessons, strategy_ids=strategies)
    return plan


def gate_candidate_next_plan(
    protocol: Mapping[str, Any],
    plan: Mapping[str, Any],
    contract: Mapping[str, Any],
    memory: MemoryWriter,
    *,
    bound_fingerprint: Mapping[str, Any] | None = None,
    gate: GateEngine | None = None,
) -> GateVerdict:
    """Gate a materialized N+1 Plan against written memory; cite on APPROVED.

    Does not execute GPU. Does not call Planner.
    """
    verdict = (gate or GateEngine()).evaluate(
        protocol,
        plan,
        contract,
        bound_fingerprint=bound_fingerprint,
        memory=memory,
    )
    if verdict.status == GateStatus.APPROVED:
        memory.record_plan_citation(plan)
    return verdict
