"""Freeze Planner: WHAT/WHY. Rules-first. Does not write model CLI.

Reuses build_candidate_next_plan as the N+1 skeleton and fills scientific
semantics from Protocol + written Memory. Not a new Agent kind. No LLM in v1.
Does not KEEP/DISCARD. Does not invent lesson ids. Does not edit Protocol.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from scientist_lab.core.invariants import InvariantError
from scientist_lab.core.memory_writer import MemoryWriter
from scientist_lab.core.next_plan import (
    _as_catalog,
    build_candidate_next_plan,
    gate_candidate_next_plan,
)
from scientist_lab.core.state_machine import OrchestrationAction
from scientist_lab.instrumentation.appender import EventAppender


class PlanRefused(ValueError):
    """Planner cannot emit a Gate-legal Plan (missing memory, no target, etc.)."""

    def __init__(self, message: str, *, orchestration_action: str | None = None) -> None:
        super().__init__(message)
        self.orchestration_action = orchestration_action or OrchestrationAction.NEED_HUMAN.value


@dataclass(frozen=True)
class PlanPacket:
    plan: dict[str, Any]
    decision_summary: dict[str, Any]
    source: str
    orchestration_action: str


_NEGATIVE_TYPES = frozenset({"negative_evidence", "constraint_violation"})
_SWITCH_ACTIONS = frozenset({"deprioritize"})
_CONTINUE_ACTIONS = frozenset({"prioritize", "keep"})


def _primary_metric(protocol: Mapping[str, Any]) -> str:
    return str(((protocol.get("objective") or {}).get("primary") or {}).get("metric") or "APS")


def _editable_targets(protocol: Mapping[str, Any]) -> list[str]:
    frozen = set(protocol.get("frozen_scope") or [])
    return [str(tok) for tok in (protocol.get("editable_scope") or []) if str(tok) not in frozen]


def _select_memory_refs(
    lessons: Mapping[str, Mapping[str, Any]],
    strategies: Mapping[str, Mapping[str, Any]],
    *,
    parent_run_id: str,
) -> dict[str, list[str]]:
    """Prefer DISCARD / negative_evidence; never invent ids."""
    negative = [
        lid
        for lid, row in lessons.items()
        if str(row.get("type") or "") in _NEGATIVE_TYPES
        and (not parent_run_id or parent_run_id in list(row.get("created_from") or []))
    ]
    if not negative:
        negative = [
            lid for lid, row in lessons.items() if str(row.get("type") or "") in _NEGATIVE_TYPES
        ]
    from_run = [
        lid
        for lid, row in lessons.items()
        if parent_run_id in list(row.get("created_from") or [])
    ]
    lesson_ids = negative or from_run or [str(lid) for lid in lessons]
    cited = set(lesson_ids)
    strategy_ids = [
        sid
        for sid, row in strategies.items()
        if cited.intersection(row.get("reason_lesson_ids") or [])
    ]
    if not strategy_ids:
        strategy_ids = [str(sid) for sid in strategies]
    return {"lesson_ids": lesson_ids, "strategy_ids": strategy_ids}


def _first_strategy(
    strategies: Mapping[str, Mapping[str, Any]], strategy_ids: Sequence[str]
) -> dict[str, Any]:
    for sid in strategy_ids:
        row = strategies.get(sid)
        if row:
            return dict(row)
    if strategies:
        return dict(next(iter(strategies.values())))
    return {}


def _first_lesson(
    lessons: Mapping[str, Mapping[str, Any]], lesson_ids: Sequence[str]
) -> dict[str, Any]:
    for lid in lesson_ids:
        row = lessons.get(lid)
        if row:
            return dict(row)
    if lessons:
        return dict(next(iter(lessons.values())))
    return {}


def _scientific_semantics(
    *,
    protocol: Mapping[str, Any],
    previous_plan: Mapping[str, Any],
    lesson: Mapping[str, Any],
    strategy: Mapping[str, Any],
    last_review_decision: str | None,
    observation: str | None,
) -> dict[str, Any]:
    """WHAT/WHY from strategy.action + last review. No CLI. No invented modules."""
    editable = _editable_targets(protocol)
    if not editable:
        raise PlanRefused(
            "Protocol has no editable_scope; Planner cannot choose a target module",
            orchestration_action=OrchestrationAction.NEED_HUMAN.value,
        )
    prev_scope = list(previous_plan.get("modification_scope") or [])
    discarded = str(
        (lesson.get("scope") or {}).get("module")
        or strategy.get("target")
        or (prev_scope[0] if prev_scope else "")
    )
    action = str(strategy.get("action") or "")
    lesson_type = str(lesson.get("type") or "")
    review = str(last_review_decision or "")
    metric = _primary_metric(protocol)
    switch = (
        action in _SWITCH_ACTIONS
        or review == "DISCARD"
        or lesson_type in _NEGATIVE_TYPES
    )
    if switch:
        alternatives = [tok for tok in editable if tok != discarded]
        if not alternatives:
            raise PlanRefused(
                "No remaining editable target after deprioritize/DISCARD; NEED_HUMAN",
                orchestration_action=OrchestrationAction.NEED_HUMAN.value,
            )
        target = alternatives[0]
        hypothesis = (
            f"After negative evidence on {discarded or 'the prior module'}, "
            f"a {target} change is a better probe of {metric} than repeating "
            f"the discarded direction."
        )
        summary = (
            f"Probe {target} after DISCARD/{lesson_type or 'negative_evidence'} "
            f"on {discarded or 'prior target'}; do not repeat that change."
        )
        selected = f"switch_to_{target}"
    else:
        target = discarded if discarded in editable else (prev_scope[0] if prev_scope else editable[0])
        if target not in editable:
            target = editable[0]
        if action == "prioritize" or review in {"KEEP", "VALIDATE"}:
            hypothesis = (
                f"Further {target} adjustment should increase {metric} "
                f"given supporting or priority evidence."
            )
            summary = f"Continue {target} modification under the same evaluation protocol."
            selected = f"continue_{target}"
        else:
            hypothesis = (
                f"Replicating the {target} change should stabilize {metric} "
                f"before claiming a directional effect."
            )
            summary = f"Replicate {target} modification with matched evaluation."
            selected = f"replicate_{target}"

    frozen = list(protocol.get("frozen_scope") or [])
    obs = observation or str(lesson.get("statement") or "").strip() or (
        f"Memory {lesson.get('type') or 'written'} on run "
        f"{(lesson.get('created_from') or ['unknown'])[0]}."
    )
    direction = "increase"
    primary = (protocol.get("objective") or {}).get("primary") or {}
    if str(primary.get("direction") or "") == "minimize":
        direction = "decrease"
    return {
        "modification_scope": [target],
        "proposed_changes": [{"target": target, "summary": summary}],
        "hypothesis": hypothesis,
        "observation": obs,
        "controlled_variables": frozen or list(previous_plan.get("controlled_variables") or ["evaluator"]),
        "expected_effect": {
            "primary_metric": metric,
            "direction": direction,
            "rationale": f"Protocol objective.primary={metric}; rules-first Planner.",
        },
        "selected_action": selected,
        "switch": switch,
        "target": target,
    }


def resolve_planner_backend(explicit: str | None = None) -> str:
    """Default is rules. llm only when constructor/CLI/env explicitly opens it."""
    if explicit is not None and str(explicit).strip():
        name = str(explicit).strip().lower()
    else:
        name = str(os.environ.get("SCIENTIST_LAB_PLANNER_BACKEND") or "rules").strip().lower()
    if name not in {"rules", "llm"}:
        raise PlanRefused(f"unknown planner_backend={name!r}; expected rules|llm")
    return name


class Planner:
    """Thin Freeze Planner. Default rules-first; optional LLM Gateway backend.

    LLM is not a fifth Agent. Fail closed: bad JSON / over-scope / FDPN does
    not APPROVE and does not silently fall back to rules.
    """

    def __init__(
        self,
        *,
        backend: str | None = None,
        provider: Any | None = None,
        fallback_to_rules: bool = False,
        live: bool = False,
    ) -> None:
        self.backend = resolve_planner_backend(backend)
        self.provider = provider
        self.fallback_to_rules = bool(fallback_to_rules)
        self.live = bool(live)

    def next_plan(
        self,
        *,
        protocol: Mapping[str, Any],
        memory: MemoryWriter | Mapping[str, Any],
        previous_plan: Mapping[str, Any],
        parent_run_id: str,
        best_run_id: str | None = None,
        relevant_runs: Sequence[str] | None = None,
        observation: str | None = None,
        last_review_decision: str | None = None,
        events: EventAppender | None = None,
    ) -> PlanPacket:
        if self.backend == "llm":
            try:
                return self._next_plan_llm(
                    protocol=protocol,
                    memory=memory,
                    previous_plan=previous_plan,
                    parent_run_id=parent_run_id,
                    best_run_id=best_run_id,
                    relevant_runs=relevant_runs,
                    observation=observation,
                    last_review_decision=last_review_decision,
                    events=events,
                )
            except PlanRefused as exc:
                if not self.fallback_to_rules:
                    raise
                if events is not None:
                    events.append(
                        {
                            "project_id": protocol["project_id"],
                            "run_id": str(parent_run_id),
                            "protocol_version": protocol.get("protocol_version"),
                            "fingerprint_id": protocol.get("fingerprint_id"),
                            "event_type": "plan_proposal",
                            "actor_role": "planner",
                            "phase": "planning",
                            "payload": {
                                "source": "llm_fallback_rules",
                                "fail_closed": True,
                                "fallback_to_rules": True,
                                "reason": str(exc),
                            },
                        }
                    )
                packet = self._next_plan_rules(
                    protocol=protocol,
                    memory=memory,
                    previous_plan=previous_plan,
                    parent_run_id=parent_run_id,
                    best_run_id=best_run_id,
                    relevant_runs=relevant_runs,
                    observation=observation,
                    last_review_decision=last_review_decision,
                    events=events,
                )
                return PlanPacket(
                    plan=packet.plan,
                    decision_summary=packet.decision_summary,
                    source="llm_fallback_rules",
                    orchestration_action=packet.orchestration_action,
                )
        return self._next_plan_rules(
            protocol=protocol,
            memory=memory,
            previous_plan=previous_plan,
            parent_run_id=parent_run_id,
            best_run_id=best_run_id,
            relevant_runs=relevant_runs,
            observation=observation,
            last_review_decision=last_review_decision,
            events=events,
        )

    def _next_plan_rules(
        self,
        *,
        protocol: Mapping[str, Any],
        memory: MemoryWriter | Mapping[str, Any],
        previous_plan: Mapping[str, Any],
        parent_run_id: str,
        best_run_id: str | None = None,
        relevant_runs: Sequence[str] | None = None,
        observation: str | None = None,
        last_review_decision: str | None = None,
        events: EventAppender | None = None,
    ) -> PlanPacket:
        lessons, strategies = _as_catalog(memory)
        prev_round = int(previous_plan.get("round_index") or 0)
        next_round = prev_round + 1
        bootstrap = bool(previous_plan.get("bootstrap")) and prev_round <= 0
        if next_round >= 1 and not bootstrap:
            if not lessons and not strategies:
                raise PlanRefused(
                    "Round>=1 Planner requires written lessons or strategies; "
                    "refusing empty memory rather than emitting unresolved refs",
                    orchestration_action=OrchestrationAction.NEED_MEMORY.value,
                )

        refs = _select_memory_refs(lessons, strategies, parent_run_id=str(parent_run_id))
        if next_round >= 1 and not refs["lesson_ids"] and not refs["strategy_ids"]:
            raise PlanRefused(
                "Round>=1 Planner has no resolvable memory_refs to cite",
                orchestration_action=OrchestrationAction.NEED_MEMORY.value,
            )

        known_lessons = set(lessons)
        known_strategies = set(strategies)
        if any(lid not in known_lessons for lid in refs["lesson_ids"]):
            raise PlanRefused("Planner refused to emit unresolved lesson_ids")
        if any(sid not in known_strategies for sid in refs["strategy_ids"]):
            raise PlanRefused("Planner refused to emit unresolved strategy_ids")

        lesson = _first_lesson(lessons, refs["lesson_ids"])
        strategy = _first_strategy(strategies, refs["strategy_ids"])
        review = last_review_decision or (
            "DISCARD"
            if str(lesson.get("type") or "") in _NEGATIVE_TYPES
            else None
        )
        semantics = _scientific_semantics(
            protocol=protocol,
            previous_plan=previous_plan,
            lesson=lesson,
            strategy=strategy,
            last_review_decision=review,
            observation=observation,
        )
        if not semantics["proposed_changes"] or not semantics["modification_scope"]:
            raise PlanRefused(
                "Planner refuses a vague Plan without target/proposed_changes; "
                "will not ask Adapter to invent a module",
                orchestration_action=OrchestrationAction.NEED_HUMAN.value,
            )

        evidence_runs = [str(parent_run_id)]
        if best_run_id and str(best_run_id) not in evidence_runs:
            evidence_runs.append(str(best_run_id))
        for run_id in relevant_runs or []:
            if str(run_id) not in evidence_runs:
                evidence_runs.append(str(run_id))

        decision_summary = {
            "problem_observed": semantics["observation"],
            "hypothesis": semantics["hypothesis"],
            "candidate_actions": [
                "continue_module",
                "switch_module",
                "replicate",
                "NEED_HUMAN",
            ],
            "selected_action": semantics["selected_action"],
            "decision_basis": [
                f"strategy.action={strategy.get('action')}",
                f"lesson.type={lesson.get('type')}",
                f"review_decision={review}",
                f"cited_lessons={refs['lesson_ids']}",
                f"cited_strategies={refs['strategy_ids']}",
                "rules-first Planner; no CoT; no Protocol edit",
            ],
            "expected_effect": (
                f"{semantics['expected_effect']['direction']} "
                f"{semantics['expected_effect']['primary_metric']}"
            ),
            "risk": "rules-first Planner; Adapter remains HOW-only",
        }
        try:
            plan = build_candidate_next_plan(
                previous_plan,
                memory,
                parent_run_id=str(parent_run_id),
                proposed_changes=semantics["proposed_changes"],
                modification_scope=semantics["modification_scope"],
                hypothesis=semantics["hypothesis"],
                observation=semantics["observation"],
                expected_effect=semantics["expected_effect"],
                evaluation=dict(previous_plan.get("evaluation") or {"method": "fast_eval"}),
                budget_class=str(previous_plan.get("budget_class") or "probe"),
                risk_level=str(previous_plan.get("risk_level") or "auto"),
                rationale=(
                    f"Rules-first Planner citing {refs['lesson_ids']} / "
                    f"{refs['strategy_ids']} from {parent_run_id}."
                ),
                decision_summary=decision_summary,
                memory_refs=refs,
                evidence_runs=evidence_runs,
                controlled_variables=semantics["controlled_variables"],
            )
        except InvariantError as exc:
            raise PlanRefused(
                f"Planner refused to emit an illegal Plan: {exc}",
                orchestration_action=OrchestrationAction.NEED_HUMAN.value,
            ) from exc

        if events is not None:
            events.append(
                {
                    "project_id": protocol["project_id"],
                    "run_id": str(parent_run_id),
                    "plan_id": plan["plan_id"],
                    "protocol_version": protocol.get("protocol_version"),
                    "fingerprint_id": protocol.get("fingerprint_id"),
                    "event_type": "plan_proposal",
                    "actor_role": "planner",
                    "phase": "planning",
                    "memory_refs": dict(plan["memory_refs"]),
                    "decision_summary": dict(decision_summary),
                    "payload": {
                        "round_index": plan["round_index"],
                        "source": "rules_first",
                        "parent_run_id": str(parent_run_id),
                        "evidence_runs": list(plan["evidence_runs"]),
                    },
                }
            )
        return PlanPacket(
            plan=plan,
            decision_summary=decision_summary,
            source="rules_first",
            orchestration_action=OrchestrationAction.NEED_GATE.value,
        )

    def _next_plan_llm(
        self,
        *,
        protocol: Mapping[str, Any],
        memory: MemoryWriter | Mapping[str, Any],
        previous_plan: Mapping[str, Any],
        parent_run_id: str,
        best_run_id: str | None = None,
        relevant_runs: Sequence[str] | None = None,
        observation: str | None = None,
        last_review_decision: str | None = None,
        events: EventAppender | None = None,
    ) -> PlanPacket:
        from scientist_lab.core.schema_registry import validate_named
        from scientist_lab.llm.config import redact_secrets
        from scientist_lab.llm.gateway import complete_chat
        from scientist_lab.llm.planner_contract import (
            PlannerContractError,
            build_contract_input,
            build_planner_request,
            parse_planner_completion,
            prompt_hash,
        )

        lessons, strategies = _as_catalog(memory)
        prev_round = int(previous_plan.get("round_index") or 0)
        next_round = prev_round + 1
        bootstrap = bool(previous_plan.get("bootstrap")) and prev_round <= 0
        if next_round >= 1 and not bootstrap:
            if not lessons and not strategies:
                raise PlanRefused(
                    "Round>=1 Planner requires written lessons or strategies; "
                    "refusing empty memory rather than emitting unresolved refs",
                    orchestration_action=OrchestrationAction.NEED_MEMORY.value,
                )

        refs = _select_memory_refs(lessons, strategies, parent_run_id=str(parent_run_id))
        if next_round >= 1 and not refs["lesson_ids"] and not refs["strategy_ids"]:
            raise PlanRefused(
                "Round>=1 Planner has no resolvable memory_refs to cite",
                orchestration_action=OrchestrationAction.NEED_MEMORY.value,
            )

        evidence_runs = [str(parent_run_id)]
        if best_run_id and str(best_run_id) not in evidence_runs:
            evidence_runs.append(str(best_run_id))
        for run_id in relevant_runs or []:
            if str(run_id) not in evidence_runs:
                evidence_runs.append(str(run_id))

        contract_in = build_contract_input(
            protocol=protocol,
            previous_plan=previous_plan,
            memory_refs=refs,
            memory_catalog={
                "lesson_ids": list(lessons),
                "strategy_ids": list(strategies),
                "lessons": {
                    str(lid): {
                        "lesson_id": str(lid),
                        "type": row.get("type"),
                        "scope": dict(row.get("scope") or {}),
                        "statement": row.get("statement"),
                    }
                    for lid, row in lessons.items()
                },
                "strategies": {
                    str(sid): {
                        "strategy_id": str(sid),
                        "action": row.get("action"),
                        "target": row.get("target"),
                    }
                    for sid, row in strategies.items()
                },
            },
            evidence={"evidence_runs": evidence_runs},
            last_review_decision=last_review_decision,
            parent_run_id=str(parent_run_id),
            observation=observation,
        )
        request = build_planner_request(contract_in)
        hashed = prompt_hash(request.messages)
        response = None
        raw = ""
        try:
            response = complete_chat(
                request,
                provider=self.provider,
                live=self.live,
            )
            raw = redact_secrets(response.content or "")
            mapped = parse_planner_completion(
                response,
                contract_in,
                known_lesson_ids=list(lessons),
                known_strategy_ids=list(strategies),
            )
        except PlannerContractError as exc:
            self._emit_llm_failure(
                protocol=protocol,
                parent_run_id=str(parent_run_id),
                events=events,
                prompt_hash=hashed,
                raw=raw,
                response=response,
                reason=str(exc),
            )
            raise PlanRefused(
                str(exc),
                orchestration_action=OrchestrationAction.NEED_HUMAN.value,
            ) from exc
        except Exception as exc:  # noqa: BLE001 — fail closed, do not invent a plan
            self._emit_llm_failure(
                protocol=protocol,
                parent_run_id=str(parent_run_id),
                events=events,
                prompt_hash=hashed,
                raw=raw,
                response=response,
                reason=f"fail_closed: gateway error: {exc}",
            )
            raise PlanRefused(
                f"fail_closed: gateway error: {exc}",
                orchestration_action=OrchestrationAction.NEED_HUMAN.value,
            ) from exc

        decision_summary = {
            "problem_observed": mapped["observation"],
            "hypothesis": mapped["hypothesis"],
            "candidate_actions": [
                "continue_module",
                "switch_module",
                "replicate",
                "NEED_HUMAN",
            ],
            "selected_action": mapped["selected_action"],
            "decision_basis": [
                "llm_planner_contract",
                f"requested_module={mapped['modification_scope'][0]}",
                f"review_decision={last_review_decision}",
                f"cited_lessons={mapped['memory_refs']['lesson_ids']}",
                f"cited_strategies={mapped['memory_refs']['strategy_ids']}",
                f"prompt_hash={hashed}",
                "Adapter remains HOW-only; Gate is not bypassed",
            ],
            "expected_effect": (
                f"{mapped['expected_effect']['direction']} "
                f"{mapped['expected_effect']['primary_metric']}"
            ),
            "risk": "llm Planner; fail closed; Adapter remains HOW-only",
        }
        try:
            plan = build_candidate_next_plan(
                previous_plan,
                memory,
                parent_run_id=str(parent_run_id),
                proposed_changes=mapped["proposed_changes"],
                modification_scope=mapped["modification_scope"],
                hypothesis=mapped["hypothesis"],
                observation=mapped["observation"],
                expected_effect=mapped["expected_effect"],
                evaluation=dict(previous_plan.get("evaluation") or {"method": "fast_eval"}),
                budget_class=str(mapped.get("budget_class") or previous_plan.get("budget_class") or "probe"),
                risk_level=str(previous_plan.get("risk_level") or "auto"),
                rationale=(
                    f"LLM Planner citing {mapped['memory_refs']['lesson_ids']} / "
                    f"{mapped['memory_refs']['strategy_ids']} from {parent_run_id}."
                ),
                decision_summary=decision_summary,
                memory_refs=mapped["memory_refs"],
                evidence_runs=evidence_runs,
                controlled_variables=list(
                    previous_plan.get("controlled_variables")
                    or protocol.get("frozen_scope")
                    or ["evaluator"]
                ),
            )
        except InvariantError as exc:
            self._emit_llm_failure(
                protocol=protocol,
                parent_run_id=str(parent_run_id),
                events=events,
                prompt_hash=hashed,
                raw=raw,
                response=response,
                reason=f"fail_closed: illegal Plan: {exc}",
            )
            raise PlanRefused(
                f"fail_closed: Planner refused to emit an illegal Plan: {exc}",
                orchestration_action=OrchestrationAction.NEED_HUMAN.value,
            ) from exc

        provider_name = response.provider if response is not None else "unknown"
        model_name = response.model if response is not None else "unknown"
        plan["llm_trace"] = {
            "backend": "llm",
            "provider": provider_name,
            "model": model_name,
            "prompt_hash": hashed,
            "raw_output": raw,
            "parsed": mapped.get("parsed") or {},
            "selected_candidate_id": str(mapped.get("selected_candidate_id") or "selected"),
            "memory_refs": dict(mapped["memory_refs"]),
        }
        if mapped.get("candidate_experiments"):
            plan["candidate_experiments"] = list(mapped["candidate_experiments"])
        validate_named("experiment_plan", plan)

        if events is not None:
            events.append(
                {
                    "project_id": protocol["project_id"],
                    "run_id": str(parent_run_id),
                    "plan_id": plan["plan_id"],
                    "protocol_version": protocol.get("protocol_version"),
                    "fingerprint_id": protocol.get("fingerprint_id"),
                    "event_type": "plan_proposal",
                    "actor_role": "planner",
                    "phase": "planning",
                    "memory_refs": dict(plan["memory_refs"]),
                    "decision_summary": dict(decision_summary),
                    "payload": {
                        "round_index": plan["round_index"],
                        "source": "llm",
                        "parent_run_id": str(parent_run_id),
                        "evidence_runs": list(plan["evidence_runs"]),
                        "provider": provider_name,
                        "model": model_name,
                        "prompt_hash": hashed,
                        "raw_output": raw[:8000],
                        "fail_closed": False,
                        "fallback_to_rules": False,
                    },
                }
            )
        return PlanPacket(
            plan=plan,
            decision_summary=decision_summary,
            source="llm",
            orchestration_action=OrchestrationAction.NEED_GATE.value,
        )

    def _emit_llm_failure(
        self,
        *,
        protocol: Mapping[str, Any],
        parent_run_id: str,
        events: EventAppender | None,
        prompt_hash: str,
        raw: str,
        response: Any,
        reason: str,
    ) -> None:
        if events is None:
            return
        events.append(
            {
                "project_id": protocol["project_id"],
                "run_id": str(parent_run_id),
                "protocol_version": protocol.get("protocol_version"),
                "fingerprint_id": protocol.get("fingerprint_id"),
                "event_type": "plan_proposal",
                "actor_role": "planner",
                "phase": "planning",
                "payload": {
                    "source": "llm",
                    "fail_closed": True,
                    "fallback_to_rules": False,
                    "reason": reason,
                    "provider": getattr(response, "provider", None),
                    "model": getattr(response, "model", None),
                    "prompt_hash": prompt_hash,
                    "raw_output": (raw or "")[:8000],
                },
            }
        )


def propose_and_gate_next(
    *,
    protocol: Mapping[str, Any],
    memory: MemoryWriter,
    previous_plan: Mapping[str, Any],
    parent_run_id: str,
    events: EventAppender | None = None,
    last_review_decision: str | None = None,
    observation: str | None = None,
    adapter: Any | None = None,
    planner: Planner | None = None,
    backend: str | None = None,
    provider: Any | None = None,
    fallback_to_rules: bool = False,
    live: bool = False,
) -> dict[str, Any]:
    """Planner.next_plan → Adapter.materialize → Gate. No Manager loop. No GPU."""
    actor = planner or Planner(
        backend=backend,
        provider=provider,
        fallback_to_rules=fallback_to_rules,
        live=live,
    )
    packet = actor.next_plan(
        protocol=protocol,
        memory=memory,
        previous_plan=previous_plan,
        parent_run_id=parent_run_id,
        last_review_decision=last_review_decision,
        observation=observation,
        events=events,
    )
    if adapter is None:
        from scientist_lab.adapters.dfine.adapter import DFINEAdapter

        adapter = DFINEAdapter(events=events)
    contract = adapter.materialize_contract(packet.plan, protocol)
    verdict = gate_candidate_next_plan(protocol, packet.plan, contract, memory)
    return {
        "plan": packet.plan,
        "decision_summary": packet.decision_summary,
        "source": packet.source,
        "contract": contract,
        "gate": verdict.to_dict(),
        "packet": packet,
    }
