"""Freeze Planner: WHAT/WHY. Rules-first. Does not write model CLI.

Reuses build_candidate_next_plan as the N+1 skeleton and fills scientific
semantics from Protocol + written Memory. Not a new Agent kind. No LLM in v1.
Does not KEEP/DISCARD. Does not invent lesson ids. Does not edit Protocol.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from scientist_lab.core.invariants import InvariantError
from scientist_lab.core.memory_writer import MemoryWriter
from scientist_lab.core.next_plan import (
    _as_catalog,
    build_candidate_next_plan,
    gate_candidate_next_plan,
)
from scientist_lab.core.round_control import (
    apply_next_round_seed,
    contrast_fusion_how_id,
    plan_how_token,
    plan_seed,
    should_contrast_fusion_how,
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


def _bind_next_round_seed(
    plan: Mapping[str, Any],
    *,
    previous_plan: Mapping[str, Any],
    last_review_decision: str | None,
    requested_seed: Any = None,
    how_id: str | None = None,
) -> dict[str, Any]:
    next_plan = dict(plan)
    token = str(how_id or next_plan.get("how_id") or "").strip().upper()
    if token:
        next_plan["how_id"] = token
    return apply_next_round_seed(
        next_plan,
        previous_plan=previous_plan,
        last_review_decision=last_review_decision,
        requested_seed=requested_seed,
    )


def _apply_fusion_how_policy(
    *,
    previous_plan: Mapping[str, Any],
    last_review_decision: str | None,
    last_primary_delta: float | None,
    selected_how: str | None,
) -> dict[str, Any]:
    """Rewrite idle F0/F1/F3 repeats into a catalog contrast. Does not invent HOW."""
    current = plan_how_token(previous_plan)
    if not should_contrast_fusion_how(
        previous_plan=previous_plan,
        last_review_decision=last_review_decision,
        last_primary_delta=last_primary_delta,
    ):
        token = str(selected_how or current or "").strip().upper()
        return {"rewrite": False, "how_id": token or None, "current": current}
    nxt = contrast_fusion_how_id(current)
    chosen = str(selected_how or "").strip().upper()
    if chosen in {"F0", "F1", "F3"} and chosen != current:
        return {"rewrite": False, "how_id": chosen, "current": current}
    return {
        "rewrite": True,
        "how_id": nxt,
        "current": current,
        "reason": (
            f"idle HOW {current or 'F1'} blocked after "
            f"{last_review_decision or 'no-review'} delta={last_primary_delta}; "
            f"contrast {nxt}"
        ),
    }


def _scientific_semantics(
    *,
    protocol: Mapping[str, Any],
    previous_plan: Mapping[str, Any],
    lesson: Mapping[str, Any],
    strategy: Mapping[str, Any],
    last_review_decision: str | None,
    observation: str | None,
    last_primary_delta: float | None = None,
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
    fusion_policy = _apply_fusion_how_policy(
        previous_plan=previous_plan,
        last_review_decision=review,
        last_primary_delta=last_primary_delta,
        selected_how=previous_plan.get("how_id"),
    )
    if fusion_policy.get("how_id") and (
        fusion_policy["rewrite"]
        or should_contrast_fusion_how(
            previous_plan=previous_plan,
            last_review_decision=review,
            last_primary_delta=last_primary_delta,
        )
    ):
        nxt = str(fusion_policy["how_id"])
        current = str(fusion_policy.get("current") or plan_how_token(previous_plan) or "F1")
        hypothesis = (
            f"After {current} ({review or 'no-review'}, delta={last_primary_delta}), "
            f"registered HOW {nxt} is the next probe of {metric} rather than "
            f"idling the same fusion switch."
        )
        summary = (
            f"Contrast registered HOW {nxt} after {current}; Adapter remains HOW-only."
        )
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
            "modification_scope": ["fusion"],
            "proposed_changes": [
                {
                    "target": "fusion",
                    "summary": summary,
                    "detail": {"how_id": nxt, "neck_type": "standard"},
                }
            ],
            "hypothesis": hypothesis,
            "observation": obs,
            "controlled_variables": frozen
            or list(previous_plan.get("controlled_variables") or ["evaluator"]),
            "expected_effect": {
                "primary_metric": metric,
                "direction": direction,
                "rationale": f"Protocol objective.primary={metric}; fusion HOW contrast.",
            },
            "selected_action": f"contrast_{nxt}",
            "switch": False,
            "target": "fusion",
            "how_id": nxt,
            "how_rewrite": fusion_policy.get("reason"),
        }
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
        "how_id": None if switch else previous_plan.get("how_id"),
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
        pending_store: Path | str | None = None,
        literature_packet: Mapping[str, Any] | None = None,
        literature_environ: Mapping[str, str] | None = None,
    ) -> None:
        self.backend = resolve_planner_backend(backend)
        self.provider = provider
        self.fallback_to_rules = bool(fallback_to_rules)
        self.live = bool(live)
        self.pending_store = Path(pending_store) if pending_store else None
        self.literature_packet = dict(literature_packet) if literature_packet else None
        self.literature_environ = dict(literature_environ) if literature_environ is not None else None

    def _bind_literature(
        self,
        *,
        protocol: Mapping[str, Any],
        parent_run_id: str,
        previous_plan: Mapping[str, Any] | None = None,
        last_review_decision: str | None = None,
        last_metrics: Mapping[str, Any] | None = None,
        last_primary_delta: float | None = None,
        memory: MemoryWriter | Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Scout now and persist before LLM. Humans must see papers or an explicit fail."""
        literature = dict(self.literature_packet) if self.literature_packet else {}
        if self.pending_store is None:
            return literature
        from scientist_lab.core.how_pending import (
            load_store,
            persist_scout,
            resolve_scout_query,
            scout_literature_for_how,
        )
        from scientist_lab.core.round_control import plan_how_token

        if not literature:
            lessons, strategies = _as_catalog(memory) if memory is not None else ({}, {})
            evidence = {
                "last_how_id": plan_how_token(previous_plan or {}),
                "last_review_decision": last_review_decision,
                "last_metrics": dict(last_metrics or {}),
                "last_primary_delta": last_primary_delta,
                "keep_is_not_claim": True,
                "lessons": [
                    {
                        "statement": str(row.get("statement") or ""),
                        "type": str(row.get("type") or ""),
                    }
                    for row in list(lessons.values())[:3]
                    if isinstance(row, Mapping)
                ],
                "strategies": [
                    {
                        "action": str(row.get("action") or ""),
                        "target": str(row.get("target") or ""),
                    }
                    for row in list(strategies.values())[:2]
                    if isinstance(row, Mapping)
                ],
            }
            store = load_store(self.pending_store)
            resolved = resolve_scout_query(
                store,
                protocol,
                evidence,
                previous_plan,
                live=self.live,
                provider=self.provider if self.live else None,
            )
            literature = scout_literature_for_how(
                protocol=protocol,
                live=self.live,
                round_id=str(parent_run_id or "round_unspecified"),
                provenance_dir=self.pending_store.parent / "literature",
                query=resolved["query"],
                queries=list(resolved.get("queries") or [resolved["query"]]),
                environ=self.literature_environ,
                year_from=int(resolved.get("year_from") or 2022),
                research_question=str(resolved.get("research_question") or ""),
                ledger=list(store.get("literature_ledger") or []),
            )
            literature["source"] = resolved["source"]
            literature["intent_source"] = resolved["source"]
            literature["intent_why"] = resolved["why"]
            literature["intent_fallback"] = bool(resolved.get("fallback"))
            literature["query"] = resolved["query"]
            literature["queries"] = list(resolved.get("queries") or [resolved["query"]])
            literature["research_question"] = resolved.get("research_question")
        persist_scout(self.pending_store, literature, dialogue=True)
        return literature

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
        last_metrics: Mapping[str, Any] | None = None,
        last_primary_delta: float | None = None,
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
                    last_metrics=last_metrics,
                    last_primary_delta=last_primary_delta,
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
                    last_metrics=last_metrics,
                    last_primary_delta=last_primary_delta,
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
            last_metrics=last_metrics,
            last_primary_delta=last_primary_delta,
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
        last_metrics: Mapping[str, Any] | None = None,
        last_primary_delta: float | None = None,
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
            last_primary_delta=last_primary_delta,
        )
        if not semantics["proposed_changes"] or not semantics["modification_scope"]:
            raise PlanRefused(
                "Planner refuses a vague Plan without target/proposed_changes; "
                "will not ask Adapter to invent a module",
                orchestration_action=OrchestrationAction.NEED_HUMAN.value,
            )
        self._bind_literature(
            protocol=protocol,
            parent_run_id=str(parent_run_id),
            previous_plan=previous_plan,
            last_review_decision=review,
            last_metrics=last_metrics,
            last_primary_delta=last_primary_delta,
            memory=memory,
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
                f"last_primary_delta={last_primary_delta}",
                f"last_metrics={sorted((last_metrics or {}).keys())}",
                "rules-first Planner; no CoT; no Protocol edit",
            ]
            + (
                [str(semantics.get("how_rewrite"))]
                if semantics.get("how_rewrite")
                else []
            ),
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
                how_id=semantics.get("how_id"),
            )
        except InvariantError as exc:
            raise PlanRefused(
                f"Planner refused to emit an illegal Plan: {exc}",
                orchestration_action=OrchestrationAction.NEED_HUMAN.value,
            ) from exc
        plan = _bind_next_round_seed(
            plan,
            previous_plan=previous_plan,
            last_review_decision=review,
            how_id=semantics.get("how_id"),
        )

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
        last_metrics: Mapping[str, Any] | None = None,
        last_primary_delta: float | None = None,
    ) -> PlanPacket:
        from scientist_lab.core.schema_registry import validate_named
        from scientist_lab.llm.config import redact_secrets
        from scientist_lab.llm.errors import StructuredOutputValidationError
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

        literature = self._bind_literature(
            protocol=protocol,
            parent_run_id=str(parent_run_id),
            previous_plan=previous_plan,
            last_review_decision=last_review_decision,
            last_metrics=last_metrics,
            last_primary_delta=last_primary_delta,
            memory=memory,
        )
        from scientist_lab.core.how_pending import (
            load_store,
            overlay_from_store,
            planner_literature_context,
        )

        experiment_brief: dict[str, Any] = {}
        human_steer: dict[str, Any] = {}
        idea_snapshot: dict[str, Any] = {}
        if self.pending_store is not None:
            campaign_path = self.pending_store.parent / "campaign.json"
            if campaign_path.is_file():
                import json as _json

                from scientist_lab.services.campaign_notebook import build_round_cards
                from scientist_lab.services.evaluation_matrix import build_live_experiment_brief
                from scientist_lab.core.campaign_steer import (
                    active_steer_for_planner,
                    load_store as load_steer,
                    steer_path,
                )

                camp = _json.loads(campaign_path.read_text(encoding="utf-8"))
                metric_token = _primary_metric(protocol)
                rounds = build_round_cards(camp, metric=metric_token)
                pending_blob = load_store(self.pending_store)
                stored = camp.get("live_m1_brief")
                if isinstance(stored, Mapping) and stored.get("unused_smoked_plugins") is not None:
                    experiment_brief = dict(stored)
                else:
                    experiment_brief = build_live_experiment_brief(
                        camp,
                        protocol=protocol,
                        rounds=rounds,
                        how_pending=pending_blob,
                    )
                # Always refresh unused-plugin priority from live overlay.
                refreshed = build_live_experiment_brief(
                    camp,
                    protocol=protocol,
                    rounds=rounds,
                    how_pending=pending_blob,
                )
                if refreshed.get("unused_smoked_plugins"):
                    experiment_brief = refreshed
                snap = camp.get("idea_snapshot")
                if isinstance(snap, Mapping):
                    idea_snapshot = dict(snap)
                steer_payload = active_steer_for_planner(load_steer(steer_path(self.pending_store.parent)))
                if steer_payload:
                    human_steer = dict(steer_payload)
                    if steer_payload.get("planner_may_use") and steer_payload.get("text"):
                        steer_line = f"Human steer (next round): {steer_payload['text']}"
                        observation = (
                            f"{observation} | {steer_line}" if observation else steer_line
                        )

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
            evidence={
                "evidence_runs": evidence_runs,
                "last_metrics": dict(last_metrics or {}),
                "last_primary_delta": last_primary_delta,
                "last_seed": plan_seed(previous_plan),
                "last_how_id": plan_how_token(previous_plan),
            },
            last_review_decision=last_review_decision,
            parent_run_id=str(parent_run_id),
            observation=observation,
            literature=planner_literature_context(literature),
            how_overlay=overlay_from_store(load_store(self.pending_store))
            if self.pending_store is not None
            else {},
            experiment_brief=experiment_brief,
            human_steer=human_steer,
            idea_snapshot=idea_snapshot,
        )
        request = build_planner_request(contract_in)
        hashed = prompt_hash(request.messages)
        response = None
        raw = ""
        try:
            # Live M1 plans are larger; 60s ReadTimeout is common on remote MaaS.
            import os

            current_timeout = float(os.environ.get("LLM_TIMEOUT_SECONDS") or 60.0)
            if self.live and current_timeout < 180.0:
                os.environ["LLM_TIMEOUT_SECONDS"] = "180"
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
            if self.pending_store is not None:
                from scientist_lab.core.how_pending import (
                    ingest_unmaterializable_selected,
                    try_ingest_llm_candidates,
                )

                try_ingest_llm_candidates(
                    self.pending_store,
                    mapped.get("how_candidates") or [],
                    literature=literature,
                    round_id=str(parent_run_id or ""),
                )
                pending_row = mapped.get("pending_unmaterializable")
                if isinstance(pending_row, Mapping) and pending_row.get("how_id"):
                    ingest_unmaterializable_selected(
                        self.pending_store,
                        pending_row,
                        round_id=str(parent_run_id or ""),
                    )
            if mapped.get("how_executable") is False:
                hid = str(mapped.get("how_id") or "")
                raise PlanRefused(
                    f"selected HOW {hid!r} is not materializable; written as pending "
                    "draft (proposed / approved_pending_adapter). Will not start GPU "
                    "and will not rewrite it to another catalog HOW.",
                    orchestration_action=OrchestrationAction.NEED_HUMAN.value,
                )
        except PlanRefused:
            raise
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
        except StructuredOutputValidationError as exc:
            detail = "; ".join(list(exc.issues or [])[:6] or [str(exc)])
            blob = exc.content or exc.prior_content or raw
            self._emit_llm_failure(
                protocol=protocol,
                parent_run_id=str(parent_run_id),
                events=events,
                prompt_hash=hashed,
                raw=blob,
                response=response,
                reason=f"fail_closed: gateway error: {exc} | issues={detail}",
            )
            raise PlanRefused(
                f"fail_closed: gateway error: {exc} | issues={detail}",
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
                "LLM selected HOW; rules contrast is not applied",
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
                how_id=mapped.get("how_id"),
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
        plan = _bind_next_round_seed(
            plan,
            previous_plan=previous_plan,
            last_review_decision=last_review_decision,
            requested_seed=mapped.get("seed"),
            how_id=mapped.get("how_id"),
        )

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
        if mapped.get("verification_plan"):
            plan["verification_plan"] = dict(mapped["verification_plan"])
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
