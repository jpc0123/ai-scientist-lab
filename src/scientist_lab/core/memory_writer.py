"""MemoryWriter consumes Canonical Research Events; it does not own them.

Persists evidence-linked lessons/strategies and a Research Trace projection.
Does not invent lessons from metrics. Does not KEEP/DISCARD.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Mapping

from scientist_lab.core.invariants import (
    InvariantError,
    assert_lesson_has_evidence,
    assert_memory_refs_resolvable,
    assert_plan_memory_policy,
)
from scientist_lab.core.schema_registry import SchemaValidationError, validate_named
from scientist_lab.instrumentation.appender import EventAppender

_TRACE_TYPES = (
    "derived_from",
    "supported_by",
    "contradicted_by",
    "used_by",
    "supersedes",
)


class MemoryWriter:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lessons_path = self.root / "research_memory.json"
        self.strategies_path = self.root / "strategy_memory.json"
        self.trace_path = self.root / "research_trace.json"

    def load_lessons(self) -> dict[str, dict[str, Any]]:
        return dict(self._load(self.lessons_path).get("lessons") or {})

    def load_strategies(self) -> dict[str, dict[str, Any]]:
        return dict(self._load(self.strategies_path).get("strategies") or {})

    def load_trace(self) -> list[dict[str, Any]]:
        return list(self._load(self.trace_path).get("edges") or [])

    def _load(self, path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _dump(self, path: Path, payload: Mapping[str, Any]) -> None:
        path.write_text(
            json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def persist_lesson(self, lesson: Mapping[str, Any]) -> dict[str, Any]:
        assert_lesson_has_evidence(lesson)
        validate_named("research_lesson", dict(lesson))
        lessons = self.load_lessons()
        lessons[str(lesson["lesson_id"])] = dict(lesson)
        self._dump(self.lessons_path, {"schema_version": "1.0.0", "lessons": lessons})
        return dict(lesson)

    def persist_semantic_proposal(
        self,
        proposal: Mapping[str, Any],
        *,
        run_id: str,
        review_decision: str,
        lesson_type: str = "negative_evidence",
        module: str = "unknown",
        task: str = "rgbt_detection",
        metric: str = "APS",
        delta: float | None = None,
    ) -> dict[str, Any]:
        """Accept an LLM Reviewer lesson proposal only after evidence_refs validate.

        LLM does not write Memory. Missing evidence_refs is a hard refuse.
        Does not persist strategies (Reviewer LLM is not HOW / not Rubric).
        """
        from scientist_lab.llm.reviewer_contract import (
            ReviewerContractError,
            proposal_to_research_lesson,
        )

        if not list(proposal.get("evidence_refs") or []):
            raise InvariantError("semantic proposal refused: missing evidence_refs")
        try:
            lesson = proposal_to_research_lesson(
                proposal,
                run_id=str(run_id),
                review_decision=str(review_decision),
                lesson_type=str(lesson_type),
                module=str(module),
                task=str(task),
                metric=str(metric),
                delta=delta,
            )
        except ReviewerContractError as exc:
            raise InvariantError(str(exc)) from exc
        return self.persist_lesson(lesson)

    def persist_strategy(self, strategy: Mapping[str, Any]) -> dict[str, Any]:
        validate_named("strategy", dict(strategy))
        lessons = self.load_lessons()
        missing = [
            lid for lid in (strategy.get("reason_lesson_ids") or []) if lid not in lessons
        ]
        if missing:
            raise InvariantError(f"Strategy reason_lesson_ids missing from memory: {missing}")
        strategies = self.load_strategies()
        strategies[str(strategy["strategy_id"])] = dict(strategy)
        self._dump(
            self.strategies_path,
            {"schema_version": "1.0.0", "strategies": strategies},
        )
        return dict(strategy)

    def consume(
        self,
        events: EventAppender | Sequence[Mapping[str, Any]],
        *,
        plan: Mapping[str, Any] | None = None,
        review: Mapping[str, Any] | None = None,
        require_resolved_refs: bool = False,
    ) -> dict[str, Any]:
        """Consume events + optional Reviewer structure. Never invents lessons."""
        rows = events.load_all() if isinstance(events, EventAppender) else list(events)
        refused: list[str] = []
        written_lessons: list[str] = []
        written_strategies: list[str] = []

        if review:
            for lesson in list(review.get("research_lessons") or []):
                try:
                    self.persist_lesson(lesson)
                    written_lessons.append(str(lesson["lesson_id"]))
                except (InvariantError, SchemaValidationError) as exc:
                    refused.append(f"lesson:{lesson.get('lesson_id')}:{exc}")
            strategies = [
                dict(row)
                for row in list(review.get("strategies") or [])
                if isinstance(row, Mapping)
            ]
            single = review.get("strategy_update")
            if isinstance(single, Mapping):
                sid = str(single.get("strategy_id") or "")
                if sid and sid not in {str(s.get("strategy_id") or "") for s in strategies}:
                    strategies.append(dict(single))
            for strategy in strategies:
                try:
                    self.persist_strategy(strategy)
                    written_strategies.append(str(strategy["strategy_id"]))
                except (InvariantError, SchemaValidationError) as exc:
                    refused.append(f"strategy:{strategy.get('strategy_id')}:{exc}")

        if plan is not None and require_resolved_refs:
            try:
                assert_memory_refs_resolvable(
                    plan,
                    lesson_ids=self.load_lessons(),
                    strategy_ids=self.load_strategies(),
                )
            except InvariantError as exc:
                refused.append(str(exc))

        edges = self.project_trace(rows, plan=plan)
        self._dump(self.trace_path, {"schema_version": "1.0.0", "edges": edges})
        return {
            "lessons_written": written_lessons,
            "strategies_written": written_strategies,
            "refused": refused,
            "trace_edges": len(edges),
            "invented_from_metrics": False,
        }

    def project_trace(
        self,
        events: list[Mapping[str, Any]],
        *,
        plan: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        edges: list[dict[str, Any]] = []
        lessons = self.load_lessons()
        strategies = self.load_strategies()

        def add(src: str, dst: str, kind: str, event_id: str | None = None) -> None:
            if kind not in _TRACE_TYPES or not src or not dst:
                return
            edge: dict[str, Any] = {"from": src, "to": dst, "type": kind}
            if event_id:
                edge["event_id"] = event_id
            edges.append(edge)

        for event in events:
            et = str(event.get("event_type") or "")
            eid = event.get("event_id")
            run_id = event.get("run_id")
            if et == "evidence_check" and run_id:
                add(str(run_id), str(eid or "evidence_check"), "supported_by", eid)
            if et == "strategy_revision" and eid:
                payload = dict(event.get("payload") or {})
                lesson_id = payload.get("lesson_id")
                strategy_id = payload.get("strategy_id")
                if run_id and lesson_id:
                    add(str(run_id), str(lesson_id), "derived_from", eid)
                if lesson_id and strategy_id:
                    add(str(lesson_id), str(strategy_id), "used_by", eid)
            if et == "memory_write" and eid:
                payload = dict(event.get("payload") or {})
                for lesson_id in payload.get("lesson_ids") or []:
                    if run_id:
                        add(str(run_id), str(lesson_id), "derived_from", eid)
            if et == "plan_proposal":
                plan_id = str(event.get("plan_id") or "")
                refs = dict(event.get("memory_refs") or {})
                payload = dict(event.get("payload") or {})
                parent = payload.get("parent_run_id") or event.get("run_id")
                if plan_id and parent:
                    add(str(parent), plan_id, "derived_from", eid)
                for lesson_id in refs.get("lesson_ids") or []:
                    if plan_id:
                        add(str(lesson_id), plan_id, "used_by", eid)
                for strategy_id in refs.get("strategy_ids") or []:
                    if plan_id:
                        add(str(strategy_id), plan_id, "used_by", eid)
                for evid in payload.get("evidence_runs") or []:
                    if plan_id:
                        add(str(evid), plan_id, "supported_by", eid)

        for lesson in lessons.values():
            for run_id in lesson.get("created_from") or []:
                add(str(run_id), str(lesson["lesson_id"]), "derived_from")
            for other in lesson.get("contradicted_by") or []:
                add(str(other), str(lesson["lesson_id"]), "contradicted_by")
            for other in lesson.get("supersedes") or []:
                add(str(lesson["lesson_id"]), str(other), "supersedes")

        for strategy in strategies.values():
            for lesson_id in strategy.get("reason_lesson_ids") or []:
                add(str(lesson_id), str(strategy["strategy_id"]), "used_by")

        if plan:
            plan_id = str(plan.get("plan_id") or "")
            parent = plan.get("parent_run_id")
            if plan_id and parent:
                add(str(parent), plan_id, "derived_from")
            refs = plan.get("memory_refs") or {}
            for lesson_id in refs.get("lesson_ids") or []:
                if plan_id:
                    add(str(lesson_id), plan_id, "used_by")
            for strategy_id in refs.get("strategy_ids") or []:
                if plan_id:
                    add(str(strategy_id), plan_id, "used_by")
            for run_id in plan.get("evidence_runs") or []:
                if plan_id:
                    add(str(run_id), plan_id, "supported_by")

        return _unique_edges(edges)

    def record_plan_citation(self, plan: Mapping[str, Any]) -> list[dict[str, Any]]:
        """When Plan N+1 is gated/materialized, persist used_by / derived_from / supported_by.

        Refs must resolve against written memory. Merges with existing Trace so
        Run N → lesson/strategy edges are kept. Does not invent lessons.
        """
        assert_plan_memory_policy(plan)
        assert_memory_refs_resolvable(
            plan,
            lesson_ids=self.load_lessons(),
            strategy_ids=self.load_strategies(),
        )
        added = self.project_trace([], plan=plan)
        merged = _unique_edges([*self.load_trace(), *added])
        self._dump(self.trace_path, {"schema_version": "1.0.0", "edges": merged})
        return merged


def _unique_edges(edges: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str, str]] = set()
    unique: list[dict[str, Any]] = []
    for edge in edges:
        row = dict(edge)
        key = (row["from"], row["to"], row["type"], str(row.get("event_id") or ""))
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique
