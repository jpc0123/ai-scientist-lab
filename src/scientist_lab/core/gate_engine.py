"""Deterministic GateEngine. Does not KEEP/DISCARD; does not call GPU."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from scientist_lab.core.invariants import (
    InvariantError,
    assert_memory_refs_resolvable,
    assert_plan_memory_policy,
)


class GateStatus:
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    HUMAN_REQUIRED = "HUMAN_REQUIRED"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class GateVerdict:
    status: str
    reasons: tuple[str, ...] = ()
    fingerprint_comparable: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reasons": list(self.reasons),
            "fingerprint_comparable": self.fingerprint_comparable,
        }


class GateEngine:
    """Protocol / Risk / memory_refs / Frozen Fingerprint."""

    def evaluate(
        self,
        protocol: Mapping[str, Any],
        plan: Mapping[str, Any],
        contract: Mapping[str, Any],
        *,
        bound_fingerprint: Mapping[str, Any] | None = None,
        memory: Mapping[str, Any] | Any | None = None,
    ) -> GateVerdict:
        reasons: list[str] = []
        status = GateStatus.APPROVED
        comparable: bool | None = None

        rank = {
            GateStatus.APPROVED: 0,
            GateStatus.HUMAN_REQUIRED: 1,
            GateStatus.REJECTED: 2,
            GateStatus.BLOCKED: 3,
        }

        def bump(new: str) -> None:
            nonlocal status
            if rank[new] > rank[status]:
                status = new

        try:
            assert_plan_memory_policy(plan)
        except InvariantError as exc:
            reasons.append(str(exc))
            bump(GateStatus.REJECTED)

        if memory is not None:
            lessons, strategies = _memory_catalog(memory)
            try:
                assert_memory_refs_resolvable(
                    plan, lesson_ids=lessons, strategy_ids=strategies
                )
            except InvariantError as exc:
                reasons.append(str(exc))
                bump(GateStatus.REJECTED)

        frozen = set(protocol.get("frozen_scope") or [])
        scope = set(plan.get("modification_scope") or []) | set(
            contract.get("allowed_changes") or []
        )
        overlap = sorted(scope & frozen)
        if overlap:
            reasons.append(f"forbidden/frozen overlap: {overlap}")
            bump(GateStatus.REJECTED)

        # Lazy import: core must not import Adapter at package init.
        from scientist_lab.adapters.dfine.fingerprint import (
            compute_fingerprint,
            fingerprints_equivalent,
        )

        bound = bound_fingerprint or compute_fingerprint(protocol, None)
        actual = compute_fingerprint(protocol, contract)
        comparable = fingerprints_equivalent(bound, actual)
        if not comparable:
            reasons.append("Frozen Fingerprint CHANGED; comparability contract broken")
            bump(GateStatus.BLOCKED)

        if plan.get("risk_level") == "approval_required":
            reasons.append("plan.risk_level=approval_required")
            bump(GateStatus.HUMAN_REQUIRED)
        elif plan.get("budget_class") == "formal":
            full = (protocol.get("risk_policy") or {}).get("full_training", "approval_required")
            if full == "approval_required":
                reasons.append("formal budget requires Human Gate")
                bump(GateStatus.HUMAN_REQUIRED)
            elif full == "forbidden":
                reasons.append("formal training forbidden by protocol")
                bump(GateStatus.REJECTED)

        return GateVerdict(status=status, reasons=tuple(reasons), fingerprint_comparable=comparable)


def _memory_catalog(memory: Any) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Accept MemoryWriter or {lessons, strategies} / {lesson_ids, strategy_ids}."""
    if hasattr(memory, "load_lessons") and hasattr(memory, "load_strategies"):
        return memory.load_lessons(), memory.load_strategies()
    if not isinstance(memory, Mapping):
        return {}, {}
    lessons = memory.get("lessons") or memory.get("lesson_ids") or {}
    strategies = memory.get("strategies") or memory.get("strategy_ids") or {}
    if isinstance(lessons, (list, tuple, set)):
        lessons = {str(i): {} for i in lessons}
    if isinstance(strategies, (list, tuple, set)):
        strategies = {str(i): {} for i in strategies}
    return dict(lessons), dict(strategies)
