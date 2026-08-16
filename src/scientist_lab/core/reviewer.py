"""Reviewer role: rules-first scientific decision after VALID evidence.

DecisionRubric computes objective/constraint checks. Reviewer maps those
checks to review_decision + structured lessons/strategy. Not CoT.
Does not recompute primary metrics. Does not run on non-VALID evidence.
Does not start Planner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from scientist_lab.core.decision_rubric import RubricResult
from scientist_lab.core.evidence_validator import EvidenceVerdict
from scientist_lab.core.schema_registry import validate_named
from scientist_lab.core.state_machine import ReviewDecisionValue


class ReviewRefused(ValueError):
    """Raised when Reviewer is asked to judge non-VALID evidence."""


@dataclass(frozen=True)
class ReviewPacket:
    review_decision: str
    document: dict[str, Any]
    decision_summary: dict[str, Any]
    research_lessons: list[dict[str, Any]]
    strategies: list[dict[str, Any]]

    def to_memory_review(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "research_lessons": list(self.research_lessons),
            "strategies": list(self.strategies),
        }
        if self.strategies:
            payload["strategy_update"] = self.strategies[0]
        return payload


def _as_rubric(rubric: RubricResult | Mapping[str, Any]) -> RubricResult:
    if isinstance(rubric, RubricResult):
        return rubric
    return RubricResult(
        objective_check=dict(rubric.get("objective_check") or {}),
        constraint_check=dict(rubric.get("constraint_check") or {}),
        primary_delta=rubric.get("primary_delta"),
        constraints_ok=bool(rubric.get("constraints_ok", True)),
        suggest_validate=bool(rubric.get("suggest_validate")),
        suggest_discard_threshold=bool(rubric.get("suggest_discard_threshold")),
        keep_threshold_ok=bool(rubric.get("keep_threshold_ok", True)),
    )


def _evidence_status(evidence: EvidenceVerdict | Mapping[str, Any]) -> str:
    if isinstance(evidence, EvidenceVerdict):
        return evidence.evidence_status
    return str(evidence.get("evidence_status") or "")


def _review_allowed(evidence: EvidenceVerdict | Mapping[str, Any]) -> bool:
    if isinstance(evidence, EvidenceVerdict):
        return evidence.review_allowed
    return bool(evidence.get("review_allowed"))


def _scope_target(
    plan: Mapping[str, Any] | None, contract: Mapping[str, Any]
) -> str:
    scope = list((plan or {}).get("modification_scope") or [])
    if not scope:
        scope = list(contract.get("allowed_changes") or [])
    if scope:
        return str(scope[0])
    changes = list((plan or {}).get("proposed_changes") or [])
    if changes and changes[0].get("target"):
        return str(changes[0]["target"])
    return "unknown"


def _decide(rubric: RubricResult) -> tuple[str, str, str, str]:
    """Map rubric flags to (review_decision, hypothesis_status, lesson_type, strategy_action)."""
    if not rubric.constraints_ok:
        return (
            ReviewDecisionValue.DISCARD.value,
            "REJECTED",
            "constraint_violation",
            "deprioritize",
        )
    if rubric.suggest_discard_threshold:
        return (
            ReviewDecisionValue.DISCARD.value,
            "REJECTED",
            "negative_evidence",
            "deprioritize",
        )
    if rubric.suggest_validate:
        return (
            ReviewDecisionValue.VALIDATE.value,
            "NEEDS_VALIDATION",
            "positive_evidence",
            "prioritize",
        )
    if rubric.primary_delta is None:
        return (
            ReviewDecisionValue.REPLICATE.value,
            "NEEDS_REPLICATION",
            "inconclusive",
            "keep",
        )
    if rubric.keep_threshold_ok:
        delta = rubric.primary_delta
        if delta is not None and delta > 0:
            return (
                ReviewDecisionValue.KEEP.value,
                "SUPPORTED",
                "positive_evidence",
                "prioritize",
            )
        if delta is not None and delta < 0:
            return (
                ReviewDecisionValue.KEEP.value,
                "INCONCLUSIVE",
                "inconclusive",
                "keep",
            )
        return (
            ReviewDecisionValue.KEEP.value,
            "INCONCLUSIVE",
            "inconclusive",
            "keep",
        )
    return (
        ReviewDecisionValue.REPLICATE.value,
        "NEEDS_REPLICATION",
        "inconclusive",
        "keep",
    )


def _statement(
    *,
    lesson_type: str,
    metric: str,
    delta: float | None,
    decision: str,
    target: str,
) -> str:
    delta_txt = "unknown" if delta is None else f"{delta:+.4g}"
    if lesson_type == "negative_evidence":
        return (
            f"Primary {metric} declined ({delta_txt}) past discard_if; "
            f"{decision} candidate targeting {target}."
        )
    if lesson_type == "constraint_violation":
        return (
            f"Constraints failed while judging {metric} ({delta_txt}); "
            f"{decision} candidate targeting {target}."
        )
    if lesson_type == "positive_evidence":
        return (
            f"Primary {metric} improved ({delta_txt}) within constraints; "
            f"{decision} candidate targeting {target}."
        )
    return (
        f"Primary {metric} delta {delta_txt} did not cross validate/discard "
        f"thresholds; {decision} candidate targeting {target}."
    )


class Reviewer:
    """Thin Reviewer role. Rules-first; no LLM; no Planner."""

    def review(
        self,
        *,
        result: Mapping[str, Any],
        evidence: EvidenceVerdict | Mapping[str, Any],
        rubric: RubricResult | Mapping[str, Any],
        contract: Mapping[str, Any],
        protocol: Mapping[str, Any],
        plan: Mapping[str, Any] | None = None,
        memory: Mapping[str, Any] | None = None,
    ) -> ReviewPacket:
        del memory  # read-only context reserved for later; unused in rules-first MVP
        status = _evidence_status(evidence)
        if status != "VALID" or not _review_allowed(evidence):
            raise ReviewRefused(
                f"Reviewer refuses non-VALID evidence (evidence_status={status})"
            )

        rubric_obj = _as_rubric(rubric)
        run_id = str(result.get("run_id") or contract.get("run_id") or "")
        if not run_id:
            raise ReviewRefused("Reviewer requires run_id on VALID result")

        decision, hypothesis, lesson_type, strategy_action = _decide(rubric_obj)
        primary = ((protocol.get("objective") or {}).get("primary") or {})
        metric = str(primary.get("metric") or "APS")
        target = _scope_target(plan, contract)
        delta = rubric_obj.primary_delta
        current_metrics = dict(result.get("metrics") or {})
        current = current_metrics.get(metric)
        baseline = None
        check = (rubric_obj.objective_check or {}).get(metric) or {}
        if isinstance(check, Mapping):
            baseline = check.get("baseline")
            if current is None:
                current = check.get("current")

        lesson_id = f"LESSON-{run_id}-001"
        strategy_id = f"STRATEGY-{run_id}-001"
        lesson = {
            "lesson_id": lesson_id,
            "type": lesson_type,
            "statement": _statement(
                lesson_type=lesson_type,
                metric=metric,
                delta=delta,
                decision=decision,
                target=target,
            ),
            "status": "active",
            "evidence": [{"run_id": run_id, "metric": metric, "delta": delta}],
            "scope": {
                "task": str(
                    ((protocol.get("goal") or {}).get("task_type") or "rgbt_detection")
                ),
                "module": target,
            },
            "confidence": "medium",
            "created_from": [run_id],
            "contradicted_by": [],
            "supersedes": [],
            "expires_when": [],
        }
        strategy = {
            "strategy_id": strategy_id,
            "action": strategy_action,
            "target": target,
            "reason_lesson_ids": [lesson_id],
            "status": "active",
        }

        hypothesis_text = str((plan or {}).get("hypothesis") or contract.get("hypothesis") or "")
        decision_summary = {
            "problem_observed": str((plan or {}).get("observation") or "VALID evidence ready for review"),
            "hypothesis": hypothesis_text,
            "candidate_actions": ["KEEP", "DISCARD", "REPLICATE", "VALIDATE", "ESCALATE"],
            "selected_action": decision,
            "decision_basis": [
                f"suggest_discard_threshold={rubric_obj.suggest_discard_threshold}",
                f"suggest_validate={rubric_obj.suggest_validate}",
                f"keep_threshold_ok={rubric_obj.keep_threshold_ok}",
                f"constraints_ok={rubric_obj.constraints_ok}",
                f"primary_delta={delta}",
            ],
            "expected_effect": str(
                ((plan or {}).get("expected_effect") or {}).get("direction")
                or f"judge {metric}"
            ),
            "risk": "rules-first Reviewer; no free-form CoT",
        }
        document = {
            "schema_version": "1.0.0",
            "run_id": run_id,
            "hypothesis_status": hypothesis,
            "review_decision": decision,
            "reasoning_summary": (
                f"{decision} from DecisionRubric: {metric} delta={delta}, "
                f"discard={rubric_obj.suggest_discard_threshold}, "
                f"validate={rubric_obj.suggest_validate}, "
                f"keep_ok={rubric_obj.keep_threshold_ok}, "
                f"constraints_ok={rubric_obj.constraints_ok}."
            ),
            "objective_check": rubric_obj.objective_check,
            "constraint_check": rubric_obj.constraint_check,
            "research_lessons": [lesson],
            "strategies": [strategy],
            "primary_metric_judgment": {
                "metric": metric,
                "before": baseline,
                "after": current,
                "delta": delta,
                "within_constraints": rubric_obj.constraints_ok,
            },
        }
        validate_named("review_decision", document)
        validate_named("research_lesson", lesson)
        validate_named("strategy", strategy)
        return ReviewPacket(
            review_decision=decision,
            document=document,
            decision_summary=decision_summary,
            research_lessons=[lesson],
            strategies=[strategy],
        )
