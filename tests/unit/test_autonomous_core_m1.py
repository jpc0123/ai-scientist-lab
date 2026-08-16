"""Architecture Freeze domain schema increment + DecisionRubric + Instrumentation."""

from __future__ import annotations

from pathlib import Path

import pytest

from scientist_lab.adapters import DFINEAdapter, MaterializeRejected
from scientist_lab.core import (
    InvariantError,
    assert_plan_memory_policy,
    evaluate_rubric,
    evaluate_stop_rules,
    list_schema_names,
    load_json,
    project_scientific_outcome,
    validate_named,
)
from scientist_lab.core.schema_registry import SCHEMA_DIR
from scientist_lab.core.state_machine import EvidenceStatus, RunState
from scientist_lab.instrumentation import EventAppender

EXAMPLES = SCHEMA_DIR / "examples"

LESSON = {
    "lesson_id": "LESSON-017",
    "type": "positive_evidence",
    "statement": "High-resolution neck enhancement improved small-object performance.",
    "status": "active",
    "evidence": [{"run_id": "EXP-007", "metric": "APS", "delta": 0.67}],
    "scope": {"task": "rgbt_tiny_detection", "module": "neck"},
    "confidence": "medium",
    "created_from": ["EXP-007"],
    "contradicted_by": [],
    "supersedes": [],
    "expires_when": [],
}

STRATEGY = {
    "strategy_id": "STRATEGY-009",
    "action": "prioritize",
    "target": "neck.high_resolution_path",
    "reason_lesson_ids": ["LESSON-017"],
    "status": "active",
}


def _round1_plan(**overrides):
    plan = {
        "schema_version": "1.0.0",
        "plan_id": "plan_round1_neck_hr",
        "project_id": "project_rgbt_cuda_001",
        "protocol_id": "research_protocol_rgbt_dfine_v1",
        "protocol_version": 1,
        "parent_run_id": "EXP-007",
        "round_index": 1,
        "observation": "APS improved with FDPN but localization still weak.",
        "hypothesis": "Enhancing high-res neck path further improves small-object APS.",
        "modification_scope": ["neck"],
        "proposed_changes": [
            {"target": "neck", "summary": "Adjust high-resolution feature path"}
        ],
        "controlled_variables": ["backbone", "fusion", "dataset_split", "evaluator"],
        "expected_effect": {
            "primary_metric": "APS",
            "direction": "increase",
            "rationale": "Better small-object localization features.",
        },
        "evaluation": {"method": "fast_eval", "seeds": [42]},
        "budget_class": "probe",
        "risk_level": "auto",
        "memory_refs": {
            "lesson_ids": ["LESSON-017"],
            "strategy_ids": ["STRATEGY-009"],
        },
        "evidence_runs": ["EXP-007"],
        "bootstrap": False,
        "rationale": "Round 1 cites EXP-007 / LESSON-017.",
        "decision_summary": {
            "problem_observed": "localization still weak",
            "hypothesis": "high-res neck path helps APS",
            "candidate_actions": ["increase resolution", "add FDPN", "modify loss"],
            "selected_action": "adjust high-res neck path",
            "decision_basis": ["targets small-object features", "fits editable_scope"],
            "expected_effect": "improve APS",
            "risk": "extra FLOPs",
        },
    }
    plan.update(overrides)
    return plan


def test_schema_dir_lists_learning_and_event_contracts() -> None:
    names = list_schema_names()
    for name in (
        "research_lesson",
        "strategy",
        "frozen_fingerprint",
        "research_event",
        "research_protocol",
        "experiment_plan",
        "claim",
        "claim_gate_result",
    ):
        assert name in names
    assert "trajectory_step" not in names


def test_example_research_protocol_validates() -> None:
    doc = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")
    validate_named("research_protocol", doc)
    assert doc["risk_policy"]["dataset_change"] == "forbidden"
    assert doc["objective"]["primary"]["metric"] == "APS"
    assert doc["decision_policy"]["validate_if"]["primary_gain_at_least"] == 0.5
    assert doc["stop_rules"]["max_rounds"] == 5


def test_fingerprint_example_validates() -> None:
    doc = load_json(EXAMPLES / "frozen_fingerprint_rgbt_dfine_v1.json")
    validate_named("frozen_fingerprint", doc)


def test_lesson_and_strategy_require_evidence_links() -> None:
    validate_named("research_lesson", LESSON)
    validate_named("strategy", STRATEGY)
    bad = dict(LESSON)
    bad["created_from"] = []
    with pytest.raises(Exception):
        validate_named("research_lesson", bad)


def test_experiment_plan_example_validates() -> None:
    plan = _round1_plan()
    validate_named("experiment_plan", plan)
    assert_plan_memory_policy(plan)


def test_round1_plan_without_memory_refs_fails_invariant() -> None:
    plan = _round1_plan(memory_refs={"lesson_ids": [], "strategy_ids": []}, evidence_runs=[])
    validate_named("experiment_plan", plan)
    with pytest.raises(InvariantError):
        assert_plan_memory_policy(plan)


def test_bootstrap_round0_may_omit_memory() -> None:
    plan = _round1_plan(
        round_index=0,
        bootstrap=True,
        parent_run_id=None,
        memory_refs={"lesson_ids": [], "strategy_ids": []},
        evidence_runs=[],
    )
    validate_named("experiment_plan", plan)
    assert_plan_memory_policy(plan)


def test_contract_result_review_chain_validates() -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")
    plan = _round1_plan()
    contract = DFINEAdapter().materialize_contract(plan, protocol)
    result = {
        "schema_version": "1.0.0",
        "run_id": contract["run_id"],
        "experiment_sha": "abcdef12",
        "metrics": {"APS": 45.02, "mAP50_95": 62.7, "params_m": 10.4},
        "metrics_delta": {"APS": 0.67, "mAP50_95": 0.08},
        "artifacts": {
            "paths": ["metrics.json", "checkpoint_selection.json"],
            "missing_expected": [],
        },
        "execution": {
            "status": "success",
            "exit_code": 0,
            "runtime_seconds": 512.0,
            "error_type": None,
            "error_message": None,
        },
        "scientific_outcome": "POSITIVE",
        "constraints_check": {"ok": True, "violations": []},
    }
    rubric = evaluate_rubric(
        protocol,
        current_metrics=result["metrics"],
        baseline_metrics={"APS": 44.35, "params_m": 10.4, "flops_g": 40.0},
    )
    review = {
        "schema_version": "1.0.0",
        "run_id": contract["run_id"],
        "hypothesis_status": "SUPPORTED",
        "review_decision": "KEEP",
        "reasoning_summary": "APS up within constraints; keep neck change.",
        "objective_check": rubric.objective_check,
        "constraint_check": rubric.constraint_check,
        "research_lessons": [LESSON],
        "strategies": [STRATEGY],
        "primary_metric_judgment": {
            "metric": "APS",
            "before": 44.35,
            "after": 45.02,
            "delta": 0.67,
            "within_constraints": True,
        },
    }
    run = {
        "schema_version": "1.0.0",
        "run_id": contract["run_id"],
        "project_id": "project_rgbt_cuda_001",
        "protocol_id": "research_protocol_rgbt_dfine_v1",
        "protocol_version": 1,
        "round_index": 1,
        "plan_id": plan["plan_id"],
        "run_state": "MEMORY_WRITTEN",
        "evidence_status": "VALID",
        "review_decision": "KEEP",
        "scientific_outcome": "POSITIVE",
        "fingerprint_id": "FP-RGBT-DFINE-V1",
        "experiment_sha": "abcdef12",
        "contract_ref": "contracts/run_0001.json",
        "result_ref": "results/run_0001.json",
        "review_ref": "reviews/run_0001.json",
    }
    validate_named("experiment_contract", contract)
    validate_named("experiment_result", result)
    validate_named("review_decision", review)
    validate_named("experiment_run", run)


def test_scientific_outcome_not_mapped_from_review_decision() -> None:
    assert (
        project_scientific_outcome(
            run_state=RunState.COMPLETED,
            evidence_status=EvidenceStatus.VALID,
            primary_delta=0.02,
        )
        == "POSITIVE"
    )
    assert (
        project_scientific_outcome(
            run_state=RunState.FAILED,
            evidence_status=EvidenceStatus.NOT_APPLICABLE,
            primary_delta=None,
        )
        == "NOT_EVALUATED"
    )


def test_keep_may_accompany_inconclusive_outcome() -> None:
    """INCONCLUSIVE + KEEP is allowed: worth validating further."""
    outcome = project_scientific_outcome(
        run_state=RunState.COMPLETED,
        evidence_status=EvidenceStatus.VALID,
        primary_delta=0.0,
    )
    assert outcome == "INCONCLUSIVE"


def test_stop_rules_novelty_exhausted() -> None:
    action = evaluate_stop_rules(
        {"max_duplicate_plan_rejects": 3},
        round_index=2,
        duplicate_plan_rejects=3,
    )
    assert action == "NOVELTY_EXHAUSTED"


def test_adapter_rejects_vague_plan() -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")
    plan = _round1_plan(proposed_changes=[])
    with pytest.raises(MaterializeRejected):
        DFINEAdapter().materialize_contract(plan, protocol)


def test_event_appender_writes_canonical_facts(tmp_path: Path) -> None:
    path = tmp_path / "research_events.jsonl"
    writer = EventAppender(path)
    event = writer.append(
        {
            "project_id": "project_rgbt_cuda_001",
            "round_id": "round_01",
            "run_id": "run_0001",
            "event_type": "execution",
            "actor_role": "adapter",
            "phase": "experiment",
            "protocol_version": 1,
            "plan_id": "plan_round1_neck_hr",
            "fingerprint_id": "FP-RGBT-DFINE-V1",
            "payload": {"tool": "run_experiment", "status": "ok"},
        }
    )
    rows = writer.load_all()
    assert len(rows) == 1
    assert rows[0]["event_id"] == event["event_id"]
    assert rows[0]["reconstructed"] is False


def test_negative_result_state_triplet() -> None:
    from scientist_lab.core.state_machine import (
        ExperimentRunState,
        OrchestrationAction,
        ReviewDecisionValue,
        apply_happy_path_negative_result,
        next_orchestration_action,
    )

    s = ExperimentRunState.initial()
    s = apply_happy_path_negative_result(s)
    assert s.run_state == RunState.MEMORY_WRITTEN
    assert s.evidence_status == EvidenceStatus.VALID
    assert s.review_decision == ReviewDecisionValue.DISCARD
    assert next_orchestration_action(s) == OrchestrationAction.NEXT_ROUND


def test_cannot_keep_without_valid_evidence() -> None:
    from scientist_lab.core.state_machine import (
        ExperimentRunState,
        InvalidTransition,
        ReviewDecisionValue,
        transition,
    )

    s = ExperimentRunState(
        run_state=RunState.COMPLETED,
        evidence_status=EvidenceStatus.INVALID,
        review_decision=ReviewDecisionValue.PENDING,
    )
    with pytest.raises(InvalidTransition):
        transition(s, review_decision=ReviewDecisionValue.KEEP)


def test_failed_suggests_retry_execution_not_replicate() -> None:
    from scientist_lab.core.state_machine import (
        ExperimentRunState,
        OrchestrationAction,
        ReviewDecisionValue,
        next_orchestration_action,
    )

    s = ExperimentRunState(
        run_state=RunState.FAILED,
        evidence_status=EvidenceStatus.NOT_APPLICABLE,
        review_decision=ReviewDecisionValue.PENDING,
    )
    assert next_orchestration_action(s) == OrchestrationAction.RETRY_EXECUTION


def test_illegal_run_state_jump() -> None:
    from scientist_lab.core.state_machine import ExperimentRunState, InvalidTransition, transition

    s = ExperimentRunState.initial()
    with pytest.raises(InvalidTransition):
        transition(s, run_state=RunState.RUNNING)


def test_auto_path_skips_human_gate_state() -> None:
    from scientist_lab.core.state_machine import (
        ExperimentRunState,
        OrchestrationAction,
        next_orchestration_action,
        transition,
    )

    s = ExperimentRunState.initial()
    s = transition(s, run_state=RunState.PLANNED)
    s = transition(s, run_state=RunState.MATERIALIZED)
    s = transition(s, run_state=RunState.APPROVED)
    assert next_orchestration_action(s) == OrchestrationAction.NEED_EXECUTION
