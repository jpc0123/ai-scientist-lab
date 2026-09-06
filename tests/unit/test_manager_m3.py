"""Thin Freeze Manager: state-machine dispatch. No GPU. No LLM loop."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from scientist_lab.adapters.dfine.adapter import DFINEAdapter
from scientist_lab.agents.manager import Manager as RoleManager
from scientist_lab.core.manager import Manager, _apply_state
from scientist_lab.core.schema_registry import SCHEMA_DIR, load_json
from scientist_lab.core.state_machine import (
    EvidenceStatus,
    ExperimentRunState,
    OrchestrationAction,
    ReviewDecisionValue,
    RunState,
    next_orchestration_action,
    transition,
)

EXAMPLES = SCHEMA_DIR / "examples"


def _seed_plan(**overrides):
    plan = {
        "schema_version": "1.0.0",
        "plan_id": "plan_round1_neck_hr",
        "project_id": "project_rgbt_cuda_001",
        "protocol_id": "research_protocol_rgbt_dfine_v1",
        "protocol_version": 1,
        "parent_run_id": "EXP-007",
        "round_index": 1,
        "observation": "localization still weak",
        "hypothesis": "high-res neck path helps APS",
        "modification_scope": ["neck"],
        "proposed_changes": [{"target": "neck", "summary": "Adjust high-res path"}],
        "controlled_variables": ["evaluator"],
        "expected_effect": {"primary_metric": "APS", "direction": "increase"},
        "evaluation": {"method": "fast_eval", "seeds": [42]},
        "budget_class": "probe",
        "risk_level": "auto",
        "memory_refs": {"lesson_ids": ["LESSON-017"], "strategy_ids": ["STRATEGY-009"]},
        "evidence_runs": ["EXP-007"],
    }
    plan.update(overrides)
    return plan


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _install_replay(root: Path, metrics_name: str = "recovered_k2c44_last_metrics.json") -> dict:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")
    best = load_json(EXAMPLES / "recovered_k2c44_best_metrics.json")
    shutil.copyfile(EXAMPLES / metrics_name, root / "metrics.json")
    shutil.copyfile(
        EXAMPLES / "recovered_k2c44_checkpoint_selection.json",
        root / "checkpoint_selection.json",
    )
    _write_json(root / "protocol.json", protocol)
    _write_json(root / "plan.json", _seed_plan())
    _write_json(root / "baseline_metrics.json", {"APS": best["APS"], "mAP50_95": best["mAP50_95"]})
    return protocol


def _walk_to_completed(state: ExperimentRunState) -> ExperimentRunState:
    for rs in (
        RunState.PLANNED,
        RunState.MATERIALIZED,
        RunState.APPROVED,
        RunState.RUNNING,
        RunState.COMPLETED,
    ):
        state = transition(state, run_state=rs)
    return state


def test_step_is_single_action(tmp_path: Path) -> None:
    protocol = _install_replay(tmp_path)
    mgr = Manager(tmp_path, protocol=protocol, execute=False, max_extra_rounds=0)
    mgr.initialize_run(round_index=1, plan_id="plan_round1_neck_hr")
    assert mgr.peek_action() == OrchestrationAction.NEED_PLAN.value
    first = mgr.step()
    assert first.action == OrchestrationAction.NEED_PLAN.value
    assert first.state.run_state == RunState.PLANNED
    assert mgr.peek_action() == OrchestrationAction.NEED_MATERIALIZE.value
    second = mgr.step()
    assert second.action == OrchestrationAction.NEED_MATERIALIZE.value
    assert second.state.run_state == RunState.MATERIALIZED


def test_replay_from_created_reaches_memory_written(tmp_path: Path) -> None:
    protocol = _install_replay(tmp_path)
    mgr = RoleManager(tmp_path, protocol=protocol, execute=False, max_extra_rounds=0)
    mgr.initialize_run(round_index=1, plan_id="plan_round1_neck_hr")
    steps = mgr.run_until(max_steps=20)
    assert steps
    assert steps[-1].state.run_state == RunState.MEMORY_WRITTEN
    assert steps[-1].state.evidence_status == EvidenceStatus.VALID
    assert steps[-1].state.review_decision == ReviewDecisionValue.DISCARD
    assert mgr.peek_action() == OrchestrationAction.NEXT_ROUND.value
    actions = [s.action for s in steps]
    assert OrchestrationAction.NEED_REVIEW.value in actions
    assert OrchestrationAction.NEED_MEMORY.value in actions
    events = (tmp_path / "research_events.jsonl").read_text(encoding="utf-8")
    assert '"actor_role": "manager"' in events
    lessons = mgr.memory.load_lessons()
    assert lessons
    assert next(iter(lessons.values()))["type"] == "negative_evidence"


def test_completed_valid_starts_at_review(tmp_path: Path) -> None:
    protocol = _install_replay(tmp_path)
    mgr = Manager(tmp_path, protocol=protocol, execute=False, max_extra_rounds=0)
    plan = _seed_plan()
    contract = DFINEAdapter().materialize_contract(plan, protocol)
    last = load_json(EXAMPLES / "recovered_k2c44_last_metrics.json")
    result = {
        "schema_version": "1.0.0",
        "run_id": contract["run_id"],
        "metrics": {
            k: last[k]
            for k in ("APS", "mAP50_95", "mAP50", "params_m", "flops_g", "gpu_memory_gb")
        },
        "artifacts": {
            "paths": ["metrics.json", "checkpoint_selection.json"],
            "missing_expected": [],
        },
        "execution": {"status": "success", "exit_code": 0},
        "raw_metric_refs": ["metrics.json"],
    }
    _write_json(tmp_path / "contract.json", contract)
    _write_json(tmp_path / "result.json", result)
    _write_json(
        tmp_path / "handle.json",
        {"status": "completed", "dry_run": True, "fingerprint_comparable": True},
    )
    doc = mgr.initialize_run(round_index=1, plan_id=plan["plan_id"], run_id=contract["run_id"])
    state = _walk_to_completed(ExperimentRunState.initial())
    state = transition(state, evidence_status=EvidenceStatus.VALID)
    _write_json(tmp_path / "experiment_run.json", _apply_state(doc, state))
    assert mgr.peek_action() == OrchestrationAction.NEED_REVIEW.value
    steps = mgr.run_until(max_steps=8)
    assert steps[-1].state.run_state == RunState.MEMORY_WRITTEN
    assert steps[-1].state.review_decision == ReviewDecisionValue.DISCARD


def test_invalid_evidence_never_reviews_keep(tmp_path: Path) -> None:
    protocol = _install_replay(tmp_path)
    mgr = Manager(tmp_path, protocol=protocol, execute=False, max_extra_rounds=0)
    doc = mgr.initialize_run(round_index=1)
    state = _walk_to_completed(ExperimentRunState.initial())
    state = transition(state, evidence_status=EvidenceStatus.INVALID)
    _write_json(tmp_path / "experiment_run.json", _apply_state(doc, state))
    assert mgr.peek_action() == OrchestrationAction.NEED_MEMORY.value
    step = mgr.step()
    assert step.action == OrchestrationAction.NEED_MEMORY.value
    assert step.state.run_state == RunState.MEMORY_WRITTEN
    assert step.state.review_decision == ReviewDecisionValue.PENDING
    assert not (tmp_path / "review.json").is_file()
    dumped = json.dumps(step.to_dict())
    assert '"KEEP"' not in dumped
    assert mgr.memory.load_lessons() == {}


def test_stop_rules_max_rounds_triggers_stop(tmp_path: Path) -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")
    protocol = dict(protocol)
    protocol["stop_rules"] = {**dict(protocol.get("stop_rules") or {}), "max_rounds": 1}
    mgr = Manager(tmp_path, protocol=protocol, execute=False)
    mgr.initialize_run(round_index=1)
    step = mgr.step()
    assert step.action == OrchestrationAction.STOP.value
    assert step.idle is True
    assert step.state.run_state == RunState.STOPPED
    assert any("stop_rules" in r for r in step.reasons)


def test_need_plan_without_memory_does_not_forge_refs(tmp_path: Path) -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")
    mgr = Manager(tmp_path, protocol=protocol, execute=False)
    mgr.initialize_run(round_index=1)
    _write_json(tmp_path / "previous_plan.json", _seed_plan())
    step = mgr.step()
    assert step.action == OrchestrationAction.NEED_HUMAN.value
    assert step.idle is True
    assert step.report.get("forged_refs") is False
    assert mgr.load_state().run_state == RunState.CREATED
    assert not (tmp_path / "plan.json").is_file()
    assert mgr.memory.load_lessons() == {}


def test_next_orchestration_not_applicable_skips_reviewer() -> None:
    s = ExperimentRunState(
        run_state=RunState.COMPLETED,
        evidence_status=EvidenceStatus.NOT_APPLICABLE,
        review_decision=ReviewDecisionValue.PENDING,
    )
    assert next_orchestration_action(s) == OrchestrationAction.NEED_MEMORY
