"""M4 Manager multi-round orchestration (stub live_runner). No GPU. No Bounded Tree."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from scientist_lab.core.manager import Manager
from scientist_lab.core.schema_registry import SCHEMA_DIR, load_json
from scientist_lab.core.state_machine import (
    EvidenceStatus,
    OrchestrationAction,
    ReviewDecisionValue,
    RunState,
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


def _install_seed(root: Path, *, protocol_overrides: dict | None = None) -> dict:
    protocol = dict(load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json"))
    if protocol_overrides:
        protocol.update(protocol_overrides)
    best = load_json(EXAMPLES / "recovered_k2c44_best_metrics.json")
    _write_json(root / "protocol.json", protocol)
    _write_json(root / "plan.json", _seed_plan())
    _write_json(root / "baseline_metrics.json", {"APS": best["APS"], "mAP50_95": best["mAP50_95"]})
    return protocol


def _copy_artifacts(dest: Path, metrics_name: str) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(EXAMPLES / metrics_name, dest / "metrics.json")
    shutil.copyfile(
        EXAMPLES / "recovered_k2c44_checkpoint_selection.json",
        dest / "checkpoint_selection.json",
    )


def _stub_by_round(calls: list):
    """Round 1 last=DISCARD; round 2 best=KEEP; round 3 last=DISCARD. No GPU."""

    def runner(contract, output_dir):
        n = len(calls)
        calls.append({"run_id": contract.get("run_id"), "plan_id": contract.get("plan_id")})
        name = (
            "recovered_k2c44_best_metrics.json"
            if n == 1
            else "recovered_k2c44_last_metrics.json"
        )
        _copy_artifacts(Path(output_dir), name)
        return {"status": "completed"}

    return runner


def test_three_stub_rounds_cite_memory_and_switch_module(tmp_path: Path) -> None:
    protocol = _install_seed(tmp_path)
    calls: list = []
    mgr = Manager(
        tmp_path,
        protocol=protocol,
        execute=True,
        live_runner=_stub_by_round(calls),
        max_extra_rounds=2,
    )
    mgr.initialize_run(round_index=1, plan_id="plan_round1_neck_hr")
    steps = mgr.run_until(max_steps=48)
    assert len(steps) <= 48
    assert len(calls) == 3
    first_archive = tmp_path / "runs" / str(calls[0]["run_id"])
    assert (first_archive / "result.json").is_file()

    execs = [s for s in steps if s.action == OrchestrationAction.NEED_EXECUTION.value]
    assert len(execs) == 3
    assert all(s.report.get("runner_called") is True for s in execs)

    reviews = [
        s.report.get("review", {}).get("review_decision")
        for s in steps
        if s.action == OrchestrationAction.NEED_REVIEW.value
    ]
    assert reviews[0] == ReviewDecisionValue.DISCARD.value
    assert reviews[1] == ReviewDecisionValue.KEEP.value
    assert all(s.state.evidence_status == EvidenceStatus.VALID for s in steps if s.action == OrchestrationAction.NEED_REVIEW.value)

    next_plans = [s for s in steps if s.action == OrchestrationAction.NEXT_ROUND.value]
    assert len(next_plans) >= 2
    assert next_plans[-1].idle is True
    assert "max_extra_rounds" in " ".join(next_plans[-1].reasons)

    round2 = load_json(tmp_path / "previous_plan.json")
    round3 = load_json(tmp_path / "plan.json")
    assert round2["round_index"] == 2
    assert round3["round_index"] == 3
    assert round2["modification_scope"] != ["neck"]
    assert round2["modification_scope"] == ["fusion"]
    assert "run_plan_round1_neck_hr" in " ".join(round2.get("evidence_runs") or [])
    lesson_ids = list((round2.get("memory_refs") or {}).get("lesson_ids") or [])
    strategy_ids = list((round2.get("memory_refs") or {}).get("strategy_ids") or [])
    assert lesson_ids
    assert any(lid in mgr.memory.load_lessons() for lid in lesson_ids)
    assert any(sid in mgr.memory.load_strategies() for sid in strategy_ids)

    round3_refs = list((round3.get("memory_refs") or {}).get("lesson_ids") or [])
    assert round3_refs
    assert any(lid in mgr.memory.load_lessons() for lid in round3_refs)
    assert str(round3.get("parent_run_id") or "") == str(calls[1]["run_id"])

    trace = mgr.memory.load_trace()
    used = {(e["from"], e["to"]) for e in trace if e.get("type") == "used_by"}
    derived = {(e["from"], e["to"]) for e in trace if e.get("type") == "derived_from"}
    assert any(src in lesson_ids and dst == round2["plan_id"] for src, dst in used)
    assert any(dst in lesson_ids for _src, dst in derived)
    assert any(src in strategy_ids and dst == round2["plan_id"] for src, dst in used)

    assert steps[-1].idle is True
    assert mgr.peek_action() in {
        OrchestrationAction.NEXT_ROUND.value,
        OrchestrationAction.STOP.value,
        OrchestrationAction.IDLE.value,
    }


def test_stop_rules_max_rounds_halts_before_extra_gpu(tmp_path: Path) -> None:
    protocol = _install_seed(
        tmp_path,
        protocol_overrides={
            "stop_rules": {
                "max_rounds": 2,
                "max_consecutive_discards": 5,
                "max_execution_failures": 5,
                "stop_if_no_improvement_for": 5,
                "max_duplicate_plan_rejects": 5,
            }
        },
    )
    calls: list = []
    mgr = Manager(
        tmp_path,
        protocol=protocol,
        execute=True,
        live_runner=_stub_by_round(calls),
        max_extra_rounds=5,
    )
    mgr.initialize_run(round_index=1, plan_id="plan_round1_neck_hr")
    steps = mgr.run_until(max_steps=32)
    assert len(calls) == 1
    assert any(s.action == OrchestrationAction.STOP.value for s in steps)
    assert steps[-1].idle is True
    assert steps[-1].state.run_state == RunState.STOPPED


def test_max_steps_hard_cap(tmp_path: Path) -> None:
    protocol = _install_seed(tmp_path)
    calls: list = []
    mgr = Manager(
        tmp_path,
        protocol=protocol,
        execute=True,
        live_runner=_stub_by_round(calls),
        max_extra_rounds=2,
    )
    mgr.initialize_run(round_index=1, plan_id="plan_round1_neck_hr")
    steps = mgr.run_until(max_steps=3)
    assert len(steps) == 3
    assert len(calls) == 0
