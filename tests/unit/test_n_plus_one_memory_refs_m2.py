"""N→N+1 Plan citability after REPLAY MemoryWriter. No Planner, no GPU."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from scientist_lab.adapters.base import MaterializeRejected
from scientist_lab.adapters.dfine.adapter import DFINEAdapter
from scientist_lab.adapters.dfine.run_loop import run_gated_dfine
from scientist_lab.core.gate_engine import GateEngine, GateStatus
from scientist_lab.core.invariants import (
    InvariantError,
    assert_memory_refs_resolvable,
    assert_plan_memory_policy,
)
from scientist_lab.core.memory_writer import MemoryWriter
from scientist_lab.core.next_plan import (
    build_candidate_next_plan,
    gate_candidate_next_plan,
)
from scientist_lab.core.schema_registry import SCHEMA_DIR, load_json, validate_named

EXAMPLES = SCHEMA_DIR / "examples"


def _plan(**overrides):
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


def _install_recovered(dest: Path, metrics_name: str) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(EXAMPLES / metrics_name, dest / "metrics.json")
    shutil.copyfile(
        EXAMPLES / "recovered_k2c44_checkpoint_selection.json",
        dest / "checkpoint_selection.json",
    )


def _replay_discard(tmp_path: Path) -> tuple[dict, MemoryWriter, dict, dict]:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")
    best = load_json(EXAMPLES / "recovered_k2c44_best_metrics.json")
    out = tmp_path / "round_n"
    _install_recovered(out, "recovered_k2c44_last_metrics.json")
    previous = _plan()
    report = run_gated_dfine(
        protocol=protocol,
        plan=previous,
        output_dir=out,
        execute=False,
        baseline_metrics={"APS": best["APS"], "mAP50_95": best["mAP50_95"]},
    )
    writer = MemoryWriter(out / "memory")
    return report, writer, protocol, previous


def test_recovered_valid_run_next_plan_with_memory_refs_passes(tmp_path: Path) -> None:
    report, writer, protocol, previous = _replay_discard(tmp_path)
    assert report["evidence"]["evidence_status"] == "VALID"
    assert report["review_decision"] == "DISCARD"
    assert report["memory"]["lessons_written"]
    run_id = report["contract_run_id"]
    lesson = next(iter(writer.load_lessons().values()))
    strategy = next(iter(writer.load_strategies().values()))
    assert lesson["type"] == "negative_evidence"

    next_plan = build_candidate_next_plan(
        previous, writer, parent_run_id=run_id
    )
    validate_named("experiment_plan", next_plan)
    assert_plan_memory_policy(next_plan)
    assert_memory_refs_resolvable(
        next_plan,
        lesson_ids=writer.load_lessons(),
        strategy_ids=writer.load_strategies(),
    )
    assert next_plan["round_index"] == 2
    assert next_plan["bootstrap"] is False
    assert next_plan["evidence_runs"] == [run_id]
    assert lesson["lesson_id"] in next_plan["memory_refs"]["lesson_ids"]
    assert strategy["strategy_id"] in next_plan["memory_refs"]["strategy_ids"]
    assert "FDPN" not in str(next_plan["proposed_changes"])
    assert next_plan["modification_scope"] == ["neck"]

    contract = DFINEAdapter().materialize_contract(next_plan, protocol)
    verdict = gate_candidate_next_plan(protocol, next_plan, contract, writer)
    assert verdict.status == GateStatus.APPROVED

    kinds = {(e["from"], e["to"], e["type"]) for e in writer.load_trace()}
    plan_id = next_plan["plan_id"]
    assert (run_id, lesson["lesson_id"], "derived_from") in kinds
    assert (lesson["lesson_id"], strategy["strategy_id"], "used_by") in kinds
    assert (lesson["lesson_id"], plan_id, "used_by") in kinds
    assert (strategy["strategy_id"], plan_id, "used_by") in kinds
    assert (run_id, plan_id, "supported_by") in kinds
    assert (run_id, plan_id, "derived_from") in kinds


def test_next_plan_missing_memory_refs_fails(tmp_path: Path) -> None:
    report, writer, protocol, previous = _replay_discard(tmp_path)
    next_plan = build_candidate_next_plan(
        previous, writer, parent_run_id=report["contract_run_id"]
    )
    next_plan = dict(next_plan)
    next_plan["memory_refs"] = {"lesson_ids": [], "strategy_ids": []}
    next_plan["evidence_runs"] = []
    validate_named("experiment_plan", next_plan)
    with pytest.raises(InvariantError, match="lesson_ids or strategy_ids"):
        assert_plan_memory_policy(next_plan)
    contract = DFINEAdapter().materialize_contract(next_plan, protocol)
    verdict = GateEngine().evaluate(protocol, next_plan, contract, memory=writer)
    assert verdict.status == GateStatus.REJECTED
    assert any("lesson_ids or strategy_ids" in r or "evidence_runs" in r for r in verdict.reasons)


def test_next_plan_bogus_lesson_ids_fails(tmp_path: Path) -> None:
    report, writer, protocol, previous = _replay_discard(tmp_path)
    next_plan = build_candidate_next_plan(
        previous, writer, parent_run_id=report["contract_run_id"]
    )
    next_plan = dict(next_plan)
    next_plan["memory_refs"] = {
        "lesson_ids": ["LESSON-FAKE-NOT-WRITTEN"],
        "strategy_ids": list(next_plan["memory_refs"]["strategy_ids"]),
    }
    with pytest.raises(InvariantError, match="not resolvable"):
        assert_memory_refs_resolvable(
            next_plan,
            lesson_ids=writer.load_lessons(),
            strategy_ids=writer.load_strategies(),
        )
    contract = DFINEAdapter().materialize_contract(next_plan, protocol)
    verdict = GateEngine().evaluate(protocol, next_plan, contract, memory=writer)
    assert verdict.status == GateStatus.REJECTED
    assert any("not resolvable" in r for r in verdict.reasons)
    with pytest.raises(InvariantError, match="not resolvable"):
        writer.record_plan_citation(next_plan)


def test_discard_negative_evidence_lesson_is_citable(tmp_path: Path) -> None:
    report, writer, protocol, previous = _replay_discard(tmp_path)
    lesson = next(iter(writer.load_lessons().values()))
    assert lesson["type"] == "negative_evidence"
    assert report["review_decision"] == "DISCARD"
    next_plan = build_candidate_next_plan(
        previous, writer, parent_run_id=report["contract_run_id"]
    )
    assert lesson["lesson_id"] in next_plan["memory_refs"]["lesson_ids"]
    validate_named("experiment_plan", next_plan)
    assert_plan_memory_policy(next_plan)
    contract = DFINEAdapter().materialize_contract(next_plan, protocol)
    verdict = GateEngine().evaluate(protocol, next_plan, contract, memory=writer)
    assert verdict.status == GateStatus.APPROVED


def test_keep_lesson_is_also_citable(tmp_path: Path) -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")
    best = load_json(EXAMPLES / "recovered_k2c44_best_metrics.json")
    out = tmp_path / "keep"
    _install_recovered(out, "recovered_k2c44_best_metrics.json")
    previous = _plan()
    report = run_gated_dfine(
        protocol=protocol,
        plan=previous,
        output_dir=out,
        execute=False,
        baseline_metrics={"APS": best["APS"]},
    )
    assert report["review_decision"] == "KEEP"
    writer = MemoryWriter(out / "memory")
    lesson = next(iter(writer.load_lessons().values()))
    next_plan = build_candidate_next_plan(
        previous, writer, parent_run_id=report["contract_run_id"]
    )
    assert lesson["lesson_id"] in next_plan["memory_refs"]["lesson_ids"]
    contract = DFINEAdapter().materialize_contract(next_plan, protocol)
    verdict = GateEngine().evaluate(protocol, next_plan, contract, memory=writer)
    assert verdict.status == GateStatus.APPROVED


def test_helper_and_adapter_do_not_invent_modules(tmp_path: Path) -> None:
    report, writer, protocol, previous = _replay_discard(tmp_path)
    stripped = dict(previous)
    stripped["proposed_changes"] = []
    with pytest.raises(InvariantError, match="proposed_changes"):
        build_candidate_next_plan(
            stripped, writer, parent_run_id=report["contract_run_id"]
        )
    next_plan = build_candidate_next_plan(
        previous, writer, parent_run_id=report["contract_run_id"]
    )
    next_plan = dict(next_plan)
    next_plan["proposed_changes"] = []
    with pytest.raises(MaterializeRejected, match="will not invent"):
        DFINEAdapter().materialize_contract(next_plan, protocol)


def test_run_loop_rejects_unresolved_refs_once_memory_exists(tmp_path: Path) -> None:
    report, writer, protocol, previous = _replay_discard(tmp_path)
    next_plan = build_candidate_next_plan(
        previous, writer, parent_run_id=report["contract_run_id"]
    )
    next_plan = dict(next_plan)
    next_plan["memory_refs"] = {
        "lesson_ids": ["LESSON-FAKE-NOT-WRITTEN"],
        "strategy_ids": ["STRATEGY-FAKE-NOT-WRITTEN"],
    }
    follow = run_gated_dfine(
        protocol=protocol,
        plan=next_plan,
        output_dir=tmp_path / "round_n1",
        execute=False,
        memory_writer=writer,
    )
    assert follow["gate"]["status"] == GateStatus.REJECTED
    assert follow["executed"] is False
    assert any("not resolvable" in r for r in follow["gate"]["reasons"])
