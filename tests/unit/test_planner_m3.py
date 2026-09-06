"""Freeze Planner (WHAT/WHY, rules-first). No GPU. No Manager loop."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from scientist_lab.adapters.dfine.adapter import DFINEAdapter
from scientist_lab.adapters.dfine.run_loop import run_gated_dfine
from scientist_lab.agents.planner import Planner as RolePlanner
from scientist_lab.core.gate_engine import GateEngine, GateStatus
from scientist_lab.core.invariants import (
    assert_memory_refs_resolvable,
    assert_plan_memory_policy,
)
from scientist_lab.core.memory_writer import MemoryWriter
from scientist_lab.core.planner import PlanRefused, Planner, propose_and_gate_next
from scientist_lab.core.schema_registry import SCHEMA_DIR, load_json, validate_named
from scientist_lab.core.state_machine import OrchestrationAction
from scientist_lab.instrumentation.appender import EventAppender

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


def test_round_ge1_without_memory_is_refused(tmp_path: Path) -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")
    empty = MemoryWriter(tmp_path / "empty_memory")
    with pytest.raises(PlanRefused, match="written lessons or strategies") as exc:
        Planner().next_plan(
            protocol=protocol,
            memory=empty,
            previous_plan=_plan(),
            parent_run_id="run_plan_round1_neck_hr",
        )
    assert exc.value.orchestration_action == OrchestrationAction.NEED_MEMORY.value


def test_planner_emits_resolvable_plan_citing_discard_lesson(tmp_path: Path) -> None:
    report, writer, protocol, previous = _replay_discard(tmp_path)
    assert report["review_decision"] == "DISCARD"
    lesson = next(iter(writer.load_lessons().values()))
    strategy = next(iter(writer.load_strategies().values()))
    assert lesson["type"] == "negative_evidence"
    events = EventAppender(tmp_path / "planner_events.jsonl")
    packet = Planner().next_plan(
        protocol=protocol,
        memory=writer,
        previous_plan=previous,
        parent_run_id=report["contract_run_id"],
        last_review_decision=report["review_decision"],
        events=events,
    )
    plan = packet.plan
    validate_named("experiment_plan", plan)
    assert_plan_memory_policy(plan)
    assert_memory_refs_resolvable(
        plan, lesson_ids=writer.load_lessons(), strategy_ids=writer.load_strategies()
    )
    assert lesson["lesson_id"] in plan["memory_refs"]["lesson_ids"]
    assert strategy["strategy_id"] in plan["memory_refs"]["strategy_ids"]
    assert report["contract_run_id"] in plan["evidence_runs"]
    assert "LESSON-FAKE" not in plan["memory_refs"]["lesson_ids"]
    assert "LESSON-017" not in plan["memory_refs"]["lesson_ids"]
    assert plan["proposed_changes"]
    assert plan["proposed_changes"][0]["target"]
    assert plan["proposed_changes"][0]["summary"]
    assert "FDPN" not in str(plan)
    assert "optimize small object" not in plan["hypothesis"].lower()
    assert packet.decision_summary["selected_action"]
    assert packet.source == "rules_first"
    rows = events.load_all()
    assert any(row.get("event_type") == "plan_proposal" for row in rows)
    proposal = next(row for row in rows if row["event_type"] == "plan_proposal")
    assert proposal["actor_role"] == "planner"
    assert proposal["phase"] == "planning"
    assert proposal["phase"] != "planner"


def test_planner_does_not_emit_fake_ids(tmp_path: Path) -> None:
    report, writer, protocol, previous = _replay_discard(tmp_path)
    packet = Planner().next_plan(
        protocol=protocol,
        memory=writer,
        previous_plan=previous,
        parent_run_id=report["contract_run_id"],
    )
    known = set(writer.load_lessons()) | set(writer.load_strategies())
    refs = packet.plan["memory_refs"]
    for lid in refs["lesson_ids"]:
        assert lid in known
    for sid in refs["strategy_ids"]:
        assert sid in known


def test_adapter_materializes_planner_plan_without_inventing_fdpn(tmp_path: Path) -> None:
    report, writer, protocol, previous = _replay_discard(tmp_path)
    packet = RolePlanner().next_plan(
        protocol=protocol,
        memory=writer,
        previous_plan=previous,
        parent_run_id=report["contract_run_id"],
        last_review_decision="DISCARD",
    )
    contract = DFINEAdapter().materialize_contract(packet.plan, protocol)
    assert contract["hypothesis"] == packet.plan["hypothesis"]
    assert contract["materialization"]["how_only"] is True
    assert "FDPN" not in str(contract)
    how = contract["materialization"]["how"]
    assert packet.plan["modification_scope"] == ["fusion"]
    assert how["primary_module"] == "fusion"
    assert how["fusion_method"] == "early_concat"
    assert how["input_mode"] == "rgbt"
    assert how["invented_operators"] == []
    verdict = GateEngine().evaluate(protocol, packet.plan, contract, memory=writer)
    assert verdict.status == GateStatus.APPROVED


def test_propose_and_gate_next_after_replay(tmp_path: Path) -> None:
    report, writer, protocol, previous = _replay_discard(tmp_path)
    events = EventAppender(tmp_path / "chain_events.jsonl")
    result = propose_and_gate_next(
        protocol=protocol,
        memory=writer,
        previous_plan=previous,
        parent_run_id=report["contract_run_id"],
        last_review_decision="DISCARD",
        events=events,
    )
    assert result["gate"]["status"] == GateStatus.APPROVED
    plan_id = result["plan"]["plan_id"]
    lesson = next(iter(writer.load_lessons().values()))
    kinds = {(e["from"], e["to"], e["type"]) for e in writer.load_trace()}
    assert (lesson["lesson_id"], plan_id, "used_by") in kinds
    assert (report["contract_run_id"], plan_id, "supported_by") in kinds


def test_keep_memory_also_yields_legal_plan(tmp_path: Path) -> None:
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
    packet = Planner().next_plan(
        protocol=protocol,
        memory=writer,
        previous_plan=previous,
        parent_run_id=report["contract_run_id"],
        last_review_decision="KEEP",
    )
    validate_named("experiment_plan", packet.plan)
    assert_memory_refs_resolvable(
        packet.plan,
        lesson_ids=writer.load_lessons(),
        strategy_ids=writer.load_strategies(),
    )
    DFINEAdapter().materialize_contract(packet.plan, protocol)


def test_replicate_same_module_writes_next_seed(tmp_path: Path) -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")
    best = load_json(EXAMPLES / "recovered_k2c44_best_metrics.json")
    out = tmp_path / "replicate"
    _install_recovered(out, "recovered_k2c44_best_metrics.json")
    previous = _plan()
    report = run_gated_dfine(
        protocol=protocol,
        plan=previous,
        output_dir=out,
        execute=False,
        baseline_metrics={"APS": best["APS"]},
    )
    writer = MemoryWriter(out / "memory")
    packet = Planner().next_plan(
        protocol=protocol,
        memory=writer,
        previous_plan=previous,
        parent_run_id=report["contract_run_id"],
        last_review_decision="REPLICATE",
    )
    assert previous["evaluation"]["seeds"] == [42]
    assert packet.plan["evaluation"]["seeds"] == [43]
    contract = DFINEAdapter().materialize_contract(packet.plan, protocol)
    assert contract["seed"] == 43


def test_discard_switches_off_failed_module(tmp_path: Path) -> None:
    report, writer, protocol, previous = _replay_discard(tmp_path)
    packet = Planner().next_plan(
        protocol=protocol,
        memory=writer,
        previous_plan=previous,
        parent_run_id=report["contract_run_id"],
        last_review_decision="DISCARD",
    )
    assert packet.plan["modification_scope"] != ["neck"]
    assert packet.plan["proposed_changes"][0]["target"] != "neck"
    assert packet.plan["proposed_changes"][0]["target"] in protocol["editable_scope"]
