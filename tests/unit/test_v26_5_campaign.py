"""V26.5 campaign wiring. No GPU. No fifth Agent."""

from __future__ import annotations

import json
from pathlib import Path

from scientist_lab.cli import main
from scientist_lab.core.manager import Manager
from scientist_lab.core.schema_registry import SCHEMA_DIR, load_json
from scientist_lab.core.state_machine import OrchestrationAction, RunState
from scientist_lab.llm.planner_contract import PlannerContractInput, parse_planner_completion
from scientist_lab.tasks.rgbt_detection.v26_r0 import (
    R0_LESSON_ID,
    seed_r0_campaign_memory,
)

EXAMPLES = SCHEMA_DIR / "examples"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _stub_runner(calls: list):
    def runner(contract, output_dir):
        calls.append({"run_id": contract.get("run_id"), "output_dir": str(output_dir)})
        dest = Path(output_dir)
        dest.mkdir(parents=True, exist_ok=True)
        _write_json(dest / "metrics.json", {"APS": 0.01, "mAP50_95": 0.02})
        _write_json(dest / "checkpoint_selection.json", {"best_epoch": 1})
        return {"status": "completed"}

    return runner


def test_formal_without_confirm_stays_gated(tmp_path: Path) -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v26.json")
    plan = load_json(EXAMPLES / "experiment_plan_rgbt_dfine_v26_r0.json")
    _write_json(tmp_path / "plan.json", plan)
    calls: list = []
    mgr = Manager(
        tmp_path,
        protocol=protocol,
        execute=True,
        live_runner=_stub_runner(calls),
        max_extra_rounds=0,
        confirm_human_gate=False,
    )
    mgr.initialize_run(round_index=0, plan_id=plan["plan_id"])
    steps = mgr.run_until(max_steps=12)
    assert calls == []
    assert steps[-1].idle is True
    assert steps[-1].state.run_state == RunState.GATED
    assert steps[-1].action == OrchestrationAction.NEED_HUMAN.value


def test_confirm_human_gate_lets_formal_execute(tmp_path: Path) -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v26.json")
    plan = load_json(EXAMPLES / "experiment_plan_rgbt_dfine_v26_r0.json")
    _write_json(tmp_path / "plan.json", plan)
    calls: list = []
    mgr = Manager(
        tmp_path,
        protocol=protocol,
        execute=True,
        live_runner=_stub_runner(calls),
        max_extra_rounds=0,
        confirm_human_gate=True,
    )
    mgr.initialize_run(round_index=0, plan_id=plan["plan_id"])
    steps = mgr.run_until(max_steps=20)
    assert calls, "campaign Human Gate must reach live_runner"
    gate_steps = [s for s in steps if s.action == OrchestrationAction.NEED_GATE.value]
    assert gate_steps
    assert gate_steps[-1].report.get("campaign_human_gate") is True
    assert gate_steps[-1].state.run_state == RunState.APPROVED


def test_cli_parses_confirm_human_gate(monkeypatch, tmp_path: Path) -> None:
    seen: dict = {}

    def fake(*_a, **kwargs):
        seen.update(kwargs)
        return {"exit_code": 0, "live_ready": True}

    monkeypatch.setattr("scientist_lab.core.manager_cli.run_manager_from_files", fake)
    protocol = tmp_path / "protocol.json"
    plan = tmp_path / "plan.json"
    protocol.write_text("{}", encoding="utf-8")
    plan.write_text("{}", encoding="utf-8")
    code = main(
        [
            "manager-run",
            "--protocol",
            str(protocol),
            "--plan",
            str(plan),
            "--output-dir",
            str(tmp_path / "out"),
            "--confirm-human-gate",
            "--llm-live",
        ]
    )
    assert code == 0
    assert seen.get("confirm_human_gate") is True
    assert seen.get("llm_live") is True


def test_seed_r0_memory_is_citable(tmp_path: Path) -> None:
    seeded = seed_r0_campaign_memory(tmp_path / "memory", run_id="exec_31eff20c4e0c", aps_lowlight=0.0045926865160844455)
    assert seeded["lesson_id"] == R0_LESSON_ID
    payload = json.loads((tmp_path / "memory" / "research_memory.json").read_text(encoding="utf-8"))
    lesson = payload["lessons"][R0_LESSON_ID]
    assert "0.0045926865160844455" in lesson["statement"]
    assert lesson["type"] == "process"


def test_parse_planner_maps_registered_how_id() -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v26.json")
    previous = load_json(EXAMPLES / "experiment_plan_rgbt_dfine_v26_r0.json")
    payload = PlannerContractInput(
        goal=dict(protocol.get("goal") or {}),
        protocol=protocol,
        previous_plan=previous,
        evidence={},
        rubric={},
        memory={"lesson_ids": [R0_LESSON_ID], "strategy_ids": []},
        memory_refs={"lesson_ids": [R0_LESSON_ID], "strategy_ids": []},
        budget={"budget_class": "formal"},
        last_review_decision=None,
        parent_run_id="exec_31eff20c4e0c",
    )
    raw = json.dumps(
        {
            "selected": {
                "requested_module": "fusion",
                "how_id": "F3",
                "hypothesis": "gated_multiscale may help low-light small objects vs R0 F1.",
                "observation": "R0 APS_lowlight is low; LiteratureEvidence is not ExperimentEvidence.",
                "proposed_changes": [
                    {
                        "target": "fusion",
                        "summary": "Switch registered HOW to F3 gated_multiscale.",
                        "detail": {"how_id": "F3"},
                    }
                ],
                "expected_effect": {"primary_metric": "APS_lowlight", "direction": "increase"},
                "budget_class": "formal",
            },
            "candidates": [],
            "invented_operators": [],
            "memory_refs": {"lesson_ids": [R0_LESSON_ID], "strategy_ids": []},
        }
    )
    mapped = parse_planner_completion(raw, payload, known_lesson_ids=[R0_LESSON_ID])
    assert mapped["how_id"] == "F3"
    assert mapped["budget_class"] == "formal"
    assert mapped["proposed_changes"][0]["detail"]["how_id"] == "F3"


def test_parse_planner_maps_f0_control() -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v26.json")
    previous = load_json(EXAMPLES / "experiment_plan_rgbt_dfine_v26_r0.json")
    previous = dict(previous)
    previous["how_id"] = "F3"
    previous["round_index"] = 1
    payload = PlannerContractInput(
        goal=dict(protocol.get("goal") or {}),
        protocol=protocol,
        previous_plan=previous,
        evidence={},
        rubric={},
        memory={"lesson_ids": [R0_LESSON_ID], "strategy_ids": []},
        memory_refs={"lesson_ids": [R0_LESSON_ID], "strategy_ids": []},
        budget={"budget_class": "formal"},
        last_review_decision="KEEP",
        parent_run_id="run_plan_round1_from_exec_31eff20c4e0c",
    )
    raw = json.dumps(
        {
            "selected": {
                "requested_module": "fusion",
                "how_id": "F0",
                "hypothesis": "RGB-only F0 tests whether thermal is necessary after F3 KEEP.",
                "proposed_changes": [
                    {
                        "target": "fusion",
                        "summary": "RGB-only control HOW F0. Not A4.",
                        "detail": {"how_id": "F0", "neck_type": "standard"},
                    }
                ],
                "expected_effect": {"primary_metric": "APS_lowlight", "direction": "decrease"},
                "budget_class": "formal",
            },
            "candidates": [
                {
                    "candidate_id": "cand_f3",
                    "requested_module": "fusion",
                    "how_id": "F3",
                    "reason_not_selected": "F3 already KEEP on seed 42; same-seed repeat is not 2-seed G2.",
                }
            ],
            "invented_operators": [],
            "memory_refs": {"lesson_ids": [R0_LESSON_ID], "strategy_ids": []},
        }
    )
    mapped = parse_planner_completion(raw, payload, known_lesson_ids=[R0_LESSON_ID])
    assert mapped["how_id"] == "F0"
    assert mapped["budget_class"] == "formal"


def test_parse_planner_coerces_relative_direction() -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v26.json")
    previous = load_json(EXAMPLES / "experiment_plan_rgbt_dfine_v26_r0.json")
    payload = PlannerContractInput(
        goal=dict(protocol.get("goal") or {}),
        protocol=protocol,
        previous_plan=previous,
        evidence={},
        rubric={},
        memory={"lesson_ids": [R0_LESSON_ID], "strategy_ids": []},
        memory_refs={"lesson_ids": [R0_LESSON_ID], "strategy_ids": []},
        budget={"budget_class": "formal"},
        last_review_decision="KEEP",
        parent_run_id="run_plan_round1_from_exec_31eff20c4e0c",
    )
    raw = json.dumps(
        {
            "selected": {
                "requested_module": "fusion",
                "how_id": "F0",
                "hypothesis": "F0 RGB-only control should fall relative to KEEP F3 if thermal helps.",
                "proposed_changes": [
                    {
                        "target": "fusion",
                        "summary": "RGB-only control HOW F0.",
                        "detail": {"how_id": "F0"},
                    }
                ],
                "expected_effect": {
                    "primary_metric": "APS_lowlight",
                    "direction": "decrease_relative_to_F3",
                },
                "budget_class": "formal",
            },
            "candidates": [],
            "invented_operators": [],
            "memory_refs": {"lesson_ids": [R0_LESSON_ID], "strategy_ids": []},
        }
    )
    mapped = parse_planner_completion(raw, payload, known_lesson_ids=[R0_LESSON_ID])
    assert mapped["expected_effect"]["direction"] == "decrease"
