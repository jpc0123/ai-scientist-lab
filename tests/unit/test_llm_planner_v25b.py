"""v2.5-B historical REPLAY exam on frozen M4 rounds3 Round1 DISCARD. No GPU."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from scientist_lab.adapters.dfine.adapter import DFINEAdapter
from scientist_lab.cli import main
from scientist_lab.core.gate_engine import GateEngine, GateStatus
from scientist_lab.core.memory_writer import MemoryWriter
from scientist_lab.core.planner import PlanRefused, Planner
from scientist_lab.core.schema_registry import load_json, validate_named
from scientist_lab.llm.gateway import ScriptedProvider
from scientist_lab.llm.plan_replay import run_llm_plan_replay

FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "llm_plan_replay" / "m4_rounds3_discard"
)
REAL_LESSON = "LESSON-run_plan_round1_neck_hr-001"
REAL_STRATEGY = "STRATEGY-run_plan_round1_neck_hr-001"


def _copy_fixture(tmp_path: Path) -> Path:
    dest = tmp_path / "m4_rounds3_discard"
    shutil.copytree(FIXTURE, dest)
    return dest


def _selected_json(*, module: str = "fusion", candidates: list | None = None, **extra) -> str:
    selected = {
        "candidate_id": "cand_selected",
        "requested_module": module,
        "observation": "DISCARD on neck; switch to an allowed HOW module.",
        "hypothesis": f"A {module} change probes APS without repeating the discarded neck path.",
        "proposed_changes": [
            {"target": module, "summary": f"Probe {module} with existing Adapter HOW."}
        ],
        "expected_effect": {"primary_metric": "APS", "direction": "increase"},
        "budget_class": "probe",
        "selected_action": f"switch_to_{module}",
    }
    selected.update(extra.pop("selected_updates", {}))
    body = {
        "selected": selected,
        "candidates": candidates
        if candidates is not None
        else [
            {
                "candidate_id": "cand_a",
                "requested_module": "neck",
                "reason_not_selected": "Repeats DISCARD module.",
            },
            {
                "candidate_id": "cand_b",
                "requested_module": "hyperparameter",
                "reason_not_selected": "No Adapter HOW.",
            },
        ],
        "invented_operators": extra.pop("invented_operators", []),
        "memory_refs": extra.pop(
            "memory_refs",
            {"lesson_ids": [REAL_LESSON], "strategy_ids": [REAL_STRATEGY]},
        ),
    }
    body.update(extra)
    return json.dumps(body, ensure_ascii=False)


def test_exam_pack_is_real_round1_discard() -> None:
    source = load_json(FIXTURE / "SOURCE.json")
    review = load_json(FIXTURE / "review.json")
    memory = MemoryWriter(FIXTURE / "memory")
    previous = load_json(FIXTURE / "previous_plan.json")
    assert source["origin_run_dir"] == ".run/real_fast_eval_m4_rounds3"
    assert source["exam_point"] == "after_round1_DISCARD"
    assert review["review_decision"] == "DISCARD"
    assert REAL_LESSON in memory.load_lessons()
    assert memory.load_lessons()[REAL_LESSON]["type"] == "negative_evidence"
    assert memory.load_lessons()[REAL_LESSON]["scope"]["module"] == "neck"
    assert previous["modification_scope"] == ["neck"]
    assert previous["plan_id"] == "plan_round1_neck_hr"


def test_mock_understands_discard_and_cites_real_lesson(tmp_path: Path) -> None:
    dest = _copy_fixture(tmp_path)
    lessons_before = set(MemoryWriter(dest / "memory").load_lessons())
    report = run_llm_plan_replay(dest, live=False, provider="mock", ab=True)
    assert report["ok"] is True
    assert report["gpu"] is False
    assert report["execute"] is False
    assert report["discarded_module"] == "neck"
    plan = report["llm_plan"]
    validate_named("experiment_plan", plan)
    assert plan["modification_scope"] != ["neck"]
    assert plan["modification_scope"] == ["fusion"]
    assert REAL_LESSON in plan["memory_refs"]["lesson_ids"]
    assert REAL_STRATEGY in plan["memory_refs"]["strategy_ids"]
    blob = f"{plan['observation']} {plan['hypothesis']}".lower()
    assert "discard" in blob or "negative_evidence" in blob
    assert len(plan.get("candidate_experiments") or []) >= 2
    discarded_attach = [
        row
        for row in plan["candidate_experiments"]
        if str(row.get("requested_module") or "") == "neck"
    ]
    assert discarded_attach
    assert any("reason_not_selected" in row for row in discarded_attach)
    assert report["gate"]["status"] == GateStatus.APPROVED
    how = report["contract"]["how"]
    assert how["primary_module"] == "fusion"
    assert how["fusion_method"] == "early_concat"
    assert how["invented_operators"] == []
    assert report["memory_unchanged"] is True
    assert report["review_decision_unchanged"] is True
    assert plan["budget_class"] != "formal"
    assert set(MemoryWriter(dest / "memory").load_lessons()) == lessons_before
    assert load_json(dest / "review.json")["review_decision"] == "DISCARD"


def test_ab_rules_vs_llm(tmp_path: Path) -> None:
    dest = _copy_fixture(tmp_path)
    report = run_llm_plan_replay(dest, live=False, provider="mock", ab=True)
    ab = report["ab"]
    assert ab["compared"] is True
    assert ab["rules_selected_module"] == "fusion"
    assert ab["llm_selected_module"] == "fusion"
    assert ab["same_selected_module"] is True
    assert ab["rules_has_candidate_experiments"] is False
    assert ab["llm_candidate_count"] >= 2
    assert ab["rules_is_template_switch"] is True
    assert ab["llm_has_selection_rationale"] is True
    llm_basis = " ".join(ab["llm_decision_summary"].get("decision_basis") or [])
    rules_basis = " ".join(ab["rules_decision_summary"].get("decision_basis") or [])
    assert "llm_planner_contract" in llm_basis
    assert "rules-first Planner" in rules_basis
    assert ab["historical_rules_dead_cycle"]["round3_selected"] == "neck"
    assert (dest / ".llm_plan_replay" / "replay_report.json").is_file()
    saved = load_json(dest / ".llm_plan_replay" / "replay_report.json")
    assert "rules_plan" in saved and "llm_plan" in saved
    assert saved["live"] is False


def test_repeat_discarded_neck_fail_closed(tmp_path: Path) -> None:
    dest = _copy_fixture(tmp_path)
    bundle_memory = MemoryWriter(dest / "memory")
    previous = load_json(dest / "previous_plan.json")
    protocol = load_json(dest / "protocol.json")
    planner = Planner(
        backend="llm",
        provider=ScriptedProvider(_selected_json(module="neck")),
    )
    with pytest.raises(PlanRefused, match="repeats DISCARD"):
        planner.next_plan(
            protocol=protocol,
            memory=bundle_memory,
            previous_plan=previous,
            parent_run_id="run_plan_round1_neck_hr",
            last_review_decision="DISCARD",
        )


def test_fdpn_and_over_scope_fail_closed(tmp_path: Path) -> None:
    dest = _copy_fixture(tmp_path)
    memory = MemoryWriter(dest / "memory")
    previous = load_json(dest / "previous_plan.json")
    protocol = load_json(dest / "protocol.json")
    fdpn = Planner(
        backend="llm",
        provider=ScriptedProvider(_selected_json(invented_operators=["FDPN"])),
    )
    with pytest.raises(PlanRefused, match="invented operators"):
        fdpn.next_plan(
            protocol=protocol,
            memory=memory,
            previous_plan=previous,
            parent_run_id="run_plan_round1_neck_hr",
            last_review_decision="DISCARD",
        )
    scoped = Planner(
        backend="llm",
        provider=ScriptedProvider(_selected_json(module="evaluator")),
    )
    with pytest.raises(PlanRefused, match="outside editable_scope"):
        scoped.next_plan(
            protocol=protocol,
            memory=memory,
            previous_plan=previous,
            parent_run_id="run_plan_round1_neck_hr",
            last_review_decision="DISCARD",
        )


def test_selected_plan_gates_with_existing_how(tmp_path: Path) -> None:
    dest = _copy_fixture(tmp_path)
    report = run_llm_plan_replay(dest, live=False, provider="mock")
    plan = report["llm_plan"]
    protocol = load_json(dest / "protocol.json")
    memory = MemoryWriter(dest / ".llm_plan_replay" / "memory_llm")
    contract = DFINEAdapter().materialize_contract(plan, protocol)
    verdict = GateEngine().evaluate(protocol, plan, contract, memory=memory)
    assert verdict.status == GateStatus.APPROVED
    assert contract["allowed_changes"] == ["fusion"]


def test_cli_ab_replay_no_gpu(tmp_path: Path) -> None:
    dest = _copy_fixture(tmp_path)
    code = main(["llm-plan-replay", "--ab", "--run-dir", str(dest)])
    assert code == 0
    report = load_json(dest / ".llm_plan_replay" / "replay_report.json")
    assert report["ok"] is True
    assert report["ab"]["llm_candidate_count"] >= 2
    assert report["gpu"] is False


def test_live_without_key_does_not_fake_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = _copy_fixture(tmp_path)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    code = main(["llm-plan-replay", "--run-dir", str(dest), "--live"])
    assert code == 1
    report = load_json(dest / ".llm_plan_replay" / "replay_report.json")
    assert report["ok"] is False
    assert report["fail_closed"] is True
    assert report["live"] is True
    assert report.get("llm_plan") in (None, {})
    assert report["gate"]["status"] != "APPROVED"
