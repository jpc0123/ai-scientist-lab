"""v2.6 fusion HOW contrast: after R0 F1, next shot must change HOW. No GPU."""

from __future__ import annotations

from pathlib import Path

from scientist_lab.core.memory_writer import MemoryWriter
from scientist_lab.core.planner import Planner
from scientist_lab.core.schema_registry import SCHEMA_DIR, load_json
from scientist_lab.llm.gateway import ScriptedProvider

EXAMPLES = SCHEMA_DIR / "examples"
RUN_ID = "run_plan_v26_r0_dfine_f1"


def _memory(tmp_path: Path) -> MemoryWriter:
    writer = MemoryWriter(tmp_path / "memory")
    writer.persist_lesson(
        {
            "lesson_id": f"LESSON-{RUN_ID}-001",
            "type": "inconclusive",
            "statement": "R0 F1 APS_lowlight recorded; do not idle the same HOW.",
            "status": "active",
            "evidence": [{"run_id": RUN_ID, "metric": "APS_lowlight", "delta": None}],
            "scope": {"task": "rgbt_detection", "module": "fusion"},
            "confidence": "medium",
            "created_from": [RUN_ID],
            "contradicted_by": [],
            "supersedes": [],
            "expires_when": [],
        }
    )
    writer.persist_strategy(
        {
            "strategy_id": f"STRATEGY-{RUN_ID}-001",
            "action": "keep",
            "target": "fusion",
            "reason_lesson_ids": [f"LESSON-{RUN_ID}-001"],
            "status": "active",
        }
    )
    return writer


def test_rules_planner_contrasts_f1_to_f3_after_bootstrap(tmp_path: Path) -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v26.json")
    previous = load_json(EXAMPLES / "experiment_plan_rgbt_dfine_v26_r0.json")
    packet = Planner(backend="rules").next_plan(
        protocol=protocol,
        memory=_memory(tmp_path),
        previous_plan=previous,
        parent_run_id=RUN_ID,
        last_review_decision="REPLICATE",
    )
    assert packet.plan["how_id"] == "F3"
    assert packet.plan["evaluation"]["seeds"] == [42]
    assert packet.decision_summary["selected_action"] == "contrast_F3"


def test_rules_planner_switches_how_after_negative_delta(tmp_path: Path) -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v26.json")
    previous = load_json(EXAMPLES / "experiment_plan_rgbt_dfine_v26_r0.json")
    previous = dict(previous)
    previous["bootstrap"] = False
    previous["how_id"] = "F3"
    previous["round_index"] = 1
    packet = Planner(backend="rules").next_plan(
        protocol=protocol,
        memory=_memory(tmp_path),
        previous_plan=previous,
        parent_run_id=RUN_ID,
        last_review_decision="KEEP",
        last_primary_delta=-0.03,
    )
    assert packet.plan["how_id"] == "F0"


def test_llm_keeps_selected_f1_and_bumps_seed(tmp_path: Path) -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v26.json")
    previous = load_json(EXAMPLES / "experiment_plan_rgbt_dfine_v26_r0.json")
    raw = """
    {"selected":{"requested_module":"fusion","how_id":"F1",
      "hypothesis":"Replicate R0 F1 on a new seed.",
      "proposed_changes":[{"target":"fusion","summary":"Keep F1.","detail":{"how_id":"F1"}}],
      "expected_effect":{"primary_metric":"APS_lowlight","direction":"stabilize"},
      "budget_class":"formal"},
     "candidates":[{"candidate_id":"cand_f3","requested_module":"fusion","how_id":"F3",
       "reason_not_selected":"deferred"}],
     "invented_operators":[],
     "memory_refs":{"lesson_ids":["LESSON-run_plan_v26_r0_dfine_f1-001"],
       "strategy_ids":["STRATEGY-run_plan_v26_r0_dfine_f1-001"]}}
    """
    packet = Planner(
        backend="llm",
        provider=ScriptedProvider(raw),
        live=False,
    ).next_plan(
        protocol=protocol,
        memory=_memory(tmp_path),
        previous_plan=previous,
        parent_run_id=RUN_ID,
        last_review_decision="REPLICATE",
    )
    assert packet.plan["how_id"] == "F1"
    assert packet.plan["evaluation"]["seeds"] == [43]
    assert packet.decision_summary["selected_action"] != "contrast_F3"
    assert any(
        "rules contrast is not applied" in str(item)
        for item in packet.decision_summary["decision_basis"]
    )


def test_llm_selected_f3_is_not_overridden(tmp_path: Path) -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v26.json")
    previous = load_json(EXAMPLES / "experiment_plan_rgbt_dfine_v26_r0.json")
    raw = """
    {"selected":{"requested_module":"fusion","how_id":"F3",
      "hypothesis":"Probe gated fusion after R0 F1.",
      "proposed_changes":[{"target":"fusion","summary":"Use F3.","detail":{"how_id":"F3"}}],
      "expected_effect":{"primary_metric":"APS_lowlight","direction":"increase"},
      "budget_class":"formal"},
     "candidates":[{"candidate_id":"cand_f1","requested_module":"fusion","how_id":"F1",
       "reason_not_selected":"R0 already ran F1."}],
     "invented_operators":[],
     "memory_refs":{"lesson_ids":["LESSON-run_plan_v26_r0_dfine_f1-001"],
       "strategy_ids":["STRATEGY-run_plan_v26_r0_dfine_f1-001"]}}
    """
    packet = Planner(
        backend="llm",
        provider=ScriptedProvider(raw),
        live=False,
    ).next_plan(
        protocol=protocol,
        memory=_memory(tmp_path),
        previous_plan=previous,
        parent_run_id=RUN_ID,
        last_review_decision="REPLICATE",
    )
    assert packet.plan["how_id"] == "F3"
    assert packet.plan["evaluation"]["seeds"] == [42]
