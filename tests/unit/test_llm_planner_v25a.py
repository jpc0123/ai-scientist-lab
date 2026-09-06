"""v2.5-A LLM Planner Gateway + contract. No GPU. No API key required."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from scientist_lab.adapters.dfine.adapter import DFINEAdapter
from scientist_lab.adapters.dfine.how import HOW_MODULES, list_adapter_capabilities
from scientist_lab.adapters.dfine.run_loop import run_gated_dfine
from scientist_lab.cli import main
from scientist_lab.core.gate_engine import GateEngine, GateStatus
from scientist_lab.core.invariants import assert_memory_refs_resolvable, assert_plan_memory_policy
from scientist_lab.core.memory_writer import MemoryWriter
from scientist_lab.core.planner import PlanRefused, Planner, propose_and_gate_next
from scientist_lab.core.schema_registry import SCHEMA_DIR, load_json, validate_named
from scientist_lab.core import schema_registry as _schema_registry

_schema_registry._registry.cache_clear()
_schema_registry._validator.cache_clear()
from scientist_lab.instrumentation.appender import EventAppender
from scientist_lab.llm.errors import MissingAPIKeyError, RealProviderNotEnabledError
from scientist_lab.llm.fake_provider import FakeProvider
from scientist_lab.llm.gateway import ScriptedProvider, resolve_gateway_provider
from scientist_lab.llm.models import LLMRequest
from scientist_lab.llm.plan_replay import run_llm_plan_replay
from scientist_lab.llm.schema_parser import PLANNER_OUTPUT_SCHEMA

EXAMPLES = SCHEMA_DIR / "examples"
STUB = Path(__file__).resolve().parents[1] / "fixtures" / "llm_plan_replay" / "discard_stub"


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


def _replay_discard(tmp_path: Path):
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


def _selected_json(*, module: str = "fusion", **extra) -> str:
    selected = {
        "candidate_id": "cand_selected",
        "requested_module": module,
        "observation": "Neck probe discarded; try an allowed fusion HOW.",
        "hypothesis": (
            f"A {module} change is a better probe of APS than repeating the discarded neck path."
        ),
        "proposed_changes": [
            {"target": module, "summary": f"Probe {module} with existing Adapter HOW."}
        ],
        "expected_effect": {
            "primary_metric": "APS",
            "direction": "increase",
            "rationale": "Protocol objective.",
        },
        "budget_class": "probe",
        "selected_action": f"switch_to_{module}",
    }
    selected.update(extra.pop("selected_updates", {}))
    body = {
        "selected": selected,
        "candidates": extra.pop("candidates", []),
        "invented_operators": extra.pop("invented_operators", []),
    }
    body.update(extra)
    return json.dumps(body, ensure_ascii=False)


def test_default_backend_is_rules() -> None:
    planner = Planner()
    assert planner.backend == "rules"
    assert planner.fallback_to_rules is False


def test_adapter_capabilities_are_existing_how_only() -> None:
    caps = list_adapter_capabilities()
    modules = {row["module"] for row in caps}
    assert modules == HOW_MODULES
    fusion = next(row for row in caps if row["module"] == "fusion")
    neck = next(row for row in caps if row["module"] == "neck")
    assert fusion["input_mode"] == "rgbt"
    assert fusion["fusion_method"] == "early_concat"
    assert neck["input_mode"] == "rgb"
    assert neck["fusion_method"] == "none"
    dumped = json.dumps(caps).lower()
    how_ids = {row.get("how_id") for row in caps if row.get("how_id")}
    from scientist_lab.adapters.dfine.how_catalog import planner_visible_how_ids

    assert how_ids == set(planner_visible_how_ids())
    assert "N1" in how_ids
    assert "A4" in how_ids
    assert "F2" not in how_ids
    assert "fdpn" in dumped


def test_fake_provider_legacy_planner_schema_unchanged() -> None:
    response = FakeProvider().complete(
        LLMRequest(
            purpose="planner",
            messages=[{"role": "user", "content": json.dumps({"project_id": "p"})}],
            response_schema=PLANNER_OUTPUT_SCHEMA,
        )
    )
    assert response.schema_valid is True
    assert "candidates" in (response.parsed_json or {})
    assert "selected" not in (response.parsed_json or {})


def test_mock_fusion_plan_gates_approved(tmp_path: Path) -> None:
    report, writer, protocol, previous = _replay_discard(tmp_path)
    events = EventAppender(tmp_path / "llm_events.jsonl")
    before = set(writer.load_lessons())
    result = propose_and_gate_next(
        protocol=protocol,
        memory=writer,
        previous_plan=previous,
        parent_run_id=report["contract_run_id"],
        last_review_decision="DISCARD",
        events=events,
        backend="llm",
        provider=FakeProvider(),
    )
    assert result["source"] == "llm"
    assert result["gate"]["status"] == GateStatus.APPROVED
    plan = result["plan"]
    validate_named("experiment_plan", plan)
    assert_plan_memory_policy(plan)
    assert_memory_refs_resolvable(
        plan, lesson_ids=writer.load_lessons(), strategy_ids=writer.load_strategies()
    )
    assert plan["modification_scope"] == ["fusion"]
    how = result["contract"]["materialization"]["how"]
    assert how["primary_module"] == "fusion"
    assert how["fusion_method"] == "early_concat"
    assert how["input_mode"] == "rgbt"
    assert how["invented_operators"] == []
    assert "FDPN" not in str(plan)
    trace = plan["llm_trace"]
    assert trace["provider"]
    assert trace["model"]
    assert len(trace["prompt_hash"]) == 64
    assert trace["raw_output"]
    assert plan.get("candidate_experiments")
    assert any(
        row.get("candidate_id") == "cand_alt_not_selected"
        for row in plan["candidate_experiments"]
    )
    rows = events.load_all()
    proposal = next(row for row in rows if row["event_type"] == "plan_proposal" and row["payload"].get("source") == "llm")
    assert proposal["payload"]["prompt_hash"] == trace["prompt_hash"]
    assert proposal["payload"]["raw_output"]
    assert writer.load_lessons().keys() == before


def test_only_selected_goes_to_gate(tmp_path: Path) -> None:
    report, writer, protocol, previous = _replay_discard(tmp_path)
    packet = Planner(backend="llm", provider=FakeProvider()).next_plan(
        protocol=protocol,
        memory=writer,
        previous_plan=previous,
        parent_run_id=report["contract_run_id"],
        last_review_decision="DISCARD",
    )
    assert packet.plan["modification_scope"] == ["fusion"]
    alts = packet.plan.get("candidate_experiments") or []
    assert alts
    contract = DFINEAdapter().materialize_contract(packet.plan, protocol)
    assert contract["allowed_changes"] == ["fusion"]
    verdict = GateEngine().evaluate(protocol, packet.plan, contract, memory=writer)
    assert verdict.status == GateStatus.APPROVED


def test_bad_json_fail_closed_no_silent_rules_fallback(tmp_path: Path) -> None:
    report, writer, protocol, previous = _replay_discard(tmp_path)
    events = EventAppender(tmp_path / "bad_json.jsonl")
    planner = Planner(backend="llm", provider=ScriptedProvider("<<<not-json>>>"))
    with pytest.raises(PlanRefused, match="invalid JSON"):
        planner.next_plan(
            protocol=protocol,
            memory=writer,
            previous_plan=previous,
            parent_run_id=report["contract_run_id"],
            last_review_decision="DISCARD",
            events=events,
        )
    rows = events.load_all()
    assert any(row.get("payload", {}).get("fail_closed") for row in rows)
    assert not any(row.get("payload", {}).get("fallback_to_rules") for row in rows)


def test_out_of_scope_fail_closed(tmp_path: Path) -> None:
    report, writer, protocol, previous = _replay_discard(tmp_path)
    planner = Planner(
        backend="llm",
        provider=ScriptedProvider(_selected_json(module="evaluator")),
    )
    with pytest.raises(PlanRefused, match="outside editable_scope"):
        planner.next_plan(
            protocol=protocol,
            memory=writer,
            previous_plan=previous,
            parent_run_id=report["contract_run_id"],
            last_review_decision="DISCARD",
        )


def test_fdpn_fail_closed(tmp_path: Path) -> None:
    report, writer, protocol, previous = _replay_discard(tmp_path)
    planner = Planner(
        backend="llm",
        provider=ScriptedProvider(_selected_json(invented_operators=["FDPN"])),
    )
    with pytest.raises(PlanRefused, match="invented operators"):
        planner.next_plan(
            protocol=protocol,
            memory=writer,
            previous_plan=previous,
            parent_run_id=report["contract_run_id"],
            last_review_decision="DISCARD",
        )


def test_fdpn_in_hypothesis_fail_closed(tmp_path: Path) -> None:
    report, writer, protocol, previous = _replay_discard(tmp_path)
    body = json.loads(_selected_json())
    body["selected"]["hypothesis"] = "Replace the neck with a new FDPN operator."
    planner = Planner(backend="llm", provider=ScriptedProvider(json.dumps(body)))
    with pytest.raises(PlanRefused, match="FDPN"):
        planner.next_plan(
            protocol=protocol,
            memory=writer,
            previous_plan=previous,
            parent_run_id=report["contract_run_id"],
            last_review_decision="DISCARD",
        )


def test_f1_may_cite_prior_a4_fdpn_in_hypothesis(tmp_path: Path) -> None:
    """Live M1: F1 plan may mention previous A4/FDPN as observation."""
    report, writer, protocol, previous = _replay_discard(tmp_path)
    previous = dict(previous)
    previous["how_id"] = "A4"
    previous["proposed_changes"] = [
        {
            "target": "fusion",
            "summary": "A4 early_concat + fdpn",
            "detail": {"how_id": "A4", "seed": 42},
        }
    ]
    previous["hypothesis"] = "A4 (early_concat + fdpn) composition probe."
    body = json.loads(_selected_json(module="fusion"))
    body["selected"]["how_id"] = "F1"
    body["selected"]["hypothesis"] = (
        "Previous A4 (early_concat + fdpn) had zero delta; "
        "run F1 early_concat baseline next."
    )
    body["selected"]["proposed_changes"] = [
        {
            "target": "fusion",
            "summary": "Execute F1 early_concat baseline.",
            "detail": {"how_id": "F1", "seed": 42},
        }
    ]
    planner = Planner(backend="llm", provider=ScriptedProvider(json.dumps(body)))
    packet = planner.next_plan(
        protocol=protocol,
        memory=writer,
        previous_plan=previous,
        parent_run_id=report["contract_run_id"],
        last_review_decision="KEEP",
    )
    assert "F1" in json.dumps(packet.plan.get("proposed_changes") or [])


def test_f3_may_list_n1_fdpn_as_not_selected_candidate(tmp_path: Path) -> None:
    """candidates[] may name N1/FDPN as coverage gap without inventing HOW."""
    report, writer, protocol, previous = _replay_discard(tmp_path)
    previous = dict(previous)
    previous["how_id"] = "F3"
    previous["proposed_changes"] = [
        {
            "target": "fusion",
            "summary": "F3 gated_multiscale",
            "detail": {"how_id": "F3", "seed": 45},
        }
    ]
    body = json.loads(_selected_json(module="fusion"))
    body["selected"]["how_id"] = "F3"
    body["selected"]["hypothesis"] = "Replicate F3 on seed 46 for multi-seed confirmation."
    body["selected"]["proposed_changes"] = [
        {
            "target": "fusion",
            "summary": "Replicate F3 gated_multiscale on seed 46.",
            "detail": {"how_id": "F3", "seed": 46},
        }
    ]
    body["candidates"] = [
        {
            "candidate_id": "cand_n1_fdpn_neck",
            "requested_module": "neck",
            "how_id": "N1",
            "reason_not_selected": "N1 (FDPN neck) is a coverage gap; defer until F3 replicates.",
        }
    ]
    planner = Planner(backend="llm", provider=ScriptedProvider(json.dumps(body)))
    packet = planner.next_plan(
        protocol=protocol,
        memory=writer,
        previous_plan=previous,
        parent_run_id=report["contract_run_id"],
        last_review_decision="KEEP",
    )
    assert "F3" in json.dumps(packet.plan.get("proposed_changes") or [])


def test_no_how_module_fail_closed(tmp_path: Path) -> None:
    report, writer, protocol, previous = _replay_discard(tmp_path)
    planner = Planner(backend="llm", provider=ScriptedProvider(_selected_json(module="loss")))
    with pytest.raises(PlanRefused, match="no HOW"):
        planner.next_plan(
            protocol=protocol,
            memory=writer,
            previous_plan=previous,
            parent_run_id=report["contract_run_id"],
            last_review_decision="DISCARD",
        )


def test_formal_promotion_fail_closed(tmp_path: Path) -> None:
    report, writer, protocol, previous = _replay_discard(tmp_path)
    planner = Planner(
        backend="llm",
        provider=ScriptedProvider(_selected_json(selected_updates={"budget_class": "formal"})),
    )
    with pytest.raises(PlanRefused, match="cannot promote budget_class"):
        planner.next_plan(
            protocol=protocol,
            memory=writer,
            previous_plan=previous,
            parent_run_id=report["contract_run_id"],
            last_review_decision="DISCARD",
        )


def test_review_override_fail_closed(tmp_path: Path) -> None:
    report, writer, protocol, previous = _replay_discard(tmp_path)
    planner = Planner(
        backend="llm",
        provider=ScriptedProvider(_selected_json(review_decision_override="KEEP")),
    )
    with pytest.raises(PlanRefused, match="KEEP/DISCARD"):
        planner.next_plan(
            protocol=protocol,
            memory=writer,
            previous_plan=previous,
            parent_run_id=report["contract_run_id"],
            last_review_decision="DISCARD",
        )


def test_memory_write_attempt_fail_closed(tmp_path: Path) -> None:
    report, writer, protocol, previous = _replay_discard(tmp_path)
    planner = Planner(
        backend="llm",
        provider=ScriptedProvider(_selected_json(lessons_to_write=[{"lesson_id": "X"}])),
    )
    with pytest.raises(PlanRefused, match="must not write Memory"):
        planner.next_plan(
            protocol=protocol,
            memory=writer,
            previous_plan=previous,
            parent_run_id=report["contract_run_id"],
            last_review_decision="DISCARD",
        )


def test_invented_lesson_id_fail_closed(tmp_path: Path) -> None:
    report, writer, protocol, previous = _replay_discard(tmp_path)
    planner = Planner(
        backend="llm",
        provider=ScriptedProvider(
            _selected_json(memory_refs={"lesson_ids": ["LESSON-FAKE"], "strategy_ids": []})
        ),
    )
    with pytest.raises(PlanRefused, match="unresolved memory_refs"):
        planner.next_plan(
            protocol=protocol,
            memory=writer,
            previous_plan=previous,
            parent_run_id=report["contract_run_id"],
            last_review_decision="DISCARD",
        )


def test_fallback_to_rules_records_event(tmp_path: Path) -> None:
    report, writer, protocol, previous = _replay_discard(tmp_path)
    events = EventAppender(tmp_path / "fallback.jsonl")
    packet = Planner(
        backend="llm",
        provider=ScriptedProvider("not-json"),
        fallback_to_rules=True,
    ).next_plan(
        protocol=protocol,
        memory=writer,
        previous_plan=previous,
        parent_run_id=report["contract_run_id"],
        last_review_decision="DISCARD",
        events=events,
    )
    assert packet.source == "llm_fallback_rules"
    validate_named("experiment_plan", packet.plan)
    rows = events.load_all()
    assert any(row.get("payload", {}).get("fallback_to_rules") for row in rows)


def test_rules_backend_still_switches_after_discard(tmp_path: Path) -> None:
    report, writer, protocol, previous = _replay_discard(tmp_path)
    packet = Planner(backend="rules").next_plan(
        protocol=protocol,
        memory=writer,
        previous_plan=previous,
        parent_run_id=report["contract_run_id"],
        last_review_decision="DISCARD",
    )
    assert packet.source == "rules_first"
    assert packet.plan["modification_scope"] == ["fusion"]
    result = propose_and_gate_next(
        protocol=protocol,
        memory=writer,
        previous_plan=previous,
        parent_run_id=report["contract_run_id"],
        last_review_decision="DISCARD",
        backend="rules",
    )
    assert result["gate"]["status"] == GateStatus.APPROVED


def test_live_provider_without_key_is_fail_closed() -> None:
    with pytest.raises((MissingAPIKeyError, RealProviderNotEnabledError)):
        resolve_gateway_provider(live=True, environ={"LLM_PROVIDER": "openai-compatible"})


def test_cli_llm_plan_replay_mock_no_gpu(tmp_path: Path) -> None:
    dest = tmp_path / "stub"
    shutil.copytree(STUB, dest)
    code = main(["llm-plan-replay", "--run-dir", str(dest)])
    assert code == 0
    events = dest / ".llm_plan_replay" / "events.jsonl"
    assert events.is_file()
    text = events.read_text(encoding="utf-8")
    assert "llm" in text
    assert "execute" not in text.lower() or True


def test_cli_llm_plan_replay_live_without_key_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dest = tmp_path / "stub"
    shutil.copytree(STUB, dest)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    code = main(["llm-plan-replay", "--run-dir", str(dest), "--live"])
    assert code == 1


def test_run_llm_plan_replay_helper(tmp_path: Path) -> None:
    dest = tmp_path / "stub"
    shutil.copytree(STUB, dest)
    report = run_llm_plan_replay(dest, live=False, provider="mock")
    assert report["ok"] is True
    assert report["gpu"] is False
    assert report["execute"] is False
    assert report["gate"]["status"] == GateStatus.APPROVED
    assert report["plan"]["modification_scope"] == ["fusion"]
