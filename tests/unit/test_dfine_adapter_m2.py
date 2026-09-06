"""M2: DFINEAdapter execute / parse_metrics / fingerprint (no GPU)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scientist_lab.adapters import DFINEAdapter, MaterializeRejected, fingerprints_equivalent
from scientist_lab.adapters.dfine.cuda_runner import freeze_to_legacy_contract
from scientist_lab.core.gate_engine import GateEngine, GateStatus
from scientist_lab.core.schema_registry import SCHEMA_DIR, load_json, validate_named
from scientist_lab.instrumentation import EventAppender

EXAMPLES = SCHEMA_DIR / "examples"


def _plan() -> dict:
    return {
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


def test_dry_run_execute_emits_instrumentation(tmp_path: Path) -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")
    events = EventAppender(tmp_path / "research_events.jsonl")
    adapter = DFINEAdapter(events=events)
    contract = adapter.materialize_contract(_plan(), protocol)
    handle = adapter.execute(contract, protocol, output_dir=tmp_path / "run", dry_run=True)
    assert handle["status"] == "dry_run"
    assert handle["fingerprint"]["protocol_id"] == protocol["protocol_id"]
    types = [row["event_type"] for row in events.load_all()]
    assert "fingerprint_check" in types
    assert "tool_call" in types
    assert "execution" in types
    result = adapter.evaluate(contract, handle)
    validate_named("experiment_result", result)


def test_parse_metrics_does_not_invent_aps_from_map(tmp_path: Path) -> None:
    (tmp_path / "metrics.json").write_text(
        json.dumps({"mAP50_95": 0.082, "mAP50": 0.21, "params_m": 10.4}),
        encoding="utf-8",
    )
    parsed = DFINEAdapter().parse_metrics(tmp_path)
    assert parsed["metrics"]["mAP50_95"] == pytest.approx(0.082)
    assert parsed["metrics"]["APS"] is None


def test_parse_metrics_reads_aps_and_nested_block(tmp_path: Path) -> None:
    (tmp_path / "metrics.json").write_text(
        json.dumps({"metrics": {"APS": 0.41, "mAP50_95": 0.077}, "params_m": 11.0}),
        encoding="utf-8",
    )
    parsed = DFINEAdapter().parse_metrics(tmp_path)
    assert parsed["metrics"]["APS"] == pytest.approx(0.41)
    assert parsed["metrics"]["params_m"] == pytest.approx(11.0)


def test_fingerprint_changes_when_split_changes() -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")
    adapter = DFINEAdapter()
    contract = adapter.materialize_contract(_plan(), protocol)
    fp1 = adapter.compute_fingerprint(protocol, contract)
    contract2 = dict(contract)
    contract2["dataset"] = {
        **dict(contract["dataset"]),
        "split_reference": "split:other",
    }
    fp2 = adapter.compute_fingerprint(protocol, contract2)
    assert not fingerprints_equivalent(fp1, fp2)
    assert fingerprints_equivalent(fp1, adapter.compute_fingerprint(protocol, contract))


def test_existing_artifacts_recovered_without_gpu(tmp_path: Path) -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")
    adapter = DFINEAdapter()
    contract = adapter.materialize_contract(_plan(), protocol)
    out = tmp_path / "run"
    out.mkdir()
    (out / "metrics.json").write_text(
        json.dumps({"APS": 0.083, "mAP50_95": 0.077, "params_m": 10.4, "flops_g": 40.0}),
        encoding="utf-8",
    )
    (out / "checkpoint_selection.json").write_text("{}", encoding="utf-8")
    handle = adapter.execute(contract, protocol, output_dir=out, dry_run=True)
    assert handle["status"] == "completed"
    assert handle["metrics"]["APS"] == pytest.approx(0.083)
    result = adapter.evaluate(contract, handle)
    assert "metrics.json" in result["artifacts"]["paths"]
    assert result["execution"]["status"] == "success"


def test_live_execute_without_runner_is_rejected(tmp_path: Path) -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")
    adapter = DFINEAdapter()
    contract = adapter.materialize_contract(_plan(), protocol)
    with pytest.raises(MaterializeRejected):
        adapter.execute(contract, protocol, output_dir=tmp_path / "run", dry_run=False)


def test_live_runner_hook_completes(tmp_path: Path) -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")
    adapter = DFINEAdapter()
    contract = adapter.materialize_contract(_plan(), protocol)
    out = tmp_path / "run"

    def runner(_contract, output: Path):
        output.mkdir(parents=True, exist_ok=True)
        (output / "metrics.json").write_text(
            json.dumps({"APS": 0.09, "mAP50_95": 0.08}),
            encoding="utf-8",
        )
        return {"status": "completed", "backend": "stub"}

    handle = adapter.execute(
        contract, protocol, output_dir=out, dry_run=False, live_runner=runner
    )
    assert handle["status"] == "completed"
    assert handle["run_view"]["backend"] == "stub"
    assert handle["metrics"]["APS"] == pytest.approx(0.09)


def _fusion_plan() -> dict:
    plan = _plan()
    plan["plan_id"] = "plan_round2_fusion_early"
    plan["modification_scope"] = ["fusion"]
    plan["proposed_changes"] = [
        {"target": "fusion", "summary": "Probe fusion after DISCARD on neck"}
    ]
    plan["hypothesis"] = (
        "After negative evidence on neck, a fusion change is a better probe of APS "
        "than repeating the discarded direction."
    )
    return plan


def test_materialize_neck_vs_fusion_how_differs() -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")
    adapter = DFINEAdapter()
    neck = adapter.materialize_contract(_plan(), protocol)
    fusion = adapter.materialize_contract(_fusion_plan(), protocol)
    neck_how = neck["materialization"]["how"]
    fusion_how = fusion["materialization"]["how"]
    assert neck_how["primary_module"] == "neck"
    assert fusion_how["primary_module"] == "fusion"
    assert neck_how["signature"] != fusion_how["signature"]
    assert neck["materialization"]["legacy_parameters"] != fusion[
        "materialization"
    ]["legacy_parameters"]
    assert neck_how["input_mode"] == "rgb"
    assert neck_how["fusion_method"] == "none"
    assert neck_how["neck_type"] == "standard"
    assert fusion_how["input_mode"] == "rgbt"
    assert fusion_how["fusion_method"] == "early_concat"
    assert fusion_how["neck_type"] == "standard"
    assert neck_how["invented_operators"] == []
    assert fusion_how["invented_operators"] == []
    assert "FDPN" not in str(neck)
    assert "FDPN" not in str(fusion)
    assert "GatedMultiscaleFusion" not in str(fusion)
    neck_legacy = freeze_to_legacy_contract(neck)["parameters"]
    fusion_legacy = freeze_to_legacy_contract(fusion)["parameters"]
    assert neck_legacy["fusion_method"] == "none"
    assert fusion_legacy["fusion_method"] == "early_concat"
    assert neck_legacy["input_mode"] != fusion_legacy["input_mode"]
    gate = GateEngine()
    assert gate.evaluate(protocol, _plan(), neck).status == GateStatus.APPROVED
    assert gate.evaluate(protocol, _fusion_plan(), fusion).status == GateStatus.APPROVED


def test_neck_fusion_frozen_fingerprint_equivalent_how_notes_differ() -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")
    adapter = DFINEAdapter()
    neck = adapter.materialize_contract(_plan(), protocol)
    fusion = adapter.materialize_contract(_fusion_plan(), protocol)
    fp_neck = adapter.compute_fingerprint(protocol, neck)
    fp_fusion = adapter.compute_fingerprint(protocol, fusion)
    fp_bound = adapter.compute_fingerprint(protocol, None)
    assert fingerprints_equivalent(fp_neck, fp_fusion)
    assert fingerprints_equivalent(fp_bound, fp_neck)
    assert fingerprints_equivalent(fp_bound, fp_fusion)
    assert fp_neck["notes"] != fp_fusion["notes"]
    assert neck["materialization"]["how_signature"] in fp_neck["notes"]
    assert fusion["materialization"]["how_signature"] in fp_fusion["notes"]
    validate_named("frozen_fingerprint", fp_neck)
    validate_named("frozen_fingerprint", fp_fusion)


def test_instrumentation_carries_how_signature(tmp_path: Path) -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")
    events = EventAppender(tmp_path / "research_events.jsonl")
    adapter = DFINEAdapter(events=events)
    contract = adapter.materialize_contract(_fusion_plan(), protocol)
    adapter.execute(contract, protocol, output_dir=tmp_path / "run", dry_run=True)
    rows = events.load_all()
    fp_evt = next(row for row in rows if row["event_type"] == "fingerprint_check")
    tool_evt = next(row for row in rows if row["event_type"] == "tool_call")
    sig = contract["materialization"]["how_signature"]
    assert fp_evt["payload"]["how_signature"] == sig
    assert fp_evt["payload"]["how"]["fusion_method"] == "early_concat"
    assert fp_evt["payload"]["frozen_hashes_exclude_how"] is True
    assert tool_evt["payload"]["how_signature"] == sig
    assert tool_evt["payload"]["legacy_parameters"]["fusion_method"] == "early_concat"
    assert tool_evt["payload"]["legacy_parameters"]["input_mode"] == "rgbt"


def test_formal_how_full_train_drops_probe_subset() -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_formal_c1_v1.json")
    plan = load_json(EXAMPLES / "experiment_plan_formal_01_rgb_none.json")
    fusion = load_json(EXAMPLES / "experiment_plan_formal_02_early_concat.json")
    validate_named("research_protocol", protocol)
    validate_named("experiment_plan", plan)
    validate_named("experiment_plan", fusion)
    adapter = DFINEAdapter()
    base = adapter.materialize_contract(plan, protocol)
    cand = adapter.materialize_contract(fusion, protocol)
    assert base["budget_class"] == "formal"
    assert base["budget"]["timeout_seconds"] == 14400
    assert base["materialization"]["execution_mode"] == "full_train"
    assert "max_train_images" not in base["materialization"]["legacy_parameters"]
    legacy = freeze_to_legacy_contract(base)
    assert legacy["execution_mode"] == "full_train"
    assert "max_train_images" not in legacy["parameters"]
    assert "max_val_images" not in legacy["parameters"]
    assert legacy["parameters"]["epochs"] == 2
    assert legacy["parameters"]["scale_queries_to_tokens"] is True
    assert legacy["parameters"]["input_mode"] == "rgb"
    assert legacy["parameters"]["fusion_method"] == "none"
    assert legacy["resources"]["timeout_seconds"] == 14400
    assert legacy["task_config"]["evaluation_scope"] != "fast_eval_subset"
    fusion_legacy = freeze_to_legacy_contract(cand)
    assert fusion_legacy["execution_mode"] == "full_train"
    assert fusion_legacy["parameters"]["input_mode"] == "rgbt"
    assert fusion_legacy["parameters"]["fusion_method"] == "early_concat"
    assert "FDPN" not in json.dumps(fusion_legacy)
    fp_base = adapter.compute_fingerprint(protocol, base)
    fp_cand = adapter.compute_fingerprint(protocol, cand)
    assert fingerprints_equivalent(fp_base, fp_cand)
    gate = GateEngine()
    assert gate.evaluate(protocol, plan, base).status == GateStatus.APPROVED
    assert gate.evaluate(protocol, fusion, cand).status == GateStatus.APPROVED

