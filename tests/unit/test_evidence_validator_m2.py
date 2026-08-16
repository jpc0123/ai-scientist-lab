"""EvidenceValidator / ResultParser / ExceptionHandler (no GPU, no Planner)."""

from __future__ import annotations

import json
from pathlib import Path

from scientist_lab.core.evidence_validator import EvidenceValidator
from scientist_lab.core.exception_handler import next_exception_action
from scientist_lab.core.result_parser import ResultParser
from scientist_lab.core.schema_registry import SCHEMA_DIR, load_json, validate_named
from scientist_lab.adapters.dfine.run_loop import run_gated_dfine

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


def _result(**overrides):
    result = {
        "schema_version": "1.0.0",
        "run_id": "run_plan_round1_neck_hr",
        "metrics": {"APS": 0.1, "mAP50_95": 0.09},
        "artifacts": {"paths": ["metrics.json", "checkpoint_selection.json"], "missing_expected": []},
        "execution": {"status": "success", "exit_code": 0},
        "raw_metric_refs": ["metrics.json"],
    }
    result.update(overrides)
    return result


def test_valid_evidence_does_not_keep_or_discard() -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")
    verdict = EvidenceValidator().validate(
        _result(),
        protocol=protocol,
        expected_artifacts=["metrics.json", "checkpoint_selection.json"],
        fingerprint_comparable=True,
    )
    assert verdict.evidence_status == "VALID"
    assert verdict.review_allowed is True


def test_missing_eval_file_is_incomplete() -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")
    verdict = EvidenceValidator().validate(
        _result(artifacts={"paths": ["metrics.json"], "missing_expected": ["checkpoint_selection.json"]}),
        protocol=protocol,
        expected_artifacts=["metrics.json", "checkpoint_selection.json"],
    )
    assert verdict.evidence_status == "INCOMPLETE"
    assert verdict.review_allowed is False


def test_failed_oom_is_not_applicable_and_retryable() -> None:
    result = _result(
        metrics={},
        artifacts={"paths": [], "missing_expected": ["metrics.json"]},
        execution={"status": "failed", "error_type": "OOM", "exit_code": 1},
    )
    verdict = EvidenceValidator().validate(result, handle_status="failed")
    assert verdict.evidence_status == "NOT_APPLICABLE"
    assert next_exception_action(result, evidence_status=verdict.evidence_status) == "RETRY_EXECUTION"


def test_nan_primary_is_invalid() -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")
    verdict = EvidenceValidator().validate(
        _result(metrics={"APS": float("nan"), "mAP50_95": 0.09}),
        protocol=protocol,
        fingerprint_comparable=True,
    )
    assert verdict.evidence_status == "INVALID"
    assert verdict.review_allowed is False


def test_fingerprint_mismatch_is_invalid() -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")
    verdict = EvidenceValidator().validate(
        _result(),
        protocol=protocol,
        fingerprint_comparable=False,
    )
    assert verdict.evidence_status == "INVALID"


def test_result_parser_validates_schema() -> None:
    handle = {
        "status": "completed",
        "metrics": {"APS": 0.083, "mAP50_95": 0.077},
        "artifacts": {"paths": ["metrics.json"], "missing_expected": []},
        "raw_metric_refs": ["metrics.json"],
    }
    contract = {"run_id": "run_x"}
    result = ResultParser().from_handle(contract, handle)
    validate_named("experiment_result", result)
    assert result["execution"]["status"] == "success"


def test_dry_run_without_artifacts_is_not_applicable(tmp_path: Path) -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")
    report = run_gated_dfine(
        protocol=protocol,
        plan=_plan(),
        output_dir=tmp_path,
        execute=False,
    )
    assert report["gate"]["status"] == "APPROVED"
    assert report["evidence"]["evidence_status"] == "NOT_APPLICABLE"
    assert report["review_decision"] == "PENDING"
    assert "rubric" not in report
    assert report["scientific_outcome"] == "NOT_EVALUATED"
    events = (tmp_path / "research_events.jsonl").read_text(encoding="utf-8")
    assert "gate_decision" in events
    assert "evidence_check" in events


def test_recovered_run_missing_checkpoint_is_incomplete(tmp_path: Path) -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")

    def runner(_contract, output: Path):
        output.mkdir(parents=True, exist_ok=True)
        (output / "metrics.json").write_text(
            json.dumps({"APS": 0.09, "mAP50_95": 0.08}), encoding="utf-8"
        )
        return {"status": "completed"}

    report = run_gated_dfine(
        protocol=protocol,
        plan=_plan(),
        output_dir=tmp_path,
        execute=True,
        live_runner=runner,
    )
    assert report["evidence"]["evidence_status"] == "INCOMPLETE"
    assert report["review_decision"] == "PENDING"
    assert "KEEP" not in json.dumps(report)
    assert "DISCARD" not in json.dumps(report)
