"""v1.5.6 Quality Gate tests."""

from __future__ import annotations

from pathlib import Path

from scientist_lab.llm_eval.quality_gate import (
    LLMQualityThresholds,
    verify_quality_gate,
)
from scientist_lab.llm_eval.runner import run_evaluation_suite
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings


def test_quality_gate_blocks_low_schema():
    scorecard = {
        "status": "completed",
        "planner": {
            "schema_valid_rate": 0.5,
            "protocol_compliance_rate": 1.0,
            "duplicate_candidate_rate": 0.0,
            "case_count": 10,
        },
        "safety": {"pass": True, "pass_rate": 1.0, "violation_count": 0, "case_count": 5},
        "operations": {},
    }
    gate = verify_quality_gate(scorecard)
    assert gate.status == "blocked"
    assert any("schema_valid_rate" in r for r in gate.blocking_reasons)


def test_quality_gate_blocks_safety_failure():
    scorecard = {
        "status": "completed",
        "planner": {
            "schema_valid_rate": 1.0,
            "protocol_compliance_rate": 1.0,
            "duplicate_candidate_rate": 0.0,
        },
        "safety": {"pass": False, "pass_rate": 0.0, "violation_count": 2},
        "operations": {},
    }
    gate = verify_quality_gate(scorecard)
    assert gate.status == "blocked"


def test_quality_gate_warns_on_duplicates():
    scorecard = {
        "status": "completed",
        "planner": {
            "schema_valid_rate": 1.0,
            "protocol_compliance_rate": 1.0,
            "duplicate_candidate_rate": 0.25,
            "evidence_gap_relevance_rate": 0.9,
            "case_count": 10,
        },
        "safety": {"pass": True, "pass_rate": 1.0, "violation_count": 0, "case_count": 5},
        "operations": {"average_latency_ms": 100},
    }
    gate = verify_quality_gate(scorecard)
    assert gate.status == "passed_with_warnings"
    assert gate.warnings


def test_service_verify_after_mock_run(tmp_path: Path):
    root = Path(__file__).resolve().parents[2]
    service = ExperimentService(
        settings=Settings(
            project_root=root,
            db_path=tmp_path / "test.db",
            runtime_dir=tmp_path / "runtime",
            outputs_dir=tmp_path / "outputs",
            experiment_app_dir=root / "experiment_app",
        ).resolve()
    )
    # Safety-only run is a clean hard-pass for the gate.
    result = run_evaluation_suite(
        "eval_suite_v1",
        provider="mock",
        project_id="project_rgbt_003",
        output_root=tmp_path / "outputs",
        task_types=["safety"],
        repository=service.llm_evals,
        profile=None,
    )
    # Persist via service API for verify path
    service.llm_evals.save_evaluation(
        evaluation_id=result["evaluation_id"],
        profile_id=result["profile_id"],
        suite_version=result["suite_version"],
        status=result["status"],
        result=result,
        report_path=(result.get("paths") or {}).get("evaluation_report_json"),
        case_rows=[],
    )
    gate = service.verify_llm_evaluation(result["evaluation_id"])
    assert gate["status"] in {"passed", "passed_with_warnings"}
    assert gate["status"] != "blocked"
