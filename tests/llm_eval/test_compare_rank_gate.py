"""v1.5.7–v1.5.9: compare, rank, planning quality gate."""

from __future__ import annotations

from pathlib import Path

from scientist_lab.llm_eval.compare import compare_evaluations
from scientist_lab.llm_eval.profiles import LLMModelProfile
from scientist_lab.llm_eval.ranking import rank_profiles
from scientist_lab.llm_eval.runner import run_evaluation_suite
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings


def _service(tmp_path: Path) -> ExperimentService:
    root = Path(__file__).resolve().parents[2]
    return ExperimentService(
        settings=Settings(
            project_root=root,
            db_path=tmp_path / "test.db",
            runtime_dir=tmp_path / "runtime",
            outputs_dir=tmp_path / "outputs",
            experiment_app_dir=root / "experiment_app",
        ).resolve()
    )


def _passing_scorecard(**overrides) -> dict:
    base = {
        "status": "completed",
        "profile_id": "p1",
        "suite_version": "eval_suite_v1",
        "planner": {
            "schema_valid_rate": 1.0,
            "protocol_compliance_rate": 1.0,
            "evidence_gap_relevance_rate": 0.9,
            "duplicate_candidate_rate": 0.0,
            "pass_rate": 1.0,
            "case_count": 10,
        },
        "critic": {
            "valid_acceptance_rate": 0.9,
            "invalid_rejection_rate": 0.9,
            "claim_overreach_detection_rate": 1.0,
            "pass_rate": 0.9,
            "case_count": 10,
        },
        "safety": {"pass": True, "pass_rate": 1.0, "violation_count": 0, "case_count": 5},
        "operations": {
            "average_latency_ms": 1000,
            "total_tokens": 100,
            "estimated_cost_usd": 0.1,
        },
        "metadata": {
            "planner_prompt_version": "mock_v1",
            "critic_prompt_version": "mock_v1",
        },
    }
    base.update(overrides)
    return base


def test_compare_detects_evidence_regression():
    baseline = _passing_scorecard()
    candidate = _passing_scorecard(
        planner={
            "schema_valid_rate": 1.0,
            "protocol_compliance_rate": 1.0,
            "evidence_gap_relevance_rate": 0.7,
            "duplicate_candidate_rate": 0.0,
            "pass_rate": 0.8,
            "case_count": 10,
        }
    )
    result = compare_evaluations(
        baseline, candidate, baseline_id="b1", candidate_id="c1"
    )
    assert result.regression_detected is True
    assert any("Evidence-gap" in r for r in result.blocking_reasons)


def test_rank_excludes_unqualified():
    entries = [
        {
            "profile_id": "good",
            "evaluation_id": "e1",
            "scorecard": _passing_scorecard(profile_id="good"),
        },
        {
            "profile_id": "bad_safety",
            "evaluation_id": "e2",
            "scorecard": _passing_scorecard(
                profile_id="bad_safety",
                safety={
                    "pass": False,
                    "pass_rate": 0.0,
                    "violation_count": 1,
                    "case_count": 5,
                },
            ),
        },
    ]
    ranked = rank_profiles(entries)
    assert ranked[0].profile_id == "good"
    assert ranked[0].qualified is True
    bad = next(r for r in ranked if r.profile_id == "bad_safety")
    assert bad.qualified is False
    assert bad.profile_score is None


def test_service_compare_rank_select_and_gate(tmp_path: Path):
    service = _service(tmp_path)
    profile = LLMModelProfile(
        profile_id="mock_default",
        provider="mock",
        model="mock-planner-v1",
        api_mode="offline",
        planner_prompt_version="mock_v1",
        critic_prompt_version="mock_v1",
    )
    service.register_llm_profile(profile=profile)

    # Persist two evaluations: baseline good, candidate regressed.
    service.llm_evals.save_evaluation(
        evaluation_id="llm_eval_base",
        profile_id="mock_default",
        suite_version="eval_suite_v1",
        status="completed",
        result=_passing_scorecard(evaluation_id="llm_eval_base"),
        report_path=None,
        case_rows=[],
    )
    bad = _passing_scorecard(
        evaluation_id="llm_eval_cand",
        planner={
            "schema_valid_rate": 1.0,
            "protocol_compliance_rate": 1.0,
            "evidence_gap_relevance_rate": 0.5,
            "duplicate_candidate_rate": 0.2,
            "pass_rate": 0.5,
            "case_count": 10,
        },
    )
    service.llm_evals.save_evaluation(
        evaluation_id="llm_eval_cand",
        profile_id="mock_default",
        suite_version="eval_suite_v1",
        status="completed",
        result=bad,
        report_path=None,
        case_rows=[],
    )

    compared = service.compare_llm_evaluations("llm_eval_base", "llm_eval_cand")
    assert compared["regression_detected"] is True

    # Replace latest with a passing eval for ranking / planning gate.
    service.llm_evals.save_evaluation(
        evaluation_id="llm_eval_ok",
        profile_id="mock_default",
        suite_version="eval_suite_v1",
        status="completed",
        result=_passing_scorecard(evaluation_id="llm_eval_ok"),
        report_path=None,
        case_rows=[],
    )
    ranked = service.rank_llm_profiles(suite_version="eval_suite_v1")
    assert ranked["qualified_count"] >= 1
    assert ranked["ranking"][0]["qualified"] is True

    selected = service.select_llm_profile("mock_default")
    assert selected["default_profile_id"] == "mock_default"

    # Unqualified: no eval for other profile
    service.register_llm_profile(
        profile=LLMModelProfile(
            profile_id="unevaluated",
            provider="mock",
            model="x",
            api_mode="offline",
            planner_prompt_version="mock_v1",
            critic_prompt_version="mock_v1",
        )
    )
    blocked = service.plan_next(
        "project_missing",
        provider="mock",
        model_profile="unevaluated",
        require_quality_gate=True,
    )
    assert blocked["status"] == "profile_not_qualified"

    bypassed = service.plan_next(
        "project_missing",
        provider="mock",
        model_profile="unevaluated",
        require_quality_gate=True,
        allow_unqualified_profile=True,
    )
    # May planner_failed due to missing project, but gate bypass recorded if planning ran,
    # or profile gate allows through then planner runs.
    assert bypassed.get("quality_gate", {}).get("quality_gate_bypassed") is True
    assert bypassed.get("formal_approval_eligible") is False


def test_safety_suite_run_persists_for_rank(tmp_path: Path):
    service = _service(tmp_path)
    service.register_llm_profile(
        Path(__file__).resolve().parents[2] / "examples" / "llm_profile.json"
    )
    result = run_evaluation_suite(
        "eval_suite_v1",
        provider="mock",
        project_id="project_rgbt_003",
        output_root=tmp_path / "outputs",
        task_types=["safety"],
        repository=service.llm_evals,
        profile=service.llm_evals.get_profile("mock_default"),
    )
    assert result["status"] == "completed"
    ranked = service.rank_llm_profiles(suite_version="eval_suite_v1")
    # Safety-only scorecards qualify (no planner hard metrics).
    assert any(r["profile_id"] == "mock_default" and r["qualified"] for r in ranked["ranking"])
