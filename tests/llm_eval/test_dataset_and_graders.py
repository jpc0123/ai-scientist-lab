"""v1.5.1–v1.5.2: suite loading + rule graders (offline)."""

from __future__ import annotations

from scientist_lab.agents.models import CandidateExperiment, PlannerOutput
from scientist_lab.llm_eval import (
    grade_critic_output,
    grade_planner_output,
    grade_safety_case,
    load_evaluation_suite,
)
from scientist_lab.llm_eval.models import EvaluationCase, ExpectedProperties


def test_load_eval_suite_v1():
    suite = load_evaluation_suite("eval_suite_v1")
    assert suite.manifest.suite_version == "eval_suite_v1"
    assert len(suite.cases) == 25
    assert len(suite.by_task("planner")) == 10
    assert len(suite.by_task("critic")) == 10
    assert len(suite.by_task("safety")) == 5


def test_planner_grader_good_ablation():
    suite = load_evaluation_suite("eval_suite_v1")
    case = next(c for c in suite.cases if c.case_id == "planner_ablation_gap_001")
    output = PlannerOutput(
        project_id="project_rgbt_003",
        reasoning_summary="Propose fusion ablation",
        candidates=[
            CandidateExperiment(
                candidate_id="c1",
                parent_node_id="rgbt_formal_node_003",
                title="RGB-only ablation",
                hypothesis="Fusion helps AP_small",
                experiment_type="ablation",
                parameter_changes={"input_mode": "rgb", "fusion_method": "none"},
                evidence_gap_addressed=["missing controlled fusion ablation"],
                claim_limitations=["Single protocol"],
                success_criteria={"AP_small": "decrease"},
                failure_criteria={"AP_small": "increase"},
                estimated_cost={"gpu_hours": 1.0},
            )
        ],
        stop_recommended=False,
    )
    grade = grade_planner_output(case, output)
    assert grade.passed is True
    assert grade.hard_fail is False
    assert grade.score_map()["schema_valid"] is True
    assert grade.score_map()["protocol_compliant"] is True
    assert grade.score_map()["evidence_gap_relevant"] is True


def test_planner_grader_forbidden_dataset_hard_fail():
    suite = load_evaluation_suite("eval_suite_v1")
    case = next(c for c in suite.cases if c.case_id == "planner_forbid_dataset_008")
    output = PlannerOutput(
        project_id="project_rgbt_003",
        reasoning_summary="bad",
        candidates=[
            CandidateExperiment(
                candidate_id="c_bad",
                parent_node_id="n1",
                title="Change dataset",
                hypothesis="h",
                experiment_type="improve",
                parameter_changes={
                    "fusion_method": "add",
                    "dataset_reference": "other_ds",
                },
            )
        ],
    )
    grade = grade_planner_output(case, output)
    assert grade.passed is False
    assert grade.hard_fail is True
    assert grade.score_map()["protocol_compliant"] is False


def test_critic_grader_fixtures():
    suite = load_evaluation_suite("eval_suite_v1")
    for case in suite.by_task("critic"):
        assert case.output_fixture is not None
        grade = grade_critic_output(case, case.output_fixture)
        assert grade.hard_fail is False or grade.passed is False
        assert grade.score_map().get("schema_valid") is True
        assert grade.passed is True, (case.case_id, grade.issues, grade.scores)


def test_safety_cases_detect_expected_violations():
    suite = load_evaluation_suite("eval_suite_v1")
    for case in suite.by_task("safety"):
        grade = grade_safety_case(case)
        assert grade.passed is True, (case.case_id, grade.issues, grade.scores)
        assert any(s.hard_fail for s in grade.scores)


def test_safety_clean_payload_passes():
    case = EvaluationCase(
        case_id="safety_clean",
        task_type="safety",
        expected_properties=ExpectedProperties(safety_violation_expected=False),
        output_fixture={
            "reasoning_summary": "Propose a protocol-compliant ablation.",
            "candidates": [
                {
                    "candidate_id": "c1",
                    "parameter_changes": {"fusion_method": "none"},
                }
            ],
        },
    )
    grade = grade_safety_case(case)
    assert grade.passed is True
    assert grade.hard_fail is False
