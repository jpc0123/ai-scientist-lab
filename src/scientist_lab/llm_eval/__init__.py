"""scientist_lab.llm_eval — versioned LLM quality evaluation (v1.5)."""

from scientist_lab.llm_eval.aggregator import aggregate_scorecard, write_evaluation_artifacts
from scientist_lab.llm_eval.critic_grader import grade_critic_output
from scientist_lab.llm_eval.dataset import DatasetError, load_evaluation_suite
from scientist_lab.llm_eval.models import (
    CaseGrade,
    EvaluationCase,
    EvaluationSuite,
    EvaluationSuiteManifest,
    ExpectedProperties,
    MetricScore,
)
from scientist_lab.llm_eval.planner_grader import grade_planner_output
from scientist_lab.llm_eval.profiles import LLMModelProfile, default_mock_profile
from scientist_lab.llm_eval.runner import run_evaluation_suite
from scientist_lab.llm_eval.safety_grader import grade_safety_case, grade_safety_payload

__all__ = [
    "CaseGrade",
    "DatasetError",
    "EvaluationCase",
    "EvaluationSuite",
    "EvaluationSuiteManifest",
    "ExpectedProperties",
    "LLMModelProfile",
    "MetricScore",
    "aggregate_scorecard",
    "default_mock_profile",
    "grade_critic_output",
    "grade_planner_output",
    "grade_safety_case",
    "grade_safety_payload",
    "load_evaluation_suite",
    "run_evaluation_suite",
    "write_evaluation_artifacts",
]
