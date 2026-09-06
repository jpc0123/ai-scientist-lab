"""Sanitize expected_effect extras that break experiment_plan schema."""

from scientist_lab.core.schema_registry import validate_named
from scientist_lab.llm.planner_contract import sanitize_expected_effect


def test_sanitize_expected_effect_strips_reference_last() -> None:
    cleaned = sanitize_expected_effect(
        {
            "primary_metric": "APS_lowlight",
            "direction": "increase",
            "rationale": "vs last",
            "reference_last": 0.02,
            "reference_last_delta": 0.01,
        }
    )
    assert cleaned == {
        "primary_metric": "APS_lowlight",
        "direction": "increase",
        "rationale": "vs last",
    }
    plan = {
        "schema_version": "1.0.0",
        "plan_id": "plan_sanitize_expected",
        "project_id": "project_rgbt_cuda_001",
        "protocol_id": "research_protocol_rgbt_dfine_v26",
        "protocol_version": 2,
        "parent_run_id": "run_prev",
        "round_index": 1,
        "observation": "sanitize test",
        "hypothesis": "extras must not break schema",
        "modification_scope": ["fusion"],
        "proposed_changes": [{"target": "fusion", "summary": "keep F1"}],
        "controlled_variables": ["evaluator"],
        "expected_effect": cleaned,
        "evaluation": {"method": "fast_eval"},
        "budget_class": "formal",
        "risk_level": "auto",
        "memory_refs": {"lesson_ids": [], "strategy_ids": []},
        "evidence_runs": ["run_prev"],
        "bootstrap": False,
        "rationale": "unit",
    }
    validate_named("experiment_plan", plan)
