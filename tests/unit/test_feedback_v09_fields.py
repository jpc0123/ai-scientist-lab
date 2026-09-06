from __future__ import annotations

from scientist_lab.domain.feedback import FeedbackInput
from scientist_lab.services.feedback_analyzer import analyze_feedback
from scientist_lab.tasks.rgbt_detection.feedback_rules import (
    annotate_feedback_for_detection,
)


def _fast_eval_comparison(**overrides):
    base = {
        "hypothesis_status": "supported_with_repeated_evidence",
        "primary_metric": "mAP50_95",
        "mean_delta": 0.02,
        "candidate_win_count": 3,
        "baseline_win_count": 0,
        "shared_seeds": [42, 43, 44],
        "stable_improvement": True,
        "claim_level": "exploratory_comparison",
        "parameter_changes": {
            "input_mode": {"from": "rgb", "to": "rgbt"},
            "fusion_method": {"from": "none", "to": "early_concat"},
        },
        "metric_relations": {
            "mAP50_95": "candidate_better",
            "AP_small": "candidate_better",
        },
        "resource_relations": {
            "duration_seconds": "candidate_worse",
            "peak_gpu_memory_mb": "candidate_worse",
        },
        "verification": {"valid": True},
        "implementation": "stand_in",
    }
    base.update(overrides)
    return base


def test_feedback_includes_v09_structured_fields():
    report = analyze_feedback(
        FeedbackInput(
            baseline_node_id="rgbt_formal_node_001",
            candidate_node_id="rgbt_formal_node_003",
            comparison=_fast_eval_comparison(),
            baseline_contract={
                "execution_mode": "fast_eval",
                "task_config": {
                    "claim_level": "exploratory_comparison",
                    "implementation": "stand_in",
                    "evaluation_scope": "fast_eval_subset",
                },
            },
            candidate_contract={
                "execution_mode": "fast_eval",
                "task_config": {
                    "claim_level": "exploratory_comparison",
                    "implementation": "stand_in",
                    "evaluation_scope": "fast_eval_subset",
                },
                "parameters": {"input_mode": "rgbt", "fusion_method": "early_concat"},
            },
        )
    )
    payload = report.model_dump()
    for key in (
        "scientific_interpretation",
        "engineering_findings",
        "performance_findings",
        "resource_tradeoffs",
        "evidence_gaps",
        "claim_restrictions",
        "recommended_next_experiments",
    ):
        assert key in payload
        assert isinstance(payload[key], list)
        assert payload[key], f"{key} should be non-empty"

    blob = " ".join(payload["claim_restrictions"]).lower()
    assert "sota" in blob or "state-of-the-art" in blob
    assert "rgbt-tiny" in blob or "full" in blob
    assert any("stand-in" in item.lower() for item in payload["engineering_findings"])
    assert any("fast eval" in item.lower() for item in payload["evidence_gaps"])


def test_weak_evidence_blocks_sota_and_full_benchmark_in_claim_gate():
    base = analyze_feedback(
        FeedbackInput(
            baseline_node_id="a",
            candidate_node_id="b",
            comparison=_fast_eval_comparison(),
            baseline_contract={"execution_mode": "fast_eval"},
            candidate_contract={
                "execution_mode": "fast_eval",
                "task_config": {
                    "claim_level": "exploratory_comparison",
                    "implementation": "stand_in",
                    "evaluation_scope": "fast_eval_subset",
                },
                "parameters": {"epochs": 5},
            },
        )
    ).model_dump()
    annotated = annotate_feedback_for_detection(
        base,
        execution_mode="fast_eval",
        evaluation_scope="fast_eval_subset",
        claim_level="exploratory_comparison",
        baseline_key="dfine_s",
    )
    restrictions = " ".join(annotated["claim_restrictions"]).lower()
    assert "sota" in restrictions or "state-of-the-art" in restrictions
    assert "rgbt-tiny" in restrictions or "full" in restrictions
    assert annotated["evidence_strength"] == "weak"
    assert any(
        "exploratory" in item.lower()
        for item in annotated["scientific_interpretation"]
    )
    assert annotated["claim_gate"]["forbid_sota"] is True
