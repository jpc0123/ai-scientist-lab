from __future__ import annotations

import pytest

from scientist_lab.tasks.rgbt_detection.decision_rules import validate_smoke_decision
from scientist_lab.tasks.rgbt_detection.feedback_rules import (
    annotate_feedback_for_detection,
    apply_detection_claim_gate,
)


def test_exploratory_claim_gate():
    gate = apply_detection_claim_gate(
        execution_mode="fast_eval",
        evaluation_scope="fast_eval_subset",
        claim_level="exploratory_comparison",
        baseline_key="dfine_s",
    )
    assert gate["claim_level"] == "exploratory_comparison"
    assert gate["max_evidence_strength"] == "weak"
    assert gate["forbid_sota"] is True
    assert gate["suggested_decision_type"] == "ready_for_full_evaluation"
    blocked = " ".join(item["claim"] for item in gate["blocked_claims"])
    assert "state-of-the-art" in blocked.lower() or "SOTA" in blocked or "state-of-the-art" in blocked


def test_exploratory_decision_allows_ready_for_full_evaluation():
    result = validate_smoke_decision(
        "ready_for_full_evaluation",
        evidence_strength="strong",
        claim_level="exploratory_comparison",
    )
    assert result["decision_type"] == "ready_for_full_evaluation"
    assert result["evidence_strength"] == "weak"
    assert result["claim_level"] == "exploratory_comparison"


def test_exploratory_decision_forbids_sota():
    with pytest.raises(ValueError, match="forbids"):
        validate_smoke_decision(
            "sota",
            claim_level="exploratory_comparison",
        )


def test_annotate_feedback_exploratory():
    feedback = annotate_feedback_for_detection(
        {"recommendations": [], "hypothesis_status": "supported_with_repeated_evidence"},
        execution_mode="fast_eval",
        evaluation_scope="fast_eval_subset",
        claim_level="exploratory_comparison",
        baseline_key="dfine_s",
    )
    assert feedback["hypothesis_status"] == "inconclusive"
    assert feedback["evidence_strength"] == "weak"
    assert feedback["recommended_action"] == "compare_fast_eval_nodes"
    assert feedback["claim_gate"]["claim_level"] == "exploratory_comparison"
