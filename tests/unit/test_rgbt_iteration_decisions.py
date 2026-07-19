from __future__ import annotations

import pytest

from scientist_lab.tasks.rgbt_detection.decision_rules import (
    node_is_smoke_detection,
    validate_smoke_decision,
)


def test_smoke_decision_allows_pipeline_validated():
    result = validate_smoke_decision(
        "pipeline_validated", evidence_strength="strong"
    )
    assert result["decision_type"] == "pipeline_validated"
    assert result["evidence_strength"] == "weak"
    assert result["claim_level"] == "pipeline_validation_only"


def test_smoke_decision_forbids_performance_winner():
    with pytest.raises(ValueError, match="forbids decision_type"):
        validate_smoke_decision("performance_winner")


def test_smoke_decision_forbids_sota_aliases():
    for dtype in ("best_model", "superior_method", "state_of_the_art", "sota"):
        with pytest.raises(ValueError):
            validate_smoke_decision(dtype)


def test_node_is_smoke_detection():
    assert node_is_smoke_detection(
        {
            "task_type": "rgbt_detection",
            "execution_mode": "smoke_train",
            "task_config": {"claim_level": "pipeline_validation_only"},
        }
    )
    assert not node_is_smoke_detection(
        {"task_type": "classification", "execution_mode": "smoke_train"}
    )
