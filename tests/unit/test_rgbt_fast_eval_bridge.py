from __future__ import annotations

import pytest

from scientist_lab.iteration.service import (
    FAST_EVAL_MINIMUM_SUCCESSFUL_SEEDS,
    FAST_EVAL_SEEDS,
    IterationService,
    SMOKE_DETECTION_MINIMUM_SUCCESSFUL_SEEDS,
    SMOKE_DETECTION_SEEDS,
)
from scientist_lab.tasks.rgbt_detection.decision_rules import (
    node_is_fast_eval,
    node_is_smoke_train_only,
    validate_smoke_decision,
)
from scientist_lab.tasks.rgbt_detection.fast_eval_bridge import (
    build_fast_eval_contract,
    relative_checkpoint_source,
)
from scientist_lab.tasks.rgbt_detection.feedback_rules import (
    annotate_feedback_for_detection,
    apply_detection_claim_gate,
)


def test_ready_for_fast_eval_decision_allowed():
    result = validate_smoke_decision(
        "ready_for_fast_eval", evidence_strength="moderate"
    )
    assert result["decision_type"] == "ready_for_fast_eval"
    assert result["evidence_strength"] == "weak"


def test_build_fast_eval_contract_sets_mode_and_checkpoint():
    smoke = {
        "schema_version": "1.1",
        "project_id": "project_rgbt_001",
        "node_id": "rgbt_node_001",
        "title": "RGB smoke",
        "task_type": "rgbt_detection",
        "execution_mode": "smoke_train",
        "parameters": {
            "input_mode": "rgb",
            "epochs": 2,
            "learning_rate": 0.01,
            "fusion_method": "none",
        },
        "task_config": {"claim_level": "pipeline_validation_only"},
        "seed": 42,
    }
    contract = build_fast_eval_contract(
        smoke,
        checkpoint_source="project_rgbt_001/exec_abc/checkpoint/last.npz",
    )
    assert contract["execution_mode"] == "fast_eval"
    assert contract["parameters"]["epochs"] == 0
    assert contract["parameters"]["learning_rate"] == 0.0
    assert (
        contract["parameters"]["checkpoint_source"]
        == "project_rgbt_001/exec_abc/checkpoint/last.npz"
    )
    assert contract["parent_node_id"] == "rgbt_node_001"
    assert node_is_fast_eval(contract)
    assert not node_is_smoke_train_only(contract)


def test_smoke_claim_gate_suggests_ready_for_fast_eval():
    gate = apply_detection_claim_gate(execution_mode="smoke_train")
    assert gate["suggested_decision_type"] == "ready_for_fast_eval"
    feedback = annotate_feedback_for_detection(
        {"recommendations": [], "hypothesis_status": "inconclusive"},
        execution_mode="smoke_train",
    )
    assert feedback["recommended_action"] == "prepare_fast_eval"
    assert feedback["claim_gate"]["suggested_decision_type"] == "ready_for_fast_eval"
    hints = [
        rec.get("next_decision_hint")
        for rec in feedback["recommendations"]
        if rec.get("next_decision_hint")
    ]
    assert "ready_for_fast_eval" in hints


def test_relative_checkpoint_source():
    assert (
        relative_checkpoint_source(
            project_id="project_rgbt_001", execution_id="exec_1"
        )
        == "project_rgbt_001/exec_1/checkpoint/last.npz"
    )


def test_iteration_seed_defaults_distinguish_smoke_and_fast_eval():
    class _Node:
        def __init__(self, contract):
            self.contract_json = contract

    smoke = _Node(
        {
            "task_type": "rgbt_detection",
            "execution_mode": "smoke_train",
            "task_config": {"claim_level": "pipeline_validation_only"},
        }
    )
    fast = _Node(
        {
            "task_type": "rgbt_detection",
            "execution_mode": "fast_eval",
            "task_config": {"claim_level": "pipeline_validation_only"},
        }
    )
    assert IterationService._is_smoke_train_pair(smoke, smoke)
    assert IterationService._is_fast_eval_pair(fast, fast)
    assert not IterationService._is_smoke_train_pair(fast, fast)
    assert FAST_EVAL_SEEDS == [42, 43, 44]
    assert FAST_EVAL_MINIMUM_SUCCESSFUL_SEEDS == 3
    assert SMOKE_DETECTION_SEEDS == [42]
    assert SMOKE_DETECTION_MINIMUM_SUCCESSFUL_SEEDS == 1


def test_build_fast_eval_rejects_non_rgbt():
    with pytest.raises(ValueError, match="rgbt_detection"):
        build_fast_eval_contract(
            {"task_type": "classification", "execution_mode": "fast_eval"},
            checkpoint_source="x/y/checkpoint/last.npz",
        )
