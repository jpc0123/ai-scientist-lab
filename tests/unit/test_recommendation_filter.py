from __future__ import annotations

from scientist_lab.domain.feedback import ExperimentRecommendation
from scientist_lab.services.recommendation_filter import (
    annotate_recommendations_against_tested,
    canonicalize_parameters,
)


def test_already_evaluated_and_deferred_interior_probe():
    source = {
        "learning_rate": 0.001,
        "epochs": 30,
        "hidden_units": 96,
        "batch_size": 64,
        "test_size": 0.2,
    }
    tested_nodes = [
        {
            "node_id": "node_003",
            "parameters": {
                "learning_rate": 0.001,
                "epochs": 30,
                "hidden_units": 64,
                "batch_size": 64,
                "test_size": 0.2,
            },
        },
        {
            "node_id": "node_004",
            "parameters": {
                "learning_rate": 0.001,
                "epochs": 30,
                "hidden_units": 128,
                "batch_size": 64,
                "test_size": 0.2,
            },
        },
        {
            "node_id": "node_005",
            "parameters": {
                "learning_rate": 0.001,
                "epochs": 30,
                "hidden_units": 96,
                "batch_size": 64,
                "test_size": 0.2,
            },
        },
    ]
    tested = {
        canonicalize_parameters(node["parameters"]): node["node_id"]
        for node in tested_nodes
    }
    recs = [
        ExperimentRecommendation(
            recommendation_type="expand_range",
            priority=0.65,
            rationale="Expand capacity.",
            parameter_changes={"hidden_units": 128},
        ),
        ExperimentRecommendation(
            recommendation_type="intermediate_value",
            priority=0.9,
            rationale="Try 80.",
            parameter_changes={"hidden_units": 80},
        ),
        ExperimentRecommendation(
            recommendation_type="intermediate_value",
            priority=0.88,
            rationale="First-step intermediate toward 128 was historically useful.",
            parameter_changes={"hidden_units": 96},
        ),
    ]
    out = annotate_recommendations_against_tested(
        recs,
        source_parameters=source,
        tested_index=tested,
        tested_nodes=tested_nodes,
    )
    by_units = {
        item.parameter_changes["hidden_units"]: item
        for item in out
        if item.parameter_changes
    }
    assert by_units[128].status == "already_evaluated"
    assert by_units[128].already_evaluated_node_id == "node_004"
    assert by_units[80].status == "deferred"
    assert by_units[96].status == "already_evaluated"
    assert by_units[96].already_evaluated_node_id == "node_005"


def test_first_intermediate_stays_active_when_only_two_endpoints_exist():
    source = {
        "learning_rate": 0.001,
        "epochs": 30,
        "hidden_units": 64,
        "batch_size": 64,
        "test_size": 0.2,
    }
    tested_nodes = [
        {
            "node_id": "node_003",
            "parameters": {**source, "hidden_units": 64},
        },
        {
            "node_id": "node_004",
            "parameters": {**source, "hidden_units": 128},
        },
    ]
    tested = {
        canonicalize_parameters(node["parameters"]): node["node_id"]
        for node in tested_nodes
    }
    out = annotate_recommendations_against_tested(
        [
            ExperimentRecommendation(
                recommendation_type="intermediate_value",
                priority=0.9,
                rationale="Try midpoint.",
                parameter_changes={"hidden_units": 96},
            )
        ],
        source_parameters=source,
        tested_index=tested,
        tested_nodes=tested_nodes,
    )
    assert out[0].status == "active"
    assert out[0].parameter_changes["hidden_units"] == 96
