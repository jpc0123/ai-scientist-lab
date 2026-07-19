from __future__ import annotations

from scientist_lab.domain.feedback import ExperimentRecommendation, FeedbackReport
from scientist_lab.services.proposal import build_next_contract, select_proposal_recommendation
from scientist_lab.services.recommendation_filter import (
    annotate_recommendations_against_tested,
    canonicalize_parameters,
    collect_tested_parameter_index,
    parameter_fingerprint,
)


def test_parameter_fingerprint_stable_and_sensitive():
    params = {
        "learning_rate": 0.001,
        "epochs": 30,
        "hidden_units": 96,
        "batch_size": 64,
        "test_size": 0.2,
    }
    a = parameter_fingerprint(
        "digits-mlp-v1", "sklearn:digits", "local:experiment_app", params
    )
    b = parameter_fingerprint(
        "digits-mlp-v1", "sklearn:digits", "local:experiment_app", dict(params)
    )
    assert a == b
    c = parameter_fingerprint(
        "digits-mlp-v1", "sklearn:digits", "local:experiment_app",
        {**params, "hidden_units": 128},
    )
    assert a != c


def test_duplicate_config_marked_already_evaluated():
    source = {
        "learning_rate": 0.001,
        "epochs": 30,
        "hidden_units": 96,
        "batch_size": 64,
        "test_size": 0.2,
    }
    tested_nodes = [
        {
            "node_id": "node_004",
            "environment_key": "digits-mlp-v1",
            "dataset_reference": "sklearn:digits",
            "code_reference": "local:experiment_app",
            "parameters": {
                "learning_rate": 0.001,
                "epochs": 30,
                "hidden_units": 128,
                "batch_size": 64,
                "test_size": 0.2,
            },
        }
    ]
    tested = collect_tested_parameter_index(tested_nodes)
    recs = [
        ExperimentRecommendation(
            recommendation_type="expand_range",
            priority=0.8,
            rationale="Try 128 again.",
            parameter_changes={"hidden_units": 128},
        ),
        ExperimentRecommendation(
            recommendation_type="intermediate_value",
            priority=0.9,
            rationale="Try 112.",
            parameter_changes={"hidden_units": 112},
        ),
    ]
    out = annotate_recommendations_against_tested(
        recs,
        source_parameters=source,
        tested_index=tested,
        tested_nodes=tested_nodes,
        source_environment_key="digits-mlp-v1",
        source_dataset_reference="sklearn:digits",
        source_code_reference="local:experiment_app",
    )
    by_units = {item.parameter_changes["hidden_units"]: item for item in out}
    assert by_units[128].status == "already_evaluated"
    assert by_units[128].already_evaluated_node_id == "node_004"
    assert by_units[112].status == "active"


def test_build_next_contract_skips_already_evaluated():
    report = FeedbackReport(
        baseline_node_id="node_003",
        candidate_node_id="node_004",
        hypothesis_status="supported_with_repeated_evidence",
        evidence_strength="moderate",
        positive_findings=["up"],
        negative_findings=[],
        tradeoffs=[],
        uncertainties=[],
        recommendations=[
            ExperimentRecommendation(
                recommendation_type="expand_range",
                priority=0.9,
                rationale="128 again",
                parameter_changes={"hidden_units": 128},
                status="already_evaluated",
                already_evaluated_node_id="node_004",
            ),
            ExperimentRecommendation(
                recommendation_type="intermediate_value",
                priority=0.85,
                rationale="Try 96",
                parameter_changes={"hidden_units": 96},
                status="active",
            ),
        ],
        recommended_action="continue",
        comparison_summary={},
    )
    selected = select_proposal_recommendation(report)
    assert selected is not None
    assert selected["parameter_changes"]["hidden_units"] == 96

    proposal = build_next_contract(
        baseline_contract={
            "project_id": "project_001",
            "node_id": "node_003",
            "parameters": {
                "learning_rate": 0.001,
                "epochs": 30,
                "hidden_units": 64,
                "batch_size": 64,
                "test_size": 0.2,
            },
        },
        candidate_contract={
            "project_id": "project_001",
            "node_id": "node_004",
            "environment_key": "digits-mlp-v1",
            "dataset_reference": "sklearn:digits",
            "code_reference": "local:experiment_app",
            "parameters": {
                "learning_rate": 0.001,
                "epochs": 30,
                "hidden_units": 128,
                "batch_size": 64,
                "test_size": 0.2,
            },
        },
        feedback=report,
        existing_node_ids={"node_003", "node_004"},
    )
    assert proposal is not None
    assert proposal["contract"]["parameters"]["hidden_units"] == 96
    assert proposal["contract"]["node_id"] == "node_005"


def test_canonicalize_still_works_as_legacy_key():
    params = {"hidden_units": 128, "epochs": 30, "learning_rate": 0.001}
    assert "hidden_units" in canonicalize_parameters(params)
