from scientist_lab.domain.contracts import ExperimentContract


def test_contract_defaults():
    contract = ExperimentContract(
        project_id="p1",
        node_id="n1",
        environment_key="scientist-experiment-v1",
        code_reference="local:experiment_app",
        dataset_reference="debug-dataset-v1",
        parameters={"learning_rate": 0.001},
    )
    assert contract.runner_profile == "local"
    assert "metrics.json" in contract.expected_outputs
