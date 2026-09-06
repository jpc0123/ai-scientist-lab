from __future__ import annotations

from scientist_lab.domain import JobStatus
from scientist_lab.domain.models import ExecutionAttempt
from scientist_lab.services.verifier import verify_fair_comparison


def _attempt(
    *,
    execution_id: str,
    node_id: str,
    status: JobStatus = JobStatus.COMPLETED,
    seed: int = 42,
    hidden_units: int = 64,
    epochs: int = 30,
    accuracy: float = 0.95,
    primary: str = "accuracy",
) -> ExecutionAttempt:
    return ExecutionAttempt(
        execution_id=execution_id,
        node_id=node_id,
        attempt_index=1,
        runner_profile="local",
        status=status,
        image_reference="scientist-experiment:v2",
        code_version="local:experiment_app",
        dataset_version="sklearn:digits",
        result_json={
            "metrics": {
                "primary_metric": primary,
                "metrics": {"accuracy": accuracy, "macro_f1": accuracy - 0.01},
                "training": {"duration_seconds": 1.2},
            },
            "contract": {
                "seed": seed,
                "dataset_reference": "sklearn:digits",
                "code_reference": "local:experiment_app",
                "environment_key": "digits-mlp-v1",
                "entrypoint": "run_experiment.py",
                "parameters": {
                    "learning_rate": 0.001,
                    "epochs": epochs,
                    "hidden_units": hidden_units,
                    "batch_size": 64,
                    "test_size": 0.2,
                },
            },
        },
        created_at="2026-01-01T00:00:00+00:00",
    )


def test_verifier_accepts_single_variable_change():
    a = _attempt(execution_id="exec_a", node_id="node_003", hidden_units=64)
    b = _attempt(execution_id="exec_b", node_id="node_004", hidden_units=128)
    report = verify_fair_comparison(a, b)
    assert report.valid is True
    assert report.blocking_issues == []
    assert any("random seed" in w.lower() for w in report.warnings)


def test_verifier_blocks_different_seed():
    a = _attempt(execution_id="exec_a", node_id="node_003", seed=42)
    b = _attempt(execution_id="exec_b", node_id="node_005", seed=99)
    report = verify_fair_comparison(a, b)
    assert report.valid is False
    assert any("seed differs" in issue for issue in report.blocking_issues)


def test_verifier_blocks_multi_variable_change():
    a = _attempt(execution_id="exec_a", node_id="node_003", hidden_units=64, epochs=30)
    b = _attempt(
        execution_id="exec_b", node_id="node_004", hidden_units=128, epochs=40
    )
    report = verify_fair_comparison(a, b)
    assert report.valid is False
    assert any("single-variable" in issue for issue in report.blocking_issues)


def test_verifier_blocks_failed_status():
    a = _attempt(execution_id="exec_a", node_id="node_003")
    b = _attempt(
        execution_id="exec_b",
        node_id="node_004",
        status=JobStatus.FAILED,
        hidden_units=128,
    )
    report = verify_fair_comparison(a, b)
    assert report.valid is False
    assert any("Candidate status" in issue for issue in report.blocking_issues)
