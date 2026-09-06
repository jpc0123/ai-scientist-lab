from __future__ import annotations

from pathlib import Path

from scientist_lab.domain import JobStatus, NodeStage, NodeStatus, NodeType, ProjectStatus
from scientist_lab.domain.models import ExecutionAttempt, ExperimentNode, ResearchProject
from scientist_lab.services.aggregation import (
    NodeAggregationService,
    compare_node_groups,
)
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings


def _service(tmp_path: Path) -> ExperimentService:
    root = Path(__file__).resolve().parents[2]
    settings = Settings(
        project_root=root,
        db_path=tmp_path / "test.db",
        runtime_dir=tmp_path / "runtime",
        outputs_dir=tmp_path / "outputs",
        experiment_app_dir=root / "experiment_app",
    ).resolve()
    return ExperimentService(settings=settings)


def _seed_attempt(
    *,
    execution_id: str,
    node_id: str,
    seed: int,
    accuracy: float,
    hidden_units: int,
    attempt_index: int,
) -> ExecutionAttempt:
    return ExecutionAttempt(
        execution_id=execution_id,
        node_id=node_id,
        attempt_index=attempt_index,
        runner_profile="local",
        status=JobStatus.COMPLETED,
        image_reference="scientist-experiment:v2",
        code_version="local:experiment_app",
        dataset_version="sklearn:digits",
        result_json={
            "metrics": {
                "primary_metric": "accuracy",
                "metrics": {
                    "accuracy": accuracy,
                    "macro_f1": accuracy - 0.001,
                    "log_loss": 0.2 - accuracy * 0.05,
                },
            },
            "contract": {
                "seed": seed,
                "dataset_reference": "sklearn:digits",
                "code_reference": "local:experiment_app",
                "environment_key": "digits-mlp-v1",
                "entrypoint": "run_experiment.py",
                "parameters": {
                    "learning_rate": 0.001,
                    "epochs": 30,
                    "hidden_units": hidden_units,
                    "batch_size": 64,
                    "test_size": 0.2,
                },
            },
        },
        created_at=f"2026-01-01T00:00:{attempt_index:02d}+00:00",
        completed_at=f"2026-01-01T00:01:{attempt_index:02d}+00:00",
    )


def _bootstrap_nodes(service: ExperimentService) -> None:
    now = "2026-01-01T00:00:00+00:00"
    service.repo.upsert_project(
        ResearchProject(
            project_id="project_001",
            title="t",
            research_goal="g",
            status=ProjectStatus.ACTIVE,
            created_at=now,
            updated_at=now,
        )
    )
    for node_id, hidden in (("node_003", 64), ("node_004", 128)):
        service.repo.upsert_node(
            ExperimentNode(
                node_id=node_id,
                project_id="project_001",
                node_type=NodeType.BASELINE,
                stage=NodeStage.DONE,
                status=NodeStatus.SUCCEEDED,
                depth=0,
                contract_json={},
                created_at=now,
                updated_at=now,
            )
        )
        accuracies = {
            "node_003": [0.95, 0.951, 0.949, 0.952, 0.948],
            "node_004": [0.956, 0.957, 0.955, 0.958, 0.954],
        }[node_id]
        for index, (seed, acc) in enumerate(
            zip([42, 43, 44, 45, 46], accuracies, strict=True), start=1
        ):
            service.repo.upsert_attempt(
                _seed_attempt(
                    execution_id=f"exec_{node_id}_{seed}",
                    node_id=node_id,
                    seed=seed,
                    accuracy=acc,
                    hidden_units=hidden,
                    attempt_index=index,
                )
            )


def test_aggregate_node_metrics(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap_nodes(service)
    agg = service.aggregate_node("node_003")
    assert agg["seed_count"] == 5
    assert agg["seeds"] == [42, 43, 44, 45, 46]
    assert "accuracy" in agg["aggregate_metrics"]
    assert agg["aggregate_metrics"]["accuracy"]["std"] > 0
    path = Path(agg["aggregate_path"])
    assert path.exists()
    node = service.repo.get_node("node_003")
    assert node is not None
    assert node.feedback_json is not None
    assert "aggregate_metrics" in node.feedback_json


def test_compare_node_groups_stable_improvement(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap_nodes(service)
    result = service.compare_node_groups("node_003", "node_004")
    assert result["shared_seeds"] == [42, 43, 44, 45, 46]
    assert result["candidate_win_count"] == 5
    assert result["stable_improvement"] is True
    assert result["hypothesis_status"] == "supported_with_repeated_evidence"
    assert result["verification"]["valid"] is True


def test_compare_node_groups_invalid_with_one_shared_seed(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap_nodes(service)
    # Keep only seed 42 on candidate
    for seed in (43, 44, 45, 46):
        row = service.repo.get_attempt(f"exec_node_004_{seed}")
        assert row is not None
        row.status = JobStatus.FAILED
        row.result_json = {"contract": {"seed": seed}}
        service.repo.upsert_attempt(row)

    aggregation = NodeAggregationService(service.repo, service.settings.outputs_dir)
    result = compare_node_groups(aggregation, "node_003", "node_004")
    assert result["hypothesis_status"] == "invalid_comparison"
    assert result["verification"]["valid"] is False
