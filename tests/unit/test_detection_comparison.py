from __future__ import annotations

from pathlib import Path

import pytest

from scientist_lab.domain import JobStatus, NodeStage, NodeStatus, NodeType, ProjectStatus
from scientist_lab.domain.models import ExecutionAttempt, ExperimentNode, ResearchProject
from scientist_lab.services.detection_comparison import (
    build_metric_relations,
    build_resource_relations,
    classify_higher_is_better,
    classify_lower_is_better,
    enrich_group_comparison,
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


def _detection_attempt(
    *,
    execution_id: str,
    node_id: str,
    seed: int,
    map50_95: float,
    map50: float,
    ap_small: float,
    precision: float,
    recall: float,
    duration: float,
    peak_gpu: float,
    params: int,
    input_mode: str,
    fusion_method: str,
    attempt_index: int,
) -> ExecutionAttempt:
    return ExecutionAttempt(
        execution_id=execution_id,
        node_id=node_id,
        attempt_index=attempt_index,
        runner_profile="local",
        status=JobStatus.COMPLETED,
        image_reference="scientist-rgbt-detection:v2",
        code_version="image:rgbt-detection-v2",
        dataset_version="dataset:rgbt_fast_eval_v1",
        result_json={
            "metrics": {
                "primary_metric": "mAP50_95",
                "metrics": {
                    "mAP50_95": map50_95,
                    "mAP50": map50,
                    "AP_small": ap_small,
                    "precision": precision,
                    "recall": recall,
                    "duration_seconds": duration,
                    "peak_gpu_memory_mb": peak_gpu,
                    "parameter_count": float(params),
                },
                "claim_level": "exploratory_comparison",
            },
            "contract": {
                "seed": seed,
                "protocol_id": "protocol_rgbt_001",
                "dataset_reference": "dataset:rgbt_fast_eval_v1",
                "code_reference": "image:rgbt-detection-v2",
                "environment_key": "rgbt-detection-v2",
                "entrypoint": "run_detection_experiment.py",
                "execution_mode": "fast_eval",
                "task_config": {
                    "claim_level": "exploratory_comparison",
                    "primary_metric": "mAP50_95",
                },
                "parameters": {
                    "baseline": "dfine_s",
                    "input_mode": input_mode,
                    "fusion_method": fusion_method,
                    "epochs": 5,
                    "batch_size": 4,
                    "learning_rate": 0.0001,
                },
            },
        },
        created_at=f"2026-01-01T00:00:{attempt_index:02d}+00:00",
        completed_at=f"2026-01-01T00:01:{attempt_index:02d}+00:00",
    )


def _bootstrap_detection_nodes(service: ExperimentService) -> None:
    now = "2026-01-01T00:00:00+00:00"
    service.repo.upsert_project(
        ResearchProject(
            project_id="project_rgbt_003",
            title="formal",
            research_goal="g",
            status=ProjectStatus.ACTIVE,
            created_at=now,
            updated_at=now,
        )
    )
    # RGB weaker / cheaper; Fusion stronger / more expensive
    specs = {
        "rgbt_formal_node_001": {
            "input_mode": "rgb",
            "fusion_method": "none",
            "maps": [0.10, 0.11, 0.09],
            "map50": [0.20, 0.21, 0.19],
            "ap_small": [0.05, 0.06, 0.04],
            "precision": [0.40, 0.41, 0.39],
            "recall": [0.30, 0.31, 0.29],
            "duration": [10.0, 11.0, 9.0],
            "peak": [100.0, 100.0, 100.0],
            "params": 1000,
        },
        "rgbt_formal_node_003": {
            "input_mode": "rgbt",
            "fusion_method": "early_concat",
            "maps": [0.12, 0.13, 0.11],
            "map50": [0.22, 0.23, 0.21],
            "ap_small": [0.07, 0.08, 0.06],
            "precision": [0.38, 0.39, 0.37],  # slightly worse
            "recall": [0.32, 0.33, 0.31],
            "duration": [20.0, 21.0, 19.0],
            "peak": [200.0, 200.0, 200.0],
            "params": 1500,
        },
    }
    for node_id, spec in specs.items():
        service.repo.upsert_node(
            ExperimentNode(
                node_id=node_id,
                project_id="project_rgbt_003",
                node_type=NodeType.BASELINE,
                stage=NodeStage.DONE,
                status=NodeStatus.SUCCEEDED,
                depth=0,
                contract_json={},
                created_at=now,
                updated_at=now,
            )
        )
        for index, seed in enumerate([42, 43, 44], start=1):
            i = index - 1
            service.repo.upsert_attempt(
                _detection_attempt(
                    execution_id=f"exec_{node_id}_{seed}",
                    node_id=node_id,
                    seed=seed,
                    map50_95=spec["maps"][i],
                    map50=spec["map50"][i],
                    ap_small=spec["ap_small"][i],
                    precision=spec["precision"][i],
                    recall=spec["recall"][i],
                    duration=spec["duration"][i],
                    peak_gpu=spec["peak"][i],
                    params=spec["params"],
                    input_mode=spec["input_mode"],
                    fusion_method=spec["fusion_method"],
                    attempt_index=index,
                )
            )


def test_classify_helpers():
    assert classify_higher_is_better(0.1, 0.12) == "candidate_better"
    assert classify_higher_is_better(0.12, 0.1) == "baseline_better"
    assert classify_higher_is_better(0.1, 0.1) == "equivalent"
    assert classify_higher_is_better(None, 0.1) == "missing"
    assert classify_lower_is_better(10.0, 20.0) == "candidate_worse"
    assert classify_lower_is_better(20.0, 10.0) == "candidate_better"


def test_build_relations_from_aggregates():
    baseline = {
        "aggregate_metrics": {
            "mAP50_95": {"mean": 0.10},
            "mAP50": {"mean": 0.20},
            "AP_small": {"mean": 0.05},
            "precision": {"mean": 0.40},
            "recall": {"mean": 0.30},
            "duration_seconds": {"mean": 10.0},
            "peak_gpu_memory_mb": {"mean": 100.0},
            "parameter_count": {"mean": 1000.0},
        }
    }
    candidate = {
        "aggregate_metrics": {
            "mAP50_95": {"mean": 0.12},
            "mAP50": {"mean": 0.22},
            "AP_small": {"mean": 0.07},
            "precision": {"mean": 0.38},
            "recall": {"mean": 0.32},
            "duration_seconds": {"mean": 20.0},
            "peak_gpu_memory_mb": {"mean": 200.0},
            "parameter_count": {"mean": 1500.0},
        }
    }
    metrics = build_metric_relations(baseline, candidate)
    resources = build_resource_relations(baseline, candidate)
    assert metrics["mAP50_95"] == "candidate_better"
    assert metrics["precision"] == "baseline_better"
    assert resources["duration_seconds"] == "candidate_worse"
    assert resources["parameter_count"] == "candidate_worse"


def test_compare_node_groups_emits_detection_relations(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap_detection_nodes(service)
    result = service.compare_node_groups(
        "rgbt_formal_node_001", "rgbt_formal_node_003"
    )
    assert result["primary_metric"] == "mAP50_95"
    assert result["paired_win_count"] == 3
    assert result["paired_loss_count"] == 0
    assert result["mean_delta"] == pytest.approx(0.02)
    assert result["metric_relations"]["mAP50_95"] == "candidate_better"
    assert result["metric_relations"]["AP_small"] == "candidate_better"
    assert result["metric_relations"]["precision"] == "baseline_better"
    assert result["resource_relations"]["duration_seconds"] == "candidate_worse"
    assert result["resource_relations"]["peak_gpu_memory_mb"] == "candidate_worse"
    assert result["resource_relations"]["parameter_count"] == "candidate_worse"
    assert result["claim_level"] == "exploratory_comparison"
    assert result["comparison_schema"] == "detection_group_v0_9_3"


def test_missing_resource_metrics_warn(tmp_path: Path):
    enriched = enrich_group_comparison(
        {
            "candidate_win_count": 2,
            "baseline_win_count": 1,
            "verification": {"valid": True, "warnings": [], "blocking_issues": []},
            "baseline_aggregate": {
                "aggregate_metrics": {"mAP50_95": {"mean": 0.1}},
            },
            "candidate_aggregate": {
                "aggregate_metrics": {"mAP50_95": {"mean": 0.12}},
            },
        }
    )
    assert enriched["resource_relations"]["duration_seconds"] == "missing"
    assert any(
        "duration_seconds" in w for w in enriched["verification"]["warnings"]
    )
