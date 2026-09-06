from __future__ import annotations

from pathlib import Path

from scientist_lab.domain import JobStatus, NodeStage, NodeStatus, NodeType, ProjectStatus
from scientist_lab.domain.models import (
    ExecutionAttempt,
    ExperimentArtifact,
    ExperimentNode,
    ResearchProject,
)
from scientist_lab.evidence.claim_gate import assess_evidence_strength
from scientist_lab.evidence.builder import build_paired_comparison_evidence
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
    map50_95: float,
    duration: float,
    attempt_index: int,
    input_mode: str,
    fusion_method: str,
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
                    "mAP50": map50_95 + 0.1,
                    "AP_small": map50_95 / 2,
                    "precision": 0.4,
                    "recall": 0.3,
                    "duration_seconds": duration,
                    "peak_gpu_memory_mb": 100.0 if input_mode == "rgb" else 200.0,
                    "parameter_count": 1000.0 if input_mode == "rgb" else 1500.0,
                },
            },
            "contract": {
                "seed": seed,
                "project_id": "project_rgbt_003",
                "protocol_id": "protocol_rgbt_001",
                "dataset_reference": "dataset:rgbt_fast_eval_v1",
                "code_reference": "image:rgbt-detection-v2",
                "environment_key": "rgbt-detection-v2",
                "entrypoint": "run_detection_experiment.py",
                "execution_mode": "fast_eval",
                "task_config": {
                    "claim_level": "exploratory_comparison",
                    "evaluation_scope": "fast_eval_subset",
                    "implementation": "stand_in",
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


def _bootstrap(service: ExperimentService) -> None:
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
    specs = {
        "rgbt_formal_node_001": {
            "input_mode": "rgb",
            "fusion_method": "none",
            "maps": [0.10, 0.11, 0.09],
            "duration": [10.0, 11.0, 9.0],
        },
        "rgbt_formal_node_003": {
            "input_mode": "rgbt",
            "fusion_method": "early_concat",
            "maps": [0.12, 0.13, 0.11],
            "duration": [20.0, 21.0, 19.0],
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
            attempt = _seed_attempt(
                execution_id=f"exec_{node_id}_{seed}",
                node_id=node_id,
                seed=seed,
                map50_95=spec["maps"][index - 1],
                duration=spec["duration"][index - 1],
                attempt_index=index,
                input_mode=spec["input_mode"],
                fusion_method=spec["fusion_method"],
            )
            service.repo.upsert_attempt(attempt)
            service.repo.add_artifact(
                ExperimentArtifact(
                    artifact_id=f"art_{node_id}_{seed}",
                    execution_id=attempt.execution_id,
                    artifact_type="metrics",
                    relative_path="metrics.json",
                    size_bytes=12,
                    sha256="abc",
                    created_at=now,
                )
            )


def test_fast_eval_standin_cannot_be_strong():
    result = assess_evidence_strength(
        seed_count=5,
        execution_mode="fast_eval",
        evaluation_scope="fast_eval_subset",
        implementation="stand_in",
        claim_level="exploratory_comparison",
        has_protocol=True,
        has_ablation=True,
        formal_implementation=False,
        stable_direction=True,
    )
    assert result["evidence_strength"] == "weak"
    assert result["allows_strong"] is False
    assert any("Stand-in" in item for item in result["limitations"])


def test_single_seed_cannot_be_repeated_experiment():
    result = assess_evidence_strength(
        seed_count=1,
        requested_type="repeated_experiment",
        execution_mode="fast_eval",
        implementation="stand_in",
    )
    assert result["effective_evidence_type"] == "single_execution"


def test_build_evidence_traces_execution_and_artifact(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap(service)
    payload = service.build_evidence(
        "rgbt_formal_node_001", "rgbt_formal_node_003"
    )
    assert len(payload["evidence_ids"]) == 2
    paired = payload["records"][0]
    assert paired["evidence_type"] == "paired_comparison"
    assert paired["evidence_strength"] == "weak"
    assert paired["protocol_id"] == "protocol_rgbt_001"
    assert set(paired["source_node_ids"]) == {
        "rgbt_formal_node_001",
        "rgbt_formal_node_003",
    }
    assert len(paired["source_execution_ids"]) == 6
    # Seeded metrics artifacts plus any node_aggregate artifacts from aggregation.
    assert len(paired["source_artifact_ids"]) >= 6
    for seed in (42, 43, 44):
        assert f"art_rgbt_formal_node_001_{seed}" in paired["source_artifact_ids"]
        assert f"art_rgbt_formal_node_003_{seed}" in paired["source_artifact_ids"]
    assert paired["metric_summary"]["seed_count"] == 3
    assert "stand_in" in " ".join(paired["limitations"]).lower() or any(
        "Stand-in" in item for item in paired["limitations"]
    )

    evidence_id = paired["evidence_id"]
    shown = service.show_evidence(evidence_id)
    assert shown["evidence_id"] == evidence_id
    assert Path(shown["evidence_path"]).is_file()

    listed = service.list_evidence(project_id="project_rgbt_003")
    assert len(listed) == 2
    assert any(item["evidence_type"] == "resource_comparison" for item in listed)


def test_standin_evidence_blocks_formal_dfine_claim_language():
    comparison = {
        "baseline_node_id": "a",
        "candidate_node_id": "b",
        "shared_seeds": [42, 43, 44],
        "primary_metric": "mAP50_95",
        "mean_delta": 0.02,
        "paired_win_count": 2,
        "paired_loss_count": 1,
        "stable_improvement": True,
        "claim_level": "exploratory_comparison",
        "paired_deltas": [
            {
                "seed": 42,
                "baseline_execution_id": "e1",
                "candidate_execution_id": "e2",
            }
        ],
        "verification": {"valid": True},
        "metric_relations": {"mAP50_95": "candidate_better"},
        "resource_relations": {"duration_seconds": "candidate_worse"},
    }
    record = build_paired_comparison_evidence(
        comparison,
        project_id="project_rgbt_003",
        sample_contract={
            "protocol_id": "protocol_rgbt_001",
            "execution_mode": "fast_eval",
            "task_config": {
                "implementation": "stand_in",
                "evaluation_scope": "fast_eval_subset",
                "claim_level": "exploratory_comparison",
            },
        },
        source_artifact_ids=["art_1"],
    )
    assert record.evidence_strength == "weak"
    assert record.source_artifact_ids == ["art_1"]
    assert any("DFINE" in item for item in record.limitations)
