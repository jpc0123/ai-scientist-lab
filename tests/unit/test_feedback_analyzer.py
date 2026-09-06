from __future__ import annotations

from pathlib import Path

from scientist_lab.domain import JobStatus, NodeStage, NodeStatus, NodeType, ProjectStatus
from scientist_lab.domain.feedback import FeedbackInput
from scientist_lab.domain.models import ExecutionAttempt, ExperimentNode, ResearchProject
from scientist_lab.services.feedback_analyzer import analyze_feedback
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.services.proposal import build_next_contract
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
    duration: float = 0.1,
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
                    "log_loss": 0.20 - accuracy * 0.05,
                },
                "training": {"duration_seconds": duration},
            },
            "contract": {
                "schema_version": "1.0",
                "project_id": "project_001",
                "node_id": node_id,
                "title": f"Digits MLP hidden units {hidden_units}",
                "research_goal": "Validate digits MLP",
                "seed": seed,
                "dataset_reference": "sklearn:digits",
                "code_reference": "local:experiment_app",
                "environment_key": "digits-mlp-v1",
                "entrypoint": "run_experiment.py",
                "execution_mode": "fast_eval",
                "parameters": {
                    "learning_rate": 0.001,
                    "epochs": 30,
                    "hidden_units": hidden_units,
                    "batch_size": 64,
                    "test_size": 0.2,
                },
                "resources": {
                    "gpu_count": 0,
                    "cpu_count": 2,
                    "memory_gb": 2,
                    "timeout_seconds": 300,
                },
            },
        },
        created_at=f"2026-01-01T00:00:{attempt_index:02d}+00:00",
        completed_at=f"2026-01-01T00:01:{attempt_index:02d}+00:00",
    )


def _bootstrap(
    service: ExperimentService,
    *,
    candidate_acc: list[float] | None = None,
    candidate_duration: float = 0.30,
) -> None:
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
    base_acc = [0.950, 0.951, 0.949, 0.952, 0.948]
    cand_acc = candidate_acc or [0.956, 0.957, 0.955, 0.958, 0.954]
    for node_id, hidden, accuracies, duration in (
        ("node_003", 64, base_acc, 0.10),
        ("node_004", 128, cand_acc, candidate_duration),
    ):
        service.repo.upsert_node(
            ExperimentNode(
                node_id=node_id,
                project_id="project_001",
                node_type=NodeType.BASELINE,
                stage=NodeStage.DONE,
                status=NodeStatus.SUCCEEDED,
                depth=0,
                contract_json={
                    "project_id": "project_001",
                    "node_id": node_id,
                    "parameters": {"hidden_units": hidden, "epochs": 30},
                },
                created_at=now,
                updated_at=now,
            )
        )
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
                    duration=duration,
                )
            )


def test_supported_feedback_recommends_intermediate_and_expand(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap(service, candidate_duration=0.30)
    report = service.analyze_feedback("node_003", "node_004")
    assert report["hypothesis_status"] == "supported_with_repeated_evidence"
    assert report["evidence_strength"] == "moderate"
    assert report["recommended_action"] == "continue"
    types = {item["recommendation_type"] for item in report["recommendations"]}
    assert "intermediate_value" in types or "efficiency_tradeoff" in types
    assert "expand_range" in types
    # With only endpoints 64/128 known, midpoint 96 stays active for propose-next.
    assert any(
        item.get("parameter_changes", {}).get("hidden_units") == 96
        and item.get("status", "active") == "active"
        for item in report["recommendations"]
    )
    assert Path(report["feedback_path"]).exists()


def test_duration_tradeoff_when_time_jumps(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap(service, candidate_duration=0.40)  # +300%
    report = service.analyze_feedback("node_003", "node_004")
    assert report["tradeoffs"]
    assert any(
        item["recommendation_type"] == "efficiency_tradeoff"
        for item in report["recommendations"]
    )


def test_rejected_feedback_suggests_revert(tmp_path: Path):
    service = _service(tmp_path)
    worse = [0.940, 0.941, 0.939, 0.942, 0.938]
    _bootstrap(service, candidate_acc=worse)
    report = service.analyze_feedback("node_003", "node_004")
    assert report["hypothesis_status"] == "rejected_with_repeated_evidence"
    assert report["recommended_action"] == "revise"
    assert any(
        item["parameter_changes"].get("hidden_units") == 64
        for item in report["recommendations"]
    )


def test_inconclusive_recommends_more_seeds(tmp_path: Path):
    service = _service(tmp_path)
    # Mean gain exists, but candidate loses majority of paired seeds.
    base = [0.950, 0.960, 0.970, 0.940, 0.970]
    cand = [0.955, 0.955, 0.968, 0.965, 0.968]
    _bootstrap(service, candidate_acc=cand)
    # overwrite baseline accuracies
    for seed, acc, index in zip(
        [42, 43, 44, 45, 46], base, [1, 2, 3, 4, 5], strict=True
    ):
        service.repo.upsert_attempt(
            _seed_attempt(
                execution_id=f"exec_node_003_{seed}",
                node_id="node_003",
                seed=seed,
                accuracy=acc,
                hidden_units=64,
                attempt_index=index,
                duration=0.10,
            )
        )
    report = service.analyze_feedback("node_003", "node_004")
    assert report["hypothesis_status"] == "inconclusive"
    assert report.get("comparison_summary", {}).get("practically_equivalent") is not True
    assert any(
        item["recommendation_type"] == "repeat_seeds"
        for item in report["recommendations"]
    )
    assert any(
        item["parameter_changes"].get("hidden_units") == 96
        for item in report["recommendations"]
    )
    assert any(
        item["parameter_changes"].get("hidden_units") == 96
        and item.get("status", "active") == "active"
        for item in report["recommendations"]
    )


def test_practically_equivalent_efficiency_tradeoff(tmp_path: Path):
    service = _service(tmp_path)
    # Nearly identical accuracy; candidate slightly faster but less stable.
    same = [0.9661, 0.9662, 0.9663, 0.9664, 0.9665]
    _bootstrap(
        service,
        candidate_acc=same,
        candidate_duration=0.23,
    )
    for seed, acc, index in zip(
        [42, 43, 44, 45, 46], same, [1, 2, 3, 4, 5], strict=True
    ):
        service.repo.upsert_attempt(
            _seed_attempt(
                execution_id=f"exec_node_003_{seed}",
                node_id="node_003",
                seed=seed,
                accuracy=acc,
                hidden_units=128,
                attempt_index=index,
                duration=0.26,
            )
        )
        # Make baseline more stable by using flatter values; candidate use more spread
    spread = [0.961, 0.972, 0.960, 0.970, 0.970]
    for seed, acc, index in zip(
        [42, 43, 44, 45, 46], spread, [1, 2, 3, 4, 5], strict=True
    ):
        service.repo.upsert_attempt(
            _seed_attempt(
                execution_id=f"exec_node_004_{seed}",
                node_id="node_004",
                seed=seed,
                accuracy=acc,
                hidden_units=96,
                attempt_index=index,
                duration=0.23,
            )
        )

    cmp = service.compare_node_groups("node_003", "node_004")
    # Force names: treat node_003 as "128-like baseline" with flat accuracy for std
    # Re-seed baseline flat for lower std
    flat = [0.9666, 0.9667, 0.9665, 0.9666, 0.9667]
    for seed, acc, index in zip(
        [42, 43, 44, 45, 46], flat, [1, 2, 3, 4, 5], strict=True
    ):
        service.repo.upsert_attempt(
            _seed_attempt(
                execution_id=f"exec_node_003_{seed}",
                node_id="node_003",
                seed=seed,
                accuracy=acc,
                hidden_units=128,
                attempt_index=index,
                duration=0.26,
            )
        )

    cmp = service.compare_node_groups("node_003", "node_004")
    assert cmp["practically_equivalent"] is True
    assert cmp["efficiency_relation"] == "candidate_better"
    assert cmp["stability_relation"] == "baseline_better"
    assert cmp["tradeoff_status"] == "candidate_is_efficiency_tradeoff"

    report = service.analyze_feedback("node_003", "node_004")
    assert report["tradeoffs"]
    assert any("practically equivalent" in x.lower() for x in report["positive_findings"])
    assert not any(
        item["parameter_changes"].get("hidden_units") in {112, 192, 256}
        for item in report["recommendations"]
    )
    assert any(
        item["recommendation_type"] == "efficiency_tradeoff"
        for item in report["recommendations"]
    )


def test_invalid_comparison_blocks_performance_claims(tmp_path: Path):
    comparison = {
        "hypothesis_status": "invalid_comparison",
        "verification": {"valid": False, "blocking_issues": ["seed differs"], "warnings": []},
        "shared_seeds": [],
        "mean_delta": 0.01,
        "parameter_changes": {},
    }
    report = analyze_feedback(
        FeedbackInput(
            baseline_node_id="node_003",
            candidate_node_id="node_004",
            comparison=comparison,
            baseline_contract={},
            candidate_contract={},
        )
    )
    assert report.hypothesis_status == "invalid_comparison"
    assert report.recommended_action == "revise"
    assert not any(
        "improved" in finding.lower() for finding in report.positive_findings
    )


def test_propose_next_keeps_controls_and_sets_parent(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap(service)
    proposal = service.propose_next("node_003", "node_004")
    contract = proposal["contract"]
    assert contract["node_id"] == "node_005"
    assert contract["parent_node_id"] == "node_004"
    assert contract["parameters"]["hidden_units"] == 96
    assert contract["parameters"]["epochs"] == 30
    assert contract["parameters"]["learning_rate"] == 0.001
    assert contract["dataset_reference"] == "sklearn:digits"
    assert Path(proposal["contract_path"]).exists()


def test_no_parameter_change_skips_contract():
    from scientist_lab.domain.feedback import ExperimentRecommendation, FeedbackReport

    report = FeedbackReport(
        baseline_node_id="a",
        candidate_node_id="b",
        hypothesis_status="inconclusive",
        evidence_strength="weak",
        recommendations=[
            ExperimentRecommendation(
                recommendation_type="repeat_seeds",
                priority=0.9,
                rationale="Need more seeds",
                parameter_changes={},
            )
        ],
        recommended_action="verify",
    )
    assert (
        build_next_contract(
            baseline_contract={"parameters": {"hidden_units": 64}},
            candidate_contract={"parameters": {"hidden_units": 64}},
            feedback=report,
            existing_node_ids=set(),
        )
        is None
    )


def test_aggregation_policy_and_std_metadata(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap(service)
    agg = service.aggregate_node("node_003")
    assert agg["aggregation_policy"] == "latest_success_per_seed"
    assert agg["aggregate_metrics"]["accuracy"]["std_type"] == "sample"
    assert agg["aggregate_metrics"]["accuracy"]["ddof"] == 1
    assert all("execution_id" in item for item in agg["executions"])
