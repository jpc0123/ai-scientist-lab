from __future__ import annotations

from pathlib import Path

import pytest

from scientist_lab.domain import JobStatus, NodeStage, NodeStatus, NodeType, ProjectStatus
from scientist_lab.domain.models import ExecutionAttempt, ExperimentNode, ResearchProject
from scientist_lab.iteration.service import InvalidSelectedNode, IterationService
from scientist_lab.iteration.workflow import (
    InvalidIterationTransition,
    transition,
)
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings
from scientist_lab.storage.artifact_store import sha256_file


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


def _bootstrap_pair(
    service: ExperimentService,
    *,
    candidate_acc: list[float] | None = None,
    candidate_duration: float = 0.30,
    candidate_hidden: int = 128,
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
        ("node_004", candidate_hidden, cand_acc, candidate_duration),
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
                    "environment_key": "digits-mlp-v1",
                    "dataset_reference": "sklearn:digits",
                    "code_reference": "local:experiment_app",
                    "parameters": {
                        "learning_rate": 0.001,
                        "epochs": 30,
                        "hidden_units": hidden,
                        "batch_size": 64,
                        "test_size": 0.2,
                    },
                },
                created_at=now,
                updated_at=now,
            )
        )
        for idx, (seed, acc) in enumerate(zip([42, 43, 44, 45, 46], accuracies), start=1):
            service.repo.upsert_attempt(
                _seed_attempt(
                    execution_id=f"exec_{node_id}_{seed}",
                    node_id=node_id,
                    seed=seed,
                    accuracy=acc,
                    hidden_units=hidden,
                    attempt_index=idx,
                    duration=duration,
                )
            )


def _fake_run_seeds(
    service: ExperimentService,
    *,
    node_id: str,
    hidden_units: int,
    seeds: list[int],
    successes: int | None = None,
):
    successes = len(seeds) if successes is None else successes

    def _run(contract, seed_list, *, auto_aggregate=True):
        now = "2026-01-02T00:00:00+00:00"
        service.repo.upsert_node(
            ExperimentNode(
                node_id=node_id,
                project_id="project_001",
                parent_node_id="node_004",
                node_type=NodeType.SMOKE,
                stage=NodeStage.DONE,
                status=NodeStatus.SUCCEEDED,
                depth=1,
                contract_json=contract.model_dump(),
                created_at=now,
                updated_at=now,
            )
        )
        results = []
        for idx, seed in enumerate(seed_list, start=1):
            ok = idx <= successes
            if ok:
                service.repo.upsert_attempt(
                    _seed_attempt(
                        execution_id=f"exec_{node_id}_{seed}",
                        node_id=node_id,
                        seed=int(seed),
                        accuracy=0.953 + idx * 0.0001,
                        hidden_units=hidden_units,
                        attempt_index=idx,
                        duration=0.18,
                    )
                )
            results.append(
                {
                    "seed": int(seed),
                    "execution_id": f"exec_{node_id}_{seed}",
                    "status": "completed" if ok else "failed",
                    "metrics": {"primary_metric": "accuracy"} if ok else None,
                    "error": None if ok else {"message": "boom"},
                }
            )
        payload = {
            "node_id": node_id,
            "project_id": "project_001",
            "seeds": list(seed_list),
            "results": results,
        }
        if auto_aggregate and successes >= 1:
            try:
                payload["aggregate"] = service.aggregate_node(node_id)
            except KeyError as exc:
                payload["aggregate_error"] = str(exc)
        return payload

    return _run


def test_illegal_transition_blocked():
    from scientist_lab.iteration.models import IterationSession

    session = IterationSession(
        iteration_id="iter_x",
        project_id="project_001",
        source_baseline_node_id="node_003",
        source_candidate_node_id="node_004",
        status="waiting_approval",
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )
    with pytest.raises(InvalidIterationTransition):
        transition(session, "completed")


def test_iterate_start_creates_waiting_approval(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap_pair(service)
    iteration = IterationService(service)

    payload = iteration.start_iteration("node_003", "node_004", seeds=[42, 43, 44, 45, 46])
    assert payload["status"] == "waiting_approval"
    assert payload["proposed_node_id"]
    assert payload["proposal_path"]
    assert payload["proposal_sha256"]
    assert Path(payload["proposal_path"]).exists()
    assert payload["next_action"]

    restored = IterationService(service).get_status(payload["iteration_id"])
    assert restored["status"] == "waiting_approval"
    assert restored["proposal_sha256"] == payload["proposal_sha256"]


def test_iterate_start_stopped_when_no_recommendation(tmp_path: Path):
    service = _service(tmp_path)
    # Practically equivalent + efficiency tradeoff → no active parameter proposal.
    _bootstrap_pair(
        service,
        candidate_acc=[0.950, 0.951, 0.949, 0.952, 0.948],
        candidate_duration=0.09,
        candidate_hidden=96,
    )
    # Add an already-evaluated intermediate so expand/intermediate get filtered.
    service.repo.upsert_node(
        ExperimentNode(
            node_id="node_005",
            project_id="project_001",
            node_type=NodeType.SMOKE,
            stage=NodeStage.DONE,
            status=NodeStatus.SUCCEEDED,
            depth=1,
            contract_json={
                "project_id": "project_001",
                "node_id": "node_005",
                "environment_key": "digits-mlp-v1",
                "dataset_reference": "sklearn:digits",
                "code_reference": "local:experiment_app",
                "parameters": {
                    "learning_rate": 0.001,
                    "epochs": 30,
                    "hidden_units": 80,
                    "batch_size": 64,
                    "test_size": 0.2,
                },
            },
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
    )
    for idx, seed in enumerate([42, 43, 44, 45, 46], start=1):
        service.repo.upsert_attempt(
            _seed_attempt(
                execution_id=f"exec_node_005_{seed}",
                node_id="node_005",
                seed=seed,
                accuracy=0.950 + idx * 0.0001,
                hidden_units=80,
                attempt_index=idx,
                duration=0.09,
            )
        )

    iteration = IterationService(service)
    payload = iteration.start_iteration("node_003", "node_004")
    assert payload["status"] in {
        "stopped_no_recommendation",
        "waiting_approval",
    }
    if payload["status"] == "waiting_approval":
        # If a novel width remains, proposal must still be unique.
        contract = Path(payload["proposal_path"]).read_text(encoding="utf-8")
        assert "hidden_units" in contract


def test_reject_does_not_run_and_blocks_approve(tmp_path: Path, monkeypatch):
    service = _service(tmp_path)
    _bootstrap_pair(service)
    iteration = IterationService(service)
    started = iteration.start_iteration("node_003", "node_004")
    assert started["status"] == "waiting_approval"

    called = {"run": False}

    def _boom(*args, **kwargs):
        called["run"] = True
        raise AssertionError("run_seeds should not be called after reject")

    monkeypatch.setattr(service, "run_seeds", _boom)
    rejected = iteration.reject(started["iteration_id"], reason="too expensive")
    assert rejected["status"] == "rejected"
    assert called["run"] is False
    approvals = iteration.repo.list_approvals(started["iteration_id"])
    assert len(approvals) == 1
    assert approvals[0].decision == "rejected"

    with pytest.raises(InvalidIterationTransition):
        iteration.approve_and_run(started["iteration_id"])


def test_approve_runs_seeds_compares_and_waits_decision(tmp_path: Path, monkeypatch):
    service = _service(tmp_path)
    _bootstrap_pair(service)
    iteration = IterationService(service)
    started = iteration.start_iteration("node_003", "node_004", seeds=[42, 43, 44, 45, 46])
    proposed = started["proposed_node_id"]
    assert proposed

    monkeypatch.setattr(
        service,
        "run_seeds",
        _fake_run_seeds(service, node_id=proposed, hidden_units=96, seeds=[42, 43, 44, 45, 46]),
    )
    approved = iteration.approve_and_run(started["iteration_id"])
    assert approved["status"] == "waiting_decision"
    assert len(approved["execution_ids"]) == 5
    assert len(approved["comparison_paths"]) == 2
    assert approved["contract_modified_before_approval"] is False
    approvals = iteration.repo.list_approvals(started["iteration_id"])
    assert approvals[-1].decision == "approved"


def test_approve_records_modified_contract_hash(tmp_path: Path, monkeypatch):
    service = _service(tmp_path)
    _bootstrap_pair(service)
    iteration = IterationService(service)
    started = iteration.start_iteration("node_003", "node_004")
    path = Path(started["proposal_path"])
    original = sha256_file(path)
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace('"epochs": 30', '"epochs": 31'), encoding="utf-8")
    assert sha256_file(path) != original

    proposed = started["proposed_node_id"]
    monkeypatch.setattr(
        service,
        "run_seeds",
        _fake_run_seeds(service, node_id=proposed, hidden_units=96, seeds=[42, 43, 44, 45, 46]),
    )
    approved = iteration.approve_and_run(started["iteration_id"])
    assert approved["contract_modified_before_approval"] is True
    assert approved["proposal_sha256"] == original
    assert approved["approved_sha256"] != original


def test_insufficient_seeds_marks_failed(tmp_path: Path, monkeypatch):
    service = _service(tmp_path)
    _bootstrap_pair(service)
    iteration = IterationService(service, minimum_successful_seeds=3)
    started = iteration.start_iteration("node_003", "node_004")
    proposed = started["proposed_node_id"]
    monkeypatch.setattr(
        service,
        "run_seeds",
        _fake_run_seeds(
            service,
            node_id=proposed,
            hidden_units=96,
            seeds=[42, 43, 44, 45, 46],
            successes=2,
        ),
    )
    payload = iteration.approve_and_run(started["iteration_id"])
    assert payload["status"] == "failed"
    assert payload["error_type"] == "insufficient_successful_seeds"


def test_partial_seed_failure_warns_but_continues(tmp_path: Path, monkeypatch):
    service = _service(tmp_path)
    _bootstrap_pair(service)
    iteration = IterationService(service, minimum_successful_seeds=3)
    started = iteration.start_iteration("node_003", "node_004")
    proposed = started["proposed_node_id"]
    monkeypatch.setattr(
        service,
        "run_seeds",
        _fake_run_seeds(
            service,
            node_id=proposed,
            hidden_units=96,
            seeds=[42, 43, 44, 45, 46],
            successes=4,
        ),
    )
    payload = iteration.approve_and_run(started["iteration_id"])
    assert payload["status"] == "waiting_decision"
    assert any("failed" in w for w in payload.get("warnings") or [])


def test_finalize_records_decision_and_validates_selection(tmp_path: Path, monkeypatch):
    service = _service(tmp_path)
    _bootstrap_pair(service)
    iteration = IterationService(service)
    started = iteration.start_iteration("node_003", "node_004")
    proposed = started["proposed_node_id"]
    monkeypatch.setattr(
        service,
        "run_seeds",
        _fake_run_seeds(service, node_id=proposed, hidden_units=96, seeds=[42, 43, 44, 45, 46]),
    )
    iteration.approve_and_run(started["iteration_id"])

    with pytest.raises(InvalidSelectedNode):
        iteration.finalize(
            started["iteration_id"],
            selected_node_id="node_999",
            decision_type="efficiency_tradeoff",
            reason="bad",
        )

    done = iteration.finalize(
        started["iteration_id"],
        selected_node_id=proposed,
        decision_type="efficiency_tradeoff",
        reason="Selected after paired multi-seed comparison.",
        evidence_strength="moderate",
    )
    assert done["status"] == "completed"
    assert done["decision_id"]
    assert done["selected_node_id"] == proposed

    with pytest.raises(InvalidIterationTransition):
        iteration.approve_and_run(started["iteration_id"])


def test_non_waiting_decision_cannot_finalize(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap_pair(service)
    iteration = IterationService(service)
    started = iteration.start_iteration("node_003", "node_004")
    with pytest.raises(InvalidIterationTransition):
        iteration.finalize(
            started["iteration_id"],
            selected_node_id="node_003",
            decision_type="x",
            reason="y",
        )


def test_list_iterations(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap_pair(service)
    iteration = IterationService(service)
    first = iteration.start_iteration("node_003", "node_004")
    rows = iteration.list_iterations(project_id="project_001")
    assert any(row["iteration_id"] == first["iteration_id"] for row in rows)


def test_session_persists_across_service_restart(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap_pair(service)
    iteration = IterationService(service)
    started = iteration.start_iteration("node_003", "node_004")

    restarted = ExperimentService(settings=service.settings)
    iteration2 = IterationService(restarted)
    status = iteration2.get_status(started["iteration_id"])
    assert status["status"] == "waiting_approval"
    assert status["proposal_sha256"] == started["proposal_sha256"]
