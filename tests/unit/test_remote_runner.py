from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

import pytest
import uvicorn

from scientist_lab.domain.contracts import ExperimentContract
from scientist_lab.runners.normalize import normalize_execution_result
from scientist_lab.runners.profile_registry import RunnerProfileRegistry
from scientist_lab.runners.remote_docker_runner import RemoteDockerRunner
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings
from scientist_lab.storage.database import init_db
from scientist_worker.api import create_app
from scientist_worker.service import WorkerService
from scientist_worker.settings import WorkerSettings


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


@pytest.fixture()
def worker_server(tmp_path: Path):
    port = _free_port()
    settings = WorkerSettings(
        data_root=tmp_path / "worker-data",
        executor_mode="mock",
        host="127.0.0.1",
        port=port,
        require_auth=False,
    ).resolve()
    app = create_app(settings, WorkerService(settings))
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 5
    while time.time() < deadline and not server.started:
        time.sleep(0.05)
    assert server.started
    yield f"http://127.0.0.1:{port}", settings
    server.should_exit = True
    thread.join(timeout=5)


def test_register_runner_profile_rejects_duplicate(tmp_path: Path):
    session_factory = init_db(str(tmp_path / "profiles.db"))
    registry = RunnerProfileRegistry(session_factory)
    registry.register(
        profile_key="remote_a",
        runner_type="remote_docker",
        endpoint="http://127.0.0.1:8080",
        allowed_environment_keys=["mock-detection-v1"],
    )
    with pytest.raises(ValueError, match="already exists"):
        registry.register(
            profile_key="remote_a",
            runner_type="remote_docker",
            endpoint="http://127.0.0.1:8080",
            allowed_environment_keys=["mock-detection-v1"],
        )


def test_register_runner_profile_rejects_bad_endpoint(tmp_path: Path):
    session_factory = init_db(str(tmp_path / "profiles2.db"))
    registry = RunnerProfileRegistry(session_factory)
    with pytest.raises(ValueError):
        registry.register(
            profile_key="bad",
            runner_type="remote_docker",
            endpoint="ftp://nope",
            allowed_environment_keys=["mock-detection-v1"],
        )


def test_remote_mock_contract_via_remote_runner(tmp_path: Path, worker_server):
    endpoint, _ = worker_server
    settings = Settings(
        project_root=tmp_path,
        db_path=tmp_path / "lab.db",
        runtime_dir=tmp_path / "runtime",
        outputs_dir=tmp_path / "outputs",
        experiment_app_dir=tmp_path / "app",
        poll_interval_seconds=0.05,
    ).resolve()
    (tmp_path / "app").mkdir()
    service = ExperimentService(settings=settings)
    service.register_runner_profile(
        profile_key="remote_mock_01",
        runner_type="remote_docker",
        endpoint=endpoint,
        allowed_environments=["mock-detection-v1"],
    )
    check = service.check_runner("remote_mock_01")
    assert check["ok"] is True

    contract = ExperimentContract.model_validate(
        {
            "schema_version": "1.2",
            "project_id": "project_rgbt_002",
            "node_id": "rgbt_remote_mock_001",
            "title": "remote mock",
            "research_goal": "g",
            "hypothesis": "h",
            "task_type": "rgbt_detection",
            "runner_profile": "remote_mock_01",
            "environment_key": "mock-detection-v1",
            "code_reference": "local:rgbt_detection_real",
            "dataset_reference": "dataset:rgbt_fast_eval_v1",
            "entrypoint": "run_detection_experiment.py",
            "execution_mode": "fast_eval",
            "parameters": {"baseline": "mock", "epochs": 0},
            "task_config": {
                "claim_level": "pipeline_validation_only",
                "evaluation_scope": "debug_subset",
                "primary_metric": "mAP50_95",
            },
            "seed": 42,
            "resources": {
                "gpu_count": 0,
                "cpu_count": 1,
                "memory_gb": 1,
                "timeout_seconds": 60,
            },
            "expected_outputs": ["metrics.json"],
        }
    )
    result = service.run_contract(contract, wait=True)
    assert str(result.status) == "completed"
    assert result.metrics.get("status") == "completed"
    assert (Path(result.output_directory) / "metrics.json").exists()
    assert (Path(result.output_directory) / "model_summary.json").exists()

    normalized = normalize_execution_result(
        {
            "status": str(result.status),
            "metrics": result.metrics,
            "artifacts": [a.model_dump() for a in result.artifacts],
            "error": None if result.error is None else result.error.model_dump(),
        }
    )
    assert normalized["metrics"]["status"] == "completed"
    assert "metrics.json" in normalized["artifact_names"]


def test_remote_rejects_disallowed_environment(tmp_path: Path, worker_server):
    endpoint, _ = worker_server
    settings = Settings(
        project_root=tmp_path,
        db_path=tmp_path / "lab2.db",
        runtime_dir=tmp_path / "runtime2",
        outputs_dir=tmp_path / "outputs2",
        experiment_app_dir=tmp_path / "app2",
        poll_interval_seconds=0.05,
    ).resolve()
    (tmp_path / "app2").mkdir()
    service = ExperimentService(settings=settings)
    service.register_runner_profile(
        profile_key="remote_mock_02",
        runner_type="remote_docker",
        endpoint=endpoint,
        allowed_environments=["mock-detection-v1"],
    )
    contract = ExperimentContract.model_validate(
        {
            "schema_version": "1.2",
            "project_id": "p",
            "node_id": "n",
            "title": "t",
            "research_goal": "g",
            "hypothesis": "h",
            "task_type": "rgbt_detection",
            "runner_profile": "remote_mock_02",
            "environment_key": "rgbt-detection-v2",
            "code_reference": "local:rgbt_detection_real",
            "dataset_reference": "dataset:rgbt_fast_eval_v1",
            "entrypoint": "run_detection_experiment.py",
            "execution_mode": "fast_eval",
            "parameters": {"baseline": "mock", "epochs": 0},
            "task_config": {},
            "seed": 1,
            "resources": {
                "gpu_count": 0,
                "cpu_count": 1,
                "memory_gb": 1,
                "timeout_seconds": 30,
            },
            "expected_outputs": ["metrics.json"],
        }
    )
    with pytest.raises(Exception, match="remote_capability_mismatch|not allowed"):
        service.run_contract(contract, wait=True)
