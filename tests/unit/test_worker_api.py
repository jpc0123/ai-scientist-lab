from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scientist_worker.api import create_app
from scientist_worker.models import JobSubmitRequest
from scientist_worker.service import WorkerService
from scientist_worker.settings import WorkerSettings


@pytest.fixture()
def worker_env(tmp_path: Path):
    settings = WorkerSettings(
        data_root=tmp_path / "worker-data",
        executor_mode="mock",
        require_auth=False,
        auth_token=None,
        supported_environment_keys=["mock-detection-v1", "rgbt-detection-v2"],
    ).resolve()
    service = WorkerService(settings)
    app = create_app(settings, service)
    client = TestClient(app)
    return settings, service, client


def _payload(request_id: str = "req_1", execution_id: str = "exec_1") -> dict:
    return {
        "request_id": request_id,
        "execution_id": execution_id,
        "contract": {
            "project_id": "project_mock",
            "node_id": "mock_node_001",
            "task_type": "rgbt_detection",
            "environment_key": "mock-detection-v1",
            "entrypoint": "run_detection_experiment.py",
            "execution_mode": "fast_eval",
            "seed": 42,
            "parameters": {"baseline": "mock", "epochs": 0},
            "task_config": {
                "claim_level": "pipeline_validation_only",
                "evaluation_scope": "debug_subset",
            },
        },
        "environment": {"environment_key": "mock-detection-v1"},
    }


def test_health(worker_env):
    _, _, client = worker_env
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["executor_mode"] == "mock"


def test_capabilities(worker_env):
    _, _, client = worker_env
    resp = client.get("/v1/capabilities")
    assert resp.status_code == 200
    data = resp.json()
    assert "mock-detection-v1" in data["supported_environment_keys"]


def test_submit_mock_job_and_collect(worker_env):
    _, service, client = worker_env
    resp = client.post("/v1/jobs", json=_payload())
    assert resp.status_code == 200
    body = resp.json()
    job_id = body["job_id"]
    assert body["status"] in {"queued", "received", "running", "completed"}

    # Wait for background mock executor.
    deadline = time.time() + 5
    status = None
    while time.time() < deadline:
        status = client.get(f"/v1/jobs/{job_id}").json()
        if status["status"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.05)
    assert status is not None
    assert status["status"] == "completed"

    logs = client.get(f"/v1/jobs/{job_id}/logs").json()
    assert "mock job" in logs["content"] or "worker" in logs["content"]
    assert logs["complete"] is True

    result = client.get(f"/v1/jobs/{job_id}/result").json()
    assert result["status"] == "completed"
    assert result["result"]["metrics"]["status"] == "completed"

    artifacts = client.get(f"/v1/jobs/{job_id}/artifacts")
    assert artifacts.status_code == 200
    assert len(artifacts.content) > 20

    # bundle exists on disk
    path = service.artifacts_path(job_id)
    assert path.exists()


def test_idempotent_request_id(worker_env):
    _, _, client = worker_env
    first = client.post("/v1/jobs", json=_payload("req_dup", "exec_dup")).json()
    second = client.post("/v1/jobs", json=_payload("req_dup", "exec_other")).json()
    assert first["job_id"] == second["job_id"]
    assert first["execution_id"] == second["execution_id"]


def test_reject_unknown_environment(worker_env):
    _, _, client = worker_env
    payload = _payload("req_bad", "exec_bad")
    payload["environment"]["environment_key"] = "no-such-env"
    payload["contract"]["environment_key"] = "no-such-env"
    resp = client.post("/v1/jobs", json=payload)
    assert resp.status_code == 400
    assert "remote_capability_mismatch" in resp.json()["detail"]


def test_auth_required_when_token_set(tmp_path: Path):
    settings = WorkerSettings(
        data_root=tmp_path / "worker-auth",
        auth_token="secret",
        require_auth=True,
        executor_mode="mock",
    ).resolve()
    client = TestClient(create_app(settings))
    denied = client.get("/v1/capabilities")
    assert denied.status_code == 401
    ok = client.get(
        "/v1/capabilities", headers={"Authorization": "Bearer secret"}
    )
    assert ok.status_code == 200


def test_service_submit_direct(tmp_path: Path):
    settings = WorkerSettings(
        data_root=tmp_path / "worker-direct", executor_mode="mock"
    ).resolve()
    service = WorkerService(settings)
    submitted = service.submit(JobSubmitRequest.model_validate(_payload("req_d", "exec_d")))
    deadline = time.time() + 5
    while time.time() < deadline:
        status = service.get_status(submitted.job_id)
        if status.status == "completed":
            break
        time.sleep(0.05)
    assert service.get_status(submitted.job_id).status == "completed"
