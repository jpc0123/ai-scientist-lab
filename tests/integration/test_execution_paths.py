from __future__ import annotations

import json
from pathlib import Path

import pytest

from scientist_lab.domain import JobStatus
from scientist_lab.domain.contracts import ExperimentContract
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"


def _docker_available() -> bool:
    try:
        import docker

        client = docker.from_env()
        client.ping()
        client.images.get("scientist-experiment:v1")
        return True
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(
    not _docker_available(),
    reason="需要 Docker Desktop 与 scientist-experiment:v1 镜像",
)


def _service(tmp_path: Path) -> ExperimentService:
    settings = Settings(
        project_root=ROOT,
        db_path=tmp_path / "test.db",
        runtime_dir=tmp_path / "runtime",
        outputs_dir=tmp_path / "outputs",
        experiment_app_dir=ROOT / "experiment_app",
        poll_interval_seconds=0.2,
    ).resolve()
    return ExperimentService(settings=settings)


def _run(service: ExperimentService, name: str) -> JobStatus:
    data = json.loads((EXAMPLES / name).read_text(encoding="utf-8"))
    # Keep ids unique-ish by using temp suffixes via parameters only?
    contract = ExperimentContract.model_validate(data)
    result = service.run_contract(contract, wait=True)
    return result.status


def test_success_path(tmp_path: Path):
    service = _service(tmp_path)
    status = _run(service, "smoke_test_contract.json")
    assert status == JobStatus.COMPLETED


def test_exception_path(tmp_path: Path):
    service = _service(tmp_path)
    status = _run(service, "fail_exception.json")
    assert status == JobStatus.FAILED
    attempts = service.list_executions(limit=1)
    assert attempts
    assert attempts[0].error_json is not None
    assert attempts[0].error_json["error_type"] == "experiment_code_error"


def test_missing_metrics_path(tmp_path: Path):
    service = _service(tmp_path)
    status = _run(service, "fail_missing_metrics.json")
    assert status == JobStatus.FAILED
    attempts = service.list_executions(limit=1)
    assert attempts
    assert attempts[0].error_json is not None
    assert attempts[0].error_json["error_type"] == "missing_artifact"


def test_timeout_path(tmp_path: Path):
    service = _service(tmp_path)
    status = _run(service, "fail_timeout.json")
    assert status == JobStatus.TIMED_OUT
    attempts = service.list_executions(limit=1)
    assert attempts
    assert attempts[0].error_json is not None
    assert attempts[0].error_json["error_type"] == "timeout"


def test_multi_attempt_and_compare(tmp_path: Path):
    service = _service(tmp_path)
    a = service.run_contract(
        ExperimentContract.model_validate(
            json.loads((EXAMPLES / "multi_attempt_a.json").read_text(encoding="utf-8"))
        ),
        wait=True,
    )
    b = service.run_contract(
        ExperimentContract.model_validate(
            json.loads((EXAMPLES / "multi_attempt_b.json").read_text(encoding="utf-8"))
        ),
        wait=True,
    )
    assert a.status == JobStatus.COMPLETED
    assert b.status == JobStatus.COMPLETED
    attempts = service.list_executions(node_id="node_retry_demo", limit=10)
    assert len(attempts) >= 2
    assert {a.attempt_index for a in attempts} >= {1, 2}

    cmp = service.compare_executions(a.execution_id, b.execution_id)
    assert "metric_changes" in cmp
    assert "accuracy" in cmp["metric_changes"]
    assert "hypothesis_status" in cmp
    assert "experiment_valid" in cmp


def test_cancel_path(tmp_path: Path):
    service = _service(tmp_path)
    data = json.loads((EXAMPLES / "fail_timeout.json").read_text(encoding="utf-8"))
    data["node_id"] = "node_cancel_demo"
    data["resources"]["timeout_seconds"] = 60
    contract = ExperimentContract.model_validate(data)
    submitted = service.submit_contract(contract)
    # Give container a moment to start
    import time

    time.sleep(1.5)
    cancelled = service.cancel_execution(submitted.execution_id)
    assert cancelled.status == JobStatus.CANCELLED
