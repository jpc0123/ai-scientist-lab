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


def _docker_v2_available() -> bool:
    try:
        import docker

        client = docker.from_env()
        client.ping()
        client.images.get("scientist-experiment:v2")
        return True
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(
    not _docker_v2_available(),
    reason="需要 Docker Desktop 与 scientist-experiment:v2 镜像",
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


def _load(name: str) -> ExperimentContract:
    data = json.loads((EXAMPLES / name).read_text(encoding="utf-8"))
    return ExperimentContract.model_validate(data)


def test_digits_single_variable_compare(tmp_path: Path):
    service = _service(tmp_path)
    a = service.run_contract(_load("digits_real_contract.json"), wait=True)
    b = service.run_contract(_load("digits_real_contract_02.json"), wait=True)
    assert a.status == JobStatus.COMPLETED
    assert b.status == JobStatus.COMPLETED

    cmp = service.compare_executions(a.execution_id, b.execution_id)
    assert cmp["experiment_valid"] is True
    assert cmp["parameter_changes"] == {
        "hidden_units": {"from": 64, "to": 128}
    }
    assert "accuracy" in cmp["metric_changes"]
    assert cmp["hypothesis_status"] in {"supported", "rejected", "inconclusive"}
    assert cmp["conclusion"]

    nodes = service.compare_nodes("node_003", "node_004")
    assert nodes["baseline_execution_id"] == a.execution_id
    assert nodes["candidate_execution_id"] == b.execution_id
    assert nodes["experiment_valid"] is True


def test_invalid_seed_compare_inconclusive(tmp_path: Path):
    service = _service(tmp_path)
    a = service.run_contract(_load("digits_real_contract.json"), wait=True)
    b = service.run_contract(_load("compare_invalid_seed.json"), wait=True)
    assert a.status == JobStatus.COMPLETED
    assert b.status == JobStatus.COMPLETED

    cmp = service.compare_executions(a.execution_id, b.execution_id)
    assert cmp["experiment_valid"] is False
    assert cmp["hypothesis_status"] == "inconclusive"
    assert any("seed differs" in x for x in cmp["verification"]["blocking_issues"])
