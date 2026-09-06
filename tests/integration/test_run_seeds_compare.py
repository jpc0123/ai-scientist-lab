from __future__ import annotations

import json
from pathlib import Path

import pytest

from scientist_lab.domain.contracts import ExperimentContract
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"


def _docker_v1_available() -> bool:
    try:
        import docker

        client = docker.from_env()
        client.ping()
        client.images.get("scientist-experiment:v1")
        return True
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(
    not _docker_v1_available(),
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


def test_run_seeds_mock_and_compare_groups(tmp_path: Path):
    service = _service(tmp_path)
    base = json.loads((EXAMPLES / "smoke_test_contract.json").read_text(encoding="utf-8"))
    base["node_id"] = "node_seed_a"
    base["parameters"]["learning_rate"] = 0.002
    cand = json.loads((EXAMPLES / "smoke_test_contract.json").read_text(encoding="utf-8"))
    cand["node_id"] = "node_seed_b"
    cand["parameters"]["learning_rate"] = 0.003

    seeds = [42, 43, 44]
    a = service.run_seeds(ExperimentContract.model_validate(base), seeds)
    b = service.run_seeds(ExperimentContract.model_validate(cand), seeds)
    assert a["aggregate"]["seed_count"] == 3
    assert b["aggregate"]["seed_count"] == 3

    cmp = service.compare_node_groups("node_seed_a", "node_seed_b")
    assert cmp["shared_seeds"] == seeds
    assert "hypothesis_status" in cmp
    assert "stable_improvement" in cmp
    assert Path(a["aggregate"]["aggregate_path"]).exists()
