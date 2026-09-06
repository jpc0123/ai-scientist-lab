"""v2.3.4 unit tests — CUDA Formal Triad orchestration (offline dry-run)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scientist_lab.protocols.formal_triad import (
    DEFAULT_CUDA_FORMAL_PROTOCOL_ID,
    DEFAULT_CUDA_FORMAL_TRIAD_NODES,
    infer_triad_role,
)
from scientist_lab.services.experiment_service import ExperimentService, load_contract
from scientist_lab.settings import Settings
from scientist_lab.tasks.rgbt_detection.dfine_cuda_formal_triad import (
    ORCHESTRATOR_ID,
    run_dfine_cuda_formal_triad,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def service(tmp_path: Path) -> ExperimentService:
    return ExperimentService(
        settings=Settings(
            project_root=ROOT,
            db_path=tmp_path / "triad.db",
            runtime_dir=tmp_path / "runtime",
            outputs_dir=tmp_path / "outputs",
            experiment_app_dir=ROOT / "experiment_app",
        ).resolve()
    )


def test_infer_cuda_formal_node_ids():
    for role, node_id in DEFAULT_CUDA_FORMAL_TRIAD_NODES.items():
        path = ROOT / "examples" / f"rgbt_formal_cuda_{role}_contract.json"
        contract = load_contract(path)
        assert contract.node_id == node_id
        assert infer_triad_role(contract) == role


def test_cuda_formal_triad_dry_run(service: ExperimentService):
    result = run_dfine_cuda_formal_triad(
        service,
        dry_run=True,
        probe_runtime=False,
        require_live_ready=False,
    )
    assert result["orchestrator"] == ORCHESTRATOR_ID
    assert result["dry_run"] is True
    assert result["formal_success"] is False
    assert result["formal_superiority_claimed"] is False
    assert result["validation"]["ok"] is True
    assert result["protocol"]["protocol_id"] == DEFAULT_CUDA_FORMAL_PROTOCOL_ID
    assert set(result["would_run"]) == {"rgb", "thermal", "fusion"}
    for role, item in result["would_run"].items():
        assert item["environment_key"] == "rgbt-detection-v2-cuda"
        assert item["dfine_backend"] == "dfine"


def test_service_facade(service: ExperimentService):
    data = service.dfine_cuda_formal_triad(dry_run=True, probe_runtime=False)
    assert data["status"] == "dry_run"
    assert data["validation"]["protocol_id"] == DEFAULT_CUDA_FORMAL_PROTOCOL_ID
