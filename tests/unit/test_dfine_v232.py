"""v2.3.2 unit tests — DFINE CUDA Fast Eval orchestrator (offline dry-run)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings
from scientist_lab.tasks.rgbt_detection.dfine_cuda_orchestrator import (
    DfineCudaOrchestratorError,
    DfineCudaFastEvalOrchestrator,
    ORCHESTRATOR_ID,
    run_dfine_cuda_fast_eval,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def service(tmp_path: Path) -> ExperimentService:
    return ExperimentService(
        settings=Settings(
            project_root=ROOT,
            db_path=tmp_path / "orch.db",
            runtime_dir=tmp_path / "runtime",
            outputs_dir=tmp_path / "outputs",
            experiment_app_dir=ROOT / "experiment_app",
        ).resolve()
    )


def test_plan_and_dry_run(service: ExperimentService):
    result = run_dfine_cuda_fast_eval(
        service,
        dry_run=True,
        probe_runtime=False,
        require_live_ready=False,
    )
    assert result["orchestrator"] == ORCHESTRATOR_ID
    assert result["dry_run"] is True
    assert result["exploratory_only"] is True
    assert result["formal_success"] is False
    assert result["status"] == "dry_run"
    assert result["would_run"]["environment_key"] == "rgbt-detection-v2-cuda"
    assert result["would_run"]["dfine_backend"] == "dfine"
    assert result["claim_gate"]["formal_claim_allowed"] is False


def test_require_live_ready_fails_without_gpu(service: ExperimentService):
    orch = DfineCudaFastEvalOrchestrator(service)
    report = orch.doctor(probe_runtime=True)
    if report.get("live_ready"):
        pytest.skip("environment already live_ready (GPU+image present)")
    with pytest.raises(DfineCudaOrchestratorError, match="live_ready"):
        orch.plan(
            probe_runtime=True,
            require_live_ready=True,
        )


def test_service_facade_dry_run(service: ExperimentService):
    data = service.dfine_cuda_fast_eval(dry_run=True, probe_runtime=False)
    assert data["dry_run"] is True
    assert data["plan"]["node_id"]


def test_record_metadata_without_artifacts(service: ExperimentService, tmp_path: Path):
    orch = DfineCudaFastEvalOrchestrator(service)
    plan = orch.plan(probe_runtime=False, require_live_ready=False)
    meta = orch.record_execution_metadata(
        "exec_dry_meta",
        plan=plan,
        run_result={"status": "completed", "output_directory": str(tmp_path)},
    )
    assert meta["exploratory_only"] is True
    assert meta["formal_success"] is False
    assert meta["standin_or_vendor"] in {"vendor", "unknown"}
    assert Path(meta["metadata_path"]).is_file()
