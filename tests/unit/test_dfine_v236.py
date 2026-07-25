"""v2.3.6 unit tests — gated real DFINE acceptance contract (offline)."""

from __future__ import annotations

from pathlib import Path

from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings
from scientist_lab.tasks.rgbt_detection.dfine_real_acceptance import (
    CORE_LIVE_STEPS,
    OPTIONAL_TRIAD_STEP,
    PIPELINE_ID,
    describe_real_acceptance_plan,
    real_dfine_acceptance_gates,
    run_real_dfine_acceptance,
)

ROOT = Path(__file__).resolve().parents[2]


def test_gates_skip_without_env():
    ready, reason, gates = real_dfine_acceptance_gates(
        ROOT, environ={"RUN_REAL_DFINE_TESTS": "", "RUN_REAL_CUDA": ""}
    )
    assert ready is False
    assert "RUN_REAL_DFINE_TESTS" in reason
    assert gates["run_real_dfine_tests"] is False


def test_gates_require_both_env_flags():
    ready, reason, gates = real_dfine_acceptance_gates(
        ROOT,
        environ={
            "RUN_REAL_DFINE_TESTS": "1",
            "RUN_REAL_CUDA": "",
        },
    )
    assert ready is False
    assert "RUN_REAL_CUDA" in reason
    assert gates["run_real_dfine_tests"] is True
    assert gates["run_real_cuda"] is False


def test_describe_plan_core_and_triad():
    core = describe_real_acceptance_plan(include_formal_triad=False)
    assert core["pipeline"] == PIPELINE_ID
    assert core["steps"] == list(CORE_LIVE_STEPS)
    assert core["formal_success"] is False
    assert core["formal_superiority_claimed"] is False
    assert OPTIONAL_TRIAD_STEP not in core["steps"]

    with_triad = describe_real_acceptance_plan(include_formal_triad=True)
    assert OPTIONAL_TRIAD_STEP in with_triad["steps"]
    assert with_triad["steps"][:4] == list(CORE_LIVE_STEPS)


def test_dry_run_acceptance_pipeline(tmp_path: Path):
    service = ExperimentService(
        settings=Settings(
            project_root=ROOT,
            db_path=tmp_path / "real.db",
            runtime_dir=tmp_path / "runtime",
            outputs_dir=tmp_path / "outputs",
            experiment_app_dir=ROOT / "experiment_app",
        ).resolve()
    )
    result = run_real_dfine_acceptance(
        service,
        dry_run=True,
        include_formal_triad=True,
        probe_runtime=False,
    )
    assert result["pipeline"] == PIPELINE_ID
    assert result["dry_run"] is True
    assert result["status"] == "dry_run"
    assert result["formal_success"] is False
    assert result["formal_superiority_claimed"] is False
    assert result["would_run"]["require_live_ready"] is True
    assert OPTIONAL_TRIAD_STEP in result["would_run"]["steps"]
    assert len(result["checks"]) == len(result["plan"]["steps"])


def test_service_facade_dry_run(tmp_path: Path):
    service = ExperimentService(
        settings=Settings(
            project_root=ROOT,
            db_path=tmp_path / "real2.db",
            runtime_dir=tmp_path / "runtime",
            outputs_dir=tmp_path / "outputs",
            experiment_app_dir=ROOT / "experiment_app",
        ).resolve()
    )
    data = service.dfine_real_acceptance(dry_run=True, include_formal_triad=False)
    assert data["dry_run"] is True
    assert data["plan"]["steps"] == list(CORE_LIVE_STEPS)
