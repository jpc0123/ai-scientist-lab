"""v2.3.7 unit tests — Doctor GPU depth + offline demo contract."""

from __future__ import annotations

from pathlib import Path

from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings
from scientist_lab.system.doctor import SystemDoctor
from scientist_lab.tasks.rgbt_detection.cuda_doctor import (
    build_dfine_cuda_doctor,
    summarize_gpu_readiness,
)

ROOT = Path(__file__).resolve().parents[2]


def test_doctor_offline_includes_protocol_and_triad_checks():
    settings = Settings(project_root=ROOT).resolve()
    report = build_dfine_cuda_doctor(
        ROOT,
        image_registry=dict(settings.image_registry or {}),
        probe_runtime=False,
    )
    assert report["ok"] is True
    assert report["doctor_version"] == "v2.3.7"
    ids = {c["id"] for c in report["checks"]}
    assert "cuda_protocol" in ids
    assert "formal_triad_contracts" in ids
    assert report["docs"]["runthrough"].endswith("dfine-cuda-runthrough.md")


def test_summarize_gpu_readiness_blockers():
    summary = summarize_gpu_readiness(
        nvidia={"ok": False, "gpu_count": 0, "gpus": []},
        docker_nvidia={"ok": False},
        cuda_image={"ok": False},
    )
    assert summary["ready_for_gpu_container"] is False
    assert "nvidia-smi unavailable" in summary["blockers"]
    assert "docker nvidia runtime missing" in summary["blockers"]

    ready = summarize_gpu_readiness(
        nvidia={
            "ok": True,
            "gpu_count": 1,
            "gpus": [{"name": "FakeGPU", "driver_version": "0", "memory_total": "1 MiB"}],
        },
        docker_nvidia={"ok": True},
        cuda_image={"ok": True},
    )
    assert ready["ready_for_gpu_container"] is True
    assert ready["blockers"] == []


def test_system_doctor_includes_dfine_cuda(tmp_path: Path):
    service = ExperimentService(
        settings=Settings(
            project_root=ROOT,
            db_path=tmp_path / "sys.db",
            runtime_dir=tmp_path / "runtime",
            outputs_dir=tmp_path / "outputs",
            experiment_app_dir=ROOT / "experiment_app",
        ).resolve()
    )
    report = SystemDoctor(service).run()
    by_id = {c["id"]: c for c in report["checks"]}
    assert "dfine_cuda" in by_id
    assert by_id["dfine_cuda"]["level"] in {"ok", "warning"}
    assert "environment_key" in by_id["dfine_cuda"]["details"]
