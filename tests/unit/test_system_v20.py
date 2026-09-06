from __future__ import annotations

from pathlib import Path

from scientist_lab.domain import JobStatus, NodeStage, NodeStatus, NodeType
from scientist_lab.domain.models import ExecutionAttempt, ExperimentNode, utc_now_iso
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings
from scientist_lab.system.doctor import SystemDoctor
from scientist_lab.system.recovery import RecoveryService


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


def test_system_doctor_runs(tmp_path: Path):
    service = _service(tmp_path)
    report = SystemDoctor(service).run()
    assert report["overall"] in {"ok", "warning", "error"}
    assert report["summary"]["ok"] + report["summary"]["warning"] + report["summary"][
        "error"
    ] == len(report["checks"])
    ids = {item["id"] for item in report["checks"]}
    assert "python" in ids
    assert "database" in ids
    assert "schema" in ids
    assert "security" in ids
    assert report["components"]["shell_available_in_ui"] is False


def test_recover_marks_orphaned_running_execution(tmp_path: Path):
    service = _service(tmp_path)
    now = utc_now_iso()
    service.repo.upsert_node(
        ExperimentNode(
            node_id="node_recover_1",
            project_id="project_recover",
            node_type=NodeType.BASELINE,
            stage=NodeStage.EXECUTING,
            status=NodeStatus.RUNNING,
            depth=0,
            contract_json={"node_id": "node_recover_1", "project_id": "project_recover"},
            created_at=now,
            updated_at=now,
        )
    )
    service.repo.upsert_attempt(
        ExecutionAttempt(
            execution_id="exec_recover_1",
            node_id="node_recover_1",
            attempt_index=1,
            runner_profile="local_docker",
            status=JobStatus.RUNNING,
            image_reference="scientist-experiment:v1",
            created_at=now,
            started_at=now,
        )
    )

    dry = RecoveryService(service).recover(dry_run=True)
    assert any(
        a.get("resource_id") == "exec_recover_1" and a.get("action") == "mark_interrupted"
        for a in dry["actions"]
    )
    assert dry["policy"]["auto_rerun_experiments"] is False

    applied = RecoveryService(service).recover(dry_run=False)
    assert any(
        a.get("resource_id") == "exec_recover_1" and a.get("applied")
        for a in applied["actions"]
    )
    attempt = service.repo.get_attempt("exec_recover_1")
    assert attempt is not None
    assert str(attempt.status) == "interrupted"


def test_service_wrappers(tmp_path: Path):
    service = _service(tmp_path)
    doctor = service.system_doctor()
    recover = service.recover(dry_run=True)
    assert "checks" in doctor
    assert recover["dry_run"] is True
