"""v1.8.1 release / workspace unit tests (offline)."""

from __future__ import annotations

from pathlib import Path

from scientist_lab.release.service import ReleaseService
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings
from scientist_lab.storage.database import init_db


def _service(tmp_path: Path) -> ExperimentService:
    root = Path(__file__).resolve().parents[2]
    return ExperimentService(
        settings=Settings(
            project_root=root,
            db_path=tmp_path / "release.db",
            runtime_dir=tmp_path / "runtime",
            outputs_dir=tmp_path / "outputs",
            experiment_app_dir=root / "experiment_app",
        ).resolve()
    )


def test_workspace_summary_never_writable_main(tmp_path: Path):
    service = _service(tmp_path)
    summary = service.workspace_summary()
    assert summary["can_write_main"] is False
    assert summary["main_workspace_modified"] is False
    kinds = {w["kind"] for w in summary["workspaces"]}
    assert kinds == {"main", "sandbox", "frozen"}
    main = next(w for w in summary["workspaces"] if w["kind"] == "main")
    assert main["writable"] is False


def test_release_create_freeze_discard(tmp_path: Path):
    service = _service(tmp_path)
    created = service.create_release(
        project_id="project_release_001",
        title="Demo freeze",
        patch_ids=["patch_demo"],
    )
    assert created["status"] == "draft"
    assert created["can_freeze"] is True
    assert created["can_write_main"] is False

    frozen = service.freeze_release(created["release_id"], notes="unit freeze")
    assert frozen["status"] == "frozen"
    assert frozen["snapshot"]["main_workspace_modified"] is False
    assert frozen["manifest"]["main_workspace_modified"] is False
    assert frozen["can_freeze"] is False

    discarded = service.discard_release(created["release_id"], reason="not needed")
    assert discarded["status"] == "discarded"


def test_release_repository_roundtrip(tmp_path: Path):
    root = Path(__file__).resolve().parents[2]
    session_factory = init_db(str(tmp_path / "rel.db"))
    releases = ReleaseService(
        session_factory,
        project_root=root,
        outputs_root=tmp_path / "outputs",
    )
    created = releases.create(project_id="p1", title="t1")
    shown = releases.show(created["release_id"])
    assert shown["title"] == "t1"
    listed = releases.list_releases(project_id="p1")
    assert len(listed) == 1
