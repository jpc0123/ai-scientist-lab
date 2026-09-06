"""v2.0.1 ResearchProject lifecycle tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from scientist_lab.domain import ProjectStatus
from scientist_lab.projects.lifecycle import assert_transition
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings


def _service(tmp_path: Path) -> ExperimentService:
    root = Path(__file__).resolve().parents[2]
    return ExperimentService(
        settings=Settings(
            project_root=root,
            db_path=tmp_path / "projects.db",
            runtime_dir=tmp_path / "runtime",
            outputs_dir=tmp_path / "outputs",
            experiment_app_dir=root / "experiment_app",
        ).resolve()
    )


def test_illegal_transition_rejected():
    with pytest.raises(ValueError, match="illegal project status"):
        assert_transition(ProjectStatus.DRAFT, ProjectStatus.COMPLETED)


def test_create_project_wizard_ready_and_persist(tmp_path: Path):
    service = _service(tmp_path)
    created = service.create_project(
        title="Workbench Demo",
        research_question="Can RGB-T fusion beat RGB-only?",
        task_type="rgbt_detection",
        dataset_keys=["rgbt_debug_v1"],
        runner_profile_keys=["local"],
        protocol_draft={"claim_level": "debug"},
    )
    assert created["status"] == "ready"
    assert created["wizard_completed"] is True
    assert created["task_type"] == "rgbt_detection"
    pid = created["project_id"]

    # Simulate restart with a new service on same DB.
    again = _service(tmp_path)
    shown = again.get_project(pid)
    assert shown["project_id"] == pid
    assert shown["status"] == "ready"
    assert shown["research_question"].startswith("Can RGB-T")
    assert shown["dataset_keys"] == ["rgbt_debug_v1"]

    archived = again.archive_project(pid)
    assert archived["status"] == "archived"
    with pytest.raises(ValueError):
        again.projects.transition(pid, ProjectStatus.READY)


def test_create_rejects_bad_task_type(tmp_path: Path):
    service = _service(tmp_path)
    with pytest.raises(ValueError, match="task_type"):
        service.create_project(title="x", task_type="not_a_task")
