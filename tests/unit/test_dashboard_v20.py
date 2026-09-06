"""v2.0.2 unified dashboard summary tests."""

from __future__ import annotations

from pathlib import Path

from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings


def test_system_summary_workbench_shape(tmp_path: Path):
    root = Path(__file__).resolve().parents[2]
    service = ExperimentService(
        settings=Settings(
            project_root=root,
            db_path=tmp_path / "dash.db",
            runtime_dir=tmp_path / "runtime",
            outputs_dir=tmp_path / "outputs",
            experiment_app_dir=root / "experiment_app",
        ).resolve()
    )
    service.create_project(
        title="Dash Demo",
        research_question="Does dashboard surface todos?",
        task_type="general_ml",
    )
    summary = service.system_summary()
    assert summary["project_count"] >= 1
    assert "todos" in summary
    assert "todo_count" in summary
    assert "answers" in summary
    assert "claim_summary" in summary
    assert "system_health" in summary
    assert summary["system_health"]["shell_available"] is False
    assert summary["system_health"]["network_default"] is False
    assert "best_nodes" in summary
    assert "recent_activity" in summary
    assert summary["answers"]["what_awaits_me"] == summary["todo_count"]
