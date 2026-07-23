"""Project service facade (v2.0.1)."""

from __future__ import annotations

from typing import Any

from scientist_lab.domain import ProjectStatus
from scientist_lab.domain.models import ResearchProject, new_id, utc_now_iso
from scientist_lab.projects.lifecycle import (
    assert_transition,
    normalize_status,
    validate_task_type,
)
from scientist_lab.storage.repositories import Repository


class ProjectService:
    def __init__(self, repo: Repository) -> None:
        self.repo = repo

    def create_from_wizard(
        self,
        *,
        title: str,
        research_question: str = "",
        research_goal: str = "",
        description: str = "",
        task_type: str = "general_ml",
        dataset_keys: list[str] | None = None,
        protocol_ids: list[str] | None = None,
        runner_profile_keys: list[str] | None = None,
        default_llm_profile_id: str | None = None,
        expected_metrics: dict[str, Any] | None = None,
        constraints: dict[str, Any] | None = None,
        protocol_draft: dict[str, Any] | None = None,
        project_id: str | None = None,
        mark_ready: bool = True,
    ) -> ResearchProject:
        title = (title or "").strip()
        if not title:
            raise ValueError("title is required")
        question = (research_question or "").strip()
        goal = (research_goal or question or title).strip()
        now = utc_now_iso()
        project = ResearchProject(
            project_id=(project_id or "").strip() or new_id("project"),
            title=title,
            research_goal=goal,
            description=(description or "").strip(),
            research_question=question or goal,
            task_type=validate_task_type(task_type),
            status=ProjectStatus.READY if mark_ready else ProjectStatus.CONFIGURING,
            dataset_keys=list(dataset_keys or []),
            protocol_ids=list(protocol_ids or []),
            runner_profile_keys=list(runner_profile_keys or []),
            default_llm_profile_id=default_llm_profile_id,
            expected_metrics=dict(expected_metrics or {}),
            constraints=dict(constraints or {}),
            protocol_draft=dict(protocol_draft or {}),
            wizard_completed=True,
            created_at=now,
            updated_at=now,
        )
        self.repo.upsert_project(project)
        return project

    def archive(self, project_id: str) -> ResearchProject:
        project = self.repo.get_project(project_id)
        if project is None:
            raise KeyError(f"project not found: {project_id}")
        assert_transition(project.status, ProjectStatus.ARCHIVED)
        project.status = ProjectStatus.ARCHIVED
        project.touch()
        self.repo.upsert_project(project)
        return project

    def transition(self, project_id: str, target: ProjectStatus) -> ResearchProject:
        project = self.repo.get_project(project_id)
        if project is None:
            raise KeyError(f"project not found: {project_id}")
        assert_transition(project.status, target)
        project.status = target
        project.touch()
        self.repo.upsert_project(project)
        return project

    def to_view(self, project: ResearchProject) -> dict[str, Any]:
        data = project.model_dump(mode="json")
        data["status"] = str(normalize_status(project.status))
        data["legacy_status"] = str(project.status)
        return data
