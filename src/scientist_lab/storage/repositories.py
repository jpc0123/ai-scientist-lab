from __future__ import annotations

import json
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from scientist_lab.domain import (
    JobStatus,
    NodeStage,
    NodeStatus,
    NodeType,
    ProjectStatus,
)
from scientist_lab.domain.models import (
    ExecutionAttempt,
    ExperimentArtifact,
    ExperimentNode,
    ResearchProject,
)
from scientist_lab.storage.database import (
    ExecutionAttemptRow,
    ExperimentArtifactRow,
    ExperimentNodeRow,
    ResearchProjectRow,
)


def _dump(data: dict[str, Any] | None) -> str | None:
    if data is None:
        return None
    return json.dumps(data, ensure_ascii=False)


def _load(raw: str | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    return json.loads(raw)


class Repository:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory

    def _session(self) -> Session:
        return self._session_factory()

    def upsert_project(self, project: ResearchProject) -> None:
        payload = project.model_dump(mode="json")
        # Columns keep primary display fields; payload holds full model.
        with self._session() as session:
            row = session.get(ResearchProjectRow, project.project_id)
            if row is None:
                row = ResearchProjectRow(project_id=project.project_id)
                session.add(row)
            row.title = project.title
            row.research_goal = project.research_goal or ""
            row.status = str(project.status)
            row.created_at = project.created_at
            row.updated_at = project.updated_at
            row.payload_json = json.dumps(payload, ensure_ascii=False)
            session.commit()

    def get_project(self, project_id: str) -> ResearchProject | None:
        with self._session() as session:
            row = session.get(ResearchProjectRow, project_id)
            if row is None:
                return None
            return self._row_to_project(row)

    def _row_to_project(self, row: ResearchProjectRow) -> ResearchProject:
        if row.payload_json:
            try:
                data = json.loads(row.payload_json)
                data["project_id"] = row.project_id
                data["title"] = row.title or data.get("title") or ""
                data["research_goal"] = row.research_goal or data.get("research_goal") or ""
                data["status"] = row.status or data.get("status") or "draft"
                data["created_at"] = row.created_at or data.get("created_at") or ""
                data["updated_at"] = row.updated_at or data.get("updated_at") or ""
                return ResearchProject.model_validate(data)
            except Exception:  # noqa: BLE001
                pass
        try:
            status = ProjectStatus(row.status)
        except ValueError:
            status = ProjectStatus.READY
        return ResearchProject(
            project_id=row.project_id,
            title=row.title,
            research_goal=row.research_goal,
            research_question=row.research_goal,
            status=status,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    def upsert_node(self, node: ExperimentNode) -> None:
        with self._session() as session:
            row = session.get(ExperimentNodeRow, node.node_id)
            if row is None:
                row = ExperimentNodeRow(node_id=node.node_id)
                session.add(row)
            row.project_id = node.project_id
            row.parent_node_id = node.parent_node_id
            row.node_type = str(node.node_type)
            row.stage = str(node.stage)
            row.hypothesis = node.hypothesis
            row.status = str(node.status)
            row.depth = node.depth
            row.contract_json = json.dumps(node.contract_json, ensure_ascii=False)
            row.feedback_json = _dump(node.feedback_json)
            row.score = node.score
            row.created_at = node.created_at
            row.updated_at = node.updated_at
            session.commit()

    def get_node(self, node_id: str) -> ExperimentNode | None:
        with self._session() as session:
            row = session.get(ExperimentNodeRow, node_id)
            if row is None:
                return None
            return ExperimentNode(
                node_id=row.node_id,
                project_id=row.project_id,
                parent_node_id=row.parent_node_id,
                node_type=NodeType(row.node_type),
                stage=NodeStage(row.stage),
                hypothesis=row.hypothesis,
                status=NodeStatus(row.status),
                depth=row.depth,
                contract_json=json.loads(row.contract_json),
                feedback_json=_load(row.feedback_json),
                score=row.score,
                created_at=row.created_at,
                updated_at=row.updated_at,
            )

    def next_attempt_index(self, node_id: str) -> int:
        with self._session() as session:
            rows = (
                session.query(ExecutionAttemptRow)
                .filter(ExecutionAttemptRow.node_id == node_id)
                .all()
            )
            return len(rows) + 1

    def upsert_attempt(self, attempt: ExecutionAttempt) -> None:
        with self._session() as session:
            row = session.get(ExecutionAttemptRow, attempt.execution_id)
            if row is None:
                row = ExecutionAttemptRow(execution_id=attempt.execution_id)
                session.add(row)
            row.node_id = attempt.node_id
            row.attempt_index = attempt.attempt_index
            row.runner_profile = attempt.runner_profile
            row.status = str(attempt.status)
            row.container_id = attempt.container_id
            row.image_reference = attempt.image_reference
            row.code_version = attempt.code_version
            row.dataset_version = attempt.dataset_version
            row.result_json = _dump(attempt.result_json)
            row.error_json = _dump(attempt.error_json)
            row.started_at = attempt.started_at
            row.completed_at = attempt.completed_at
            row.created_at = attempt.created_at
            session.commit()

    def get_attempt(self, execution_id: str) -> ExecutionAttempt | None:
        with self._session() as session:
            row = session.get(ExecutionAttemptRow, execution_id)
            if row is None:
                return None
            return self._row_to_attempt(row)

    def list_projects(self) -> list[ResearchProject]:
        with self._session() as session:
            rows = (
                session.query(ResearchProjectRow)
                .order_by(ResearchProjectRow.updated_at.desc())
                .all()
            )
            return [self._row_to_project(row) for row in rows]

    def count_nodes(self, project_id: str) -> int:
        with self._session() as session:
            return (
                session.query(ExperimentNodeRow)
                .filter(ExperimentNodeRow.project_id == project_id)
                .count()
            )

    def list_nodes(self, project_id: str | None = None) -> list[ExperimentNode]:
        with self._session() as session:
            query = session.query(ExperimentNodeRow)
            if project_id:
                query = query.filter(ExperimentNodeRow.project_id == project_id)
            rows = query.order_by(ExperimentNodeRow.updated_at.desc()).all()
            return [
                ExperimentNode(
                    node_id=row.node_id,
                    project_id=row.project_id,
                    parent_node_id=row.parent_node_id,
                    node_type=NodeType(row.node_type),
                    stage=NodeStage(row.stage),
                    hypothesis=row.hypothesis,
                    status=NodeStatus(row.status),
                    depth=row.depth,
                    contract_json=json.loads(row.contract_json),
                    feedback_json=_load(row.feedback_json),
                    score=row.score,
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                )
                for row in rows
            ]

    def list_attempts(
        self,
        limit: int = 20,
        project_id: str | None = None,
        node_id: str | None = None,
    ) -> list[ExecutionAttempt]:
        with self._session() as session:
            query = session.query(ExecutionAttemptRow)
            if project_id:
                query = query.join(
                    ExperimentNodeRow,
                    ExecutionAttemptRow.node_id == ExperimentNodeRow.node_id,
                ).filter(ExperimentNodeRow.project_id == project_id)
            if node_id:
                query = query.filter(ExecutionAttemptRow.node_id == node_id)
            rows = (
                query.order_by(ExecutionAttemptRow.created_at.desc())
                .limit(limit)
                .all()
            )
            return [self._row_to_attempt(row) for row in rows]

    def latest_attempt_for_project(
        self, project_id: str
    ) -> ExecutionAttempt | None:
        attempts = self.list_attempts(limit=1, project_id=project_id)
        return attempts[0] if attempts else None

    def add_artifact(self, artifact: ExperimentArtifact) -> None:
        with self._session() as session:
            row = ExperimentArtifactRow(
                artifact_id=artifact.artifact_id,
                execution_id=artifact.execution_id,
                artifact_type=artifact.artifact_type,
                relative_path=artifact.relative_path,
                size_bytes=artifact.size_bytes,
                sha256=artifact.sha256,
                metadata_json=_dump(artifact.metadata_json),
                created_at=artifact.created_at,
            )
            session.add(row)
            session.commit()

    def list_artifacts(self, execution_id: str) -> list[ExperimentArtifact]:
        with self._session() as session:
            rows = (
                session.query(ExperimentArtifactRow)
                .filter(ExperimentArtifactRow.execution_id == execution_id)
                .all()
            )
            return [
                ExperimentArtifact(
                    artifact_id=row.artifact_id,
                    execution_id=row.execution_id,
                    artifact_type=row.artifact_type,
                    relative_path=row.relative_path,
                    size_bytes=row.size_bytes,
                    sha256=row.sha256,
                    metadata_json=_load(row.metadata_json),
                    created_at=row.created_at,
                )
                for row in rows
            ]

    @staticmethod
    def _row_to_attempt(row: ExecutionAttemptRow) -> ExecutionAttempt:
        return ExecutionAttempt(
            execution_id=row.execution_id,
            node_id=row.node_id,
            attempt_index=row.attempt_index,
            runner_profile=row.runner_profile,
            status=JobStatus(row.status),
            container_id=row.container_id,
            image_reference=row.image_reference,
            code_version=row.code_version,
            dataset_version=row.dataset_version,
            result_json=_load(row.result_json),
            error_json=_load(row.error_json),
            started_at=row.started_at,
            completed_at=row.completed_at,
            created_at=row.created_at,
        )
