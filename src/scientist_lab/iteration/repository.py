from __future__ import annotations

import json

from sqlalchemy.orm import Session, sessionmaker

from scientist_lab.iteration.models import ApprovalRecord, IterationSession
from scientist_lab.storage.database import ApprovalRecordRow, IterationSessionRow


def _dump_list(values: list) -> str:
    return json.dumps(values, ensure_ascii=False)


def _load_list(raw: str | None) -> list:
    if not raw:
        return []
    data = json.loads(raw)
    return data if isinstance(data, list) else []


class IterationRepository:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory

    def _session(self) -> Session:
        return self._session_factory()

    def save_session(self, session_obj: IterationSession) -> None:
        with self._session() as session:
            row = session.get(IterationSessionRow, session_obj.iteration_id)
            if row is None:
                row = IterationSessionRow(iteration_id=session_obj.iteration_id)
                session.add(row)
            row.project_id = session_obj.project_id
            row.source_baseline_node_id = session_obj.source_baseline_node_id
            row.source_candidate_node_id = session_obj.source_candidate_node_id
            row.proposed_node_id = session_obj.proposed_node_id
            row.status = session_obj.status
            row.seeds_json = _dump_list(session_obj.seeds)
            row.feedback_path = session_obj.feedback_path
            row.proposal_path = session_obj.proposal_path
            row.proposal_sha256 = session_obj.proposal_sha256
            row.approved_sha256 = session_obj.approved_sha256
            row.execution_ids_json = _dump_list(session_obj.execution_ids)
            row.comparison_paths_json = _dump_list(session_obj.comparison_paths)
            row.selected_node_id = session_obj.selected_node_id
            row.decision_id = session_obj.decision_id
            row.error_type = session_obj.error_type
            row.error_message = session_obj.error_message
            row.created_at = session_obj.created_at
            row.updated_at = session_obj.updated_at
            session.commit()

    def get_session(self, iteration_id: str) -> IterationSession | None:
        with self._session() as session:
            row = session.get(IterationSessionRow, iteration_id)
            if row is None:
                return None
            return self._row_to_session(row)

    def list_sessions(
        self,
        *,
        project_id: str | None = None,
        limit: int = 50,
    ) -> list[IterationSession]:
        with self._session() as session:
            query = session.query(IterationSessionRow)
            if project_id:
                query = query.filter(IterationSessionRow.project_id == project_id)
            rows = (
                query.order_by(IterationSessionRow.created_at.desc())
                .limit(limit)
                .all()
            )
            return [self._row_to_session(row) for row in rows]

    def save_approval(self, approval: ApprovalRecord) -> None:
        with self._session() as session:
            row = session.get(ApprovalRecordRow, approval.approval_id)
            if row is None:
                row = ApprovalRecordRow(approval_id=approval.approval_id)
                session.add(row)
            row.iteration_id = approval.iteration_id
            row.decision = approval.decision
            row.reason = approval.reason
            row.contract_path = approval.contract_path
            row.contract_sha256 = approval.contract_sha256
            row.created_at = approval.created_at
            session.commit()

    def list_approvals(self, iteration_id: str) -> list[ApprovalRecord]:
        with self._session() as session:
            rows = (
                session.query(ApprovalRecordRow)
                .filter(ApprovalRecordRow.iteration_id == iteration_id)
                .order_by(ApprovalRecordRow.created_at.asc())
                .all()
            )
            return [
                ApprovalRecord(
                    approval_id=row.approval_id,
                    iteration_id=row.iteration_id,
                    decision=row.decision,  # type: ignore[arg-type]
                    reason=row.reason,
                    contract_path=row.contract_path,
                    contract_sha256=row.contract_sha256,
                    created_at=row.created_at,
                )
                for row in rows
            ]

    @staticmethod
    def _row_to_session(row: IterationSessionRow) -> IterationSession:
        return IterationSession(
            iteration_id=row.iteration_id,
            project_id=row.project_id,
            source_baseline_node_id=row.source_baseline_node_id,
            source_candidate_node_id=row.source_candidate_node_id,
            proposed_node_id=row.proposed_node_id,
            status=row.status,  # type: ignore[arg-type]
            seeds=[int(v) for v in _load_list(row.seeds_json)],
            feedback_path=row.feedback_path,
            proposal_path=row.proposal_path,
            proposal_sha256=row.proposal_sha256,
            approved_sha256=row.approved_sha256,
            execution_ids=[str(v) for v in _load_list(row.execution_ids_json)],
            comparison_paths=[str(v) for v in _load_list(row.comparison_paths_json)],
            selected_node_id=row.selected_node_id,
            decision_id=row.decision_id,
            error_type=row.error_type,
            error_message=row.error_message,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )
