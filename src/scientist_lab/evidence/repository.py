from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column, sessionmaker

from scientist_lab.evidence.models import ClaimSupportMatrix, EvidenceRecord
from scientist_lab.storage.database import Base


class EvidenceRecordRow(Base):
    __tablename__ = "evidence_records"

    evidence_id: Mapped[str] = mapped_column(String, primary_key=True)
    project_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    evidence_type: Mapped[str] = mapped_column(String, nullable=False)
    protocol_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    evidence_strength: Mapped[str] = mapped_column(String, nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)


class ClaimSupportMatrixRow(Base):
    __tablename__ = "claim_support_matrices"

    project_id: Mapped[str] = mapped_column(String, primary_key=True)
    protocol_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[str] = mapped_column(String, nullable=False)


def ensure_evidence_schema(engine) -> None:
    Base.metadata.create_all(
        engine,
        tables=[EvidenceRecordRow.__table__, ClaimSupportMatrixRow.__table__],
    )


class EvidenceRepository:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory

    def upsert(self, record: EvidenceRecord) -> EvidenceRecord:
        with self._session_factory() as session:
            row = session.get(EvidenceRecordRow, record.evidence_id)
            payload = self._to_row_dict(record)
            if row is None:
                session.add(EvidenceRecordRow(**payload))
            else:
                for key, value in payload.items():
                    setattr(row, key, value)
            session.commit()
        return record

    def get(self, evidence_id: str) -> EvidenceRecord | None:
        with self._session_factory() as session:
            row = session.get(EvidenceRecordRow, evidence_id)
            return self._from_row(row) if row else None

    def list_evidence(
        self,
        *,
        project_id: str | None = None,
        protocol_id: str | None = None,
        evidence_type: str | None = None,
    ) -> list[EvidenceRecord]:
        from sqlalchemy import select

        with self._session_factory() as session:
            stmt = select(EvidenceRecordRow)
            if project_id:
                stmt = stmt.where(EvidenceRecordRow.project_id == project_id)
            if protocol_id:
                stmt = stmt.where(EvidenceRecordRow.protocol_id == protocol_id)
            if evidence_type:
                stmt = stmt.where(EvidenceRecordRow.evidence_type == evidence_type)
            stmt = stmt.order_by(
                EvidenceRecordRow.created_at.desc(),
                EvidenceRecordRow.evidence_id,
            )
            rows = session.scalars(stmt).all()
            return [self._from_row(row) for row in rows]

    def upsert_claim_matrix(self, matrix: ClaimSupportMatrix) -> ClaimSupportMatrix:
        from datetime import timezone

        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        created = matrix.created_at
        if isinstance(created, datetime):
            created_at = created.replace(microsecond=0).isoformat()
        elif created:
            created_at = str(created)
        else:
            created_at = now

        with self._session_factory() as session:
            row = session.get(ClaimSupportMatrixRow, matrix.project_id)
            payload = {
                "project_id": matrix.project_id,
                "protocol_id": matrix.protocol_id,
                "payload_json": matrix.model_dump_json(),
                "created_at": created_at if row is None else row.created_at,
                "updated_at": now,
            }
            if row is None:
                session.add(ClaimSupportMatrixRow(**payload))
            else:
                for key, value in payload.items():
                    setattr(row, key, value)
            session.commit()
        return matrix

    def get_claim_matrix(self, project_id: str) -> ClaimSupportMatrix | None:
        with self._session_factory() as session:
            row = session.get(ClaimSupportMatrixRow, project_id)
            if row is None:
                return None
            return ClaimSupportMatrix.model_validate(json.loads(row.payload_json))

    @staticmethod
    def _to_row_dict(record: EvidenceRecord) -> dict[str, Any]:
        created = record.created_at
        if isinstance(created, datetime):
            created_at = created.replace(microsecond=0).isoformat()
        elif created:
            created_at = str(created)
        else:
            from datetime import timezone

            created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        return {
            "evidence_id": record.evidence_id,
            "project_id": record.project_id,
            "evidence_type": record.evidence_type,
            "protocol_id": record.protocol_id,
            "evidence_strength": record.evidence_strength,
            "payload_json": record.model_dump_json(),
            "created_at": created_at,
        }

    @staticmethod
    def _from_row(row: EvidenceRecordRow) -> EvidenceRecord:
        return EvidenceRecord.model_validate(json.loads(row.payload_json))
