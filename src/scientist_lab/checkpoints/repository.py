from __future__ import annotations

import json
from typing import Any

from sqlalchemy import Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, sessionmaker

from scientist_lab.checkpoints.models import CheckpointRecord
from scientist_lab.storage.database import Base


class CheckpointRow(Base):
    __tablename__ = "checkpoints"

    checkpoint_id: Mapped[str] = mapped_column(String, primary_key=True)
    project_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    execution_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    node_id: Mapped[str | None] = mapped_column(String, nullable=True)
    relative_path: Mapped[str] = mapped_column(Text, nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False)
    format: Mapped[str] = mapped_column(String, nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256: Mapped[str] = mapped_column(String, nullable=False)
    baseline_key: Mapped[str | None] = mapped_column(String, nullable=True)
    verified: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    verification_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[str] = mapped_column(String, nullable=False)


def ensure_checkpoint_schema(engine) -> None:
    Base.metadata.create_all(engine, tables=[CheckpointRow.__table__])


class CheckpointRepository:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory

    def upsert(self, record: CheckpointRecord) -> CheckpointRecord:
        with self._session_factory() as session:
            row = session.get(CheckpointRow, record.checkpoint_id)
            payload = self._to_row_dict(record)
            if row is None:
                session.add(CheckpointRow(**payload))
            else:
                for key, value in payload.items():
                    setattr(row, key, value)
            session.commit()
        return record

    def get(self, checkpoint_id: str) -> CheckpointRecord | None:
        with self._session_factory() as session:
            row = session.get(CheckpointRow, checkpoint_id)
            return self._from_row(row) if row else None

    def list_checkpoints(
        self,
        *,
        project_id: str | None = None,
        execution_id: str | None = None,
    ) -> list[CheckpointRecord]:
        from sqlalchemy import select

        with self._session_factory() as session:
            stmt = select(CheckpointRow)
            if project_id:
                stmt = stmt.where(CheckpointRow.project_id == project_id)
            if execution_id:
                stmt = stmt.where(CheckpointRow.execution_id == execution_id)
            stmt = stmt.order_by(CheckpointRow.created_at.desc(), CheckpointRow.checkpoint_id)
            rows = session.scalars(stmt).all()
            return [self._from_row(row) for row in rows]

    def get_by_execution_path(
        self, execution_id: str, relative_path: str
    ) -> CheckpointRecord | None:
        from sqlalchemy import select

        rel = relative_path.replace("\\", "/")
        with self._session_factory() as session:
            row = session.scalars(
                select(CheckpointRow).where(
                    CheckpointRow.execution_id == execution_id,
                    CheckpointRow.relative_path == rel,
                )
            ).first()
            return self._from_row(row) if row else None

    @staticmethod
    def _to_row_dict(record: CheckpointRecord) -> dict[str, Any]:
        return {
            "checkpoint_id": record.checkpoint_id,
            "project_id": record.project_id,
            "execution_id": record.execution_id,
            "node_id": record.node_id,
            "relative_path": record.relative_path,
            "role": record.role,
            "format": record.format,
            "size_bytes": int(record.size_bytes),
            "sha256": record.sha256,
            "baseline_key": record.baseline_key,
            "verified": 1 if record.verified else 0,
            "verification_message": record.verification_message,
            "metadata_json": json.dumps(record.metadata or {}, ensure_ascii=False),
            "created_at": str(record.created_at),
            "updated_at": str(record.updated_at),
        }

    @staticmethod
    def _from_row(row: CheckpointRow) -> CheckpointRecord:
        return CheckpointRecord(
            checkpoint_id=row.checkpoint_id,
            project_id=row.project_id,
            execution_id=row.execution_id,
            node_id=row.node_id,
            relative_path=row.relative_path,
            role=row.role,  # type: ignore[arg-type]
            format=row.format,  # type: ignore[arg-type]
            size_bytes=row.size_bytes,
            sha256=row.sha256,
            baseline_key=row.baseline_key,
            verified=bool(row.verified),
            verification_message=row.verification_message,
            metadata=json.loads(row.metadata_json or "{}"),
            created_at=row.created_at,
            updated_at=row.updated_at,
        )
