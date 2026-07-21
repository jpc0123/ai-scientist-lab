from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column, sessionmaker

from scientist_lab.protocols.models import ExperimentProtocol
from scientist_lab.storage.database import Base


class ExperimentProtocolRow(Base):
    __tablename__ = "experiment_protocols"

    protocol_id: Mapped[str] = mapped_column(String, primary_key=True)
    project_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)


def ensure_protocol_schema(engine) -> None:
    Base.metadata.create_all(engine, tables=[ExperimentProtocolRow.__table__])


class ProtocolRepository:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory

    def upsert(self, protocol: ExperimentProtocol) -> ExperimentProtocol:
        with self._session_factory() as session:
            row = session.get(ExperimentProtocolRow, protocol.protocol_id)
            payload = self._to_row_dict(protocol)
            if row is None:
                session.add(ExperimentProtocolRow(**payload))
            else:
                for key, value in payload.items():
                    setattr(row, key, value)
            session.commit()
        return protocol

    def get(self, protocol_id: str) -> ExperimentProtocol | None:
        with self._session_factory() as session:
            row = session.get(ExperimentProtocolRow, protocol_id)
            return self._from_row(row) if row else None

    def list_protocols(
        self,
        *,
        project_id: str | None = None,
    ) -> list[ExperimentProtocol]:
        from sqlalchemy import select

        with self._session_factory() as session:
            stmt = select(ExperimentProtocolRow)
            if project_id:
                stmt = stmt.where(ExperimentProtocolRow.project_id == project_id)
            stmt = stmt.order_by(
                ExperimentProtocolRow.created_at.desc(),
                ExperimentProtocolRow.protocol_id,
            )
            rows = session.scalars(stmt).all()
            return [self._from_row(row) for row in rows]

    @staticmethod
    def _to_row_dict(protocol: ExperimentProtocol) -> dict[str, Any]:
        created = protocol.created_at
        if isinstance(created, datetime):
            created_at = created.replace(microsecond=0).isoformat()
        else:
            created_at = str(created)
        return {
            "protocol_id": protocol.protocol_id,
            "project_id": protocol.project_id,
            "title": protocol.title,
            "payload_json": protocol.model_dump_json(),
            "created_at": created_at,
        }

    @staticmethod
    def _from_row(row: ExperimentProtocolRow) -> ExperimentProtocol:
        data = json.loads(row.payload_json)
        return ExperimentProtocol.model_validate(data)
