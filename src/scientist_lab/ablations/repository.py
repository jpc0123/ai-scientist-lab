from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column, sessionmaker

from scientist_lab.ablations.models import AblationPlan
from scientist_lab.storage.database import Base


class AblationPlanRow(Base):
    __tablename__ = "ablation_plans"

    ablation_id: Mapped[str] = mapped_column(String, primary_key=True)
    project_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    protocol_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    reference_node_id: Mapped[str] = mapped_column(String, nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)


def ensure_ablation_schema(engine) -> None:
    Base.metadata.create_all(engine, tables=[AblationPlanRow.__table__])


class AblationRepository:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory

    def upsert(self, plan: AblationPlan) -> AblationPlan:
        with self._session_factory() as session:
            row = session.get(AblationPlanRow, plan.ablation_id)
            payload = self._to_row_dict(plan)
            if row is None:
                session.add(AblationPlanRow(**payload))
            else:
                for key, value in payload.items():
                    setattr(row, key, value)
            session.commit()
        return plan

    def get(self, ablation_id: str) -> AblationPlan | None:
        with self._session_factory() as session:
            row = session.get(AblationPlanRow, ablation_id)
            return self._from_row(row) if row else None

    def list_plans(
        self,
        *,
        project_id: str | None = None,
        protocol_id: str | None = None,
    ) -> list[AblationPlan]:
        from sqlalchemy import select

        with self._session_factory() as session:
            stmt = select(AblationPlanRow)
            if project_id:
                stmt = stmt.where(AblationPlanRow.project_id == project_id)
            if protocol_id:
                stmt = stmt.where(AblationPlanRow.protocol_id == protocol_id)
            stmt = stmt.order_by(
                AblationPlanRow.created_at.desc(),
                AblationPlanRow.ablation_id,
            )
            rows = session.scalars(stmt).all()
            return [self._from_row(row) for row in rows]

    @staticmethod
    def _to_row_dict(plan: AblationPlan) -> dict[str, Any]:
        created = plan.created_at
        if isinstance(created, datetime):
            created_at = created.replace(microsecond=0).isoformat()
        elif created:
            created_at = str(created)
        else:
            from datetime import timezone

            created_at = (
                datetime.now(timezone.utc).replace(microsecond=0).isoformat()
            )
        return {
            "ablation_id": plan.ablation_id,
            "project_id": plan.project_id,
            "protocol_id": plan.protocol_id,
            "reference_node_id": plan.reference_node_id,
            "payload_json": plan.model_dump_json(),
            "created_at": created_at,
        }

    @staticmethod
    def _from_row(row: AblationPlanRow) -> AblationPlan:
        data = json.loads(row.payload_json)
        return AblationPlan.model_validate(data)
