from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, sessionmaker

from scientist_lab.budget.models import ExperimentBudget
from scientist_lab.storage.database import Base


class ExperimentBudgetRow(Base):
    __tablename__ = "experiment_budgets"

    project_id: Mapped[str] = mapped_column(String, primary_key=True)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[str] = mapped_column(String, nullable=False)


def ensure_budget_schema(engine) -> None:
    Base.metadata.create_all(engine, tables=[ExperimentBudgetRow.__table__])


class BudgetRepository:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory

    def upsert(self, budget: ExperimentBudget) -> ExperimentBudget:
        now = datetime.now(timezone.utc).replace(microsecond=0)
        budget = budget.model_copy(update={"updated_at": now})
        with self._session_factory() as session:
            row = session.get(ExperimentBudgetRow, budget.project_id)
            payload = {
                "project_id": budget.project_id,
                "payload_json": budget.model_dump_json(),
                "updated_at": now.isoformat(),
            }
            if row is None:
                session.add(ExperimentBudgetRow(**payload))
            else:
                for key, value in payload.items():
                    setattr(row, key, value)
            session.commit()
        return budget

    def get(self, project_id: str) -> ExperimentBudget | None:
        with self._session_factory() as session:
            row = session.get(ExperimentBudgetRow, project_id)
            if row is None:
                return None
            return ExperimentBudget.model_validate(json.loads(row.payload_json))
