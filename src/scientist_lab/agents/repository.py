from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, sessionmaker

from scientist_lab.agents.models import ExperimentCandidateRecord, ExperimentPlanRecord
from scientist_lab.storage.database import Base


class ExperimentPlanRow(Base):
    __tablename__ = "experiment_plans"

    plan_id: Mapped[str] = mapped_column(String, primary_key=True)
    project_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    context_json: Mapped[str] = mapped_column(Text, nullable=False)
    planner_output_json: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    model_provider: Mapped[str | None] = mapped_column(String, nullable=True)
    model_name: Mapped[str | None] = mapped_column(String, nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String, nullable=True)
    context_sha256: Mapped[str | None] = mapped_column(String, nullable=True)
    output_sha256: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[str] = mapped_column(String, nullable=False)


class ExperimentCandidateRow(Base):
    __tablename__ = "experiment_candidates"

    candidate_id: Mapped[str] = mapped_column(String, primary_key=True)
    plan_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    candidate_json: Mapped[str] = mapped_column(Text, nullable=False)
    verification_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    critic_review_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    final_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)


def ensure_agent_plan_schema(engine) -> None:
    Base.metadata.create_all(
        engine,
        tables=[ExperimentPlanRow.__table__, ExperimentCandidateRow.__table__],
    )


class AgentPlanRepository:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory

    def upsert_plan(self, plan: ExperimentPlanRecord) -> ExperimentPlanRecord:
        with self._session_factory() as session:
            row = session.get(ExperimentPlanRow, plan.plan_id)
            payload = self._plan_to_row(plan)
            if row is None:
                session.add(ExperimentPlanRow(**payload))
            else:
                for key, value in payload.items():
                    setattr(row, key, value)
            session.commit()
        return plan

    def get_plan(self, plan_id: str) -> ExperimentPlanRecord | None:
        with self._session_factory() as session:
            row = session.get(ExperimentPlanRow, plan_id)
            return self._plan_from_row(row) if row else None

    def list_plans(
        self,
        *,
        project_id: str | None = None,
    ) -> list[ExperimentPlanRecord]:
        from sqlalchemy import select

        with self._session_factory() as session:
            stmt = select(ExperimentPlanRow)
            if project_id:
                stmt = stmt.where(ExperimentPlanRow.project_id == project_id)
            stmt = stmt.order_by(
                ExperimentPlanRow.created_at.desc(),
                ExperimentPlanRow.plan_id,
            )
            return [self._plan_from_row(row) for row in session.scalars(stmt).all()]

    def upsert_candidate(
        self, candidate: ExperimentCandidateRecord
    ) -> ExperimentCandidateRecord:
        with self._session_factory() as session:
            row = session.get(ExperimentCandidateRow, candidate.candidate_id)
            payload = self._candidate_to_row(candidate)
            if row is None:
                session.add(ExperimentCandidateRow(**payload))
            else:
                for key, value in payload.items():
                    setattr(row, key, value)
            session.commit()
        return candidate

    def list_candidates(self, plan_id: str) -> list[ExperimentCandidateRecord]:
        from sqlalchemy import select

        with self._session_factory() as session:
            stmt = (
                select(ExperimentCandidateRow)
                .where(ExperimentCandidateRow.plan_id == plan_id)
                .order_by(
                    ExperimentCandidateRow.candidate_id,
                )
            )
            return [
                self._candidate_from_row(row) for row in session.scalars(stmt).all()
            ]

    def get_candidate(self, candidate_id: str) -> ExperimentCandidateRecord | None:
        with self._session_factory() as session:
            row = session.get(ExperimentCandidateRow, candidate_id)
            return self._candidate_from_row(row) if row else None

    @staticmethod
    def _iso(value: datetime | None) -> str:
        if isinstance(value, datetime):
            return value.replace(microsecond=0).isoformat()
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    def _plan_to_row(self, plan: ExperimentPlanRecord) -> dict[str, Any]:
        return {
            "plan_id": plan.plan_id,
            "project_id": plan.project_id,
            "context_json": json.dumps(plan.context_json, ensure_ascii=False),
            "planner_output_json": json.dumps(
                plan.planner_output_json, ensure_ascii=False
            ),
            "status": plan.status,
            "model_provider": plan.model_provider,
            "model_name": plan.model_name,
            "prompt_version": plan.prompt_version,
            "context_sha256": plan.context_sha256,
            "output_sha256": plan.output_sha256,
            "created_at": self._iso(plan.created_at),
            "updated_at": self._iso(plan.updated_at),
        }

    @staticmethod
    def _plan_from_row(row: ExperimentPlanRow) -> ExperimentPlanRecord:
        return ExperimentPlanRecord(
            plan_id=row.plan_id,
            project_id=row.project_id,
            status=row.status,  # type: ignore[arg-type]
            context_json=json.loads(row.context_json or "{}"),
            planner_output_json=json.loads(row.planner_output_json or "{}"),
            model_provider=row.model_provider,
            model_name=row.model_name,
            prompt_version=row.prompt_version,
            context_sha256=row.context_sha256,
            output_sha256=row.output_sha256,
            created_at=datetime.fromisoformat(row.created_at),
            updated_at=datetime.fromisoformat(row.updated_at),
        )

    def _candidate_to_row(
        self, candidate: ExperimentCandidateRecord
    ) -> dict[str, Any]:
        return {
            "candidate_id": candidate.candidate_id,
            "plan_id": candidate.plan_id,
            "candidate_json": json.dumps(candidate.candidate_json, ensure_ascii=False),
            "verification_json": (
                json.dumps(candidate.verification_json, ensure_ascii=False)
                if candidate.verification_json is not None
                else None
            ),
            "critic_review_json": (
                json.dumps(candidate.critic_review_json, ensure_ascii=False)
                if candidate.critic_review_json is not None
                else None
            ),
            "final_score": candidate.final_score,
            "rank": candidate.rank,
            "status": candidate.status,
            "created_at": self._iso(candidate.created_at),
        }

    @staticmethod
    def _candidate_from_row(row: ExperimentCandidateRow) -> ExperimentCandidateRecord:
        return ExperimentCandidateRecord(
            candidate_id=row.candidate_id,
            plan_id=row.plan_id,
            candidate_json=json.loads(row.candidate_json or "{}"),
            verification_json=(
                json.loads(row.verification_json)
                if row.verification_json
                else None
            ),
            critic_review_json=(
                json.loads(row.critic_review_json)
                if row.critic_review_json
                else None
            ),
            final_score=row.final_score,
            rank=row.rank,
            status=row.status,  # type: ignore[arg-type]
            created_at=datetime.fromisoformat(row.created_at),
        )
