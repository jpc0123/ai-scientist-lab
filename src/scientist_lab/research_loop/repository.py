"""SQLite persistence for real research loop sessions and rounds (v2.1.1)."""

from __future__ import annotations

import json

from sqlalchemy import Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, sessionmaker

from scientist_lab.research_loop.models import (
    FeedbackUsageRecord,
    RealResearchLoopSession,
    ResearchLoopRound,
)
from scientist_lab.storage.database import Base


class RealResearchLoopSessionRow(Base):
    __tablename__ = "real_research_loop_sessions"

    session_id: Mapped[str] = mapped_column(String, primary_key=True)
    project_id: Mapped[str] = mapped_column(String, nullable=False)
    profile_id: Mapped[str] = mapped_column(String, nullable=False)
    protocol_id: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    current_round: Mapped[int] = mapped_column(Integer, nullable=False)
    required_rounds: Mapped[int] = mapped_column(Integer, nullable=False)
    fallback_allowed: Mapped[int] = mapped_column(Integer, nullable=False)
    fallback_used: Mapped[int] = mapped_column(Integer, nullable=False)
    session_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[str] = mapped_column(String, nullable=False)
    completed_at: Mapped[str | None] = mapped_column(String, nullable=True)


class RealResearchLoopRoundRow(Base):
    __tablename__ = "real_research_loop_rounds"

    round_id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    round_number: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    context_id: Mapped[str] = mapped_column(String, nullable=False)
    planner_call_id: Mapped[str | None] = mapped_column(String, nullable=True)
    critic_call_ids_json: Mapped[str] = mapped_column(Text, nullable=False)
    candidate_ids_json: Mapped[str] = mapped_column(Text, nullable=False)
    approved_candidate_id: Mapped[str | None] = mapped_column(String, nullable=True)
    execution_node_id: Mapped[str | None] = mapped_column(String, nullable=True)
    evidence_ids_json: Mapped[str] = mapped_column(Text, nullable=False)
    round_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    completed_at: Mapped[str | None] = mapped_column(String, nullable=True)


class FeedbackUsageRecordRow(Base):
    __tablename__ = "feedback_usage_records"

    feedback_usage_id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    source_round_id: Mapped[str] = mapped_column(String, nullable=False)
    target_round_id: Mapped[str] = mapped_column(String, nullable=False)
    verified: Mapped[int] = mapped_column(Integer, nullable=False)
    record_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)


def ensure_research_loop_schema(engine) -> None:
    Base.metadata.create_all(
        engine,
        tables=[
            RealResearchLoopSessionRow.__table__,
            RealResearchLoopRoundRow.__table__,
            FeedbackUsageRecordRow.__table__,
        ],
    )


def _dump_list(values: list) -> str:
    return json.dumps(values, ensure_ascii=False)


class RealResearchLoopRepository:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory

    def upsert_session(self, session_obj: RealResearchLoopSession) -> RealResearchLoopSession:
        session_obj.touch()
        with self._session_factory() as db:
            row = db.get(RealResearchLoopSessionRow, session_obj.session_id)
            payload = {
                "session_id": session_obj.session_id,
                "project_id": session_obj.project_id,
                "profile_id": session_obj.provider_profile_id,
                "protocol_id": session_obj.protocol_id,
                "status": session_obj.status,
                "current_round": int(session_obj.current_round),
                "required_rounds": int(session_obj.required_rounds),
                "fallback_allowed": 1 if session_obj.fallback_allowed else 0,
                "fallback_used": 1 if session_obj.fallback_used else 0,
                "session_json": session_obj.model_dump_json(),
                "created_at": session_obj.created_at,
                "updated_at": session_obj.updated_at,
                "completed_at": session_obj.completed_at,
            }
            if row is None:
                db.add(RealResearchLoopSessionRow(**payload))
            else:
                for key, value in payload.items():
                    setattr(row, key, value)
            db.commit()
        return session_obj

    def get_session(self, session_id: str) -> RealResearchLoopSession | None:
        with self._session_factory() as db:
            row = db.get(RealResearchLoopSessionRow, session_id)
            if row is None:
                return None
            return RealResearchLoopSession.model_validate_json(row.session_json)

    def require_session(self, session_id: str) -> RealResearchLoopSession:
        item = self.get_session(session_id)
        if item is None:
            raise KeyError(f"real-loop session not found: {session_id}")
        return item

    def list_sessions(
        self,
        *,
        project_id: str | None = None,
        limit: int = 50,
    ) -> list[RealResearchLoopSession]:
        with self._session_factory() as db:
            query = db.query(RealResearchLoopSessionRow)
            if project_id:
                query = query.filter(RealResearchLoopSessionRow.project_id == project_id)
            rows = (
                query.order_by(RealResearchLoopSessionRow.updated_at.desc())
                .limit(limit)
                .all()
            )
            return [
                RealResearchLoopSession.model_validate_json(row.session_json)
                for row in rows
            ]

    def upsert_round(self, round_obj: ResearchLoopRound) -> ResearchLoopRound:
        with self._session_factory() as db:
            row = db.get(RealResearchLoopRoundRow, round_obj.round_id)
            payload = {
                "round_id": round_obj.round_id,
                "session_id": round_obj.session_id,
                "round_number": int(round_obj.round_number),
                "status": round_obj.status,
                "context_id": round_obj.planning_context_id or "",
                "planner_call_id": round_obj.planner_call_id,
                "critic_call_ids_json": _dump_list(round_obj.critic_call_ids),
                "candidate_ids_json": _dump_list(round_obj.candidate_ids),
                "approved_candidate_id": round_obj.approved_candidate_id,
                "execution_node_id": round_obj.execution_node_id,
                "evidence_ids_json": _dump_list(round_obj.evidence_ids),
                "round_json": round_obj.model_dump_json(),
                "created_at": round_obj.created_at,
                "completed_at": round_obj.completed_at,
            }
            if row is None:
                db.add(RealResearchLoopRoundRow(**payload))
            else:
                for key, value in payload.items():
                    setattr(row, key, value)
            db.commit()
        return round_obj

    def list_rounds(self, session_id: str) -> list[ResearchLoopRound]:
        with self._session_factory() as db:
            rows = (
                db.query(RealResearchLoopRoundRow)
                .filter(RealResearchLoopRoundRow.session_id == session_id)
                .order_by(RealResearchLoopRoundRow.round_number.asc())
                .all()
            )
            return [
                ResearchLoopRound.model_validate_json(row.round_json) for row in rows
            ]

    def get_round(self, round_id: str) -> ResearchLoopRound | None:
        with self._session_factory() as db:
            row = db.get(RealResearchLoopRoundRow, round_id)
            if row is None:
                return None
            return ResearchLoopRound.model_validate_json(row.round_json)

    def upsert_feedback_usage(self, record: FeedbackUsageRecord) -> FeedbackUsageRecord:
        with self._session_factory() as db:
            row = db.get(FeedbackUsageRecordRow, record.feedback_usage_id)
            payload = {
                "feedback_usage_id": record.feedback_usage_id,
                "session_id": record.session_id,
                "source_round_id": record.source_round_id,
                "target_round_id": record.target_round_id,
                "verified": 1 if record.verified else 0,
                "record_json": record.model_dump_json(),
                "created_at": record.created_at,
            }
            if row is None:
                db.add(FeedbackUsageRecordRow(**payload))
            else:
                for key, value in payload.items():
                    setattr(row, key, value)
            db.commit()
        return record

    def get_feedback_usage(self, feedback_usage_id: str) -> FeedbackUsageRecord | None:
        with self._session_factory() as db:
            row = db.get(FeedbackUsageRecordRow, feedback_usage_id)
            if row is None:
                return None
            return FeedbackUsageRecord.model_validate_json(row.record_json)

    def list_feedback_usage(
        self, session_id: str, *, limit: int = 20
    ) -> list[FeedbackUsageRecord]:
        with self._session_factory() as db:
            rows = (
                db.query(FeedbackUsageRecordRow)
                .filter(FeedbackUsageRecordRow.session_id == session_id)
                .order_by(FeedbackUsageRecordRow.created_at.desc())
                .limit(limit)
                .all()
            )
            return [
                FeedbackUsageRecord.model_validate_json(row.record_json) for row in rows
            ]

    def latest_feedback_usage(
        self, session_id: str, *, target_round_number: int | None = None
    ) -> FeedbackUsageRecord | None:
        items = self.list_feedback_usage(session_id, limit=50)
        if target_round_number is None:
            return items[0] if items else None
        for item in items:
            if int(item.target_round_number) == int(target_round_number):
                return item
        return None
