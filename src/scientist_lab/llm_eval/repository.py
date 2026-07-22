"""SQLite persistence for LLM profiles and evaluation indexes (v1.5)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Integer, String, Text, select
from sqlalchemy.orm import Mapped, mapped_column, sessionmaker

from scientist_lab.llm_eval.profiles import LLMModelProfile
from scientist_lab.storage.database import Base


class LLMModelProfileRow(Base):
    __tablename__ = "llm_model_profiles"

    profile_id: Mapped[str] = mapped_column(String, primary_key=True)
    provider: Mapped[str] = mapped_column(String, nullable=False)
    model: Mapped[str] = mapped_column(String, nullable=False)
    api_mode: Mapped[str] = mapped_column(String, nullable=False)
    planner_prompt_version: Mapped[str] = mapped_column(String, nullable=False)
    critic_prompt_version: Mapped[str] = mapped_column(String, nullable=False)
    config_json: Mapped[str] = mapped_column(Text, nullable=False)
    enabled: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[str] = mapped_column(String, nullable=False)


class LLMEvaluationRow(Base):
    __tablename__ = "llm_evaluations"

    evaluation_id: Mapped[str] = mapped_column(String, primary_key=True)
    profile_id: Mapped[str] = mapped_column(String, nullable=False)
    suite_version: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    result_json: Mapped[str] = mapped_column(Text, nullable=False)
    report_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    completed_at: Mapped[str | None] = mapped_column(String, nullable=True)


class LLMEvaluationCaseRow(Base):
    __tablename__ = "llm_evaluation_cases"

    case_result_id: Mapped[str] = mapped_column(String, primary_key=True)
    evaluation_id: Mapped[str] = mapped_column(String, nullable=False)
    case_id: Mapped[str] = mapped_column(String, nullable=False)
    task_type: Mapped[str] = mapped_column(String, nullable=False)
    passed: Mapped[int] = mapped_column(Integer, nullable=False)
    scores_json: Mapped[str] = mapped_column(Text, nullable=False)
    issues_json: Mapped[str] = mapped_column(Text, nullable=False)
    call_id: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[str] = mapped_column(String, nullable=False)


class LLMRuntimeSettingRow(Base):
    __tablename__ = "llm_runtime_settings"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    value_json: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[str] = mapped_column(String, nullable=False)


def ensure_llm_eval_schema(engine) -> None:
    Base.metadata.create_all(
        engine,
        tables=[
            LLMModelProfileRow.__table__,
            LLMEvaluationRow.__table__,
            LLMEvaluationCaseRow.__table__,
            LLMRuntimeSettingRow.__table__,
        ],
    )


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class LLMEvalRepository:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory

    def upsert_profile(self, profile: LLMModelProfile) -> LLMModelProfile:
        now = _now()
        with self._session_factory() as session:
            row = session.get(LLMModelProfileRow, profile.profile_id)
            payload = {
                "profile_id": profile.profile_id,
                "provider": profile.provider,
                "model": profile.model,
                "api_mode": profile.api_mode,
                "planner_prompt_version": profile.planner_prompt_version,
                "critic_prompt_version": profile.critic_prompt_version,
                "config_json": json.dumps(profile.safe_dict(), ensure_ascii=False),
                "enabled": 1 if profile.enabled else 0,
                "updated_at": now,
            }
            if row is None:
                session.add(LLMModelProfileRow(created_at=now, **payload))
            else:
                for key, value in payload.items():
                    setattr(row, key, value)
            session.commit()
        return profile

    def get_profile(self, profile_id: str) -> LLMModelProfile | None:
        with self._session_factory() as session:
            row = session.get(LLMModelProfileRow, profile_id)
            if row is None:
                return None
            data = json.loads(row.config_json)
            return LLMModelProfile.model_validate(data)

    def list_profiles(self, *, enabled_only: bool = False) -> list[LLMModelProfile]:
        with self._session_factory() as session:
            rows = list(session.scalars(select(LLMModelProfileRow)).all())
            out: list[LLMModelProfile] = []
            for row in rows:
                if enabled_only and not row.enabled:
                    continue
                out.append(LLMModelProfile.model_validate(json.loads(row.config_json)))
            return out

    def save_evaluation(
        self,
        *,
        evaluation_id: str,
        profile_id: str,
        suite_version: str,
        status: str,
        result: dict[str, Any],
        report_path: str | None,
        case_rows: list[dict[str, Any]],
    ) -> None:
        now = _now()
        with self._session_factory() as session:
            existing = session.get(LLMEvaluationRow, evaluation_id)
            payload = {
                "evaluation_id": evaluation_id,
                "profile_id": profile_id,
                "suite_version": suite_version,
                "status": status,
                "result_json": json.dumps(result, ensure_ascii=False),
                "report_path": report_path,
                "completed_at": now,
            }
            if existing is None:
                session.add(LLMEvaluationRow(created_at=now, **payload))
            else:
                for key, value in payload.items():
                    setattr(existing, key, value)
            for item in case_rows:
                cid = str(item["case_result_id"])
                crow = session.get(LLMEvaluationCaseRow, cid)
                cpayload = {
                    "case_result_id": cid,
                    "evaluation_id": evaluation_id,
                    "case_id": item["case_id"],
                    "task_type": item["task_type"],
                    "passed": 1 if item["passed"] else 0,
                    "scores_json": json.dumps(item.get("scores") or [], ensure_ascii=False),
                    "issues_json": json.dumps(item.get("issues") or [], ensure_ascii=False),
                    "call_id": item.get("call_id"),
                    "created_at": now,
                }
                if crow is None:
                    session.add(LLMEvaluationCaseRow(**cpayload))
                else:
                    for key, value in cpayload.items():
                        setattr(crow, key, value)
            session.commit()

    def get_evaluation(self, evaluation_id: str) -> dict[str, Any] | None:
        with self._session_factory() as session:
            row = session.get(LLMEvaluationRow, evaluation_id)
            if row is None:
                return None
            return {
                "evaluation_id": row.evaluation_id,
                "profile_id": row.profile_id,
                "suite_version": row.suite_version,
                "status": row.status,
                "result": json.loads(row.result_json),
                "report_path": row.report_path,
                "created_at": row.created_at,
                "completed_at": row.completed_at,
            }

    def list_evaluations(
        self,
        *,
        profile_id: str | None = None,
        suite_version: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        with self._session_factory() as session:
            stmt = select(LLMEvaluationRow)
            if profile_id:
                stmt = stmt.where(LLMEvaluationRow.profile_id == profile_id)
            if suite_version:
                stmt = stmt.where(LLMEvaluationRow.suite_version == suite_version)
            rows = list(session.scalars(stmt).all())
            rows.sort(
                key=lambda r: (
                    r.completed_at or r.created_at or "",
                    r.evaluation_id,
                ),
                reverse=True,
            )
            out = []
            for row in rows[: max(1, int(limit))]:
                out.append(
                    {
                        "evaluation_id": row.evaluation_id,
                        "profile_id": row.profile_id,
                        "suite_version": row.suite_version,
                        "status": row.status,
                        "result": json.loads(row.result_json),
                        "report_path": row.report_path,
                        "created_at": row.created_at,
                        "completed_at": row.completed_at,
                    }
                )
            return out

    def latest_evaluation_for_profile(
        self, profile_id: str, *, suite_version: str | None = None
    ) -> dict[str, Any] | None:
        rows = self.list_evaluations(
            profile_id=profile_id, suite_version=suite_version, limit=1
        )
        return rows[0] if rows else None

    def set_default_profile(self, profile_id: str) -> str:
        profile = self.get_profile(profile_id)
        if profile is None:
            raise KeyError(f"llm profile not found: {profile_id}")
        if not profile.enabled:
            raise ValueError(f"llm profile is disabled: {profile_id}")
        now = _now()
        with self._session_factory() as session:
            row = session.get(LLMRuntimeSettingRow, "default_profile_id")
            payload = json.dumps({"profile_id": profile_id}, ensure_ascii=False)
            if row is None:
                session.add(
                    LLMRuntimeSettingRow(
                        key="default_profile_id",
                        value_json=payload,
                        updated_at=now,
                    )
                )
            else:
                row.value_json = payload
                row.updated_at = now
            session.commit()
        return profile_id

    def get_default_profile_id(self) -> str | None:
        with self._session_factory() as session:
            row = session.get(LLMRuntimeSettingRow, "default_profile_id")
            if row is None:
                return None
            data = json.loads(row.value_json)
            return data.get("profile_id")
