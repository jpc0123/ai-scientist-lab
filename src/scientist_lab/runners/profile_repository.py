from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy import Integer, String, Text, create_engine
from sqlalchemy.orm import Mapped, Session, mapped_column, sessionmaker

from scientist_lab.runners.profiles import RunnerProfile
from scientist_lab.storage.database import Base


class RunnerProfileRow(Base):
    __tablename__ = "runner_profiles"

    profile_key: Mapped[str] = mapped_column(String, primary_key=True)
    runner_type: Mapped[str] = mapped_column(String, nullable=False)
    enabled: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    endpoint: Mapped[str | None] = mapped_column(Text, nullable=True)
    auth_token_env: Mapped[str | None] = mapped_column(String, nullable=True)
    allowed_environment_keys_json: Mapped[str] = mapped_column(Text, nullable=False)
    default_timeout_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    metadata_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[str] = mapped_column(String, nullable=False)


def ensure_runner_profile_schema(engine) -> None:
    Base.metadata.create_all(engine, tables=[RunnerProfileRow.__table__])


class RunnerProfileRepository:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory

    def upsert(self, profile: RunnerProfile) -> RunnerProfile:
        with self._session_factory() as session:
            row = session.get(RunnerProfileRow, profile.profile_key)
            payload = self._to_row_dict(profile)
            if row is None:
                session.add(RunnerProfileRow(**payload))
            else:
                for key, value in payload.items():
                    setattr(row, key, value)
            session.commit()
        return profile

    def get(self, profile_key: str) -> RunnerProfile | None:
        with self._session_factory() as session:
            row = session.get(RunnerProfileRow, profile_key)
            return self._from_row(row) if row else None

    def list_profiles(self) -> list[RunnerProfile]:
        from sqlalchemy import select

        with self._session_factory() as session:
            rows = session.scalars(
                select(RunnerProfileRow).order_by(RunnerProfileRow.profile_key)
            ).all()
            return [self._from_row(row) for row in rows]

    def delete(self, profile_key: str) -> bool:
        with self._session_factory() as session:
            row = session.get(RunnerProfileRow, profile_key)
            if row is None:
                return False
            session.delete(row)
            session.commit()
            return True

    @staticmethod
    def _to_row_dict(profile: RunnerProfile) -> dict[str, Any]:
        return {
            "profile_key": profile.profile_key,
            "runner_type": profile.runner_type,
            "enabled": 1 if profile.enabled else 0,
            "endpoint": profile.endpoint,
            "auth_token_env": profile.auth_token_env,
            "allowed_environment_keys_json": json.dumps(
                profile.allowed_environment_keys, ensure_ascii=False
            ),
            "default_timeout_seconds": int(profile.default_timeout_seconds),
            "metadata_json": json.dumps(profile.metadata or {}, ensure_ascii=False),
            "created_at": str(profile.created_at),
            "updated_at": str(profile.updated_at),
        }

    @staticmethod
    def _from_row(row: RunnerProfileRow) -> RunnerProfile:
        return RunnerProfile(
            profile_key=row.profile_key,
            runner_type=row.runner_type,  # type: ignore[arg-type]
            enabled=bool(row.enabled),
            endpoint=row.endpoint,
            auth_token_env=row.auth_token_env,
            allowed_environment_keys=json.loads(row.allowed_environment_keys_json),
            default_timeout_seconds=row.default_timeout_seconds,
            metadata=json.loads(row.metadata_json or "{}"),
            created_at=row.created_at,
            updated_at=row.updated_at,
        )
