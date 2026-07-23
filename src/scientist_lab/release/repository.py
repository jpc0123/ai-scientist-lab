"""SQLite persistence for ReleasePackage (v1.8.1)."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import String, Text, select
from sqlalchemy.orm import Mapped, mapped_column, sessionmaker

from scientist_lab.release.models import ReleasePackage
from scientist_lab.storage.database import Base


class ReleasePackageRow(Base):
    __tablename__ = "release_packages"

    release_id: Mapped[str] = mapped_column(String, primary_key=True)
    project_id: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False, default="")
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[str] = mapped_column(String, nullable=False)


def ensure_release_schema(engine) -> None:
    Base.metadata.create_all(engine, tables=[ReleasePackageRow.__table__])


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class ReleaseRepository:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory

    def upsert(self, package: ReleasePackage) -> ReleasePackage:
        package.touch()
        with self._session_factory() as session:
            row = session.get(ReleasePackageRow, package.release_id)
            payload = {
                "release_id": package.release_id,
                "project_id": package.project_id,
                "status": package.status,
                "title": package.title or "",
                "payload_json": package.model_dump_json(),
                "created_at": package.created_at or _now(),
                "updated_at": package.updated_at or _now(),
            }
            if row is None:
                session.add(ReleasePackageRow(**payload))
            else:
                for key, value in payload.items():
                    setattr(row, key, value)
            session.commit()
        return package

    def get(self, release_id: str) -> ReleasePackage | None:
        with self._session_factory() as session:
            row = session.get(ReleasePackageRow, release_id)
            if row is None:
                return None
            return ReleasePackage.model_validate_json(row.payload_json)

    def require(self, release_id: str) -> ReleasePackage:
        item = self.get(release_id)
        if item is None:
            raise KeyError(f"release not found: {release_id}")
        return item

    def list_all(
        self, *, project_id: str | None = None, limit: int = 100
    ) -> list[ReleasePackage]:
        with self._session_factory() as session:
            stmt = select(ReleasePackageRow)
            if project_id:
                stmt = stmt.where(ReleasePackageRow.project_id == project_id)
            rows = list(session.scalars(stmt).all())
            rows.sort(key=lambda r: r.updated_at, reverse=True)
            return [
                ReleasePackage.model_validate_json(row.payload_json)
                for row in rows[: max(1, int(limit))]
            ]
