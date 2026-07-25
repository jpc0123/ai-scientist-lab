"""SQLite + filesystem persistence for CodeContextBundle (v2.2.1)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import String, Text, select
from sqlalchemy.orm import Mapped, mapped_column, sessionmaker

from scientist_lab.patching.context_models import CodeContextBundle
from scientist_lab.storage.artifact_store import write_json
from scientist_lab.storage.database import Base


class CodeContextBundleRow(Base):
    __tablename__ = "code_context_bundles"

    bundle_id: Mapped[str] = mapped_column(String, primary_key=True)
    request_id: Mapped[str] = mapped_column(String, nullable=False)
    project_id: Mapped[str] = mapped_column(String, nullable=False)
    source_commit: Mapped[str] = mapped_column(String, nullable=False, default="")
    context_sha256: Mapped[str] = mapped_column(String, nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[str] = mapped_column(String, nullable=False)


def ensure_context_schema(engine) -> None:
    Base.metadata.create_all(engine, tables=[CodeContextBundleRow.__table__])


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class CodeContextRepository:
    """Persist and reload CodeContextBundle for audit / restart reproducibility."""

    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory

    def upsert(self, bundle: CodeContextBundle) -> CodeContextBundle:
        bundle.touch()
        with self._session_factory() as session:
            row = session.get(CodeContextBundleRow, bundle.bundle_id)
            payload = {
                "bundle_id": bundle.bundle_id,
                "request_id": bundle.request_id,
                "project_id": bundle.project_id,
                "source_commit": bundle.source_commit or "",
                "context_sha256": bundle.context_sha256,
                "payload_json": bundle.model_dump_json(),
                "created_at": bundle.created_at or _now(),
                "updated_at": _now(),
            }
            if row is None:
                session.add(CodeContextBundleRow(**payload))
            else:
                for key, value in payload.items():
                    setattr(row, key, value)
            session.commit()
        return bundle

    def get(self, bundle_id: str) -> CodeContextBundle | None:
        with self._session_factory() as session:
            row = session.get(CodeContextBundleRow, bundle_id)
            if row is None:
                return None
            return CodeContextBundle.model_validate_json(row.payload_json)

    def require(self, bundle_id: str) -> CodeContextBundle:
        item = self.get(bundle_id)
        if item is None:
            raise KeyError(f"code context bundle not found: {bundle_id}")
        return item

    def get_by_sha256(self, context_sha256: str) -> CodeContextBundle | None:
        with self._session_factory() as session:
            stmt = select(CodeContextBundleRow).where(
                CodeContextBundleRow.context_sha256 == context_sha256
            )
            row = session.scalars(stmt).first()
            if row is None:
                return None
            return CodeContextBundle.model_validate_json(row.payload_json)

    def list_for_project(
        self, project_id: str, *, limit: int = 50
    ) -> list[CodeContextBundle]:
        with self._session_factory() as session:
            stmt = select(CodeContextBundleRow).where(
                CodeContextBundleRow.project_id == project_id
            )
            rows = list(session.scalars(stmt).all())
            rows.sort(key=lambda r: r.updated_at, reverse=True)
            return [
                CodeContextBundle.model_validate_json(row.payload_json)
                for row in rows[: max(1, int(limit))]
            ]


def export_code_context_bundle(
    bundle: CodeContextBundle,
    *,
    output_path: Path,
) -> dict[str, Any]:
    """Write a portable JSON export (audit / replay input later)."""
    path = Path(output_path)
    payload = bundle.model_dump(mode="json")
    write_json(path, payload)
    return {
        "path": str(path.resolve()),
        "bundle_id": bundle.bundle_id,
        "context_sha256": bundle.context_sha256,
        "source_commit": bundle.source_commit,
        "snapshot_count": len(bundle.snapshots),
        "total_bytes": bundle.total_bytes,
    }


def load_exported_code_context_bundle(path: Path) -> CodeContextBundle:
    text = Path(path).read_text(encoding="utf-8")
    return CodeContextBundle.model_validate_json(text)
