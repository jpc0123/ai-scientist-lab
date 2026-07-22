"""SQLite persistence for PatchProposal (v1.6.1)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import String, Text, select
from sqlalchemy.orm import Mapped, mapped_column, sessionmaker

from scientist_lab.patching.models import PatchProposal
from scientist_lab.storage.database import Base


class PatchProposalRow(Base):
    __tablename__ = "patch_proposals"

    patch_id: Mapped[str] = mapped_column(String, primary_key=True)
    project_id: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    unified_diff: Mapped[str] = mapped_column(Text, nullable=False)
    files_json: Mapped[str] = mapped_column(Text, nullable=False)
    fingerprint_sha256: Mapped[str] = mapped_column(String, nullable=False)
    provider: Mapped[str] = mapped_column(String, nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[str] = mapped_column(String, nullable=False)


def ensure_patching_schema(engine) -> None:
    Base.metadata.create_all(engine, tables=[PatchProposalRow.__table__])


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class PatchRepository:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory

    def upsert(self, proposal: PatchProposal) -> PatchProposal:
        proposal.touch()
        with self._session_factory() as session:
            row = session.get(PatchProposalRow, proposal.patch_id)
            payload = {
                "patch_id": proposal.patch_id,
                "project_id": proposal.project_id,
                "status": proposal.status,
                "title": proposal.title,
                "rationale": proposal.rationale,
                "unified_diff": proposal.unified_diff,
                "files_json": json.dumps(proposal.files_touched, ensure_ascii=False),
                "fingerprint_sha256": proposal.fingerprint_sha256,
                "provider": proposal.provider,
                "payload_json": proposal.model_dump_json(),
                "created_at": proposal.created_at or _now(),
                "updated_at": proposal.updated_at or _now(),
            }
            if row is None:
                session.add(PatchProposalRow(**payload))
            else:
                for key, value in payload.items():
                    setattr(row, key, value)
            session.commit()
        return proposal

    def get(self, patch_id: str) -> PatchProposal | None:
        with self._session_factory() as session:
            row = session.get(PatchProposalRow, patch_id)
            if row is None:
                return None
            return PatchProposal.model_validate_json(row.payload_json)

    def require(self, patch_id: str) -> PatchProposal:
        item = self.get(patch_id)
        if item is None:
            raise KeyError(f"patch not found: {patch_id}")
        return item

    def find_by_fingerprint(
        self, fingerprint_sha256: str, *, exclude_patch_id: str | None = None
    ) -> PatchProposal | None:
        with self._session_factory() as session:
            stmt = select(PatchProposalRow).where(
                PatchProposalRow.fingerprint_sha256 == fingerprint_sha256
            )
            for row in session.scalars(stmt).all():
                if exclude_patch_id and row.patch_id == exclude_patch_id:
                    continue
                return PatchProposal.model_validate_json(row.payload_json)
        return None

    def list_for_project(self, project_id: str) -> list[PatchProposal]:
        with self._session_factory() as session:
            stmt = select(PatchProposalRow).where(
                PatchProposalRow.project_id == project_id
            )
            rows = list(session.scalars(stmt).all())
            rows.sort(key=lambda r: r.updated_at, reverse=True)
            return [
                PatchProposal.model_validate_json(row.payload_json) for row in rows
            ]
