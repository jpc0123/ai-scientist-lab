from __future__ import annotations

import json

from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker

from scientist_lab.datasets.models import DatasetRegistration
from scientist_lab.storage.database import DatasetRegistrationRow


class DatasetRepository:
    """SQLite persistence for dataset registrations."""

    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory

    def _session(self) -> Session:
        return self._session_factory()

    def get(self, dataset_key: str) -> DatasetRegistration | None:
        with self._session() as session:
            row = session.get(DatasetRegistrationRow, dataset_key)
            if row is None:
                return None
            return self._row_to_model(row)

    def list_all(self) -> list[DatasetRegistration]:
        with self._session() as session:
            rows = (
                session.query(DatasetRegistrationRow)
                .order_by(DatasetRegistrationRow.updated_at.desc())
                .all()
            )
            return [self._row_to_model(row) for row in rows]

    def exists(self, dataset_key: str) -> bool:
        with self._session() as session:
            return session.get(DatasetRegistrationRow, dataset_key) is not None

    def insert(self, registration: DatasetRegistration) -> DatasetRegistration:
        with self._session() as session:
            if session.get(DatasetRegistrationRow, registration.dataset_key) is not None:
                raise ValueError(
                    f"dataset_key already registered: {registration.dataset_key}"
                )
            row = DatasetRegistrationRow(
                dataset_key=registration.dataset_key,
                task_type=registration.task_type,
                host_path=registration.host_path,
                container_path=registration.container_path,
                read_only=1 if registration.read_only else 0,
                enabled=1 if registration.enabled else 0,
                metadata_json=json.dumps(registration.metadata, ensure_ascii=False),
                created_at=_as_iso(registration.created_at),
                updated_at=_as_iso(registration.updated_at),
            )
            session.add(row)
            session.commit()
            return self._row_to_model(row)

    def update_flags(
        self,
        dataset_key: str,
        *,
        enabled: bool | None = None,
        updated_at: str,
    ) -> DatasetRegistration:
        with self._session() as session:
            row = session.get(DatasetRegistrationRow, dataset_key)
            if row is None:
                raise KeyError(f"数据集未注册：{dataset_key}")
            if enabled is not None:
                row.enabled = 1 if enabled else 0
            row.updated_at = updated_at
            session.commit()
            return self._row_to_model(row)

    @staticmethod
    def _row_to_model(row: DatasetRegistrationRow) -> DatasetRegistration:
        metadata: dict = {}
        if row.metadata_json:
            metadata = json.loads(row.metadata_json)
        enabled = True
        if hasattr(row, "enabled") and row.enabled is not None:
            enabled = bool(row.enabled)
        return DatasetRegistration(
            dataset_key=row.dataset_key,
            task_type=row.task_type,
            host_path=row.host_path,
            container_path=row.container_path,
            read_only=bool(row.read_only),
            enabled=enabled,
            metadata=metadata,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )


def ensure_dataset_schema(engine) -> None:
    """Add missing columns / index for older SQLite files."""
    with engine.begin() as conn:
        rows = conn.execute(text("PRAGMA table_info(dataset_registrations)")).fetchall()
        if not rows:
            return
        columns = {row[1] for row in rows}
        if "enabled" not in columns:
            conn.execute(
                text(
                    "ALTER TABLE dataset_registrations "
                    "ADD COLUMN enabled INTEGER NOT NULL DEFAULT 1"
                )
            )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_datasets_task_type "
                "ON dataset_registrations(task_type)"
            )
        )


def _as_iso(value: object) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()  # type: ignore[no-any-return]
    return str(value)
