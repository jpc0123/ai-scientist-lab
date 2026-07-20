from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from scientist_worker.models import JobRecord, utc_now_iso


class WorkerJobRepository:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS worker_jobs (
                    job_id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL UNIQUE,
                    execution_id TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL,
                    contract_json TEXT NOT NULL,
                    environment_key TEXT NOT NULL,
                    container_id TEXT,
                    output_path TEXT NOT NULL,
                    log_path TEXT NOT NULL,
                    error_type TEXT,
                    error_message TEXT,
                    progress REAL,
                    stage TEXT,
                    submitted_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    updated_at TEXT NOT NULL,
                    result_json TEXT
                )
                """
            )
            conn.commit()

    def get_by_request_id(self, request_id: str) -> JobRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM worker_jobs WHERE request_id = ?",
                (request_id,),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def get_by_job_id(self, job_id: str) -> JobRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM worker_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def get_by_execution_id(self, execution_id: str) -> JobRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM worker_jobs WHERE execution_id = ?",
                (execution_id,),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def insert(self, record: JobRecord) -> JobRecord:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO worker_jobs (
                    job_id, request_id, execution_id, status, contract_json,
                    environment_key, container_id, output_path, log_path,
                    error_type, error_message, progress, stage,
                    submitted_at, started_at, finished_at, updated_at, result_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.job_id,
                    record.request_id,
                    record.execution_id,
                    record.status,
                    json.dumps(record.contract_json, ensure_ascii=False),
                    record.environment_key,
                    record.container_id,
                    record.output_path,
                    record.log_path,
                    record.error_type,
                    record.error_message,
                    record.progress,
                    record.stage,
                    record.submitted_at,
                    record.started_at,
                    record.finished_at,
                    record.updated_at,
                    json.dumps(record.result_json, ensure_ascii=False)
                    if record.result_json is not None
                    else None,
                ),
            )
            conn.commit()
        return record

    def update(self, record: JobRecord) -> JobRecord:
        record.updated_at = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE worker_jobs SET
                    status=?,
                    container_id=?,
                    error_type=?,
                    error_message=?,
                    progress=?,
                    stage=?,
                    started_at=?,
                    finished_at=?,
                    updated_at=?,
                    result_json=?
                WHERE job_id=?
                """,
                (
                    record.status,
                    record.container_id,
                    record.error_type,
                    record.error_message,
                    record.progress,
                    record.stage,
                    record.started_at,
                    record.finished_at,
                    record.updated_at,
                    json.dumps(record.result_json, ensure_ascii=False)
                    if record.result_json is not None
                    else None,
                    record.job_id,
                ),
            )
            conn.commit()
        return record

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> JobRecord:
        data: dict[str, Any] = dict(row)
        data["contract_json"] = json.loads(data["contract_json"])
        if data.get("result_json"):
            data["result_json"] = json.loads(data["result_json"])
        return JobRecord.model_validate(data)
