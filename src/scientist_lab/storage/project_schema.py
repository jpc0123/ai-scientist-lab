"""Lightweight schema ensure for research_projects.payload_json (v2.0.1)."""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Engine


def ensure_project_schema(engine: Engine) -> None:
    """Add payload_json column when missing (existing local SQLite DBs)."""
    with engine.begin() as conn:
        rows = conn.execute(text("PRAGMA table_info(research_projects)")).fetchall()
        columns = {str(row[1]) for row in rows}
        if "payload_json" not in columns:
            conn.execute(
                text("ALTER TABLE research_projects ADD COLUMN payload_json TEXT")
            )
