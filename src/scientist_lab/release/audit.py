"""Lightweight audit trail helpers for merge/release actions (v1.9.8)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scientist_lab.domain.models import utc_now_iso


def append_audit_event(
    outputs_root: Path,
    *,
    project_id: str,
    event_type: str,
    payload: dict[str, Any],
) -> Path:
    """Append one JSONL audit event under outputs/<project>/audit_trail/."""
    root = Path(outputs_root) / (project_id or "_release") / "audit_trail"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "events.jsonl"
    record = {
        "event_type": event_type,
        "recorded_at": utc_now_iso(),
        "payload": payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return path
