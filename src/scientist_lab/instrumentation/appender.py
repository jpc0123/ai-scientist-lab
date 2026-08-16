"""Append-only Canonical Research Events. Not a domain aggregate.

MemoryWriter consumes these events; it does not own them.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from scientist_lab.core.schema_registry import validate_named


class EventAppender:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(event)
        payload.setdefault("schema_version", "1.0.0")
        payload.setdefault("event_id", f"evt_{uuid4().hex[:12]}")
        payload.setdefault("ts", datetime.now(timezone.utc).isoformat())
        payload.setdefault("reconstructed", False)
        validate_named("research_event", payload)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return payload

    def load_all(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows
