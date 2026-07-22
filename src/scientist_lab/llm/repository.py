"""Filesystem persistence for LLM call audit / replay records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scientist_lab.llm.models import LLMCallRecord
from scientist_lab.storage.artifact_store import write_json


class LLMCallRepository:
    """Store call records under ``<root>/llm_audit/`` for audit + replay."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.calls_dir = self.root / "llm_audit" / "calls"
        self.index_dir = self.root / "llm_audit" / "by_fingerprint"
        self.calls_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def save(self, record: LLMCallRecord) -> LLMCallRecord:
        path = self.calls_dir / f"{record.request_id}.json"
        payload = record.model_dump(mode="json")
        payload["path"] = str(path)
        write_json(path, payload)
        index_path = self.index_dir / f"{record.request_fingerprint}.json"
        write_json(
            index_path,
            {
                "request_fingerprint": record.request_fingerprint,
                "request_id": record.request_id,
                "path": str(path),
                "purpose": record.purpose,
                "provider": record.provider,
                "model": record.model,
            },
        )
        return record.model_copy(update={"path": str(path)})

    def get(self, request_id: str) -> LLMCallRecord | None:
        path = self.calls_dir / f"{request_id}.json"
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return LLMCallRecord.model_validate(data)

    def get_by_fingerprint(self, fingerprint: str) -> LLMCallRecord | None:
        index_path = self.index_dir / f"{fingerprint}.json"
        if not index_path.is_file():
            return None
        index = json.loads(index_path.read_text(encoding="utf-8"))
        request_id = str(index.get("request_id") or "")
        if not request_id:
            return None
        return self.get(request_id)

    def require_by_fingerprint(self, fingerprint: str) -> LLMCallRecord:
        record = self.get_by_fingerprint(fingerprint)
        if record is None:
            raise KeyError(f"no LLM call recorded for fingerprint: {fingerprint}")
        return record

    def list_calls(self, *, limit: int = 100) -> list[LLMCallRecord]:
        paths = sorted(self.calls_dir.glob("call_*.json"), reverse=True)
        # Also accept request_* and llm_* prefixes from new_id.
        if not paths:
            paths = sorted(self.calls_dir.glob("*.json"), reverse=True)
        out: list[LLMCallRecord] = []
        for path in paths[: max(1, int(limit))]:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                out.append(LLMCallRecord.model_validate(data))
            except Exception:  # noqa: BLE001
                continue
        return out
