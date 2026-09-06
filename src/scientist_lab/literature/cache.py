"""Local literature cache + provenance. No secrets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scientist_lab.domain.models import new_id
from scientist_lab.literature.models import LiteratureQueryRecord, PaperRecord


def default_cache_dir() -> Path:
    return Path(".literature_cache")


def cache_key(
    *,
    provider: str,
    method: str,
    query: str,
    year_from: int | None,
    limit: int,
    paper_id: str = "",
) -> str:
    blob = json.dumps(
        {
            "provider": provider,
            "method": method,
            "query": query,
            "year_from": year_from,
            "limit": limit,
            "paper_id": paper_id,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


class LiteratureCache:
    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root is not None else default_cache_dir()

    def get(self, key: str) -> list[dict[str, Any]] | None:
        path = self.root / f"{key}.json"
        if not path.is_file():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        rows = raw.get("papers") if isinstance(raw, dict) else raw
        if not isinstance(rows, list):
            return None
        return [dict(row) for row in rows if isinstance(row, dict)]

    def put(self, key: str, papers: list[PaperRecord]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{key}.json"
        payload = {
            "papers": [paper.to_dict() for paper in papers],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_provenance(
    record: LiteratureQueryRecord,
    *,
    output_dir: Path | str,
) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{record.literature_query_id}.json"
    path.write_text(
        json.dumps(record.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    ledger = root / "literature_queries.jsonl"
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    return path


def new_query_id() -> str:
    return new_id("litq")
