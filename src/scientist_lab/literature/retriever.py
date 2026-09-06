"""LiteratureRetriever facade. Not a fifth Agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from scientist_lab.literature.cache import (
    LiteratureCache,
    cache_key,
    new_query_id,
    write_provenance,
)
from scientist_lab.literature.factory import resolve_literature_provider
from scientist_lab.literature.gate import gate_paper
from scientist_lab.literature.models import LiteratureQueryRecord, PaperRecord
from scientist_lab.llm.http_transport import HttpTransport


class LiteratureRetriever:
    """search / get_paper / references / citations + cache + provenance + gate."""

    def __init__(
        self,
        *,
        live: bool = False,
        provider: str | None = None,
        cache_dir: Path | str | None = None,
        provenance_dir: Path | str | None = None,
        environ: Mapping[str, str] | None = None,
        transport: HttpTransport | None = None,
        use_cache: bool = True,
    ) -> None:
        self.live = bool(live)
        self.provider = resolve_literature_provider(
            live=self.live,
            provider=provider,
            environ=environ,
            transport=transport,
        )
        self.cache = LiteratureCache(cache_dir)
        self.provenance_dir = Path(provenance_dir) if provenance_dir else Path(".run") / "literature"
        self.use_cache = bool(use_cache)

    def _load_cached(self, key: str, query: str) -> list[PaperRecord] | None:
        if not self.use_cache:
            return None
        rows = self.cache.get(key)
        if rows is None:
            return None
        return [
            PaperRecord.from_mapping(
                row,
                retrieval_query=str(row.get("retrieval_query") or query),
                source=str(row.get("source") or self.provider.name),
                retrieved_at=row.get("retrieved_at"),
            )
            for row in rows
        ]

    def _store(self, key: str, papers: list[PaperRecord]) -> None:
        # Never cache empty live misses — polluted queries / transient S2 blanks
        # would otherwise stick and block a later cleaned query with the same key.
        if self.use_cache and papers:
            self.cache.put(key, papers)

    def _packet(
        self,
        papers: list[PaperRecord],
        *,
        query: str,
        round_id: str,
        used_by_plan: list[str] | None,
        year_from: int | None,
        limit: int | None,
        method: str,
        cached: bool,
    ) -> dict[str, Any]:
        gated = [gate_paper(paper) for paper in papers]
        refs = [str(item["paper"]["paper_id"]) for item in gated]
        used = [item for item in (used_by_plan or []) if item in refs]
        record = LiteratureQueryRecord(
            literature_query_id=new_query_id(),
            round_id=round_id,
            queries=[query],
            paper_refs=refs,
            used_by_plan=used,
            provider=self.provider.name,
            year_from=year_from,
            limit=limit,
        )
        provenance_path = write_provenance(record, output_dir=self.provenance_dir)
        return {
            "ok": True,
            "method": method,
            "cached": cached,
            "provider": self.provider.name,
            "live": self.live,
            "provenance": record.to_dict(),
            "provenance_path": str(provenance_path),
            "papers": gated,
            "planner_admissible": [row for row in gated if row.get("planner_admissible")],
            "rejected": [row for row in gated if not row.get("planner_admissible")],
            "claim_gate_note": (
                "LiteratureEvidence cannot enter ClaimGate; only ExperimentEvidence can."
            ),
        }

    def search(
        self,
        query: str,
        *,
        year_from: int | None = 2022,
        limit: int = 20,
        round_id: str = "round_unspecified",
        used_by_plan: list[str] | None = None,
    ) -> dict[str, Any]:
        key = cache_key(
            provider=self.provider.name,
            method="search",
            query=query,
            year_from=year_from,
            limit=limit,
        )
        cached = self._load_cached(key, query)
        papers = cached if cached is not None else self.provider.search(
            query, year_from=year_from, limit=limit
        )
        if cached is None:
            self._store(key, papers)
        return self._packet(
            papers,
            query=query,
            round_id=round_id,
            used_by_plan=used_by_plan,
            year_from=year_from,
            limit=limit,
            method="search",
            cached=cached is not None,
        )

    def get_paper(
        self,
        paper_id: str,
        *,
        retrieval_query: str = "",
        round_id: str = "round_unspecified",
        used_by_plan: list[str] | None = None,
    ) -> dict[str, Any]:
        query = retrieval_query or paper_id
        key = cache_key(
            provider=self.provider.name,
            method="get_paper",
            query=query,
            year_from=None,
            limit=1,
            paper_id=paper_id,
        )
        cached = self._load_cached(key, query)
        if cached:
            papers = cached[:1]
        else:
            papers = [self.provider.get_paper(paper_id, retrieval_query=query)]
            self._store(key, papers)
        return self._packet(
            papers,
            query=query,
            round_id=round_id,
            used_by_plan=used_by_plan,
            year_from=None,
            limit=1,
            method="get_paper",
            cached=cached is not None,
        )

    def related(
        self,
        paper_id: str,
        *,
        kind: str = "references",
        limit: int = 20,
        round_id: str = "round_unspecified",
        used_by_plan: list[str] | None = None,
    ) -> dict[str, Any]:
        method = "citations" if str(kind).strip().lower() == "citations" else "references"
        query = f"{method} of {paper_id}"
        key = cache_key(
            provider=self.provider.name,
            method=method,
            query=query,
            year_from=None,
            limit=limit,
            paper_id=paper_id,
        )
        cached = self._load_cached(key, query)
        if cached is not None:
            papers = cached
        elif method == "citations":
            papers = self.provider.citations(paper_id, limit=limit, retrieval_query=query)
            self._store(key, papers)
        else:
            papers = self.provider.references(paper_id, limit=limit, retrieval_query=query)
            self._store(key, papers)
        return self._packet(
            papers,
            query=query,
            round_id=round_id,
            used_by_plan=used_by_plan,
            year_from=None,
            limit=limit,
            method=method,
            cached=cached is not None,
        )
