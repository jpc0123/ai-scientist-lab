"""Structured PaperRecord. Metadata/abstract only — never full text."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def paper_public_url(
    *,
    url: str | None = None,
    doi: str | None = None,
    arxiv_id: str | None = None,
    paper_id: str | None = None,
    source: str | None = None,
) -> str | None:
    """Stable public link for audit. Fake corpus keeps its own url; do not mint S2 URLs for it."""
    href = str(url or "").strip()
    if href:
        return href
    doi_v = str(doi or "").strip()
    if doi_v:
        return f"https://doi.org/{doi_v}"
    arxiv = str(arxiv_id or "").strip()
    if arxiv:
        return f"https://arxiv.org/abs/{arxiv}"
    pid = str(paper_id or "").strip()
    if str(source or "").strip() == "semantic_scholar" and pid.lower().startswith("s2:"):
        bare = pid.split(":", 1)[1].strip()
        if bare:
            return f"https://www.semanticscholar.org/paper/{bare}"
    return None


def s2_paper_id(raw: str) -> str:
    value = str(raw or "").strip()
    if not value:
        return ""
    if value.lower().startswith("s2:"):
        return "S2:" + value.split(":", 1)[1]
    return f"S2:{value}"


@dataclass
class PaperRecord:
    paper_id: str
    title: str
    source: str
    retrieval_query: str
    retrieved_at: str
    year: int | None = None
    authors: list[str] = field(default_factory=list)
    abstract: str | None = None
    doi: str | None = None
    venue: str | None = None
    citation_count: int | None = None
    url: str | None = None
    arxiv_id: str | None = None

    def identifier(self) -> str | None:
        return paper_public_url(
            url=self.url,
            doi=self.doi,
            arxiv_id=self.arxiv_id,
            paper_id=self.paper_id,
            source=self.source,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        retrieval_query: str,
        source: str,
        retrieved_at: str | None = None,
    ) -> "PaperRecord":
        authors_raw = raw.get("authors") or []
        authors: list[str] = []
        for item in authors_raw:
            if isinstance(item, str) and item.strip():
                authors.append(item.strip())
            elif isinstance(item, Mapping):
                name = str(item.get("name") or "").strip()
                if name:
                    authors.append(name)
        year = raw.get("year")
        try:
            year_i = int(year) if year is not None and str(year).strip() else None
        except (TypeError, ValueError):
            year_i = None
        cites = raw.get("citation_count", raw.get("citationCount"))
        try:
            cites_i = int(cites) if cites is not None and str(cites).strip() != "" else None
        except (TypeError, ValueError):
            cites_i = None
        external = dict(raw.get("externalIds") or raw.get("external_ids") or {})
        doi = raw.get("doi") or external.get("DOI")
        arxiv = raw.get("arxiv_id") or external.get("ArXiv")
        paper_id = str(raw.get("paper_id") or raw.get("paperId") or "").strip()
        if source == "semantic_scholar" or paper_id.startswith("S2:") or raw.get("paperId"):
            paper_id = s2_paper_id(paper_id)
        return cls(
            paper_id=paper_id,
            title=str(raw.get("title") or "").strip(),
            source=source,
            retrieval_query=retrieval_query,
            retrieved_at=retrieved_at or utc_now(),
            year=year_i,
            authors=authors,
            abstract=(str(raw["abstract"]).strip() if raw.get("abstract") else None),
            doi=(str(doi).strip() if doi else None),
            venue=(str(raw.get("venue") or "").strip() or None),
            citation_count=cites_i,
            url=(str(raw.get("url") or "").strip() or None),
            arxiv_id=(str(arxiv).strip() if arxiv else None),
        )


@dataclass
class LiteratureQueryRecord:
    literature_query_id: str
    round_id: str
    queries: list[str]
    paper_refs: list[str]
    used_by_plan: list[str] = field(default_factory=list)
    provider: str = "fake"
    year_from: int | None = None
    limit: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
