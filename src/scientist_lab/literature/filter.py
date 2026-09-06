"""Metadata filter on fields Semantic Scholar already returns.

Year, citation_count, venue only. Do not invent modality/task fields S2 does not give.
Fail-soft: if every paper would drop, keep the original list.
Never promotes literature into ClaimGate.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from scientist_lab.literature.models import PaperRecord

DEFAULT_VENUE_BLOCK = ("survey journal",)


def _as_record(paper: PaperRecord | Mapping[str, Any]) -> PaperRecord:
    if isinstance(paper, PaperRecord):
        return paper
    return PaperRecord.from_mapping(
        paper,
        retrieval_query=str(paper.get("retrieval_query") or "unknown"),
        source=str(paper.get("source") or "fake"),
        retrieved_at=paper.get("retrieved_at"),
    )


def metadata_reasons(
    paper: PaperRecord | Mapping[str, Any],
    *,
    year_from: int | None = 2022,
    min_citations: int = 0,
    venue_block: Sequence[str] | None = None,
) -> list[str]:
    """Why this row fails metadata gates. Empty list = keep."""
    record = _as_record(paper)
    blocked = [str(x).strip().lower() for x in (venue_block if venue_block is not None else DEFAULT_VENUE_BLOCK) if str(x).strip()]
    reasons: list[str] = []
    if year_from is not None and record.year is not None and int(record.year) < int(year_from):
        reasons.append(f"year {record.year} < {year_from}")
    if min_citations > 0 and record.citation_count is not None and int(record.citation_count) < int(min_citations):
        reasons.append(f"citation_count {record.citation_count} < {min_citations}")
    venue = str(record.venue or "").lower()
    for token in blocked:
        if token and token in venue:
            reasons.append(f"venue blocked ({record.venue})")
            break
    return reasons


def filter_papers(
    papers: Sequence[PaperRecord | Mapping[str, Any]],
    *,
    year_from: int | None = 2022,
    min_citations: int = 0,
    venue_block: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Drop old / survey-venue / under-cited rows. Fail-soft if nothing remains."""
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for paper in papers:
        record = _as_record(paper)
        reasons = metadata_reasons(
            record,
            year_from=year_from,
            min_citations=min_citations,
            venue_block=venue_block,
        )
        row = {"paper": record, "reasons": reasons, "can_enter_claim_gate": False}
        if reasons:
            dropped.append(row)
        else:
            kept.append(row)
    used_fallback = False
    if not kept and list(papers):
        used_fallback = True
        kept = [
            {"paper": _as_record(paper), "reasons": [], "can_enter_claim_gate": False}
            for paper in papers
        ]
        dropped = []
    return {
        "kept": kept,
        "dropped": dropped,
        "used_fallback": used_fallback,
        "year_from": year_from,
        "min_citations": int(min_citations),
        "can_enter_claim_gate": False,
    }


def filter_gated_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    year_from: int | None = 2022,
    min_citations: int = 0,
    venue_block: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Same gates on LiteratureRetriever gated packets. Does not mutate schema fields."""
    papers: list[Any] = []
    by_id: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        paper = row.get("paper") if isinstance(row.get("paper"), Mapping) else row
        record = _as_record(paper if isinstance(paper, Mapping) else {})
        pid = str(record.paper_id or "") or f"anon-{len(order)}"
        papers.append(record)
        by_id[pid] = dict(row)
        order.append(pid)
    ranked = filter_papers(
        papers, year_from=year_from, min_citations=min_citations, venue_block=venue_block
    )
    kept_ids = {str(item["paper"].paper_id) for item in ranked["kept"]}
    kept_rows = [by_id[pid] for pid in order if pid in kept_ids and pid in by_id]
    dropped_rows = [
        {
            "paper_id": pid,
            "reasons": metadata_reasons(
                by_id[pid].get("paper") if isinstance(by_id[pid].get("paper"), Mapping) else by_id[pid],
                year_from=year_from,
                min_citations=min_citations,
                venue_block=venue_block,
            ),
            "can_enter_claim_gate": False,
        }
        for pid in order
        if pid not in kept_ids
    ]
    return {
        "kept": kept_rows,
        "dropped": dropped_rows,
        "used_fallback": bool(ranked["used_fallback"]),
        "can_enter_claim_gate": False,
    }
