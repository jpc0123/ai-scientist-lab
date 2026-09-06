"""Rerank scout hits against the current research question.

Heuristic is the testable default. Optional live LLM rerank is best-effort.
Never fetches full text. Never promotes literature into ClaimGate.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from scientist_lab.literature.models import PaperRecord
from scientist_lab.literature.query_synth import (
    catalog_query_terms,
    key_terms,
    protocol_fingerprint,
    protocol_research_question,
)

_GENERIC_SURVEY = re.compile(
    r"\b((a |an )?(comprehensive |critical )?(survey|review) of "
    r"(deep learning|machine learning|neural networks?|cnns?|artificial intelligence)\b"
    r"|deep learning (survey|review)"
    r"|survey of deep learning)",
    re.IGNORECASE,
)
_GENERIC_TITLE = re.compile(
    r"^(a |an )?(comprehensive )?(survey|review) of deep learning\b",
    re.IGNORECASE,
)


def _norm(text: str) -> str:
    return re.sub(r"[-_]+", " ", str(text or "").lower())


def _blob(paper: PaperRecord | Mapping[str, Any]) -> str:
    if isinstance(paper, PaperRecord):
        return f"{paper.title} {paper.abstract or ''}"
    return f"{paper.get('title') or ''} {paper.get('abstract') or ''}"


def _title(paper: PaperRecord | Mapping[str, Any]) -> str:
    if isinstance(paper, PaperRecord):
        return str(paper.title or "")
    return str(paper.get("title") or "")


def _as_record(paper: PaperRecord | Mapping[str, Any]) -> PaperRecord:
    if isinstance(paper, PaperRecord):
        return paper
    return PaperRecord.from_mapping(
        paper,
        retrieval_query=str(paper.get("retrieval_query") or "unknown"),
        source=str(paper.get("source") or "fake"),
        retrieved_at=paper.get("retrieved_at"),
    )


def is_generic_survey(paper: PaperRecord | Mapping[str, Any]) -> bool:
    title = _title(paper)
    blob = _blob(paper)
    if _GENERIC_TITLE.search(title.strip()):
        return True
    head = blob[:280]
    return bool(_GENERIC_SURVEY.search(title) or _GENERIC_SURVEY.search(head))


def _aliases(token: str) -> list[str]:
    raw = str(token or "").strip()
    if not raw:
        return []
    cleaned = raw.replace("dataset:", "").strip()
    cleaned = re.sub(r"_v\d+$", "", cleaned, flags=re.IGNORECASE)
    variants = {
        cleaned,
        cleaned.replace("_", "-"),
        cleaned.replace("-", " "),
        cleaned.replace("_", " "),
        cleaned.replace("-", ""),
        cleaned.replace("_", ""),
    }
    return [item for item in variants if len(item) >= 3]


def protocol_clues(
    paper: PaperRecord | Mapping[str, Any],
    *,
    protocol: Mapping[str, Any] | None = None,
    research_question: str = "",
) -> dict[str, Any]:
    """Whether title/abstract mention the current experiment's dataset/metric/task.

    A miss is a clue, not a Claim. Catalog HOW terms are labels, not new operators.
    """
    record = _as_record(paper)
    blob = _norm(_blob(record))
    fp = protocol_fingerprint(protocol)
    dataset = str(fp.get("dataset") or "")
    metric = str(fp.get("metric") or "")
    rq = str(research_question or "").strip() or protocol_research_question(protocol)
    dataset_hit = any(_norm(alias) and _norm(alias) in blob for alias in _aliases(dataset))
    metric_l = metric.lower().replace("_", " ")
    metric_hit = False
    if metric:
        metric_hit = _norm(metric) in blob or metric_l in blob
        head = metric.split("_", 1)[0]
        if len(head) >= 3 and _norm(head) in blob:
            metric_hit = True
    task_hits = [term for term in key_terms(rq, limit=12) if _norm(term) and _norm(term) in blob]
    how_hits: list[str] = []
    seen: set[str] = set()
    for hid in ("F0", "F1", "F3", "N0", "N1", "A4"):
        for term in catalog_query_terms(hid):
            token = _norm(term)
            if token and token in blob and token not in seen:
                seen.add(token)
                how_hits.append(term)
    bits: list[str] = []
    if dataset_hit:
        bits.append(f"dataset~{dataset}")
    else:
        bits.append("dataset not in abstract")
    if metric_hit:
        bits.append(f"metric~{metric}")
    else:
        bits.append("metric not in abstract")
    if how_hits:
        bits.append("catalog terms: " + ", ".join(how_hits[:4]))
    aligned = bool(task_hits) and (dataset_hit or metric_hit)
    return {
        "dataset_hit": dataset_hit,
        "metric_hit": metric_hit,
        "task_hits": task_hits,
        "how_terms": how_hits[:4],
        "protocol_aligned": aligned,
        "clue": "protocol: " + ", ".join(bits),
        "can_enter_claim_gate": False,
    }


def relevance_score(
    paper: PaperRecord | Mapping[str, Any],
    *,
    research_question: str,
    query: str = "",
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score title+abstract against the research question. Not Claim evidence."""
    record = _as_record(paper)
    blob = _norm(_blob(record))
    terms = key_terms(research_question, limit=12)
    query_terms = key_terms(query, limit=8)
    hits = [term for term in terms if _norm(term) and _norm(term) in blob]
    q_hits = [term for term in query_terms if _norm(term) and _norm(term) in blob]
    generic = is_generic_survey(record)
    clues = protocol_clues(record, protocol=protocol, research_question=research_question)
    score = 0.0
    if terms:
        score += 0.7 * (len(hits) / max(len(terms), 1))
    if query_terms:
        score += 0.25 * (len(q_hits) / max(len(query_terms), 1))
    if record.year:
        score += 0.02
    if record.citation_count:
        score += min(0.06, 0.01 * (float(record.citation_count) ** 0.5))
    if clues["protocol_aligned"]:
        score += 0.12
    elif clues["dataset_hit"] or clues["metric_hit"]:
        score += 0.04
    if generic:
        score *= 0.15
    keep = True
    why = ""
    if generic and not hits:
        keep = False
        why = "generic deep-learning survey; no research-question entities"
    elif terms and not hits:
        keep = False
        why = "title/abstract miss research-question key entities"
        score = min(score, 0.08)
    elif hits:
        why = "hits: " + ", ".join(hits[:6])
        if generic:
            why += "; generic-survey title down-ranked"
        why += " · " + str(clues["clue"])
    else:
        why = "weak lexical overlap with query"
    disclaimer = " · not Claim evidence"
    if "not Claim evidence" not in why:
        why = why[: max(0, 240 - len(disclaimer))] + disclaimer
    return {
        "score": round(float(score), 4),
        "why_relevant": why[:240],
        "keep": keep,
        "hits": hits,
        "generic_survey": generic,
        "dataset_hit": clues["dataset_hit"],
        "metric_hit": clues["metric_hit"],
        "protocol_aligned": clues["protocol_aligned"],
        "can_enter_claim_gate": False,
    }


def rerank_papers(
    papers: Sequence[PaperRecord | Mapping[str, Any]],
    *,
    research_question: str,
    query: str = "",
    protocol: Mapping[str, Any] | None = None,
    limit: int = 8,
) -> dict[str, Any]:
    """Filter obvious misses, then sort by score. Fail-soft if everything would drop."""
    scored: list[dict[str, Any]] = []
    for paper in papers:
        record = _as_record(paper)
        meta = relevance_score(
            record, research_question=research_question, query=query, protocol=protocol
        )
        scored.append({"paper": record, **meta})
    scored.sort(key=lambda row: float(row["score"]), reverse=True)
    kept = [row for row in scored if row["keep"]]
    dropped = [row for row in scored if not row["keep"]]
    chosen = kept if kept else scored
    chosen = chosen[: max(1, min(int(limit), 20))] if chosen else []
    return {
        "kept": chosen,
        "dropped": dropped,
        "research_question": research_question,
        "can_enter_claim_gate": False,
    }


def research_question_for_rerank(
    protocol: Mapping[str, Any] | None,
    query: str = "",
) -> str:
    return protocol_research_question(protocol) or str(query or "").strip()
