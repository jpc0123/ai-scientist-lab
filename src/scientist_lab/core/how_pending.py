"""Pending HOW store. LLM drafts; humans freeze-register. Not the executable catalog."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from scientist_lab.adapters.dfine.how_catalog import (
    ALLOWED_HOW,
    CATALOG_ID,
    NOT_REGISTERED,
    adapter_can_map_candidate,
    candidate_plugin_kind,
    family_needs_adapter_work,
    overlay_is_plugin,
    plugin_overlay_spec,
    resolve_how_id,
)
from scientist_lab.core.schema_registry import validate_named
from scientist_lab.llm.errors import MissingAPIKeyError, RealProviderNotEnabledError
from scientist_lab.llm.planner_contract import PlannerContractError
from scientist_lab.literature.models import paper_public_url

STORE_NAME = "how_pending.json"
_ALLOWED_FAMILIES = frozenset({"fusion", "neck", "training", "backbone_wrap"})
_CODE_HINT = re.compile(
    r"(```|write python|\bdef\s+\w|\bclass\s+\w+\s*[:\(]|\bimport\s+\w|"
    r"\bfrom\s+\w+\s+import\b|new network|\bfdpn\b)",
    re.IGNORECASE,
)

STATUS_PROPOSED = "proposed"
STATUS_REJECTED = "rejected"
STATUS_PENDING_ADAPTER = "approved_pending_adapter"
STATUS_REGISTERED = "registered"

INTENT_SOURCES = frozenset({"human", "llm", "fallback"})
INTENT_ACTIVE = "active"
INTENT_PROPOSED = "proposed"
INTENT_CLEARED = "cleared"
MAX_SCOUT_DIALOGUE = 40
MAX_SCOUT_QUERY = 400
LEDGER_MAX = 40
_INTENT_CANNOT = ("register_how", "start_gpu", "write_claim")


class HowPendingError(ValueError):
    """Human freeze or ingest refused a HOW candidate."""


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def pending_path(campaign_dir: Path | str) -> Path:
    return Path(campaign_dir) / STORE_NAME


def empty_store() -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "catalog_id": CATALOG_ID,
        "llm_may_invent_how": False,
        "can_enter_claim_gate": False,
        "scout": None,
        "scout_intent": None,
        "scout_dialogue": [],
        "literature_ledger": [],
        "candidates": [],
        "registered_overlay": {},
        "updated_at": _now(),
    }


def load_store(path: Path | str) -> dict[str, Any]:
    dest = Path(path)
    if not dest.is_file():
        return empty_store()
    raw = json.loads(dest.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise HowPendingError("how_pending.json must be an object")
    store = empty_store()
    store.update(raw)
    store["candidates"] = [
        dict(row) for row in (store.get("candidates") or []) if isinstance(row, Mapping)
    ]
    store["registered_overlay"] = dict(store.get("registered_overlay") or {})
    store["scout_dialogue"] = [
        dict(row) for row in (store.get("scout_dialogue") or []) if isinstance(row, Mapping)
    ]
    store["literature_ledger"] = [
        dict(row) for row in (store.get("literature_ledger") or []) if isinstance(row, Mapping)
    ]
    intent = store.get("scout_intent")
    store["scout_intent"] = dict(intent) if isinstance(intent, Mapping) else None
    scout = store.get("scout")
    if isinstance(scout, Mapping):
        store["scout"] = summarize_scout(scout)
    else:
        store["scout"] = None
    # Campaign Human Gate may flip invent for Stage B; do not force-false on load.
    store["llm_may_invent_how"] = bool(store.get("llm_may_invent_how"))
    store["can_enter_claim_gate"] = False
    return store


def save_store(path: Path | str, store: Mapping[str, Any]) -> dict[str, Any]:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(store)
    payload["llm_may_invent_how"] = bool(payload.get("llm_may_invent_how"))
    payload["can_enter_claim_gate"] = False
    payload["updated_at"] = _now()
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def set_llm_may_invent_how(path: Path | str, enabled: bool) -> dict[str, Any]:
    """Persist Stage B invent gate on how_pending. ClaimGate stays closed."""
    store = load_store(path)
    store["llm_may_invent_how"] = bool(enabled)
    store["can_enter_claim_gate"] = False
    return save_store(path, store)


def literature_paper_ids(packet: Mapping[str, Any] | None) -> set[str]:
    ids: set[str] = set()
    blob = dict(packet or {})
    for row in blob.get("planner_admissible") or []:
        if not isinstance(row, Mapping):
            continue
        paper = row.get("paper") if isinstance(row.get("paper"), Mapping) else row
        token = str((paper or {}).get("paper_id") or "").strip()
        if token:
            ids.add(token)
    provenance = blob.get("provenance") if isinstance(blob.get("provenance"), Mapping) else {}
    for token in list(blob.get("paper_refs") or []) + list((provenance or {}).get("paper_refs") or []):
        if str(token).strip():
            ids.add(str(token).strip())
    return ids


def literature_query_id(packet: Mapping[str, Any] | None) -> str:
    blob = dict(packet or {})
    provenance = blob.get("provenance") if isinstance(blob.get("provenance"), Mapping) else {}
    return str(
        blob.get("literature_query_id")
        or (provenance or {}).get("literature_query_id")
        or ""
    ).strip()


def _paper_row(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
    blob = dict(row or {})
    paper = blob.get("paper") if isinstance(blob.get("paper"), Mapping) else blob
    if not isinstance(paper, Mapping):
        return None
    pid = str(paper.get("paper_id") or "").strip()
    title = str(paper.get("title") or "").strip()
    if not pid and not title:
        return None
    url = str(paper.get("url") or "").strip() or None
    doi = str(paper.get("doi") or "").strip() or None
    arxiv = str(paper.get("arxiv_id") or "").strip() or None
    source = str(paper.get("source") or blob.get("source") or "").strip() or None
    url = paper_public_url(
        url=url,
        doi=doi,
        arxiv_id=arxiv,
        paper_id=pid,
        source=source,
    )
    why = str(blob.get("why_relevant") or paper.get("why_relevant") or "").strip() or None
    score = blob.get("score", paper.get("score"))
    try:
        score_f = float(score) if score is not None and str(score).strip() != "" else None
    except (TypeError, ValueError):
        score_f = None
    abstract = str(paper.get("abstract") or "").strip()
    snippet = abstract[:240] if abstract else None
    hop_raw = blob.get("hop", paper.get("hop"))
    try:
        hop_i = int(hop_raw) if hop_raw is not None and str(hop_raw).strip() != "" else 0
    except (TypeError, ValueError):
        hop_i = 0
    aligned = blob.get("protocol_aligned", paper.get("protocol_aligned"))
    retrieval = str(
        blob.get("retrieval_source") or paper.get("retrieval_source") or ""
    ).strip().lower()
    if retrieval not in {"retriever", "library", "hop"}:
        retrieval = "hop" if hop_i else "retriever"
    rank_raw = blob.get("rank", paper.get("rank"))
    try:
        rank_i = int(rank_raw) if rank_raw is not None and str(rank_raw).strip() != "" else None
    except (TypeError, ValueError):
        rank_i = None
    return {
        "paper_id": pid,
        "title": title or pid,
        "url": url,
        "link_ok": bool(url),
        "year": paper.get("year"),
        "doi": doi,
        "score": score_f,
        "why_relevant": why,
        "abstract": snippet,
        "hop": hop_i,
        "rank": rank_i,
        "retrieval_source": retrieval,
        "source": source,
        "protocol_aligned": bool(aligned) if aligned is not None else None,
        "title_zh": str(blob.get("title_zh") or paper.get("title_zh") or "").strip() or None,
        "title_en": str(blob.get("title_en") or paper.get("title_en") or "").strip() or None,
        "why_relevant_zh": str(blob.get("why_relevant_zh") or paper.get("why_relevant_zh") or "").strip()
        or None,
        "why_relevant_en": str(
            blob.get("why_relevant_en") or paper.get("why_relevant_en") or ""
        ).strip()
        or None,
        "display_translated": bool(blob.get("display_translated") or paper.get("display_translated")),
        "can_enter_claim_gate": False,
    }


def scout_paper_summaries(packet: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    blob = dict(packet or {})
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    rows: list[Any] = []
    for key in ("papers", "planner_admissible"):
        rows.extend(list(blob.get(key) or []))
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        item = _paper_row(row)
        if item is None:
            continue
        token = str(item.get("paper_id") or item.get("title") or "")
        if token in seen:
            continue
        seen.add(token)
        out.append(item)
    for index, item in enumerate(out, start=1):
        item["rank"] = index
    return out


def merge_literature_ledger(
    store: Mapping[str, Any] | None,
    papers: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Accumulate verifiable papers. Ranked by best score. Not Claim evidence."""
    existing: dict[str, dict[str, Any]] = {}
    for row in list((store or {}).get("literature_ledger") or []):
        if not isinstance(row, Mapping):
            continue
        pid = str(row.get("paper_id") or "").strip()
        if pid:
            existing[pid] = dict(row)
    for row in papers or []:
        if not isinstance(row, Mapping):
            continue
        item = _paper_row(row)
        if item is None or not item.get("paper_id") or not item.get("url"):
            continue
        pid = str(item["paper_id"])
        prev = dict(existing.get(pid) or {})
        prev_score = prev.get("best_score")
        score = item.get("score")
        try:
            score_f = float(score) if score is not None else None
        except (TypeError, ValueError):
            score_f = None
        try:
            prev_f = float(prev_score) if prev_score is not None else None
        except (TypeError, ValueError):
            prev_f = None
        better = score_f is not None and (prev_f is None or score_f >= prev_f)
        merged = {
            **prev,
            "paper_id": pid,
            "title": item.get("title") or prev.get("title"),
            "url": item.get("url") or prev.get("url"),
            "doi": item.get("doi") or prev.get("doi"),
            "year": item.get("year") if item.get("year") is not None else prev.get("year"),
            "abstract": item.get("abstract") or prev.get("abstract"),
            "source": item.get("source") or prev.get("source"),
            "link_ok": True,
            "can_enter_claim_gate": False,
        }
        if better:
            merged["best_score"] = score_f
            merged["why_relevant"] = item.get("why_relevant") or prev.get("why_relevant")
        elif "best_score" not in merged:
            merged["best_score"] = score_f
        existing[pid] = merged
    ranked = sorted(
        existing.values(),
        key=lambda row: float(row.get("best_score") or 0.0),
        reverse=True,
    )
    return ranked[:LEDGER_MAX]


def rank_library_ledger(
    ledger: Sequence[Mapping[str, Any]] | None,
    *,
    research_question: str,
    query: str,
    protocol: Mapping[str, Any] | None = None,
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Re-rank the campaign paper library. Does not invent papers."""
    rows = [dict(row) for row in (ledger or []) if isinstance(row, Mapping) and row.get("paper_id")]
    if not rows:
        return []
    from scientist_lab.literature.rerank import rerank_papers

    ranked = rerank_papers(
        rows,
        research_question=research_question,
        query=query,
        protocol=protocol,
        limit=limit,
    )
    out: list[dict[str, Any]] = []
    for item in ranked["kept"]:
        record = item["paper"]
        summary = dict(record.to_dict() if hasattr(record, "to_dict") else {})
        summary["score"] = item["score"]
        summary["why_relevant"] = item["why_relevant"]
        summary["protocol_aligned"] = item.get("protocol_aligned")
        summary["retrieval_source"] = "library"
        summary["hop"] = 0
        summary["can_enter_claim_gate"] = False
        out.append(summary)
    return scout_paper_summaries({"papers": out})


def _scout_closed_packet(
    *,
    query: str,
    live: bool,
    error: str,
    round_id: str | None = None,
    queries: Sequence[str] | None = None,
) -> dict[str, Any]:
    reason = str(error or "").strip() or "missing literature key or live search not enabled"
    q_list = [str(x).strip() for x in (queries or [query]) if str(x).strip()] or [query]
    return {
        "ok": False,
        "fail_closed": True,
        "live": bool(live),
        "actual_search": False,
        "provider": None,
        "planner_admissible": [],
        "papers": [],
        "paper_refs": [],
        "query": query,
        "queries": q_list,
        "literature_query_id": "",
        "error": f"没有实际检索：{reason}",
        "round_id": round_id,
        "can_enter_claim_gate": False,
        "claim_gate_note": (
            "LiteratureEvidence cannot enter ClaimGate; scout fail-closed. "
            "No live literature search happened."
        ),
    }


def scout_literature_for_how(
    *,
    protocol: Mapping[str, Any] | None = None,
    live: bool,
    round_id: str,
    provenance_dir: Path | str | None,
    query: str | None = None,
    queries: Sequence[str] | None = None,
    environ: Mapping[str, str] | None = None,
    year_from: int | None = 2022,
    research_question: str | None = None,
    min_citations: int = 0,
    hop: bool = False,
    ledger: Sequence[Mapping[str, Any]] | None = None,
    include_library: bool = True,
) -> dict[str, Any]:
    """Search papers for HOW drafts. Live without a key fail-closes this scout only.

    Live path uses a single primary query (no multi-query burst) to reduce S2 429s.
    Optional 1-hop references is off by default; pass hop=True to enable.
    """
    from scientist_lab.literature.filter import filter_gated_rows
    from scientist_lab.literature.query_synth import (
        last_ditch_fallback_query,
        strip_internal_search_tokens,
    )
    from scientist_lab.literature.rerank import rerank_papers, research_question_for_rerank

    q_list = [str(x).strip() for x in (queries or []) if str(x).strip()]
    primary = str(query or "").strip()
    if primary and primary not in q_list:
        q_list = [primary, *q_list]
    if not q_list:
        q_list = [last_ditch_fallback_query(protocol)]
    authored_queries = list(q_list)
    # Fire academic strings only — lab ids (rgbt_tiny_v1 / APS_lowlight) → 0 S2 hits.
    cleaned: list[str] = []
    for item in q_list:
        text = strip_internal_search_tokens(item) or last_ditch_fallback_query(protocol)
        if text and text.lower() not in {c.lower() for c in cleaned}:
            cleaned.append(text)
    q_list = cleaned or [last_ditch_fallback_query(protocol)]
    q = q_list[0]
    rq = str(research_question or "").strip() or research_question_for_rerank(protocol, q)
    library_rows = (
        [dict(row) for row in (ledger or []) if isinstance(row, Mapping)]
        if include_library
        else []
    )

    def _closed(error: str) -> dict[str, Any]:
        packet = _scout_closed_packet(
            query=q, live=bool(live), error=error, round_id=round_id, queries=q_list
        )
        lib = rank_library_ledger(
            library_rows, research_question=rq, query=q, protocol=protocol, limit=8
        )
        if lib:
            packet["papers"] = lib
            packet["ranked_table"] = lib
            packet["library_only"] = True
            packet["research_question"] = rq
            packet["fail_closed"] = False
            packet["ok"] = True
            packet["literature_query_id"] = f"lq_library_{round_id or 'round'}"
            packet["paper_refs"] = [
                str(row.get("paper_id") or "").strip()
                for row in lib
                if str(row.get("paper_id") or "").strip()
            ]
            packet["note"] = (
                "联网检索失败，降级为战役文献库。"
                " 仍可规划目录 HOW；不会凭记忆发明论文。"
            )
            err_lower = str(error or "").lower()
            if "429" in str(error) or "rate limit" in err_lower:
                packet["degraded_reason"] = "rate_limited"
        return packet
    try:
        from scientist_lab.literature.runtime_secrets import apply_runtime_literature_env
        from scientist_lab.settings import get_settings

        if live:
            apply_runtime_literature_env(get_settings().runtime_dir)
    except OSError:
        pass
    try:
        from scientist_lab.literature.retriever import LiteratureRetriever

        retriever = LiteratureRetriever(
            live=bool(live),
            provenance_dir=provenance_dir,
            cache_dir=(Path(provenance_dir) / "cache") if provenance_dir else None,
            environ=environ,
        )
        merged_gated: list[dict[str, Any]] = []
        seen: set[str] = set()
        hop_ids: set[str] = set()
        library_ids: set[str] = set()
        packet: dict[str, Any] = {}

        def _absorb(part: Mapping[str, Any], *, hop_i: int) -> None:
            nonlocal packet
            if not packet:
                packet = dict(part)
            for row in list(part.get("papers") or []):
                if not isinstance(row, Mapping):
                    continue
                paper = row.get("paper") if isinstance(row.get("paper"), Mapping) else row
                token = str((paper or {}).get("paper_id") or (paper or {}).get("doi") or "").strip()
                if not token or token in seen:
                    continue
                seen.add(token)
                merged_gated.append(dict(row))
                if hop_i:
                    hop_ids.add(token)

        # Live: one query only (composed extras kept for provenance, not fired).
        # Offline fake: still allow up to 3 so unit fixtures stay expressive.
        search_queries = q_list[:1] if live else q_list[:3]
        empty_retries: list[str] = []
        for item in search_queries:
            _absorb(
                retriever.search(item, limit=8, round_id=round_id, year_from=year_from),
                hop_i=0,
            )
        # Live empty miss: try a short S2-safe fallback (no D-FINE / RGBT-Tiny).
        if live and not merged_gated:
            from scientist_lab.literature.query_synth import _SAFE_S2_FALLBACK

            for fallback_q in (
                last_ditch_fallback_query(protocol),
                _SAFE_S2_FALLBACK,
            ):
                if not fallback_q:
                    continue
                if fallback_q.lower() in {x.lower() for x in search_queries}:
                    continue
                if fallback_q.lower() in {x.lower() for x in empty_retries}:
                    continue
                empty_retries.append(fallback_q)
                _absorb(
                    retriever.search(
                        fallback_q, limit=8, round_id=round_id, year_from=year_from
                    ),
                    hop_i=0,
                )
                search_queries = list(search_queries) + [fallback_q]
                if merged_gated:
                    break
        filtered = filter_gated_rows(
            merged_gated, year_from=year_from, min_citations=int(min_citations)
        )
        working = list(filtered["kept"])
        ranked = rerank_papers(
            [dict(row.get("paper") or row) for row in working],
            research_question=rq,
            query=q,
            protocol=protocol,
            limit=8,
        )
        seed_id = ""
        if hop and ranked.get("kept"):
            seed_id = str(getattr(ranked["kept"][0]["paper"], "paper_id", "") or "")
        hop_error = None
        if hop and seed_id:
            try:
                _absorb(
                    retriever.related(
                        seed_id, kind="references", limit=5, round_id=round_id
                    ),
                    hop_i=1,
                )
                filtered = filter_gated_rows(
                    merged_gated, year_from=year_from, min_citations=int(min_citations)
                )
                working = list(filtered["kept"])
                ranked = rerank_papers(
                    [dict(row.get("paper") or row) for row in working],
                    research_question=rq,
                    query=q,
                    protocol=protocol,
                    limit=8,
                )
            except Exception as exc:  # noqa: BLE001 — 1-hop is best-effort
                hop_error = str(exc)
        for row in library_rows:
            pid = str(row.get("paper_id") or "").strip()
            if not pid or pid in seen or not row.get("url"):
                continue
            seen.add(pid)
            library_ids.add(pid)
            merged_gated.append({"paper": dict(row), "planner_admissible": False, "can_enter_claim_gate": False})
        if library_ids:
            filtered = filter_gated_rows(
                merged_gated, year_from=year_from, min_citations=int(min_citations)
            )
            working = list(filtered["kept"])
            ranked = rerank_papers(
                [dict(item.get("paper") or item) for item in working],
                research_question=rq,
                query=q,
                protocol=protocol,
                limit=8,
            )
        kept_ids = []
        kept_gated: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        by_id = {}
        for row in working:
            paper = row.get("paper") if isinstance(row.get("paper"), Mapping) else row
            by_id[str((paper or {}).get("paper_id") or "")] = dict(row)
        for item in ranked["kept"]:
            record = item["paper"]
            pid = str(getattr(record, "paper_id", "") or "")
            gated = dict(by_id.get(pid) or {})
            if gated:
                kept_gated.append(gated)
            summary_src = dict(record.to_dict() if hasattr(record, "to_dict") else {})
            summary_src["score"] = item["score"]
            summary_src["why_relevant"] = item["why_relevant"]
            summary_src["protocol_aligned"] = item.get("protocol_aligned")
            summary_src["hop"] = 1 if pid in hop_ids else 0
            if pid in library_ids:
                summary_src["retrieval_source"] = "library"
            elif pid in hop_ids:
                summary_src["retrieval_source"] = "hop"
            else:
                summary_src["retrieval_source"] = "retriever"
            summary_src["can_enter_claim_gate"] = False
            summaries.append(summary_src)
            if pid:
                kept_ids.append(pid)
        packet = dict(packet or {})
        packet["fail_closed"] = False
        packet["live"] = bool(live)
        packet["actual_search"] = bool(live)
        packet["can_enter_claim_gate"] = False
        packet["query"] = q
        packet["queries"] = list(search_queries)
        packet["queries_composed"] = authored_queries[:3]
        packet["queries_authored"] = authored_queries[:3]
        packet["queries_skipped"] = [x for x in q_list[:3] if x not in search_queries]
        packet["empty_retries"] = list(empty_retries)
        packet["round_id"] = round_id
        packet["research_question"] = rq
        packet["year_from"] = year_from
        packet["min_citations"] = int(min_citations)
        packet["papers"] = summaries
        packet["planner_admissible"] = [
            row for row in kept_gated if row.get("planner_admissible")
        ]
        packet["rejected"] = [
            row for row in kept_gated if not row.get("planner_admissible")
        ]
        packet["dropped"] = [
            {
                "paper_id": str(getattr(row["paper"], "paper_id", "") or ""),
                "title": str(getattr(row["paper"], "title", "") or ""),
                "why_relevant": row.get("why_relevant"),
                "score": row.get("score"),
                "can_enter_claim_gate": False,
            }
            for row in ranked.get("dropped") or []
        ]
        packet["dropped_count"] = len(list(ranked.get("dropped") or [])) + len(
            list(filtered.get("dropped") or [])
        )
        packet["filter_dropped"] = len(list(filtered.get("dropped") or []))
        packet["hop_seed"] = seed_id or None
        packet["hop_added"] = len([pid for pid in kept_ids if pid in hop_ids])
        packet["hop_error"] = hop_error
        qid = literature_query_id(packet)
        if qid:
            packet["literature_query_id"] = qid
        packet["paper_refs"] = kept_ids or sorted(literature_paper_ids(packet))
        provenance = dict(packet.get("provenance") or {})
        if provenance:
            provenance["queries"] = list(search_queries)
            provenance["queries_composed"] = authored_queries[:3]
            provenance["paper_refs"] = list(packet["paper_refs"])
            packet["provenance"] = provenance
        papers = scout_paper_summaries(packet)
        packet["papers"] = papers
        packet["ranked_table"] = papers
        packet["library_hits"] = len([row for row in papers if row.get("retrieval_source") == "library"])
        notes: list[str] = [
            f"按研究问题重排后保留 {len(papers)} 篇，最优在前。文献不能进 ClaimGate。"
        ]
        if packet["library_hits"]:
            notes.append(f"其中 {packet['library_hits']} 篇来自本战役文献库。")
        if packet["filter_dropped"]:
            notes.append(f"元数据过滤丢掉 {packet['filter_dropped']} 篇（年/venue/引用）。")
        if packet["dropped_count"] and not packet["filter_dropped"]:
            notes.append(f"丢掉 {packet['dropped_count']} 篇不相关或通用综述。")
        if empty_retries:
            notes.append(
                "首枪空结果后用学术回退 query 再搜了一次"
                f"（{empty_retries[0][:80]}）。"
            )
        if authored_queries and authored_queries[0] != q:
            notes.append("已去掉实验室 dataset/slice/metric id，避免 Semantic Scholar 零命中。")
        if seed_id:
            notes.append(
                f"对 {seed_id} 做了 1 跳 references，最终留下 {packet['hop_added']} 篇。"
            )
        if hop_error:
            notes.append("1 跳引用扩展失败，已忽略。")
        packet["rerank_note"] = " ".join(notes)
        if not papers:
            packet["ok"] = False
            packet["error"] = "检索已执行，但结果为空"
        if not live:
            packet["error"] = packet.get("error")
            packet["note"] = "未 live：使用离线语料，没有实际联网检索"
        return packet
    except (MissingAPIKeyError, RealProviderNotEnabledError) as exc:
        return _closed(str(exc))
    except Exception as exc:  # noqa: BLE001 — scout fail-closed; do not invent papers
        return _closed(str(exc))


def summarize_scout(packet: Mapping[str, Any] | None) -> dict[str, Any]:
    """Human-facing last-scout row. Not ClaimGate evidence."""
    blob = dict(packet or {})
    papers = scout_paper_summaries(blob)
    fail_closed = bool(blob.get("fail_closed"))
    live = bool(blob.get("live"))
    error = str(blob.get("error") or "").strip() or None
    note = str(blob.get("note") or "").strip() or None
    if fail_closed and blob.get("library_only") and papers:
        note = str(blob.get("note") or "").strip() or (
            "本轮没有实际联网检索。下表来自本战役已核验文献库，不是 LLM 凭记忆编造。"
        )
    elif fail_closed:
        if error and not str(error).startswith("没有实际检索"):
            error = f"没有实际检索：{error}"
        error = error or "没有实际检索：缺 Key 或未允许联网"
        note = note or error
    elif not live:
        note = note or "未 live：使用离线语料，没有实际联网检索"
    elif not papers:
        note = note or "检索已执行，但结果为空"
        error = error or note
    source = str(blob.get("intent_source") or blob.get("source") or "").strip()
    fallback = bool(blob.get("intent_fallback") if "intent_fallback" in blob else blob.get("fallback"))
    if source == "fallback":
        fallback = True
    origin = source or ("fallback" if fallback else None)
    queries = [str(x).strip() for x in (blob.get("queries") or []) if str(x).strip()]
    if not queries and blob.get("query"):
        queries = [str(blob.get("query"))]
    rerank_note = str(blob.get("rerank_note") or "").strip() or None
    if rerank_note and note and rerank_note not in str(note):
        note = f"{note} · {rerank_note}"
    elif rerank_note and not note:
        note = rerank_note
    return {
        "ok": bool(blob.get("ok", not fail_closed and bool(papers))),
        "fail_closed": fail_closed,
        "live": live,
        "actual_search": bool(blob.get("actual_search", live and not fail_closed)),
        "provider": blob.get("provider"),
        "literature_query_id": literature_query_id(blob),
        "query": blob.get("query"),
        "queries": queries,
        "research_question": blob.get("research_question"),
        "source": origin,
        "intent_source": origin,
        "intent_why": blob.get("intent_why") or blob.get("why"),
        "fallback": fallback,
        "error": error,
        "note": note,
        "rerank_note": rerank_note,
        "dropped_count": int(blob.get("dropped_count") or 0),
        "filter_dropped": int(blob.get("filter_dropped") or 0),
        "hop_seed": blob.get("hop_seed"),
        "hop_added": int(blob.get("hop_added") or 0),
        "library_only": bool(blob.get("library_only")),
        "library_hits": int(
            blob.get("library_hits")
            or len([row for row in papers if row.get("retrieval_source") == "library"])
        ),
        "papers": papers,
        "ranked_table": papers,
        "paper_refs": [str(x) for x in (blob.get("paper_refs") or []) if str(x).strip()]
        or [str(row["paper_id"]) for row in papers if row.get("paper_id")],
        "how_ingest_error": blob.get("how_ingest_error"),
        "round_id": blob.get("round_id"),
        "year_from": blob.get("year_from"),
        "can_enter_claim_gate": False,
    }


def persist_scout(
    path: Path | str,
    literature: Mapping[str, Any] | None,
    *,
    dialogue: bool = False,
    locale: str | None = None,
    live: bool = False,
    provider: Any | None = None,
) -> dict[str, Any]:
    """Write last scout to how_pending.json immediately. Do not wait for LLM ingest."""
    store = load_store(path)
    summary = summarize_scout(literature)
    from scientist_lab.literature.display import apply_identity_labels, translate_ranked_papers

    papers = [apply_identity_labels(row) for row in list(summary.get("papers") or [])]
    if locale:
        papers = translate_ranked_papers(papers, locale=locale, live=live, provider=provider)
    summary["papers"] = papers
    summary["ranked_table"] = papers
    store["scout"] = summary
    store["literature_ledger"] = merge_literature_ledger(store, papers)
    saved = save_store(path, store)
    if not dialogue:
        return saved
    summary = saved["scout"]
    n_papers = len(list(summary.get("papers") or []))
    source = str(summary.get("source") or summary.get("intent_source") or "fallback")
    query = str(summary.get("query") or "")
    queries = [str(x) for x in (summary.get("queries") or []) if str(x).strip()]
    extra = ""
    if len(queries) > 1:
        extra = f" · queries={queries}"
    why = str((summary.get("papers") or [{}])[0].get("why_relevant") or "") if summary.get("papers") else ""
    if summary.get("fail_closed"):
        text = str(summary.get("error") or "没有实际检索")
    elif not summary.get("live"):
        text = (
            f"已检索（未 live / 离线语料，不是真实联网）。"
            f"query={query} · source={source} · 命中 {n_papers} 篇。"
            f"{extra} literature_query_id={summary.get('literature_query_id') or '—'}"
            f"{' · ' + str(summary.get('rerank_note')) if summary.get('rerank_note') else ''}"
        )
    elif n_papers == 0:
        text = f"检索已执行但结果为空。query={query} · source={source}{extra}"
    else:
        text = (
            f"已检索。query={query} · source={source} · 命中 {n_papers} 篇。"
            f"{extra} literature_query_id={summary.get('literature_query_id') or '—'}"
            f"{' · ' + str(summary.get('rerank_note')) if summary.get('rerank_note') else ''}"
        )
        if why:
            text += f" 首篇为何留下：{why}"
    return append_scout_dialogue(
        path,
        role="assistant",
        text=text,
        intent_action="scout_result",
        refused=bool(summary.get("fail_closed")),
    )


def localize_scout_store(
    path: Path | str,
    *,
    locale: str,
    live: bool = False,
    provider: Any | None = None,
) -> dict[str, Any]:
    """Fill locale labels on the last ranked table. Does not search or invent papers."""
    from scientist_lab.literature.display import apply_identity_labels, translate_ranked_papers

    store = load_store(path)
    scout = dict(store.get("scout") or {})
    papers = [apply_identity_labels(row) for row in list(scout.get("papers") or [])]
    if not papers:
        return store
    papers = translate_ranked_papers(papers, locale=locale, live=live, provider=provider)
    scout["papers"] = papers
    scout["ranked_table"] = papers
    store["scout"] = scout
    return save_store(path, store)


def execute_llm_scout(
    path: Path | str,
    *,
    protocol: Mapping[str, Any] | None,
    evidence: Mapping[str, Any] | None = None,
    live: bool = False,
    provider: Any | None = None,
    locale: str | None = None,
    provenance_dir: Path | str | None = None,
    environ: Mapping[str, str] | None = None,
    round_id: str = "llm_scout",
) -> dict[str, Any]:
    """LLM writes a query and LiteratureRetriever searches now. Not a fifth Agent."""
    dest = Path(path)
    store = load_store(dest)
    human_holds = False
    intent = dict(store.get("scout_intent") or {})
    if (
        str(intent.get("status") or "") == INTENT_ACTIVE
        and str(intent.get("source") or "") == "human"
        and str(intent.get("query") or "").strip()
    ):
        human_holds = True
    from scientist_lab.literature.query_synth import compose_scout_queries

    composed = compose_scout_queries(
        {} if human_holds else store,
        protocol,
        evidence,
        live=live,
        provider=provider,
    )
    if not human_holds:
        set_scout_intent(
            dest,
            source=str(composed.get("source") or "llm"),
            query=str(composed["query"]),
            why=str(composed.get("why") or "LLM scout"),
            status=INTENT_ACTIVE,
            drafted_by="llm",
        )
    packet = scout_literature_for_how(
        protocol=protocol,
        live=bool(live),
        round_id=round_id,
        provenance_dir=provenance_dir or dest.parent / "literature",
        query=str(composed["query"]),
        queries=list(composed.get("queries") or [composed["query"]]),
        environ=environ,
        year_from=int(composed.get("year_from") or 2022),
        research_question=str(composed.get("research_question") or ""),
        ledger=[],
        include_library=False,
    )
    packet["source"] = str(composed.get("source") or "llm")
    packet["intent_source"] = packet["source"]
    packet["intent_why"] = composed.get("why")
    packet["intent_fallback"] = bool(composed.get("fallback"))
    saved = persist_scout(
        dest,
        packet,
        dialogue=False,
        locale=locale or "zh",
        live=live,
        provider=provider,
    )
    n_papers = len(list((saved.get("scout") or {}).get("papers") or []))
    if human_holds:
        note = (
            f"LLM 已即时检索（query={composed['query']}，命中 {n_papers} 篇）。"
            f"人指定的下一枪 query 未改：{intent.get('query')}。"
            "本轮走检索器，不查战役文献库。文献不能进 ClaimGate。"
        )
    else:
        note = (
            f"LLM 已自己检索。query={composed['query']} · source={composed.get('source')} · "
            f"命中 {n_papers} 篇。走 Semantic Scholar / 离线检索器，不查战役文献库。"
            "下一枪也用这条 query。文献不能进 ClaimGate。"
        )
    saved = append_scout_dialogue(
        dest, role="assistant", text=note, intent_action="llm_search"
    )
    return {
        "ok": True,
        "refused": False,
        "action": "llm_search",
        "reply": note,
        "human_holds": human_holds,
        "query": composed["query"],
        "source": composed.get("source"),
        "scout": saved.get("scout"),
        "scout_intent": saved.get("scout_intent"),
        "scout_dialogue": saved.get("scout_dialogue") or [],
        "can_enter_claim_gate": False,
        "cannot": list(_INTENT_CANNOT),
    }


def execute_library_scout(
    path: Path | str,
    *,
    protocol: Mapping[str, Any] | None,
    evidence: Mapping[str, Any] | None = None,
    locale: str | None = None,
    live: bool = False,
    provider: Any | None = None,
    round_id: str = "library_scout",
) -> dict[str, Any]:
    """Rank the campaign paper library only. Does not search the web or invent papers."""
    dest = Path(path)
    store = load_store(dest)
    intent = dict(store.get("scout_intent") or {})
    query = str(intent.get("query") or "").strip()
    from scientist_lab.literature.query_synth import compose_scout_queries, protocol_research_question
    from scientist_lab.literature.rerank import research_question_for_rerank

    if query:
        rq = protocol_research_question(protocol) or research_question_for_rerank(protocol, query)
        origin = "library"
        why = str(intent.get("why") or "campaign literature library")
    else:
        composed = compose_scout_queries(store, protocol, evidence, live=False)
        query = str(composed.get("query") or "")
        rq = str(composed.get("research_question") or "") or research_question_for_rerank(
            protocol, query
        )
        origin = "library"
        why = str(composed.get("why") or "campaign literature library")
    papers = rank_library_ledger(
        list(store.get("literature_ledger") or []),
        research_question=rq,
        query=query,
        protocol=protocol,
        limit=8,
    )
    if papers:
        note = (
            f"已检索本战役文献库。query={query} · 命中 {len(papers)} 篇。"
            "没有联网，不是 LLM 凭记忆编造。文献不能进 ClaimGate。"
        )
        error = None
    else:
        note = "战役文献库为空。没有可核验论文，不会编造，也不会改去搜网。"
        error = note
    packet = {
        "ok": bool(papers),
        "fail_closed": False,
        "live": False,
        "actual_search": False,
        "library_only": True,
        "provider": "library",
        "query": query,
        "queries": [query] if query else [],
        "research_question": rq,
        "source": origin,
        "intent_source": origin,
        "intent_why": why,
        "papers": papers,
        "ranked_table": papers,
        "paper_refs": [str(row.get("paper_id") or "") for row in papers if row.get("paper_id")],
        "note": note,
        "error": error,
        "round_id": round_id,
        "can_enter_claim_gate": False,
    }
    saved = persist_scout(
        dest,
        packet,
        dialogue=False,
        locale=locale or "zh",
        live=live,
        provider=provider,
    )
    saved = append_scout_dialogue(
        dest, role="assistant", text=note, intent_action="library_search"
    )
    return {
        "ok": True,
        "refused": False,
        "action": "library_search",
        "reply": note,
        "human_holds": str(intent.get("source") or "") == "human"
        and str(intent.get("status") or "") == INTENT_ACTIVE,
        "query": query,
        "source": origin,
        "scout": saved.get("scout"),
        "scout_intent": saved.get("scout_intent"),
        "scout_dialogue": saved.get("scout_dialogue") or [],
        "can_enter_claim_gate": False,
        "cannot": list(_INTENT_CANNOT),
    }


def _candidate_id(how_id: str, query_id: str, paper_refs: Sequence[str]) -> str:
    paper = str(paper_refs[0] if paper_refs else "paper").replace(":", "_")
    return f"howc_{how_id}_{query_id}_{paper}"


def normalize_llm_candidates(
    raw_rows: Sequence[Mapping[str, Any]] | None,
    *,
    literature: Mapping[str, Any] | None,
    round_id: str | None = None,
) -> list[dict[str, Any]]:
    """Validate LLM drafts against literature provenance. Does not register HOW."""
    rows = [dict(item) for item in (raw_rows or []) if isinstance(item, Mapping)]
    if not rows:
        return []
    blob = dict(literature or {})
    if blob.get("fail_closed"):
        raise PlannerContractError(
            "fail_closed: literature scout failed; refusing HOW candidates invented from model memory"
        )
    known_papers = literature_paper_ids(blob)
    qid = literature_query_id(blob)
    if not qid or not known_papers:
        raise PlannerContractError(
            "fail_closed: HOW candidates require literature_query_id and admissible paper_refs"
        )
    out: list[dict[str, Any]] = []
    for item in rows:
        how_id = str(item.get("how_id") or "").strip().upper()
        family = str(item.get("family") or "fusion").strip().lower()
        mechanism = str(item.get("mechanism") or item.get("hypothesis") or "").strip()
        query_id = str(item.get("literature_query_id") or qid).strip()
        papers = [str(x).strip() for x in (item.get("paper_refs") or []) if str(x).strip()]
        invented = [str(x).strip() for x in (item.get("invented_operators") or []) if str(x).strip()]
        if invented:
            raise PlannerContractError(
                f"fail_closed: invented operators not allowed on HOW candidates: {invented}"
            )
        if not how_id or not mechanism:
            raise PlannerContractError("fail_closed: HOW candidate needs how_id and mechanism")
        if family not in _ALLOWED_FAMILIES:
            raise PlannerContractError(f"fail_closed: HOW candidate family {family!r} is not allowed")
        if query_id != qid:
            raise PlannerContractError(
                "fail_closed: HOW candidate literature_query_id does not match the scout packet"
            )
        if not papers or any(pid not in known_papers for pid in papers):
            raise PlannerContractError(
                "fail_closed: HOW candidate paper_refs must be scout-admissible paper ids"
            )
        map_to = str(item.get("map_to_existing") or "").strip().upper() or None
        fusion = str(item.get("fusion_method") or "").strip() or None
        neck = str(item.get("neck_type") or "").strip() or None
        intent = str(item.get("implementation_intent") or "").strip() or None
        if intent and _CODE_HINT.search(intent):
            raise PlannerContractError(
                "fail_closed: implementation_intent must not include Python or new networks"
            )
        hay_item = {
            key: value for key, value in item.items() if key != "implementation_intent"
        }
        hay = mechanism + json.dumps(hay_item, ensure_ascii=False)
        if _CODE_HINT.search(hay):
            raise PlannerContractError(
                "fail_closed: HOW candidate must not include Python or new networks"
            )
        draft = {
            "candidate_id": str(item.get("candidate_id") or _candidate_id(how_id, qid, papers)),
            "how_id": how_id,
            "family": family,
            "mechanism": mechanism,
            "implementation_intent": intent,
            "status": STATUS_PROPOSED,
            "fusion_method": fusion,
            "neck_type": neck or "standard",
            "plugin_kind": family if family in {"fusion", "neck", "backbone_wrap"} else None,
            "map_to_existing": map_to,
            "needs_adapter_work": True,
            "literature_query_id": qid,
            "paper_refs": papers,
            "invented_operators": [],
            "source": "llm",
            "round_id": round_id,
            "human_decision": None,
            "human_actor": None,
            "decided_at": None,
            "decision_note": None,
            "created_at": _now(),
            "can_enter_claim_gate": False,
        }
        if adapter_can_map_candidate(draft) and how_id in ALLOWED_HOW:
            draft["needs_adapter_work"] = False
        validate_named("how_candidate", draft)
        out.append(draft)
    return out


def ingest_llm_candidates(
    path: Path | str,
    raw_rows: Sequence[Mapping[str, Any]] | None,
    *,
    literature: Mapping[str, Any] | None,
    round_id: str | None = None,
) -> dict[str, Any]:
    drafts = normalize_llm_candidates(raw_rows, literature=literature, round_id=round_id)
    store = load_store(path)
    if literature is not None:
        store["scout"] = summarize_scout(literature)
    existing = {str(row.get("candidate_id")): dict(row) for row in store["candidates"]}
    for draft in drafts:
        cid = str(draft["candidate_id"])
        prev = existing.get(cid)
        if prev and str(prev.get("status") or "") != STATUS_PROPOSED:
            continue
        existing[cid] = draft
    store["candidates"] = list(existing.values())
    return save_store(path, store)


def overlay_from_store(store: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    extra = dict(store.get("registered_overlay") or {})
    return {str(k).upper(): dict(v) for k, v in extra.items() if isinstance(v, Mapping)}


def resolve_pending_how(how_id: str, path: Path | str | None) -> dict[str, Any]:
    overlay = overlay_from_store(load_store(path)) if path else {}
    return resolve_how_id(how_id, overlay=overlay)


def decide_candidate(
    path: Path | str,
    candidate_id: str,
    *,
    decision: str,
    actor: str = "human",
    note: str | None = None,
    confirm_human_gate: bool = False,
) -> dict[str, Any]:
    if not confirm_human_gate:
        raise HowPendingError("HOW catalog freeze requires confirm_human_gate")
    action = str(decision or "").strip().lower()
    if action not in {"reject", "register"}:
        raise HowPendingError("decision must be reject or register")
    store = load_store(path)
    found = None
    for row in store["candidates"]:
        if str(row.get("candidate_id")) == candidate_id:
            found = row
            break
    if found is None:
        raise HowPendingError(f"unknown HOW candidate: {candidate_id}")
    if str(found.get("status")) == STATUS_REGISTERED and action == "register":
        return store
    found["decided_by"] = actor
    found["human_actor"] = actor
    found["decided_at"] = _now()
    found["decision_note"] = note
    found["human_decision"] = action
    how_id = str(found.get("how_id") or "").upper()
    if action == "reject":
        found["requires_human_review"] = False
        found["status"] = STATUS_REJECTED
        store["registered_overlay"].pop(how_id, None)
    elif bool(found.get("requires_human_review")) and not bool(found.get("smoke_ok")):
        # Invent-fallback / human-gated draft: UI "批准" means allow authoring,
        # not overlay register. Clear the gate and keep proposed for lifecycle.
        found["requires_human_review"] = False
        found["status"] = STATUS_PROPOSED
        family = str(found.get("family") or "fusion").strip().lower() or "fusion"
        found["needs_adapter_work"] = family_needs_adapter_work(family)
        found["human_decision"] = "release_for_author"
        if note is None:
            found["decision_note"] = "human approved invent draft for lifecycle author"
    elif (
        not bool(found.get("smoke_ok"))
        and bool(found.get("needs_adapter_work", True))
        and str(found.get("source") or "") in {"draft_arm", "invent_fallback", "llm", "plugin_explore"}
        and (
            str(found.get("fusion_method") or "").lower().startswith("plugin:")
            or str(found.get("neck_type") or "").lower().startswith("plugin:")
            or str(found.get("backbone_wrap_method") or "").lower().startswith("plugin:")
            or found.get("implementation_intent")
            or str(found.get("how_id") or "").upper().startswith("P")
        )
    ):
        # Literature / plugin drafts: human approve → author path, not dead pending.
        found["requires_human_review"] = False
        found["status"] = STATUS_PROPOSED
        family = str(found.get("family") or "fusion").strip().lower() or "fusion"
        found["needs_adapter_work"] = family_needs_adapter_work(family)
        found["human_decision"] = "release_for_author"
        if note is None:
            found["decision_note"] = "human approved plugin draft for lifecycle author"
    elif bool(found.get("smoke_ok")) and (
        overlay_is_plugin(found)
        or str(found.get("fusion_method") or "").lower().startswith("plugin:")
        or str(found.get("neck_type") or "").lower().startswith("plugin:")
        or found.get("plugin_relpath")
    ):
        found["requires_human_review"] = False
        found["status"] = STATUS_REGISTERED
        found["needs_adapter_work"] = False
        spec = plugin_overlay_spec(
            how_id, smoke_ok=True, plugin_kind=candidate_plugin_kind(found)
        )
        spec["plugin_relpath"] = found.get("plugin_relpath") or spec["plugin_relpath"]
        store.setdefault("registered_overlay", {})[how_id] = spec
    elif adapter_can_map_candidate(found) and how_id not in NOT_REGISTERED:
        found["requires_human_review"] = False
        found["status"] = STATUS_REGISTERED
        mapped = str(found.get("map_to_existing") or how_id).upper()
        spec = dict(ALLOWED_HOW.get(mapped) or ALLOWED_HOW.get(how_id) or {})
        if spec:
            spec = dict(spec)
            spec["id"] = how_id
            store.setdefault("registered_overlay", {})[how_id] = spec
    else:
        found["requires_human_review"] = False
        found["status"] = STATUS_PENDING_ADAPTER
        found["needs_adapter_work"] = True
    validate_named("how_candidate", found)
    return save_store(path, store)


def protocol_metric(protocol: Mapping[str, Any] | None) -> str:
    from scientist_lab.literature.query_synth import protocol_metric as _metric

    return _metric(protocol)


def fallback_scout_query(protocol: Mapping[str, Any] | None = None) -> str:
    from scientist_lab.literature.query_synth import last_ditch_fallback_query

    return last_ditch_fallback_query(protocol)


def _clean_query(query: str) -> str:
    raw = " ".join(str(query or "").split())
    if not raw:
        raise HowPendingError("scout query is empty")
    if len(raw) > MAX_SCOUT_QUERY:
        raise HowPendingError(f"scout query exceeds {MAX_SCOUT_QUERY} characters")
    if _CODE_HINT.search(raw):
        raise HowPendingError("scout query must not include Python or new networks")
    return raw


def make_scout_intent(
    *,
    source: str,
    query: str,
    why: str,
    status: str,
    drafted_by: str | None = None,
) -> dict[str, Any]:
    src = str(source or "").strip().lower()
    if src not in INTENT_SOURCES:
        raise HowPendingError("scout_intent source must be human, llm, or fallback")
    st = str(status or "").strip().lower()
    if st not in {INTENT_ACTIVE, INTENT_PROPOSED, INTENT_CLEARED}:
        raise HowPendingError("scout_intent status must be active, proposed, or cleared")
    payload = {
        "source": src,
        "status": st,
        "query": _clean_query(query) if st != INTENT_CLEARED else "",
        "why": " ".join(str(why or "").split())[:400],
        "drafted_by": drafted_by or src,
        "updated_at": _now(),
        "can_enter_claim_gate": False,
        "cannot": list(_INTENT_CANNOT),
    }
    return payload


def set_scout_intent(
    path: Path | str,
    *,
    source: str,
    query: str,
    why: str = "",
    status: str = INTENT_ACTIVE,
    drafted_by: str | None = None,
) -> dict[str, Any]:
    store = load_store(path)
    store["scout_intent"] = make_scout_intent(
        source=source,
        query=query,
        why=why,
        status=status,
        drafted_by=drafted_by,
    )
    return save_store(path, store)


def clear_scout_intent(path: Path | str, *, why: str = "") -> dict[str, Any]:
    store = load_store(path)
    store["scout_intent"] = make_scout_intent(
        source="fallback",
        query=fallback_scout_query(None),
        why=why or "human cleared scout_intent; next scout uses fallback",
        status=INTENT_CLEARED,
        drafted_by="human",
    )
    return save_store(path, store)


def accept_proposed_intent(path: Path | str) -> dict[str, Any]:
    store = load_store(path)
    intent = dict(store.get("scout_intent") or {})
    if str(intent.get("status") or "") != INTENT_PROPOSED or not str(intent.get("query") or "").strip():
        raise HowPendingError("no proposed scout_intent to accept")
    intent["status"] = INTENT_ACTIVE
    intent["updated_at"] = _now()
    intent["can_enter_claim_gate"] = False
    intent["cannot"] = list(_INTENT_CANNOT)
    store["scout_intent"] = intent
    return save_store(path, store)


def reject_proposed_intent(path: Path | str) -> dict[str, Any]:
    store = load_store(path)
    intent = dict(store.get("scout_intent") or {})
    if str(intent.get("status") or "") != INTENT_PROPOSED:
        raise HowPendingError("no proposed scout_intent to reject")
    return clear_scout_intent(path, why="human rejected LLM scout query draft")


def resolve_scout_query(
    store: Mapping[str, Any] | None,
    protocol: Mapping[str, Any] | None = None,
    evidence: Mapping[str, Any] | None = None,
    previous_plan: Mapping[str, Any] | None = None,
    *,
    live: bool = False,
    provider: Any | None = None,
) -> dict[str, Any]:
    from scientist_lab.literature.query_synth import compose_scout_queries

    return compose_scout_queries(
        store,
        protocol,
        evidence,
        previous_plan,
        live=live,
        provider=provider,
    )


def planner_literature_context(packet: Mapping[str, Any] | None) -> dict[str, Any]:
    """Slim read-only clues for Planner. Not Claim evidence. Not a HOW catalog."""
    blob = dict(packet or {})
    papers = scout_paper_summaries(blob)
    clues = []
    for row in papers[:8]:
        clues.append(
            {
                "rank": row.get("rank"),
                "paper_id": row.get("paper_id"),
                "title": row.get("title"),
                "year": row.get("year"),
                "url": row.get("url"),
                "doi": row.get("doi"),
                "link_ok": bool(row.get("link_ok")),
                "why_relevant": row.get("why_relevant"),
                "score": row.get("score"),
                "hop": int(row.get("hop") or 0),
                "retrieval_source": row.get("retrieval_source") or "retriever",
                "protocol_aligned": row.get("protocol_aligned"),
                "can_enter_claim_gate": False,
            }
        )
    admissible = []
    for row in blob.get("planner_admissible") or []:
        if not isinstance(row, Mapping):
            continue
        paper = row.get("paper") if isinstance(row.get("paper"), Mapping) else row
        if not isinstance(paper, Mapping):
            continue
        admissible.append(
            {
                "paper_id": paper.get("paper_id"),
                "title": paper.get("title"),
                "year": paper.get("year"),
                "doi": paper.get("doi"),
            }
        )
    return {
        "fail_closed": bool(blob.get("fail_closed")),
        "actual_search": bool(blob.get("actual_search")),
        "live": bool(blob.get("live")),
        "literature_query_id": literature_query_id(blob),
        "query": blob.get("query"),
        "queries": list(blob.get("queries") or []),
        "source": blob.get("intent_source") or blob.get("source"),
        "research_question": blob.get("research_question"),
        "paper_refs": [str(x) for x in (blob.get("paper_refs") or []) if str(x).strip()],
        "planner_admissible": admissible,
        "clues": clues,
        "dropped_count": int(blob.get("dropped_count") or 0),
        "filter_dropped": int(blob.get("filter_dropped") or 0),
        "hop_seed": blob.get("hop_seed"),
        "hop_added": int(blob.get("hop_added") or 0),
        "can_enter_claim_gate": False,
        "usage": (
            "Literature is a ranked read-only table (rank/title/year/url/why_relevant). "
            "Use only rows with url. It is NOT Claim evidence and MUST NOT enter ClaimGate. "
            "Do not invent a paper that is not in this table. "
            "Do not invent an unregistered HOW because a paper used it. "
            "To follow a paper: change scout_intent, emit how_candidates[] drafts "
            "with paper_refs from this payload, or propose a materializable registered HOW "
            "/ new experiment draft via the existing pending/propose path. "
            "KEEP is not a Claim."
        ),
        "claim_gate_note": str(
            blob.get("claim_gate_note")
            or "LiteratureEvidence cannot enter ClaimGate; only ExperimentEvidence can."
        ),
    }


def append_scout_dialogue(
    path: Path | str,
    *,
    role: str,
    text: str,
    intent_action: str | None = None,
    refused: bool = False,
) -> dict[str, Any]:
    store = load_store(path)
    rows = list(store.get("scout_dialogue") or [])
    rows.append(
        {
            "role": str(role),
            "text": str(text or "").strip()[:2000],
            "at": _now(),
            "intent_action": intent_action,
            "refused": bool(refused),
            "can_enter_claim_gate": False,
            "cannot": list(_INTENT_CANNOT),
        }
    )
    store["scout_dialogue"] = rows[-MAX_SCOUT_DIALOGUE:]
    return save_store(path, store)


def ingest_unmaterializable_selected(
    path: Path | str,
    row: Mapping[str, Any],
    *,
    round_id: str | None = None,
) -> dict[str, Any]:
    """LLM selected an unmaterializable HOW. Keep the id; do not rewrite to F1. No GPU."""
    how_id = str(row.get("how_id") or "").strip().upper()
    if not how_id:
        raise HowPendingError("unmaterializable selected HOW missing how_id")
    family = str(row.get("family") or "fusion").strip().lower()
    if family not in _ALLOWED_FAMILIES:
        family = "fusion"
    mechanism = str(row.get("mechanism") or "").strip() or (
        f"LLM selected unmaterializable HOW {how_id}; Adapter cannot run it"
    )
    draft = {
        "candidate_id": str(row.get("candidate_id") or f"howc_selected_{how_id}_{round_id or 'round'}"),
        "how_id": how_id,
        "family": family,
        "mechanism": mechanism,
        "status": STATUS_PROPOSED,
        "fusion_method": str(row.get("fusion_method") or "").strip() or None,
        "neck_type": str(row.get("neck_type") or "standard").strip() or "standard",
        "map_to_existing": None,
        "needs_adapter_work": True,
        "literature_query_id": str(row.get("literature_query_id") or ""),
        "paper_refs": [str(x).strip() for x in (row.get("paper_refs") or []) if str(x).strip()],
        "invented_operators": [],
        "source": "llm",
        "from_selected": True,
        "round_id": round_id,
        "human_decision": None,
        "human_actor": None,
        "decided_at": None,
        "decision_note": None,
        "created_at": _now(),
        "can_enter_claim_gate": False,
    }
    if adapter_can_map_candidate(draft) and how_id in ALLOWED_HOW:
        draft["needs_adapter_work"] = False
    validate_named("how_candidate", draft)
    store = load_store(path)
    existing = {str(item.get("candidate_id")): dict(item) for item in store["candidates"]}
    cid = str(draft["candidate_id"])
    prev = existing.get(cid)
    if not (prev and str(prev.get("status") or "") != STATUS_PROPOSED):
        existing[cid] = draft
    store["candidates"] = list(existing.values())
    return save_store(path, store)


def try_ingest_llm_candidates(
    path: Path | str,
    raw_rows: Sequence[Mapping[str, Any]] | None,
    *,
    literature: Mapping[str, Any] | None,
    round_id: str | None = None,
) -> dict[str, Any]:
    """Drop illegal HOW drafts. Do not refuse the selected registered Plan."""
    try:
        return ingest_llm_candidates(
            path, raw_rows, literature=literature, round_id=round_id
        )
    except PlannerContractError as exc:
        store = load_store(path)
        blob = dict(literature or {})
        scout = summarize_scout(blob)
        scout["how_ingest_error"] = str(exc)
        store["scout"] = scout
        return save_store(path, store)


def ingest_catalog_exploration_candidate(
    path: Path | str,
    how_id: str,
    *,
    round_id: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    """Queue a trusted catalog HOW for the next GPU round. No literature required."""
    hid = str(how_id or "").strip().upper()
    if hid not in ALLOWED_HOW:
        raise HowPendingError(f"catalog exploration refused: {hid!r} not in {CATALOG_ID}")
    spec = dict(ALLOWED_HOW[hid])
    rid = str(round_id or "round")
    draft: dict[str, Any] = {
        "candidate_id": f"howc_explore_{hid}_{rid}",
        "how_id": hid,
        "family": str(spec.get("family") or "fusion"),
        "mechanism": str(reason or f"Explore catalog HOW {hid} toward SOTA."),
        "status": STATUS_PROPOSED,
        "fusion_method": spec.get("fusion_method"),
        "neck_type": str(spec.get("neck_type") or "standard"),
        "map_to_existing": hid,
        "needs_adapter_work": False,
        "literature_query_id": "evidence_explore",
        "paper_refs": [],
        "invented_operators": [],
        "source": "evidence_explore",
        "round_id": rid,
        "human_decision": None,
        "human_actor": None,
        "decided_at": None,
        "decision_note": None,
        "created_at": _now(),
        "can_enter_claim_gate": False,
    }
    validate_named("how_candidate", draft)
    store = load_store(path)
    existing = {str(row.get("candidate_id")): dict(row) for row in store["candidates"]}
    existing[str(draft["candidate_id"])] = draft
    store["candidates"] = list(existing.values())
    return save_store(path, store)


def ingest_plugin_exploration_candidate(
    path: Path | str,
    how_id: str,
    mechanism: str,
    *,
    round_id: str | None = None,
    semantic_ref: str | None = None,
) -> dict[str, Any]:
    """Queue a plugin HOW slot for LLM author + smoke. Not a catalog preset."""
    hid = str(how_id or "").strip().upper()
    if not hid or hid in ALLOWED_HOW:
        raise HowPendingError("plugin exploration requires a non-catalog HOW id (e.g. P1)")
    rid = str(round_id or "round")
    body = str(mechanism or "").strip()
    if semantic_ref:
        body = f"{body} (from reviewer: {semantic_ref})"
    draft: dict[str, Any] = {
        "candidate_id": f"howc_plugin_{hid}_{rid}",
        "how_id": hid,
        "family": "fusion",
        "mechanism": body[:1200],
        "implementation_intent": "Author FeatureFusion plugin in sandbox; smoke before GPU.",
        "status": STATUS_PROPOSED,
        "fusion_method": f"plugin:{hid.lower()}",
        "neck_type": "standard",
        "map_to_existing": None,
        "needs_adapter_work": True,
        "literature_query_id": "semantic_evidence",
        "paper_refs": [],
        "invented_operators": [],
        "source": "plugin_explore",
        "round_id": rid,
        "human_decision": None,
        "human_actor": None,
        "decided_at": None,
        "decision_note": None,
        "created_at": _now(),
        "can_enter_claim_gate": False,
    }
    validate_named("how_candidate", draft)
    store = load_store(path)
    existing = {str(row.get("candidate_id")): dict(row) for row in store["candidates"]}
    existing[str(draft["candidate_id"])] = draft
    store["candidates"] = list(existing.values())
    return save_store(path, store)
