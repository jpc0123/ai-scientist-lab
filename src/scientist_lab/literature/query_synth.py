"""Compose scout queries from intent, research question, and last-shot evidence.

Not a fifth Agent. Fallback is always labeled. Human intent is never overwritten.
Literature queries are clues, not Claim evidence.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from scientist_lab.llm.errors import MissingAPIKeyError, RealProviderNotEnabledError
from scientist_lab.llm.gateway import complete_chat
from scientist_lab.llm.models import LLMRequest
from scientist_lab.llm.schema_parser import extract_json_object

INTENT_SOURCES = frozenset({"human", "llm", "fallback"})
INTENT_ACTIVE = "active"
MAX_QUERY = 400
MAX_SUPPLEMENTAL = 2

_STOP = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "for",
        "in",
        "on",
        "to",
        "vs",
        "versus",
        "does",
        "do",
        "did",
        "is",
        "are",
        "under",
        "with",
        "from",
        "and",
        "or",
        "that",
        "this",
        "what",
        "how",
        "when",
        "can",
        "we",
        "our",
        "into",
        "than",
        "then",
        "its",
        "via",
        "using",
        "use",
        "used",
        "not",
        "but",
        "as",
        "by",
        "at",
        "be",
        "been",
        "was",
        "were",
        "will",
        "would",
        "should",
        "may",
        "might",
        "if",
        "it",
        "their",
        "whether",
        "after",
        "before",
        "about",
        "over",
        "between",
        "within",
        "without",
        "across",
        "again",
        "same",
        "new",
        "next",
        "last",
        "one",
        "two",
        "three",
        "both",
        "all",
        "any",
        "each",
        "research",
        "question",
        "study",
        "paper",
        "method",
        "approach",
        "based",
        "constitution",
        "campaign",
        "autonomous",
        "loop",
        "scientist",
        "lab",
        "official",
        "notes",
        "historical",
        "replay",
        "baseline",
        "compared",
        "compare",
        "comparison",
        "relative",
        "help",
        "change",
        "changes",
        "changed",
    }
)
_KEEP_SHORT = frozenset(
    {
        "aps",
        "ap",
        "f0",
        "f1",
        "f3",
        "n0",
        "n1",
        "a4",
        "rgb",
        "map",
        "val",
        "iou",
        "neck",
        "how",
    }
)
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_+/-]{1,}|[\u4e00-\u9fff]{2,}")
_BOILERPLATE = re.compile(
    r"campaign constitution|autonomous loop|historical replay|not official",
    re.IGNORECASE,
)
_CONSTITUTION_NOTES = re.compile(
    r"\bconstitution\b|\bhistorical replay\b|\bautonomous loop\b|human gate may|not official",
    re.IGNORECASE,
)
_V26_STOCK = (
    "RGB-T fusion low-light small object detection gated weighting"
)

# Internal lab tokens that poison Semantic Scholar keyword search (Live A: 0 hits).
_INTERNAL_ID = re.compile(
    r"\b(?:dataset:)?(?:rgbt[_-]?tiny[_-]?v\d+|low[_-]?light[_-]?subset[_-]?v\d+|"
    r"aps[_-]?lowlight|map50[_-]?95[_-]?lowlight|ap50[_-]?lowlight|"
    r"fp-[a-z0-9_-]+|research_protocol_[a-z0-9_]+)\b",
    re.IGNORECASE,
)
_HOW_TOKEN = re.compile(r"\b(?:F0|F1|F3|N0|N1|A4)\b")
_DATASET_GLOSS = {
    "rgbt_tiny_v1": "RGB-T",
    "rgbt_tiny": "RGB-T",
}
_SLICE_GLOSS = {
    "low_light_subset_v1": "low-light",
    "low_light": "low-light",
}
_METRIC_DROP = frozenset(
    {
        "aps_lowlight",
        "map50_95_lowlight",
        "ap50_lowlight",
        "map50_95",
        "map50",
    }
)
# Model / dataset proper nouns that over-constrain Semantic Scholar AND search.
_S2_NOISE_TERMS = frozenset(
    {
        "d-fine",
        "dfine",
        "d_fine",
        "rgbt-tiny",
        "rgbt_tiny",
        "improvement",
        "improve",
        "enhanced",
        "enhancement",
        "under",
        "using",
        "via",
        "based",
        "method",
        "methods",
        "approach",
        "complementary",
    }
)
_MAX_S2_TOKENS = 8
_SAFE_S2_FALLBACK = "RGB-T low-light small object detection"

SCOUT_QUERIES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["query", "why"],
    "additionalProperties": True,
    "properties": {
        "query": {"type": "string", "minLength": 8, "maxLength": 400},
        "supplemental_queries": {
            "type": "array",
            "maxItems": 2,
            "items": {"type": "string", "maxLength": 400},
        },
        "why": {"type": "string", "maxLength": 400},
    },
}


def protocol_metric(protocol: Mapping[str, Any] | None) -> str:
    objective = dict((protocol or {}).get("objective") or (protocol or {}).get("goal") or {})
    primary = objective.get("primary") if isinstance(objective.get("primary"), Mapping) else {}
    return str((primary or {}).get("metric") or "APS_lowlight")


def _strip_boilerplate(text: str) -> str:
    cleaned = _BOILERPLATE.sub(" ", str(text or ""))
    return " ".join(cleaned.split()).strip()


def protocol_research_question(protocol: Mapping[str, Any] | None) -> str:
    """Read the experiment's research question. LLM drafts store it in goal.notes."""
    blob = dict(protocol or {})
    direct = _strip_boilerplate(str(blob.get("research_question") or ""))
    if direct:
        return direct
    notes = str(blob.get("notes") or "")
    marker = re.search(
        r"LLM research question:\s*(.+?)(?:\s+KEEP|\s+This document|\s*$)",
        notes,
        re.IGNORECASE | re.DOTALL,
    )
    if marker:
        found = _strip_boilerplate(marker.group(1))
        if found:
            return found
    goal = dict(blob.get("goal") or {})
    goal_notes = str(goal.get("notes") or "").strip()
    if goal_notes and not _CONSTITUTION_NOTES.search(goal_notes):
        cleaned = _strip_boilerplate(goal_notes)
        if cleaned:
            return cleaned
    title = _strip_boilerplate(str(blob.get("title") or ""))
    if title:
        return title
    return ""


def protocol_fingerprint(
    protocol: Mapping[str, Any] | None,
    previous_plan: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    blob = dict(protocol or {})
    baseline = dict(blob.get("baseline") or {})
    slice_row = dict(blob.get("condition_slice") or {})
    dataset = str(baseline.get("dataset") or "").replace("dataset:", "").strip()
    slice_id = str(slice_row.get("id") or "").strip()
    how = str((previous_plan or {}).get("how_id") or "").strip().upper()
    return {
        "dataset": dataset,
        "slice_id": slice_id,
        "metric": protocol_metric(blob),
        "adapter": str(baseline.get("adapter") or "").strip(),
        "seed_how_id": how,
        "fingerprint_id": str(blob.get("fingerprint_id") or "").strip(),
    }


def key_terms(*parts: Any, limit: int = 10) -> list[str]:
    """Domain tokens from the research question / fingerprint. Not stopwords."""
    hay = " ".join(str(p or "") for p in parts)
    found: list[str] = []
    seen: set[str] = set()
    for raw in _TOKEN.findall(hay):
        token = raw.strip()
        low = token.lower().replace("_", "-")
        if low in _STOP:
            continue
        if len(token) < 3 and low not in _KEEP_SHORT:
            continue
        if low in seen:
            continue
        seen.add(low)
        found.append(token)
        if len(found) >= limit:
            break
    return found


def _clean_query(query: str) -> str:
    raw = " ".join(str(query or "").split())
    return raw[:MAX_QUERY].strip()


def strip_internal_search_tokens(query: str) -> str:
    """Remove lab dataset/slice/metric/HOW ids that make Semantic Scholar return 0 hits.

    Keeps academic surface terms (RGB-T, low-light, gated multiscale, …).
    """
    text = _INTERNAL_ID.sub(" ", str(query or ""))
    text = _HOW_TOKEN.sub(" ", text)
    return compact_s2_query(text)


def compact_s2_query(query: str, *, max_tokens: int = _MAX_S2_TOKENS) -> str:
    """Drop model/dataset proper nouns and cap token count for S2 keyword AND search.

    Live A showed: '... D-FINE ... RGBT-Tiny ...' → 0 hits, while
    'RGB-T low-light small object detection' → 5 hits.
    """
    raw = _clean_query(query)
    kept: list[str] = []
    seen: set[str] = set()
    for token in raw.split():
        low = token.lower().strip(".,;:()[]{}\"'")
        if not low or low in _S2_NOISE_TERMS or low in _METRIC_DROP:
            continue
        if _INTERNAL_ID.search(token) or _HOW_TOKEN.fullmatch(token):
            continue
        if low in seen:
            continue
        seen.add(low)
        kept.append(token)
        if len(kept) >= max_tokens:
            break
    return _clean_query(" ".join(kept))


def safe_s2_fallback_query(protocol: Mapping[str, Any] | None = None) -> str:
    """Short query proven to hit S2 for this research line. No D-FINE / RGBT-Tiny."""
    academic = [
        t
        for t in fingerprint_academic_terms(protocol)
        if t.lower() not in _S2_NOISE_TERMS
    ]
    if academic:
        return compact_s2_query(" ".join(academic[:5] + ["detection"]), max_tokens=6) or _SAFE_S2_FALLBACK
    return _SAFE_S2_FALLBACK


def fingerprint_academic_terms(
    protocol: Mapping[str, Any] | None = None,
    previous_plan: Mapping[str, Any] | None = None,
) -> list[str]:
    """Map frozen fingerprint ids to academic English. Never emit raw lab ids."""
    fp = protocol_fingerprint(protocol, previous_plan)
    out: list[str] = []
    seen: set[str] = set()

    def _add(token: str) -> None:
        raw = str(token or "").strip()
        if not raw:
            return
        low = raw.lower()
        if low in seen or low in _METRIC_DROP or low in _S2_NOISE_TERMS:
            return
        seen.add(low)
        out.append(raw)

    dataset = str(fp.get("dataset") or "").replace("dataset:", "").strip()
    slice_id = str(fp.get("slice_id") or "").strip()
    dataset_key = dataset.lower().replace("-", "_")
    slice_key = slice_id.lower().replace("-", "_")
    _add(_DATASET_GLOSS.get(dataset_key) or "")
    if dataset and dataset_key not in _DATASET_GLOSS:
        # Unknown dataset: keep human words only (drop *_vN ids).
        gloss = strip_internal_search_tokens(dataset.replace("_", " "))
        if gloss and not re.search(r"\bv\d+\b", gloss, re.IGNORECASE):
            _add(gloss)
    _add(_SLICE_GLOSS.get(slice_key) or "")
    if slice_id and slice_key not in _SLICE_GLOSS:
        gloss = strip_internal_search_tokens(slice_id.replace("_", " "))
        if gloss and "subset" not in gloss.lower():
            _add(gloss)
    rq = protocol_research_question(protocol)
    for term in key_terms(rq, limit=8):
        if term.lower() in _METRIC_DROP or term.lower() in _S2_NOISE_TERMS:
            continue
        if _INTERNAL_ID.search(term) or _HOW_TOKEN.fullmatch(term):
            continue
        _add(term)
    return out


def ensure_key_terms(query: str, terms: Sequence[str]) -> str:
    """If a generated query dropped research-question entities, put them back.

    Skips internal lab ids — those belong in provenance, not in the S2 string.
    """
    text = strip_internal_search_tokens(query)
    blob = text.lower()
    academic = [
        term
        for term in terms
        if term
        and term.lower() not in _METRIC_DROP
        and term.lower() not in _S2_NOISE_TERMS
        and not _INTERNAL_ID.search(str(term))
        and not _HOW_TOKEN.fullmatch(str(term))
    ]
    missing = [term for term in academic if term.lower() not in blob]
    if missing:
        text = _clean_query(f"{text} {' '.join(missing[:4])}")
    return strip_internal_search_tokens(text)


def last_ditch_fallback_query(protocol: Mapping[str, Any] | None = None) -> str:
    """Labeled last resort. Prefer a short S2-safe academic string."""
    return safe_s2_fallback_query(protocol)


def catalog_query_terms(how_id: str) -> list[str]:
    """Registered HOW surface terms only. Never late/mid fusion or other unregistered ops."""
    token = str(how_id or "").strip().upper()
    if not token:
        return []
    from scientist_lab.adapters.dfine.how_catalog import ALLOWED_HOW, NOT_REGISTERED

    if token in NOT_REGISTERED:
        return [token]
    spec = dict(ALLOWED_HOW.get(token) or {})
    terms: list[str] = [token]
    for key in ("fusion_method", "neck_type", "existing_capability", "input_mode"):
        raw = str(spec.get(key) or "").replace("_", " ").strip()
        if raw and raw.lower() not in {"none", "standard"}:
            terms.append(raw)
    return terms


def memory_query_terms(evidence: Mapping[str, Any] | None, limit: int = 6) -> list[str]:
    """Key tokens from written lessons/strategies. Not a vector memory."""
    blob = dict(evidence or {})
    parts: list[str] = []
    for row in list(blob.get("lessons") or [])[:3]:
        if isinstance(row, Mapping):
            parts.append(str(row.get("statement") or ""))
        else:
            parts.append(str(row or ""))
    for row in list(blob.get("strategies") or [])[:2]:
        if isinstance(row, Mapping):
            parts.append(str(row.get("target") or ""))
            parts.append(str(row.get("action") or ""))
        else:
            parts.append(str(row or ""))
    return key_terms(*parts, limit=limit)


def _heuristic_queries(
    *,
    protocol: Mapping[str, Any] | None,
    evidence: Mapping[str, Any] | None,
    previous_plan: Mapping[str, Any] | None,
) -> dict[str, Any]:
    blob = dict(evidence or {})
    fp = protocol_fingerprint(protocol, previous_plan)
    rq = protocol_research_question(protocol) or str(blob.get("research_question") or "")
    last_how = str(blob.get("last_how_id") or fp.get("seed_how_id") or "").upper()
    decision = str(blob.get("last_review_decision") or "").upper()
    academic = fingerprint_academic_terms(protocol, previous_plan)
    # Catalog surface words only (gated multiscale / fdpn), never bare F3/N1 tokens.
    how_surface: list[str] = []
    seen_how: set[str] = set()
    for token in catalog_query_terms(last_how)[1:]:
        low = str(token).strip().lower()
        if not low or low in seen_how:
            continue
        seen_how.add(low)
        how_surface.append(str(token).strip())
    mem = [
        t
        for t in memory_query_terms(blob)
        if t.lower() not in _METRIC_DROP and not _INTERNAL_ID.search(t)
    ]
    terms = list(academic)
    for token in (*how_surface, *mem):
        if token and token.lower() not in {t.lower() for t in terms}:
            terms.append(token)
    parts = list(terms)
    why_bits = [
        "no active scout_intent; heuristic from research question + academic fingerprint glosses "
        "(no lab dataset/slice/metric ids in the S2 string)"
    ]
    extra: list[str] = []
    how_phrase = " ".join(how_surface[:2]).strip()
    if mem:
        extra.extend(mem[:4])
        why_bits.append("写入的 Lesson/Strategy 术语进了 query，不是论文向量库。")
    if decision == "DISCARD" and (how_phrase or last_how):
        extra.append(f"alternative to {how_phrase or 'prior fusion'}")
        why_bits.append(f"上一枪 HOW={last_how} DISCARD；查替代，不把文献当 Claim。")
    elif decision == "KEEP":
        extra.append("complementary method")
        if how_phrase:
            extra.append(how_phrase)
        why_bits.append(f"上一枪 HOW={last_how or '—'} KEEP（不是 Claim）；查互补线索。")
    elif decision:
        why_bits.append(f"上一枪 {decision} HOW={last_how or '—'}")
    primary = ensure_key_terms(_clean_query(" ".join(parts + extra)), academic)
    if decision == "DISCARD" and "alternative" not in primary.lower():
        primary = compact_s2_query(f"alternative {primary}")
    if not primary:
        primary = last_ditch_fallback_query(protocol)
    supplemental: list[str] = []
    if academic:
        supp = strip_internal_search_tokens(
            " ".join(
                academic[:6]
                + ([f"alternative {how_phrase}"] if decision == "DISCARD" and how_phrase else [])
            )
        )
        if supp and supp.lower() != primary.lower():
            supplemental.append(supp)
    if how_phrase:
        supp2 = strip_internal_search_tokens(
            f"{' '.join(academic[:3])} {how_phrase} small object detection"
        )
        if supp2 and supp2.lower() not in {primary.lower(), *(s.lower() for s in supplemental)}:
            supplemental.append(supp2)
    return {
        "query": primary,
        "queries": [primary, *supplemental[:MAX_SUPPLEMENTAL]],
        "source": "fallback",
        "why": " ".join(why_bits)[:400],
        "status": "fallback",
        "fallback": True,
        "research_question": rq,
        "key_terms": academic,
        "year_from": 2022,
        "fingerprint": fp,
    }


def _llm_compose(
    *,
    protocol: Mapping[str, Any] | None,
    evidence: Mapping[str, Any] | None,
    previous_plan: Mapping[str, Any] | None,
    live: bool,
    provider: Any | None,
) -> dict[str, Any] | None:
    """Live only. Missing key → None (caller falls back, labeled)."""
    if not live:
        return None
    rq = protocol_research_question(protocol)
    fp = protocol_fingerprint(protocol, previous_plan)
    blob = dict(evidence or {})
    last_how = str((blob.get("last_how_id") or fp.get("seed_how_id") or "")).upper()
    terms = fingerprint_academic_terms(protocol, previous_plan)
    how_surface = [t for t in catalog_query_terms(last_how)[1:] if t]
    user = {
        "task": "Propose literature search queries for the next HOW scout.",
        "rules": [
            "Return JSON {query, supplemental_queries, why} only.",
            "query is the primary Semantic Scholar keyword string.",
            "Optional supplemental_queries: at most 2 narrower strings.",
            "MUST include the research-question academic entities (e.g. RGB-T, low-light, small object).",
            "Do NOT put internal lab ids into the query: no rgbt_tiny_v1, low_light_subset_v1, APS_lowlight, FP-*, F0/F1/F3/N0/N1/A4 tokens.",
            "Do NOT put D-FINE, DFINE, or RGBT-Tiny into the query — Semantic Scholar returns zero hits when those are AND-combined.",
            "Keep the primary query short (<=8 content words). Prefer 'RGB-T low-light small object detection'.",
            "Translate fingerprint ids to academic English (dataset→RGB-T, slice→low-light).",
            "If last HOW was DISCARD, search alternatives using that HOW's catalog surface terms (gated multiscale, fdpn), not bare letter codes.",
            "Do not register HOW, start GPU, or write a Claim.",
            "Do not invent operators or Python.",
            "Do not reuse a generic RGB-T gated-weighting template if the research question is different.",
            "Literature is a clue only; KEEP is not a Claim.",
        ],
        "research_question": rq,
        "key_terms_must_appear": terms,
        "catalog_surface_terms_for_last_how": how_surface,
        "fingerprint_for_context_only_not_for_query": fp,
        "evidence": dict(evidence or {}),
        "previous_plan_how_id": str((previous_plan or {}).get("how_id") or ""),
    }
    request = LLMRequest(
        purpose="other",
        messages=[
            {
                "role": "system",
                "content": (
                    "You write keyword queries for Semantic Scholar. "
                    "Stay faithful to the experiment research question in academic English. "
                    "Never paste internal dataset_id / slice_id / metric names into the query string -- "
                    "those make Semantic Scholar return zero hits. "
                    "You cannot change the HOW catalog or enter ClaimGate."
                ),
            },
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        response_schema=SCOUT_QUERIES_SCHEMA,
        temperature=0.0,
        metadata={"scout_query_synth": True},
    )
    try:
        response = complete_chat(request, provider=provider, live=live)
    except (MissingAPIKeyError, RealProviderNotEnabledError):
        return None
    except Exception:  # noqa: BLE001 — query synth must not sink the scout
        return None
    raw = response.parsed_json if isinstance(response.parsed_json, dict) else None
    if raw is None:
        try:
            raw = extract_json_object(response.content or "")
        except (ValueError, json.JSONDecodeError):
            return None
    query = ensure_key_terms(str((raw or {}).get("query") or "").strip(), terms)
    if len(query) < 8:
        return None
    extra_raw = list((raw or {}).get("supplemental_queries") or [])
    supplemental: list[str] = []
    for item in extra_raw[:MAX_SUPPLEMENTAL]:
        text = ensure_key_terms(str(item or "").strip(), terms[:4])
        if len(text) >= 8 and text.lower() != query.lower():
            supplemental.append(text)
    why = str((raw or {}).get("why") or "").strip() or "LLM 按研究问题写的检索词，source=llm。"
    return {
        "query": query,
        "queries": [query, *supplemental],
        "source": "llm",
        "why": why[:400],
        "status": "generated",
        "fallback": False,
        "research_question": rq,
        "key_terms": terms,
        "year_from": 2022,
        "fingerprint": fp,
    }


def compose_scout_queries(
    store: Mapping[str, Any] | None = None,
    protocol: Mapping[str, Any] | None = None,
    evidence: Mapping[str, Any] | None = None,
    previous_plan: Mapping[str, Any] | None = None,
    *,
    live: bool = False,
    provider: Any | None = None,
) -> dict[str, Any]:
    """Priority: active human intent > accepted LLM draft > live LLM synth > heuristic fallback."""
    intent = dict((store or {}).get("scout_intent") or {})
    query = str(intent.get("query") or "").strip()
    status = str(intent.get("status") or "")
    source = str(intent.get("source") or "")
    rq = protocol_research_question(protocol)
    fp = protocol_fingerprint(protocol, previous_plan)
    if status == INTENT_ACTIVE and query:
        origin = source if source in INTENT_SOURCES else "human"
        authored = _clean_query(query)
        # Human query is preserved verbatim for intent/UI. Live fire path in
        # scout_literature_for_how strips lab ids before calling Semantic Scholar.
        queries = [authored]
        if origin != "human":
            queries = [strip_internal_search_tokens(authored) or authored]
            extra = _heuristic_queries(
                protocol=protocol, evidence=evidence, previous_plan=previous_plan
            )
            for item in extra.get("queries") or []:
                text = strip_internal_search_tokens(str(item))
                if text and text.lower() not in {q.lower() for q in queries}:
                    queries.append(text)
                if len(queries) >= 1 + MAX_SUPPLEMENTAL:
                    break
        return {
            "query": queries[0],
            "queries": queries[: 1 + MAX_SUPPLEMENTAL],
            "source": origin,
            "why": str(intent.get("why") or ""),
            "status": INTENT_ACTIVE,
            "fallback": False,
            "research_question": rq,
            "key_terms": fingerprint_academic_terms(protocol, previous_plan)
            if origin != "human"
            else key_terms(authored),
            "year_from": 2022,
            "fingerprint": fp,
            "authored_query": authored,
        }
    generated = _llm_compose(
        protocol=protocol,
        evidence=evidence,
        previous_plan=previous_plan,
        live=bool(live),
        provider=provider,
    )
    if generated is not None:
        return generated
    return _heuristic_queries(
        protocol=protocol, evidence=evidence, previous_plan=previous_plan
    )
