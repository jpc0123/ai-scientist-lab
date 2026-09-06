"""Locale labels for ranked scout rows. Original title/url stay for audit.

LLM translates display fields only. Fail-soft: missing live LLM keeps originals.
Never invents papers, URLs, or Claim evidence.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from scientist_lab.llm.errors import MissingAPIKeyError, RealProviderNotEnabledError
from scientist_lab.llm.gateway import complete_chat
from scientist_lab.llm.models import LLMRequest
from scientist_lab.llm.schema_parser import extract_json_object

LOCALES = frozenset({"zh", "en"})
_CJK = re.compile(r"[\u4e00-\u9fff]")

DISPLAY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["papers"],
    "additionalProperties": True,
    "properties": {
        "papers": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["paper_id", "title"],
                "additionalProperties": True,
                "properties": {
                    "paper_id": {"type": "string"},
                    "title": {"type": "string", "maxLength": 400},
                    "why_relevant": {"type": "string", "maxLength": 400},
                },
            },
        }
    },
}


def normalize_locale(locale: str | None) -> str:
    raw = str(locale or "").strip().lower()
    if raw.startswith("zh"):
        return "zh"
    if raw.startswith("en"):
        return "en"
    return "zh"


def has_cjk(text: str) -> bool:
    return bool(_CJK.search(str(text or "")))


def identity_locale(text: str) -> str:
    return "zh" if has_cjk(text) else "en"


def pick_display_text(row: Mapping[str, Any] | None, field: str, locale: str | None) -> str:
    """UI helper: locale-specific label, else original."""
    blob = dict(row or {})
    loc = normalize_locale(locale)
    original = str(blob.get(field) or "").strip()
    labeled = str(blob.get(f"{field}_{loc}") or "").strip()
    return labeled or original


def apply_identity_labels(row: Mapping[str, Any]) -> dict[str, Any]:
    """Copy originals into title_en/title_zh when the script already matches."""
    item = dict(row)
    title = str(item.get("title") or "").strip()
    why = str(item.get("why_relevant") or "").strip()
    if title:
        if has_cjk(title):
            if not str(item.get("title_zh") or "").strip():
                item["title_zh"] = title
        elif not str(item.get("title_en") or "").strip():
            item["title_en"] = title
    if why:
        if has_cjk(why):
            if not str(item.get("why_relevant_zh") or "").strip():
                item["why_relevant_zh"] = why
        elif not str(item.get("why_relevant_en") or "").strip():
            item["why_relevant_en"] = why
    item["can_enter_claim_gate"] = False
    return item


def _needs_llm(row: Mapping[str, Any], locale: str) -> bool:
    loc = normalize_locale(locale)
    title = str(row.get("title") or "")
    if loc == "zh":
        return bool(title) and not str(row.get("title_zh") or "").strip()
    return bool(title) and has_cjk(title) and not str(row.get("title_en") or "").strip()


def translate_ranked_papers(
    papers: Sequence[Mapping[str, Any]] | None,
    *,
    locale: str,
    live: bool = False,
    provider: Any | None = None,
) -> list[dict[str, Any]]:
    """Fill title_{locale} / why_relevant_{locale}. Does not change url/rank/paper_id."""
    loc = normalize_locale(locale)
    rows = [apply_identity_labels(row) for row in (papers or []) if isinstance(row, Mapping)]
    pending = [row for row in rows if _needs_llm(row, loc)]
    if not pending or not live:
        return rows
    payload = {
        "task": "Translate literature scout display fields.",
        "target_locale": loc,
        "rules": [
            "Return JSON {papers:[{paper_id,title,why_relevant}]} only.",
            "Keep paper_id unchanged. Do not add or drop papers.",
            "Do not change URLs, ranks, scores, or DOIs.",
            "Do not invent a paper. Do not write a Claim.",
            "title/why_relevant must be in the target locale.",
        ],
        "papers": [
            {
                "paper_id": row.get("paper_id"),
                "title": row.get("title"),
                "why_relevant": row.get("why_relevant") or "",
            }
            for row in pending
        ],
    }
    request = LLMRequest(
        purpose="other",
        messages=[
            {
                "role": "system",
                "content": (
                    "You translate academic paper titles and one-line relevance notes. "
                    "You cannot search the web or invent papers."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        response_schema=DISPLAY_SCHEMA,
        temperature=0.0,
        metadata={"scout_display_translate": True, "locale": loc},
    )
    try:
        response = complete_chat(request, provider=provider, live=live)
    except (MissingAPIKeyError, RealProviderNotEnabledError):
        return rows
    except Exception:  # noqa: BLE001 — display translate must not sink scout
        return rows
    raw = response.parsed_json if isinstance(response.parsed_json, dict) else None
    if raw is None:
        try:
            raw = extract_json_object(response.content or "")
        except (ValueError, json.JSONDecodeError):
            return rows
    by_id: dict[str, dict[str, Any]] = {}
    for item in list((raw or {}).get("papers") or []):
        if not isinstance(item, Mapping):
            continue
        pid = str(item.get("paper_id") or "").strip()
        if pid:
            by_id[pid] = dict(item)
    for row in rows:
        pid = str(row.get("paper_id") or "")
        hit = by_id.get(pid)
        if not hit:
            continue
        title = str(hit.get("title") or "").strip()[:400]
        why = str(hit.get("why_relevant") or "").strip()[:400]
        if loc == "zh":
            if title and not has_cjk(title):
                title = ""
            if why and not has_cjk(why):
                why = ""
        elif loc == "en":
            if title and has_cjk(title):
                title = ""
            if why and has_cjk(why):
                why = ""
        wrote = False
        if title:
            row[f"title_{loc}"] = title
            wrote = True
        if why:
            row[f"why_relevant_{loc}"] = why
            wrote = True
        if wrote:
            row["display_translated"] = True
        row["can_enter_claim_gate"] = False
    return rows
