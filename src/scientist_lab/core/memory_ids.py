"""Stable, citation-friendly LESSON/STRATEGY id bodies from run_id.

Nested campaign run_ids grow as ``run_plan_roundN_from_<parent>...`` and
quickly exceed what live planners can cite verbatim. Keep short run_ids
unchanged; compress only when the body would blow past the cite budget.
Also provide filesystem-safe archive slugs (Windows component limit 255).
"""

from __future__ import annotations

import hashlib
import re

# Live LLM memory_refs often clip past ~80–100 chars of the full LESSON-... id.
_MEMORY_ID_BODY_MAX = 56
# Leave headroom under Windows 255-char path component + ".json".
_ARCHIVE_SLUG_MAX = 96


def memory_id_body_from_run_id(run_id: str) -> str:
    rid = str(run_id or "").strip() or "unknown"
    if len(rid) <= _MEMORY_ID_BODY_MAX:
        return rid
    digest = hashlib.sha1(rid.encode("utf-8")).hexdigest()[:12]
    tip = re.sub(r"[^A-Za-z0-9_]+", "_", rid[-24:]).strip("_") or "x"
    return f"{digest}_{tip}"


def lesson_id_for_run(run_id: str, *, semantic: bool = False) -> str:
    body = memory_id_body_from_run_id(run_id)
    if semantic:
        return f"LESSON-{body}-semantic-001"
    return f"LESSON-{body}-001"


def strategy_id_for_run(run_id: str) -> str:
    return f"STRATEGY-{memory_id_body_from_run_id(run_id)}-001"


def next_plan_id(*, round_index: int, parent_run_id: str) -> str:
    """Short plan_id for Round N+1. Parent lineage stays in parent_run_id field.

    Legacy form ``plan_roundN_from_<full_parent>`` nested until Windows paths
    overflowed (~255 char component). New form is bounded.
    """
    parent = str(parent_run_id or "").strip() or "unknown"
    digest = hashlib.sha1(parent.encode("utf-8")).hexdigest()[:10]
    return f"plan_r{int(round_index)}_{digest}"


def archive_run_slug(run_id: str, *, max_len: int = _ARCHIVE_SLUG_MAX) -> str:
    """Filesystem-safe directory / file stem for runs/<slug>[.json]."""
    rid = str(run_id or "").strip() or "unknown"
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", rid).strip(" ._") or "unknown"
    limit = max(32, int(max_len))
    if len(safe) <= limit:
        return safe
    digest = hashlib.sha1(rid.encode("utf-8")).hexdigest()[:16]
    tip = re.sub(r"[^A-Za-z0-9_]+", "_", safe[:40]).strip("_") or "run"
    return f"{tip}_{digest}"
