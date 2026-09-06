"""Draft Arm: literature → HOW drafts; invent only as human-gated fallback.

Not a fifth Agent. Does not write Python, start GPU, or enter ClaimGate.
Planner still selects materializable HOW only; this arm only fills pending drafts.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from scientist_lab.adapters.dfine.how_catalog import (
    family_needs_adapter_work,
    protocol_invent_kinds,
)
from scientist_lab.core.how_pending import (
    STATUS_PROPOSED,
    HowPendingError,
    ingest_llm_candidates,
    literature_paper_ids,
    literature_query_id,
    load_store,
    save_store,
    summarize_scout,
)
from scientist_lab.core.schema_registry import validate_named
from scientist_lab.llm.errors import MissingAPIKeyError, RealProviderNotEnabledError
from scientist_lab.llm.models import LLMRequest
from scientist_lab.llm.schema_parser import extract_json_object, validate_against_schema

MAX_DRAFT_ARM_TICKS = 6
_CODE_HINT = re.compile(
    r"(```|write python|\bdef\s+\w|\bclass\s+\w+\s*[:\(]|\bimport\s+\w|"
    r"\bfrom\s+\w+\s+import\b|new network|\bfdpn\b)",
    re.IGNORECASE,
)
_PLUGIN_HOW = re.compile(r"^P\d+[A-Z]?$", re.IGNORECASE)

LIT_SCHEMA = {
    "type": "object",
    "required": ["action", "reason", "how_candidates"],
    "properties": {
        "action": {
            "type": "string",
            "enum": ["draft_from_literature", "cannot_draft"],
        },
        "reason": {"type": "string"},
        "how_candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["how_id", "family", "mechanism", "paper_refs"],
                "properties": {
                    "how_id": {"type": "string"},
                    "family": {
                        "type": "string",
                        "enum": ["fusion", "neck", "training", "backbone_wrap"],
                    },
                    "mechanism": {"type": "string"},
                    "implementation_intent": {"type": "string"},
                    "paper_refs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "fusion_method": {"type": ["string", "null"]},
                },
            },
        },
    },
}

INVENT_SCHEMA = {
    "type": "object",
    "required": ["how_id", "family", "mechanism", "implementation_intent", "reason"],
    "properties": {
        "how_id": {"type": "string"},
        "family": {"type": "string", "enum": ["fusion", "neck", "training", "backbone_wrap"]},
        "mechanism": {"type": "string"},
        "implementation_intent": {"type": "string"},
        "reason": {"type": "string"},
        "fusion_method": {"type": ["string", "null"]},
    },
}

def _slot_lines(kinds: frozenset[str]) -> str:
    allowed = kinds or frozenset({"fusion"})
    parts: list[str] = []
    if "fusion" in allowed:
        parts.append(
            "family=fusion → PLUGIN_KIND=fusion and def build_fusion → FeatureFusion"
        )
    if "neck" in allowed:
        parts.append(
            "family=neck → PLUGIN_KIND=neck and def build_neck → HybridEncoder-compatible "
            "encoder (list×3 P3/P4/P5 in, hidden_dim channels out); same D-FINE transformer slot as N1"
        )
    if "backbone_wrap" in allowed:
        parts.append(
            "family=backbone_wrap → PLUGIN_KIND=backbone_wrap and def build_backbone_wrap; "
            "requires protocol.baseline.architecture_id and a new R0"
        )
    return "; ".join(parts)


def _lit_system(kinds: frozenset[str]) -> str:
    allowed = kinds or frozenset({"fusion"})
    families = ", ".join(sorted(allowed))
    return (
        "You write HOW plugin drafts for Scientist Lab from scouted papers only. "
        "Protocol is frozen. Return JSON "
        '{"action":"draft_from_literature"|"cannot_draft","reason":"...","how_candidates":[...]}. '
        "Each candidate needs how_id (Pn plugin id unused in suggested_how_id / occupied list; "
        "never reuse catalog F0/F1/F3/N0/N1/A4), "
        f"family MUST be one of [{families}] ({_slot_lines(allowed)}), mechanism "
        "(natural language), paper_refs (ids from the payload papers list), optional "
        "implementation_intent (NL only, no Python). "
        "If you describe spatial attention, intent MUST say HxW / per-pixel mask on feature "
        "maps — never GAP-only channel vectors mislabeled as spatial. "
        "Do not invent paper ids. Do not write Python. invented_operators stay empty. "
        "If papers are empty or irrelevant to RGB-T detection, action=cannot_draft. "
        "KEEP is not a Claim. Literature cannot enter ClaimGate."
    )


def _invent_system(kinds: frozenset[str]) -> str:
    allowed = kinds or frozenset({"fusion"})
    families = ", ".join(sorted(allowed))
    return (
        "Literature did not yield a usable HOW draft. Propose ONE invent-fallback plugin "
        "idea for RGB-T detection. Return JSON with how_id "
        "(use suggested_how_id from the user payload; do not reuse occupied_how_ids), "
        f"family in [{families}] ({_slot_lines(allowed)}), mechanism, implementation_intent "
        "(NL only), reason. No Python. No fake paper citations. "
        "Do not edit vendor D-FINE; only how_plugins/<HOW>/plugin.py after human review. "
        "If the idea is spatial attention, implementation_intent MUST require a true "
        "spatial (N,1,H,W) or (N,C,H,W) mask from thermal/RGB feature maps; "
        "do NOT specify Global Average Pooling channel-only gates as 'spatial'. "
        "This draft requires human review before authoring code. "
        "KEEP is not a Claim. Do not enter ClaimGate."
    )


LIT_SYSTEM = _lit_system(frozenset({"fusion"}))
INVENT_SYSTEM = _invent_system(frozenset({"fusion"}))


def _dump_user(payload: Mapping[str, Any]) -> str:
    return json.dumps(dict(payload), ensure_ascii=False, sort_keys=True)


def _parse(raw: str, schema: Mapping[str, Any]) -> dict[str, Any]:
    data = extract_json_object(raw)
    errors = validate_against_schema(data, dict(schema))
    if errors:
        raise HowPendingError("how draft arm schema: " + "; ".join(errors))
    return data


def _complete(
    *,
    system: str,
    user: Mapping[str, Any],
    schema: Mapping[str, Any],
    provider: Any | None,
    fallback: dict[str, Any],
    live: bool = False,
) -> dict[str, Any]:
    if provider is None:
        return dict(fallback)
    from scientist_lab.llm.config import redact_secrets
    from scientist_lab.llm.gateway import complete_chat

    request = LLMRequest(
        purpose="other",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": _dump_user(user)},
        ],
        response_schema=dict(schema),
        metadata={"how_draft_arm": True, "can_enter_claim_gate": False},
    )
    response = complete_chat(request, provider=provider, live=bool(live))
    raw = redact_secrets(response.content or "")
    return _parse(raw, schema)


def scout_has_papers(scout: Mapping[str, Any] | None) -> bool:
    blob = dict(scout or {})
    if blob.get("fail_closed"):
        return False
    return bool(literature_paper_ids(blob)) and bool(literature_query_id(blob))


def lifecycle_eligible_candidates(store: Mapping[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in store.get("candidates") or []:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("status") or "") != STATUS_PROPOSED:
            continue
        if bool(row.get("requires_human_review")):
            continue
        out.append(dict(row))
    return out


def invent_awaiting_human(store: Mapping[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in store.get("candidates") or []:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("status") or "") != STATUS_PROPOSED:
            continue
        if bool(row.get("requires_human_review")) or str(row.get("source") or "") == "invent_fallback":
            out.append(dict(row))
    return out


_CATALOG_BLOCKED = frozenset({"F0", "F1", "F3", "N0", "N1", "A4"})


def _disk_plugin_how_ids(project_root: Path | str | None = None) -> set[str]:
    """Ids already present under how_plugins/ (tree or campaign-visible code root)."""
    from scientist_lab.core.how_plugin_author import _lab_root, _train_app_root

    root = Path(project_root).resolve() if project_root else _lab_root()
    plugins = _train_app_root(root) / "models" / "how_plugins"
    out: set[str] = set()
    if not plugins.is_dir():
        return out
    for child in plugins.iterdir():
        if not child.is_dir():
            continue
        name = child.name.strip().upper()
        if not name or name.startswith("_"):
            continue
        if (child / "plugin.py").is_file() or any(child.iterdir()):
            out.add(name)
    return out


def occupied_plugin_how_ids(
    store: Mapping[str, Any],
    *,
    project_root: Path | str | None = None,
) -> set[str]:
    """Union of catalog-blocked, campaign candidates/overlay, and on-disk plugins."""
    used = set(_CATALOG_BLOCKED)
    used |= {
        str(row.get("how_id") or "").strip().upper()
        for row in (store.get("candidates") or [])
        if isinstance(row, Mapping)
    }
    used |= {str(k).upper() for k in (store.get("registered_overlay") or {})}
    used |= _disk_plugin_how_ids(project_root)
    used.discard("")
    return used


def _next_plugin_how_id(
    store: Mapping[str, Any],
    *,
    project_root: Path | str | None = None,
) -> str:
    used = occupied_plugin_how_ids(store, project_root=project_root)
    for idx in range(1, 64):
        token = f"P{idx}"
        if token not in used:
            return token
    return "P99"


def _lit_fallback(scout: Mapping[str, Any], how_id: str) -> dict[str, Any]:
    papers = sorted(literature_paper_ids(scout))
    if not papers:
        return {
            "action": "cannot_draft",
            "reason": "deterministic: no admissible papers",
            "how_candidates": [],
        }
    title = ""
    why = ""
    for row in list(scout.get("papers") or []):
        if not isinstance(row, Mapping):
            continue
        if str(row.get("paper_id") or "") == papers[0]:
            title = str(row.get("title") or "")
            why = str(row.get("why_relevant") or "")
            break
    mech = (
        f"Paper {papers[0]} ({title[:120]}): adapt RGB-T fusion cue "
        f"into a FeatureFusion plugin. Clue: {why[:240] or 'cross-modal gating'}."
    )
    return {
        "action": "draft_from_literature",
        "reason": "deterministic literature draft",
        "how_candidates": [
            {
                "how_id": how_id,
                "family": "fusion",
                "mechanism": mech[:800],
                "implementation_intent": (
                    "Author a FeatureFusion plugin that blends RGB and thermal "
                    "features with a gated residual path suggested by the paper clue."
                ),
                "paper_refs": papers[:2],
                "fusion_method": f"plugin:{how_id.lower()}",
            }
        ],
    }


def _invent_fallback(how_id: str, *, why: str) -> dict[str, Any]:
    return {
        "how_id": how_id,
        "family": "fusion",
        "mechanism": (
            "Invent-fallback: confidence-gated RGB-T residual blend inside FeatureFusion; "
            "thermal path only when RGB lowlight confidence is weak."
        ),
        "implementation_intent": (
            "Author a FeatureFusion overlay plugin using a small gated residual mix of "
            "RGB and thermal channels; keep the detector backbone unchanged."
        ),
        "reason": why,
        "fusion_method": f"plugin:{how_id.lower()}",
    }


def ingest_invent_fallback_candidate(
    path: Path | str,
    row: Mapping[str, Any],
    *,
    round_id: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    """Queue invent-fallback draft. Requires human review; no auto-author."""
    how_id = str(row.get("how_id") or "").strip().upper()
    if not how_id:
        raise HowPendingError("invent fallback missing how_id")
    if how_id in {"F0", "F1", "F3", "N0", "N1", "A4"}:
        raise HowPendingError("invent fallback must use a non-catalog HOW id")
    family = str(row.get("family") or "fusion").strip().lower() or "fusion"
    if family not in {"fusion", "neck", "training", "backbone_wrap"}:
        family = "fusion"
    mechanism = str(row.get("mechanism") or "").strip()
    intent = str(row.get("implementation_intent") or "").strip() or None
    if not mechanism:
        raise HowPendingError("invent fallback needs mechanism")
    hay = mechanism + " " + (intent or "")
    if _CODE_HINT.search(hay):
        raise HowPendingError("invent fallback must not include Python")
    fusion = str(row.get("fusion_method") or "").strip() or None
    neck = str(row.get("neck_type") or "").strip() or "standard"
    wrap = str(row.get("backbone_wrap_method") or "").strip() or None
    if family == "neck":
        fusion = fusion or "none"
        if not neck.lower().startswith("plugin:"):
            neck = f"plugin:{how_id}"
    elif family == "backbone_wrap":
        fusion = fusion or "none"
        wrap = wrap or f"plugin:{how_id}"
    else:
        fusion = fusion or f"plugin:{how_id.lower()}"
    rid = str(round_id or "round")
    note = str(reason or row.get("reason") or "literature draft arm could not ground a paper")
    draft: dict[str, Any] = {
        "candidate_id": f"howc_invent_{how_id}_{rid}",
        "how_id": how_id,
        "family": family,
        "mechanism": mechanism[:1200],
        "implementation_intent": intent,
        "status": STATUS_PROPOSED,
        "fusion_method": fusion,
        "neck_type": neck,
        "backbone_wrap_method": wrap,
        "map_to_existing": None,
        # Fusion/neck/backbone_wrap invent drafts author HOW plugins; training needs Adapter work.
        "needs_adapter_work": family_needs_adapter_work(family),
        "plugin_kind": family if family in {"fusion", "neck", "backbone_wrap"} else "fusion",
        "literature_query_id": "invent_fallback",
        "paper_refs": [],
        "invented_operators": [],
        "source": "invent_fallback",
        "requires_human_review": True,
        "draft_origin": "invent",
        "round_id": rid,
        "human_decision": None,
        "human_actor": None,
        "decided_at": None,
        "decision_note": note[:800],
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "can_enter_claim_gate": False,
    }
    validate_named("how_candidate", draft)
    store = load_store(path)
    existing = {str(item.get("candidate_id")): dict(item) for item in store["candidates"]}
    existing[str(draft["candidate_id"])] = draft
    store["candidates"] = list(existing.values())
    store["draft_arm_last"] = {
        "action": "invent_awaiting_human",
        "how_id": how_id,
        "candidate_id": draft["candidate_id"],
        "reason": note[:400],
        "can_enter_claim_gate": False,
    }
    return save_store(path, store)


def release_invent_for_lifecycle(
    path: Path | str,
    candidate_id: str,
    *,
    confirm_human_gate: bool,
    actor: str = "human",
    note: str | None = None,
) -> dict[str, Any]:
    """Human clears invent draft for lifecycle authoring. Does not register overlay."""
    if not confirm_human_gate:
        raise HowPendingError("releasing invent draft requires confirm_human_gate")
    store = load_store(path)
    found = None
    for row in store["candidates"]:
        if str(row.get("candidate_id")) == candidate_id:
            found = row
            break
    if found is None:
        raise HowPendingError(f"unknown HOW candidate: {candidate_id}")
    if str(found.get("status") or "") != STATUS_PROPOSED:
        raise HowPendingError("only proposed invent drafts can be released")
    found["requires_human_review"] = False
    found["human_actor"] = actor
    found["human_decision"] = "release_for_author"
    found["decision_note"] = str(note or "human released invent draft for lifecycle author")
    found["decided_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    validate_named("how_candidate", found)
    return save_store(path, store)


def run_how_draft_arm(
    pending_path: Path | str,
    *,
    confirm_human_gate: bool,
    live: bool = False,
    provider: Any | None = None,
    round_id: str | None = None,
    allow_invent_fallback: bool = True,
    project_root: Path | str | None = None,
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fill proposed HOW drafts: literature first, invent→human if needed."""
    if not confirm_human_gate:
        raise HowPendingError("HOW draft arm requires the campaign Human Gate")
    dest = Path(pending_path)
    store = load_store(dest)
    kinds = protocol_invent_kinds(protocol)
    ticks = int(store.get("draft_arm_ticks") or 0)
    if ticks >= MAX_DRAFT_ARM_TICKS:
        return {
            "progressed": False,
            "ok": True,
            "action": "cap",
            "reason": "how draft arm tick cap",
            "gpu": False,
            "can_enter_claim_gate": False,
        }
    if lifecycle_eligible_candidates(store):
        return {
            "progressed": False,
            "ok": True,
            "action": "idle",
            "reason": "lifecycle-eligible HOW draft already present",
            "gpu": False,
            "can_enter_claim_gate": False,
        }
    waiting = invent_awaiting_human(store)
    if waiting:
        row = waiting[0]
        return {
            "progressed": True,
            "ok": True,
            "action": "invent_awaiting_human",
            "reason": "invent-fallback HOW draft waiting for human review",
            "candidate_id": row.get("candidate_id"),
            "how_id": row.get("how_id"),
            "gpu": False,
            "can_enter_claim_gate": False,
            "requires_human_review": True,
        }

    scout = dict(store.get("scout") or {})
    rid = str(round_id or f"draft_{ticks + 1}")
    store["draft_arm_ticks"] = ticks + 1
    save_store(dest, store)
    how_id = _next_plugin_how_id(store, project_root=project_root)
    occupied = sorted(occupied_plugin_how_ids(store, project_root=project_root))
    # Stage B: invent-first (still human-gated). Literature can fill later ticks.
    invent_first = bool(store.get("llm_may_invent_how")) and allow_invent_fallback

    if scout_has_papers(scout) and not invent_first:
        lit = _complete(
            system=_lit_system(kinds),
            user={
                "literature_query_id": literature_query_id(scout),
                "papers": list(scout.get("papers") or [])[:8],
                "research_question": scout.get("research_question"),
                "suggested_how_id": how_id,
                "occupied_how_ids": occupied,
                "keep_is_not_claim": True,
                "can_enter_claim_gate": False,
                "allowed_plugin_kinds": sorted(kinds),
            },
            schema=LIT_SCHEMA,
            provider=provider,
            fallback=_lit_fallback(scout, how_id),
            live=live,
        )
        action = str(lit.get("action") or "").strip().lower()
        rows = [dict(x) for x in (lit.get("how_candidates") or []) if isinstance(x, Mapping)]
        if action == "draft_from_literature" and rows:
            try:
                for row in rows:
                    row.setdefault("literature_query_id", literature_query_id(scout))
                    row["source"] = "draft_arm"
                    row["draft_origin"] = "literature"
                    row["requires_human_review"] = False
                ingest_llm_candidates(dest, rows, literature=scout, round_id=rid)
                # stamp draft_arm metadata after normalize (source forced to llm there)
                store = load_store(dest)
                for cand in store["candidates"]:
                    if str(cand.get("status")) != STATUS_PROPOSED:
                        continue
                    if str(cand.get("round_id") or "") != rid:
                        continue
                    cand["source"] = "draft_arm"
                    cand["draft_origin"] = "literature"
                    cand["requires_human_review"] = False
                    validate_named("how_candidate", cand)
                store["draft_arm_last"] = {
                    "action": "draft_from_literature",
                    "count": len(rows),
                    "reason": lit.get("reason"),
                    "can_enter_claim_gate": False,
                }
                save_store(dest, store)
                return {
                    "progressed": True,
                    "ok": True,
                    "action": "draft_from_literature",
                    "reason": lit.get("reason") or "drafted HOW from literature",
                    "count": len(rows),
                    "gpu": False,
                    "can_enter_claim_gate": False,
                }
            except Exception as exc:  # noqa: BLE001 — fall through to invent
                lit_fail_reason = f"literature ingest failed: {exc}"
        else:
            lit_fail_reason = str(lit.get("reason") or "model could not draft from literature")
    else:
        lit_fail_reason = (
            "stage_B invent-first: propose plugin HOW before literature draft"
            if invent_first
            else "no scout papers for grounded HOW draft"
        )

    if not allow_invent_fallback:
        return {
            "progressed": False,
            "ok": True,
            "action": "cannot_draft",
            "reason": lit_fail_reason,
            "gpu": False,
            "can_enter_claim_gate": False,
        }

    invent = _complete(
        system=_invent_system(kinds),
        user={
            "why_literature_failed": lit_fail_reason,
            "suggested_how_id": how_id,
            "occupied_how_ids": occupied,
            "scout_summary": summarize_scout(scout) if scout else None,
            "keep_is_not_claim": True,
            "requires_human_review": True,
            "can_enter_claim_gate": False,
            "allowed_plugin_kinds": sorted(kinds),
        },
        schema=INVENT_SCHEMA,
        provider=provider,
        fallback=_invent_fallback(how_id, why=lit_fail_reason),
        live=live,
    )
    hid = str(invent.get("how_id") or how_id).strip().upper()
    # Never collide with catalog / on-disk / already-queued plugin ids.
    occupied_now = occupied_plugin_how_ids(store, project_root=project_root)
    if (not _PLUGIN_HOW.match(hid)) or hid in occupied_now:
        hid = how_id
    invent["how_id"] = hid
    family = str(invent.get("family") or "fusion").strip().lower()
    if family not in kinds:
        family = sorted(kinds)[0] if kinds else "fusion"
    invent["family"] = family
    try:
        ingest_invent_fallback_candidate(
            dest,
            invent,
            round_id=rid,
            reason=str(invent.get("reason") or lit_fail_reason),
        )
    except HowPendingError as exc:
        # Model often slips words like "from/class" into NL; do not fail the campaign.
        safe = _invent_fallback(hid, why=f"{lit_fail_reason}; invent sanitized after: {exc}")
        safe["how_id"] = hid
        ingest_invent_fallback_candidate(
            dest,
            safe,
            round_id=rid,
            reason=str(safe.get("reason") or lit_fail_reason),
        )
        invent = safe
    return {
        "progressed": True,
        "ok": True,
        "action": "invent_awaiting_human",
        "reason": str(invent.get("reason") or lit_fail_reason),
        "how_id": hid,
        "gpu": False,
        "can_enter_claim_gate": False,
        "requires_human_review": True,
    }
