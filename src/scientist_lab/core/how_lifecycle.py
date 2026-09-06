"""Campaign LLM HOW lifecycle: add? write plugin? accept code?

Human freezes Protocol. This is not a fifth Agent.
Does not edit Protocol, merge the main tree, start GPU, or enter ClaimGate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from scientist_lab.adapters.dfine.how_catalog import candidate_plugin_kind
from scientist_lab.core.how_pending import (
    STATUS_PENDING_ADAPTER,
    STATUS_PROPOSED,
    HowPendingError,
    decide_candidate,
    load_store,
    overlay_from_store,
    save_store,
)
from scientist_lab.core.how_plugin_author import HowPluginAuthorError, author_how_patch
from scientist_lab.core.how_plugin_worker import FakePluginWorker
from scientist_lab.llm.errors import MissingAPIKeyError, RealProviderNotEnabledError
from scientist_lab.llm.models import LLMRequest
from scientist_lab.llm.schema_parser import extract_json_object, validate_against_schema

MAX_LIFECYCLE_TICKS = 24
MAX_ATTEMPTS_PER_CANDIDATE = 2

ADD_SCHEMA = {
    "type": "object",
    "required": ["action", "reason"],
    "properties": {
        "action": {"type": "string", "enum": ["author", "reject"]},
        "reason": {"type": "string"},
    },
}
ACCEPT_SCHEMA = {
    "type": "object",
    "required": ["decision", "reason"],
    "properties": {
        "decision": {"type": "string", "enum": ["register", "reject"]},
        "reason": {"type": "string"},
    },
}

ADD_SYSTEM = (
    "You decide whether Scientist Lab should author a HOW plugin "
    "for this campaign. Protocol is frozen; you cannot change it. "
    "Return JSON {\"action\":\"author\"|\"reject\",\"reason\":\"...\"}. "
    "author = write how_plugins/<HOW>/plugin.py in the Diff sandbox. "
    "reject = drop the draft. Do not start GPU. Do not write ClaimGate. "
    "KEEP is not a Claim. "
    "IMPORTANT: smoke_ok=false on a proposed/pending draft means it has NOT "
    "been authored yet — that is the normal reason to choose author, NOT reject. "
    "plugin_kind=fusion → FeatureFusion. plugin_kind=neck → HybridEncoder-slot neck. "
    "plugin_kind=backbone_wrap → backbone wrap (new R0). "
    "Only reject if the mechanism is out of the allowed invent_policy.kinds "
    "(e.g. preprocessing-only CLAHE, new detector Adapter family, vendor D-FINE rewrite)."
)
ACCEPT_SYSTEM = (
    "You decide whether a smoked HOW plugin may enter this campaign's "
    "registered_overlay. Protocol is frozen. Return JSON "
    "{\"decision\":\"register\"|\"reject\",\"reason\":\"...\"}. "
    "register = Planner may select this HOW next (KEEP exploration only). "
    "reject = drop it. Do not merge the main tree. Do not start GPU. "
    "KEEP is not a Claim. "
    "IMPORTANT: can_enter_claim_gate=false is NORMAL and REQUIRED — it means "
    "the plugin must NOT enter ClaimGate. It is NOT a reason to reject overlay. "
    "If smoke_ok=true and the plugin matches its PLUGIN_KIND contract, choose register."
)


def _dump_user(payload: Mapping[str, Any]) -> str:
    return json.dumps(dict(payload), ensure_ascii=False, sort_keys=True)


def _parse(raw: str, schema: Mapping[str, Any]) -> dict[str, Any]:
    data = extract_json_object(raw)
    errors = validate_against_schema(data, dict(schema))
    if errors:
        raise HowPendingError("how lifecycle schema: " + "; ".join(errors))
    return data


def _complete(
    *,
    system: str,
    user: Mapping[str, Any],
    schema: Mapping[str, Any],
    provider: Any | None,
    live: bool,
    fallback: dict[str, Any],
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
        metadata={"how_lifecycle": True, "can_enter_claim_gate": False},
    )
    response = complete_chat(request, provider=provider, live=False)
    raw = redact_secrets(response.content or "")
    return _parse(raw, schema)


def _pick_candidate(store: Mapping[str, Any]) -> dict[str, Any] | None:
    """Prefer proposed drafts; also author approved_pending_adapter without smoke."""
    pending_adapter: dict[str, Any] | None = None
    for row in store.get("candidates") or []:
        if not isinstance(row, Mapping):
            continue
        # Invent-fallback drafts stay with the human until released.
        if bool(row.get("requires_human_review")):
            continue
        attempts = int(row.get("lifecycle_attempts") or 0)
        if attempts >= MAX_ATTEMPTS_PER_CANDIDATE:
            continue
        status = str(row.get("status") or "")
        if status == STATUS_PROPOSED:
            return dict(row)
        if (
            pending_adapter is None
            and status == STATUS_PENDING_ADAPTER
            and not bool(row.get("smoke_ok"))
            and bool(row.get("needs_adapter_work", True))
        ):
            pending_adapter = dict(row)
    return pending_adapter


def _bump_attempt(path: Path | str, candidate_id: str) -> None:
    store = load_store(path)
    for row in store["candidates"]:
        if str(row.get("candidate_id")) == candidate_id:
            row["lifecycle_attempts"] = int(row.get("lifecycle_attempts") or 0) + 1
            break
    save_store(path, store)


def run_how_lifecycle_tick(
    pending_path: Path | str,
    *,
    project_root: Path | str,
    confirm_human_gate: bool,
    live: bool = False,
    provider: Any | None = None,
    plugin_worker: Any | None = None,
    sandbox_root: Path | str | None = None,
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One add → author → accept step. Overlay only. No GPU."""
    if not confirm_human_gate:
        raise HowPendingError("HOW lifecycle requires the campaign Human Gate")
    dest = Path(pending_path)
    store = load_store(dest)
    ticks = int(store.get("lifecycle_ticks") or 0)
    if ticks >= MAX_LIFECYCLE_TICKS:
        return {
            "progressed": False,
            "ok": True,
            "action": "cap",
            "reason": "how lifecycle tick cap",
            "gpu": False,
            "registered": False,
            "can_enter_claim_gate": False,
        }
    found = _pick_candidate(store)
    if found is None:
        return {
            "progressed": False,
            "ok": True,
            "action": "idle",
            "reason": "no proposed HOW draft",
            "gpu": False,
            "registered": False,
            "can_enter_claim_gate": False,
        }
    cid = str(found["candidate_id"])
    how_id = str(found.get("how_id") or "")
    _bump_attempt(dest, cid)
    store = load_store(dest)
    store["lifecycle_ticks"] = ticks + 1
    save_store(dest, store)

    add = _complete(
        system=ADD_SYSTEM,
        user={
            "how_id": how_id,
            "candidate_id": cid,
            "mechanism": found.get("mechanism") or found.get("implementation_intent"),
            "plugin_kind": candidate_plugin_kind(found),
            "family": found.get("family"),
            "smoke_ok": bool(found.get("smoke_ok")),
            "keep_is_not_claim": True,
        },
        schema=ADD_SCHEMA,
        provider=provider,
        live=live,
        fallback={"action": "author", "reason": "deterministic: author proposed fusion draft"},
    )
    action = str(add.get("action") or "").strip().lower()
    # Unsmoked drafts must be authored first. Models often misread smoke_ok=false
    # as "failed smoke" and reject — that blocks Stage A plugin materialization.
    if action != "author" and not bool(found.get("smoke_ok")):
        action = "author"
        add = {
            "action": "author",
            "reason": (
                "override: draft has no smoke yet; author FeatureFusion plugin "
                f"(model said reject: {add.get('reason')})"
            ),
        }
    if action != "author":
        decide_candidate(
            dest,
            cid,
            decision="reject",
            actor="llm",
            note=str(add.get("reason") or "LLM rejected adding this HOW"),
            confirm_human_gate=True,
        )
        return {
            "progressed": True,
            "ok": True,
            "action": "reject_add",
            "candidate_id": cid,
            "how_id": how_id,
            "reason": add.get("reason"),
            "gpu": False,
            "registered": False,
            "can_enter_claim_gate": False,
        }

    refreshed = load_store(dest)
    row = next(
        (dict(item) for item in refreshed["candidates"] if str(item.get("candidate_id")) == cid),
        found,
    )
    if not bool(row.get("smoke_ok")):
        worker: Any | None = None
        worker_name: str | None = None
        author_live = False
        author_provider = None
        kind = candidate_plugin_kind(row)
        if isinstance(plugin_worker, str):
            token = plugin_worker.strip().lower() or "llm"
            if token == "harness":
                worker_name = "harness"
            elif token == "llm":
                if live and provider is not None:
                    worker_name = "llm"
                    author_provider = provider
                    author_live = True
                elif not live:
                    worker = FakePluginWorker(
                        project_root=project_root, plugin_kind=kind
                    )
                else:
                    worker_name = "llm"
            else:
                worker_name = token
        elif plugin_worker is not None:
            worker = plugin_worker
        elif not live:
            worker = FakePluginWorker(project_root=project_root, plugin_kind=kind)
        else:
            worker_name = "llm"
            author_provider = provider
            author_live = bool(live and provider is not None)
        try:
            author_how_patch(
                dest,
                cid,
                project_root=project_root,
                provider=author_provider,
                worker=worker,
                worker_name=worker_name if worker is None else None,
                live=author_live,
                sandbox_root=sandbox_root,
                protocol=protocol,
            )
        except (
            HowPendingError,
            HowPluginAuthorError,
            MissingAPIKeyError,
            RealProviderNotEnabledError,
        ) as exc:
            return {
                "progressed": True,
                "ok": False,
                "action": "author_failed",
                "candidate_id": cid,
                "how_id": how_id,
                "reason": str(exc),
                "gpu": False,
                "registered": False,
                "can_enter_claim_gate": False,
            }

    smoked = load_store(dest)
    current = next(
        (dict(item) for item in smoked["candidates"] if str(item.get("candidate_id")) == cid),
        {},
    )
    accept = _complete(
        system=ACCEPT_SYSTEM,
        user={
            "how_id": how_id,
            "candidate_id": cid,
            "smoke_ok": bool(current.get("smoke_ok")),
            "smoke_detail": current.get("smoke_detail"),
            "plugin_relpath": current.get("plugin_relpath"),
            "keep_is_not_claim": True,
            # Explicit so the model does not treat the store flag as a reject cue.
            "can_enter_claim_gate": False,
            "claim_gate_note": (
                "false means overlay KEEP only; never ClaimGate. Still register if smoke_ok."
            ),
        },
        schema=ACCEPT_SCHEMA,
        provider=provider,
        live=live,
        fallback=(
            {"decision": "register", "reason": "deterministic: smoke_ok overlay"}
            if bool(current.get("smoke_ok"))
            else {"decision": "reject", "reason": "deterministic: smoke missing"}
        ),
    )
    decision = str(accept.get("decision") or "").strip().lower()
    # Models often reject smoked plugins citing can_enter_claim_gate=false —
    # that flag is intentional (KEEP≠Claim) and must not block overlay.
    if decision != "register" and bool(current.get("smoke_ok")):
        decision = "register"
        accept = {
            "decision": "register",
            "reason": (
                "override: smoke_ok plugin enters registered_overlay for KEEP "
                f"(model said reject: {accept.get('reason')})"
            ),
        }
    decide_candidate(
        dest,
        cid,
        decision=decision if decision in {"register", "reject"} else "reject",
        actor="llm",
        note=str(accept.get("reason") or ""),
        confirm_human_gate=True,
    )
    overlay = overlay_from_store(load_store(dest))
    registered = how_id.upper() in overlay and decision == "register"
    return {
        "progressed": True,
        "ok": True,
        "action": "register" if registered else "reject_code",
        "candidate_id": cid,
        "how_id": how_id,
        "reason": accept.get("reason"),
        "gpu": False,
        "registered": registered,
        "can_enter_claim_gate": False,
        "overlay_ids": sorted(overlay),
    }
