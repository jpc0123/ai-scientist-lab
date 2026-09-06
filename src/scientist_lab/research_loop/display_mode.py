"""Derive REAL / MOCK / REPLAY display mode for Web Console (v2.1.8)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def derive_display_mode(
    session: Mapping[str, Any] | None,
    *,
    rounds: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """Return one of REAL | MOCK | REPLAY for UI badges."""
    data = dict(session or {})
    real_only = bool(data.get("real_only", True))
    fallback_allowed = bool(data.get("fallback_allowed", False))
    fallback_used = bool(data.get("fallback_used", False))

    providers: list[str] = []
    for rnd in rounds or []:
        audit = rnd.get("provider_audit_json") or {}
        if isinstance(audit, str):
            continue
        if not isinstance(audit, Mapping):
            continue
        for key in ("actual_provider", "requested_provider"):
            val = str(audit.get(key) or "").strip().lower()
            if val:
                providers.append(val)
        critic = audit.get("critic")
        if isinstance(critic, Mapping):
            for key in ("actual_provider", "requested_provider"):
                val = str(critic.get(key) or "").strip().lower()
                if val:
                    providers.append(val)

    if any(p == "replay" or p.startswith("replay:") for p in providers):
        return "REPLAY"
    if (
        fallback_used
        or fallback_allowed
        or (not real_only)
        or any(p in {"mock", "fake"} or p.startswith("mock:") for p in providers)
    ):
        return "MOCK"
    return "REAL"


def enrich_session_view(view: dict[str, Any]) -> dict[str, Any]:
    rounds = view.get("rounds") if isinstance(view.get("rounds"), list) else []
    mode = derive_display_mode(view, rounds=rounds)
    view["display_mode"] = mode
    view["mode_badge"] = mode
    return view
