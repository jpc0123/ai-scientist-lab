"""Real-only provider / profile gates for research loops (v2.1.3)."""

from __future__ import annotations

from typing import Any

from scientist_lab.agents.provider_bridge import normalize_provider_mode
from scientist_lab.llm_eval.profile_gate import (
    LLMProfileNotQualifiedError,
    assert_profile_qualified_for_planning,
)
from scientist_lab.research_loop.errors import (
    RealLoopBudgetMissingError,
    RealLoopFallbackDetectedError,
    RealLoopProfileNotQualifiedError,
    RealLoopProviderMismatchError,
    RealLoopValidationError,
)


REAL_PROVIDER_MODES = frozenset({"real", "openai-compatible"})
FORBIDDEN_IN_REAL_ONLY = frozenset({"mock", "fake", "replay"})


def unwrap_provider_label(mode: str | None) -> str | None:
    """Strip AuditingProvider / LimitingProvider wrappers before mode checks."""
    if mode is None:
        return None
    text = str(mode).strip().lower()
    changed = True
    while changed and text:
        changed = False
        for prefix in ("audit:", "limit:"):
            if text.startswith(prefix):
                text = text[len(prefix) :].strip()
                changed = True
    return text or None


def canonical_real_provider(mode: str | None) -> str | None:
    raw = unwrap_provider_label(mode)
    if raw is None:
        return None
    return normalize_provider_mode(raw)


def provider_audit(
    *,
    requested_provider: str,
    actual_provider: str | None,
    fallback_allowed: bool = False,
    fallback_used: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    requested = canonical_real_provider(requested_provider) or normalize_provider_mode(
        requested_provider
    )
    # Keep raw actual for audit trail, but normalize comparison fields.
    actual_raw = str(actual_provider).strip() if actual_provider else None
    actual = canonical_real_provider(actual_provider)
    payload = {
        "requested_provider": requested,
        "actual_provider": actual_raw or actual,
        "actual_provider_canonical": actual,
        "fallback_allowed": bool(fallback_allowed),
        "fallback_used": bool(fallback_used),
    }
    if extra:
        payload.update(extra)
    return payload


def assert_real_only_provider(mode: str, *, real_only: bool = True) -> str:
    resolved = canonical_real_provider(mode) or normalize_provider_mode(mode)
    if not real_only:
        return resolved
    if resolved in FORBIDDEN_IN_REAL_ONLY:
        raise RealLoopProviderMismatchError(
            f"real_only loop rejects provider={resolved!r}; "
            "use openai-compatible / real with explicit network gates"
        )
    if resolved not in REAL_PROVIDER_MODES:
        raise RealLoopProviderMismatchError(
            f"unsupported provider for real loop: {resolved!r}"
        )
    return resolved


def assert_no_fallback(
    *,
    requested_provider: str,
    actual_provider: str | None,
    fallback_allowed: bool,
    fallback_used: bool,
) -> None:
    if fallback_allowed:
        raise RealLoopValidationError(
            "fallback_allowed must be false in real research loops"
        )
    if fallback_used:
        raise RealLoopFallbackDetectedError(
            "silent provider fallback detected (fallback_used=true)"
        )
    requested = canonical_real_provider(requested_provider) or normalize_provider_mode(
        requested_provider
    )
    actual = canonical_real_provider(actual_provider)
    if actual is None:
        return
    # Treat openai-compatible and real as equivalent real backends.
    if requested in REAL_PROVIDER_MODES and actual in REAL_PROVIDER_MODES:
        return
    if actual != requested:
        raise RealLoopFallbackDetectedError(
            f"provider mismatch: requested={requested!r} actual={actual!r}"
        )
    if actual in FORBIDDEN_IN_REAL_ONLY:
        raise RealLoopFallbackDetectedError(
            f"actual provider fell back to offline mode: {actual!r}"
        )


def check_profile_for_real_loop(
    *,
    profile: Any | None,
    evaluation: dict[str, Any] | None,
    suite_version: str | None = "eval_suite_v1",
    require_quality_gate: bool = True,
    budget_configured: bool | None = None,
    network_enabled: bool | None = None,
) -> dict[str, Any]:
    """Offline profile qualification report (default: no network)."""
    checks: list[dict[str, Any]] = []
    issues: list[str] = []

    def _add(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})
        if not ok:
            issues.append(f"{name}: {detail}")

    if profile is None:
        _add("profile_configured", False, "missing")
        return {
            "overall": "failed",
            "checks": checks,
            "issues": issues,
            "network_used": False,
        }

    profile_id = getattr(profile, "profile_id", None) or profile.get("profile_id")
    provider = canonical_real_provider(
        getattr(profile, "provider", None) or profile.get("provider")
    ) or normalize_provider_mode(
        getattr(profile, "provider", None) or profile.get("provider")
    )
    enabled = bool(getattr(profile, "enabled", True) if not isinstance(profile, dict) else profile.get("enabled", True))
    planner_prompt = getattr(profile, "planner_prompt_version", None) or (
        profile.get("planner_prompt_version") if isinstance(profile, dict) else None
    )
    critic_prompt = getattr(profile, "critic_prompt_version", None) or (
        profile.get("critic_prompt_version") if isinstance(profile, dict) else None
    )

    _add("profile_configured", True, str(profile_id))
    _add("profile_enabled", enabled, f"enabled={enabled}")
    _add(
        "provider_type",
        provider in REAL_PROVIDER_MODES,
        provider,
    )
    _add(
        "planner_prompt",
        bool(str(planner_prompt or "").strip()),
        str(planner_prompt or "missing"),
    )
    _add(
        "critic_prompt",
        bool(str(critic_prompt or "").strip()),
        str(critic_prompt or "missing"),
    )

    quality_detail = "not_checked"
    quality_ok = True
    if require_quality_gate:
        try:
            audit = assert_profile_qualified_for_planning(
                profile=profile,
                evaluation=evaluation,
                suite_version=suite_version,
                require_quality_gate=True,
                allow_unqualified=False,
            )
            quality_ok = True
            quality_detail = str(audit.get("gate_status") or "passed")
        except LLMProfileNotQualifiedError as exc:
            quality_ok = False
            quality_detail = str(exc)
    _add("quality_gate", quality_ok, quality_detail)

    if budget_configured is not None:
        _add("budget", bool(budget_configured), "configured" if budget_configured else "missing")
    else:
        _add("budget", True, "not_checked")

    if network_enabled is None:
        _add("network_gate", True, "disabled")
    else:
        _add(
            "network_gate",
            True,
            "enabled" if network_enabled else "disabled",
        )

    overall = "ok" if not issues else "failed"
    return {
        "overall": overall,
        "profile_id": profile_id,
        "provider": provider,
        "checks": checks,
        "issues": issues,
        "network_used": False,
        "require_quality_gate": require_quality_gate,
    }


def require_profile_qualified_for_real_loop(
    *,
    profile: Any | None,
    evaluation: dict[str, Any] | None,
    suite_version: str | None = "eval_suite_v1",
    budget_configured: bool = True,
) -> dict[str, Any]:
    report = check_profile_for_real_loop(
        profile=profile,
        evaluation=evaluation,
        suite_version=suite_version,
        require_quality_gate=True,
        budget_configured=budget_configured,
    )
    if report["overall"] != "ok":
        # Map to typed errors for callers.
        if any(c["name"] == "provider_type" and not c["ok"] for c in report["checks"]):
            raise RealLoopProviderMismatchError(
                "; ".join(report["issues"]) or "provider mismatch"
            )
        if any(c["name"] == "budget" and not c["ok"] for c in report["checks"]):
            raise RealLoopBudgetMissingError("real-loop budget is not configured")
        if any(c["name"] == "quality_gate" and not c["ok"] for c in report["checks"]):
            raise RealLoopProfileNotQualifiedError(
                "; ".join(report["issues"]) or "profile not qualified"
            )
        raise RealLoopProfileNotQualifiedError(
            "; ".join(report["issues"]) or "profile not qualified"
        )
    return report
