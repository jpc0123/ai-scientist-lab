"""Real-provider force mode + call budget for restricted Diff (v2.2.4)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from scientist_lab.llm.limits import LimitingProvider, ProviderLimits, UsageLedger
from scientist_lab.llm.provider import LLMProvider
from scientist_lab.research_loop.provider_gate import (
    FORBIDDEN_IN_REAL_ONLY,
    REAL_PROVIDER_MODES,
    assert_real_only_provider,
    unwrap_provider_label,
)


class PatchRealModeError(ValueError):
    """Raised when patch real-mode policy is violated."""


class PatchProviderBudget(BaseModel):
    """Hard caps for RealPatchPlanner provider calls (session or project)."""

    max_calls: int = 5
    max_total_tokens: int = 50_000
    max_cost_usd: float = 1.0
    max_latency_ms: float | None = 120_000.0

    def to_provider_limits(self) -> ProviderLimits:
        return ProviderLimits(
            max_calls=self.max_calls,
            max_total_tokens=self.max_total_tokens,
            max_cost_usd=self.max_cost_usd,
            max_latency_ms=self.max_latency_ms,
        )


DEFAULT_PATCH_PROVIDER_BUDGET = PatchProviderBudget()


@dataclass
class PatchRealModePolicy:
    """Force openai-compatible semantics; never silent mock/fake/replay."""

    force_real_only: bool = True
    allow_explicit_non_real: bool = False
    require_transport_or_network: bool = True
    default_budget: PatchProviderBudget = field(
        default_factory=lambda: PatchProviderBudget()
    )

    def budget(self) -> PatchProviderBudget:
        return self.default_budget or DEFAULT_PATCH_PROVIDER_BUDGET

    def assert_mode(
        self,
        *,
        requested_provider: str,
        real_only: bool,
        allow_network: bool,
        transport: Any | None,
        provider: LLMProvider | None,
    ) -> str:
        if self.force_real_only and not real_only:
            if not self.allow_explicit_non_real:
                raise PatchRealModeError(
                    "patch propose_real requires real_only=True "
                    "(v2.2.4 force mode); refusing non-real session"
                )
        requested = assert_real_only_provider(
            requested_provider, real_only=bool(real_only or self.force_real_only)
        )
        if provider is None and self.require_transport_or_network:
            if not allow_network and transport is None:
                raise PatchRealModeError(
                    "real patch provider requires allow_network=True "
                    "or an explicit offline transport (e.g. MockTransport); "
                    "silent mock fallback is forbidden"
                )
        return requested

    def wrap_with_budget(
        self,
        provider: LLMProvider,
        *,
        budget: PatchProviderBudget | None = None,
        ledger: UsageLedger | None = None,
    ) -> tuple[LLMProvider, UsageLedger, PatchProviderBudget]:
        caps = budget or self.budget()
        usage = ledger or UsageLedger()
        limited = LimitingProvider(
            provider, caps.to_provider_limits(), ledger=usage
        )
        return limited, usage, caps


class PatchBudgetStore:
    """Persist per-project patch-provider usage under outputs/_patch_budgets/."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, project_id: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in project_id)
        return self.root / f"{safe}.json"

    def load(self, project_id: str) -> dict[str, Any]:
        path = self._path(project_id)
        if not path.is_file():
            return {
                "project_id": project_id,
                "call_count": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
            }
        return json.loads(path.read_text(encoding="utf-8"))

    def add_usage(self, project_id: str, ledger: UsageLedger) -> dict[str, Any]:
        current = self.load(project_id)
        current["call_count"] = int(current.get("call_count") or 0) + int(
            ledger.call_count
        )
        current["prompt_tokens"] = int(current.get("prompt_tokens") or 0) + int(
            ledger.prompt_tokens
        )
        current["completion_tokens"] = int(current.get("completion_tokens") or 0) + int(
            ledger.completion_tokens
        )
        current["total_tokens"] = int(current.get("total_tokens") or 0) + int(
            ledger.total_tokens
        )
        current["cost_usd"] = round(
            float(current.get("cost_usd") or 0.0) + float(ledger.cost_usd), 8
        )
        path = self._path(project_id)
        path.write_text(
            json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return current

    def status(
        self,
        project_id: str,
        *,
        budget: PatchProviderBudget | None = None,
    ) -> dict[str, Any]:
        caps = budget or DEFAULT_PATCH_PROVIDER_BUDGET
        used = self.load(project_id)
        remaining_calls = max(0, caps.max_calls - int(used.get("call_count") or 0))
        remaining_tokens = max(
            0, caps.max_total_tokens - int(used.get("total_tokens") or 0)
        )
        remaining_cost = max(
            0.0, caps.max_cost_usd - float(used.get("cost_usd") or 0.0)
        )
        exhausted = (
            remaining_calls <= 0 or remaining_tokens <= 0 or remaining_cost <= 0.0
        )
        return {
            "project_id": project_id,
            "budget": caps.model_dump(),
            "used": used,
            "remaining": {
                "calls": remaining_calls,
                "total_tokens": remaining_tokens,
                "cost_usd": round(remaining_cost, 8),
            },
            "exhausted": exhausted,
            "force_real_only": True,
            "forbidden_providers": sorted(FORBIDDEN_IN_REAL_ONLY),
            "allowed_providers": sorted(REAL_PROVIDER_MODES),
        }


def patch_provider_doctor(
    *,
    requested_provider: str = "openai-compatible",
    allow_network: bool = False,
    has_api_key: bool = False,
    has_base_url: bool = False,
    has_model: bool = False,
    transport_injected: bool = False,
) -> dict[str, Any]:
    """Offline readiness report for real patch provider (no network)."""
    checks: list[dict[str, Any]] = []

    def _add(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})

    try:
        resolved = assert_real_only_provider(requested_provider, real_only=True)
        _add("provider_mode", True, resolved)
    except Exception as exc:  # noqa: BLE001
        resolved = unwrap_provider_label(requested_provider)
        _add("provider_mode", False, str(exc))

    offline_ok = transport_injected and not allow_network
    live_ok = allow_network and has_api_key and has_base_url and has_model
    _add(
        "transport_or_live_gate",
        offline_ok or live_ok,
        "mock_transport"
        if offline_ok
        else ("live_ready" if live_ok else "blocked"),
    )
    _add("api_key", has_api_key or offline_ok, "present" if has_api_key else "missing")
    _add(
        "base_url",
        has_base_url or offline_ok,
        "present" if has_base_url else "missing",
    )
    _add("model", has_model or offline_ok, "present" if has_model else "missing")
    _add("fallback_forbidden", True, "mock/fake/replay rejected in real_only")

    overall = "ok" if all(c["ok"] for c in checks) else "failed"
    return {
        "overall": overall,
        "requested_provider": resolved,
        "checks": checks,
        "network_used": False,
    }
