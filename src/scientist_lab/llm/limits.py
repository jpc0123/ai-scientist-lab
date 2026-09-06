"""Token / cost / latency limits for LLMProvider calls (v1.3.9).

Offline-first: costs use configurable synthetic rates. Real cloud billing is
deferred until a Real provider exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from scientist_lab.llm.models import LLMRequest, LLMResponse, TokenUsage
from scientist_lab.llm.provider import LLMProvider


class ProviderLimitExceeded(RuntimeError):
    """Raised when a call would violate configured provider limits."""

    def __init__(self, reason: str, *, limit: str, observed: Any = None) -> None:
        self.reason = reason
        self.limit = limit
        self.observed = observed
        super().__init__(reason)


class ProviderLimits(BaseModel):
    """Hard caps applied by LimitingProvider (session-scoped unless noted)."""

    max_calls: int | None = None
    max_prompt_tokens: int | None = None
    max_completion_tokens: int | None = None
    max_total_tokens: int | None = None
    max_latency_ms: float | None = None
    max_cost_usd: float | None = None

    # Synthetic offline rates (USD per 1K tokens). Fake/Replay metering only.
    cost_per_1k_prompt_tokens: float = 0.001
    cost_per_1k_completion_tokens: float = 0.002


def estimate_cost_usd(usage: TokenUsage, limits: ProviderLimits) -> float:
    prompt = float(usage.prompt_tokens or 0) / 1000.0
    completion = float(usage.completion_tokens or 0) / 1000.0
    return (
        prompt * float(limits.cost_per_1k_prompt_tokens)
        + completion * float(limits.cost_per_1k_completion_tokens)
    )


@dataclass
class UsageLedger:
    """In-memory session totals for limiting + reporting."""

    call_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    last_latency_ms: float = 0.0
    latencies_ms: list[float] = field(default_factory=list)

    def record(self, response: LLMResponse, *, cost_usd: float) -> None:
        usage = response.usage or TokenUsage()
        self.call_count += 1
        self.prompt_tokens += int(usage.prompt_tokens or 0)
        self.completion_tokens += int(usage.completion_tokens or 0)
        self.total_tokens += int(usage.total_tokens or 0)
        self.cost_usd += float(cost_usd)
        self.last_latency_ms = float(response.latency_ms or 0.0)
        self.latencies_ms.append(self.last_latency_ms)

    def as_dict(self) -> dict[str, Any]:
        mean_latency = (
            sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else 0.0
        )
        return {
            "call_count": self.call_count,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 8),
            "last_latency_ms": self.last_latency_ms,
            "mean_latency_ms": mean_latency,
        }


class LimitingProvider:
    """Enforce ProviderLimits around an inner LLMProvider."""

    def __init__(
        self,
        inner: LLMProvider,
        limits: ProviderLimits | None = None,
        *,
        ledger: UsageLedger | None = None,
    ) -> None:
        self._inner = inner
        self.limits = limits or ProviderLimits()
        self.ledger = ledger or UsageLedger()

    @property
    def name(self) -> str:
        return f"limit:{self._inner.name}"

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def inner(self) -> LLMProvider:
        return self._inner

    def _check_before_call(self) -> None:
        lim = self.limits
        if lim.max_calls is not None and self.ledger.call_count >= lim.max_calls:
            raise ProviderLimitExceeded(
                f"max_calls exceeded: {self.ledger.call_count} >= {lim.max_calls}",
                limit="max_calls",
                observed=self.ledger.call_count,
            )
        if (
            lim.max_total_tokens is not None
            and self.ledger.total_tokens >= lim.max_total_tokens
        ):
            raise ProviderLimitExceeded(
                f"max_total_tokens already exhausted: {self.ledger.total_tokens}",
                limit="max_total_tokens",
                observed=self.ledger.total_tokens,
            )
        if lim.max_cost_usd is not None and self.ledger.cost_usd >= lim.max_cost_usd:
            raise ProviderLimitExceeded(
                f"max_cost_usd already exhausted: {self.ledger.cost_usd}",
                limit="max_cost_usd",
                observed=self.ledger.cost_usd,
            )

    def _check_after_call(self, response: LLMResponse, *, cost_usd: float) -> None:
        lim = self.limits
        usage = response.usage or TokenUsage()
        if (
            lim.max_latency_ms is not None
            and float(response.latency_ms or 0.0) > float(lim.max_latency_ms)
        ):
            raise ProviderLimitExceeded(
                f"max_latency_ms exceeded: {response.latency_ms} > {lim.max_latency_ms}",
                limit="max_latency_ms",
                observed=response.latency_ms,
            )
        if (
            lim.max_prompt_tokens is not None
            and self.ledger.prompt_tokens > lim.max_prompt_tokens
        ):
            raise ProviderLimitExceeded(
                f"max_prompt_tokens exceeded: {self.ledger.prompt_tokens}",
                limit="max_prompt_tokens",
                observed=self.ledger.prompt_tokens,
            )
        if (
            lim.max_completion_tokens is not None
            and self.ledger.completion_tokens > lim.max_completion_tokens
        ):
            raise ProviderLimitExceeded(
                f"max_completion_tokens exceeded: {self.ledger.completion_tokens}",
                limit="max_completion_tokens",
                observed=self.ledger.completion_tokens,
            )
        if (
            lim.max_total_tokens is not None
            and self.ledger.total_tokens > lim.max_total_tokens
        ):
            raise ProviderLimitExceeded(
                f"max_total_tokens exceeded: {self.ledger.total_tokens}",
                limit="max_total_tokens",
                observed=self.ledger.total_tokens,
            )
        if lim.max_cost_usd is not None and self.ledger.cost_usd > lim.max_cost_usd:
            raise ProviderLimitExceeded(
                f"max_cost_usd exceeded: {self.ledger.cost_usd}",
                limit="max_cost_usd",
                observed=self.ledger.cost_usd,
            )
        # Silence unused when all limits None.
        _ = usage
        _ = cost_usd

    def complete(self, request: LLMRequest) -> LLMResponse:
        self._check_before_call()
        response = self._inner.complete(request)
        cost = estimate_cost_usd(response.usage or TokenUsage(), self.limits)
        self.ledger.record(response, cost_usd=cost)
        self._check_after_call(response, cost_usd=cost)
        return response
