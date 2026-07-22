"""LLM call budget (separate from experiment GPU budget) — v1.4.3."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from scientist_lab.domain.models import new_id
from scientist_lab.llm.errors import LLMBudgetExceededError
from scientist_lab.llm.models import TokenUsage


class ModelPricing(BaseModel):
    provider: str = "openai-compatible"
    model_pattern: str = "*"
    input_cost_per_million_tokens: float | None = None
    output_cost_per_million_tokens: float | None = None
    currency: str = "USD"
    effective_date: str | None = None

    def estimate_cost_usd(self, usage: TokenUsage) -> float | None:
        if (
            self.input_cost_per_million_tokens is None
            and self.output_cost_per_million_tokens is None
        ):
            return None
        inp = float(usage.prompt_tokens or 0) / 1_000_000.0
        out = float(usage.completion_tokens or 0) / 1_000_000.0
        cost = 0.0
        if self.input_cost_per_million_tokens is not None:
            cost += inp * float(self.input_cost_per_million_tokens)
        if self.output_cost_per_million_tokens is not None:
            cost += out * float(self.output_cost_per_million_tokens)
        return cost


class LLMBudget(BaseModel):
    budget_id: str = Field(default_factory=lambda: new_id("llmbudget"))
    project_id: str

    max_requests: int = Field(default=20, ge=0)
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_total_tokens: int | None = None
    max_cost_usd: float | None = None

    used_requests: int = 0
    used_input_tokens: int = 0
    used_output_tokens: int = 0
    used_total_tokens: int = 0
    used_cost_usd: float = 0.0

    enabled: bool = True
    pricing: ModelPricing | None = None

    def remaining_requests(self) -> int:
        return max(0, int(self.max_requests) - int(self.used_requests))

    def check_can_call(
        self,
        *,
        projected_output_tokens: int | None = None,
        projected_cost_usd: float | None = None,
    ) -> None:
        if not self.enabled:
            return
        if self.used_requests >= self.max_requests:
            raise LLMBudgetExceededError(
                f"LLM max_requests exceeded: {self.used_requests}/{self.max_requests}"
            )
        if (
            self.max_total_tokens is not None
            and self.used_total_tokens >= self.max_total_tokens
        ):
            raise LLMBudgetExceededError(
                f"LLM max_total_tokens already exhausted: {self.used_total_tokens}"
            )
        if self.max_cost_usd is not None and self.used_cost_usd >= self.max_cost_usd:
            raise LLMBudgetExceededError(
                f"LLM max_cost_usd already exhausted: {self.used_cost_usd}"
            )
        if (
            projected_output_tokens is not None
            and self.max_output_tokens is not None
            and self.used_output_tokens + int(projected_output_tokens)
            > self.max_output_tokens
        ):
            raise LLMBudgetExceededError(
                "projected output tokens would exceed max_output_tokens"
            )
        if (
            projected_cost_usd is not None
            and self.max_cost_usd is not None
            and self.used_cost_usd + float(projected_cost_usd) > self.max_cost_usd
        ):
            raise LLMBudgetExceededError(
                "projected cost would exceed max_cost_usd"
            )

    def record(
        self,
        usage: TokenUsage,
        *,
        cost_usd: float | None = None,
        usage_unknown: bool = False,
    ) -> dict[str, Any]:
        """Update counters after a call. Unknown usage still counts as 1 request."""
        self.used_requests += 1
        if not usage_unknown:
            self.used_input_tokens += int(usage.prompt_tokens or 0)
            self.used_output_tokens += int(usage.completion_tokens or 0)
            self.used_total_tokens += int(usage.total_tokens or 0)
        estimated = cost_usd
        if estimated is None and self.pricing is not None and not usage_unknown:
            estimated = self.pricing.estimate_cost_usd(usage)
        if estimated is not None:
            self.used_cost_usd += float(estimated)
        return {
            "used_requests": self.used_requests,
            "used_input_tokens": self.used_input_tokens,
            "used_output_tokens": self.used_output_tokens,
            "used_total_tokens": self.used_total_tokens,
            "used_cost_usd": self.used_cost_usd,
            "usage_unknown": usage_unknown,
            "cost_status": "unknown" if estimated is None else "estimated",
        }

    def snapshot(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
