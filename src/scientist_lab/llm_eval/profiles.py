"""LLM model / prompt profiles (v1.5.4)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LLMModelProfile(BaseModel):
    profile_id: str
    provider: str = "mock"
    model: str = "mock-planner-v1"
    api_mode: str = "chat_completions"

    planner_prompt_version: str = "mock_v1"
    critic_prompt_version: str = "mock_v1"

    temperature: float = 0.0
    max_output_tokens: int = 2048

    enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    def safe_dict(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")
        # Never embed secrets in profile metadata dumps.
        meta = dict(data.get("metadata") or {})
        for key in list(meta.keys()):
            low = str(key).lower()
            if any(tok in low for tok in ("api_key", "authorization", "token", "password")):
                meta[key] = "[REDACTED]"
        data["metadata"] = meta
        return data


def default_mock_profile() -> LLMModelProfile:
    return LLMModelProfile(
        profile_id="mock_default",
        provider="mock",
        model="mock-planner-v1",
        api_mode="offline",
        planner_prompt_version="mock_v1",
        critic_prompt_version="mock_v1",
        enabled=True,
        metadata={"builtin": True},
    )
