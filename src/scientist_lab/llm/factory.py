"""Provider factory with network gates (v1.4.2)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from scientist_lab.llm.errors import (
    RealProviderNotEnabledError,
    UnsupportedProviderError,
)
from scientist_lab.llm.fake_provider import FakeProvider
from scientist_lab.llm.http_transport import HttpTransport
from scientist_lab.llm.openai_compatible_provider import OpenAICompatibleProvider
from scientist_lab.llm.openai_config import OpenAICompatibleConfig
from scientist_lab.llm.provider import LLMProvider
from scientist_lab.llm.replay_provider import ReplayProvider
from scientist_lab.llm.repository import LLMCallRepository


def _env_truthy(name: str, environ: dict[str, str] | None = None) -> bool:
    env = environ if environ is not None else dict(os.environ)
    return str(env.get(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def create_llm_provider(
    provider: str = "mock",
    *,
    allow_network: bool = False,
    environ: dict[str, str] | None = None,
    openai_config: OpenAICompatibleConfig | None = None,
    transport: HttpTransport | None = None,
    audit_root: Path | str | None = None,
    budget: Any = None,
    concurrency: Any = None,
    retry_policy: Any = None,
    **_kwargs: Any,
) -> LLMProvider:
    """Create an LLMProvider.

    Real openai-compatible instances require ``allow_network=True`` (or an
    explicit offline ``transport`` such as MockTransport). Environment
    ``LLM_ALLOW_NETWORK`` alone is not enough for CLI; callers must pass
    allow_network explicitly for real traffic.
    """
    name = (provider or "mock").strip().lower()
    aliases = {
        "real": "openai-compatible",
        "openai": "openai-compatible",
        "openai_compatible": "openai-compatible",
    }
    name = aliases.get(name, name)

    if name in {"mock", "fake"}:
        return FakeProvider()

    if name == "replay":
        if audit_root is None:
            raise UnsupportedProviderError(
                "replay provider requires audit_root with prior call records"
            )
        return ReplayProvider(LLMCallRepository(Path(audit_root)))

    if name == "openai-compatible":
        env_allows = _env_truthy("LLM_ALLOW_NETWORK", environ)
        kwargs = {
            "budget": budget,
            "concurrency": concurrency,
            "retry_policy": retry_policy,
        }
        # Offline MockTransport path: transport provided without network flag.
        if transport is not None and not allow_network and not env_allows:
            if openai_config is None:
                raise RealProviderNotEnabledError(
                    "openai-compatible with MockTransport still needs openai_config"
                )
            cfg = openai_config.model_copy(update={"allow_network": False})
            return OpenAICompatibleProvider(cfg, transport=transport, **kwargs)

        if not allow_network and not env_allows:
            raise RealProviderNotEnabledError(
                "real provider blocked: pass allow_network=True and set "
                "LLM_ALLOW_NETWORK=true (triple gate with CLI --provider real)"
            )
        if openai_config is None:
            raise RealProviderNotEnabledError(
                "openai-compatible requires OpenAICompatibleConfig"
            )
        if not openai_config.allow_network and transport is None:
            raise RealProviderNotEnabledError(
                "OpenAICompatibleConfig.allow_network must be True for live HTTP"
            )
        cfg = openai_config
        if allow_network or env_allows:
            cfg = openai_config.model_copy(update={"allow_network": True})
        return OpenAICompatibleProvider(cfg, transport=transport, **kwargs)

    raise UnsupportedProviderError(f"unsupported LLM provider: {provider!r}")
