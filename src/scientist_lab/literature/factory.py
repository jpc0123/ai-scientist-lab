"""Resolve LiteratureProvider. Keys stay in the environment."""

from __future__ import annotations

import os
from typing import Mapping

from scientist_lab.llm.errors import (
    MissingAPIKeyError,
    RealProviderNotEnabledError,
    UnsupportedProviderError,
)
from scientist_lab.llm.http_transport import HttpTransport
from scientist_lab.literature.fake_provider import FakeLiteratureProvider
from scientist_lab.literature.semantic_scholar import S2_DEFAULT_BASE, SemanticScholarProvider

ENV_S2_KEY = "SEMANTIC_SCHOLAR_API_KEY"
FUTURE_PROVIDERS = frozenset({"arxiv", "crossref", "openalex"})
OFFLINE_PROVIDERS = frozenset({"fake", "mock"})


def resolve_literature_provider(
    *,
    live: bool = False,
    provider: str | None = None,
    environ: Mapping[str, str] | None = None,
    transport: HttpTransport | None = None,
    api_key: str | None = None,
) -> Any:
    env = dict(environ if environ is not None else os.environ)
    name = str(provider or ("semantic_scholar" if live else "fake")).strip().lower()
    if name in {"mock"}:
        name = "fake"
    if name in OFFLINE_PROVIDERS:
        if live:
            raise RealProviderNotEnabledError(
                "fake literature provider cannot be used with --live"
            )
        return FakeLiteratureProvider()
    if name in FUTURE_PROVIDERS:
        raise UnsupportedProviderError(
            f"literature provider {name!r} is reserved for a later v2.6 Provider; "
            "P1 implements semantic_scholar only"
        )
    if name != "semantic_scholar":
        raise UnsupportedProviderError(f"unknown literature provider: {name}")
    if not live:
        raise RealProviderNotEnabledError(
            "semantic_scholar requires live=True; without --live the fake corpus is used"
        )
    key = str(api_key if api_key is not None else env.get(ENV_S2_KEY) or "").strip()
    if not key:
        raise MissingAPIKeyError(
            "SEMANTIC_SCHOLAR_API_KEY is required for live literature search; "
            "key stays in env"
        )
    return SemanticScholarProvider(
        api_key=key,
        transport=transport,
        base_url=str(env.get("SEMANTIC_SCHOLAR_BASE_URL") or S2_DEFAULT_BASE),
    )
