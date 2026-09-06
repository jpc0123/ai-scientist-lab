"""Semantic Scholar Graph API provider. Live only. Key stays in env."""

from __future__ import annotations

import json
import time
from typing import Any
from urllib.parse import quote, urlencode

from scientist_lab.literature.errors import LiteratureProviderError
from scientist_lab.literature.models import PaperRecord, s2_paper_id, utc_now
from scientist_lab.llm.errors import LLMConnectionError, LLMRateLimitError, LLMTimeoutError
from scientist_lab.llm.http_transport import HttpTransport, HttpxTransport, map_http_error

S2_DEFAULT_BASE = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = (
    "paperId,title,year,authors,abstract,externalIds,venue,citationCount,url"
)
S2_CITED_FIELDS = ",".join(f"citedPaper.{part}" for part in S2_FIELDS.split(","))
S2_CITING_FIELDS = ",".join(f"citingPaper.{part}" for part in S2_FIELDS.split(","))


def _bare_id(paper_id: str) -> str:
    value = str(paper_id or "").strip()
    if value.lower().startswith("s2:"):
        return value.split(":", 1)[1]
    return value


class SemanticScholarProvider:
    def __init__(
        self,
        *,
        api_key: str,
        transport: HttpTransport | None = None,
        base_url: str = S2_DEFAULT_BASE,
    ) -> None:
        key = str(api_key or "").strip()
        if not key:
            raise ValueError("SemanticScholarProvider requires api_key")
        self._api_key = key
        self._transport = transport or HttpxTransport()
        self._base = str(base_url or S2_DEFAULT_BASE).rstrip("/")

    @property
    def name(self) -> str:
        return "semantic_scholar"

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self._api_key, "User-Agent": "scientist-lab-literature/v26"}

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        query = urlencode({k: v for k, v in params.items() if v is not None and v != ""})
        url = f"{self._base}{path}"
        if query:
            url = f"{url}?{query}"
        last_error: Exception | None = None
        # Scout used to fire 3 searches + hop; free/low-tier S2 keys 429 easily.
        # Also retry ConnectTimeout / transport blips (Live A round 5).
        # Free-tier 429 needs longer gaps than 2s/5s.
        delays = (0.0, 5.0, 15.0, 30.0)
        for attempt, delay in enumerate(delays):
            if delay:
                time.sleep(delay)
            try:
                response = self._transport.request(
                    "GET",
                    url,
                    headers=self._headers(),
                    json_body=None,
                    timeout_seconds=30.0,
                    connect_timeout_seconds=15.0,
                )
            except (LLMTimeoutError, LLMConnectionError) as exc:
                last_error = LiteratureProviderError(str(exc))
                if attempt + 1 < len(delays):
                    continue
                raise last_error from exc
            mapped = map_http_error(response.status_code, response.body)
            if mapped is None:
                try:
                    payload = json.loads(response.body or "{}")
                except json.JSONDecodeError as exc:
                    raise LiteratureProviderError("Semantic Scholar returned non-JSON") from exc
                if not isinstance(payload, dict):
                    raise LiteratureProviderError("Semantic Scholar returned a non-object")
                return payload
            last_error = mapped
            if isinstance(mapped, LLMRateLimitError) and attempt + 1 < len(delays):
                continue
            raise LiteratureProviderError(str(mapped)) from mapped
        raise LiteratureProviderError(str(last_error or "Semantic Scholar request failed"))

    def _to_record(self, raw: dict[str, Any], query: str) -> PaperRecord:
        return PaperRecord.from_mapping(
            raw,
            retrieval_query=query,
            source="semantic_scholar",
            retrieved_at=utc_now(),
        )

    def search(
        self,
        query: str,
        *,
        year_from: int | None = 2022,
        limit: int = 20,
    ) -> list[PaperRecord]:
        params: dict[str, Any] = {
            "query": query,
            "limit": max(1, min(int(limit), 100)),
            "fields": S2_FIELDS,
        }
        if year_from is not None:
            params["year"] = f"{int(year_from)}-"
        payload = self._get("/paper/search", params)
        rows = payload.get("data") or []
        return [self._to_record(dict(row), query) for row in rows if isinstance(row, dict)]

    def get_paper(self, paper_id: str, *, retrieval_query: str = "") -> PaperRecord:
        bare = _bare_id(paper_id)
        payload = self._get(f"/paper/{quote(bare, safe='')}", {"fields": S2_FIELDS})
        return self._to_record(payload, retrieval_query or s2_paper_id(bare))

    def references(
        self,
        paper_id: str,
        *,
        limit: int = 20,
        retrieval_query: str = "",
    ) -> list[PaperRecord]:
        bare = _bare_id(paper_id)
        payload = self._get(
            f"/paper/{quote(bare, safe='')}/references",
            {
                "limit": max(1, min(int(limit), 100)),
                "fields": S2_CITED_FIELDS,
            },
        )
        query = retrieval_query or f"references of {s2_paper_id(bare)}"
        out: list[PaperRecord] = []
        for row in payload.get("data") or []:
            if not isinstance(row, dict):
                continue
            cited = row.get("citedPaper")
            if isinstance(cited, dict) and cited.get("paperId"):
                out.append(self._to_record(cited, query))
        return out

    def citations(
        self,
        paper_id: str,
        *,
        limit: int = 20,
        retrieval_query: str = "",
    ) -> list[PaperRecord]:
        bare = _bare_id(paper_id)
        payload = self._get(
            f"/paper/{quote(bare, safe='')}/citations",
            {
                "limit": max(1, min(int(limit), 100)),
                "fields": S2_CITING_FIELDS,
            },
        )
        query = retrieval_query or f"citations of {s2_paper_id(bare)}"
        out: list[PaperRecord] = []
        for row in payload.get("data") or []:
            if not isinstance(row, dict):
                continue
            citing = row.get("citingPaper")
            if isinstance(citing, dict) and citing.get("paperId"):
                out.append(self._to_record(citing, query))
        return out
