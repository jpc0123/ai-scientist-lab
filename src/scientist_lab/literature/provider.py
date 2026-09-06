"""LiteratureProvider protocol. Planner does not care which corpus."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from scientist_lab.literature.models import PaperRecord


@runtime_checkable
class LiteratureProvider(Protocol):
    @property
    def name(self) -> str: ...

    def search(
        self,
        query: str,
        *,
        year_from: int | None = 2022,
        limit: int = 20,
    ) -> list[PaperRecord]: ...

    def get_paper(self, paper_id: str, *, retrieval_query: str = "") -> PaperRecord: ...

    def references(
        self,
        paper_id: str,
        *,
        limit: int = 20,
        retrieval_query: str = "",
    ) -> list[PaperRecord]: ...

    def citations(
        self,
        paper_id: str,
        *,
        limit: int = 20,
        retrieval_query: str = "",
    ) -> list[PaperRecord]: ...
