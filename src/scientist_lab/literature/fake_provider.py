"""Offline literature corpus. No network. Safe default."""

from __future__ import annotations

from scientist_lab.literature.models import PaperRecord, utc_now

_CORPUS: list[dict] = [
    {
        "paper_id": "S2:lowlight-rgbt-001",
        "title": "Low-Light RGB-T Small Object Detection via Early Fusion",
        "year": 2024,
        "authors": ["Fake Author A"],
        "abstract": (
            "RGB cues degrade under low illumination. Thermal imagery can provide "
            "complementary information. Early concatenation is a common fusion baseline "
            "for small-object RGB-T detection."
        ),
        "doi": "10.0000/fake.lowlight.early",
        "venue": "Fake CV Workshop",
        "citation_count": 12,
        "url": "https://example.invalid/papers/lowlight-rgbt-001",
        "arxiv_id": None,
    },
    {
        "paper_id": "S2:rgbt-weight-002",
        "title": "Adaptive Modality Weighting for Visible-Thermal Detection",
        "year": 2023,
        "authors": ["Fake Author B"],
        "abstract": (
            "Fixed early fusion can inject thermal noise. Adaptive modality weighting "
            "is reported to reduce redundancy when RGB quality varies."
        ),
        "doi": "10.0000/fake.weighting",
        "venue": "Fake Sensors Journal",
        "citation_count": 37,
        "url": "https://example.invalid/papers/rgbt-weight-002",
    },
    {
        "paper_id": "S2:small-obj-neck-003",
        "title": "Multi-scale Necks for Tiny Object Detection",
        "year": 2025,
        "authors": ["Fake Author C"],
        "abstract": (
            "Small-object AP often fails because of weak multi-scale representation. "
            "This is not evidence that any specific neck works in a given codebase."
        ),
        "doi": "10.0000/fake.tiny.neck",
        "venue": "Fake Detection Conf",
        "citation_count": 5,
        "url": "https://example.invalid/papers/small-obj-neck-003",
    },
    {
        "paper_id": "S2:brightness-bucket-004",
        "title": "Brightness-bucket error analysis for RGB-T detection on RGBT-Tiny",
        "year": 2024,
        "authors": ["Fake Author E"],
        "abstract": (
            "APS by illumination / brightness buckets on RGBT-Tiny. "
            "Error analysis of RGB-T fusion under low versus mid brightness."
        ),
        "doi": "10.0000/fake.brightness.bucket",
        "venue": "Fake Analysis Workshop",
        "citation_count": 3,
        "url": "https://example.invalid/papers/brightness-bucket-004",
    },
    {
        "paper_id": "S2:dl-survey-999",
        "title": "A Comprehensive Survey of Deep Learning",
        "year": 2023,
        "authors": ["Fake Author D"],
        "abstract": (
            "We review convolutional networks, ImageNet classification, and generic "
            "deep learning practice. This paper is a methods survey, not a detection study."
        ),
        "doi": "10.0000/fake.dl.survey",
        "venue": "Fake Survey Journal",
        "citation_count": 400,
        "url": "https://example.invalid/papers/dl-survey-999",
        "arxiv_id": None,
    },
    {
        "paper_id": "S2:hop-only-005",
        "title": "Brightness-bucket follow-up on RGBT-Tiny APS",
        "year": 2024,
        "authors": ["Fake Author F"],
        "abstract": (
            "Brightness-bucket error analysis and APS on RGBT-Tiny. "
            "Only reachable via references of the seed paper in the fake corpus."
        ),
        "doi": "10.0000/fake.hop.only",
        "venue": "Fake Detection Conf",
        "citation_count": 2,
        "url": "https://example.invalid/papers/hop-only-005",
        "searchable": False,
    },
    {
        "paper_id": "S2:old-brightness-010",
        "title": "Brightness-bucket APS on RGBT-Tiny (2019)",
        "year": 2019,
        "authors": ["Fake Author G"],
        "abstract": "Brightness-bucket error analysis APS RGBT-Tiny from an old venue.",
        "doi": "10.0000/fake.old.brightness",
        "venue": "Fake Old Workshop",
        "citation_count": 88,
        "url": "https://example.invalid/papers/old-brightness-010",
        "searchable": False,
    },
]

class FakeLiteratureProvider:
    """Deterministic in-memory papers for tests and offline CLI."""

    def __init__(self, papers: list[dict] | None = None) -> None:
        self._papers = list(papers or _CORPUS)

    @property
    def name(self) -> str:
        return "fake"

    def _all(self, *, query: str) -> list[PaperRecord]:
        now = utc_now()
        return [
            PaperRecord.from_mapping(row, retrieval_query=query, source="fake", retrieved_at=now)
            for row in self._papers
        ]

    def search(
        self,
        query: str,
        *,
        year_from: int | None = 2022,
        limit: int = 20,
    ) -> list[PaperRecord]:
        tokens = [t for t in str(query).lower().replace("-", " ").split() if t]
        hits: list[PaperRecord] = []
        for row, paper in zip(self._papers, self._all(query=query), strict=True):
            if row.get("searchable") is False:
                continue
            if year_from is not None and paper.year is not None and paper.year < int(year_from):
                continue
            blob = " ".join(
                [
                    paper.title,
                    paper.abstract or "",
                    " ".join(paper.authors),
                    paper.venue or "",
                ]
            ).lower()
            if not tokens or any(tok in blob for tok in tokens):
                hits.append(paper)
        return hits[: max(1, min(int(limit), 100))]

    def get_paper(self, paper_id: str, *, retrieval_query: str = "") -> PaperRecord:
        needle = str(paper_id).strip()
        for paper in self._all(query=retrieval_query or needle):
            if paper.paper_id == needle or paper.paper_id.endswith(needle):
                if retrieval_query:
                    paper.retrieval_query = retrieval_query
                return paper
        raise KeyError(f"paper not found: {needle}")

    def references(
        self,
        paper_id: str,
        *,
        limit: int = 20,
        retrieval_query: str = "",
    ) -> list[PaperRecord]:
        self.get_paper(paper_id)
        others = [
            p
            for p in self._all(query=retrieval_query or f"references of {paper_id}")
            if p.paper_id != paper_id and not p.paper_id.endswith(paper_id)
        ]
        return others[: max(1, min(int(limit), 100))]

    def citations(
        self,
        paper_id: str,
        *,
        limit: int = 20,
        retrieval_query: str = "",
    ) -> list[PaperRecord]:
        return self.references(
            paper_id, limit=limit, retrieval_query=retrieval_query or f"citations of {paper_id}"
        )
