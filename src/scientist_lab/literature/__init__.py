"""Controlled literature retrieval tool (v2.6-P1). Not a fifth Agent."""

from scientist_lab.literature.factory import resolve_literature_provider
from scientist_lab.literature.fake_provider import FakeLiteratureProvider
from scientist_lab.literature.gate import gate_paper, looks_like_literature_evidence
from scientist_lab.literature.models import PaperRecord
from scientist_lab.literature.retriever import LiteratureRetriever

__all__ = [
    "FakeLiteratureProvider",
    "LiteratureRetriever",
    "PaperRecord",
    "gate_paper",
    "looks_like_literature_evidence",
    "resolve_literature_provider",
]
