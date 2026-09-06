"""Literature Evidence Gate. Hard rules, not an Agent, not ClaimGate.

Literature answers: what have others done?
ExperimentEvidence answers: what did our GPU run measure?
"""

from __future__ import annotations

from typing import Any, Mapping

from scientist_lab.literature.models import PaperRecord

ALLOWED_METADATA = "paper_exists"
ALLOWED_ABSTRACT = "abstract_summary"
BLOCKED_FULL_TEXT = "no_full_text_body_claim"
BLOCKED_CONSENSUS = "single_paper_is_not_field_consensus"
BLOCKED_CLAIM_GATE = "literature_cannot_substitute_experiment_evidence"


def gate_paper(paper: PaperRecord | Mapping[str, Any]) -> dict[str, Any]:
    """Admit a PaperRecord into Planner context, or explain why not."""
    record = paper if isinstance(paper, PaperRecord) else PaperRecord.from_mapping(
        paper,
        retrieval_query=str(paper.get("retrieval_query") or "unknown"),
        source=str(paper.get("source") or "fake"),
        retrieved_at=paper.get("retrieved_at"),
    )
    missing: list[str] = []
    for field in ("paper_id", "title", "source", "retrieval_query", "retrieved_at"):
        if not str(getattr(record, field, "") or "").strip():
            missing.append(field)
    if record.year is None:
        missing.append("year")
    if not (record.abstract or "").strip():
        missing.append("abstract")
    if not record.identifier():
        missing.append("url_or_doi_or_arxiv")

    completeness = "abstract" if (record.abstract or "").strip() else "metadata"
    # P1 never fetches full text.
    if completeness == "abstract" and not missing:
        completeness = "abstract"

    planner_admissible = not missing
    allowed = [ALLOWED_METADATA]
    blocked = [BLOCKED_FULL_TEXT, BLOCKED_CONSENSUS, BLOCKED_CLAIM_GATE]
    if (record.abstract or "").strip():
        allowed.append(ALLOWED_ABSTRACT)

    reject = None
    if missing:
        reject = "missing required fields: " + ", ".join(missing)

    packet = {
        "schema_version": "1.0.0",
        "evidence_kind": "literature",
        "paper": record.to_dict(),
        "completeness": completeness,
        "planner_admissible": planner_admissible,
        "can_enter_claim_gate": False,
        "allowed_statements": allowed,
        "blocked_statements": blocked,
        "reject_reason": reject,
    }
    return packet


def looks_like_literature_evidence(evidence: Mapping[str, Any] | None) -> bool:
    """True only when the blob *is* literature, not GPU evidence with literature provenance.

    ExperimentEvidence may later carry ``literature_query_id`` as round provenance.
    That must not make ClaimGate treat a GPU result as a paper.
    Do not read generic ``kind``: inspector packs use kind=fixture|llm_loop|formal_c1.
    """
    blob = dict(evidence or {})
    kind = str(blob.get("evidence_kind") or "").strip().lower()
    if kind in {"literature", "literature_evidence"}:
        return True
    if blob.get("paper") and blob.get("can_enter_claim_gate") is False:
        return True
    if blob.get("paper_records"):
        return True
    # Retriever search packet accidentally fed to ClaimGate.
    if blob.get("papers") and (
        blob.get("planner_admissible") is not None or blob.get("claim_gate_note")
    ):
        return True
    return False
