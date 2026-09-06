"""v2.3.5 unit tests — Formal DFINE path Claim Gate."""

from __future__ import annotations

from scientist_lab.evidence.claim_matrix import (
    build_claim_support_matrix,
    evaluate_formal_dfine_claim,
    evaluate_formal_dfine_path_claim,
)
from scientist_lab.evidence.formal_dfine_gate import assess_formal_dfine_path_gate
from scientist_lab.evidence.models import EvidenceRecord
from scientist_lab.tasks.rgbt_detection.vendor_audit import BASELINE_IMPLEMENTATION


def _vendor_record(
    *,
    evidence_id: str,
    role: str,
    node_id: str,
    protocol_id: str = "protocol_rgbt_cuda_001",
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        project_id="project_gate_v235",
        evidence_type="single_execution",
        source_node_ids=[node_id],
        source_execution_ids=[f"exec_{evidence_id}"],
        protocol_id=protocol_id,
        claim_level="exploratory_comparison",
        evidence_strength="weak",
        limitations=[
            "Fast Eval subset / budget only.",
            "Vendor DFINE Fast Eval only; not a formal benchmark acceptance.",
        ],
        metric_summary={
            "standin_or_vendor": "vendor",
            "baseline_implementation": BASELINE_IMPLEMENTATION,
            "triad_role": role,
            "evaluation_scope": "fast_eval_subset",
            "primary_metric": "mAP50_95",
            "metrics": {"mAP50_95": 0.1},
        },
    )


def test_standin_blocks_path_and_superiority():
    record = EvidenceRecord(
        evidence_id="ev_standin",
        project_id="p",
        evidence_type="single_execution",
        limitations=["Stand-in implementation was used."],
        metric_summary={"standin_or_vendor": "standin"},
    )
    gate = assess_formal_dfine_path_gate([record])
    assert gate["formal_path_open"] is False
    path = evaluate_formal_dfine_path_claim(project_id="p", records=[record])
    assert path.support_status == "blocked"
    sup = evaluate_formal_dfine_claim(project_id="p", records=[record])
    assert sup.support_status == "blocked"


def test_single_vendor_not_enough_for_path():
    record = _vendor_record(
        evidence_id="ev1",
        role="rgb",
        node_id="rgbt_formal_cuda_node_001",
    )
    gate = assess_formal_dfine_path_gate([record])
    assert gate["formal_path_open"] is False
    path = evaluate_formal_dfine_path_claim(project_id="p", records=[record])
    assert path.support_status == "unsupported"


def test_vendor_triad_opens_path_but_not_superiority():
    records = [
        _vendor_record(
            evidence_id="ev_rgb",
            role="rgb",
            node_id="rgbt_formal_cuda_node_001",
        ),
        _vendor_record(
            evidence_id="ev_th",
            role="thermal",
            node_id="rgbt_formal_cuda_node_002",
        ),
        _vendor_record(
            evidence_id="ev_fu",
            role="fusion",
            node_id="rgbt_formal_cuda_node_003",
        ),
    ]
    gate = assess_formal_dfine_path_gate(records)
    assert gate["formal_path_open"] is True
    assert gate["has_triad"] is True
    assert gate["fast_eval_only"] is True
    assert gate["formal_superiority_eligible"] is False

    path = evaluate_formal_dfine_path_claim(
        project_id="project_gate_v235", records=records
    )
    assert path.support_status == "supported"
    assert path.claim_id == "claim_formal_dfine_path"

    sup = evaluate_formal_dfine_claim(
        project_id="project_gate_v235", records=records
    )
    assert sup.support_status == "unsupported"
    assert "Fast Eval" in (sup.reason or "")

    matrix = build_claim_support_matrix(
        project_id="project_gate_v235", records=records
    )
    by_id = {c.claim_id: c for c in matrix.claims}
    assert by_id["claim_formal_dfine_path"].support_status == "supported"
    assert by_id["claim_formal_dfine"].support_status == "unsupported"
    assert by_id["claim_sota"].support_status == "blocked"
