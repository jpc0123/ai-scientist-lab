"""Formal DFINE path Claim Gate (v2.3.5).

Opens the *formal acceptance path* only for non-stand-in Vendor evidence that
meets matched-protocol thresholds. Never treats Fast Eval as proof of
formal DFINE superiority.
"""

from __future__ import annotations

from typing import Any

from scientist_lab.evidence.models import EvidenceRecord, ScientificClaim


TRIAD_ROLE_MARKERS = {
    "rgb": ("rgb", "formal_cuda_node_001", "formal_node_001", "_node_001"),
    "thermal": ("thermal", "formal_cuda_node_002", "formal_node_002", "_node_002"),
    "fusion": ("fusion", "early_concat", "formal_cuda_node_003", "formal_node_003", "_node_003"),
}


def _tag_backend(record: EvidenceRecord) -> str:
    summary = dict(record.metric_summary or {})
    tag = str(summary.get("standin_or_vendor") or "").strip().lower()
    if tag in {"vendor", "standin"}:
        return tag
    impl = str(summary.get("baseline_implementation") or "").lower()
    if "standin" in impl or "stand_in" in impl or "torch_mini" in impl:
        return "standin"
    if "vendored" in impl or impl.startswith("dfine"):
        return "vendor"
    blob = " ".join(
        [
            *record.limitations,
            str(summary.get("dfine_backend_requested") or ""),
            str(summary.get("notes") or ""),
        ]
    ).lower()
    if "stand-in" in blob or "stand_in" in blob or "standin" in blob:
        return "standin"
    return "unknown"


def _is_fast_eval_record(record: EvidenceRecord) -> bool:
    if record.claim_level == "exploratory_comparison":
        return True
    summary = dict(record.metric_summary or {})
    scope = str(summary.get("evaluation_scope") or "")
    if scope.startswith("fast_eval"):
        return True
    blob = " ".join(record.limitations).lower()
    return "fast eval" in blob


def _detect_roles(records: list[EvidenceRecord]) -> set[str]:
    found: set[str] = set()
    for record in records:
        summary = dict(record.metric_summary or {})
        role = str(summary.get("triad_role") or summary.get("role") or "").strip().lower()
        if role in TRIAD_ROLE_MARKERS:
            found.add(role)
            continue
        hay = " ".join(
            [
                role,
                str(summary.get("input_mode") or ""),
                str(summary.get("fusion_method") or ""),
                *[str(x) for x in (record.source_node_ids or [])],
            ]
        ).lower()
        for candidate, markers in TRIAD_ROLE_MARKERS.items():
            if any(marker in hay for marker in markers):
                found.add(candidate)
    return found


def assess_formal_dfine_path_gate(
    records: list[EvidenceRecord],
) -> dict[str, Any]:
    """Evaluate whether the Vendor formal acceptance path may open."""
    vendor_records = [r for r in records if r.valid and _tag_backend(r) == "vendor"]
    standin_records = [r for r in records if r.valid and _tag_backend(r) == "standin"]
    roles = _detect_roles(vendor_records)
    has_protocol = any(bool(r.protocol_id) for r in vendor_records)
    paired_vendor = [
        r
        for r in vendor_records
        if r.evidence_type == "paired_comparison"
    ]
    fast_eval_only = bool(vendor_records) and all(
        _is_fast_eval_record(r) for r in vendor_records
    )

    thresholds = {
        "non_standin_vendor": len(vendor_records) >= 1,
        "no_standin_contamination": len(standin_records) == 0,
        "has_protocol": has_protocol,
        "vendor_count_ge_3_or_triad_or_paired": (
            len(vendor_records) >= 3
            or roles >= {"rgb", "thermal", "fusion"}
            or len(paired_vendor) >= 1
        ),
    }
    blocking: list[str] = []
    if not thresholds["non_standin_vendor"]:
        blocking.append("No Vendor (non-stand-in) DFINE evidence.")
    if not thresholds["no_standin_contamination"]:
        blocking.append("Stand-in evidence is present; formal path stays closed.")
    if not thresholds["has_protocol"]:
        blocking.append("Vendor evidence lacks ExperimentProtocol linkage.")
    if not thresholds["vendor_count_ge_3_or_triad_or_paired"]:
        blocking.append(
            "Need Vendor triad (rgb/thermal/fusion), >=3 Vendor records, "
            "or Vendor paired_comparison."
        )

    standin_blocked = len(standin_records) > 0 and len(vendor_records) == 0
    path_open = all(thresholds.values()) and not standin_blocked

    return {
        "formal_path_open": path_open,
        "standin_blocked": bool(standin_records) and not path_open,
        "vendor_evidence_count": len(vendor_records),
        "standin_evidence_count": len(standin_records),
        "triad_roles": sorted(roles),
        "has_triad": roles >= {"rgb", "thermal", "fusion"},
        "has_protocol": has_protocol,
        "has_vendor_paired_comparison": len(paired_vendor) >= 1,
        "fast_eval_only": fast_eval_only,
        # Superiority remains ineligible under Fast Eval even when path is open.
        "formal_superiority_eligible": bool(path_open and not fast_eval_only),
        "thresholds": thresholds,
        "blocking_reasons": blocking,
        "note": (
            "formal_path_open means Vendor evidence may proceed toward formal "
            "acceptance workflows; it does not assert DFINE superiority."
        ),
    }


def evaluate_formal_dfine_path_claim(
    *,
    project_id: str,
    records: list[EvidenceRecord],
) -> ScientificClaim:
    """Claim: Vendor DFINE formal acceptance path is open (not superiority)."""
    gate = assess_formal_dfine_path_gate(records)
    ids = [r.evidence_id for r in records if r.valid and _tag_backend(r) == "vendor"]

    if gate["standin_evidence_count"] and not gate["vendor_evidence_count"]:
        return ScientificClaim(
            claim_id="claim_formal_dfine_path",
            project_id=project_id,
            claim_text=(
                "Vendor DFINE formal acceptance path is open under matched protocol."
            ),
            claim_type="formal_path_gate",
            required_evidence_types=["single_execution", "paired_comparison"],
            support_status="blocked",
            supporting_evidence_ids=ids,
            limitations=list(gate["blocking_reasons"])
            + ["Stand-in evidence cannot open the formal DFINE path."],
            reason="Stand-in evidence blocks the formal path gate.",
        )

    if gate["formal_path_open"]:
        limitations = [
            "Path-open does not equal formal DFINE superiority.",
        ]
        if gate["fast_eval_only"]:
            limitations.append(
                "Evidence is Fast Eval / exploratory; continue to full formal budget "
                "before superiority claims."
            )
        return ScientificClaim(
            claim_id="claim_formal_dfine_path",
            project_id=project_id,
            claim_text=(
                "Vendor DFINE formal acceptance path is open under matched protocol."
            ),
            claim_type="formal_path_gate",
            required_evidence_types=["single_execution", "paired_comparison"],
            support_status="supported",
            supporting_evidence_ids=ids,
            limitations=limitations,
            reason=None,
        )

    return ScientificClaim(
        claim_id="claim_formal_dfine_path",
        project_id=project_id,
        claim_text=(
            "Vendor DFINE formal acceptance path is open under matched protocol."
        ),
        claim_type="formal_path_gate",
        required_evidence_types=["single_execution", "paired_comparison"],
        support_status="unsupported",
        supporting_evidence_ids=ids,
        limitations=list(gate["blocking_reasons"]),
        reason="Formal path thresholds are not met.",
    )


def apply_formal_dfine_superiority_status(
    *,
    project_id: str,
    records: list[EvidenceRecord],
    supporting_evidence_ids: list[str],
) -> ScientificClaim:
    """Superiority claim: never supported from Fast Eval-only Vendor evidence."""
    gate = assess_formal_dfine_path_gate(records)

    if gate["standin_evidence_count"] and not gate["vendor_evidence_count"]:
        return ScientificClaim(
            claim_id="claim_formal_dfine",
            project_id=project_id,
            claim_text="The method demonstrates formal DFINE superiority.",
            claim_type="formal_implementation",
            required_evidence_types=["paired_comparison"],
            support_status="blocked",
            supporting_evidence_ids=supporting_evidence_ids,
            limitations=["Stand-in evidence cannot support formal DFINE claims."],
            reason="Stand-in / non-vendor DFINE evidence cannot support this claim.",
        )

    if not gate["formal_path_open"]:
        return ScientificClaim(
            claim_id="claim_formal_dfine",
            project_id=project_id,
            claim_text="The method demonstrates formal DFINE superiority.",
            claim_type="formal_implementation",
            required_evidence_types=["paired_comparison"],
            support_status="unsupported",
            supporting_evidence_ids=supporting_evidence_ids,
            limitations=list(gate["blocking_reasons"])
            + ["Formal DFINE claim still requires stronger acceptance evidence."],
            reason="Formal path thresholds are not met.",
        )

    if gate["fast_eval_only"] or not gate["formal_superiority_eligible"]:
        return ScientificClaim(
            claim_id="claim_formal_dfine",
            project_id=project_id,
            claim_text="The method demonstrates formal DFINE superiority.",
            claim_type="formal_implementation",
            required_evidence_types=["paired_comparison"],
            support_status="unsupported",
            supporting_evidence_ids=supporting_evidence_ids,
            limitations=[
                "Formal path is open, but Fast Eval / exploratory evidence "
                "cannot establish formal DFINE superiority.",
                "Escalate to full_train matched budget + ablation before support.",
            ],
            reason=(
                "Formal path open under Fast Eval; superiority remains unsupported."
            ),
        )

    # Non-Fast-Eval Vendor path: allow partial support only (still not SOTA).
    return ScientificClaim(
        claim_id="claim_formal_dfine",
        project_id=project_id,
        claim_text="The method demonstrates formal DFINE superiority.",
        claim_type="formal_implementation",
        required_evidence_types=["paired_comparison"],
        support_status="partially_supported",
        supporting_evidence_ids=supporting_evidence_ids,
        limitations=[
            "Partial support only; ablation / multi-seed formal acceptance still required.",
            "Does not authorize SOTA claims.",
        ],
        reason="Vendor non-Fast-Eval evidence opens partial formal support.",
    )
