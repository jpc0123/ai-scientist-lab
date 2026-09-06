from __future__ import annotations

from scientist_lab.services.evidence_view import (
    CLAIM_STATUS_LABELS,
    EVIDENCE_TYPE_LABELS,
    enrich_claim,
    enrich_claim_matrix,
    enrich_evidence,
    enrich_report,
)


def test_evidence_type_labels_cover_core_types():
    for key in (
        "single_execution",
        "repeated_experiment",
        "paired_comparison",
        "ablation",
        "resource_comparison",
        "failure_analysis",
    ):
        assert key in EVIDENCE_TYPE_LABELS


def test_claim_status_labels_complete():
    for key in ("supported", "partially_supported", "unsupported", "blocked"):
        assert key in CLAIM_STATUS_LABELS


def test_enrich_evidence_summary():
    payload = enrich_evidence(
        {
            "evidence_id": "ev_1",
            "project_id": "p1",
            "evidence_type": "paired_comparison",
            "evidence_strength": "moderate",
            "valid": True,
            "source_node_ids": ["n1"],
            "source_execution_ids": ["ex1"],
            "source_artifact_ids": ["a1"],
            "metric_summary": {"delta_mAP": 0.02},
            "limitations": ["small sample"],
        }
    )
    assert payload["evidence_type_label"] == "配对比较"
    assert payload["summary"]["source_execution_ids"] == ["ex1"]
    assert payload["summary"]["metric_summary"]["delta_mAP"] == 0.02


def test_enrich_claim_links_evidence_ids():
    claim = enrich_claim(
        {
            "claim_id": "c1",
            "claim": "Fusion helps",
            "support_status": "blocked",
            "evidence": ["ev_1"],
            "required_evidence_types": ["ablation"],
            "reason": "missing ablation",
            "limitations": ["debug only"],
        }
    )
    assert claim["claim_text"] == "Fusion helps"
    assert claim["supporting_evidence_ids"] == ["ev_1"]
    assert claim["support_status_label"] == "已阻断"
    assert claim["block_reason"] == "missing ablation"
    assert "ablation" in claim["next_evidence_needed"]


def test_enrich_claim_matrix_counts():
    matrix = enrich_claim_matrix(
        {
            "project_id": "p1",
            "protocol_id": "proto",
            "evidence_ids": ["ev_1"],
            "claims": [
                {
                    "claim_id": "c1",
                    "claim_text": "A",
                    "support_status": "supported",
                    "supporting_evidence_ids": ["ev_1"],
                },
                {
                    "claim_id": "c2",
                    "claim_text": "B",
                    "support_status": "blocked",
                    "supporting_evidence_ids": [],
                    "reason": "no ablation",
                },
            ],
        }
    )
    assert matrix["status_counts"]["supported"] == 1
    assert matrix["status_counts"]["blocked"] == 1
    assert matrix["sections"]["overview"]["claim_count"] == 2


def test_enrich_report_sections():
    report = enrich_report(
        {
            "report_id": "r1",
            "project_id": "p1",
            "status": "verified",
            "research_goal": "Improve detection",
            "metrics_table": [{"node": "n1", "mAP": 0.5}],
            "key_path": [{"node_id": "n1"}],
            "supported_claims": [{"claim_id": "c1", "evidence_ids": ["ev_1"]}],
            "blocked_claims": [],
            "limitations": ["debug dataset"],
            "verification": {"valid": True},
            "markdown_path": "/tmp/r1.md",
        }
    )
    assert report["sections"]["overview"]["verification_valid"] is True
    assert report["sections"]["key_results"]["metrics"][0]["mAP"] == 0.5
    assert "markdown" in report["export_options"]
