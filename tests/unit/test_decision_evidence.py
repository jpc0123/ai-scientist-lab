from __future__ import annotations

from pathlib import Path

import pytest

from scientist_lab.evidence.decision_link import cap_evidence_strength
from scientist_lab.evidence.models import EvidenceRecord
from scientist_lab.services.experiment_service import ExperimentService

from tests.unit.test_evidence_record import _bootstrap, _service


def test_cap_evidence_strength_cannot_exceed_weak_support():
    records = [
        EvidenceRecord(
            evidence_id="e1",
            project_id="project_rgbt_003",
            evidence_type="paired_comparison",
            evidence_strength="weak",
            limitations=[],
            valid=True,
        )
    ]
    assert cap_evidence_strength("strong", records) == "weak"
    assert cap_evidence_strength("moderate", records) == "weak"


def test_decision_links_evidence_and_claim_matrix(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap(service)
    # Put protocol_id on selected node contract for resolution.
    node = service.repo.get_node("rgbt_formal_node_003")
    assert node is not None
    node.contract_json = {
        "task_type": "rgbt_detection",
        "execution_mode": "fast_eval",
        "protocol_id": "protocol_rgbt_001",
        "task_config": {
            "claim_level": "exploratory_comparison",
            "evaluation_scope": "fast_eval_subset",
        },
    }
    service.repo.upsert_node(node)

    evidence = service.build_evidence(
        "rgbt_formal_node_001", "rgbt_formal_node_003"
    )
    decision = service.record_decision(
        selected_node_id="rgbt_formal_node_003",
        alternatives=["rgbt_formal_node_001"],
        decision_type="exploratory_improvement",
        reason="Fusion wins AP_small under protocol.",
        evidence_strength="strong",
        baseline_node_id="rgbt_formal_node_001",
        candidate_node_id="rgbt_formal_node_003",
    )

    assert decision["selected_node_id"] == "rgbt_formal_node_003"
    assert decision["protocol_id"] == "protocol_rgbt_001"
    assert decision["evidence_strength"] == "weak"  # capped by smoke/exploratory + evidence
    assert set(decision["supporting_evidence_ids"]) == set(evidence["evidence_ids"])
    assert decision["claim_matrix_path"]
    assert Path(decision["claim_matrix_path"]).is_file()

    shown = service.show_decision(decision["decision_id"])
    assert shown["decision_id"] == decision["decision_id"]
    assert shown["supporting_evidence_ids"] == decision["supporting_evidence_ids"]
    assert shown["claim_matrix_path"] == decision["claim_matrix_path"]


def test_decision_explicit_evidence_ids(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap(service)
    node = service.repo.get_node("rgbt_formal_node_003")
    assert node is not None
    node.contract_json = {
        "task_type": "rgbt_detection",
        "execution_mode": "fast_eval",
        "protocol_id": "protocol_rgbt_001",
        "task_config": {"claim_level": "exploratory_comparison"},
    }
    service.repo.upsert_node(node)

    evidence = service.build_evidence(
        "rgbt_formal_node_001", "rgbt_formal_node_003"
    )
    only = [evidence["evidence_ids"][0]]
    decision = service.record_decision(
        selected_node_id="rgbt_formal_node_003",
        alternatives=["rgbt_formal_node_001"],
        decision_type="exploratory_tradeoff",
        reason="Attach one evidence only.",
        evidence_strength="moderate",
        supporting_evidence_ids=only,
        auto_attach_evidence=False,
        ensure_claim_matrix=True,
    )
    assert decision["supporting_evidence_ids"] == only
    assert decision["protocol_id"] == "protocol_rgbt_001"
    assert Path(decision["claim_matrix_path"]).is_file()


def test_decision_rejects_missing_evidence_id(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap(service)
    node = service.repo.get_node("rgbt_formal_node_003")
    assert node is not None
    node.contract_json = {
        "task_type": "rgbt_detection",
        "execution_mode": "fast_eval",
        "task_config": {"claim_level": "exploratory_comparison"},
    }
    service.repo.upsert_node(node)

    with pytest.raises(KeyError, match="evidence not found"):
        service.record_decision(
            selected_node_id="rgbt_formal_node_003",
            alternatives=[],
            decision_type="exploratory_improvement",
            reason="bad id",
            supporting_evidence_ids=["evidence_does_not_exist"],
            auto_attach_evidence=False,
            ensure_claim_matrix=False,
        )
