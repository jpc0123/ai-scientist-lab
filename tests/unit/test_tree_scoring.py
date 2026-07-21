from __future__ import annotations

import json
from pathlib import Path

import pytest

from scientist_lab.domain import NodeStage, NodeStatus, NodeType, ProjectStatus
from scientist_lab.domain.models import ExperimentNode, ResearchProject
from scientist_lab.search.scoring import (
    compute_scores,
    efficiency_score,
    evidence_strength_score,
    information_gap_score,
    performance_score,
    stability_score,
)
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings


EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _service(tmp_path: Path) -> ExperimentService:
    root = Path(__file__).resolve().parents[2]
    settings = Settings(
        project_root=root,
        db_path=tmp_path / "test.db",
        runtime_dir=tmp_path / "runtime",
        outputs_dir=tmp_path / "outputs",
        experiment_app_dir=root / "experiment_app",
    ).resolve()
    return ExperimentService(settings=settings)


def _bootstrap(service: ExperimentService) -> None:
    now = "2026-01-01T00:00:00+00:00"
    service.protocols.create_from_path(EXAMPLES / "rgbt_protocol.json")
    service.set_budget("project_rgbt_003", max_new_nodes=5, max_gpu_hours=10)
    service.repo.upsert_project(
        ResearchProject(
            project_id="project_rgbt_003",
            title="RGB-T",
            research_goal="Scoring",
            status=ProjectStatus.ACTIVE,
            created_at=now,
            updated_at=now,
        )
    )
    contract = json.loads(
        (EXAMPLES / "rgbt_formal_fusion_contract.json").read_text(encoding="utf-8")
    )
    service.repo.upsert_node(
        ExperimentNode(
            node_id="rgbt_formal_node_003",
            project_id="project_rgbt_003",
            node_type=NodeType.BASELINE,
            stage=NodeStage.DONE,
            status=NodeStatus.SUCCEEDED,
            depth=0,
            contract_json=contract,
            feedback_json={
                "aggregate_metrics": {
                    "primary_metric": "mAP50_95",
                    "seed_count": 3,
                    "aggregate_metrics": {
                        "mAP50_95": {
                            "mean": 0.42,
                            "std": 0.02,
                            "min": 0.40,
                            "max": 0.44,
                        },
                        "duration_seconds": {
                            "mean": 120.0,
                            "std": 5.0,
                            "min": 110.0,
                            "max": 130.0,
                        },
                    },
                }
            },
            created_at=now,
            updated_at=now,
        )
    )


def test_performance_score_from_aggregate():
    agg = {
        "primary_metric": "mAP50_95",
        "seed_count": 3,
        "aggregate_metrics": {"mAP50_95": {"mean": 0.6, "std": 0.01}},
    }
    score, primary, _ = performance_score(agg)
    assert primary == "mAP50_95"
    assert abs(score - 0.6) < 1e-6


def test_stability_prefers_low_variance():
    stable = {
        "primary_metric": "mAP50_95",
        "seed_count": 3,
        "aggregate_metrics": {"mAP50_95": {"mean": 0.5, "std": 0.01}},
    }
    unstable = {
        "primary_metric": "mAP50_95",
        "seed_count": 3,
        "aggregate_metrics": {"mAP50_95": {"mean": 0.5, "std": 0.25}},
    }
    s1, _, _ = stability_score(stable)
    s2, _, _ = stability_score(unstable)
    assert s1 > s2


def test_efficiency_better_when_faster():
    fast = {
        "aggregate_metrics": {"duration_seconds": {"mean": 60.0}},
    }
    slow = {
        "aggregate_metrics": {"duration_seconds": {"mean": 900.0}},
    }
    e1, _ = efficiency_score(fast)
    e2, _ = efficiency_score(slow)
    assert e1 > e2


def test_evidence_strength_mapping():
    score, _ = evidence_strength_score(
        [{"evidence_strength": "strong"}, {"evidence_strength": "weak"}]
    )
    assert abs(score - 0.6) < 1e-6


def test_information_gap_from_claim_matrix():
    matrix = {
        "claims": [
            {"support_status": "blocked", "claim": "SOTA"},
            {"support_status": "unsupported", "reason": "no ablation"},
            {"support_status": "supported", "claim": "ok"},
        ]
    }
    score, count, _ = information_gap_score(matrix)
    assert count == 2
    assert score > 0.3


def test_node_score_weights_and_bounds():
    breakdown = compute_scores(
        experiment_node_id="n1",
        node_type="ablation",
        status="evaluated",
        aggregate={
            "primary_metric": "mAP50_95",
            "seed_count": 3,
            "aggregate_metrics": {
                "mAP50_95": {"mean": 0.5, "std": 0.02},
                "duration_seconds": {"mean": 100.0},
            },
        },
        evidence_records=[{"evidence_strength": "moderate"}],
        claim_matrix={
            "claims": [
                {"support_status": "unsupported", "reason": "need ablation"},
            ]
        },
        remaining_budget={"max_new_nodes": 3, "max_total_gpu_hours": 8.0},
        has_protocol=True,
        protocol_valid=True,
    )
    assert 0.0 <= breakdown.node_score <= 1.0
    assert 0.0 <= breakdown.expansion_priority <= 1.0
    assert breakdown.protocol_compliance_score == 1.0
    expected = (
        0.40 * breakdown.performance_score
        + 0.20 * breakdown.stability_score
        + 0.15 * breakdown.efficiency_score
        + 0.15 * breakdown.evidence_strength_score
        + 0.10 * breakdown.protocol_compliance_score
    )
    assert abs(breakdown.node_score - round(expected, 4)) < 1e-6


def test_duplicate_and_budget_penalties():
    base = compute_scores(
        experiment_node_id="n1",
        remaining_budget={"max_new_nodes": 3, "max_total_gpu_hours": 8.0},
    )
    penalized = compute_scores(
        experiment_node_id="n1",
        remaining_budget={"max_new_nodes": 1, "max_total_gpu_hours": 8.0},
        duplicate_fingerprint=True,
        parent_failed_streak=2,
    )
    assert penalized.expansion_priority < base.expansion_priority
    assert any("Duplicate" in p for p in penalized.penalties)


def test_tree_score_persists(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap(service)
    created = service.tree_create(
        "project_rgbt_003",
        root_node_id="rgbt_formal_node_003",
        protocol_id="protocol_rgbt_001",
        max_depth=3,
        max_nodes=8,
        max_children=3,
    )
    scored = service.tree_score(created["tree_id"])
    assert scored["scored_count"] == 1
    item = scored["scores"][0]
    assert item["node_score"] is not None
    assert item["expansion_priority"] is not None
    assert item["performance_score"] == pytest.approx(0.42, abs=1e-3)
    assert scored["ranking_by_expansion_priority"][0]["experiment_node_id"] == (
        "rgbt_formal_node_003"
    )

    nodes = service.tree_nodes(created["tree_id"])
    assert nodes[0]["score"] == item["node_score"]
    assert nodes[0]["expansion_priority"] == item["expansion_priority"]

    # Restart recovery of scores
    reopened = _service(tmp_path)
    restored = reopened.tree_nodes(created["tree_id"])
    assert restored[0]["score"] == item["node_score"]


def test_tree_score_single_node(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap(service)
    created = service.tree_create(
        "project_rgbt_003",
        root_node_id="rgbt_formal_node_003",
        protocol_id="protocol_rgbt_001",
    )
    tree_node_id = created["nodes"][0]["tree_node_id"]
    one = service.tree_score(created["tree_id"], tree_node_id=tree_node_id)
    assert one["tree_node_id"] == tree_node_id
    assert "node_score" in one
    assert "expansion_priority" in one
