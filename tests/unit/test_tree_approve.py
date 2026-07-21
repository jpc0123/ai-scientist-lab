from __future__ import annotations

import json
from pathlib import Path

import pytest

from scientist_lab.domain import NodeStage, NodeStatus, NodeType, ProjectStatus
from scientist_lab.domain.models import ExperimentNode, ResearchProject
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
            research_goal="Tree approve",
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
                            "min": 0.4,
                            "max": 0.44,
                        }
                    },
                }
            },
            created_at=now,
            updated_at=now,
        )
    )


def test_tree_approve_creates_iteration(tmp_path: Path):
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
    planned = service.tree_plan_next(created["tree_id"])
    assert planned["status"] == "planned"
    cand = planned["top_candidate_id"]
    assert cand

    approved = service.tree_approve(created["tree_id"], cand)
    assert approved["iteration_id"]
    assert approved["proposed_node_id"]
    assert approved["tree_status"] == "waiting_approval"
    assert approved["iteration"]["status"] == "waiting_approval"

    child = approved["tree_node"]
    assert child["experiment_node_id"] == approved["proposed_node_id"]
    assert child["candidate_id"] == cand
    assert child["plan_id"] == planned["plan_id"]
    assert child["iteration_id"] == approved["iteration_id"]
    assert child["status"] == "waiting_approval"
    assert child["parent_tree_node_id"] == planned["parent"]["tree_node_id"]
    assert child["depth"] == 1

    # ExperimentNode created
    exp = service.repo.get_node(approved["proposed_node_id"])
    assert exp is not None
    assert exp.parent_node_id == "rgbt_formal_node_003"


def test_tree_approve_without_plan_rejected(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap(service)
    created = service.tree_create(
        "project_rgbt_003",
        root_node_id="rgbt_formal_node_003",
        protocol_id="protocol_rgbt_001",
    )
    with pytest.raises(ValueError, match="tree-plan-next|plan_id|approve"):
        service.tree_approve(created["tree_id"], "candidate_missing")


def test_tree_approve_unknown_candidate_rejected(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap(service)
    created = service.tree_create(
        "project_rgbt_003",
        root_node_id="rgbt_formal_node_003",
        protocol_id="protocol_rgbt_001",
    )
    service.tree_plan_next(created["tree_id"])
    with pytest.raises(KeyError, match="candidate not found"):
        service.tree_approve(created["tree_id"], "candidate_does_not_exist")


def test_tree_approve_terminal_rejected(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap(service)
    created = service.tree_create(
        "project_rgbt_003",
        root_node_id="rgbt_formal_node_003",
        protocol_id="protocol_rgbt_001",
    )
    planned = service.tree_plan_next(created["tree_id"])
    service.trees.set_tree_status(created["tree_id"], "user_stopped")
    with pytest.raises(ValueError, match="terminal"):
        service.tree_approve(created["tree_id"], planned["top_candidate_id"])


def test_map_experiment_type():
    assert (
        ExperimentService._map_experiment_type_to_tree_node("ablation") == "ablation"
    )
    assert (
        ExperimentService._map_experiment_type_to_tree_node("robustness") == "improve"
    )
