from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scientist_lab.domain import NodeStage, NodeStatus, NodeType, ProjectStatus
from scientist_lab.domain.models import ExperimentNode, ResearchProject
from scientist_lab.search.models import ExperimentTree, TreeNode
from scientist_lab.search.stop_policy import (
    StopPolicy,
    evaluate_stop,
    update_improvement_counters,
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
            research_goal="Stop policy",
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
            created_at=now,
            updated_at=now,
        )
    )


def _tree(now: datetime, **kwargs) -> ExperimentTree:
    base = dict(
        tree_id="tree_x",
        project_id="p",
        protocol_id="proto",
        root_node_id="n0",
        status="active",
        max_depth=2,
        max_nodes=3,
        max_children_per_node=2,
        created_at=now,
        updated_at=now,
    )
    base.update(kwargs)
    return ExperimentTree(**base)


def test_budget_exhausted_stop():
    now = datetime.now(timezone.utc)
    tree = _tree(now)
    nodes = [
        TreeNode(
            tree_node_id="t0",
            tree_id="tree_x",
            experiment_node_id="n0",
            depth=0,
            node_type="root",
            status="evaluated",
            expansion_priority=0.5,
            created_at=now,
            updated_at=now,
        )
    ]
    decision = evaluate_stop(
        tree,
        nodes,
        remaining_budget={"max_new_nodes": 0, "max_total_gpu_hours": 5},
    )
    assert decision.should_stop
    assert decision.status == "budget_exhausted"


def test_max_nodes_stop():
    now = datetime.now(timezone.utc)
    tree = _tree(now, max_nodes=2)
    nodes = [
        TreeNode(
            tree_node_id=f"t{i}",
            tree_id="tree_x",
            experiment_node_id=f"n{i}",
            depth=i,
            node_type="root" if i == 0 else "improve",
            status="evaluated",
            expansion_priority=0.5,
            parent_tree_node_id=None if i == 0 else "t0",
            created_at=now,
            updated_at=now,
        )
        for i in range(2)
    ]
    decision = evaluate_stop(tree, nodes, remaining_budget={"max_new_nodes": 3})
    assert decision.should_stop
    assert decision.status == "no_valid_candidates"


def test_duplicate_candidates_stop():
    now = datetime.now(timezone.utc)
    tree = _tree(now)
    nodes = [
        TreeNode(
            tree_node_id="t0",
            tree_id="tree_x",
            experiment_node_id="n0",
            depth=0,
            node_type="root",
            status="evaluated",
            expansion_priority=0.8,
            created_at=now,
            updated_at=now,
        )
    ]
    decision = evaluate_stop(
        tree,
        nodes,
        remaining_budget={"max_new_nodes": 3, "max_total_gpu_hours": 5},
        last_plan_filter={
            "accepted_count": 0,
            "rejected": [
                {"tree_filter": "duplicate_configuration"},
                {"tree_filter": "duplicate_configuration"},
            ],
            "deferred": [],
        },
    )
    assert decision.should_stop
    assert "duplicate" in (decision.reason or "").lower()


def test_no_improvement_rounds_stop():
    now = datetime.now(timezone.utc)
    tree = _tree(now, no_improvement_rounds=2, max_no_improvement_rounds=2)
    nodes = [
        TreeNode(
            tree_node_id="t0",
            tree_id="tree_x",
            experiment_node_id="n0",
            depth=0,
            node_type="root",
            status="evaluated",
            expansion_priority=0.8,
            created_at=now,
            updated_at=now,
        )
    ]
    decision = evaluate_stop(
        tree,
        nodes,
        remaining_budget={"max_new_nodes": 3, "max_total_gpu_hours": 5},
    )
    assert decision.should_stop
    assert decision.status == "completed"


def test_update_improvement_counters():
    now = datetime.now(timezone.utc)
    tree = _tree(now, best_score_seen=0.5, no_improvement_rounds=1)
    policy = StopPolicy(minimum_score_improvement=0.01)
    worse = update_improvement_counters(tree, new_scores=[0.50], policy=policy)
    assert worse.no_improvement_rounds == 2
    better = update_improvement_counters(tree, new_scores=[0.55], policy=policy)
    assert better.no_improvement_rounds == 0
    assert better.best_score_seen == 0.55


def test_tree_stop_cli_path(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap(service)
    created = service.tree_create(
        "project_rgbt_003",
        root_node_id="rgbt_formal_node_003",
        protocol_id="protocol_rgbt_001",
    )
    stopped = service.tree_stop(created["tree_id"], reason="manual halt for demo")
    assert stopped["status"] == "user_stopped"
    assert stopped["stop_reason"] == "manual halt for demo"
    with pytest.raises(ValueError, match="terminal"):
        service.tree_plan_next(created["tree_id"])


def test_budget_stop_via_plan_next(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap(service)
    service.set_budget("project_rgbt_003", max_new_nodes=0, max_gpu_hours=10)
    created = service.tree_create(
        "project_rgbt_003",
        root_node_id="rgbt_formal_node_003",
        protocol_id="protocol_rgbt_001",
    )
    # Force tree active with expandable root then evaluate stop through helper.
    service.trees.set_tree_status(created["tree_id"], "active")
    stop = service._tree_evaluate_and_maybe_stop(created["tree_id"])
    assert stop is not None
    assert stop["status"] == "budget_exhausted"
    assert service.tree_status(created["tree_id"])["status"] == "budget_exhausted"
