from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scientist_lab.domain import NodeStage, NodeStatus, NodeType, ProjectStatus
from scientist_lab.domain.models import ExperimentNode, ResearchProject, new_id
from scientist_lab.search.models import ExperimentTree, TreeNode
from scientist_lab.search.selection_policy import rank_expandable_parents
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


def _now() -> str:
    return "2026-01-01T00:00:00+00:00"


def _bootstrap(service: ExperimentService) -> None:
    now = _now()
    service.protocols.create_from_path(EXAMPLES / "rgbt_protocol.json")
    service.set_budget("project_rgbt_003", max_new_nodes=5, max_gpu_hours=10)
    service.repo.upsert_project(
        ResearchProject(
            project_id="project_rgbt_003",
            title="RGB-T",
            research_goal="Best-First selection",
            status=ProjectStatus.ACTIVE,
            created_at=now,
            updated_at=now,
        )
    )
    for node_id, mean in [
        ("rgbt_formal_node_003", 0.40),
        ("rgbt_node_improve", 0.55),
        ("rgbt_node_weak", 0.30),
    ]:
        contract = json.loads(
            (EXAMPLES / "rgbt_formal_fusion_contract.json").read_text(encoding="utf-8")
        )
        contract = dict(contract)
        contract["node_id"] = node_id
        service.repo.upsert_node(
            ExperimentNode(
                node_id=node_id,
                project_id="project_rgbt_003",
                node_type=NodeType.BASELINE
                if node_id == "rgbt_formal_node_003"
                else NodeType.IMPROVEMENT,
                stage=NodeStage.DONE,
                status=NodeStatus.SUCCEEDED,
                depth=0 if node_id == "rgbt_formal_node_003" else 1,
                contract_json=contract,
                feedback_json={
                    "aggregate_metrics": {
                        "primary_metric": "mAP50_95",
                        "seed_count": 3,
                        "aggregate_metrics": {
                            "mAP50_95": {
                                "mean": mean,
                                "std": 0.02,
                                "min": mean - 0.02,
                                "max": mean + 0.02,
                            },
                            "duration_seconds": {
                                "mean": 100.0,
                                "std": 1.0,
                                "min": 90.0,
                                "max": 110.0,
                            },
                        },
                    }
                },
                created_at=now,
                updated_at=now,
            )
        )


def _tree_with_children(service: ExperimentService) -> dict:
    created = service.tree_create(
        "project_rgbt_003",
        root_node_id="rgbt_formal_node_003",
        protocol_id="protocol_rgbt_001",
        max_depth=3,
        max_nodes=8,
        max_children=3,
    )
    root = created["nodes"][0]
    # Mark root evaluated so non-root rules are comparable; root created is also ok.
    service.trees.set_node_status(root["tree_node_id"], "evaluated")
    improve = service.trees.register_experiment_node(
        created["tree_id"],
        experiment_node_id="rgbt_node_improve",
        parent_tree_node_id=root["tree_node_id"],
        node_type="improve",
    )
    weak = service.trees.register_experiment_node(
        created["tree_id"],
        experiment_node_id="rgbt_node_weak",
        parent_tree_node_id=root["tree_node_id"],
        node_type="improve",
    )
    service.trees.set_node_status(improve.tree_node_id, "evaluated")
    service.trees.set_node_status(weak.tree_node_id, "evaluated")
    return {
        "tree_id": created["tree_id"],
        "root": root,
        "improve": improve,
        "weak": weak,
    }


def test_best_first_selects_highest_priority(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap(service)
    ctx = _tree_with_children(service)
    selected = service.tree_select_parent(ctx["tree_id"])
    assert selected["selected"] is not None
    # Higher mean → higher performance → typically higher expansion among evaluated.
    assert selected["selected"]["experiment_node_id"] == "rgbt_node_improve"
    assert selected["tree_status"] == "active"
    assert selected["selected_node_id"] == "rgbt_node_improve"


def test_pruned_node_cannot_expand(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap(service)
    ctx = _tree_with_children(service)
    service.trees.set_node_status(ctx["improve"].tree_node_id, "pruned")
    # Force priorities without relying on rescoring order.
    now = datetime.now(timezone.utc).replace(microsecond=0)
    for node, prio in [
        (ctx["improve"], 0.99),
        (ctx["weak"], 0.50),
        (service.trees._repo.get_node(ctx["root"]["tree_node_id"]), 0.40),
    ]:
        assert node is not None
        service.trees._repo.upsert_node(
            node.model_copy(
                update={"expansion_priority": prio, "score": prio, "updated_at": now}
            )
        )
    result = service.tree_select_parent(ctx["tree_id"], rescore=False)
    assert result["selected"]["experiment_node_id"] != "rgbt_node_improve"
    blocked_ids = {b["experiment_node_id"] for b in result["blocked"]}
    assert "rgbt_node_improve" in blocked_ids


def test_unevaluated_non_root_cannot_be_improve_parent(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap(service)
    created = service.tree_create(
        "project_rgbt_003",
        root_node_id="rgbt_formal_node_003",
        protocol_id="protocol_rgbt_001",
    )
    root = created["nodes"][0]
    service.trees.set_node_status(root["tree_node_id"], "evaluated")
    child = service.trees.register_experiment_node(
        created["tree_id"],
        experiment_node_id="rgbt_node_improve",
        parent_tree_node_id=root["tree_node_id"],
        node_type="improve",
    )
    # child stays status=created → blocked
    now = datetime.now(timezone.utc).replace(microsecond=0)
    service.trees._repo.upsert_node(
        child.model_copy(
            update={"expansion_priority": 0.99, "updated_at": now}
        )
    )
    result = service.tree_select_parent(created["tree_id"], rescore=False)
    assert result["selected"]["experiment_node_id"] == "rgbt_formal_node_003"
    blocked = {b["experiment_node_id"]: b for b in result["blocked"]}
    assert "rgbt_node_improve" in blocked
    assert any("not yet expandable" in r for r in blocked["rgbt_node_improve"]["reasons_blocked"])


def test_max_children_blocks_parent(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap(service)
    # Need extra experiment nodes for children
    now = _now()
    for i in range(3):
        nid = f"rgbt_child_{i}"
        contract = json.loads(
            (EXAMPLES / "rgbt_formal_fusion_contract.json").read_text(encoding="utf-8")
        )
        contract = dict(contract)
        contract["node_id"] = nid
        service.repo.upsert_node(
            ExperimentNode(
                node_id=nid,
                project_id="project_rgbt_003",
                node_type=NodeType.IMPROVEMENT,
                stage=NodeStage.DONE,
                status=NodeStatus.SUCCEEDED,
                depth=1,
                contract_json=contract,
                created_at=now,
                updated_at=now,
            )
        )

    created = service.tree_create(
        "project_rgbt_003",
        root_node_id="rgbt_formal_node_003",
        protocol_id="protocol_rgbt_001",
        max_depth=3,
        max_nodes=8,
        max_children=2,
    )
    root_id = created["nodes"][0]["tree_node_id"]
    service.trees.set_node_status(root_id, "evaluated")
    for i in range(2):
        child = service.trees.register_experiment_node(
            created["tree_id"],
            experiment_node_id=f"rgbt_child_{i}",
            parent_tree_node_id=root_id,
            node_type="improve",
        )
        service.trees.set_node_status(child.tree_node_id, "evaluated")

    result = service.tree_select_parent(created["tree_id"])
    # Root has 2 children already with max_children=2 → cannot expand root
    if result["selected"]:
        assert result["selected"]["experiment_node_id"] != "rgbt_formal_node_003"
    root_blocked = [
        b for b in result["blocked"] if b["experiment_node_id"] == "rgbt_formal_node_003"
    ]
    assert root_blocked
    assert any("max_children" in r for r in root_blocked[0]["reasons_blocked"])


def test_terminal_tree_no_selection(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap(service)
    created = service.tree_create(
        "project_rgbt_003",
        root_node_id="rgbt_formal_node_003",
        protocol_id="protocol_rgbt_001",
    )
    service.trees.set_tree_status(created["tree_id"], "active")
    service.trees.set_tree_status(created["tree_id"], "user_stopped")
    result = service.tree_select_parent(created["tree_id"], rescore=False)
    assert result["selected"] is None
    assert result["stop_suggested"] is True


def test_pure_policy_ranking_order():
    now = datetime.now(timezone.utc).replace(microsecond=0)
    tree = ExperimentTree(
        tree_id="tree_x",
        project_id="p",
        protocol_id="proto",
        root_node_id="n0",
        status="active",
        max_depth=3,
        max_nodes=8,
        max_children_per_node=3,
        created_at=now,
        updated_at=now,
    )
    nodes = [
        TreeNode(
            tree_node_id="t0",
            tree_id="tree_x",
            experiment_node_id="n0",
            depth=0,
            node_type="root",
            status="evaluated",
            expansion_priority=0.4,
            score=0.4,
            created_at=now,
            updated_at=now,
        ),
        TreeNode(
            tree_node_id="t1",
            tree_id="tree_x",
            experiment_node_id="n1",
            parent_tree_node_id="t0",
            depth=1,
            node_type="improve",
            status="evaluated",
            expansion_priority=0.9,
            score=0.8,
            created_at=now,
            updated_at=now,
        ),
        TreeNode(
            tree_node_id="t2",
            tree_id="tree_x",
            experiment_node_id="n2",
            parent_tree_node_id="t0",
            depth=1,
            node_type="improve",
            status="evaluated",
            expansion_priority=0.7,
            score=0.6,
            created_at=now,
            updated_at=now,
        ),
    ]
    result = rank_expandable_parents(tree, nodes)
    assert result.selected is not None
    assert result.selected.experiment_node_id == "n1"
    assert [c.experiment_node_id for c in result.candidates] == ["n1", "n2", "n0"]
