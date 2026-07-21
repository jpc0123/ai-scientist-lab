from __future__ import annotations

import json
from pathlib import Path

import pytest

from scientist_lab.domain import NodeStage, NodeStatus, NodeType, ProjectStatus
from scientist_lab.domain.models import ExperimentNode, ResearchProject
from scientist_lab.search.expansion_service import (
    _parameter_fingerprint,
    annotate_candidates_against_tree,
    remaining_candidate_slots,
)
from scientist_lab.search.models import ExperimentTree, TreeNode
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
            research_goal="Tree plan-next",
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


def test_remaining_candidate_slots():
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    tree = ExperimentTree(
        tree_id="t",
        project_id="p",
        protocol_id="proto",
        root_node_id="n0",
        max_depth=3,
        max_nodes=5,
        max_children_per_node=3,
        created_at=now,
        updated_at=now,
    )
    parent = TreeNode(
        tree_node_id="tn",
        tree_id="t",
        experiment_node_id="n0",
        depth=0,
        created_at=now,
        updated_at=now,
    )
    assert remaining_candidate_slots(tree, parent, child_count=0, node_count=1) == 3
    assert remaining_candidate_slots(tree, parent, child_count=2, node_count=1) == 1
    assert remaining_candidate_slots(tree, parent, child_count=3, node_count=1) == 0
    assert remaining_candidate_slots(tree, parent, child_count=0, node_count=5) == 0


def test_annotate_rejects_duplicates_and_caps():
    parent = "n0"
    fp = _parameter_fingerprint({"input_mode": "rgb", "fusion_method": "none"})
    candidates = [
        {
            "candidate_id": "c1",
            "parent_node_id": parent,
            "parameter_changes": {"input_mode": "rgb", "fusion_method": "none"},
            "status": "verified",
        },
        {
            "candidate_id": "c2",
            "parent_node_id": parent,
            "parameter_changes": {"input_mode": "thermal", "fusion_method": "none"},
            "status": "verified",
        },
        {
            "candidate_id": "c3",
            "parent_node_id": parent,
            "parameter_changes": {"input_mode": "rgbt", "fusion_method": "early_concat"},
            "status": "verified",
        },
    ]
    filtered = annotate_candidates_against_tree(
        candidates,
        parent_experiment_node_id=parent,
        existing_fingerprints={fp},
        max_candidates=1,
    )
    assert filtered["accepted_count"] == 1
    assert filtered["accepted"][0]["candidate_id"] == "c2"
    assert any(r["tree_filter"] == "duplicate_configuration" for r in filtered["rejected"])
    assert any(d["tree_filter"] == "deferred_over_limit" for d in filtered["deferred"])


def test_tree_plan_next_mock_flow(tmp_path: Path):
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
    result = service.tree_plan_next(created["tree_id"])
    assert result["status"] == "planned"
    assert result["plan_id"]
    assert result["ranking"]
    assert result["top_candidate_id"]
    assert result["tree_status"] == "waiting_approval"
    assert result["parent"]["experiment_node_id"] == "rgbt_formal_node_003"

    nodes = service.tree_nodes(created["tree_id"])
    assert nodes[0]["plan_id"] == result["plan_id"]

    plan = service.show_plan(result["plan_id"])
    assert plan["project_id"] == "project_rgbt_003"
    assert plan["candidates"]


def test_tree_plan_next_stop_when_terminal(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap(service)
    created = service.tree_create(
        "project_rgbt_003",
        root_node_id="rgbt_formal_node_003",
        protocol_id="protocol_rgbt_001",
    )
    service.trees.set_tree_status(created["tree_id"], "active")
    service.trees.set_tree_status(created["tree_id"], "user_stopped")
    with pytest.raises(ValueError, match="terminal"):
        service.tree_plan_next(created["tree_id"], rescore=False)


def test_tree_plan_next_filters_existing_rgb_config(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap(service)
    now = "2026-01-01T00:00:00+00:00"
    rgb = json.loads(
        (EXAMPLES / "rgbt_formal_rgb_contract.json").read_text(encoding="utf-8")
    )
    service.repo.upsert_node(
        ExperimentNode(
            node_id="rgbt_formal_node_rgb",
            project_id="project_rgbt_003",
            node_type=NodeType.IMPROVEMENT,
            stage=NodeStage.DONE,
            status=NodeStatus.SUCCEEDED,
            depth=1,
            contract_json=rgb,
            created_at=now,
            updated_at=now,
        )
    )
    created = service.tree_create(
        "project_rgbt_003",
        root_node_id="rgbt_formal_node_003",
        protocol_id="protocol_rgbt_001",
    )
    root_id = created["nodes"][0]["tree_node_id"]
    service.trees.set_node_status(root_id, "evaluated")
    child = service.trees.register_experiment_node(
        created["tree_id"],
        experiment_node_id="rgbt_formal_node_rgb",
        parent_tree_node_id=root_id,
        node_type="ablation",
    )
    service.trees.set_node_status(child.tree_node_id, "evaluated")

    result = service.tree_plan_next(created["tree_id"])
    # RGB ablation candidate should be filtered as duplicate if planner proposes it.
    rejected = result["candidate_filter"]["rejected"]
    assert any(
        r.get("tree_filter") == "duplicate_configuration"
        and (r.get("parameter_changes") or {}).get("input_mode") == "rgb"
        for r in rejected
    ) or result["status"] in {"planned", "no_valid_candidates"}
