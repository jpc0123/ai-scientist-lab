from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from scientist_lab.domain import NodeStage, NodeStatus, NodeType, ProjectStatus
from scientist_lab.domain.models import ExperimentNode, ResearchProject
from scientist_lab.search.export import render_mermaid
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
    service.repo.upsert_project(
        ResearchProject(
            project_id="project_rgbt_003",
            title="RGB-T",
            research_goal="Tree export",
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


def test_render_mermaid_parent_child():
    now = datetime.now(timezone.utc).replace(microsecond=0)
    tree = ExperimentTree(
        tree_id="tree_demo",
        project_id="project_rgbt_003",
        protocol_id="protocol_rgbt_001",
        root_node_id="rgbt_formal_node_003",
        status="active",
        max_depth=3,
        max_nodes=8,
        max_children_per_node=3,
        created_at=now,
        updated_at=now,
    )
    root = TreeNode(
        tree_node_id="tn_root",
        tree_id="tree_demo",
        experiment_node_id="rgbt_formal_node_003",
        parent_tree_node_id=None,
        depth=0,
        node_type="root",
        status="evaluated",
        score=0.72,
        created_at=now,
        updated_at=now,
    )
    child = TreeNode(
        tree_node_id="tn_child",
        tree_id="tree_demo",
        experiment_node_id="rgbt_node_child",
        parent_tree_node_id="tn_root",
        depth=1,
        node_type="improve",
        status="evaluated",
        score=0.65,
        created_at=now,
        updated_at=now,
    )
    text = render_mermaid(tree, [root, child])
    assert text.startswith("flowchart TD")
    assert "tn_root" in text
    assert "tn_child" in text
    assert "tn_root --> tn_child" in text
    assert "rgbt_formal_node_003" in text


def test_tree_export_json_and_mermaid(tmp_path: Path):
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
    tree_id = created["tree_id"]

    as_json = service.tree_export(tree_id, format="json")
    assert as_json["format"] == "json"
    assert as_json["tree_id"] == tree_id
    assert as_json["node_count"] == 1
    assert as_json["nodes"]
    assert "flowchart TD" in as_json["mermaid"]
    assert as_json["ascii_tree"].startswith("root:")

    as_mmd = service.tree_export(tree_id, format="mermaid")
    assert as_mmd["format"] == "mermaid"
    assert as_mmd["mermaid"].startswith("flowchart TD")
    assert "nodes" not in as_mmd


def test_tree_export_rejects_bad_format(tmp_path: Path):
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
    try:
        service.tree_export(created["tree_id"], format="xml")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "unsupported export format" in str(exc)
