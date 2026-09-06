from __future__ import annotations

import json
from pathlib import Path

import pytest

from scientist_lab.domain import NodeStage, NodeStatus, NodeType, ProjectStatus
from scientist_lab.domain.models import ExperimentNode, ResearchProject, new_id
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


def _bootstrap(service: ExperimentService, *, other_project: bool = False) -> None:
    now = "2026-01-01T00:00:00+00:00"
    service.protocols.create_from_path(EXAMPLES / "rgbt_protocol.json")
    service.repo.upsert_project(
        ResearchProject(
            project_id="project_rgbt_003",
            title="RGB-T",
            research_goal="Finite tree search",
            status=ProjectStatus.ACTIVE,
            created_at=now,
            updated_at=now,
        )
    )
    contract = json.loads(
        (EXAMPLES / "rgbt_formal_fusion_contract.json").read_text(encoding="utf-8")
    )
    project_id = "project_other" if other_project else "project_rgbt_003"
    if other_project:
        service.repo.upsert_project(
            ResearchProject(
                project_id="project_other",
                title="Other",
                research_goal="mismatch",
                status=ProjectStatus.ACTIVE,
                created_at=now,
                updated_at=now,
            )
        )
        contract = dict(contract)
        contract["project_id"] = "project_other"
    service.repo.upsert_node(
        ExperimentNode(
            node_id="rgbt_formal_node_003",
            project_id=project_id,
            node_type=NodeType.BASELINE,
            stage=NodeStage.DONE,
            status=NodeStatus.SUCCEEDED,
            depth=0,
            contract_json=contract,
            created_at=now,
            updated_at=now,
        )
    )


def test_create_tree_success(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap(service)
    data = service.tree_create(
        "project_rgbt_003",
        root_node_id="rgbt_formal_node_003",
        protocol_id="protocol_rgbt_001",
        max_depth=3,
        max_nodes=8,
        max_children=3,
    )
    assert data["status"] == "created"
    assert data["node_count"] == 1
    assert data["max_nodes"] == 8


def test_auto_creates_root_tree_node(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap(service)
    data = service.tree_create(
        "project_rgbt_003",
        root_node_id="rgbt_formal_node_003",
        protocol_id="protocol_rgbt_001",
    )
    root = data["nodes"][0]
    assert root["experiment_node_id"] == "rgbt_formal_node_003"
    assert root["node_type"] == "root"
    assert root["status"] == "created"
    assert root["depth"] == 0
    assert root["parent_tree_node_id"] is None


def test_missing_root_rejected(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap(service)
    with pytest.raises(KeyError, match="root experiment node not found"):
        service.tree_create(
            "project_rgbt_003",
            root_node_id="missing_node",
            protocol_id="protocol_rgbt_001",
        )


def test_missing_protocol_rejected(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap(service)
    with pytest.raises(KeyError, match="protocol not found"):
        service.tree_create(
            "project_rgbt_003",
            root_node_id="rgbt_formal_node_003",
            protocol_id="protocol_missing",
        )


def test_project_mismatch_rejected(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap(service, other_project=True)
    with pytest.raises(ValueError, match="belongs to project"):
        service.tree_create(
            "project_rgbt_003",
            root_node_id="rgbt_formal_node_003",
            protocol_id="protocol_rgbt_001",
        )


def test_protocol_mismatch_rejected(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap(service)
    # Create a second protocol for another project id so lookup succeeds,
    # but root contract still points at protocol_rgbt_001.
    from scientist_lab.protocols.models import ExperimentProtocol
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).replace(microsecond=0)
    base = service.protocols.require("protocol_rgbt_001").model_dump(mode="json")
    base["protocol_id"] = "protocol_other_001"
    base["created_at"] = now
    service.protocols._repo.upsert(ExperimentProtocol.model_validate(base))

    with pytest.raises(ValueError, match="protocol"):
        service.tree_create(
            "project_rgbt_003",
            root_node_id="rgbt_formal_node_003",
            protocol_id="protocol_other_001",
        )


def test_max_depth_zero_rejected(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap(service)
    with pytest.raises(ValueError, match="max_depth"):
        service.tree_create(
            "project_rgbt_003",
            root_node_id="rgbt_formal_node_003",
            protocol_id="protocol_rgbt_001",
            max_depth=0,
        )


def test_illegal_max_nodes_rejected(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap(service)
    with pytest.raises(ValueError, match="max_nodes"):
        service.tree_create(
            "project_rgbt_003",
            root_node_id="rgbt_formal_node_003",
            protocol_id="protocol_rgbt_001",
            max_nodes=1,
        )
    with pytest.raises(ValueError, match="max_nodes must be >= max_children"):
        service.tree_create(
            "project_rgbt_003",
            root_node_id="rgbt_formal_node_003",
            protocol_id="protocol_rgbt_001",
            max_nodes=2,
            max_children=3,
        )


def test_duplicate_tree_id_rejected(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap(service)
    tree_id = "tree_fixed_v111"
    service.tree_create(
        "project_rgbt_003",
        root_node_id="rgbt_formal_node_003",
        protocol_id="protocol_rgbt_001",
        tree_id=tree_id,
    )
    with pytest.raises(ValueError, match="duplicate tree_id"):
        service.tree_create(
            "project_rgbt_003",
            root_node_id="rgbt_formal_node_003",
            protocol_id="protocol_rgbt_001",
            tree_id=tree_id,
        )


def test_same_experiment_node_cannot_reenter(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap(service)
    data = service.tree_create(
        "project_rgbt_003",
        root_node_id="rgbt_formal_node_003",
        protocol_id="protocol_rgbt_001",
    )
    root = data["nodes"][0]
    with pytest.raises(ValueError, match="already in tree"):
        service.trees.register_experiment_node(
            data["tree_id"],
            experiment_node_id="rgbt_formal_node_003",
            parent_tree_node_id=root["tree_node_id"],
            node_type="improve",
        )


def test_tree_show_returns_full_payload(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap(service)
    created = service.tree_create(
        "project_rgbt_003",
        root_node_id="rgbt_formal_node_003",
        protocol_id="protocol_rgbt_001",
    )
    shown = service.tree_show(created["tree_id"])
    assert shown["tree_id"] == created["tree_id"]
    assert shown["nodes"]
    assert shown["ascii_tree"].startswith("root: rgbt_formal_node_003")
    assert "created" in shown["ascii_tree"]
    status = service.tree_status(created["tree_id"])
    assert status["node_count"] == 1
    nodes = service.tree_nodes(created["tree_id"])
    assert len(nodes) == 1


def test_tree_survives_reopen(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap(service)
    created = service.tree_create(
        "project_rgbt_003",
        root_node_id="rgbt_formal_node_003",
        protocol_id="protocol_rgbt_001",
        tree_id=new_id("tree"),
    )
    tree_id = created["tree_id"]
    reopened = _service(tmp_path)
    status = reopened.tree_status(tree_id)
    assert status["status"] == "created"
    assert status["node_count"] == 1
    shown = reopened.tree_show(tree_id)
    assert shown["nodes"][0]["node_type"] == "root"
