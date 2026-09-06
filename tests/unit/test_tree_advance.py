from __future__ import annotations

import json
from pathlib import Path

import pytest

from scientist_lab.domain import NodeStage, NodeStatus, NodeType, ProjectStatus
from scientist_lab.domain.models import ExperimentNode, ResearchProject
from scientist_lab.iteration.service import IterationService
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
            research_goal="Tree advance",
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


def _plan_and_approve(service: ExperimentService) -> dict:
    created = service.tree_create(
        "project_rgbt_003",
        root_node_id="rgbt_formal_node_003",
        protocol_id="protocol_rgbt_001",
        max_depth=3,
        max_nodes=8,
        max_children=3,
    )
    planned = service.tree_plan_next(created["tree_id"])
    approved = service.tree_approve(created["tree_id"], planned["top_candidate_id"])
    return {
        "tree_id": created["tree_id"],
        "planned": planned,
        "approved": approved,
    }


def _force_iteration_completed(
    service: ExperimentService,
    *,
    iteration_id: str,
    proposed_node_id: str,
    decision_id: str = "decision_test_001",
) -> None:
    iteration = IterationService(service)
    session = iteration.repo.get_session(iteration_id)
    assert session is not None
    session.status = "completed"
    session.selected_node_id = proposed_node_id
    session.decision_id = decision_id
    iteration.repo.save_session(session)

    # Attach aggregate metrics so scoring has numbers.
    node = service.repo.get_node(proposed_node_id)
    assert node is not None
    feedback = dict(node.feedback_json or {})
    feedback["aggregate_metrics"] = {
        "primary_metric": "mAP50_95",
        "seed_count": 3,
        "aggregate_metrics": {
            "mAP50_95": {"mean": 0.38, "std": 0.03, "min": 0.35, "max": 0.41},
            "duration_seconds": {"mean": 90.0, "std": 2.0, "min": 88.0, "max": 92.0},
        },
    }
    node.feedback_json = feedback
    service.repo.upsert_node(node)


def test_tree_advance_pending_while_waiting_iteration_approval(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap(service)
    ctx = _plan_and_approve(service)
    result = service.tree_advance(ctx["tree_id"])
    assert result["status"] == "pending"
    assert result["pending_count"] == 1
    assert result["advanced_count"] == 0
    assert "iterate-approve" in (result["pending"][0]["next_action"] or "")


def test_tree_advance_backfills_completed_iteration(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap(service)
    ctx = _plan_and_approve(service)
    approved = ctx["approved"]
    _force_iteration_completed(
        service,
        iteration_id=approved["iteration_id"],
        proposed_node_id=approved["proposed_node_id"],
        decision_id="decision_abc",
    )

    result = service.tree_advance(ctx["tree_id"])
    assert result["status"] == "advanced"
    assert result["advanced_count"] == 1
    item = result["advanced"][0]
    assert item["action"] == "evaluated"
    assert item["decision_id"] == "decision_abc"
    assert item["node_score"] is not None
    assert result["tree_status"] == "active"

    nodes = service.tree_nodes(ctx["tree_id"])
    child = next(
        n for n in nodes if n["experiment_node_id"] == approved["proposed_node_id"]
    )
    assert child["status"] == "evaluated"
    assert child["decision_id"] == "decision_abc"
    assert child["score"] is not None


def test_tree_advance_failed_iteration(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap(service)
    ctx = _plan_and_approve(service)
    iteration = IterationService(service)
    session = iteration.repo.get_session(ctx["approved"]["iteration_id"])
    assert session is not None
    session.status = "failed"
    session.error_message = "boom"
    iteration.repo.save_session(session)

    result = service.tree_advance(ctx["tree_id"])
    assert result["failed_count"] == 1
    assert result["failed"][0]["action"] == "failed"
    nodes = service.tree_nodes(ctx["tree_id"])
    child = next(
        n
        for n in nodes
        if n["experiment_node_id"] == ctx["approved"]["proposed_node_id"]
    )
    assert child["status"] == "failed"


def test_tree_advance_noop_without_iterations(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap(service)
    created = service.tree_create(
        "project_rgbt_003",
        root_node_id="rgbt_formal_node_003",
        protocol_id="protocol_rgbt_001",
    )
    result = service.tree_advance(created["tree_id"])
    assert result["status"] == "noop"
