from __future__ import annotations

import json
from pathlib import Path

from scientist_lab.domain import JobStatus, NodeStage, NodeStatus, NodeType, ProjectStatus
from scientist_lab.domain.models import ExecutionAttempt, ExperimentNode, ResearchProject
from scientist_lab.iteration.service import IterationService
from scientist_lab.search.evidence_link import (
    collect_related_evidence_ids,
    diff_gaps,
    extract_open_gaps,
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


def test_diff_gaps_and_extract():
    before = ["unsupported: A", "weak limitation"]
    after = ["unsupported: A", "unsupported: B"]
    resolved, new = diff_gaps(before, after)
    assert resolved == ["weak limitation"]
    assert new == ["unsupported: B"]

    gaps = extract_open_gaps(
        {
            "claims": [
                {
                    "claim_text": "Fusion helps",
                    "support_status": "unsupported",
                },
                {
                    "claim_text": "RGB baseline",
                    "support_status": "supported",
                },
            ]
        },
        [{"limitations": ["Fast Eval only"]}],
    )
    assert gaps[0].startswith("unsupported:")
    assert "Fast Eval only" in gaps


def test_collect_related_evidence_ids():
    ids = collect_related_evidence_ids(
        [
            {"evidence_id": "e1", "source_node_ids": ["n1", "n2"]},
            {"evidence_id": "e2", "source_node_ids": ["n3"]},
            {"evidence_id": "e3", "source_node_ids": ["n2"]},
        ],
        experiment_node_ids=["n2"],
    )
    assert ids == ["e1", "e3"]


def _bootstrap(service: ExperimentService) -> None:
    now = "2026-01-01T00:00:00+00:00"
    service.protocols.create_from_path(EXAMPLES / "rgbt_protocol.json")
    service.set_budget("project_rgbt_003", max_new_nodes=5, max_gpu_hours=10)
    service.repo.upsert_project(
        ResearchProject(
            project_id="project_rgbt_003",
            title="RGB-T",
            research_goal="Tree evidence",
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
                        },
                        "duration_seconds": {
                            "mean": 100.0,
                            "std": 1.0,
                            "min": 99.0,
                            "max": 101.0,
                        },
                    },
                }
            },
            created_at=now,
            updated_at=now,
        )
    )


def _seed_attempts(
    service: ExperimentService,
    *,
    node_id: str,
    map_mean: float,
    input_mode: str = "fusion",
) -> None:
    contract = json.loads(
        (EXAMPLES / "rgbt_formal_fusion_contract.json").read_text(encoding="utf-8")
    )
    contract = dict(contract)
    contract["task_config"] = dict(contract.get("task_config") or {})
    contract["task_config"]["input_mode"] = input_mode
    for i, seed in enumerate((1, 2, 3), start=1):
        value = map_mean + (i - 2) * 0.01
        service.repo.upsert_attempt(
            ExecutionAttempt(
                execution_id=f"exec_{node_id}_{seed}",
                node_id=node_id,
                attempt_index=i,
                runner_profile="local",
                status=JobStatus.COMPLETED,
                image_reference="scientist-rgbt-detection:v2",
                code_version="image:rgbt-detection-v2",
                dataset_version="dataset:rgbt_fast_eval_v1",
                result_json={
                    "metrics": {
                        "primary_metric": "mAP50_95",
                        "metrics": {
                            "mAP50_95": value,
                            "mAP50": value + 0.1,
                            "duration_seconds": 90.0 + i,
                            "peak_gpu_memory_mb": 200.0,
                            "parameter_count": 1500.0,
                        },
                    },
                    "contract": {
                        **contract,
                        "seed": seed,
                        "project_id": "project_rgbt_003",
                        "protocol_id": "protocol_rgbt_001",
                    },
                },
                created_at="2026-01-01T00:00:00+00:00",
                updated_at="2026-01-01T00:00:00+00:00",
            )
        )


def test_tree_advance_links_evidence_and_claim_matrix(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap(service)
    _seed_attempts(service, node_id="rgbt_formal_node_003", map_mean=0.42)

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

    proposed = approved["proposed_node_id"]
    _seed_attempts(service, node_id=proposed, map_mean=0.38, input_mode="rgb")
    node = service.repo.get_node(proposed)
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

    iteration = IterationService(service)
    session = iteration.repo.get_session(approved["iteration_id"])
    assert session is not None
    session.status = "completed"
    session.selected_node_id = proposed
    session.decision_id = "decision_ev_001"
    iteration.repo.save_session(session)

    result = service.tree_advance(created["tree_id"])
    assert result["status"] == "advanced"
    item = result["advanced"][0]
    assert item["action"] == "evaluated"
    assert isinstance(item["evidence_ids"], list)
    assert isinstance(item["resolved_evidence_gaps"], list)
    assert isinstance(item["new_evidence_gaps"], list)

    nodes = service.tree_nodes(created["tree_id"])
    child = next(n for n in nodes if n["experiment_node_id"] == proposed)
    assert child["status"] == "evaluated"
    assert child["evidence_ids"]
    assert child["claim_matrix_path"]

    summary = service.tree_evidence(created["tree_id"])
    assert summary["linked_count"] >= 1
    assert summary["evidence_count"] >= 1
    assert any(
        n["tree_node_id"] == child["tree_node_id"] for n in summary["linked_nodes"]
    )
