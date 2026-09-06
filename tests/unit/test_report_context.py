from __future__ import annotations

import json
from pathlib import Path

from scientist_lab.domain import JobStatus, NodeStage, NodeStatus, NodeType, ProjectStatus
from scientist_lab.domain.models import ExecutionAttempt, ExperimentNode, ResearchProject
from scientist_lab.reporting import build_report_context, report_context_sha256
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
            research_goal="Report context",
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
                    "seed_count": 1,
                    "aggregate_metrics": {
                        "mAP50_95": {"mean": 0.42, "std": 0.0, "min": 0.42, "max": 0.42}
                    },
                }
            },
            created_at=now,
            updated_at=now,
        )
    )
    service.repo.upsert_attempt(
        ExecutionAttempt(
            execution_id="exec_report_001",
            node_id="rgbt_formal_node_003",
            attempt_index=1,
            runner_profile="local",
            status=JobStatus.COMPLETED,
            image_reference="scientist-rgbt-detection:v2",
            code_version="image:rgbt-detection-v2",
            dataset_version="dataset:rgbt_fast_eval_v1",
            result_json={
                "metrics": {
                    "primary_metric": "mAP50_95",
                    "metrics": {"mAP50_95": 0.42, "duration_seconds": 10.0},
                },
                "contract": {**contract, "seed": 1},
            },
            created_at=now,
            updated_at=now,
        )
    )


def test_build_report_context_pure():
    ctx = build_report_context(
        project_id="project_rgbt_003",
        research_goal="g",
        protocol={"protocol_id": "protocol_rgbt_001", "title": "p"},
        protocol_id="protocol_rgbt_001",
        evidence_records=[
            {
                "evidence_id": "e1",
                "limitations": ["Fast Eval only"],
                "evidence_strength": "weak",
            }
        ],
        claim_support_matrix={
            "claims": [
                {
                    "claim_id": "c1",
                    "claim_text": "SOTA",
                    "support_status": "blocked",
                    "reason": "blocked by claim gate",
                    "limitations": ["Cannot claim SOTA"],
                }
            ]
        },
    )
    assert ctx.project_id == "project_rgbt_003"
    assert ctx.context_sha256
    assert "Fast Eval only" in ctx.limitations
    assert any("blocked" in g for g in ctx.open_evidence_gaps)
    again = report_context_sha256(ctx)
    assert again == ctx.context_sha256


def test_build_report_context_from_service(tmp_path: Path):
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
    payload = service.build_report_context(
        "project_rgbt_003",
        tree_id=created["tree_id"],
        protocol_id="protocol_rgbt_001",
    )
    assert payload["project_id"] == "project_rgbt_003"
    assert payload["tree_id"] == created["tree_id"]
    assert payload["protocol_id"] == "protocol_rgbt_001"
    assert payload["nodes"]
    assert payload["executions"]
    assert payload["context_sha256"]
    assert payload["builder_version"] == "v1.2.1"
    assert payload["key_path"]
    # stable hash on second build (ignore built_at)
    payload2 = service.build_report_context(
        "project_rgbt_003",
        tree_id=created["tree_id"],
        protocol_id="protocol_rgbt_001",
    )
    assert payload2["context_sha256"] == payload["context_sha256"]
