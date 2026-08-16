from __future__ import annotations

import json
from pathlib import Path

import pytest

from scientist_lab.agents.context_builder import build_planning_context
from scientist_lab.agents.models import CandidateExperiment, PlanningContext
from scientist_lab.agents.legacy_planner import MockPlanner
from scientist_lab.domain import NodeStage, NodeStatus, NodeType, ProjectStatus
from scientist_lab.domain.models import ExperimentNode, ResearchProject
from scientist_lab.planning.candidate_verifier import (
    CandidateVerifier,
    parameter_fingerprint,
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


def _bootstrap_project(service: ExperimentService) -> None:
    now = "2026-01-01T00:00:00+00:00"
    service.protocols.create_from_path(EXAMPLES / "rgbt_protocol.json")
    service.repo.upsert_project(
        ResearchProject(
            project_id="project_rgbt_003",
            title="RGB-T",
            research_goal="Improve small-object RGB-T detection under protocol.",
            status=ProjectStatus.ACTIVE,
            created_at=now,
            updated_at=now,
        )
    )
    # Only fusion node present → MockPlanner should propose RGB/Thermal ablations.
    service.repo.upsert_node(
        ExperimentNode(
            node_id="rgbt_formal_node_003",
            project_id="project_rgbt_003",
            node_type=NodeType.BASELINE,
            stage=NodeStage.DONE,
            status=NodeStatus.SUCCEEDED,
            depth=0,
            contract_json=json.loads(
                (EXAMPLES / "rgbt_formal_fusion_contract.json").read_text(
                    encoding="utf-8"
                )
            ),
            created_at=now,
            updated_at=now,
        )
    )


def test_planning_context_scrubs_host_paths():
    context = build_planning_context(
        project_id="project_rgbt_003",
        research_goal="g",
        protocol={
            "protocol_id": "protocol_rgbt_001",
            "allowed_variables": ["input_mode", "fusion_method"],
            "host_path": "D:\\secret\\data",
        },
        nodes=[],
        evidence_records=[
            {
                "evidence_id": "evidence_1",
                "host_path": "/home/user/secret",
                "limitations": ["Missing ablation"],
            }
        ],
    )
    assert "host_path" not in context.protocol
    assert context.evidence_records[0].get("host_path") is None
    assert context.context_sha256


def test_verifier_rejects_blocked_and_disallowed_params():
    context = PlanningContext(
        project_id="project_rgbt_003",
        research_goal="g",
        protocol={
            "protocol_id": "p",
            "allowed_variables": ["input_mode", "fusion_method"],
            "fixed_parameters": {"epochs": 5},
        },
        nodes=[{"node_id": "n1"}],
        allowed_parameter_changes=["input_mode", "fusion_method"],
        tested_parameter_fingerprints=[],
        remaining_budget={"max_new_nodes": 3, "max_total_gpu_hours": 10},
    )
    bad = CandidateExperiment(
        candidate_id="candidate_bad",
        parent_node_id="n1",
        title="bad",
        hypothesis="x",
        experiment_type="improve",
        parameter_changes={"environment_key": "evil", "epochs": 99},
        evidence_gap_addressed=["gap"],
    )
    report = CandidateVerifier().verify(bad, context)
    assert report.valid is False
    assert any("environment_key" in issue for issue in report.blocking_issues)


def test_verifier_rejects_duplicate_fingerprint():
    changes = {"input_mode": "rgb", "fusion_method": "none"}
    fp = parameter_fingerprint(changes)
    context = PlanningContext(
        project_id="project_rgbt_003",
        research_goal="g",
        protocol={"allowed_variables": ["input_mode", "fusion_method"]},
        nodes=[{"node_id": "n1"}],
        allowed_parameter_changes=["input_mode", "fusion_method"],
        tested_parameter_fingerprints=[fp],
        remaining_budget={"max_new_nodes": 3},
    )
    cand = CandidateExperiment(
        candidate_id="candidate_dup",
        parent_node_id="n1",
        title="dup",
        hypothesis="x",
        experiment_type="ablation",
        parameter_changes=changes,
        evidence_gap_addressed=["gap"],
    )
    report = CandidateVerifier().verify(cand, context)
    assert report.valid is False
    assert any("duplicate" in issue for issue in report.blocking_issues)


def test_mock_planner_emits_schema_candidates():
    context = build_planning_context(
        project_id="project_rgbt_003",
        research_goal="g",
        protocol=json.loads((EXAMPLES / "rgbt_protocol.json").read_text(encoding="utf-8")),
        nodes=[
            type(
                "N",
                (),
                {
                    "node_id": "rgbt_formal_node_003",
                    "project_id": "project_rgbt_003",
                    "status": "succeeded",
                    "hypothesis": "fusion",
                    "contract_json": json.loads(
                        (EXAMPLES / "rgbt_formal_fusion_contract.json").read_text(
                            encoding="utf-8"
                        )
                    ),
                },
            )()
        ],
        claim_support_matrix={
            "claims": [
                {
                    "claim_text": "Fusion improves AP_small",
                    "support_status": "partially_supported",
                    "reason": "Missing controlled ablation",
                }
            ]
        },
    )
    output = MockPlanner().plan(context)
    assert output.candidates
    assert len(output.candidates) <= 3
    for cand in output.candidates:
        assert set(cand.parameter_changes).issubset({"input_mode", "fusion_method"})
        assert cand.evidence_gap_addressed


def test_plan_next_cli_path(tmp_path: Path):
    service = _service(tmp_path)
    _bootstrap_project(service)
    result = service.plan_next("project_rgbt_003", protocol_id="protocol_rgbt_001")
    assert result["plan_id"].startswith("plan_")
    assert result["model_name"] == "mock-planner-v1"
    assert result["valid_candidate_count"] >= 1
    listed = service.list_plans(project_id="project_rgbt_003")
    assert len(listed) == 1
    shown = service.show_plan(result["plan_id"])
    assert shown["plan_id"] == result["plan_id"]
    assert shown["context_sha256"]
    assert shown["output_sha256"]


def test_empty_budget_stops():
    context = build_planning_context(
        project_id="project_rgbt_003",
        research_goal="g",
        protocol={"allowed_variables": ["input_mode", "fusion_method"]},
        nodes=[
            type(
                "N",
                (),
                {
                    "node_id": "n1",
                    "project_id": "project_rgbt_003",
                    "status": "succeeded",
                    "hypothesis": None,
                    "contract_json": {
                        "parameters": {"input_mode": "rgbt", "fusion_method": "early_concat"}
                    },
                },
            )()
        ],
        remaining_budget={"max_new_nodes": 0, "max_total_gpu_hours": 0},
    )
    output = MockPlanner().plan(context)
    assert output.stop_recommended is True
    assert output.candidates == []
