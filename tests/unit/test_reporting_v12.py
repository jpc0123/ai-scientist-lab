from __future__ import annotations

import json
from pathlib import Path

from scientist_lab.domain import JobStatus, NodeStage, NodeStatus, NodeType, ProjectStatus
from scientist_lab.domain.models import ExecutionAttempt, ExperimentNode, ResearchProject
from scientist_lab.reporting import (
    build_result_tables,
    generate_research_report,
    verify_research_report,
)
from scientist_lab.reporting.models import ReportConclusion, ResearchReport
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


def _bootstrap(service: ExperimentService) -> str:
    now = "2026-01-01T00:00:00+00:00"
    service.protocols.create_from_path(EXAMPLES / "rgbt_protocol.json")
    service.repo.upsert_project(
        ResearchProject(
            project_id="project_rgbt_003",
            title="RGB-T",
            research_goal="v1.2 reporting",
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
    service.repo.upsert_attempt(
        ExecutionAttempt(
            execution_id="exec_report_v12",
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
                    "metrics": {"mAP50_95": 0.42, "duration_seconds": 100.0},
                },
                "contract": {**contract, "seed": 1},
            },
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
        max_children=3,
    )
    from scientist_lab.evidence.models import ClaimSupportMatrix, ScientificClaim

    service.evidence._repo.upsert_claim_matrix(
        ClaimSupportMatrix(
            project_id="project_rgbt_003",
            protocol_id="protocol_rgbt_001",
            claims=[
                ScientificClaim(
                    claim_id="claim_fast_eval_ap_small",
                    project_id="project_rgbt_003",
                    claim_text="Fusion improves AP_small under Fast Eval",
                    claim_type="exploratory_metric_comparison",
                    support_status="partially_supported",
                    supporting_evidence_ids=[],
                    limitations=["Fast Eval subset only."],
                    reason="Directional only.",
                ),
                ScientificClaim(
                    claim_id="claim_sota",
                    project_id="project_rgbt_003",
                    claim_text="Method is SOTA on full RGBT-Tiny",
                    claim_type="sota",
                    support_status="blocked",
                    supporting_evidence_ids=[],
                    limitations=["Blocked by claim gate."],
                    reason="Insufficient evidence for SOTA.",
                ),
            ],
            evidence_ids=[],
        )
    )
    return created["tree_id"]


def test_result_tables_and_report_build(tmp_path: Path):
    service = _service(tmp_path)
    tree_id = _bootstrap(service)
    context = service.reporting.build_context(
        "project_rgbt_003", tree_id=tree_id, protocol_id="protocol_rgbt_001"
    )
    tables = build_result_tables(context)
    assert tables.metrics
    assert tables.key_path
    assert tables.claims

    report = generate_research_report(context, tables=tables)
    assert report.conclusions
    for item in report.conclusions:
        assert item.limitations is not None
    verified = verify_research_report(report)
    assert verified["valid"] is True


def test_report_verifier_blocks_forbidden_phrasing():
    report = ResearchReport(
        report_id="report_bad",
        project_id="project_rgbt_003",
        conclusions=[
            ReportConclusion(
                conclusion_id="c1",
                text="Fusion 已证明在完整 RGBT-Tiny 上优于现有方法。",
                claim_id="claim_sota",
                support_status="blocked",
                evidence_ids=[],
                limitations=["x"],
                strength="strong",
            )
        ],
    )
    result = verify_research_report(report)
    assert result["valid"] is False
    assert result["blocking_issues"]


def test_report_and_audit_cli_flow(tmp_path: Path):
    service = _service(tmp_path)
    tree_id = _bootstrap(service)

    built = service.build_report(
        "project_rgbt_003", tree_id=tree_id, protocol_id="protocol_rgbt_001"
    )
    assert built["report_id"]
    assert built["metrics_table"]
    assert Path(built["json_path"]).is_file()
    assert Path(built["markdown_path"]).is_file()

    shown = service.show_report(built["report_id"])
    assert shown["report_id"] == built["report_id"]

    verified = service.verify_report(built["report_id"])
    assert verified["valid"] is True

    bundle = service.build_audit(
        "project_rgbt_003",
        tree_id=tree_id,
        protocol_id="protocol_rgbt_001",
        report_id=built["report_id"],
    )
    assert bundle["bundle_id"]
    root = Path(bundle["root_dir"])
    assert (root / "report" / "research_report.md").is_file()
    assert (root / "manifests" / "reproducibility_manifest.json").is_file()
    assert (root / "tree" / "tree.json").is_file()
    assert (root / "audit_bundle.json").is_file()

    check = service.verify_audit(bundle["bundle_id"])
    assert check["valid"] is True

    exported = service.export_audit(bundle["bundle_id"], tmp_path / "exports")
    assert Path(exported["exported_to"]).is_dir()
