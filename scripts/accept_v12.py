"""v1.2 acceptance: ReportContext → ResearchReport → Audit Bundle."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scientist_lab.domain import JobStatus, NodeStage, NodeStatus, NodeType, ProjectStatus
from scientist_lab.domain.models import ExecutionAttempt, ExperimentNode, ResearchProject
from scientist_lab.evidence.models import ClaimSupportMatrix, ScientificClaim
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings


def _check(name: str, ok: bool, detail: str = "") -> dict:
    return {"name": name, "ok": bool(ok), "detail": detail}


def seed(service: ExperimentService) -> str:
    now = "2026-07-21T00:00:00+00:00"
    try:
        service.create_protocol(ROOT / "examples" / "rgbt_protocol.json")
    except Exception:
        pass
    service.repo.upsert_project(
        ResearchProject(
            project_id="project_rgbt_003",
            title="RGB-T v1.2 acceptance",
            research_goal="Evidence report and audit bundle",
            status=ProjectStatus.ACTIVE,
            created_at=now,
            updated_at=now,
        )
    )
    contract = json.loads(
        (ROOT / "examples" / "rgbt_formal_fusion_contract.json").read_text(
            encoding="utf-8"
        )
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
            execution_id="exec_accept_v12",
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
    created = service.tree_create(
        "project_rgbt_003",
        root_node_id="rgbt_formal_node_003",
        protocol_id="protocol_rgbt_001",
        max_depth=3,
        max_nodes=8,
        max_children=3,
    )
    return created["tree_id"]


def main() -> int:
    accept_root = ROOT / "outputs" / "_accept_v12"
    settings = Settings(
        project_root=ROOT,
        db_path=accept_root / "accept.db",
        runtime_dir=accept_root / "runtime",
        outputs_dir=accept_root / "outputs",
        experiment_app_dir=ROOT / "experiment_app",
    ).resolve()
    if settings.db_path.exists():
        settings.db_path.unlink()
    service = ExperimentService(settings=settings)
    tree_id = seed(service)
    results: list[dict] = []

    context = service.build_report_context(
        "project_rgbt_003", tree_id=tree_id, protocol_id="protocol_rgbt_001"
    )
    results.append(
        _check(
            "report-context",
            bool(context.get("context_sha256")) and bool(context.get("nodes")),
            f"sha={context.get('context_sha256')}",
        )
    )

    report = service.build_report(
        "project_rgbt_003", tree_id=tree_id, protocol_id="protocol_rgbt_001"
    )
    results.append(
        _check(
            "report-build",
            bool(report.get("report_id"))
            and bool(report.get("metrics_table"))
            and bool(report.get("conclusions")),
            f"report_id={report.get('report_id')}",
        )
    )
    results.append(
        _check(
            "report-markdown",
            Path(str(report.get("markdown_path") or "")).is_file(),
            str(report.get("markdown_path")),
        )
    )

    shown = service.show_report(report["report_id"])
    results.append(
        _check(
            "report-show",
            shown.get("report_id") == report["report_id"],
            shown.get("report_id"),
        )
    )

    verified = service.verify_report(report["report_id"])
    results.append(
        _check(
            "report-verify",
            bool(verified.get("valid")),
            json.dumps(verified.get("blocking_issues") or [], ensure_ascii=False),
        )
    )

    # Claim gate: no conclusion should use forbidden SOTA phrasing as standalone words.
    import re

    forbidden = False
    for item in report.get("conclusions") or []:
        text = str(item.get("text") or "").lower()
        if re.search(r"(?<![a-z0-9_])sota(?![a-z0-9_])", text):
            forbidden = True
        if "已证明" in text or "优于现有方法" in text:
            forbidden = True
    results.append(_check("claim-gate-no-sota-phrasing", not forbidden, "ok"))

    bundle = service.build_audit(
        "project_rgbt_003",
        tree_id=tree_id,
        protocol_id="protocol_rgbt_001",
        report_id=report["report_id"],
    )
    root = Path(str(bundle.get("root_dir") or ""))
    results.append(
        _check(
            "audit-build",
            bool(bundle.get("bundle_id"))
            and (root / "audit_bundle.json").is_file()
            and (root / "manifests" / "reproducibility_manifest.json").is_file(),
            f"bundle={bundle.get('bundle_id')}",
        )
    )

    audit_ok = service.verify_audit(bundle["bundle_id"])
    results.append(
        _check(
            "audit-verify",
            bool(audit_ok.get("valid")),
            json.dumps(audit_ok.get("blocking_issues") or [], ensure_ascii=False),
        )
    )

    exported = service.export_audit(
        bundle["bundle_id"], accept_root / "exports"
    )
    results.append(
        _check(
            "audit-export",
            Path(str(exported.get("exported_to") or "")).is_dir(),
            str(exported.get("exported_to")),
        )
    )

    ok_all = all(item["ok"] for item in results)
    out_dir = settings.outputs_dir / "project_rgbt_003" / "acceptance"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_out = {
        "version": "v1.2",
        "ok": ok_all,
        "tree_id": tree_id,
        "report_id": report.get("report_id"),
        "bundle_id": bundle.get("bundle_id"),
        "steps": results,
        "passed": sum(1 for r in results if r["ok"]),
        "total": len(results),
    }
    path = out_dir / "v12_acceptance_report.json"
    path.write_text(json.dumps(report_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report_out, ensure_ascii=False, indent=2))
    print(f"\nWrote {path}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
