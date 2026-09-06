"""v1.8 acceptance: workspace views + release package + report/audit build-export APIs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scientist_lab.api.app import create_app
from scientist_lab.domain import (
    JobStatus,
    NodeStage,
    NodeStatus,
    NodeType,
    ProjectStatus,
)
from scientist_lab.domain.models import ExecutionAttempt, ExperimentNode, ResearchProject
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings


def _check(name: str, ok: bool, detail: str = "") -> dict:
    return {"name": name, "ok": bool(ok), "detail": detail}


def _bootstrap(service: ExperimentService) -> None:
    examples = ROOT / "examples"
    now = "2026-01-01T00:00:00+00:00"
    service.protocols.create_from_path(examples / "rgbt_protocol.json")
    service.repo.upsert_project(
        ResearchProject(
            project_id="project_rgbt_003",
            title="RGB-T",
            research_goal="v1.8 acceptance",
            status=ProjectStatus.ACTIVE,
            created_at=now,
            updated_at=now,
        )
    )
    contract = json.loads(
        (examples / "rgbt_formal_fusion_contract.json").read_text(encoding="utf-8")
    )
    service.repo.upsert_node(
        ExperimentNode(
            node_id="rgbt_formal_node_v18",
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
                        "mAP50_95": {
                            "mean": 0.41,
                            "std": 0.0,
                            "min": 0.41,
                            "max": 0.41,
                        }
                    },
                }
            },
            created_at=now,
            updated_at=now,
        )
    )
    service.repo.upsert_attempt(
        ExecutionAttempt(
            execution_id="exec_accept_v18",
            node_id="rgbt_formal_node_v18",
            attempt_index=1,
            runner_profile="local",
            status=JobStatus.COMPLETED,
            image_reference="scientist-rgbt-detection:v2",
            code_version="image:rgbt-detection-v2",
            dataset_version="dataset:rgbt_fast_eval_v1",
            result_json={
                "metrics": {
                    "primary_metric": "mAP50_95",
                    "metrics": {"mAP50_95": 0.41},
                },
                "contract": {**contract, "seed": 1},
            },
            created_at=now,
            updated_at=now,
        )
    )


def main() -> int:
    accept_root = ROOT / "outputs" / "_accept_v18"
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
    client = TestClient(create_app(service=service))
    results: list[dict] = []

    # 1 health reports v1.8
    health = client.get("/api/v1/health")
    results.append(
        _check(
            "api-health-v18",
            health.status_code == 200
            and str(health.json().get("version", "")).startswith("v1.8"),
            f"version={health.json().get('version')}",
        )
    )

    # 2 workspace summary never writable main
    ws = client.get("/api/v1/workspaces/summary")
    body = ws.json() if ws.status_code == 200 else {}
    results.append(
        _check(
            "workspace-main-readonly",
            ws.status_code == 200
            and body.get("can_write_main") is False
            and body.get("main_workspace_modified") is False,
            str(body.get("can_write_main")),
        )
    )

    # 3 release create → freeze
    created = client.post(
        "/api/v1/releases",
        json={"project_id": "project_accept_v18", "title": "accept freeze"},
    )
    release_id = created.json().get("release_id") if created.status_code == 200 else ""
    frozen = client.post(f"/api/v1/releases/{release_id}/freeze", json={"notes": "ok"})
    results.append(
        _check(
            "release-create-freeze",
            created.status_code == 200
            and frozen.status_code == 200
            and frozen.json().get("status") == "frozen"
            and frozen.json().get("main_workspace_modified") is False,
            f"create={created.status_code} freeze={frozen.status_code}",
        )
    )

    # 4 bootstrap + report/audit build + export linked to release
    try:
        _bootstrap(service)
        report = client.post(
            "/api/v1/reports/build",
            json={"project_id": "project_rgbt_003"},
        )
        report_id = report.json().get("report_id") if report.status_code == 200 else ""
        audit = client.post(
            "/api/v1/audits/build",
            json={"project_id": "project_rgbt_003", "report_id": report_id},
        )
        bundle_id = audit.json().get("bundle_id") if audit.status_code == 200 else ""
        rel2 = client.post(
            "/api/v1/releases",
            json={
                "project_id": "project_rgbt_003",
                "title": "with audit",
                "report_id": report_id,
                "audit_bundle_id": bundle_id,
            },
        ).json()
        client.post(f"/api/v1/releases/{rel2['release_id']}/freeze", json={})
        export_dir = accept_root / "export"
        exported = client.post(
            f"/api/v1/audits/{bundle_id}/export",
            json={
                "output_dir": str(export_dir),
                "release_id": rel2["release_id"],
            },
        )
        exp_body = exported.json() if exported.status_code == 200 else {}
        results.append(
            _check(
                "report-audit-build-export",
                report.status_code == 200
                and audit.status_code == 200
                and exported.status_code == 200
                and Path(exp_body.get("exported_to") or "").is_dir()
                and exp_body.get("release", {}).get("status") == "exported"
                and exp_body.get("release", {}).get("main_workspace_modified")
                is False,
                f"report={report.status_code} audit={audit.status_code} "
                f"export={exported.status_code}",
            )
        )
    except Exception as exc:  # noqa: BLE001
        results.append(_check("report-audit-build-export", False, str(exc)))

    # 5 openapi contains release + build paths
    schema = client.get("/openapi.json").json()
    paths = schema.get("paths") or {}
    needed = [
        "/api/v1/reports/build",
        "/api/v1/audits/build",
        "/api/v1/audits/{bundle_id}/export",
        "/api/v1/releases",
        "/api/v1/workspaces/summary",
    ]
    missing = [p for p in needed if p not in paths]
    results.append(
        _check("openapi-v18-paths", not missing, f"missing={missing}")
    )

    # 6 chat workspace page exists in frontend source
    chat_page = ROOT / "web" / "src" / "pages" / "ChatWorkspacePage.tsx"
    results.append(
        _check(
            "frontend-chat-workspace",
            chat_page.is_file(),
            str(chat_page),
        )
    )

    # 7 discard release does not write main
    discarded = client.post(
        f"/api/v1/releases/{release_id}/discard",
        json={"reason": "accept cleanup"},
    )
    results.append(
        _check(
            "release-discard-safe",
            discarded.status_code == 200
            and discarded.json().get("status") == "discarded"
            and discarded.json().get("can_write_main") is False,
            f"status={discarded.json().get('status')}",
        )
    )

    # 8 list releases page shape
    listed = client.get("/api/v1/releases", params={"limit": 10})
    results.append(
        _check(
            "releases-list-page",
            listed.status_code == 200 and "items" in listed.json(),
            f"total={listed.json().get('total')}",
        )
    )

    out_dir = ROOT / "docs" / "acceptance" / "v1.8"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "v18_acceptance_report.json"
    payload = {
        "version": "v1.8.0",
        "passed": sum(1 for r in results if r["ok"]),
        "total": len(results),
        "results": results,
    }
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "version_manifest.json").write_text(
        json.dumps(
            {
                "version": "v1.8.0",
                "tag": "v1.8.0",
                "branch": "feat/v1.8-workspace-release",
                "baseline_tag": "v1.7.0",
                "title": "Scientist Lab Workspace & Release Management",
                "principles": [
                    "Export-only releases; no main workspace writes",
                    "No git commit/push from release freeze",
                    "Report/Audit build-export via /api/v1",
                    "Patch merge remains intent-only until v1.9",
                ],
                "completed_subversions": [
                    "v1.8.1",
                    "v1.8.2",
                    "frontend-chat-workspace",
                ],
                "acceptance_script": "scripts/accept_v18.py",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    failed = [r for r in results if not r["ok"]]
    if failed:
        print(f"FAILED {len(failed)}/{len(results)}", file=sys.stderr)
        return 1
    print(f"PASSED {len(results)}/{len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
