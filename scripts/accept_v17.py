"""v1.7 acceptance: Web Console API contract + controlled patch/report surfaces."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scientist_lab.api.app import create_app
from scientist_lab.patching.service import build_mock_unified_diff
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings


def _check(name: str, ok: bool, detail: str = "") -> dict:
    return {"name": name, "ok": bool(ok), "detail": detail}


def main() -> int:
    accept_root = ROOT / "outputs" / "_accept_v17"
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

    # 1 health
    health = client.get("/api/v1/health")
    results.append(
        _check(
            "api-health",
            health.status_code == 200 and health.json().get("ok") is True,
            f"status={health.status_code}",
        )
    )

    # 2 dashboard summary
    summary = client.get("/api/v1/system/summary")
    results.append(
        _check(
            "dashboard-summary",
            summary.status_code == 200 and "project_count" in summary.json(),
            f"keys={sorted(summary.json().keys())[:8]}",
        )
    )

    # 3–4 executions list/detail shape (empty ok)
    executions = client.get("/api/v1/executions", params={"limit": 5})
    results.append(
        _check(
            "executions-list",
            executions.status_code == 200 and "items" in executions.json(),
            f"count={executions.json().get('count')}",
        )
    )
    missing_exec = client.get("/api/v1/executions/does-not-exist")
    results.append(
        _check(
            "execution-404-envelope",
            missing_exec.status_code == 404
            and missing_exec.json().get("error", {}).get("code") == "not_found",
            str(missing_exec.json()),
        )
    )

    # 5 trees list + mermaid 404 envelope
    trees = client.get("/api/v1/trees")
    results.append(
        _check(
            "trees-list",
            trees.status_code == 200 and "items" in trees.json(),
            f"total={trees.json().get('total')}",
        )
    )
    mermaid_404 = client.get("/api/v1/trees/missing/mermaid")
    results.append(
        _check(
            "tree-mermaid-not-found",
            mermaid_404.status_code == 404,
            mermaid_404.json().get("error", {}).get("code", ""),
        )
    )

    # 6 plans list
    plans = client.get("/api/v1/plans")
    results.append(
        _check(
            "plans-list",
            plans.status_code == 200 and "items" in plans.json(),
            f"total={plans.json().get('total')}",
        )
    )

    # 7–11 patch controlled lifecycle
    proposed = service.patches.propose_mock(
        "project_web_v17",
        unified_diff=build_mock_unified_diff(
            relative_path=(
                "experiment_apps/rgbt_detection_real/adapters/accept_v17_note.md"
            )
        ),
    )
    patch_id = proposed["patch_id"]

    blocked = client.post(f"/api/v1/patches/{patch_id}/apply-sandbox")
    results.append(
        _check(
            "unapproved-patch-cannot-apply",
            blocked.status_code == 409,
            blocked.json().get("error", {}).get("message", ""),
        )
    )

    shown = client.get(f"/api/v1/patches/{patch_id}")
    results.append(
        _check(
            "patch-diff-viewable",
            shown.status_code == 200
            and "unified_diff" in shown.json()
            and shown.json().get("can_apply_main") is False,
            f"status={shown.json().get('status')}",
        )
    )

    approved = client.post(
        f"/api/v1/patches/{patch_id}/approve", json={"reason": "accept_v17"}
    )
    results.append(
        _check(
            "patch-approve",
            approved.status_code == 200 and approved.json().get("status") == "approved",
            approved.json().get("status", ""),
        )
    )

    applied = client.post(f"/api/v1/patches/{patch_id}/apply-sandbox")
    results.append(
        _check(
            "approved-patch-apply-sandbox",
            applied.status_code == 200
            and applied.json().get("status") == "applied_sandbox"
            and applied.json().get("can_apply_main") is False,
            applied.json().get("status", ""),
        )
    )

    tested = client.post(
        f"/api/v1/patches/{patch_id}/test-sandbox",
        json={"profile": "mock_experiment"},
    )
    results.append(
        _check(
            "sandbox-test-viewable",
            tested.status_code == 200
            and (tested.json().get("sandbox_tests") or {}).get("ok") is True,
            f"ok={(tested.json().get('sandbox_tests') or {}).get('ok')}",
        )
    )

    evidence = client.post(
        f"/api/v1/patches/{patch_id}/record-evidence",
        json={"require_tests": True},
    )
    results.append(
        _check(
            "patch-evidence-recorded",
            evidence.status_code == 200
            and evidence.json().get("status") == "evidence_recorded",
            evidence.json().get("status", ""),
        )
    )

    merged = client.post(
        f"/api/v1/patches/{patch_id}/decide-merge",
        json={"decision": "merge", "reason": "intent only"},
    )
    results.append(
        _check(
            "merge-intent-no-main",
            merged.status_code == 200
            and merged.json().get("status") == "merged"
            and merged.json().get("applied_main") is False
            and merged.json().get("can_apply_main") is False,
            merged.json().get("warning", ""),
        )
    )

    # 12–13 evidence / claims / reports / audits list shapes
    for path, name in (
        ("/api/v1/evidence", "evidence-list"),
        ("/api/v1/claims", "claims-list"),
        ("/api/v1/reports", "reports-list"),
        ("/api/v1/audits", "audits-list"),
    ):
        resp = client.get(path)
        results.append(
            _check(
                name,
                resp.status_code == 200 and "items" in resp.json(),
                f"total={resp.json().get('total')}",
            )
        )

    # 14 path policy
    policy = client.get("/api/v1/system/path-policy")
    results.append(
        _check(
            "path-policy",
            policy.status_code == 200
            and bool(policy.json().get("allowed_prefixes")),
            f"allowed={len(policy.json().get('allowed_prefixes') or [])}",
        )
    )

    # 15 openapi has confirm-critical routes
    schema = client.get("/openapi.json").json()
    paths = schema.get("paths") or {}
    results.append(
        _check(
            "openapi-has-v1-core",
            "/api/v1/health" in paths
            and "/api/v1/system/summary" in paths
            and "/api/v1/patches/{patch_id}/apply-sandbox" in paths
            and "/api/v1/patches/{patch_id}/record-evidence" in paths
            and "/api/v1/reports/{report_id}/markdown" in paths,
            f"paths={len(paths)}",
        )
    )

    # 16 no shell endpoint surface
    shellish = [
        p
        for p in paths
        if "shell" in p.lower() or "terminal" in p.lower() or "exec-cmd" in p.lower()
    ]
    results.append(
        _check(
            "no-arbitrary-shell-api",
            len(shellish) == 0,
            f"found={shellish}",
        )
    )

    # 17 frontend build artifact exists (optional soft if dist missing)
    dist_index = ROOT / "web" / "dist" / "index.html"
    results.append(
        _check(
            "frontend-dist-present",
            dist_index.is_file(),
            str(dist_index),
        )
    )

    passed = sum(1 for r in results if r["ok"])
    total = len(results)
    report_doc = {
        "version": "v1.7.0",
        "title": "Web Console MVP API + controlled surfaces",
        "passed": passed,
        "total": total,
        "all_ok": passed == total,
        "checks": results,
        "notes": [
            "Frontend confirmation dialogs are UI-only; backend enforces state.",
            "decide-merge records intent only; never modifies main workspace.",
            "Run `npm run build` in web/ before this script for dist check.",
        ],
    }
    out_dir = ROOT / "docs" / "acceptance" / "v1.7"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "v17_acceptance_report.json"
    report_path.write_text(
        json.dumps(report_doc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report_doc, ensure_ascii=False, indent=2))
    print(f"\nWrote {report_path}")
    print(f"Acceptance: {passed}/{total}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
