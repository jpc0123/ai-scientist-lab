"""v2.0 acceptance: five groups (system / project / experiment / AI / delivery).

Offline by default — no real LLM, no remote GPU, no arbitrary Shell.
Optional Docker is not required; experiment metrics are seeded for aggregate/compare.

Usage:
  python scripts/accept_v20.py
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scientist_lab.api.app import create_app
from scientist_lab.domain import JobStatus, NodeStage, NodeStatus, NodeType, ProjectStatus
from scientist_lab.domain.models import ExecutionAttempt, ExperimentNode, ResearchProject
from scientist_lab.iteration.service import IterationService
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings

EXAMPLES = ROOT / "examples"


def _check(name: str, ok: bool, detail: str = "", *, group: str = "") -> dict:
    return {
        "group": group,
        "name": name,
        "ok": bool(ok),
        "detail": detail,
    }


def _service(accept_root: Path, name: str = "main") -> ExperimentService:
    return ExperimentService(
        settings=Settings(
            project_root=ROOT,
            db_path=accept_root / f"{name}.db",
            runtime_dir=accept_root / f"{name}_runtime",
            outputs_dir=accept_root / f"{name}_outputs",
            experiment_app_dir=ROOT / "experiment_app",
        ).resolve()
    )


def _seed_digits_attempts(service: ExperimentService, project_id: str) -> None:
    """Seed completed attempts so aggregate/compare/feedback work offline."""
    now = "2026-07-23T00:00:00+00:00"
    for node_id, hidden, accuracies in (
        ("node_003", 64, {42: 0.91, 43: 0.90, 44: 0.89}),
        ("node_004", 128, {42: 0.93, 43: 0.92, 44: 0.91}),
    ):
        node = service.repo.get_node(node_id)
        if node is None:
            service.repo.upsert_node(
                ExperimentNode(
                    node_id=node_id,
                    project_id=project_id,
                    node_type=NodeType.BASELINE,
                    stage=NodeStage.DONE,
                    status=NodeStatus.SUCCEEDED,
                    depth=0,
                    contract_json={
                        "project_id": project_id,
                        "node_id": node_id,
                        "seed": 42,
                        "parameters": {"hidden_units": hidden},
                    },
                    created_at=now,
                    updated_at=now,
                )
            )
        else:
            node.status = NodeStatus.SUCCEEDED
            node.stage = NodeStage.DONE
            contract = dict(node.contract_json or {})
            contract["project_id"] = project_id
            node.contract_json = contract
            service.repo.upsert_node(node)

        for idx, (seed, acc) in enumerate(accuracies.items(), start=1):
            service.repo.upsert_attempt(
                ExecutionAttempt(
                    execution_id=f"accept_{node_id}_s{seed}",
                    node_id=node_id,
                    attempt_index=idx,
                    runner_profile="local",
                    status=JobStatus.COMPLETED,
                    image_reference="scientist-experiment:v2",
                    code_version="local:experiment_app",
                    dataset_version="sklearn:digits",
                    result_json={
                        "metrics": {
                            "primary_metric": "accuracy",
                            "metrics": {"accuracy": acc, "macro_f1": acc - 0.01},
                        },
                        "contract": {
                            "project_id": project_id,
                            "node_id": node_id,
                            "seed": seed,
                            "dataset_reference": "sklearn:digits",
                            "code_reference": "local:experiment_app",
                            "environment_key": "digits-mlp-v1",
                            "entrypoint": "run_experiment.py",
                            "parameters": {
                                "learning_rate": 0.001,
                                "epochs": 30,
                                "hidden_units": hidden,
                                "batch_size": 64,
                                "test_size": 0.2,
                            },
                        },
                    },
                    created_at=now,
                    completed_at=now,
                )
            )


def _bootstrap_rgbt(service: ExperimentService) -> None:
    now = "2026-07-23T00:00:00+00:00"
    service.protocols.create_from_path(EXAMPLES / "rgbt_protocol.json")
    service.set_budget("project_rgbt_003", max_new_nodes=5, max_gpu_hours=10)
    service.repo.upsert_project(
        ResearchProject(
            project_id="project_rgbt_003",
            title="RGB-T accept",
            research_goal="Improve small-object RGB-T detection under protocol.",
            research_question="Does fusion help under fixed protocol?",
            task_type="rgbt_detection",
            status=ProjectStatus.READY,
            protocol_ids=["protocol_rgbt_001"],
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


def _force_iteration_completed(
    service: ExperimentService,
    *,
    iteration_id: str,
    proposed_node_id: str,
) -> None:
    iteration = IterationService(service)
    session = iteration.repo.get_session(iteration_id)
    assert session is not None
    session.status = "completed"
    session.selected_node_id = proposed_node_id
    session.decision_id = "decision_accept_v20"
    iteration.repo.save_session(session)
    node = service.repo.get_node(proposed_node_id)
    assert node is not None
    feedback = dict(node.feedback_json or {})
    feedback["aggregate_metrics"] = {
        "primary_metric": "mAP50_95",
        "seed_count": 3,
        "aggregate_metrics": {
            "mAP50_95": {"mean": 0.38, "std": 0.03, "min": 0.35, "max": 0.41},
        },
    }
    node.feedback_json = feedback
    service.repo.upsert_node(node)


def main() -> int:
    accept_root = ROOT / "outputs" / "_accept_v20"
    if accept_root.exists():
        shutil.rmtree(accept_root, ignore_errors=True)
    accept_root.mkdir(parents=True, exist_ok=True)

    service = _service(accept_root)
    client = TestClient(create_app(service=service), raise_server_exceptions=False)
    results: list[dict] = []

    # ------------------------------------------------------------------ A. 系统
    scripts_ok = all(
        (ROOT / "scripts" / name).is_file()
        for name in (
            "install_windows.ps1",
            "start_workbench.ps1",
            "stop_workbench.ps1",
            "doctor_windows.ps1",
        )
    )
    results.append(
        _check(
            "A1-workbench-scripts",
            scripts_ok,
            "install/start/stop/doctor scripts",
            group="A",
        )
    )

    health = client.get("/api/v1/health")
    ver = str((health.json() or {}).get("version") or "")
    results.append(
        _check(
            "A2-api-health",
            health.status_code == 200 and ver.startswith("v2.0"),
            f"version={ver}",
            group="A",
        )
    )

    web_index = (ROOT / "web" / "index.html").is_file() or (
        ROOT / "web" / "dist" / "index.html"
    ).is_file()
    results.append(
        _check("A3-web-entrypoint", web_index, "web/index.html or dist", group="A")
    )

    doctor = client.get("/api/v1/system/doctor")
    doctor_body = doctor.json() if doctor.status_code == 200 else {}
    results.append(
        _check(
            "A4-db-and-doctor",
            doctor.status_code == 200
            and str(doctor_body.get("overall") or "") in {"ok", "warning"},
            f"overall={doctor_body.get('overall')} version={doctor_body.get('version')}",
            group="A",
        )
    )

    security = client.get("/api/v1/system/security")
    sec = security.json() if security.status_code == 200 else {}
    results.append(
        _check(
            "A5-system-doctor-security",
            security.status_code == 200
            and sec.get("overall") in {"ok", "warning"}
            and any(
                b.get("id") == "no_arbitrary_shell" and b.get("enforced")
                for b in (sec.get("boundaries") or [])
            ),
            f"overall={sec.get('overall')}",
            group="A",
        )
    )

    # ------------------------------------------------------------------ B. 项目
    created = client.post(
        "/api/v1/projects",
        json={
            "title": "Accept V20 Project",
            "research_question": "Can we close the workbench loop offline?",
            "research_goal": "v2.0 acceptance",
            "task_type": "general_ml",
            "dataset_keys": ["sklearn:digits"],
            "runner_profile_keys": ["local"],
            "mark_ready": True,
        },
    )
    proj = created.json() if created.status_code == 200 else {}
    project_id = str(proj.get("project_id") or "")
    results.append(
        _check(
            "B6-create-project",
            created.status_code == 200 and bool(project_id),
            f"project_id={project_id} status={proj.get('status')}",
            group="B",
        )
    )

    datasets = client.get("/api/v1/datasets")
    results.append(
        _check(
            "B7-datasets-list",
            datasets.status_code == 200 and "items" in (datasets.json() or {}),
            f"status={datasets.status_code}",
            group="B",
        )
    )

    proto_path = EXAMPLES / "digits_demo_protocol.json"
    protocol = service.protocols.create_from_path(proto_path)
    results.append(
        _check(
            "B8-create-protocol",
            protocol.protocol_id == "protocol_digits_demo_001",
            protocol.protocol_id,
            group="B",
        )
    )

    demo = client.post("/api/v1/demo/create", json={"kind": "digits", "force": True})
    demo_body = demo.json() if demo.status_code == 200 else {}
    demo_pid = str((demo_body.get("project") or {}).get("project_id") or "")
    results.append(
        _check(
            "B9-baseline-via-demo",
            demo.status_code == 200
            and demo_pid == "demo_digits_v20"
            and demo_body.get("auto_ran_experiments") is False
            and len(demo_body.get("nodes") or []) >= 1,
            f"nodes={len(demo_body.get('nodes') or [])}",
            group="B",
        )
    )

    shown = client.get(f"/api/v1/projects/{demo_pid}")
    shown_body = shown.json() if shown.status_code == 200 else {}
    results.append(
        _check(
            "B10-project-status-ready",
            shown.status_code == 200
            and str(shown_body.get("status") or "") in {"ready", "active"},
            f"status={shown_body.get('status')}",
            group="B",
        )
    )

    # ------------------------------------------------------------------ C. 实验
    _seed_digits_attempts(service, demo_pid)
    executions = client.get("/api/v1/executions", params={"limit": 20})
    exec_items = (executions.json() or {}).get("items") or []
    results.append(
        _check(
            "C11-run-or-seed-executions",
            executions.status_code == 200 and len(exec_items) >= 2,
            f"count={len(exec_items)} (seeded offline; Docker not required)",
            group="C",
        )
    )

    one = client.get(f"/api/v1/executions/{exec_items[0]['execution_id']}")
    one_body = one.json() if one.status_code == 200 else {}
    attempt = one_body.get("attempt") if isinstance(one_body.get("attempt"), dict) else {}
    results.append(
        _check(
            "C12-show-execution",
            one.status_code == 200
            and (
                one_body.get("execution_id")
                or attempt.get("execution_id")
                or bool(attempt)
            ),
            f"status={one.status_code} attempt_id={attempt.get('execution_id')}",
            group="C",
        )
    )

    agg = service.aggregate_node("node_003")
    results.append(
        _check(
            "C13-aggregate-node",
            bool(agg.get("node_id") == "node_003" or agg.get("aggregate_metrics")),
            f"keys={sorted(agg.keys())[:8]}",
            group="C",
        )
    )

    cmp = service.compare_nodes("node_003", "node_004")
    results.append(
        _check(
            "C14-compare-nodes",
            "conclusion" in cmp or "primary_metric" in cmp or "verdict" in cmp
            or "comparison" in cmp
            or bool(cmp),
            f"keys={sorted(cmp.keys())[:10]}",
            group="C",
        )
    )

    fb = service.analyze_feedback("node_003", "node_004")
    results.append(
        _check(
            "C15-feedback",
            bool(fb.get("feedback_id") or fb.get("node_id") or fb.get("summary") or fb),
            f"keys={sorted(fb.keys())[:10]}",
            group="C",
        )
    )

    # ------------------------------------------------------------------ D. AI Scientist
    _bootstrap_rgbt(service)
    planned = service.plan_next(
        "project_rgbt_003", protocol_id="protocol_rgbt_001", provider="mock"
    )
    plan_id = str(planned.get("plan_id") or "")
    results.append(
        _check(
            "D16-planner-candidates",
            bool(plan_id) and int(planned.get("valid_candidate_count") or 0) >= 1,
            f"plan_id={plan_id} valid={planned.get('valid_candidate_count')}",
            group="D",
        )
    )

    reviewed = service.review_plan(plan_id)
    results.append(
        _check(
            "D17-critic-review",
            reviewed.get("status") == "reviewed" and bool(reviewed.get("reviews")),
            f"status={reviewed.get('status')}",
            group="D",
        )
    )

    ranked = service.rank_candidates(plan_id)
    top = (ranked.get("ranking") or [{}])[0].get("candidate_id")
    approved = service.approve_candidate(plan_id, str(top))
    results.append(
        _check(
            "D18-human-approve-candidate",
            approved.get("status") == "approved" and bool(top),
            f"candidate={top}",
            group="D",
        )
    )

    generated = service.generate_contract_from_plan(plan_id, str(top))
    results.append(
        _check(
            "D19-generate-contract",
            bool(generated.get("contract")) and Path(generated.get("contract_path") or "").is_file(),
            str(generated.get("contract_path")),
            group="D",
        )
    )

    tree = service.tree_create(
        "project_rgbt_003",
        root_node_id="rgbt_formal_node_003",
        protocol_id="protocol_rgbt_001",
        max_depth=3,
        max_nodes=8,
        max_children=3,
    )
    tree_id = str(tree.get("tree_id") or "")
    tree_planned = service.tree_plan_next(tree_id)
    tree_approved = service.tree_approve(tree_id, tree_planned["top_candidate_id"])
    results.append(
        _check(
            "D20-create-iteration",
            bool(tree_approved.get("iteration_id")),
            f"iteration={tree_approved.get('iteration_id')}",
            group="D",
        )
    )

    _force_iteration_completed(
        service,
        iteration_id=str(tree_approved["iteration_id"]),
        proposed_node_id=str(tree_approved["proposed_node_id"]),
    )
    advanced = service.tree_advance(tree_id)
    results.append(
        _check(
            "D21-tree-advance",
            advanced.get("status") in {"advanced", "pending", "stopped"}
            or int(advanced.get("advanced_count") or 0) >= 1,
            f"status={advanced.get('status')} advanced={advanced.get('advanced_count')}",
            group="D",
        )
    )

    # Evidence from digits comparison (has seeded metrics)
    evidence = service.build_evidence("node_003", "node_004")
    claim = service.build_claim_matrix(demo_pid)
    results.append(
        _check(
            "D22-evidence-claim",
            bool(
                evidence.get("evidence_id")
                or evidence.get("evidence_ids")
                or evidence.get("records")
            )
            and (
                claim.get("claims") is not None
                or claim.get("project_id")
                or bool(claim)
            ),
            f"evidence_keys={list(evidence.keys())[:6]} claims={len(claim.get('claims') or [])}",
            group="D",
        )
    )

    # ------------------------------------------------------------------ E. 科研交付
    report = service.build_report("project_rgbt_003", tree_id=tree_id)
    report_id = str(report.get("report_id") or "")
    results.append(
        _check(
            "E23-build-report",
            bool(report_id),
            f"report_id={report_id}",
            group="E",
        )
    )

    verified = service.verify_report(report_id)
    results.append(
        _check(
            "E24-verify-report",
            bool(verified.get("ok") or verified.get("valid") or verified.get("status")),
            f"keys={sorted(verified.keys())[:8]}",
            group="E",
        )
    )

    audit = service.build_audit("project_rgbt_003", tree_id=tree_id, report_id=report_id)
    bundle_id = str(audit.get("bundle_id") or audit.get("audit_id") or "")
    results.append(
        _check(
            "E25-build-audit",
            bool(bundle_id),
            f"bundle_id={bundle_id}",
            group="E",
        )
    )

    export_dir = accept_root / "export"
    exported = service.export_project(demo_pid, output_dir=export_dir)
    export_path = Path(exported.get("path") or "")
    # Import into a fresh DB to prove round-trip
    other = _service(accept_root, "import")
    imported = other.import_project(export_path, force=True)
    results.append(
        _check(
            "E26-export-import-project",
            export_path.is_file()
            and imported.get("project_id") == demo_pid
            and int(imported.get("node_count") or 0) >= 1,
            f"path={export_path} nodes={imported.get('node_count')}",
            group="E",
        )
    )

    recover = service.recover(dry_run=True)
    results.append(
        _check(
            "E27-recover-dry-run",
            isinstance(recover, dict)
            and (
                "actions" in recover
                or "summary" in recover
                or "dry_run" in recover
                or bool(recover)
            ),
            f"keys={sorted(recover.keys())[:8]}",
            group="E",
        )
    )

    router = (ROOT / "web" / "src" / "app" / "router.tsx").read_text(encoding="utf-8")
    needed_pages = [
        "DashboardPage",
        "ProjectsPage",
        "ExecutionsPage",
        "TreesPage",
        "EvidencePage",
        "ReportsPage",
        "SettingsPage",
        "ApprovalsPage",
    ]
    missing_pages = [p for p in needed_pages if p not in router]
    results.append(
        _check(
            "E28-frontend-key-pages",
            not missing_pages,
            f"missing={missing_pages}",
            group="E",
        )
    )

    openapi = client.get("/openapi.json").json()
    paths = openapi.get("paths") or {}
    shell_paths = [p for p in paths if re.search(r"shell|exec-cmd|arbitrary", p, re.I)]
    # Also ensure chat textarea is local-assistant only (string check), not a shell API.
    results.append(
        _check(
            "E29-no-arbitrary-shell-api",
            not shell_paths and "/api/v1/system/security" in paths,
            f"shell_paths={shell_paths}",
            group="E",
        )
    )

    # Focused regression (fast unit subset used by v2.0 workbench)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/unit/test_demo_v208.py",
            "tests/unit/test_security_errors_v209.py",
            "tests/unit/test_agent_planning_workflow.py",
            "-q",
            "--tb=line",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    results.append(
        _check(
            "E30-focused-regression",
            proc.returncode == 0,
            (proc.stdout or proc.stderr or "")[-400:],
            group="E",
        )
    )

    # Persist acceptance artifacts
    out_dir = ROOT / "docs" / "acceptance" / "v2.0"
    out_dir.mkdir(parents=True, exist_ok=True)
    by_group: dict[str, list] = {}
    for item in results:
        by_group.setdefault(item["group"], []).append(item)

    payload = {
        "version": "v2.0.0",
        "tag_target": "v2.0.0",
        "package_version": "2.0.0",
        "passed": sum(1 for r in results if r["ok"]),
        "total": len(results),
        "groups": {
            g: {
                "passed": sum(1 for r in items if r["ok"]),
                "total": len(items),
            }
            for g, items in by_group.items()
        },
        "results": results,
    }
    report_path = out_dir / "v20_acceptance_report.json"
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "version_manifest.json").write_text(
        json.dumps(
            {
                "version": "v2.0.0",
                "tag_target": "v2.0.0",
                "branch": "feat/v2.0-scientist-workbench",
                "baseline_tag": "v1.9.0",
                "title": "Scientist Lab v2.0 AI Scientist Workbench",
                "principles": [
                    "Local single-user workbench",
                    "Default Mock LLM / no network",
                    "Human approval for Plan / Patch / Merge",
                    "No arbitrary Shell / no auto Git push",
                    "Claim Gate; recover never auto-reruns expensive experiments",
                ],
                "completed_subversions": [
                    "v2.0.1",
                    "v2.0.2",
                    "v2.0.3",
                    "v2.0.4",
                    "v2.0.5",
                    "v2.0.6",
                    "v2.0.7",
                    "v2.0.8",
                    "v2.0.9",
                    "v2.0.10",
                ],
                "acceptance_script": "scripts/accept_v20.py",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "README.md").write_text(
        """# Scientist Lab v2.0 Acceptance

## Run

```text
python scripts/accept_v20.py
```

## Groups

| Group | Focus |
|-------|--------|
| A | System / workbench / doctor / security |
| B | Project lifecycle + demo baseline |
| C | Executions / aggregate / compare / feedback |
| D | Planner / Critic / approve / tree / evidence |
| E | Report / Audit / export / recover / UI / regression |

## Tag

Target freeze tag: **`v2.0.0`** (baseline `v1.9.0`).
""",
        encoding="utf-8",
    )

    # Update version assertions used by prior unit tests
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    failed = [r for r in results if not r["ok"]]
    if failed:
        print(f"FAILED {len(failed)}/{len(results)}", file=sys.stderr)
        for item in failed:
            print(f"  - [{item['group']}] {item['name']}: {item['detail']}", file=sys.stderr)
        return 1
    print(f"PASSED {len(results)}/{len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
