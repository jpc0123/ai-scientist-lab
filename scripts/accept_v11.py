"""v1.1 end-to-end acceptance: finite experiment tree (MockPlanner, no real LLM)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scientist_lab.domain import JobStatus, NodeStage, NodeStatus, NodeType, ProjectStatus
from scientist_lab.domain.models import ExecutionAttempt, ExperimentNode, ResearchProject
from scientist_lab.iteration.service import IterationService
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings


def _check(name: str, ok: bool, detail: str = "") -> dict:
    return {"name": name, "ok": bool(ok), "detail": detail}


def _seed_attempts(
    service: ExperimentService,
    *,
    node_id: str,
    map_mean: float,
    input_mode: str = "fusion",
) -> None:
    contract = json.loads(
        (ROOT / "examples" / "rgbt_formal_fusion_contract.json").read_text(
            encoding="utf-8"
        )
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
                created_at=f"2026-07-21T00:00:{i:02d}+00:00",
                updated_at=f"2026-07-21T00:01:{i:02d}+00:00",
            )
        )


def seed(service: ExperimentService) -> None:
    now = "2026-07-21T00:00:00+00:00"
    try:
        service.create_protocol(ROOT / "examples" / "rgbt_protocol.json")
    except Exception:
        pass
    service.set_budget("project_rgbt_003", max_new_nodes=5, max_gpu_hours=10)
    service.repo.upsert_project(
        ResearchProject(
            project_id="project_rgbt_003",
            title="RGB-T v1.1 acceptance",
            research_goal="Finite tree search acceptance",
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
    _seed_attempts(service, node_id="rgbt_formal_node_003", map_mean=0.42)


def main() -> int:
    accept_root = ROOT / "outputs" / "_accept_v11"
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
    seed(service)
    results: list[dict] = []

    created = service.tree_create(
        "project_rgbt_003",
        root_node_id="rgbt_formal_node_003",
        protocol_id="protocol_rgbt_001",
        max_depth=3,
        max_nodes=8,
        max_children=3,
    )
    tree_id = created["tree_id"]
    results.append(
        _check(
            "tree-create",
            created.get("status") == "created" and created.get("node_count") == 1,
            f"tree_id={tree_id} status={created.get('status')}",
        )
    )

    nodes = service.tree_nodes(tree_id)
    root = nodes[0] if nodes else {}
    exp = service.repo.get_node("rgbt_formal_node_003")
    results.append(
        _check(
            "ExperimentNode-TreeNode-separation",
            exp is not None
            and root.get("experiment_node_id") == "rgbt_formal_node_003"
            and root.get("node_type") == "root"
            and root.get("tree_node_id") != "rgbt_formal_node_003",
            f"tree_node_id={root.get('tree_node_id')}",
        )
    )

    selection = service.tree_select_parent(tree_id)
    results.append(
        _check(
            "best-first-parent",
            bool(selection.get("selected")),
            json.dumps(selection.get("selected") or {}, ensure_ascii=False)[:300],
        )
    )

    planned = service.tree_plan_next(tree_id)
    top = planned.get("top_candidate_id")
    results.append(
        _check(
            "tree-plan-next-mock",
            planned.get("status") in {"planned", "ranked", "waiting_approval"}
            or bool(top),
            f"status={planned.get('status')} top={top}",
        )
    )

    approved = service.tree_approve(tree_id, top)
    results.append(
        _check(
            "tree-approve-iteration",
            bool(approved.get("iteration_id"))
            and bool(approved.get("proposed_node_id")),
            f"iteration={approved.get('iteration_id')} "
            f"proposed={approved.get('proposed_node_id')}",
        )
    )

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
    session.decision_id = "decision_accept_v11"
    iteration.repo.save_session(session)

    advanced = service.tree_advance(tree_id)
    child_nodes = service.tree_nodes(tree_id)
    child = next(n for n in child_nodes if n["experiment_node_id"] == proposed)
    results.append(
        _check(
            "tree-advance-backfill",
            advanced.get("advanced_count", 0) >= 1
            and child.get("status") == "evaluated"
            and child.get("score") is not None,
            f"status={advanced.get('status')} score={child.get('score')}",
        )
    )
    results.append(
        _check(
            "evidence-claim-backfill",
            bool(child.get("evidence_ids")) and bool(child.get("claim_matrix_path")),
            f"evidence_ids={child.get('evidence_ids')} "
            f"matrix={child.get('claim_matrix_path')}",
        )
    )

    evidence = service.tree_evidence(tree_id)
    results.append(
        _check(
            "tree-evidence",
            evidence.get("linked_count", 0) >= 1,
            f"linked={evidence.get('linked_count')} "
            f"evidence_count={evidence.get('evidence_count')}",
        )
    )

    exported_json = service.tree_export(tree_id, format="json")
    exported_mmd = service.tree_export(tree_id, format="mermaid")
    results.append(
        _check(
            "tree-export-json",
            exported_json.get("format") == "json"
            and exported_json.get("node_count", 0) >= 2
            and bool(exported_json.get("nodes")),
            f"nodes={exported_json.get('node_count')}",
        )
    )
    results.append(
        _check(
            "tree-export-mermaid",
            "flowchart TD" in (exported_mmd.get("mermaid") or "")
            and "-->" in (exported_mmd.get("mermaid") or ""),
            (exported_mmd.get("mermaid") or "")[:200],
        )
    )

    stopped = service.tree_stop(tree_id, reason="v1.1 acceptance stop")
    terminal_blocked = False
    try:
        service.tree_plan_next(tree_id)
    except ValueError as exc:
        terminal_blocked = "terminal" in str(exc).lower()
    results.append(
        _check(
            "user-stop-blocks-expand",
            stopped.get("status") == "user_stopped" and terminal_blocked,
            f"stop={stopped.get('status')} blocked={terminal_blocked}",
        )
    )

    # Persistence / restart recovery
    service2 = ExperimentService(settings=settings)
    recovered = service2.tree_show(tree_id)
    results.append(
        _check(
            "restart-recovery",
            recovered.get("tree_id") == tree_id
            and recovered.get("status") == "user_stopped"
            and recovered.get("node_count", 0) >= 2,
            f"status={recovered.get('status')} nodes={recovered.get('node_count')}",
        )
    )

    # v1.0 single-round planning still independent
    v10 = service2.plan_next("project_rgbt_003", protocol_id="protocol_rgbt_001")
    results.append(
        _check(
            "v1.0-plan-next-independent",
            bool(v10.get("plan_id")),
            f"plan_id={v10.get('plan_id')} status={v10.get('status')}",
        )
    )

    ok_all = all(item["ok"] for item in results)
    out_dir = settings.outputs_dir / "project_rgbt_003" / "acceptance"
    out_dir.mkdir(parents=True, exist_ok=True)
    mmd_path = out_dir / "v11_tree.mmd"
    mmd_path.write_text((exported_mmd.get("mermaid") or "") + "\n", encoding="utf-8")
    json_path = out_dir / "v11_tree.json"
    json_path.write_text(
        json.dumps(exported_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = {
        "version": "v1.1",
        "ok": ok_all,
        "tree_id": tree_id,
        "project_id": "project_rgbt_003",
        "protocol_id": "protocol_rgbt_001",
        "mermaid_path": str(mmd_path),
        "json_path": str(json_path),
        "steps": results,
        "passed": sum(1 for r in results if r["ok"]),
        "total": len(results),
    }
    report_path = out_dir / "v11_acceptance_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nWrote {report_path}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
