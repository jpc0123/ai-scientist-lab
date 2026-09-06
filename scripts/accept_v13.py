"""v1.3 acceptance: offline LLM Provider + finite tree plan-next (no cloud)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scientist_lab.agents.provider_bridge import build_llm_provider
from scientist_lab.domain import JobStatus, NodeStage, NodeStatus, NodeType, ProjectStatus
from scientist_lab.domain.models import ExecutionAttempt, ExperimentNode, ResearchProject
from scientist_lab.llm.limits import ProviderLimitExceeded, ProviderLimits
from scientist_lab.llm.models import LLMRequest
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
            title="RGB-T v1.3 acceptance",
            research_goal="Offline LLM provider + finite tree",
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
    accept_root = ROOT / "outputs" / "_accept_v13"
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

    quality = service.evaluate_llm_quality(
        "project_rgbt_003",
        protocol_id="protocol_rgbt_001",
        current_best_node_id="rgbt_formal_node_003",
        include_real=True,
    )
    gates = quality.get("gates") or {}
    results.append(
        _check(
            "llm-eval-gates",
            bool(gates.get("mock_ok") and gates.get("fake_ok") and gates.get("replay_ok")),
            json.dumps(gates, ensure_ascii=False),
        )
    )
    results.append(
        _check(
            "real-deferred",
            bool(quality.get("real_deferred"))
            and any(
                m.get("mode") == "real" and m.get("status") == "skipped"
                for m in (quality.get("modes") or [])
            ),
            "real mode skipped",
        )
    )

    planned = service.plan_next(
        "project_rgbt_003",
        protocol_id="protocol_rgbt_001",
        current_best_node_id="rgbt_formal_node_003",
        provider="fake",
    )
    results.append(
        _check(
            "plan-next-fake",
            planned.get("model_provider") in {"fake", "audit:fake", "limit:audit:fake"}
            or "fake" in str(planned.get("model_provider") or ""),
            f"provider={planned.get('model_provider')} status={planned.get('status')}",
        )
    )

    replayed = service.plan_next(
        "project_rgbt_003",
        protocol_id="protocol_rgbt_001",
        current_best_node_id="rgbt_formal_node_003",
        provider="replay",
    )
    results.append(
        _check(
            "plan-next-replay",
            replayed.get("reasoning_summary") == planned.get("reasoning_summary")
            and replayed.get("model_provider") == "replay",
            f"replay_provider={replayed.get('model_provider')}",
        )
    )

    limited = build_llm_provider(
        "fake",
        audit_root=settings.outputs_dir / "project_rgbt_003" / "llm_limit_check",
        project_id="project_rgbt_003",
        limits=ProviderLimits(max_calls=1),
    )
    req = LLMRequest(
        purpose="other",
        messages=[{"role": "user", "content": "limit-check"}],
        metadata={"project_id": "project_rgbt_003"},
    )
    assert limited is not None
    limited.complete(req)
    limit_ok = False
    try:
        limited.complete(req)
    except ProviderLimitExceeded as exc:
        limit_ok = exc.limit == "max_calls"
    results.append(_check("provider-limits-max-calls", limit_ok, "max_calls=1"))

    usage = service.summarize_llm_usage("project_rgbt_003")
    results.append(
        _check(
            "llm-usage-summary",
            int(usage.get("call_count") or 0) >= 1
            and float(usage.get("total_tokens") or 0) >= 0,
            f"calls={usage.get('call_count')} tokens={usage.get('total_tokens')}",
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
    tree_id = created["tree_id"]
    results.append(
        _check(
            "tree-create",
            created.get("status") == "created" and created.get("node_count") == 1,
            f"tree_id={tree_id}",
        )
    )

    tree_fake = service.tree_plan_next(tree_id, provider="fake")
    results.append(
        _check(
            "tree-plan-next-fake",
            tree_fake.get("status") in {"planned", "no_valid_candidates", "stopped"}
            and tree_fake.get("provider") == "fake"
            and (
                bool(tree_fake.get("top_candidate_id"))
                or tree_fake.get("status") != "planner_failed"
            ),
            f"status={tree_fake.get('status')} top={tree_fake.get('top_candidate_id')} "
            f"provider={tree_fake.get('provider')}",
        )
    )

    # Second tree: mock path still works (default).
    created2 = service.tree_create(
        "project_rgbt_003",
        root_node_id="rgbt_formal_node_003",
        protocol_id="protocol_rgbt_001",
        max_depth=3,
        max_nodes=8,
        max_children=3,
    )
    tree_mock = service.tree_plan_next(created2["tree_id"], provider="mock")
    results.append(
        _check(
            "tree-plan-next-mock-default",
            tree_mock.get("provider") == "mock"
            and (
                bool(tree_mock.get("top_candidate_id"))
                or tree_mock.get("status") in {"planned", "no_valid_candidates"}
            ),
            f"status={tree_mock.get('status')} top={tree_mock.get('top_candidate_id')}",
        )
    )

    report = {
        "version": "v1.3",
        "checks": results,
        "passed": sum(1 for r in results if r["ok"]),
        "total": len(results),
        "offline_only": True,
        "real_llm": False,
        "quality_report_path": quality.get("path"),
    }
    out_dir = settings.outputs_dir / "project_rgbt_003" / "acceptance"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "v13_acceptance_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    docs_dir = ROOT / "docs" / "acceptance" / "v1.3"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "v13_acceptance_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (docs_dir / "README.md").write_text(
        "\n".join(
            [
                "# v1.3 Acceptance",
                "",
                "- Offline LLM Provider (Fake / Replay / Audit / Schema)",
                "- Planner/Critic adapters (default mock)",
                "- Quality eval + token/cost/latency limits",
                "- Finite tree plan-next with `--provider mock|fake|replay`",
                "- Real cloud LLM: false",
                "",
                f"Checks: {report['passed']}/{report['total']}",
                "",
                "Run: `python scripts/accept_v13.py`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"wrote {out_path}")
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
