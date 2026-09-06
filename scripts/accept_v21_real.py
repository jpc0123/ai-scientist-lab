"""v2.1 real acceptance: gated live LLM closed-loop report.

Default / CI: SKIP (exit 0) — never opens network sockets.

Live mode requires ALL of:
  RUN_REAL_LLM_TESTS=1
  LLM_ALLOW_NETWORK=true
  LLM_API_KEY / LLM_BASE_URL / LLM_MODEL configured

Optional Digits Docker execution:
  RUN_REAL_DIGITS=1
  (and local Docker image scientist-experiment:v2 available)

Usage:
  python scripts/accept_v21_real.py
  # skip unless gates set

  $env:RUN_REAL_LLM_TESTS=1
  $env:LLM_ALLOW_NETWORK="true"
  $env:LLM_API_KEY="..."
  $env:LLM_BASE_URL="https://..."
  $env:LLM_MODEL="..."
  python scripts/accept_v21_real.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scientist_lab.domain import JobStatus, NodeStage, NodeStatus, NodeType
from scientist_lab.domain.models import ExecutionAttempt, ExperimentNode, new_id, utc_now_iso
from scientist_lab.llm.eval_suite import real_eval_gates
from scientist_lab.llm_eval.profiles import LLMModelProfile
from scientist_lab.research_loop.replay_bundle import assert_bundle_redacted, load_replay_bundle
from scientist_lab.research_loop.state_machine import transition
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings

EXAMPLES = ROOT / "examples"


def _check(name: str, ok: bool, detail: str = "", *, skipped: bool = False) -> dict:
    return {
        "name": name,
        "ok": bool(ok),
        "skipped": bool(skipped),
        "detail": detail,
    }


def _env_truthy(name: str, environ: dict[str, str] | None = None) -> bool:
    env = environ if environ is not None else dict(os.environ)
    return str(env.get(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def real_loop_acceptance_gates(
    *,
    environ: dict[str, str] | None = None,
) -> tuple[bool, str, dict[str, bool]]:
    """Composite gate for accept_v21_real live mode."""
    env = environ if environ is not None else dict(os.environ)
    allowed, reason = real_eval_gates(
        provider="openai-compatible",
        allow_network=True,
        environ=env,
    )
    gates = {
        "run_real_llm_tests": _env_truthy("RUN_REAL_LLM_TESTS", env),
        "llm_allow_network": _env_truthy("LLM_ALLOW_NETWORK", env),
        "llm_api_key": bool(str(env.get("LLM_API_KEY") or "").strip()),
        "llm_base_url": bool(str(env.get("LLM_BASE_URL") or "").strip()),
        "llm_model": bool(str(env.get("LLM_MODEL") or "").strip()),
        "run_real_digits": _env_truthy("RUN_REAL_DIGITS", env),
    }
    return allowed, reason, gates


def _service(accept_root: Path) -> ExperimentService:
    return ExperimentService(
        settings=Settings(
            project_root=ROOT,
            db_path=accept_root / "accept_v21_real.db",
            runtime_dir=accept_root / "runtime",
            outputs_dir=accept_root / "outputs",
            experiment_app_dir=ROOT / "experiment_app",
        ).resolve()
    )


def _passing_scorecard() -> dict[str, Any]:
    return {
        "status": "completed",
        "suite_version": "eval_suite_v1",
        "planner": {
            "schema_valid_rate": 1.0,
            "protocol_compliance_rate": 1.0,
            "duplicate_candidate_rate": 0.0,
            "evidence_gap_relevance_rate": 0.95,
            "case_count": 10,
        },
        "critic": {"case_count": 5},
        "safety": {
            "pass": True,
            "pass_rate": 1.0,
            "violation_count": 0,
            "case_count": 5,
        },
        "operations": {"repair_rate": 0.0, "average_latency_ms": 50},
        "metadata": {
            "planner_prompt_version": "planner_v1",
            "critic_prompt_version": "critic_v1",
        },
    }


def _seed_baseline(service: ExperimentService) -> dict[str, str]:
    project = service.create_project(
        title="Accept v2.1 Real Closed Loop",
        research_question="real llm multi-round",
        research_goal="v2.1.9 real acceptance",
        task_type="general_ml",
        project_id="project_accept_v21_real",
        mark_ready=True,
    )
    protocol = service.protocols.create_from_path(
        EXAMPLES / "digits_demo_protocol.json"
    )
    profile = LLMModelProfile(
        profile_id="profile_accept_v21_real",
        provider="openai-compatible",
        model=str(os.environ.get("LLM_MODEL") or "gpt-test").strip(),
        planner_prompt_version="planner_v1",
        critic_prompt_version="critic_v1",
        enabled=True,
    )
    service.llm_evals.upsert_profile(profile)
    service.llm_evals.save_evaluation(
        evaluation_id="eval_accept_v21_real",
        profile_id=profile.profile_id,
        suite_version="eval_suite_v1",
        status="completed",
        result=_passing_scorecard(),
        report_path=None,
        case_rows=[],
    )
    service.set_budget(project["project_id"], max_new_nodes=6, max_gpu_hours=4.0)

    now = utc_now_iso()
    parent = ExperimentNode(
        node_id=new_id("node"),
        project_id=project["project_id"],
        node_type=NodeType.BASELINE,
        stage=NodeStage.DONE,
        status=NodeStatus.SUCCEEDED,
        hypothesis="hu=64",
        contract_json={
            "schema_version": "1.0",
            "project_id": project["project_id"],
            "title": "Digits baseline",
            "research_goal": "v2.1.9",
            "parameters": {
                "learning_rate": 0.001,
                "epochs": 30,
                "hidden_units": 64,
                "batch_size": 64,
                "test_size": 0.2,
            },
            "protocol_id": protocol.protocol_id,
            "environment_key": "digits-mlp-v1",
            "entrypoint": "run_experiment.py",
            "dataset_reference": "sklearn:digits",
            "code_reference": "local:experiment_app",
            "runner_profile": "local",
            "execution_mode": "fast_eval",
            "resources": {
                "gpu_count": 0,
                "cpu_count": 2,
                "memory_gb": 2,
                "timeout_seconds": 300,
            },
        },
        created_at=now,
        updated_at=now,
    )
    service.repo.upsert_node(parent)
    # Seed baseline attempts so compare/evidence works even if Digits is fixture.
    for idx, seed in enumerate([42, 43], start=1):
        service.repo.upsert_attempt(
            ExecutionAttempt(
                execution_id=f"accept_real_base_{seed}",
                node_id=parent.node_id,
                attempt_index=idx,
                runner_profile="local",
                status=JobStatus.COMPLETED,
                image_reference="scientist-experiment:v2",
                code_version="local:experiment_app",
                dataset_version="sklearn:digits",
                result_json={
                    "metrics": {
                        "primary_metric": "accuracy",
                        "metrics": {"accuracy": 0.94 + idx * 0.001},
                    },
                    "contract": {
                        **dict(parent.contract_json or {}),
                        "seed": seed,
                        "node_id": parent.node_id,
                    },
                },
                created_at=now,
                completed_at=now,
            )
        )

    created = service.real_loop_create(
        project["project_id"],
        profile_id=profile.profile_id,
        protocol_id=protocol.protocol_id,
        baseline_node_ids=[parent.node_id],
        rounds=2,
    )
    service._real_loop_service().mark_baseline_ready(created["session_id"])
    return {
        "session_id": created["session_id"],
        "project_id": project["project_id"],
        "profile_id": profile.profile_id,
        "protocol_id": protocol.protocol_id,
        "parent_id": parent.node_id,
    }


def _fixture_execute_and_feedback(
    service: ExperimentService, *, session_id: str, parent_id: str
) -> dict[str, Any]:
    """When Digits Docker is not requested, complete round-1 with seeded attempts."""
    shown = service.real_loop_show(session_id)
    r1 = next(r for r in shown["rounds"] if r["round_number"] == 1)
    plan_id = r1.get("plan_id")
    cand_id = (r1.get("candidate_ids") or [None])[0]
    if not plan_id or not cand_id:
        raise RuntimeError("plan/candidate missing before fixture execute")

    approved = service.real_loop_approve(
        session_id, candidate_id=str(cand_id), round_number=1, seeds=[42, 43]
    )
    node_id = str(approved.get("proposed_node_id") or "")
    now = utc_now_iso()
    node = service.repo.get_node(node_id)
    if node is not None:
        node.stage = NodeStage.DONE
        node.status = NodeStatus.SUCCEEDED
        node.updated_at = now
        service.repo.upsert_node(node)
    for idx, seed in enumerate([42, 43], start=1):
        service.repo.upsert_attempt(
            ExecutionAttempt(
                execution_id=f"accept_real_cand_{seed}",
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
                        "metrics": {"accuracy": 0.955 + idx * 0.001},
                    },
                    "contract": {
                        "project_id": shown["project_id"],
                        "node_id": node_id,
                        "seed": seed,
                        "parameters": {"hidden_units": 112},
                        "environment_key": "digits-mlp-v1",
                        "entrypoint": "run_experiment.py",
                        "dataset_reference": "sklearn:digits",
                    },
                },
                created_at=now,
                completed_at=now,
            )
        )

    loop = service._real_loop_service()
    session = loop._require(session_id)
    if session.status == "round_1_waiting_approval":
        transition(session, "round_1_executing")
        loop.repo.upsert_session(session)
    rnd = loop._require_round(session_id, 1)
    rnd.execution_node_id = node_id
    rnd.iteration_id = approved.get("iteration_id")
    rnd.status = "executing"
    loop.repo.upsert_round(rnd)
    if node_id not in session.execution_node_ids:
        session.execution_node_ids = [*session.execution_node_ids, node_id]
        loop.repo.upsert_session(session)

    fb = service.real_loop_record_execution_feedback(session_id)
    return {
        "digits_mode": "fixture",
        "approve": approved,
        "feedback": fb,
        "execution_node_id": node_id,
        "parent_id": parent_id,
    }


def run_live_closed_loop(
    service: ExperimentService,
    *,
    gates: dict[str, bool],
) -> tuple[list[dict], dict[str, Any]]:
    """Execute as much of the real closed loop as gates allow."""
    results: list[dict] = []
    meta: dict[str, Any] = {
        "network_used": False,
        "digits_mode": "skipped",
        "session_id": None,
        "bundle_dir": None,
    }

    ids = _seed_baseline(service)
    meta["session_id"] = ids["session_id"]
    results.append(
        _check(
            "session_created_real_only",
            True,
            f"session={ids['session_id']} profile={ids['profile_id']}",
        )
    )

    # Live Planner (network)
    planned = service.real_loop_plan(
        ids["session_id"],
        round_number=1,
        allow_network=True,
        provider="openai-compatible",
    )
    meta["network_used"] = True
    results.append(
        _check(
            "round1_live_plan",
            planned.get("ok", True) is not False and bool(planned.get("plan_id")),
            f"status={planned.get('status')} plan={planned.get('plan_id')} "
            f"fallback={planned.get('fallback_used')}",
        )
    )
    if planned.get("ok") is False:
        return results, meta

    reviewed = service.real_loop_review(
        ids["session_id"],
        round_number=1,
        allow_network=True,
        provider="openai-compatible",
    )
    results.append(
        _check(
            "round1_live_review",
            reviewed.get("ok", True) is not False,
            f"status={reviewed.get('status')} fallback={reviewed.get('fallback_used')}",
        )
    )

    shown = service.real_loop_show(ids["session_id"])
    results.append(
        _check(
            "display_mode_real",
            shown.get("display_mode") == "REAL",
            str(shown.get("display_mode")),
        )
    )

    # Digits: real docker or fixture
    if gates.get("run_real_digits"):
        r1 = next(r for r in shown["rounds"] if r["round_number"] == 1)
        cand = (r1.get("candidate_ids") or [None])[0]
        if cand:
            service.real_loop_approve(
                ids["session_id"], candidate_id=str(cand), seeds=[42]
            )
            executed = service.real_loop_execute(
                ids["session_id"], seeds=[42], wait=True
            )
            meta["digits_mode"] = "docker"
            results.append(
                _check(
                    "round1_real_digits_execute",
                    executed.get("ok", True) is not False
                    and executed.get("mock_execution") is False,
                    f"status={executed.get('status')} node={executed.get('execution_node_id')}",
                )
            )
            fb = service.real_loop_record_execution_feedback(ids["session_id"])
            results.append(
                _check(
                    "round1_execution_feedback",
                    fb.get("ok", True) is not False
                    and fb.get("status") == "round_1_feedback_ready",
                    f"evidence={len(fb.get('evidence_ids') or [])}",
                )
            )
        else:
            results.append(
                _check("round1_real_digits_execute", False, "no candidate_id")
            )
    else:
        fixture = _fixture_execute_and_feedback(
            service, session_id=ids["session_id"], parent_id=ids["parent_id"]
        )
        meta["digits_mode"] = "fixture"
        results.append(
            _check(
                "round1_digits_fixture_path",
                True,
                "RUN_REAL_DIGITS not set; used seeded attempts (LLM path still real)",
            )
        )
        results.append(
            _check(
                "round1_execution_feedback",
                fixture["feedback"].get("status") == "round_1_feedback_ready",
                f"node={fixture.get('execution_node_id')}",
            )
        )

    nxt = service.real_loop_next_round(ids["session_id"])
    results.append(
        _check(
            "round2_context_ready",
            nxt.get("status") == "round_2_planning"
            and bool((nxt.get("context") or {}).get("has_round_feedback")),
            str(nxt.get("status")),
        )
    )

    planned2 = service.real_loop_plan(
        ids["session_id"],
        round_number=2,
        allow_network=True,
        provider="openai-compatible",
    )
    results.append(
        _check(
            "round2_live_plan",
            planned2.get("ok", True) is not False and bool(planned2.get("plan_id")),
            f"plan={planned2.get('plan_id')} feedback_use_ok={planned2.get('feedback_use_ok')}",
        )
    )

    verify = service.real_loop_verify_feedback(ids["session_id"], round_number=2)
    results.append(
        _check(
            "feedback_use_verifier",
            verify.get("pass_status") is True,
            json.dumps(verify.get("issues") or []),
        )
    )

    bundle_dir = (
        Path(service.settings.outputs_dir)
        / ids["project_id"]
        / "real_loop"
        / ids["session_id"]
    )
    exported = service.real_loop_export(ids["session_id"], output_dir=str(bundle_dir))
    meta["bundle_dir"] = str(bundle_dir)
    results.append(
        _check(
            "replay_bundle_export",
            exported.get("ok") is True and exported.get("secrets_redacted") is True,
            str(exported.get("bundle_dir")),
        )
    )
    try:
        assert_bundle_redacted(bundle_dir)
        load_replay_bundle(bundle_dir)
        redaction_ok = True
        detail = "clean"
    except Exception as exc:  # noqa: BLE001
        redaction_ok = False
        detail = str(exc)
    results.append(_check("secret_scan_clean", redaction_ok, detail))

    return results, meta


def main() -> int:
    accept_root = ROOT / "outputs" / "_accept_v21_real"
    accept_root.mkdir(parents=True, exist_ok=True)
    report_dir = ROOT / "docs" / "acceptance" / "v2.1"
    report_dir.mkdir(parents=True, exist_ok=True)

    allowed, reason, gates = real_loop_acceptance_gates()
    results: list[dict] = [
        _check(
            "gate_run_real_llm_tests",
            gates["run_real_llm_tests"],
            "RUN_REAL_LLM_TESTS=1",
            skipped=not gates["run_real_llm_tests"],
        ),
        _check(
            "gate_llm_allow_network",
            gates["llm_allow_network"],
            "LLM_ALLOW_NETWORK=true",
            skipped=not gates["llm_allow_network"],
        ),
        _check(
            "gate_llm_credentials",
            gates["llm_api_key"] and gates["llm_base_url"] and gates["llm_model"],
            "LLM_API_KEY / LLM_BASE_URL / LLM_MODEL",
            skipped=not (
                gates["llm_api_key"] and gates["llm_base_url"] and gates["llm_model"]
            ),
        ),
    ]

    meta: dict[str, Any] = {
        "network_used": False,
        "digits_mode": "n/a",
        "session_id": None,
        "bundle_dir": None,
    }
    status = "skipped"

    if not allowed:
        results.append(
            _check(
                "live_closed_loop",
                True,
                f"skipped: {reason}",
                skipped=True,
            )
        )
        status = "skipped"
    else:
        service = _service(accept_root)
        try:
            live_results, meta = run_live_closed_loop(service, gates=gates)
            results.extend(live_results)
            failed = [r for r in live_results if not r["ok"] and not r.get("skipped")]
            status = "failed" if failed else "passed"
        except Exception as exc:  # noqa: BLE001
            results.append(_check("live_closed_loop", False, str(exc)))
            status = "failed"
            meta["network_used"] = True

    hard_failed = [
        r for r in results if (not r["ok"]) and (not r.get("skipped"))
    ]
    payload = {
        "schema_version": "v21_real_closed_loop_report",
        "version": "v2.1.9",
        "status": status,
        "gates": gates,
        "gate_reason": reason,
        "network_used": bool(meta.get("network_used")),
        "digits_mode": meta.get("digits_mode"),
        "session_id": meta.get("session_id"),
        "bundle_dir": meta.get("bundle_dir"),
        "passed": sum(1 for r in results if r["ok"]),
        "total": len(results),
        "failed_count": len(hard_failed),
        "results": results,
        "definition": (
            "Gated real-LLM multi-round closed-loop acceptance. "
            "Default CI skips without network. Digits Docker is optional "
            "(RUN_REAL_DIGITS=1)."
        ),
        "how_to_run": {
            "required_env": [
                "RUN_REAL_LLM_TESTS=1",
                "LLM_ALLOW_NETWORK=true",
                "LLM_API_KEY",
                "LLM_BASE_URL",
                "LLM_MODEL",
            ],
            "optional_env": ["RUN_REAL_DIGITS=1"],
            "command": "python scripts/accept_v21_real.py",
        },
    }

    report_path = report_dir / "v21_real_closed_loop_report.json"
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Also mirror under outputs for operators.
    (accept_root / "v21_real_closed_loop_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if status == "skipped":
        print(f"SKIPPED ({reason}) — report: {report_path}")
        return 0
    if hard_failed:
        print(f"FAILED {len(hard_failed)}/{len(results)}", file=sys.stderr)
        for item in hard_failed:
            print(f"  - {item['name']}: {item['detail']}", file=sys.stderr)
        return 1
    print(f"PASSED {payload['passed']}/{payload['total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
