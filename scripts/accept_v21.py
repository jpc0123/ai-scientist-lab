"""v2.1 offline acceptance: real research loop Replay Bundle path.

Zero network by default. Uses fixtures + research_loop unit semantics.
Does not require Docker, API keys, or real Digits training.

Usage:
  python scripts/accept_v21.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scientist_lab.domain import NodeStage, NodeStatus, NodeType
from scientist_lab.domain.models import ExperimentNode, new_id, utc_now_iso
from scientist_lab.llm_eval.profiles import LLMModelProfile
from scientist_lab.research_loop.feedback_use_verifier import verify_feedback_use
from scientist_lab.research_loop.replay_bundle import (
    assert_bundle_redacted,
    load_replay_bundle,
)
from scientist_lab.research_loop.state_machine import transition
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings

EXAMPLES = ROOT / "examples"


def _check(name: str, ok: bool, detail: str = "") -> dict:
    return {"name": name, "ok": bool(ok), "detail": detail}


def _service(accept_root: Path) -> ExperimentService:
    return ExperimentService(
        settings=Settings(
            project_root=ROOT,
            db_path=accept_root / "accept_v21.db",
            runtime_dir=accept_root / "runtime",
            outputs_dir=accept_root / "outputs",
            experiment_app_dir=ROOT / "experiment_app",
        ).resolve()
    )


def _seed_loop(service: ExperimentService) -> dict[str, str]:
    project = service.create_project(
        title="Accept v2.1 Digits Loop",
        research_question="offline closed loop",
        research_goal="v2.1.7 acceptance",
        task_type="general_ml",
        project_id="project_accept_v21",
        mark_ready=True,
    )
    protocol = service.protocols.create_from_path(
        EXAMPLES / "digits_demo_protocol.json"
    )
    profile = LLMModelProfile(
        profile_id="profile_accept_v21",
        provider="openai-compatible",
        model="gpt-test",
        planner_prompt_version="planner_v1",
        critic_prompt_version="critic_v1",
        enabled=True,
    )
    service.llm_evals.upsert_profile(profile)
    service.set_budget(project["project_id"], max_new_nodes=5, max_gpu_hours=2.0)

    now = utc_now_iso()
    parent = ExperimentNode(
        node_id=new_id("node"),
        project_id=project["project_id"],
        node_type=NodeType.BASELINE,
        stage=NodeStage.DONE,
        status=NodeStatus.SUCCEEDED,
        hypothesis="hu=64",
        contract_json={
            "parameters": {"hidden_units": 64},
            "protocol_id": protocol.protocol_id,
            "environment_key": "digits-mlp-v1",
            "entrypoint": "run_experiment.py",
            "dataset_reference": "sklearn:digits",
        },
        created_at=now,
        updated_at=now,
    )
    executed = ExperimentNode(
        node_id=new_id("node"),
        project_id=project["project_id"],
        parent_node_id=parent.node_id,
        node_type=NodeType.IMPROVEMENT,
        stage=NodeStage.DONE,
        status=NodeStatus.SUCCEEDED,
        hypothesis="hu=112",
        contract_json={
            "parameters": {"hidden_units": 112},
            "protocol_id": protocol.protocol_id,
            "environment_key": "digits-mlp-v1",
            "entrypoint": "run_experiment.py",
            "dataset_reference": "sklearn:digits",
        },
        created_at=now,
        updated_at=now,
    )
    service.repo.upsert_node(parent)
    service.repo.upsert_node(executed)

    created = service.real_loop_create(
        project["project_id"],
        profile_id=profile.profile_id,
        protocol_id=protocol.protocol_id,
        baseline_node_ids=[parent.node_id],
        rounds=2,
    )
    loop = service._real_loop_service()
    session = loop._require(created["session_id"])
    for status in (
        "baseline_ready",
        "round_1_planning",
        "round_1_reviewing",
        "round_1_waiting_approval",
        "round_1_executing",
        "round_1_feedback_ready",
        "round_2_planning",
        "round_2_reviewing",
        "completed",
    ):
        transition(session, status)
    session.current_round = 2
    session.execution_node_ids = [executed.node_id]
    loop.repo.upsert_session(session)

    fb = {
        "source_round": 1,
        "parent_node_id": parent.node_id,
        "executed_node_id": executed.node_id,
        "metric_deltas": {"accuracy": 0.012},
        "evidence_added": ["evidence_accept_v21"],
        "claims_changed": ["claim_accept_v21"],
        "outcome_label": "improved",
        "executed_parameters": {"hidden_units": 112},
        "previous_hypothesis": "hu=112",
    }
    r1 = loop._require_round(session.session_id, 1)
    r1.feedback_summary_json = fb
    r1.execution_node_id = executed.node_id
    r1.evidence_ids = ["evidence_accept_v21"]
    r1.plan_id = "plan_accept_r1"
    r1.approved_candidate_id = "candidate_digits_hu112"
    r1.planning_context_json = {"context_sha256": "r1ctx"}
    r1.provider_audit_json = {
        "requested_provider": "openai-compatible",
        "actual_provider": "openai-compatible",
        "fallback_used": False,
    }
    r1.status = "feedback_ready"
    loop.repo.upsert_round(r1)

    ctx2 = {
        "context_sha256": "r2ctx",
        "round_feedback_summary": fb,
        "recent_execution_summary": {
            "executed_node_id": executed.node_id,
            "outcome_label": "improved",
            "metric_deltas": fb["metric_deltas"],
            "evidence_added": fb["evidence_added"],
        },
        "evidence_records": [{"evidence_id": "evidence_accept_v21"}],
        "tested_parameter_fingerprints": [],
    }
    r2 = loop._require_round(session.session_id, 2)
    r2.plan_id = "plan_accept_r2"
    r2.planning_context_json = ctx2
    r2.candidate_ids = ["candidate_digits_hu96"]
    r2.provider_audit_json = {
        "requested_provider": "openai-compatible",
        "fallback_used": False,
    }
    r2.status = "reviewing"
    loop.repo.upsert_round(r2)

    def _show(plan_id: str):
        if plan_id == "plan_accept_r1":
            return {
                "plan_id": plan_id,
                "project_id": project["project_id"],
                "candidates": [
                    {
                        "candidate_id": "candidate_digits_hu112",
                        "parent_node_id": parent.node_id,
                        "parameter_changes": {"hidden_units": 112},
                        "rationale": "capacity probe",
                    }
                ],
                "critic": {"recommendation": "accept"},
            }
        return {
            "plan_id": plan_id,
            "project_id": project["project_id"],
            "candidates": [
                {
                    "candidate_id": "candidate_digits_hu96",
                    "parent_node_id": executed.node_id,
                    "parameter_changes": {"hidden_units": 96},
                    "rationale": "accuracy improved; refine capacity",
                    "evidence_gap_addressed": ["evidence_accept_v21"],
                    "expected_outcomes": [
                        {
                            "metric": "accuracy",
                            "direction": "increase",
                            "rationale": "follow round-1 metric",
                        }
                    ],
                }
            ],
            "critic": {"recommendation": "accept"},
        }

    service.show_plan = _show  # type: ignore[method-assign]
    return {
        "session_id": session.session_id,
        "project_id": project["project_id"],
        "parent_id": parent.node_id,
        "executed_id": executed.node_id,
        "profile_id": profile.profile_id,
        "protocol_id": protocol.protocol_id,
    }


def main() -> int:
    accept_root = ROOT / "outputs" / "_accept_v21"
    accept_root.mkdir(parents=True, exist_ok=True)
    report_dir = ROOT / "docs" / "acceptance" / "v2.1"
    report_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    service = _service(accept_root)
    ids = _seed_loop(service)
    loop = service._real_loop_service()
    shown = service.real_loop_show(ids["session_id"])

    results.append(
        _check(
            "create_two_round_session",
            shown.get("required_rounds", 0) >= 2
            and shown.get("real_only") is True
            and shown.get("fallback_allowed") is False,
            f"status={shown.get('status')} rounds={shown.get('required_rounds')}",
        )
    )
    results.append(
        _check(
            "session_completed_offline",
            shown.get("status") == "completed",
            str(shown.get("status")),
        )
    )

    checked = service.real_loop_check(ids["session_id"])
    results.append(
        _check(
            "offline_session_check",
            checked.get("overall") == "ok" and checked.get("network_used") is False,
            json.dumps(checked.get("issues") or []),
        )
    )

    profile = service.real_loop_check_profile(
        ids["profile_id"],
        session_id=ids["session_id"],
        require_quality_gate=False,
    )
    results.append(
        _check(
            "profile_bound_real_provider",
            profile.get("provider") in {"openai-compatible", "real"}
            or any(
                c.get("name") == "provider_type" and c.get("ok")
                for c in profile.get("checks") or []
            ),
            str(profile.get("provider") or profile.get("issues")),
        )
    )

    # Round-1 feedback present
    r1 = next(r for r in shown["rounds"] if r["round_number"] == 1)
    results.append(
        _check(
            "round1_feedback_fixture",
            bool(r1.get("feedback_summary_json")),
            "feedback_summary present" if r1.get("feedback_summary_json") else "missing",
        )
    )
    results.append(
        _check(
            "human_approval_boundary",
            bool(r1.get("approved_candidate_id")),
            str(r1.get("approved_candidate_id")),
        )
    )
    results.append(
        _check(
            "execution_fixture_backfill",
            ids["executed_id"] in (shown.get("execution_node_ids") or []),
            str(shown.get("execution_node_ids")),
        )
    )
    results.append(
        _check(
            "evidence_ids_on_round1",
            bool(r1.get("evidence_ids")),
            str(r1.get("evidence_ids")),
        )
    )

    # Round-2 context gate
    ctx = service.real_loop_build_context(ids["session_id"], round_number=2)
    results.append(
        _check(
            "round2_context_has_feedback",
            ctx.get("has_round_feedback") is True,
            f"sha={ctx.get('context_sha256')}",
        )
    )

    # FeedbackUseVerifier
    verify = service.real_loop_verify_feedback(ids["session_id"], round_number=2)
    results.append(
        _check(
            "feedback_use_verifier",
            verify.get("pass_status") is True,
            json.dumps(verify.get("issues") or []),
        )
    )
    results.append(
        _check(
            "plan_change_not_duplicate",
            bool(verify.get("checks", {}).get("output_changes_experiment_plan"))
            and bool(verify.get("checks", {}).get("output_avoids_duplicate_candidate")),
            json.dumps(verify.get("checks") or {}),
        )
    )

    # Pure verifier also rejects duplicates
    dup = verify_feedback_use(
        feedback_summary=r1["feedback_summary_json"],
        planning_context=ctx.get("context") or {},
        candidates=[
            {
                "candidate_id": "dup",
                "parent_node_id": ids["executed_id"],
                "parameter_changes": {"hidden_units": 112},
                "rationale": "accuracy",
                "evidence_gap_addressed": ["evidence_accept_v21"],
            }
        ],
        previous_executed_parameters={"hidden_units": 112},
    )
    results.append(
        _check(
            "duplicate_candidate_blocked",
            dup.pass_status is False,
            json.dumps(dup.issues),
        )
    )

    # Export Replay Bundle
    bundle_dir = accept_root / "replay_bundle"
    exported = service.real_loop_export(
        ids["session_id"], output_dir=str(bundle_dir)
    )
    results.append(
        _check(
            "replay_bundle_export",
            exported.get("ok") is True and (bundle_dir / "manifest.json").is_file(),
            str(exported.get("bundle_dir")),
        )
    )
    loaded = load_replay_bundle(bundle_dir)
    results.append(
        _check(
            "replay_manifest",
            loaded["manifest"].get("secrets_redacted") is True
            and loaded["manifest"].get("session_id") == ids["session_id"]
            and bool(loaded["manifest"].get("rounds")),
            json.dumps(
                {
                    "session_id": loaded["manifest"].get("session_id"),
                    "rounds": len(loaded["manifest"].get("rounds") or []),
                }
            ),
        )
    )
    try:
        assert_bundle_redacted(bundle_dir)
        redaction_ok = True
        redaction_detail = "clean"
    except Exception as exc:  # noqa: BLE001
        redaction_ok = False
        redaction_detail = str(exc)
    results.append(_check("secret_scan_clean", redaction_ok, redaction_detail))

    report_path = bundle_dir / "real_llm_closed_loop_report.json"
    results.append(
        _check(
            "closed_loop_report_present",
            report_path.is_file(),
            str(report_path),
        )
    )

    # Unit regression hook (v211–v217)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/unit/test_research_loop_v211.py",
            "tests/unit/test_research_loop_v212.py",
            "tests/unit/test_research_loop_v213.py",
            "tests/unit/test_research_loop_v214.py",
            "tests/unit/test_research_loop_v215.py",
            "tests/unit/test_research_loop_v216.py",
            "tests/unit/test_research_loop_v217.py",
            "-q",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    results.append(
        _check(
            "unit_regression_v211_v217",
            proc.returncode == 0,
            (proc.stdout or proc.stderr or "")[-400:],
        )
    )

    payload = {
        "version": "v2.1.7",
        "session_id": ids["session_id"],
        "network_used": False,
        "passed": sum(1 for r in results if r["ok"]),
        "total": len(results),
        "results": results,
        "bundle_dir": str(bundle_dir),
        "definition": (
            "Offline acceptance for real LLM multi-round closed-loop plumbing "
            "(Replay Bundle + FeedbackUse). Not full autonomous science."
        ),
    }
    (report_dir / "v21_acceptance_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (report_dir / "README.md").write_text(
        """# Scientist Lab v2.1 Acceptance (offline)

## Run

```text
python scripts/accept_v21.py
```

Zero network. Uses Replay Bundle export + FeedbackUseVerifier fixtures.
Real LLM / Digits Docker: `scripts/accept_v21_real.py` (later).
""",
        encoding="utf-8",
    )

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    failed = [r for r in results if not r["ok"]]
    if failed:
        print(f"FAILED {len(failed)}/{len(results)}", file=sys.stderr)
        for item in failed:
            print(f"  - {item['name']}: {item['detail']}", file=sys.stderr)
        return 1
    print(f"PASSED {len(results)}/{len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
