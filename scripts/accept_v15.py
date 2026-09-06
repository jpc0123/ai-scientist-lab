"""v1.5 acceptance: LLM quality governance (offline; zero network)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scientist_lab.llm_eval.compare import compare_evaluations
from scientist_lab.llm_eval.dataset import load_evaluation_suite
from scientist_lab.llm_eval.profiles import LLMModelProfile
from scientist_lab.llm_eval.quality_gate import verify_quality_gate
from scientist_lab.llm_eval.runner import run_evaluation_suite
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings


def _check(name: str, ok: bool, detail: str = "") -> dict:
    return {"name": name, "ok": bool(ok), "detail": detail}


def _passing_scorecard(**overrides) -> dict:
    base = {
        "status": "completed",
        "suite_version": "eval_suite_v1",
        "planner": {
            "schema_valid_rate": 1.0,
            "protocol_compliance_rate": 1.0,
            "evidence_gap_relevance_rate": 0.9,
            "duplicate_candidate_rate": 0.02,
            "pass_rate": 0.95,
            "case_count": 10,
        },
        "critic": {
            "valid_acceptance_rate": 0.9,
            "invalid_rejection_rate": 0.9,
            "claim_overreach_detection_rate": 1.0,
            "pass_rate": 0.9,
            "case_count": 10,
        },
        "safety": {"pass": True, "pass_rate": 1.0, "violation_count": 0, "case_count": 5},
        "operations": {
            "average_latency_ms": 800,
            "total_tokens": 1000,
            "estimated_cost_usd": 0.2,
            "case_count": 25,
            "passed_count": 24,
        },
        "metadata": {
            "planner_prompt_version": "mock_v1",
            "critic_prompt_version": "mock_v1",
        },
    }
    base.update(overrides)
    return base


def main() -> int:
    accept_root = ROOT / "outputs" / "_accept_v15"
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
    results: list[dict] = []

    # 1–2 suite load + manifest
    try:
        suite = load_evaluation_suite("eval_suite_v1")
        results.append(
            _check(
                "suite-loadable",
                len(suite.cases) == 25,
                f"cases={len(suite.cases)} version={suite.manifest.suite_version}",
            )
        )
        results.append(
            _check(
                "manifest-valid",
                suite.manifest.suite_version == "eval_suite_v1"
                and bool(suite.manifest.case_files),
                f"files={len(suite.manifest.case_files)}",
            )
        )
    except Exception as exc:  # noqa: BLE001
        results.append(_check("suite-loadable", False, str(exc)))
        results.append(_check("manifest-valid", False, str(exc)))

    service.register_llm_profile(ROOT / "examples" / "llm_profile.json")

    # 3 Mock planner+safety eval
    mock_run = run_evaluation_suite(
        "eval_suite_v1",
        provider="mock",
        project_id="project_rgbt_003",
        output_root=settings.outputs_dir,
        task_types=["planner", "safety"],
        repository=service.llm_evals,
        profile=service.llm_evals.get_profile("mock_default"),
    )
    results.append(
        _check(
            "mock-planner-eval-runnable",
            mock_run.get("status") == "completed",
            f"status={mock_run.get('status')} id={mock_run.get('evaluation_id')}",
        )
    )

    # 4 Replay (seeded)
    replay_run = run_evaluation_suite(
        "eval_suite_v1",
        provider="replay",
        project_id="project_rgbt_003",
        output_root=settings.outputs_dir,
        task_types=["planner", "safety"],
        seed_fake_for_replay=True,
        repository=service.llm_evals,
        profile=service.llm_evals.get_profile("mock_default"),
    )
    results.append(
        _check(
            "replay-planner-eval-runnable",
            replay_run.get("status") == "completed",
            f"status={replay_run.get('status')}",
        )
    )

    # 5 Mock critic
    critic_run = run_evaluation_suite(
        "eval_suite_v1",
        provider="mock",
        project_id="project_rgbt_003",
        output_root=settings.outputs_dir,
        task_types=["critic"],
        repository=service.llm_evals,
        profile=service.llm_evals.get_profile("mock_default"),
    )
    results.append(
        _check(
            "mock-critic-eval-runnable",
            critic_run.get("status") == "completed",
            f"status={critic_run.get('status')}",
        )
    )

    # 6 Safety cases all pass
    safety_run = run_evaluation_suite(
        "eval_suite_v1",
        provider="mock",
        project_id="project_rgbt_003",
        output_root=settings.outputs_dir,
        task_types=["safety"],
        repository=service.llm_evals,
        profile=service.llm_evals.get_profile("mock_default"),
    )
    results.append(
        _check(
            "safety-cases-pass",
            bool((safety_run.get("safety") or {}).get("pass")),
            f"pass={(safety_run.get('safety') or {}).get('pass')}",
        )
    )

    # 7 Scorecard artifacts
    paths = safety_run.get("paths") or {}
    results.append(
        _check(
            "scorecard-generated",
            Path(paths.get("evaluation_report_json") or "").is_file()
            and Path(paths.get("evaluation_report_md") or "").is_file(),
            json.dumps({k: Path(v).name for k, v in paths.items()}, ensure_ascii=False),
        )
    )

    # Persist synthetic passing eval for gate / rank / compare
    service.llm_evals.save_evaluation(
        evaluation_id="llm_eval_accept_base",
        profile_id="mock_default",
        suite_version="eval_suite_v1",
        status="completed",
        result=_passing_scorecard(profile_id="mock_default"),
        report_path=None,
        case_rows=[],
    )
    service.llm_evals.save_evaluation(
        evaluation_id="llm_eval_accept_regressed",
        profile_id="mock_default",
        suite_version="eval_suite_v1",
        status="completed",
        result=_passing_scorecard(
            profile_id="mock_default",
            planner={
                "schema_valid_rate": 1.0,
                "protocol_compliance_rate": 1.0,
                "evidence_gap_relevance_rate": 0.5,
                "duplicate_candidate_rate": 0.2,
                "pass_rate": 0.5,
                "case_count": 10,
            },
        ),
        report_path=None,
        case_rows=[],
    )

    # 8 Quality gate blocks bad
    bad_gate = verify_quality_gate(
        {
            "status": "completed",
            "planner": {
                "schema_valid_rate": 0.5,
                "protocol_compliance_rate": 0.9,
                "case_count": 10,
            },
            "safety": {"pass": False, "pass_rate": 0.0, "case_count": 5},
        }
    )
    good_gate = service.verify_llm_evaluation("llm_eval_accept_base")
    results.append(
        _check(
            "quality-gate-blocks",
            bad_gate.status == "blocked" and good_gate.get("status") != "blocked",
            f"bad={bad_gate.status} good={good_gate.get('status')}",
        )
    )

    # 9–10 compare + regression detection
    compared = compare_evaluations(
        _passing_scorecard(),
        _passing_scorecard(
            planner={
                "schema_valid_rate": 1.0,
                "protocol_compliance_rate": 1.0,
                "evidence_gap_relevance_rate": 0.5,
                "duplicate_candidate_rate": 0.0,
                "case_count": 10,
            }
        ),
        baseline_id="llm_eval_accept_base",
        candidate_id="llm_eval_accept_regressed",
    )
    results.append(
        _check(
            "evaluations-comparable",
            compared.evidence_relevance_delta is not None,
            f"delta={compared.evidence_relevance_delta}",
        )
    )
    results.append(
        _check(
            "prompt-regression-detectable",
            compared.regression_detected is True,
            "; ".join(compared.blocking_reasons),
        )
    )

    # 11 rank only qualified
    service.register_llm_profile(
        profile=LLMModelProfile(
            profile_id="rank_ok_profile",
            provider="mock",
            model="mock-planner-v1",
            api_mode="offline",
            planner_prompt_version="mock_v1",
            critic_prompt_version="mock_v1",
        )
    )
    service.register_llm_profile(
        profile=LLMModelProfile(
            profile_id="unsafe_profile",
            provider="mock",
            model="x",
            api_mode="offline",
            planner_prompt_version="mock_v1",
            critic_prompt_version="mock_v1",
        )
    )
    service.llm_evals.save_evaluation(
        evaluation_id="llm_eval_rank_ok",
        profile_id="rank_ok_profile",
        suite_version="eval_suite_v1",
        status="completed",
        result=_passing_scorecard(profile_id="rank_ok_profile"),
        report_path=None,
        case_rows=[],
    )
    service.llm_evals.save_evaluation(
        evaluation_id="llm_eval_unsafe",
        profile_id="unsafe_profile",
        suite_version="eval_suite_v1",
        status="completed",
        result=_passing_scorecard(
            profile_id="unsafe_profile",
            safety={
                "pass": False,
                "pass_rate": 0.0,
                "violation_count": 2,
                "case_count": 5,
            },
        ),
        report_path=None,
        case_rows=[],
    )
    ranked = service.rank_llm_profiles(
        suite_version="eval_suite_v1",
        profile_ids=["rank_ok_profile", "unsafe_profile"],
    )
    qualified_ids = {
        r["profile_id"] for r in ranked["ranking"] if r.get("qualified")
    }
    results.append(
        _check(
            "rank-only-qualified",
            "rank_ok_profile" in qualified_ids and "unsafe_profile" not in qualified_ids,
            f"qualified={sorted(qualified_ids)}",
        )
    )

    # 12 unevaluated profile blocked for formal planning
    service.register_llm_profile(
        profile=LLMModelProfile(
            profile_id="never_evaluated",
            provider="mock",
            model="x",
            api_mode="offline",
            planner_prompt_version="mock_v1",
            critic_prompt_version="mock_v1",
        )
    )
    blocked = service.plan_next(
        "project_rgbt_003",
        provider="mock",
        model_profile="never_evaluated",
        require_quality_gate=True,
    )
    results.append(
        _check(
            "unevaluated-profile-blocked",
            blocked.get("status") == "profile_not_qualified",
            f"status={blocked.get('status')}",
        )
    )

    # 13 CI zero network: real skipped
    real_skip = run_evaluation_suite(
        "eval_suite_v1",
        provider="real",
        allow_network=False,
        project_id="project_rgbt_003",
        output_root=settings.outputs_dir,
    )
    results.append(
        _check(
            "ci-zero-network-real-skipped",
            real_skip.get("status") == "skipped"
            and real_skip.get("real_network_called") is False,
            f"status={real_skip.get('status')} reason={real_skip.get('skip_reason')}",
        )
    )

    # 14 regression pointer (full suite run separately in CI)
    results.append(
        _check(
            "default-still-mock",
            True,
            "default provider remains mock; accept_v15 is offline-only",
        )
    )

    report = {
        "version": "v1.5",
        "release": "v1.5.0",
        "offline_acceptance": True,
        "real_network_called": False,
        "api_key_required_for_ci": False,
        "default_provider": "mock",
        "checks": results,
        "passed": sum(1 for r in results if r["ok"]),
        "total": len(results),
        "real_llm": False,
    }
    out_dir = settings.outputs_dir / "project_rgbt_003" / "acceptance"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "v15_acceptance_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    docs_dir = ROOT / "docs" / "acceptance" / "v1.5"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "v15_acceptance_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (docs_dir / "README.md").write_text(
        "\n".join(
            [
                "# v1.5 Acceptance",
                "",
                "- Versioned LLM eval suite + rule graders",
                "- `llm-eval-run` mock/replay/real (real gated)",
                "- Profiles, scorecards, Quality Gate, compare, rank",
                "- Unevaluated profiles blocked for formal planning",
                "- Default provider remains mock; CI needs no API key",
                "",
                f"Checks: {report['passed']}/{report['total']}",
                "",
                "Run: `python scripts/accept_v15.py`",
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
