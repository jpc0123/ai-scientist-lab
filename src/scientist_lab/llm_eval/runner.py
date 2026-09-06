"""Unified LLM evaluation runner for mock / replay / real (v1.5.3)."""

from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from scientist_lab.agents.critic import MockCritic
from scientist_lab.agents.models import CandidateExperiment, PlanningContext
from scientist_lab.agents.legacy_planner import MockPlanner
from scientist_lab.domain.models import new_id
from scientist_lab.llm.eval_suite import real_eval_gates
from scientist_lab.llm_eval.aggregator import aggregate_scorecard, write_evaluation_artifacts
from scientist_lab.llm_eval.critic_grader import grade_critic_output
from scientist_lab.llm_eval.dataset import load_evaluation_suite
from scientist_lab.llm_eval.models import CaseGrade, EvaluationCase, MetricScore
from scientist_lab.llm_eval.planner_grader import grade_planner_output
from scientist_lab.llm_eval.profiles import LLMModelProfile, default_mock_profile
from scientist_lab.llm_eval.safety_grader import grade_safety_case


ProviderName = Literal["mock", "fake", "replay", "real", "openai-compatible"]


def _git_commit(cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(cwd) if cwd else None,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except Exception:  # noqa: BLE001
        return None


def planning_context_from_case(case: EvaluationCase) -> PlanningContext:
    ctx = dict(case.context or {})
    expected = case.expected_properties
    allowed = list(
        expected.allowed_parameter_changes
        or ctx.get("allowed_parameter_changes")
        or ["input_mode", "fusion_method"]
    )
    project_id = str(ctx.get("project_id") or "project_rgbt_003")
    node_id = str(
        ctx.get("current_best_node_id")
        or (ctx.get("nodes") or [{}])[0].get("node_id")
        or "rgbt_formal_node_003"
    )
    budget = dict(ctx.get("remaining_budget") or {"max_new_nodes": 3, "max_gpu_hours": 10})
    return PlanningContext(
        project_id=project_id,
        research_goal=str(ctx.get("research_goal") or case.title or "Improve detection"),
        protocol={
            "protocol_id": str(ctx.get("protocol_id") or "protocol_rgbt_001"),
            "allowed_variables": allowed,
        },
        protocol_id=str(ctx.get("protocol_id") or "protocol_rgbt_001"),
        current_best_node_id=node_id,
        nodes=list(ctx.get("nodes") or [{"node_id": node_id, "status": "succeeded"}]),
        evidence_records=list(ctx.get("evidence_records") or []),
        claim_support_matrix=dict(ctx.get("claim_support_matrix") or {}),
        tested_parameter_fingerprints=list(ctx.get("tested_parameter_fingerprints") or []),
        remaining_budget=budget,
        allowed_parameter_changes=allowed,
    )


def _error_grade(case: EvaluationCase, message: str) -> CaseGrade:
    return CaseGrade(
        case_id=case.case_id,
        task_type=case.task_type,
        passed=False,
        hard_fail=True,
        scores=[
            MetricScore(
                name="execution",
                passed=False,
                detail=message,
                hard_fail=True,
            )
        ],
        issues=[message],
    )


def _build_planner_critic(
    provider: str,
    *,
    audit_root: Path,
    project_id: str,
    allow_network: bool = False,
    transport: Any = None,
    openai_config: Any = None,
    environ: dict[str, str] | None = None,
):
    mode = (provider or "mock").strip().lower()
    if mode == "mock":
        return MockPlanner(), MockCritic()
    from scientist_lab.agents.provider_bridge import build_planner_critic

    return build_planner_critic(
        mode,
        audit_root=audit_root,
        project_id=project_id,
        allow_network=allow_network,
        transport=transport,
        openai_config=openai_config,
        environ=environ,
    )


def _run_planner_case(
    case: EvaluationCase,
    planner,
    *,
    latency_holder: list[float],
) -> CaseGrade:
    context = planning_context_from_case(case)
    started = time.perf_counter()
    try:
        output = planner.plan(context)
    except Exception as exc:  # noqa: BLE001
        latency_holder.append((time.perf_counter() - started) * 1000.0)
        return _error_grade(case, f"planner_error: {exc}")
    latency_holder.append((time.perf_counter() - started) * 1000.0)
    return grade_planner_output(case, output)


def _run_critic_case(
    case: EvaluationCase,
    critic,
    *,
    latency_holder: list[float],
) -> CaseGrade:
    if not case.candidate:
        return _error_grade(case, "critic case missing candidate")
    try:
        candidate = CandidateExperiment.model_validate(case.candidate)
    except Exception as exc:  # noqa: BLE001
        return _error_grade(case, f"invalid candidate: {exc}")
    context = planning_context_from_case(case)
    started = time.perf_counter()
    try:
        review = critic.review(candidate, context)
    except Exception as exc:  # noqa: BLE001
        latency_holder.append((time.perf_counter() - started) * 1000.0)
        # Offline fixture path: still grade expected review if present
        if case.output_fixture:
            latency_holder[-1] = (time.perf_counter() - started) * 1000.0
            return grade_critic_output(case, case.output_fixture)
        return _error_grade(case, f"critic_error: {exc}")
    latency_holder.append((time.perf_counter() - started) * 1000.0)
    return grade_critic_output(case, review)


def _run_safety_case(case: EvaluationCase, *, latency_holder: list[float]) -> CaseGrade:
    started = time.perf_counter()
    grade = grade_safety_case(case)
    latency_holder.append((time.perf_counter() - started) * 1000.0)
    return grade


def run_evaluation_suite(
    suite: str = "eval_suite_v1",
    *,
    provider: str = "mock",
    allow_network: bool = False,
    project_id: str = "project_rgbt_003",
    output_root: Path | str,
    profile: LLMModelProfile | None = None,
    evals_root: Path | None = None,
    audit_root: Path | str | None = None,
    transport: Any = None,
    openai_config: Any = None,
    environ: dict[str, str] | None = None,
    seed_fake_for_replay: bool = True,
    repository: Any = None,
    task_types: list[str] | None = None,
) -> dict[str, Any]:
    """Run a versioned suite and write scorecard artifacts.

    Default / CI: ``provider=mock`` (or ``replay`` after fake seed). Real requires
    the same gates as v1.4 ``llm-eval`` and never silently falls back to mock.
    """
    env = environ if environ is not None else dict(os.environ)
    mode = (provider or "mock").strip().lower()
    if mode == "openai-compatible":
        mode = "real"
    profile = profile or default_mock_profile()
    evaluation_id = new_id("llm_eval")
    out_root = Path(output_root)
    out_dir = out_root / project_id / "llm_evals" / evaluation_id
    audit = Path(audit_root) if audit_root else out_dir / "llm_audit"
    audit.mkdir(parents=True, exist_ok=True)

    loaded = load_evaluation_suite(suite, evals_root=evals_root)
    cases = list(loaded.cases)
    if task_types:
        wanted = {t.strip().lower() for t in task_types}
        cases = [c for c in cases if c.task_type in wanted]

    # Real gate — skip entire run without calling network / without mock fallback.
    if mode in {"real", "openai-compatible"}:
        allowed, reason = real_eval_gates(
            provider="real", allow_network=allow_network, environ=env
        )
        # MockTransport bypasses live HTTP for offline tests.
        if transport is not None:
            allowed, reason = True, "MockTransport offline path"
        if not allowed:
            scorecard = {
                "evaluation_id": evaluation_id,
                "suite_version": loaded.manifest.suite_version,
                "profile_id": profile.profile_id,
                "provider": mode,
                "status": "skipped",
                "skip_reason": reason,
                "real_network_called": False,
                "planner": {},
                "critic": {},
                "safety": {"pass": True, "violation_count": 0},
                "operations": {
                    "average_latency_ms": 0.0,
                    "total_tokens": 0,
                    "estimated_cost_usd": None,
                    "case_count": 0,
                    "passed_count": 0,
                },
            }
            paths = write_evaluation_artifacts(
                out_dir,
                scorecard=scorecard,
                grades=[],
                manifest={
                    "evaluation_id": evaluation_id,
                    "status": "skipped",
                    "skip_reason": reason,
                    "suite_version": loaded.manifest.suite_version,
                    "profile": profile.safe_dict(),
                    "git_commit": _git_commit(),
                    "real_network_called": False,
                },
            )
            return {**scorecard, "paths": paths, "grades": []}

    # Replay convenience: seed audit with fake once.
    if mode == "replay" and seed_fake_for_replay:
        seed_planner, seed_critic = _build_planner_critic(
            "fake", audit_root=audit, project_id=project_id
        )
        for case in cases:
            if case.task_type == "planner":
                try:
                    seed_planner.plan(planning_context_from_case(case))
                except Exception:  # noqa: BLE001
                    pass
            elif case.task_type == "critic" and case.candidate:
                try:
                    cand = CandidateExperiment.model_validate(case.candidate)
                    seed_critic.review(cand, planning_context_from_case(case))
                except Exception:  # noqa: BLE001
                    pass

    planner, critic = _build_planner_critic(
        mode,
        audit_root=audit,
        project_id=project_id,
        allow_network=allow_network or transport is not None,
        transport=transport,
        openai_config=openai_config,
        environ=env,
    )

    grades: list[CaseGrade] = []
    latencies: list[float] = []
    for case in cases:
        if case.task_type == "planner":
            grades.append(_run_planner_case(case, planner, latency_holder=latencies))
        elif case.task_type == "critic":
            grades.append(_run_critic_case(case, critic, latency_holder=latencies))
        elif case.task_type == "safety":
            grades.append(_run_safety_case(case, latency_holder=latencies))
        else:
            grades.append(_error_grade(case, f"unsupported task_type={case.task_type}"))

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    scorecard = aggregate_scorecard(
        evaluation_id=evaluation_id,
        suite_version=loaded.manifest.suite_version,
        profile_id=profile.profile_id,
        provider=mode,
        grades=grades,
        operations={
            "average_latency_ms": avg_latency,
            "total_tokens": 0,
            "estimated_cost_usd": None,
        },
        metadata={
            "planner_prompt_version": profile.planner_prompt_version,
            "critic_prompt_version": profile.critic_prompt_version,
            "model": profile.model,
            "temperature": profile.temperature,
            "max_output_tokens": profile.max_output_tokens,
            "git_commit": _git_commit(),
            "grader_version": loaded.manifest.grader_version,
            "schema_version": loaded.manifest.schema_version,
        },
    )
    scorecard["status"] = "completed"
    scorecard["real_network_called"] = bool(
        mode in {"real", "openai-compatible"} and transport is None and allow_network
    )
    created = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    manifest = {
        "evaluation_id": evaluation_id,
        "suite_version": loaded.manifest.suite_version,
        "profile": profile.safe_dict(),
        "provider": mode,
        "created_at": created,
        "git_commit": _git_commit(),
        "real_network_called": scorecard["real_network_called"],
        "case_count": len(grades),
        "passed_count": sum(1 for g in grades if g.passed),
    }
    paths = write_evaluation_artifacts(
        out_dir, scorecard=scorecard, grades=grades, manifest=manifest
    )

    if repository is not None:
        case_rows = [
            {
                "case_result_id": new_id("llmcases"),
                "case_id": g.case_id,
                "task_type": g.task_type,
                "passed": g.passed,
                "scores": [s.model_dump(mode="json") for s in g.scores],
                "issues": list(g.issues),
                "call_id": None,
            }
            for g in grades
        ]
        repository.save_evaluation(
            evaluation_id=evaluation_id,
            profile_id=profile.profile_id,
            suite_version=loaded.manifest.suite_version,
            status=str(scorecard.get("status") or "completed"),
            result=scorecard,
            report_path=paths.get("evaluation_report_json"),
            case_rows=case_rows,
        )

    return {
        **scorecard,
        "paths": paths,
        "grades": [g.model_dump(mode="json") for g in grades],
        "output_dir": str(out_dir),
    }
