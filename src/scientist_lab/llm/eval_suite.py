"""v1.4.4 LLM evaluation suites with mock / replay / real gates.

Default and CI paths never open network sockets. Real mode is skipped unless
all of: RUN_REAL_LLM_TESTS=1, LLM_ALLOW_NETWORK=true, CLI allow_network,
and provider=real|openai-compatible.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from scientist_lab.agents.critic import MockCritic
from scientist_lab.agents.models import PlanningContext, PlannerOutput
from scientist_lab.agents.planner import MockPlanner
from scientist_lab.planning.candidate_verifier import CandidateVerifier
from scientist_lab.storage.artifact_store import write_json


EvalProvider = Literal["mock", "fake", "replay", "real", "openai-compatible"]


class EvalCase(BaseModel):
    case_id: str
    title: str = ""
    research_goal: str = "Improve RGB-T detection under a fixed protocol."
    current_best_node_id: str = "rgbt_formal_node_003"
    allowed_parameter_changes: list[str] = Field(
        default_factory=lambda: ["input_mode", "fusion_method"]
    )
    expect_stop: bool | None = None
    tags: list[str] = Field(default_factory=list)


class CaseEvalResult(BaseModel):
    case_id: str
    status: Literal["ok", "error", "skipped"] = "ok"
    schema_valid: bool = False
    protocol_compliant: bool = False
    safety_violation: bool = False
    duplicate_candidate: bool = False
    stop_recommended: bool | None = None
    stop_match: bool | None = None
    repair_used: bool = False
    latency_ms: float = 0.0
    total_tokens: int = 0
    estimated_cost_usd: float | None = None
    candidate_count: int = 0
    verified_pass: int = 0
    error: str | None = None
    notes: list[str] = Field(default_factory=list)


class SuiteEvalReport(BaseModel):
    report_id: str
    suite: str
    provider: str
    model: str | None = None
    created_at: str
    case_count: int = 0
    executed_count: int = 0
    skipped: bool = False
    skip_reason: str | None = None
    real_network_called: bool = False
    schema_valid_rate: float = 0.0
    protocol_compliance_rate: float = 0.0
    safety_violation_rate: float = 0.0
    duplicate_candidate_rate: float = 0.0
    stop_decision_accuracy: float | None = None
    repair_rate: float = 0.0
    average_latency_ms: float = 0.0
    average_tokens: float = 0.0
    estimated_cost_usd: float | None = None
    cases: list[CaseEvalResult] = Field(default_factory=list)
    path: str | None = None
    gates: dict[str, bool] = Field(default_factory=dict)


def package_evals_dir() -> Path:
    """``evals/llm`` next to the scientist-lab package root."""
    return Path(__file__).resolve().parents[3] / "evals" / "llm"


def real_eval_gates(
    *,
    provider: str,
    allow_network: bool,
    environ: dict[str, str] | None = None,
) -> tuple[bool, str]:
    """Return (allowed, reason)."""
    env = environ if environ is not None else dict(os.environ)
    name = (provider or "mock").strip().lower()
    if name not in {"real", "openai-compatible"}:
        return True, "offline provider"
    if not allow_network:
        return False, "CLI --allow-network not set"
    if str(env.get("RUN_REAL_LLM_TESTS") or "").strip() not in {"1", "true", "yes", "on"}:
        return False, "RUN_REAL_LLM_TESTS not enabled"
    if str(env.get("LLM_ALLOW_NETWORK") or "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return False, "LLM_ALLOW_NETWORK not true"
    if not (env.get("LLM_API_KEY") or "").strip():
        return False, "LLM_API_KEY not configured"
    if not (env.get("LLM_BASE_URL") or "").strip():
        return False, "LLM_BASE_URL not configured"
    if not (env.get("LLM_MODEL") or "").strip():
        return False, "LLM_MODEL not configured"
    return True, "real gates satisfied"


def resolve_suite_path(suite: str, *, evals_root: Path | None = None) -> Path:
    root = evals_root or package_evals_dir()
    key = suite.strip().lower().replace("_", "-")
    candidates = [
        root / f"{key}.jsonl",
        root / f"{key.replace('-', '_')}.jsonl",
        root / f"{suite}.jsonl",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"eval suite not found: {suite} (looked in {root})")


def load_eval_cases(
    suite: str,
    *,
    evals_root: Path | None = None,
    max_cases: int | None = None,
) -> list[EvalCase]:
    path = resolve_suite_path(suite, evals_root=evals_root)
    cases: list[EvalCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        cases.append(EvalCase.model_validate(json.loads(text)))
        if max_cases is not None and len(cases) >= int(max_cases):
            break
    if not cases:
        raise ValueError(f"eval suite empty: {path}")
    return cases


def _context_from_case(case: EvalCase, *, project_id: str) -> PlanningContext:
    node_id = case.current_best_node_id
    return PlanningContext(
        project_id=project_id,
        research_goal=case.research_goal,
        protocol={
            "protocol_id": "protocol_rgbt_001",
            "allowed_variables": list(case.allowed_parameter_changes),
        },
        protocol_id="protocol_rgbt_001",
        current_best_node_id=node_id,
        nodes=[{"node_id": node_id, "status": "succeeded"}],
        remaining_budget={"max_new_nodes": 3, "max_gpu_hours": 10},
        allowed_parameter_changes=list(case.allowed_parameter_changes),
    )


def _score_output(
    case: EvalCase,
    output: PlannerOutput,
    context: PlanningContext,
    *,
    latency_ms: float,
    total_tokens: int = 0,
    repair_used: bool = False,
    estimated_cost_usd: float | None = None,
) -> CaseEvalResult:
    verifier = CandidateVerifier()
    verified = 0
    safety = False
    duplicate = False
    for cand in output.candidates:
        result = verifier.verify(cand, context)
        if result.valid:
            verified += 1
        else:
            issues = " ".join(result.blocking_issues or []).lower()
            if "blocked" in issues or "safety" in issues:
                safety = True
            if "duplicate" in issues or "fingerprint" in issues:
                duplicate = True
    schema_valid = True  # PlannerOutput already validated by pydantic
    stop = bool(output.stop_recommended)
    stop_match = None
    if case.expect_stop is not None:
        stop_match = stop is bool(case.expect_stop)
    protocol_ok = verified == len(output.candidates) if output.candidates else True
    return CaseEvalResult(
        case_id=case.case_id,
        status="ok",
        schema_valid=schema_valid,
        protocol_compliant=protocol_ok,
        safety_violation=safety,
        duplicate_candidate=duplicate,
        stop_recommended=stop,
        stop_match=stop_match,
        repair_used=repair_used,
        latency_ms=latency_ms,
        total_tokens=total_tokens,
        estimated_cost_usd=estimated_cost_usd,
        candidate_count=len(output.candidates),
        verified_pass=verified,
    )


def _run_mock_case(case: EvalCase, *, project_id: str) -> CaseEvalResult:
    context = _context_from_case(case, project_id=project_id)
    started = time.perf_counter()
    output = MockPlanner().plan(context)
    # Critic pass for smoke (not scored deeply in planner-basic).
    critic = MockCritic()
    for cand in output.candidates:
        critic.review(cand, context)
    latency = (time.perf_counter() - started) * 1000.0
    return _score_output(case, output, context, latency_ms=latency)


def _run_fake_or_replay_case(
    case: EvalCase,
    *,
    project_id: str,
    audit_root: Path,
    mode: Literal["fake", "replay"],
) -> CaseEvalResult:
    from scientist_lab.agents.provider_bridge import build_planner_critic

    context = _context_from_case(case, project_id=project_id)
    planner, critic = build_planner_critic(
        mode, audit_root=audit_root, project_id=project_id
    )
    started = time.perf_counter()
    output = planner.plan(context)
    for cand in output.candidates:
        critic.review(cand, context)
    latency = (time.perf_counter() - started) * 1000.0
    return _score_output(case, output, context, latency_ms=latency)


def run_llm_eval_suite(
    *,
    suite: str,
    provider: str = "mock",
    project_id: str = "project_rgbt_003",
    audit_root: Path | str | None = None,
    evals_root: Path | None = None,
    max_cases: int | None = None,
    allow_network: bool = False,
    environ: dict[str, str] | None = None,
    report_id: str | None = None,
) -> SuiteEvalReport:
    """Run an eval suite. Real provider is skipped unless gates pass."""
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    rid = report_id or f"llm_eval_{suite}_{provider}_{now.replace(':', '').replace('+', 'p')}"
    name = (provider or "mock").strip().lower()
    if name == "openai-compatible":
        name = "real"

    allowed, reason = real_eval_gates(
        provider=name, allow_network=allow_network, environ=environ
    )
    if name == "real" and not allowed:
        return SuiteEvalReport(
            report_id=rid,
            suite=suite,
            provider="real",
            created_at=now,
            case_count=0,
            executed_count=0,
            skipped=True,
            skip_reason=f"real evaluation skipped: {reason}",
            real_network_called=False,
            gates={
                "real_requested": True,
                "real_allowed": False,
                "offline_only": True,
            },
        )

    cases = load_eval_cases(suite, evals_root=evals_root, max_cases=max_cases)
    root = Path(audit_root) if audit_root else Path("outputs") / project_id / "llm"
    root.mkdir(parents=True, exist_ok=True)

    results: list[CaseEvalResult] = []
    model_name: str | None = None
    real_network = False

    if name == "real":
        # Explicit real path still requires a live provider; for safety in
        # default CI we never reach here without gates. Live HTTP is out of
        # accept_v14 offline scope — mark as error if somehow misconfigured.
        return SuiteEvalReport(
            report_id=rid,
            suite=suite,
            provider="real",
            model=(environ or os.environ).get("LLM_MODEL"),
            created_at=now,
            case_count=len(cases),
            executed_count=0,
            skipped=True,
            skip_reason=(
                "real evaluation gated OK but live HTTP suite runner is "
                "intentionally not auto-executed in offline package defaults; "
                "use a dedicated smoke harness with MockTransport or explicit "
                "integration job"
            ),
            real_network_called=False,
            gates={
                "real_requested": True,
                "real_allowed": True,
                "live_suite_auto_run": False,
                "offline_only": True,
            },
        )

    for case in cases:
        try:
            if name == "mock":
                result = _run_mock_case(case, project_id=project_id)
                model_name = "mock-planner-v1"
            elif name == "fake":
                result = _run_fake_or_replay_case(
                    case, project_id=project_id, audit_root=root, mode="fake"
                )
                model_name = "fake-llm-v1"
            elif name == "replay":
                # Ensure fake seed exists for fingerprints when suite starts cold.
                _run_fake_or_replay_case(
                    case, project_id=project_id, audit_root=root, mode="fake"
                )
                result = _run_fake_or_replay_case(
                    case, project_id=project_id, audit_root=root, mode="replay"
                )
                model_name = "replay-v1"
            else:
                result = CaseEvalResult(
                    case_id=case.case_id,
                    status="error",
                    error=f"unsupported eval provider: {provider}",
                )
            results.append(result)
        except Exception as exc:  # noqa: BLE001
            results.append(
                CaseEvalResult(
                    case_id=case.case_id,
                    status="error",
                    error=str(exc)[:300],
                )
            )

    executed = [r for r in results if r.status == "ok"]
    n = len(executed) or 1
    stop_pairs = [r for r in executed if r.stop_match is not None]
    report = SuiteEvalReport(
        report_id=rid,
        suite=suite,
        provider=name,
        model=model_name,
        created_at=now,
        case_count=len(cases),
        executed_count=len(executed),
        skipped=False,
        real_network_called=real_network,
        schema_valid_rate=sum(1 for r in executed if r.schema_valid) / n,
        protocol_compliance_rate=sum(1 for r in executed if r.protocol_compliant) / n,
        safety_violation_rate=sum(1 for r in executed if r.safety_violation) / n,
        duplicate_candidate_rate=sum(1 for r in executed if r.duplicate_candidate) / n,
        stop_decision_accuracy=(
            sum(1 for r in stop_pairs if r.stop_match) / len(stop_pairs)
            if stop_pairs
            else None
        ),
        repair_rate=sum(1 for r in executed if r.repair_used) / n,
        average_latency_ms=sum(r.latency_ms for r in executed) / n,
        average_tokens=sum(r.total_tokens for r in executed) / n,
        estimated_cost_usd=None,
        cases=results,
        gates={
            "real_requested": False,
            "offline_only": True,
            "suite_ok": all(r.status == "ok" for r in results),
        },
    )
    return report


def write_suite_report(
    report: SuiteEvalReport,
    output_path: Path | str,
) -> SuiteEvalReport:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = report.model_dump(mode="json")
    payload["path"] = str(path)
    write_json(path, payload)
    return report.model_copy(update={"path": str(path)})
