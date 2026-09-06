"""Offline Mock / Fake / Replay quality evaluation (v1.3.8).

Real cloud provider comparison is reported as deferred until a Real provider
ships. This module never opens network sockets.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from scientist_lab.agents.critic import CriticReview, MockCritic
from scientist_lab.agents.models import CandidateExperiment, PlanningContext, PlannerOutput
from scientist_lab.agents.legacy_planner import MockPlanner
from scientist_lab.llm.context_codec import planning_context_to_planner_request
from scientist_lab.llm.limits import ProviderLimits, estimate_cost_usd
from scientist_lab.llm.models import TokenUsage
from scientist_lab.llm.provider import request_fingerprint
from scientist_lab.llm.repository import LLMCallRepository
from scientist_lab.planning.candidate_verifier import CandidateVerifier
from scientist_lab.storage.artifact_store import write_json


ModeName = Literal["mock", "fake", "replay", "real"]


class ModeQualityResult(BaseModel):
    mode: ModeName
    status: Literal["ok", "skipped", "error"] = "ok"
    planner_ok: bool = False
    critic_ok: bool = False
    schema_valid: bool = False
    candidate_count: int = 0
    verified_pass: int = 0
    critic_accept: int = 0
    determinism_match: bool | None = None
    replay_match: bool | None = None
    call_count: int = 0
    total_tokens: int = 0
    mean_latency_ms: float = 0.0
    estimated_cost_usd: float = 0.0
    notes: list[str] = Field(default_factory=list)
    error: str | None = None


class ProviderQualityReport(BaseModel):
    report_id: str
    project_id: str
    created_at: str
    real_deferred: bool = True
    modes: list[ModeQualityResult] = Field(default_factory=list)
    comparison: dict[str, Any] = Field(default_factory=dict)
    gates: dict[str, bool] = Field(default_factory=dict)
    path: str | None = None


def _score_candidates(
    output: PlannerOutput,
    context: PlanningContext,
    critic_reviews: list[CriticReview],
) -> tuple[int, int, int]:
    verifier = CandidateVerifier()
    verified = 0
    for candidate in output.candidates:
        result = verifier.verify(candidate, context)
        if result.valid:
            verified += 1
    accepts = sum(1 for r in critic_reviews if r.recommendation == "accept")
    return len(output.candidates), verified, accepts


def _review_all(
    critic: Any,
    candidates: list[CandidateExperiment],
    context: PlanningContext,
) -> list[CriticReview]:
    return [critic.review(c, context) for c in candidates]


def _eval_mock(context: PlanningContext) -> ModeQualityResult:
    planner = MockPlanner()
    critic = MockCritic()
    output = planner.plan(context)
    reviews = _review_all(critic, list(output.candidates), context)
    n, verified, accepts = _score_candidates(output, context, reviews)
    return ModeQualityResult(
        mode="mock",
        status="ok",
        planner_ok=True,
        critic_ok=True,
        schema_valid=True,
        candidate_count=n,
        verified_pass=verified,
        critic_accept=accepts,
        notes=["MockPlanner/MockCritic; no LLM tokens."],
    )


def _eval_fake(
    context: PlanningContext,
    audit_root: Path,
    *,
    limits: ProviderLimits | None = None,
) -> tuple[ModeQualityResult, PlannerOutput, list[CriticReview]]:
    from scientist_lab.agents.provider_bridge import (
        ProviderCritic,
        ProviderPlanner,
        build_planner_critic,
    )

    rates = limits or ProviderLimits()
    planner, critic = build_planner_critic(
        "fake", audit_root=audit_root, project_id=context.project_id
    )
    assert isinstance(planner, ProviderPlanner)
    assert isinstance(critic, ProviderCritic)

    output = planner.plan(context)
    output2 = planner.plan(context)
    determinism = output.model_dump(mode="json") == output2.model_dump(mode="json")
    reviews = _review_all(critic, list(output.candidates), context)
    n, verified, accepts = _score_candidates(output, context, reviews)

    repo = LLMCallRepository(audit_root)
    calls = repo.list_calls(limit=200)
    # Prefer calls from this project when tagged.
    project_calls = [
        c
        for c in calls
        if (c.project_id or "") in {"", context.project_id}
        or c.project_id == context.project_id
    ]
    if not project_calls:
        project_calls = calls
    total_tokens = sum(int((c.usage.total_tokens if c.usage else 0) or 0) for c in project_calls)
    latencies = [float(c.latency_ms or 0.0) for c in project_calls]
    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
    cost = sum(
        estimate_cost_usd(c.usage or TokenUsage(), rates) for c in project_calls
    )
    schema_ok = all(bool(c.schema_valid) for c in project_calls) if project_calls else False

    result = ModeQualityResult(
        mode="fake",
        status="ok",
        planner_ok=True,
        critic_ok=True,
        schema_valid=schema_ok,
        candidate_count=n,
        verified_pass=verified,
        critic_accept=accepts,
        determinism_match=determinism,
        call_count=len(project_calls),
        total_tokens=total_tokens,
        mean_latency_ms=mean_latency,
        estimated_cost_usd=round(cost, 8),
        notes=["Auditing FakeProvider; offline synthetic cost rates."],
    )
    return result, output, reviews


def _eval_replay(
    context: PlanningContext,
    audit_root: Path,
    *,
    fake_output: PlannerOutput,
    fake_reviews: list[CriticReview],
    limits: ProviderLimits | None = None,
) -> ModeQualityResult:
    from scientist_lab.agents.provider_bridge import build_planner_critic

    rates = limits or ProviderLimits()
    planner, critic = build_planner_critic(
        "replay", audit_root=audit_root, project_id=context.project_id
    )
    output = planner.plan(context)
    reviews = _review_all(critic, list(output.candidates), context)
    n, verified, accepts = _score_candidates(output, context, reviews)
    match = output.model_dump(mode="json") == fake_output.model_dump(mode="json")
    review_match = [r.model_dump(mode="json") for r in reviews] == [
        r.model_dump(mode="json") for r in fake_reviews
    ]

    # Replay does not write new audit records by default; estimate from request size.
    req = planning_context_to_planner_request(context)
    repo = LLMCallRepository(audit_root)
    recorded = repo.get_by_fingerprint(request_fingerprint(req))
    total_tokens = int((recorded.usage.total_tokens if recorded and recorded.usage else 0) or 0)
    latency = float(recorded.latency_ms if recorded else 0.0)
    cost = (
        estimate_cost_usd(recorded.usage or TokenUsage(), rates) if recorded else 0.0
    )

    return ModeQualityResult(
        mode="replay",
        status="ok",
        planner_ok=True,
        critic_ok=True,
        schema_valid=True,
        candidate_count=n,
        verified_pass=verified,
        critic_accept=accepts,
        replay_match=match and review_match,
        call_count=1 if recorded else 0,
        total_tokens=total_tokens,
        mean_latency_ms=latency,
        estimated_cost_usd=round(cost, 8),
        notes=[
            "ReplayProvider reads Fake audit fingerprints.",
            f"planner_match={match}",
            f"critic_match={review_match}",
        ],
    )


def evaluate_provider_quality(
    context: PlanningContext,
    *,
    audit_root: Path | str,
    include_real: bool = False,
    limits: ProviderLimits | None = None,
    report_id: str | None = None,
) -> ProviderQualityReport:
    """Compare Mock / Fake / Replay quality on one PlanningContext (offline)."""
    root = Path(audit_root)
    root.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    rid = report_id or f"llm_quality_{context.project_id}_{now.replace(':', '').replace('+', 'p')}"

    modes: list[ModeQualityResult] = []
    fake_output: PlannerOutput | None = None
    fake_reviews: list[CriticReview] = []

    try:
        modes.append(_eval_mock(context))
    except Exception as exc:  # noqa: BLE001
        modes.append(
            ModeQualityResult(mode="mock", status="error", error=str(exc))
        )

    try:
        fake_result, fake_output, fake_reviews = _eval_fake(
            context, root, limits=limits
        )
        modes.append(fake_result)
    except Exception as exc:  # noqa: BLE001
        modes.append(
            ModeQualityResult(mode="fake", status="error", error=str(exc))
        )

    try:
        if fake_output is None:
            raise RuntimeError("fake evaluation failed; cannot run replay")
        modes.append(
            _eval_replay(
                context,
                root,
                fake_output=fake_output,
                fake_reviews=fake_reviews,
                limits=limits,
            )
        )
    except Exception as exc:  # noqa: BLE001
        modes.append(
            ModeQualityResult(mode="replay", status="error", error=str(exc))
        )

    if include_real:
        modes.append(
            ModeQualityResult(
                mode="real",
                status="skipped",
                notes=["Real cloud provider not enabled in v1.3; deferred."],
            )
        )
    else:
        modes.append(
            ModeQualityResult(
                mode="real",
                status="skipped",
                notes=["Real provider deferred; pass include_real=True to mark explicitly."],
            )
        )

    by_mode = {m.mode: m for m in modes}
    fake_m = by_mode.get("fake")
    replay_m = by_mode.get("replay")
    mock_m = by_mode.get("mock")
    comparison = {
        "fake_determinism": bool(fake_m and fake_m.determinism_match),
        "fake_replay_identical": bool(replay_m and replay_m.replay_match),
        "mock_vs_fake_candidate_delta": (
            (fake_m.candidate_count if fake_m else 0)
            - (mock_m.candidate_count if mock_m else 0)
        ),
        "all_offline": True,
        "real_status": by_mode["real"].status if "real" in by_mode else "skipped",
    }
    gates = {
        "mock_ok": bool(mock_m and mock_m.status == "ok" and mock_m.planner_ok),
        "fake_ok": bool(
            fake_m
            and fake_m.status == "ok"
            and fake_m.schema_valid
            and fake_m.determinism_match
        ),
        "replay_ok": bool(
            replay_m and replay_m.status == "ok" and replay_m.replay_match
        ),
        "real_deferred": True,
        "offline_only": True,
    }
    return ProviderQualityReport(
        report_id=rid,
        project_id=context.project_id,
        created_at=now,
        real_deferred=True,
        modes=modes,
        comparison=comparison,
        gates=gates,
    )


def write_quality_report(
    report: ProviderQualityReport,
    output_path: Path | str,
) -> ProviderQualityReport:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = report.model_dump(mode="json")
    payload["path"] = str(path)
    write_json(path, payload)
    return report.model_copy(update={"path": str(path)})


def summarize_audit_usage(
    audit_root: Path | str,
    *,
    limits: ProviderLimits | None = None,
    limit: int = 500,
) -> dict[str, Any]:
    """Aggregate token / cost / latency from persisted LLM audit records."""
    rates = limits or ProviderLimits()
    repo = LLMCallRepository(Path(audit_root))
    calls = repo.list_calls(limit=limit)
    total_prompt = sum(int((c.usage.prompt_tokens if c.usage else 0) or 0) for c in calls)
    total_completion = sum(
        int((c.usage.completion_tokens if c.usage else 0) or 0) for c in calls
    )
    total_tokens = sum(int((c.usage.total_tokens if c.usage else 0) or 0) for c in calls)
    cost = sum(estimate_cost_usd(c.usage or TokenUsage(), rates) for c in calls)
    latencies = [float(c.latency_ms or 0.0) for c in calls]
    return {
        "audit_root": str(audit_root),
        "call_count": len(calls),
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(cost, 8),
        "mean_latency_ms": (sum(latencies) / len(latencies)) if latencies else 0.0,
        "schema_valid_rate": (
            sum(1 for c in calls if c.schema_valid) / len(calls) if calls else 0.0
        ),
        "limits": rates.model_dump(mode="json"),
    }
