"""v1.4 acceptance: OpenAI-compatible provider (offline; MockTransport only)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pydantic import SecretStr, ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scientist_lab.agents.provider_bridge import build_planner_critic
from scientist_lab.domain import JobStatus, NodeStage, NodeStatus, NodeType, ProjectStatus
from scientist_lab.domain.models import ExecutionAttempt, ExperimentNode, ResearchProject
from scientist_lab.llm.budget import LLMBudget
from scientist_lab.llm.config import LLMConfig, load_llm_config, redact_secrets
from scientist_lab.llm.errors import (
    InvalidLLMConfigError,
    LLMBudgetExceededError,
    MissingAPIKeyError,
    MissingBaseURLError,
    MissingModelError,
    RealProviderNotEnabledError,
)
from scientist_lab.llm.factory import create_llm_provider
from scientist_lab.llm.http_transport import HttpResponse, MockTransport
from scientist_lab.llm.models import LLMRequest
from scientist_lab.llm.openai_compatible_provider import OpenAICompatibleProvider
from scientist_lab.llm.openai_config import OpenAICompatibleConfig
from scientist_lab.llm.retry_policy import RetryPolicy
from scientist_lab.llm.schema_parser import PLANNER_OUTPUT_SCHEMA
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings


def _check(name: str, ok: bool, detail: str = "") -> dict:
    return {"name": name, "ok": bool(ok), "detail": detail}


def _cfg(**kwargs) -> OpenAICompatibleConfig:
    base = dict(
        base_url="https://api.example.com/v1",
        api_key=SecretStr("sk-accept-v14-secret-key-value"),
        model="gpt-accept-v14",
        allow_network=False,
        max_retries=1,
        max_concurrency=1,
    )
    base.update(kwargs)
    return OpenAICompatibleConfig(**base)


def _planner_content() -> str:
    return json.dumps(
        {
            "project_id": "project_rgbt_003",
            "reasoning_summary": "accept_v14 mock transport planner",
            "candidates": [
                {
                    "candidate_id": "candidate_accept_ablation_001",
                    "parent_node_id": "rgbt_formal_node_003",
                    "title": "Accept ablation: RGB-only control",
                    "hypothesis": (
                        "Removing fusion under a matched protocol should reduce "
                        "AP_small if fusion contributes."
                    ),
                    "experiment_type": "ablation",
                    "parameter_changes": {
                        "input_mode": "rgb",
                        "fusion_method": "none",
                    },
                    "expected_outcomes": [
                        {
                            "metric": "AP_small",
                            "direction": "decrease",
                            "rationale": "Ablating fusion should hurt small-object AP.",
                        }
                    ],
                    "evidence_gap_addressed": [
                        "Missing controlled ablation for fusion contribution."
                    ],
                    "priority": 0.81,
                    "rationale": "Deterministic accept_v14 candidate.",
                    "claim_limitations": ["Offline acceptance fixture only."],
                }
            ],
            "stop_recommended": False,
            "stop_reason": None,
        },
        ensure_ascii=False,
    )


def _critic_content() -> str:
    return json.dumps(
        {
            "candidate_id": "candidate_accept_ablation_001",
            "scientific_validity": "valid",
            "novelty_status": "new",
            "expected_information_gain": 0.7,
            "cost_effectiveness": 0.65,
            "risk_level": "low",
            "strengths": ["Single-variable ablation."],
            "weaknesses": ["Acceptance fixture review."],
            "required_revisions": [],
            "recommendation": "accept",
        },
        ensure_ascii=False,
    )


def _chat_body(content: str, *, usage: dict | None = None) -> str:
    return json.dumps(
        {
            "id": "chatcmpl-accept-v14",
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": usage
            if usage is not None
            else {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        }
    )


def _planner_req() -> LLMRequest:
    return LLMRequest(
        purpose="planner",
        messages=[{"role": "user", "content": "plan"}],
        response_schema=PLANNER_OUTPUT_SCHEMA,
        temperature=0.0,
        max_tokens=256,
    )


def _seed_attempts(service: ExperimentService, *, node_id: str, map_mean: float) -> None:
    contract = json.loads(
        (ROOT / "examples" / "rgbt_formal_fusion_contract.json").read_text(
            encoding="utf-8"
        )
    )
    contract = dict(contract)
    contract["task_config"] = dict(contract.get("task_config") or {})
    contract["task_config"]["input_mode"] = "fusion"
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
                created_at=f"2026-07-22T00:00:{i:02d}+00:00",
                updated_at=f"2026-07-22T00:01:{i:02d}+00:00",
            )
        )


def seed(service: ExperimentService) -> None:
    now = "2026-07-22T00:00:00+00:00"
    try:
        service.create_protocol(ROOT / "examples" / "rgbt_protocol.json")
    except Exception:
        pass
    service.set_budget("project_rgbt_003", max_new_nodes=5, max_gpu_hours=10)
    service.repo.upsert_project(
        ResearchProject(
            project_id="project_rgbt_003",
            title="RGB-T v1.4 acceptance",
            research_goal="OpenAI-compatible provider offline acceptance",
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
    accept_root = ROOT / "outputs" / "_accept_v14"
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

    # 1. Default provider is mock
    cfg_default = load_llm_config(environ={})
    results.append(
        _check(
            "default-provider-mock",
            cfg_default.provider == "mock" and not cfg_default.is_real,
            f"provider={cfg_default.provider}",
        )
    )

    # 2. Mock / Fake runs without API key
    fake = create_llm_provider("mock")
    fake_resp = fake.complete(
        LLMRequest(
            purpose="other",
            messages=[{"role": "user", "content": "no-key-needed"}],
        )
    )
    results.append(
        _check(
            "mock-without-api-key",
            fake_resp.provider == "fake" and bool(fake_resp.content),
            f"provider={fake_resp.provider}",
        )
    )

    # 3–5. Missing key / URL / model errors are explicit
    missing_key = False
    try:
        LLMConfig(provider="openai-compatible").require_real_ready()
    except MissingAPIKeyError:
        missing_key = True
    results.append(_check("missing-api-key-error", missing_key))

    missing_url = False
    try:
        OpenAICompatibleConfig(
            base_url="",
            api_key=SecretStr("sk-x"),
            model="m",
        )
    except (MissingBaseURLError, InvalidLLMConfigError, ValidationError, ValueError):
        missing_url = True
    results.append(_check("missing-base-url-error", missing_url))

    missing_model = False
    try:
        OpenAICompatibleConfig(
            base_url="https://api.example.com/v1",
            api_key=SecretStr("sk-x"),
            model="",
        )
    except (MissingModelError, InvalidLLMConfigError, ValidationError, ValueError):
        missing_model = True
    results.append(_check("missing-model-error", missing_model))

    # 6. Real blocked without allow_network
    blocked = False
    try:
        create_llm_provider(
            "openai-compatible",
            allow_network=False,
            openai_config=_cfg(allow_network=False),
            environ={"LLM_ALLOW_NETWORK": "false"},
        )
    except RealProviderNotEnabledError:
        blocked = True
    results.append(_check("real-blocked-without-allow-network", blocked))

    # 7–8. MockTransport Planner / Critic via service
    transport = MockTransport(
        responses=[
            HttpResponse(status_code=200, body=_chat_body(_planner_content())),
            HttpResponse(status_code=200, body=_chat_body(_critic_content())),
        ]
    )
    planned = service.plan_next(
        "project_rgbt_003",
        protocol_id="protocol_rgbt_001",
        current_best_node_id="rgbt_formal_node_003",
        provider="real",
        allow_network=False,
        transport=transport,
        openai_config=_cfg(),
    )
    results.append(
        _check(
            "plan-next-mock-transport",
            planned.get("status") not in {"real_provider_failed", "planner_failed"}
            and planned.get("requested_provider") == "real"
            and planned.get("fallback_used") is False
            and "openai-compatible" in str(planned.get("actual_provider") or ""),
            f"status={planned.get('status')} actual={planned.get('actual_provider')}",
        )
    )
    verified_ok = any(
        c.get("status") == "verified" for c in (planned.get("candidates") or [])
    )
    results.append(
        _check(
            "real-output-candidate-verifier",
            verified_ok,
            f"candidates={len(planned.get('candidates') or [])}",
        )
    )

    reviewed = service.review_plan(
        planned["plan_id"],
        provider="real",
        allow_network=False,
        transport=transport,
        openai_config=_cfg(),
    )
    results.append(
        _check(
            "review-plan-mock-transport",
            reviewed.get("status") not in {"real_provider_failed"}
            and bool(reviewed.get("reviews")),
            f"status={reviewed.get('status')}",
        )
    )

    # Fail-closed: no silent mock fallback
    failed = service.plan_next(
        "project_rgbt_003",
        protocol_id="protocol_rgbt_001",
        provider="real",
        allow_network=False,
        openai_config=_cfg(),
    )
    results.append(
        _check(
            "real-fail-closed-no-silent-mock",
            failed.get("status") == "real_provider_failed"
            and failed.get("fallback_used") is False
            and failed.get("actual_provider") is None,
            f"status={failed.get('status')} err={failed.get('error_type')}",
        )
    )

    # 9. Schema repair once
    bad = '{"project_id":"project_rgbt_003"}'
    good = json.dumps(
        {
            "project_id": "project_rgbt_003",
            "reasoning_summary": "repaired",
            "candidates": [],
            "stop_recommended": True,
            "stop_reason": "fixture",
        }
    )
    repair_transport = MockTransport(
        responses=[
            HttpResponse(status_code=200, body=_chat_body(bad)),
            HttpResponse(status_code=200, body=_chat_body(good)),
        ]
    )
    repair_provider = OpenAICompatibleProvider(
        _cfg(),
        transport=repair_transport,
        enable_schema_repair=True,
        sleep=lambda _s: None,
    )
    repaired = repair_provider.complete(_planner_req())
    results.append(
        _check(
            "schema-repair-once",
            repaired.schema_valid is True
            and repaired.metadata.get("schema_repaired") is True
            and len(repair_transport.calls) == 2,
            f"calls={len(repair_transport.calls)} repaired={repaired.metadata.get('schema_repaired')}",
        )
    )

    # 10. 429 retry then success
    retry_transport = MockTransport(
        responses=[
            HttpResponse(status_code=429, body='{"error":"rate"}'),
            HttpResponse(status_code=200, body=_chat_body(good)),
        ]
    )
    retry_provider = OpenAICompatibleProvider(
        _cfg(),
        transport=retry_transport,
        retry_policy=RetryPolicy(max_retries=1, jitter=False),
        sleep=lambda _s: None,
        enable_schema_repair=False,
    )
    after_retry = retry_provider.complete(_planner_req())
    results.append(
        _check(
            "retry-429-then-success",
            after_retry.schema_valid is True and len(retry_transport.calls) == 2,
            f"calls={len(retry_transport.calls)}",
        )
    )

    # 11. Budget exceeded blocks before network
    budget = LLMBudget(project_id="accept_v14", max_requests=0)
    budget_transport = MockTransport(
        default_response=HttpResponse(status_code=200, body=_chat_body(good))
    )
    budget_blocked = False
    try:
        OpenAICompatibleProvider(
            _cfg(),
            transport=budget_transport,
            budget=budget,
            enable_schema_repair=False,
        ).complete(_planner_req())
    except LLMBudgetExceededError:
        budget_blocked = True
    results.append(
        _check(
            "budget-exceeded-blocks",
            budget_blocked and budget_transport.calls == [],
            f"calls={len(budget_transport.calls)}",
        )
    )

    # 12. Audit / config / redact have no raw secret
    secret = "sk-accept-v14-secret-key-value"
    safe = _cfg().safe_dict()
    redacted = redact_secrets(f"Authorization: Bearer {secret}")
    results.append(
        _check(
            "no-secret-in-safe-views",
            secret not in json.dumps(safe)
            and secret not in repr(_cfg())
            and secret not in redacted
            and "[REDACTED]" in redacted,
            f"safe_api_key={safe.get('api_key')} redacted={redacted[:40]}",
        )
    )

    # 13–14. Fake then Replay
    audit_root = settings.outputs_dir / "project_rgbt_003" / "llm_accept_replay"
    planner_fake, _ = build_planner_critic(
        "fake", audit_root=audit_root, project_id="project_rgbt_003"
    )
    from scientist_lab.agents.models import PlanningContext

    ctx = PlanningContext(
        project_id="project_rgbt_003",
        research_goal="accept replay",
        protocol={"protocol_id": "protocol_rgbt_001", "allowed_variables": ["input_mode"]},
        protocol_id="protocol_rgbt_001",
        current_best_node_id="rgbt_formal_node_003",
        nodes=[{"node_id": "rgbt_formal_node_003", "status": "succeeded"}],
        remaining_budget={"max_new_nodes": 3, "max_gpu_hours": 10},
        allowed_parameter_changes=["input_mode", "fusion_method"],
    )
    fake_out = planner_fake.plan(ctx)
    planner_replay, _ = build_planner_critic(
        "replay", audit_root=audit_root, project_id="project_rgbt_003"
    )
    replay_out = planner_replay.plan(ctx)
    results.append(
        _check(
            "fake-then-replay",
            fake_out.model_dump(mode="json") == replay_out.model_dump(mode="json"),
            "replay matched fake planner output",
        )
    )

    # 15. tree-plan-next default mock
    created = service.tree_create(
        "project_rgbt_003",
        root_node_id="rgbt_formal_node_003",
        protocol_id="protocol_rgbt_001",
        max_depth=3,
        max_nodes=8,
        max_children=3,
    )
    tree_mock = service.tree_plan_next(created["tree_id"], provider="mock")
    results.append(
        _check(
            "tree-plan-next-mock-default",
            tree_mock.get("provider") == "mock"
            and tree_mock.get("status")
            in {"planned", "no_valid_candidates", "stopped", "planner_failed"},
            f"status={tree_mock.get('status')} provider={tree_mock.get('provider')}",
        )
    )

    # Real tree path fail-closed (no network)
    tree_real_fail = service.tree_plan_next(
        created["tree_id"],
        provider="real",
        allow_network=False,
        openai_config=_cfg(),
    )
    results.append(
        _check(
            "tree-plan-next-real-fail-closed",
            tree_real_fail.get("status") == "real_provider_failed"
            and tree_real_fail.get("fallback_used") is False,
            f"status={tree_real_fail.get('status')}",
        )
    )

    report = {
        "version": "v1.4",
        "release": "v1.4.0",
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
    out_path = out_dir / "v14_acceptance_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    docs_dir = ROOT / "docs" / "acceptance" / "v1.4"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "v14_acceptance_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (docs_dir / "README.md").write_text(
        "\n".join(
            [
                "# v1.4 Acceptance",
                "",
                "- First OpenAI-compatible cloud provider (offline-gated)",
                "- Default provider remains **mock**; CI needs no API key",
                "- Triple gate for live HTTP: `--provider real` + `--allow-network` + `LLM_ALLOW_NETWORK`",
                "- Failures surface as `real_provider_failed` (no silent mock fallback)",
                "- MockTransport covers Planner / Critic / retry / budget / schema repair",
                "",
                f"Checks: {report['passed']}/{report['total']}",
                "",
                "Run: `python scripts/accept_v14.py`",
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
