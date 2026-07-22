"""Offline LLM Provider quality eval and limit tests (v1.3.8 / v1.3.9)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scientist_lab.agents.models import PlanningContext
from scientist_lab.agents.provider_bridge import build_llm_provider
from scientist_lab.llm.limits import (
    LimitingProvider,
    ProviderLimitExceeded,
    ProviderLimits,
)
from scientist_lab.llm.models import LLMRequest
from scientist_lab.llm.quality import (
    evaluate_provider_quality,
    summarize_audit_usage,
    write_quality_report,
)
from scientist_lab.domain import NodeStage, NodeStatus, NodeType, ProjectStatus
from scientist_lab.domain.models import ExperimentNode, ResearchProject
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings
import json


EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _context() -> PlanningContext:
    return PlanningContext(
        project_id="project_rgbt_003",
        research_goal="quality eval",
        protocol={
            "protocol_id": "protocol_rgbt_001",
            "allowed_variables": ["input_mode", "fusion_method"],
        },
        protocol_id="protocol_rgbt_001",
        current_best_node_id="rgbt_formal_node_003",
        nodes=[{"node_id": "rgbt_formal_node_003", "status": "succeeded"}],
        remaining_budget={"max_new_nodes": 3, "max_gpu_hours": 10},
        allowed_parameter_changes=["input_mode", "fusion_method"],
    )


def test_provider_quality_mock_fake_replay(tmp_path: Path):
    report = evaluate_provider_quality(
        _context(),
        audit_root=tmp_path / "llm",
        include_real=True,
    )
    by_mode = {m.mode: m for m in report.modes}
    assert by_mode["mock"].status == "ok"
    assert by_mode["fake"].status == "ok"
    assert by_mode["fake"].determinism_match is True
    assert by_mode["fake"].schema_valid is True
    assert by_mode["replay"].status == "ok"
    assert by_mode["replay"].replay_match is True
    assert by_mode["real"].status == "skipped"
    assert report.gates["fake_ok"] is True
    assert report.gates["replay_ok"] is True
    assert report.comparison["fake_replay_identical"] is True

    out = write_quality_report(report, tmp_path / "quality.json")
    assert Path(out.path or "").is_file()


def test_limiting_provider_max_calls(tmp_path: Path):
    provider = build_llm_provider(
        "fake",
        audit_root=tmp_path / "llm",
        project_id="project_rgbt_003",
        limits=ProviderLimits(max_calls=1),
    )
    assert isinstance(provider, LimitingProvider)
    req = LLMRequest(
        purpose="other",
        messages=[{"role": "user", "content": "ping"}],
        metadata={"project_id": "project_rgbt_003"},
    )
    provider.complete(req)
    with pytest.raises(ProviderLimitExceeded) as exc:
        provider.complete(req)
    assert exc.value.limit == "max_calls"


def test_limiting_provider_max_total_tokens(tmp_path: Path):
    provider = build_llm_provider(
        "fake",
        audit_root=tmp_path / "llm",
        project_id="project_rgbt_003",
        limits=ProviderLimits(max_total_tokens=1),
    )
    req = LLMRequest(
        purpose="planner",
        messages=[{"role": "user", "content": '{"project_id":"project_rgbt_003"}'}],
        metadata={
            "project_id": "project_rgbt_003",
            "current_best_node_id": "rgbt_formal_node_003",
        },
    )
    with pytest.raises(ProviderLimitExceeded) as exc:
        provider.complete(req)
    assert exc.value.limit == "max_total_tokens"


def test_summarize_audit_usage(tmp_path: Path):
    evaluate_provider_quality(_context(), audit_root=tmp_path / "llm")
    summary = summarize_audit_usage(tmp_path / "llm")
    assert summary["call_count"] >= 2
    assert summary["total_tokens"] > 0
    assert summary["estimated_cost_usd"] >= 0.0


def _service(tmp_path: Path) -> ExperimentService:
    root = Path(__file__).resolve().parents[2]
    settings = Settings(
        project_root=root,
        db_path=tmp_path / "test.db",
        runtime_dir=tmp_path / "runtime",
        outputs_dir=tmp_path / "outputs",
        experiment_app_dir=root / "experiment_app",
    ).resolve()
    return ExperimentService(settings=settings)


def test_experiment_service_llm_eval(tmp_path: Path):
    service = _service(tmp_path)
    now = "2026-01-01T00:00:00+00:00"
    service.protocols.create_from_path(EXAMPLES / "rgbt_protocol.json")
    service.set_budget("project_rgbt_003", max_new_nodes=3, max_gpu_hours=10)
    service.repo.upsert_project(
        ResearchProject(
            project_id="project_rgbt_003",
            title="RGB-T",
            research_goal="llm eval",
            status=ProjectStatus.ACTIVE,
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
            created_at=now,
            updated_at=now,
        )
    )
    report = service.evaluate_llm_quality(
        "project_rgbt_003",
        protocol_id="protocol_rgbt_001",
        current_best_node_id="rgbt_formal_node_003",
    )
    assert report["gates"]["replay_ok"] is True
    assert Path(report["path"]).is_file()
    usage = service.summarize_llm_usage("project_rgbt_003")
    assert usage["call_count"] >= 1
