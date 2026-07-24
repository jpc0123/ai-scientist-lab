"""v2.1.7 Replay Bundle export + redaction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scientist_lab.domain import NodeStage, NodeStatus, NodeType
from scientist_lab.domain.models import ExperimentNode, new_id, utc_now_iso
from scientist_lab.llm_eval.profiles import LLMModelProfile
from scientist_lab.research_loop.errors import RealLoopValidationError
from scientist_lab.research_loop.replay_bundle import (
    assert_bundle_redacted,
    build_replay_bundle,
    load_replay_bundle,
    scrub_for_replay,
)
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"


@pytest.fixture()
def service(tmp_path: Path) -> ExperimentService:
    return ExperimentService(
        settings=Settings(
            project_root=tmp_path,
            db_path=tmp_path / "lab.db",
            runtime_dir=tmp_path / "runtime",
            outputs_dir=tmp_path / "outputs",
            experiment_app_dir=ROOT / "experiment_app",
        ).resolve()
    )


def test_scrub_for_replay_redacts_secrets_and_paths():
    payload = {
        "api_key": "sk-live-secret-value-here",
        "note": "Authorization: Bearer abcdef123456",
        "host_path": r"C:\Users\secret\project",
        "ok": "hidden_units=96",
    }
    cleaned = scrub_for_replay(payload)
    assert cleaned["api_key"] == "[REDACTED]"
    assert "sk-live" not in json.dumps(cleaned)
    assert "Bearer abcdef" not in json.dumps(cleaned)
    assert "C:\\Users" not in json.dumps(cleaned)
    assert cleaned["ok"] == "hidden_units=96"


def test_build_and_load_replay_bundle(tmp_path: Path):
    bundle_dir = tmp_path / "bundle"
    session = {
        "session_id": "realloop_demo",
        "project_id": "project_demo",
        "status": "completed",
        "real_only": True,
        "fallback_allowed": False,
        "fallback_used": False,
        "required_rounds": 2,
        "provider_profile_id": "profile_x",
        "protocol_id": "protocol_digits_demo_001",
    }
    rounds = [
        {
            "round_id": "rlround_1",
            "round_number": 1,
            "status": "feedback_ready",
            "plan_id": "plan_1",
            "approved_candidate_id": "cand_1",
            "execution_node_id": "node_exec",
            "evidence_ids": ["evidence_abc"],
            "claim_ids": ["claim_1"],
            "feedback_summary_json": {
                "source_round": 1,
                "metric_deltas": {"accuracy": 0.01},
                "evidence_added": ["evidence_abc"],
                "outcome_label": "improved",
                "executed_node_id": "node_exec",
                "parent_node_id": "node_parent",
                "executed_parameters": {"hidden_units": 112},
            },
            "planning_context_json": {"context_sha256": "sha1"},
            "provider_audit_json": {
                "requested_provider": "openai-compatible",
                "fallback_used": False,
            },
        },
        {
            "round_id": "rlround_2",
            "round_number": 2,
            "status": "reviewing",
            "plan_id": "plan_2",
            "planning_context_json": {
                "context_sha256": "sha2",
                "round_feedback_summary": {"source_round": 1},
            },
            "provider_audit_json": {"requested_provider": "openai-compatible"},
        },
    ]
    plans = {
        "plan_1": {
            "plan_id": "plan_1",
            "candidates": [
                {
                    "candidate_id": "cand_1",
                    "parameter_changes": {"hidden_units": 112},
                    "parent_node_id": "node_parent",
                }
            ],
            "critic": {"recommendation": "accept"},
        },
        "plan_2": {
            "plan_id": "plan_2",
            "candidates": [
                {
                    "candidate_id": "cand_2",
                    "parameter_changes": {"hidden_units": 96},
                    "parent_node_id": "node_exec",
                    "rationale": "accuracy improved",
                    "evidence_gap_addressed": ["evidence_abc"],
                }
            ],
        },
    }
    usage = [
        {
            "feedback_usage_id": "fbuse_1",
            "target_round_number": 2,
            "verified": True,
            "verification": {"pass_status": True, "issues": []},
        }
    ]
    llm_calls = [
        {
            "call_id": "call_planner_1",
            "request_fingerprint": "fp123",
            "provider": "openai-compatible",
            "request": {"messages": [{"role": "user", "content": "plan"}]},
            "response": {"content": "{}"},
            "api_key": "sk-should-be-removed",
        }
    ]

    result = build_replay_bundle(
        session=session,
        rounds=rounds,
        output_dir=bundle_dir,
        project_id="project_demo",
        feedback_usage=usage,
        llm_calls=llm_calls,
        plans=plans,
        evidence_records=[{"evidence_id": "evidence_abc", "type": "paired"}],
        claim_matrix={"claims": [{"claim_id": "claim_1"}]},
    )
    assert result["ok"] is True
    assert result["secrets_redacted"] is True
    assert (bundle_dir / "manifest.json").is_file()
    assert (bundle_dir / "session.json").is_file()
    assert (bundle_dir / "round_1" / "feedback_summary.json").is_file()
    assert (bundle_dir / "round_2" / "feedback_verification.json").is_file()
    assert (bundle_dir / "llm_replays" / "call_planner_1.json").is_file()
    assert (bundle_dir / "real_llm_closed_loop_report.json").is_file()

    loaded = load_replay_bundle(bundle_dir)
    assert loaded["manifest"]["session_id"] == "realloop_demo"
    assert "round_1" in loaded["rounds"]
    assert loaded["llm_replays"]
    assert "sk-should" not in json.dumps(loaded)

    assert_bundle_redacted(bundle_dir)


def test_service_export_replay(service: ExperimentService, tmp_path: Path):
    project = service.create_project(
        title="Digits Replay",
        research_question="export",
        research_goal="v2.1.7",
        task_type="general_ml",
        project_id="project_digits_v217",
        mark_ready=True,
    )
    protocol = service.protocols.create_from_path(
        EXAMPLES / "digits_demo_protocol.json"
    )
    profile = LLMModelProfile(
        profile_id="profile_v217",
        provider="openai-compatible",
        model="gpt-test",
        enabled=True,
    )
    service.llm_evals.upsert_profile(profile)
    now = utc_now_iso()
    parent = ExperimentNode(
        node_id=new_id("node"),
        project_id=project["project_id"],
        node_type=NodeType.BASELINE,
        stage=NodeStage.DONE,
        status=NodeStatus.SUCCEEDED,
        hypothesis="hu=64",
        contract_json={"parameters": {"hidden_units": 64}},
        created_at=now,
        updated_at=now,
    )
    service.repo.upsert_node(parent)
    created = service.real_loop_create(
        project["project_id"],
        profile_id=profile.profile_id,
        protocol_id=protocol.protocol_id,
        baseline_node_ids=[parent.node_id],
    )
    loop = service._real_loop_service()
    session = loop._require(created["session_id"])
    from scientist_lab.research_loop.state_machine import transition

    for status in (
        "baseline_ready",
        "round_1_planning",
        "round_1_reviewing",
        "round_1_waiting_approval",
        "round_1_executing",
        "round_1_feedback_ready",
    ):
        transition(session, status)
    loop.repo.upsert_session(session)

    r1 = loop._require_round(session.session_id, 1)
    r1.feedback_summary_json = {
        "source_round": 1,
        "parent_node_id": parent.node_id,
        "executed_node_id": parent.node_id,
        "metric_deltas": {"accuracy": 0.01},
        "evidence_added": ["evidence_demo"],
        "outcome_label": "improved",
        "executed_parameters": {"hidden_units": 112},
    }
    r1.evidence_ids = ["evidence_demo"]
    r1.planning_context_json = {"context_sha256": "x"}
    r1.status = "feedback_ready"
    loop.repo.upsert_round(r1)

    out = tmp_path / "export_bundle"
    exported = service.real_loop_export(session.session_id, output_dir=str(out))
    assert exported["ok"] is True
    assert Path(exported["bundle_dir"]).is_dir()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["secrets_redacted"] is True
    assert manifest["session_id"] == session.session_id


def test_export_rejects_too_early(service: ExperimentService):
    project = service.create_project(
        title="Digits Early",
        research_question="early",
        research_goal="v2.1.7",
        task_type="general_ml",
        project_id="project_digits_v217_early",
        mark_ready=True,
    )
    protocol = service.protocols.create_from_path(
        EXAMPLES / "digits_demo_protocol.json"
    )
    profile = LLMModelProfile(
        profile_id="profile_v217_early",
        provider="openai-compatible",
        model="gpt-test",
        enabled=True,
    )
    service.llm_evals.upsert_profile(profile)
    now = utc_now_iso()
    parent = ExperimentNode(
        node_id=new_id("node"),
        project_id=project["project_id"],
        node_type=NodeType.BASELINE,
        stage=NodeStage.DONE,
        status=NodeStatus.SUCCEEDED,
        hypothesis="hu=64",
        contract_json={"parameters": {"hidden_units": 64}},
        created_at=now,
        updated_at=now,
    )
    service.repo.upsert_node(parent)
    created = service.real_loop_create(
        project["project_id"],
        profile_id=profile.profile_id,
        protocol_id=protocol.protocol_id,
        baseline_node_ids=[parent.node_id],
    )
    with pytest.raises(RealLoopValidationError):
        service.real_loop_export(created["session_id"])
