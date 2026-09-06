"""v2.1.6 FeedbackUseVerifier + plan-change checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from scientist_lab.domain import NodeStage, NodeStatus, NodeType
from scientist_lab.domain.models import ExperimentNode, new_id, utc_now_iso
from scientist_lab.llm_eval.profiles import LLMModelProfile
from scientist_lab.research_loop.feedback_use_verifier import verify_feedback_use
from scientist_lab.research_loop.models import FeedbackUsageRecord
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"


def _feedback(**overrides):
    base = {
        "source_round": 1,
        "parent_node_id": "node_parent",
        "executed_node_id": "node_exec",
        "metric_deltas": {"accuracy": 0.012},
        "stability_deltas": {},
        "resource_deltas": {},
        "evidence_added": ["evidence_abc123"],
        "claims_changed": ["claim_1"],
        "comparison_ids": [],
        "outcome_label": "improved",
        "executed_parameters": {"hidden_units": 112},
        "previous_hypothesis": "hu=112",
    }
    base.update(overrides)
    return base


def _context(feedback: dict, *, evidence_ids: list[str] | None = None):
    ids = evidence_ids if evidence_ids is not None else list(feedback["evidence_added"])
    return {
        "context_sha256": "sha_test",
        "round_feedback_summary": feedback,
        "recent_execution_summary": {
            "executed_node_id": feedback["executed_node_id"],
            "outcome_label": feedback["outcome_label"],
            "metric_deltas": feedback["metric_deltas"],
            "evidence_added": feedback["evidence_added"],
        },
        "evidence_records": [{"evidence_id": eid} for eid in ids],
        "tested_parameter_fingerprints": [],
    }


def _cand(
    *,
    candidate_id: str,
    parent: str,
    hidden_units: int,
    evidence_ref: str | None = "evidence_abc123",
    rationale: str = "accuracy improved; try nearby capacity",
):
    gaps = [evidence_ref] if evidence_ref else ["Need another capacity point."]
    return {
        "candidate_id": candidate_id,
        "parent_node_id": parent,
        "title": f"hidden_units={hidden_units}",
        "hypothesis": f"try {hidden_units}",
        "parameter_changes": {"hidden_units": hidden_units},
        "evidence_gap_addressed": gaps,
        "rationale": rationale,
        "expected_outcomes": [
            {
                "metric": "accuracy",
                "direction": "increase",
                "rationale": "follow round-1 accuracy signal",
            }
        ],
    }


def test_verifier_passes_when_plan_uses_feedback_and_changes_params():
    fb = _feedback()
    ctx = _context(fb)
    result = verify_feedback_use(
        feedback_summary=fb,
        planning_context=ctx,
        candidates=[_cand(candidate_id="c2", parent="node_exec", hidden_units=96)],
        previous_executed_parameters={"hidden_units": 112},
    )
    assert result.pass_status is True
    assert result.output_changes_experiment_plan is True
    assert result.output_avoids_duplicate_candidate is True
    assert result.context_contains_round_1_evidence is True


def test_verifier_fails_on_duplicate_executed_params():
    fb = _feedback()
    ctx = _context(fb)
    result = verify_feedback_use(
        feedback_summary=fb,
        planning_context=ctx,
        candidates=[_cand(candidate_id="c_dup", parent="node_exec", hidden_units=112)],
        previous_executed_parameters={"hidden_units": 112},
    )
    assert result.pass_status is False
    assert result.output_avoids_duplicate_candidate is False
    assert any("duplicate" in i for i in result.issues)


def test_verifier_fails_when_context_missing_evidence():
    fb = _feedback()
    ctx = _context(fb, evidence_ids=[])
    ctx["round_feedback_summary"] = {
        **fb,
        "evidence_added": [],
    }
    ctx["recent_execution_summary"]["evidence_added"] = []
    result = verify_feedback_use(
        feedback_summary=fb,
        planning_context=ctx,
        candidates=[_cand(candidate_id="c2", parent="node_exec", hidden_units=96)],
        previous_executed_parameters={"hidden_units": 112},
    )
    assert result.context_contains_round_1_evidence is False
    assert result.pass_status is False


def test_verifier_fails_on_ghost_metric():
    fb = _feedback()
    ctx = _context(fb)
    cand = _cand(candidate_id="c_ghost", parent="node_exec", hidden_units=80)
    cand["expected_outcomes"] = [
        {"metric": "totally_fake_metric_xyz", "direction": "increase", "rationale": "x"}
    ]
    cand["rationale"] = "no accuracy mention"
    cand["evidence_gap_addressed"] = []
    result = verify_feedback_use(
        feedback_summary=fb,
        planning_context=ctx,
        candidates=[cand],
        previous_executed_parameters={"hidden_units": 112},
    )
    assert result.pass_status is False
    assert any("unknown metrics" in i for i in result.issues)


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


def test_service_verify_feedback_persists_usage(service: ExperimentService):
    project = service.create_project(
        title="Digits Feedback Use",
        research_question="verify",
        research_goal="v2.1.6",
        task_type="general_ml",
        project_id="project_digits_v216",
        mark_ready=True,
    )
    protocol = service.protocols.create_from_path(
        EXAMPLES / "digits_demo_protocol.json"
    )
    profile = LLMModelProfile(
        profile_id="profile_v216",
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
    executed = ExperimentNode(
        node_id=new_id("node"),
        project_id=project["project_id"],
        parent_node_id=parent.node_id,
        node_type=NodeType.IMPROVEMENT,
        stage=NodeStage.DONE,
        status=NodeStatus.SUCCEEDED,
        hypothesis="hu=112",
        contract_json={"parameters": {"hidden_units": 112}},
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
        "round_2_planning",
        "round_2_reviewing",
    ):
        transition(session, status)
    session.current_round = 2
    session.execution_node_ids = [executed.node_id]
    loop.repo.upsert_session(session)

    fb = _feedback(
        parent_node_id=parent.node_id,
        executed_node_id=executed.node_id,
        executed_parameters={"hidden_units": 112},
    )
    r1 = loop._require_round(session.session_id, 1)
    r1.feedback_summary_json = fb
    r1.execution_node_id = executed.node_id
    r1.evidence_ids = list(fb["evidence_added"])
    r1.status = "feedback_ready"
    loop.repo.upsert_round(r1)

    ctx = _context(fb)
    ctx["recent_execution_summary"]["executed_node_id"] = executed.node_id
    r2 = loop._require_round(session.session_id, 2)
    r2.plan_id = "plan_v216_ok"
    r2.planning_context_json = ctx
    r2.candidate_ids = ["candidate_digits_hu96"]
    r2.status = "reviewing"
    loop.repo.upsert_round(r2)

    def _show(_pid):
        return {
            "plan_id": "plan_v216_ok",
            "project_id": project["project_id"],
            "candidates": [
                _cand(
                    candidate_id="candidate_digits_hu96",
                    parent=executed.node_id,
                    hidden_units=96,
                    evidence_ref="evidence_abc123",
                )
            ],
        }

    service.show_plan = _show  # type: ignore[method-assign]
    result = service.real_loop_verify_feedback(session.session_id, round_number=2)
    assert result["pass_status"] is True
    assert result["feedback_usage"]["verified"] is True
    assert result["checks"]["output_changes_experiment_plan"] is True

    latest = loop.repo.latest_feedback_usage(session.session_id, target_round_number=2)
    assert latest is not None
    assert latest.verified is True
    assert isinstance(latest, FeedbackUsageRecord)


def test_service_verify_blocks_duplicate_plan(service: ExperimentService):
    project = service.create_project(
        title="Digits Dup",
        research_question="dup",
        research_goal="v2.1.6",
        task_type="general_ml",
        project_id="project_digits_v216_dup",
        mark_ready=True,
    )
    protocol = service.protocols.create_from_path(
        EXAMPLES / "digits_demo_protocol.json"
    )
    profile = LLMModelProfile(
        profile_id="profile_v216_dup",
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
    executed = ExperimentNode(
        node_id=new_id("node"),
        project_id=project["project_id"],
        parent_node_id=parent.node_id,
        node_type=NodeType.IMPROVEMENT,
        stage=NodeStage.DONE,
        status=NodeStatus.SUCCEEDED,
        hypothesis="hu=112",
        contract_json={"parameters": {"hidden_units": 112}},
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
        "round_2_planning",
        "round_2_reviewing",
    ):
        transition(session, status)
    session.current_round = 2
    loop.repo.upsert_session(session)

    fb = _feedback(
        parent_node_id=parent.node_id,
        executed_node_id=executed.node_id,
        executed_parameters={"hidden_units": 112},
    )
    r1 = loop._require_round(session.session_id, 1)
    r1.feedback_summary_json = fb
    r1.execution_node_id = executed.node_id
    r1.evidence_ids = list(fb["evidence_added"])
    loop.repo.upsert_round(r1)

    r2 = loop._require_round(session.session_id, 2)
    r2.plan_id = "plan_fake_dup"
    r2.planning_context_json = _context(fb)
    r2.status = "reviewing"
    loop.repo.upsert_round(r2)

    def _show(_pid):
        return {
            "plan_id": "plan_fake_dup",
            "project_id": project["project_id"],
            "candidates": [
                _cand(
                    candidate_id="candidate_dup",
                    parent=executed.node_id,
                    hidden_units=112,
                )
            ],
        }

    service.show_plan = _show  # type: ignore[method-assign]
    result = service.real_loop_verify_feedback(session.session_id, round_number=2)
    assert result["pass_status"] is False
    assert any("duplicate" in i for i in result["issues"])
