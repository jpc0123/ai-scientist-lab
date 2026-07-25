"""v2.2.7 PatchEvidence feedback into Evidence / Planner / Tree (offline)."""

from __future__ import annotations

from pathlib import Path

from scientist_lab.agents.context_builder import build_planning_context
from scientist_lab.patching.feedback import (
    build_patch_feedback_package,
    derive_patch_verdict,
    load_planner_patch_feedback,
    patch_evidence_to_claim_draft,
    patch_evidence_to_lab_record,
)
from scientist_lab.patching.service import build_mock_unified_diff
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings


ROOT = Path(__file__).resolve().parents[2]


def _service(tmp_path: Path) -> ExperimentService:
    return ExperimentService(
        settings=Settings(
            project_root=ROOT,
            db_path=tmp_path / "test.db",
            runtime_dir=tmp_path / "runtime",
            outputs_dir=tmp_path / "outputs",
            experiment_app_dir=ROOT / "experiment_app",
        ).resolve()
    )


def _applied_patch(service: ExperimentService, *, suffix: str) -> str:
    proposed = service.patches.propose_mock(
        "project_rgbt_003",
        unified_diff=build_mock_unified_diff(
            relative_path=(
                f"experiment_apps/rgbt_detection_real/adapters/fb_{suffix}.md"
            )
        ),
    )
    pid = proposed["patch_id"]
    service.patches.approve(pid)
    service.patches.apply_sandbox(pid)
    service.patches.test_sandbox(pid, profile="mock_experiment")
    return pid


def test_record_evidence_builds_feedback_package(tmp_path: Path):
    service = _service(tmp_path)
    pid = _applied_patch(service, suffix="pkg")
    recorded = service.patches.record_evidence(pid, require_tests=True)
    assert recorded["status"] == "evidence_recorded"
    evidence = recorded["patch_evidence"]
    assert evidence["verdict"] == "effective"
    assert evidence["sandbox_tests_ok"] is True
    feedback = recorded["patch_feedback"]
    assert feedback["schema_version"] == "v2.2.7"
    assert feedback["verdict"] == "effective"
    assert Path(feedback["artifact_path"]).is_file()
    assert feedback["planner_fragment"]["should_retain"] is True
    assert feedback["claim_draft"]["claim_type"] == "code_patch_effectiveness"


def test_lab_feedback_persists_evidence_record(tmp_path: Path):
    service = _service(tmp_path)
    pid = _applied_patch(service, suffix="lab")
    view = service.record_patch_evidence_feedback(pid, require_tests=True)
    lab = view["lab_feedback"]
    assert lab["persisted_evidence_id"]
    assert lab["main_workspace_modified"] is False
    stored = service.evidence.require(lab["persisted_evidence_id"])
    assert stored.evidence_type == "failure_analysis"
    assert stored.metric_summary.get("kind") == "sandbox_patch"
    assert stored.metric_summary.get("verdict") == "effective"
    assert Path(lab["claim_draft_path"]).is_file()
    planner = load_planner_patch_feedback(
        tmp_path / "outputs", "project_rgbt_003"
    )
    assert any(item.get("patch_id") == pid for item in planner)


def test_planner_context_accepts_patch_feedback(tmp_path: Path):
    service = _service(tmp_path)
    pid = _applied_patch(service, suffix="ctx")
    service.record_patch_evidence_feedback(pid, require_tests=True)
    records = load_planner_patch_feedback(
        tmp_path / "outputs", "project_rgbt_003"
    )
    context = build_planning_context(
        project_id="project_rgbt_003",
        research_goal="Improve under fixed protocol",
        protocol={"protocol_id": "p1", "allowed_variables": ["learning_rate"]},
        nodes=[],
        patch_feedback_records=records,
    )
    assert context.patch_feedback_records
    assert context.patch_feedback_records[-1]["patch_id"] == pid


def test_verdict_helpers():
    from scientist_lab.patching.evidence import PatchEvidence

    ok = PatchEvidence(
        evidence_id="patch_ev_1",
        patch_id="p1",
        project_id="proj",
        sandbox_tests_ok=True,
        evidence_strength="moderate",
    )
    assert derive_patch_verdict(ok) == "effective"
    bad = ok.model_copy(update={"sandbox_tests_ok": False, "evidence_strength": "weak"})
    assert derive_patch_verdict(bad) == "ineffective"
    claim = patch_evidence_to_claim_draft(ok)
    assert claim.support_status == "partially_supported"
    lab = patch_evidence_to_lab_record(ok)
    assert lab.scientific_evidence_level == "weak"
