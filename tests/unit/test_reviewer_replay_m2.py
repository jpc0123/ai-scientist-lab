"""REPLAY recovered evidence + rules-first Reviewer (no GPU, no Planner)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from scientist_lab.adapters.dfine.run_loop import run_gated_dfine
from scientist_lab.core.decision_rubric import evaluate_rubric
from scientist_lab.core.evidence_validator import EvidenceValidator, EvidenceVerdict
from scientist_lab.core.git_manager import GitManager
from scientist_lab.core.memory_writer import MemoryWriter
from scientist_lab.core.reviewer import ReviewRefused, Reviewer
from scientist_lab.core.schema_registry import SCHEMA_DIR, load_json, validate_named
from scientist_lab.core.state_machine import EvidenceStatus, ReviewDecisionValue

EXAMPLES = SCHEMA_DIR / "examples"


def _plan(**overrides):
    plan = {
        "schema_version": "1.0.0",
        "plan_id": "plan_round1_neck_hr",
        "project_id": "project_rgbt_cuda_001",
        "protocol_id": "research_protocol_rgbt_dfine_v1",
        "protocol_version": 1,
        "parent_run_id": "EXP-007",
        "round_index": 1,
        "observation": "localization still weak",
        "hypothesis": "high-res neck path helps APS",
        "modification_scope": ["neck"],
        "proposed_changes": [{"target": "neck", "summary": "Adjust high-res path"}],
        "controlled_variables": ["evaluator"],
        "expected_effect": {"primary_metric": "APS", "direction": "increase"},
        "evaluation": {"method": "fast_eval", "seeds": [42]},
        "budget_class": "probe",
        "risk_level": "auto",
        "memory_refs": {"lesson_ids": ["LESSON-017"], "strategy_ids": ["STRATEGY-009"]},
        "evidence_runs": ["EXP-007"],
    }
    plan.update(overrides)
    return plan


def _git_init(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.email", "lab@test.local"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Scientist Lab Test"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    (root / "model.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "model.py"], cwd=root, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "commit", "-m", "seed"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return root

EXAMPLES = SCHEMA_DIR / "examples"


def _install_recovered(dest: Path, metrics_name: str) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(EXAMPLES / metrics_name, dest / "metrics.json")
    shutil.copyfile(
        EXAMPLES / "recovered_k2c44_checkpoint_selection.json",
        dest / "checkpoint_selection.json",
    )


def _result_from_metrics(metrics: dict, run_id: str = "run_plan_round1_neck_hr") -> dict:
    return {
        "schema_version": "1.0.0",
        "run_id": run_id,
        "metrics": metrics,
        "artifacts": {
            "paths": ["metrics.json", "checkpoint_selection.json"],
            "missing_expected": [],
        },
        "execution": {"status": "success", "exit_code": 0},
        "raw_metric_refs": ["metrics.json"],
    }


def test_recovered_fixture_has_independent_aps_and_map() -> None:
    last = load_json(EXAMPLES / "recovered_k2c44_last_metrics.json")
    best = load_json(EXAMPLES / "recovered_k2c44_best_metrics.json")
    assert last["APS"] == pytest.approx(7.4)
    assert last["mAP50_95"] == pytest.approx(7.3)
    assert last["APS"] != last["mAP50_95"]
    assert best["APS"] == pytest.approx(15.5)
    assert best["mAP50_95"] == pytest.approx(14.3)
    assert best["APS"] != best["mAP50_95"]


def test_reviewer_refuses_incomplete_and_not_applicable() -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")
    plan = _plan()
    contract = {"run_id": "run_plan_round1_neck_hr", "hypothesis": plan["hypothesis"]}
    result = _result_from_metrics({"APS": 7.4, "mAP50_95": 7.3})
    rubric = evaluate_rubric(
        protocol, current_metrics=result["metrics"], baseline_metrics={"APS": 15.5}
    )
    reviewer = Reviewer()
    for status in (
        EvidenceStatus.INCOMPLETE.value,
        EvidenceStatus.NOT_APPLICABLE.value,
        EvidenceStatus.INVALID.value,
    ):
        with pytest.raises(ReviewRefused):
            reviewer.review(
                result=result,
                evidence=EvidenceVerdict(status, ("blocked",), False),
                rubric=rubric,
                contract=contract,
                protocol=protocol,
                plan=plan,
            )


def test_reviewer_discard_negative_evidence_from_rubric() -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")
    plan = _plan()
    last = load_json(EXAMPLES / "recovered_k2c44_last_metrics.json")
    best = load_json(EXAMPLES / "recovered_k2c44_best_metrics.json")
    result = _result_from_metrics(
        {k: last[k] for k in ("APS", "mAP50_95", "mAP50", "params_m", "flops_g", "gpu_memory_gb")}
    )
    evidence = EvidenceValidator().validate(
        result,
        protocol=protocol,
        expected_artifacts=["metrics.json", "checkpoint_selection.json"],
        fingerprint_comparable=True,
    )
    assert evidence.evidence_status == "VALID"
    rubric = evaluate_rubric(
        protocol,
        current_metrics=result["metrics"],
        baseline_metrics={"APS": best["APS"], "mAP50_95": best["mAP50_95"]},
    )
    assert rubric.suggest_discard_threshold is True
    packet = Reviewer().review(
        result=result,
        evidence=evidence,
        rubric=rubric,
        contract={"run_id": result["run_id"], "allowed_changes": ["neck"]},
        protocol=protocol,
        plan=plan,
    )
    assert packet.review_decision == ReviewDecisionValue.DISCARD.value
    assert packet.research_lessons[0]["type"] == "negative_evidence"
    assert packet.research_lessons[0]["created_from"] == [result["run_id"]]
    assert packet.strategies[0]["action"] == "deprioritize"
    validate_named("review_decision", packet.document)


def test_reviewer_discard_when_aps_zero_vs_control_baseline() -> None:
    """APS=0 vs control APS=0.6 → delta=-0.6 < discard_if(-0.5). No GPU."""
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")
    current = {"APS": 0.0, "mAP50_95": 0.0, "mAP50": 0.0}
    rubric = evaluate_rubric(
        protocol,
        current_metrics=current,
        baseline_metrics={"APS": 0.6},
    )
    assert rubric.primary_delta == pytest.approx(-0.6)
    assert rubric.suggest_discard_threshold is True
    assert rubric.keep_threshold_ok is False
    packet = Reviewer().review(
        result=_result_from_metrics(current),
        evidence=EvidenceVerdict("VALID", (), True),
        rubric=rubric,
        contract={"run_id": "run_plan_round1_neck_hr", "allowed_changes": ["neck"]},
        protocol=protocol,
        plan=_plan(),
    )
    assert packet.review_decision == ReviewDecisionValue.DISCARD.value
    assert packet.research_lessons[0]["type"] == "negative_evidence"
    assert packet.strategies[0]["action"] == "deprioritize"


def test_reviewer_keep_when_keep_threshold_ok() -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")
    plan = _plan()
    best = load_json(EXAMPLES / "recovered_k2c44_best_metrics.json")
    metrics = {
        k: best[k] for k in ("APS", "mAP50_95", "mAP50", "params_m", "flops_g", "gpu_memory_gb")
    }
    result = _result_from_metrics(metrics)
    evidence = EvidenceVerdict("VALID", (), True)
    rubric = evaluate_rubric(protocol, current_metrics=metrics, baseline_metrics={"APS": 15.5})
    assert rubric.keep_threshold_ok is True
    assert rubric.suggest_discard_threshold is False
    assert rubric.suggest_validate is False
    packet = Reviewer().review(
        result=result,
        evidence=evidence,
        rubric=rubric,
        contract={"run_id": result["run_id"], "allowed_changes": ["neck"]},
        protocol=protocol,
        plan=plan,
    )
    assert packet.review_decision == ReviewDecisionValue.KEEP.value
    assert packet.document["hypothesis_status"] == "INCONCLUSIVE"


def test_recovered_last_vs_best_is_valid_discard_and_writes_memory(tmp_path: Path) -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")
    best = load_json(EXAMPLES / "recovered_k2c44_best_metrics.json")
    _install_recovered(tmp_path, "recovered_k2c44_last_metrics.json")
    report = run_gated_dfine(
        protocol=protocol,
        plan=_plan(),
        output_dir=tmp_path,
        execute=False,
        baseline_metrics={"APS": best["APS"], "mAP50_95": best["mAP50_95"]},
    )
    assert report["gate"]["status"] == "APPROVED"
    assert report["evidence"]["evidence_status"] == "VALID"
    assert report["recovered"] is True
    assert report["review_decision"] == "DISCARD"
    assert report["rubric"]["suggest_discard_threshold"] is True
    assert report["result"]["metrics"]["APS"] == pytest.approx(7.4)
    assert report["result"]["metrics"]["mAP50_95"] == pytest.approx(7.3)
    assert report["memory"]["lessons_written"]
    assert report["memory"]["strategies_written"]
    assert report["memory"]["invented_from_metrics"] is False
    writer = MemoryWriter(tmp_path / "memory")
    lessons = writer.load_lessons()
    lesson = next(iter(lessons.values()))
    assert lesson["type"] == "negative_evidence"
    assert lesson["created_from"] == [report["contract_run_id"]]
    assert lesson["evidence"][0]["run_id"] == report["contract_run_id"]
    strategy = next(iter(writer.load_strategies().values()))
    assert strategy["action"] == "deprioritize"
    kinds = {(e["from"], e["to"], e["type"]) for e in writer.load_trace()}
    run_id = report["contract_run_id"]
    assert (run_id, lesson["lesson_id"], "derived_from") in kinds
    assert (lesson["lesson_id"], strategy["strategy_id"], "used_by") in kinds
    events = (tmp_path / "research_events.jsonl").read_text(encoding="utf-8")
    assert '"event_type": "review_decision"' in events
    assert '"actor_role": "reviewer"' in events
    assert '"phase": "review"' in events
    assert "strategy_revision" in events


def test_recovered_best_vs_self_is_valid_keep(tmp_path: Path) -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")
    best = load_json(EXAMPLES / "recovered_k2c44_best_metrics.json")
    _install_recovered(tmp_path, "recovered_k2c44_best_metrics.json")
    report = run_gated_dfine(
        protocol=protocol,
        plan=_plan(),
        output_dir=tmp_path,
        execute=False,
        baseline_metrics={"APS": best["APS"]},
    )
    assert report["evidence"]["evidence_status"] == "VALID"
    assert report["review_decision"] == "KEEP"
    assert report["memory"]["lessons_written"]
    assert report["review"]["review_decision"] == "KEEP"


def test_incomplete_does_not_review_or_write_lessons(tmp_path: Path) -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")

    def runner(_contract, output: Path):
        output.mkdir(parents=True, exist_ok=True)
        (output / "metrics.json").write_text(
            json.dumps({"APS": 7.4, "mAP50_95": 7.3}), encoding="utf-8"
        )
        return {"status": "completed"}

    report = run_gated_dfine(
        protocol=protocol,
        plan=_plan(),
        output_dir=tmp_path,
        execute=True,
        live_runner=runner,
        review={
            "research_lessons": [
                {
                    "lesson_id": "LESSON-INJECT",
                    "type": "positive_evidence",
                    "statement": "should not persist",
                    "status": "active",
                    "evidence": [{"run_id": "run_x"}],
                    "created_from": ["run_x"],
                }
            ]
        },
    )
    assert report["evidence"]["evidence_status"] == "INCOMPLETE"
    assert report["review_decision"] == "PENDING"
    assert report["memory"]["lessons_written"] == []
    assert not (tmp_path / "memory" / "research_memory.json").is_file()
    dumped = json.dumps(report)
    assert '"KEEP"' not in dumped
    assert '"DISCARD"' not in dumped
    events = (tmp_path / "research_events.jsonl").read_text(encoding="utf-8")
    assert "review_decision" not in events or '"event_type": "review_decision"' not in events


def test_not_applicable_dry_run_stays_pending_without_lessons(tmp_path: Path) -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")
    report = run_gated_dfine(
        protocol=protocol,
        plan=_plan(),
        output_dir=tmp_path,
        execute=False,
    )
    assert report["evidence"]["evidence_status"] == "NOT_APPLICABLE"
    assert report["review_decision"] == "PENDING"
    assert report["memory"]["lessons_written"] == []
    assert "review" not in report


def test_git_apply_keep_and_discard_only_after_valid_review(tmp_path: Path) -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")
    best = load_json(EXAMPLES / "recovered_k2c44_best_metrics.json")
    repo = _git_init(tmp_path / "repo")
    mgr = GitManager(repo)
    seed_best = mgr.load().best_sha
    (repo / "model.py").write_text("x = 9\n", encoding="utf-8")
    prepared = mgr.record_experiment(run_id="prep", paths=["model.py"])
    assert prepared.experiment_sha != seed_best
    out = tmp_path / "out"
    _install_recovered(out, "recovered_k2c44_last_metrics.json")
    report = run_gated_dfine(
        protocol=protocol,
        plan=_plan(),
        output_dir=out,
        execute=False,
        git_manager=mgr,
        baseline_metrics={"APS": best["APS"], "mAP50_95": best["mAP50_95"]},
    )
    assert report["review_decision"] == "DISCARD"
    assert report["git"]["best_sha"] == seed_best
    assert mgr.current_sha() == mgr.git.rev_parse(seed_best)
    assert mgr.commit_reachable(prepared.experiment_sha)
    assert (repo / "model.py").read_text(encoding="utf-8") == "x = 1\n"

    keep_repo = _git_init(tmp_path / "keep_repo")
    keep_mgr = GitManager(keep_repo)
    keep_out = tmp_path / "keep_out"
    _install_recovered(keep_out, "recovered_k2c44_best_metrics.json")
    keep_report = run_gated_dfine(
        protocol=protocol,
        plan=_plan(),
        output_dir=keep_out,
        execute=False,
        git_manager=keep_mgr,
        baseline_metrics={"APS": best["APS"]},
    )
    assert keep_report["review_decision"] == "KEEP"
    assert keep_report["git"]["best_sha"] == keep_report["git"]["experiment_sha"]
