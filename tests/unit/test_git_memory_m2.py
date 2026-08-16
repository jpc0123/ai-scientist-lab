"""GitManager dual pointers + MemoryWriter consume-only (no Planner)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scientist_lab.adapters.dfine.run_loop import run_gated_dfine
from scientist_lab.core.git_manager import GitManager
from scientist_lab.core.invariants import InvariantError, assert_memory_refs_resolvable
from scientist_lab.core.memory_writer import MemoryWriter
from scientist_lab.core.schema_registry import SCHEMA_DIR, load_json
from scientist_lab.release.git_adapter import GitAdapter, GitAdapterError

EXAMPLES = SCHEMA_DIR / "examples"

LESSON = {
    "lesson_id": "LESSON-017",
    "type": "positive_evidence",
    "statement": "High-resolution neck enhancement improved small-object performance.",
    "status": "active",
    "evidence": [{"run_id": "EXP-007", "metric": "APS", "delta": 0.67}],
    "scope": {"task": "rgbt_tiny_detection", "module": "neck"},
    "confidence": "medium",
    "created_from": ["EXP-007"],
    "contradicted_by": [],
    "supersedes": [],
    "expires_when": [],
}

STRATEGY = {
    "strategy_id": "STRATEGY-009",
    "action": "prioritize",
    "target": "neck.high_resolution_path",
    "reason_lesson_ids": ["LESSON-017"],
    "status": "active",
}


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


def test_git_adapter_still_denies_reset_hard(tmp_path: Path) -> None:
    root = _git_init(tmp_path / "repo")
    git = GitAdapter(root)
    with pytest.raises(GitAdapterError):
        git._run(["reset", "--hard", "HEAD"])  # noqa: SLF001


def test_record_experiment_does_not_move_best_sha(tmp_path: Path) -> None:
    root = _git_init(tmp_path / "repo")
    mgr = GitManager(root)
    seed = mgr.load().best_sha
    (root / "model.py").write_text("x = 2\n", encoding="utf-8")
    pointers = mgr.record_experiment(run_id="run_1", paths=["model.py"])
    assert pointers.experiment_sha != seed
    assert pointers.best_sha == seed
    assert pointers.runs["run_1"] == pointers.experiment_sha


def test_discard_restores_best_and_keeps_experiment_commit(tmp_path: Path) -> None:
    root = _git_init(tmp_path / "repo")
    mgr = GitManager(root)
    seed = mgr.load().best_sha
    (root / "model.py").write_text("x = 3\n", encoding="utf-8")
    pointers = mgr.record_experiment(run_id="run_neg", paths=["model.py"])
    discarded = pointers.experiment_sha
    after = mgr.apply_review("DISCARD")
    assert after.best_sha == seed
    assert after.experiment_sha == discarded
    assert mgr.current_sha() == mgr.git.rev_parse(seed)
    assert mgr.commit_reachable(discarded)
    assert (root / "model.py").read_text(encoding="utf-8") == "x = 1\n"


def test_keep_updates_best_sha(tmp_path: Path) -> None:
    root = _git_init(tmp_path / "repo")
    mgr = GitManager(root)
    (root / "model.py").write_text("x = 4\n", encoding="utf-8")
    pointers = mgr.record_experiment(run_id="run_keep", paths=["model.py"])
    after = mgr.apply_review("KEEP")
    assert after.best_sha == pointers.experiment_sha


def test_git_manager_does_not_invent_keep(tmp_path: Path) -> None:
    root = _git_init(tmp_path / "repo")
    mgr = GitManager(root)
    seed = mgr.load().best_sha
    (root / "model.py").write_text("x = 5\n", encoding="utf-8")
    pointers = mgr.record_experiment(run_id="run_pending", paths=["model.py"])
    after = mgr.apply_review("PENDING")
    assert after.best_sha == seed
    assert after.experiment_sha == pointers.experiment_sha


def test_memory_writer_refuses_lesson_without_evidence(tmp_path: Path) -> None:
    writer = MemoryWriter(tmp_path / "memory")
    with pytest.raises(InvariantError):
        writer.persist_lesson(
            {
                "lesson_id": "LESSON-BAD",
                "type": "positive_evidence",
                "statement": "FDPN seems promising",
                "status": "active",
                "evidence": [],
                "created_from": [],
            }
        )


def test_memory_writer_persists_reviewer_structure_and_trace(tmp_path: Path) -> None:
    writer = MemoryWriter(tmp_path / "memory")
    report = writer.consume(
        [],
        plan=_plan(),
        review={"research_lessons": [LESSON], "strategy_update": STRATEGY},
        require_resolved_refs=True,
    )
    assert report["lessons_written"] == ["LESSON-017"]
    assert report["strategies_written"] == ["STRATEGY-009"]
    assert report["invented_from_metrics"] is False
    edges = writer.load_trace()
    kinds = {(e["from"], e["to"], e["type"]) for e in edges}
    assert ("EXP-007", "LESSON-017", "derived_from") in kinds
    assert ("LESSON-017", "STRATEGY-009", "used_by") in kinds
    assert ("STRATEGY-009", "plan_round1_neck_hr", "used_by") in kinds


def test_memory_writer_does_not_invent_lessons_from_metrics(tmp_path: Path) -> None:
    writer = MemoryWriter(tmp_path / "memory")
    report = writer.consume(
        [{"event_type": "metrics_parsed", "payload": {"metrics": {"APS": 0.1}}}],
        plan=_plan(),
    )
    assert report["lessons_written"] == []
    assert writer.load_lessons() == {}


def test_unresolved_memory_refs_fail_when_required() -> None:
    with pytest.raises(InvariantError):
        assert_memory_refs_resolvable(_plan(), lesson_ids={}, strategy_ids={})


def test_gated_run_writes_trace_without_inventing_lessons(tmp_path: Path) -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")
    report = run_gated_dfine(
        protocol=protocol,
        plan=_plan(),
        output_dir=tmp_path,
        execute=False,
    )
    assert report["memory"]["lessons_written"] == []
    assert report["memory"]["invented_from_metrics"] is False
    assert report["memory"]["trace_edges"] >= 1
    assert report["review_decision"] == "PENDING"
    events = (tmp_path / "research_events.jsonl").read_text(encoding="utf-8")
    assert "memory_write" in events
    assert (tmp_path / "memory" / "research_trace.json").is_file()


def test_gated_run_records_git_sha_without_keep(tmp_path: Path) -> None:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")
    repo = _git_init(tmp_path / "repo")
    mgr = GitManager(repo)
    seed_best = mgr.load().best_sha
    report = run_gated_dfine(
        protocol=protocol,
        plan=_plan(),
        output_dir=tmp_path / "out",
        execute=False,
        git_manager=mgr,
    )
    assert report["git"]["best_sha"] == seed_best
    assert report["result"]["experiment_sha"]
    assert report["review_decision"] == "PENDING"
    dumped = json.dumps(report)
    assert '"KEEP"' not in dumped
    assert '"DISCARD"' not in dumped
