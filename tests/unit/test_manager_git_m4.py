"""Manager wires GitManager: KEEP/DISCARD/REPLICATE. No GPU. No reset --hard."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from scientist_lab.core.git_manager import GitManager, is_git_repo
from scientist_lab.core.manager import Manager, _apply_state
from scientist_lab.core.schema_registry import SCHEMA_DIR, load_json
from scientist_lab.core.state_machine import (
    EvidenceStatus,
    ExperimentRunState,
    ReviewDecisionValue,
    RunState,
    transition,
)

EXAMPLES = SCHEMA_DIR / "examples"


def _seed_plan(**overrides):
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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def _install_replay(
    root: Path,
    *,
    metrics_name: str = "recovered_k2c44_last_metrics.json",
    with_baseline: bool = True,
) -> dict:
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")
    shutil.copyfile(EXAMPLES / metrics_name, root / "metrics.json")
    shutil.copyfile(
        EXAMPLES / "recovered_k2c44_checkpoint_selection.json",
        root / "checkpoint_selection.json",
    )
    _write_json(root / "protocol.json", protocol)
    _write_json(root / "plan.json", _seed_plan())
    if with_baseline:
        best = load_json(EXAMPLES / "recovered_k2c44_best_metrics.json")
        _write_json(root / "baseline_metrics.json", {"APS": best["APS"], "mAP50_95": best["mAP50_95"]})
    return protocol


def _git_events(root: Path) -> list[dict]:
    path = root / "research_events.jsonl"
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        payload = event.get("payload") or {}
        if payload.get("tool") == "git_manager":
            rows.append(event)
    return rows


def _walk_to_completed(state: ExperimentRunState) -> ExperimentRunState:
    for rs in (
        RunState.PLANNED,
        RunState.MATERIALIZED,
        RunState.APPROVED,
        RunState.RUNNING,
        RunState.COMPLETED,
    ):
        state = transition(state, run_state=rs)
    return state


def test_keep_updates_best_sha_to_experiment(tmp_path: Path) -> None:
    root = _git_init(tmp_path / "repo")
    (root / "model.py").write_text("x = keep\n", encoding="utf-8")
    protocol = _install_replay(root, metrics_name="recovered_k2c44_best_metrics.json")
    seed_best = GitManager(root).load().best_sha
    mgr = Manager(
        root,
        protocol=protocol,
        execute=False,
        max_extra_rounds=0,
        git_record_paths=["model.py"],
    )
    mgr.initialize_run(round_index=1, plan_id="plan_round1_neck_hr")
    steps = mgr.run_until(max_steps=20)
    assert steps[-1].state.review_decision == ReviewDecisionValue.KEEP
    assert steps[-1].state.evidence_status == EvidenceStatus.VALID
    pointers = GitManager(root).load()
    assert pointers.experiment_sha != seed_best
    assert pointers.best_sha == pointers.experiment_sha
    apply_events = [e for e in _git_events(root) if e["payload"].get("action") == "apply_review"]
    assert apply_events
    assert apply_events[-1]["payload"]["best_sha_updated"] is True
    assert (root / "model.py").read_text(encoding="utf-8") == "x = keep\n"


def test_discard_keeps_best_and_restores_worktree(tmp_path: Path) -> None:
    root = _git_init(tmp_path / "repo")
    (root / "model.py").write_text("x = discard\n", encoding="utf-8")
    protocol = _install_replay(root)
    git = GitManager(root)
    seed_best = git.load().best_sha
    mgr = Manager(
        root,
        protocol=protocol,
        execute=False,
        max_extra_rounds=0,
        git_manager=git,
        git_record_paths=["model.py"],
    )
    mgr.initialize_run(round_index=1, plan_id="plan_round1_neck_hr")
    steps = mgr.run_until(max_steps=20)
    assert steps[-1].state.review_decision == ReviewDecisionValue.DISCARD
    pointers = git.load()
    assert pointers.best_sha == seed_best
    assert pointers.experiment_sha != seed_best
    assert git.commit_reachable(pointers.experiment_sha)
    assert git.current_sha() == git.git.rev_parse(seed_best)
    assert (root / "model.py").read_text(encoding="utf-8") == "x = 1\n"
    apply_events = [e for e in _git_events(root) if e["payload"].get("action") == "apply_review"]
    assert apply_events[-1]["payload"]["best_sha_updated"] is False


def test_replicate_does_not_move_best_sha(tmp_path: Path) -> None:
    root = _git_init(tmp_path / "repo")
    (root / "model.py").write_text("x = replicate\n", encoding="utf-8")
    protocol = _install_replay(
        root,
        metrics_name="recovered_k2c44_best_metrics.json",
        with_baseline=False,
    )
    git = GitManager(root)
    seed_best = git.load().best_sha
    mgr = Manager(
        root,
        protocol=protocol,
        execute=False,
        max_extra_rounds=0,
        git_manager=git,
        git_record_paths=["model.py"],
    )
    mgr.initialize_run(round_index=1, plan_id="plan_round1_neck_hr")
    steps = mgr.run_until(max_steps=20)
    assert steps[-1].state.review_decision == ReviewDecisionValue.REPLICATE
    pointers = git.load()
    assert pointers.best_sha == seed_best
    assert pointers.experiment_sha != seed_best
    assert (root / "model.py").read_text(encoding="utf-8") == "x = replicate\n"
    apply_events = [e for e in _git_events(root) if e["payload"].get("action") == "apply_review"]
    assert apply_events[-1]["payload"]["review_decision"] == "REPLICATE"
    assert apply_events[-1]["payload"]["best_sha_updated"] is False


def test_non_git_directory_skips_without_crash(tmp_path: Path) -> None:
    assert not is_git_repo(tmp_path)
    protocol = _install_replay(tmp_path)
    mgr = Manager(tmp_path, protocol=protocol, execute=False, max_extra_rounds=0)
    mgr.initialize_run(round_index=1, plan_id="plan_round1_neck_hr")
    steps = mgr.run_until(max_steps=20)
    assert steps[-1].state.run_state == RunState.MEMORY_WRITTEN
    snapshot = json.loads((tmp_path / "git_pointers.json").read_text(encoding="utf-8"))
    assert snapshot["skipped"] is True
    assert "not a git repository" in str(snapshot.get("reason") or "")
    git_events = _git_events(tmp_path)
    assert git_events
    assert all(e["payload"].get("skipped") is True for e in git_events)
    assert all(e["payload"].get("best_sha_updated") is not True for e in git_events)


def test_invalid_evidence_does_not_apply_keep(tmp_path: Path) -> None:
    root = _git_init(tmp_path / "repo")
    (root / "model.py").write_text("x = invalid\n", encoding="utf-8")
    protocol = _install_replay(root)
    git = GitManager(root)
    seed = git.load()
    recorded = git.record_experiment(run_id="pre_invalid", paths=["model.py"])
    assert recorded.best_sha == seed.best_sha
    mgr = Manager(
        root,
        protocol=protocol,
        execute=False,
        max_extra_rounds=0,
        git_manager=git,
        git_record_paths=["model.py"],
    )
    doc = mgr.initialize_run(round_index=1)
    state = _walk_to_completed(ExperimentRunState.initial())
    state = transition(state, evidence_status=EvidenceStatus.INVALID)
    _write_json(root / "experiment_run.json", _apply_state(doc, state))
    step = mgr.step()
    assert step.state.run_state == RunState.MEMORY_WRITTEN
    assert step.state.review_decision == ReviewDecisionValue.PENDING
    assert not (root / "review.json").is_file()
    after = git.load()
    assert after.best_sha == seed.best_sha
    assert after.experiment_sha == recorded.experiment_sha
    apply_keep = [
        e
        for e in _git_events(root)
        if e["payload"].get("action") == "apply_review"
        and e["payload"].get("review_decision") == "KEEP"
        and e["payload"].get("skipped") is not True
    ]
    assert apply_keep == []
    dumped = json.dumps(step.to_dict())
    assert '"KEEP"' not in dumped
