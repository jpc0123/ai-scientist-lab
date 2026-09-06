"""Read-only ATDP six-tuple exporter."""

from __future__ import annotations

import json
from pathlib import Path

from scientist_lab.cli import build_parser
from scientist_lab.core.schema_registry import SCHEMA_DIR, validate_named
from scientist_lab.llm.contrast_holdout import training_leaks_holdout
from scientist_lab.llm.trajectory_export import (
    EXPORTER_VERSION,
    build_steps,
    export_run,
    export_runs,
    load_run_bundle,
)


def _dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _protocol() -> dict:
    return {
        "schema_version": "1.0.0",
        "project_id": "project_rgbt_cuda_001",
        "protocol_id": "research_protocol_rgbt_dfine_v26",
        "protocol_version": 2,
        "goal": {"improve": "lowlight_small_object_detection"},
        "baseline": {"model": "DFINE-S", "dataset": "dataset:rgbt_tiny_v1", "adapter": "dfine"},
        "objective": {"primary": {"metric": "APS_lowlight", "direction": "maximize"}},
        "editable_scope": ["fusion", "neck"],
        "frozen_scope": ["dataset_split", "evaluator", "metric_definition", "condition_slice"],
        "experiment_budget": {"probe": "10min", "validation": "60min", "formal": "4h"},
        "risk_policy": {"probe_training": "auto", "full_training": "approval_required"},
        "decision_policy": {},
        "stop_rules": {"max_rounds": 12},
        "fingerprint_id": "FP-RGBT-DFINE-V26-LOWLIGHT",
        "condition_slice": {"id": "low_light_subset_v1"},
    }


def _plan(*, how_id: str, seeds: list[int], plan_id: str, parent: str | None, round_index: int) -> dict:
    return {
        "schema_version": "1.0.0",
        "plan_id": plan_id,
        "project_id": "project_rgbt_cuda_001",
        "protocol_id": "research_protocol_rgbt_dfine_v26",
        "protocol_version": 2,
        "parent_run_id": parent,
        "round_index": round_index,
        "observation": "上一轮已经跑过比较基准。",
        "hypothesis": f"对照 HOW {how_id}",
        "modification_scope": ["fusion"],
        "proposed_changes": [
            {
                "target": "fusion",
                "summary": f"use {how_id}",
                "detail": {"how_id": how_id, "neck_type": "standard"},
            }
        ],
        "controlled_variables": ["dataset_split", "evaluator", "seed"],
        "expected_effect": {
            "primary_metric": "APS_lowlight",
            "direction": "increase",
            "rationale": "single-variable fusion contrast",
        },
        "evaluation": {"method": "full_train", "seeds": seeds},
        "budget_class": "formal",
        "risk_level": "auto",
        "memory_refs": {"lesson_ids": [], "strategy_ids": []},
        "evidence_runs": [parent] if parent else [],
        "bootstrap": parent is None,
        "how_id": how_id,
        "decision_summary": {
            "problem_observed": "需要相对上一轮做对照",
            "hypothesis": f"试 {how_id}",
            "selected_action": f"contrast_{how_id}",
            "decision_basis": ["single variable", "catalog how"],
            "expected_effect": "fair contrast",
            "risk": "auto",
        },
    }


def _event(event_type: str, payload: dict, **extra) -> dict:
    row = {
        "schema_version": "1.0.0",
        "event_id": f"evt_{event_type}",
        "ts": "2026-08-19T00:00:00+00:00",
        "project_id": "project_rgbt_cuda_001",
        "event_type": event_type,
        "actor_role": extra.pop("actor_role", "manager"),
        "phase": extra.pop("phase", "planning"),
        "payload": payload,
        "reconstructed": False,
    }
    row.update(extra)
    return row


def _write_run(
    root: Path,
    *,
    run_id: str,
    how_id: str,
    seeds: list[int],
    parent: str | None,
    round_index: int,
    review: str = "KEEP",
    aps: float = 0.02,
    events: bool = True,
    idle: bool = False,
    human_reject: bool = False,
    previous_plan: dict | None = None,
) -> Path:
    plan = _plan(
        how_id=how_id,
        seeds=seeds,
        plan_id=f"plan_{run_id}",
        parent=parent,
        round_index=round_index,
    )
    if idle:
        plan["hypothesis"] = "repeat the same combination"
    contract = {
        "run_id": run_id,
        "plan_id": plan["plan_id"],
        "seed": seeds[0],
        "budget_class": "formal",
        "dataset": {"reference": "dataset:rgbt_tiny_v1"},
        "allowed_changes": ["fusion"],
        "frozen_variables": ["dataset_split", "evaluator"],
        "materialization": {
            "how": {
                "how_id": how_id,
                "fusion_method": "gated_multiscale" if how_id == "F3" else "early_concat",
                "neck_type": "standard",
            }
        },
    }
    result = {
        "run_id": run_id,
        "metrics": {"APS_lowlight": aps},
        "execution": {"status": "success"},
    }
    review_doc = {
        "review_decision": review,
        "hypothesis_status": "INCONCLUSIVE",
        "reasoning_summary": "rubric KEEP is not a claim",
        "research_lessons": [{"lesson_id": f"LESSON-{run_id}-001"}],
        "strategies": [{"strategy_id": f"STRATEGY-{run_id}-001"}],
        "primary_metric_judgment": {
            "metric": "APS_lowlight",
            "before": 0.0046,
            "after": aps,
            "delta": aps - 0.0046,
        },
    }
    run_doc = {
        "run_id": run_id,
        "project_id": "project_rgbt_cuda_001",
        "run_state": "MEMORY_WRITTEN",
        "review_decision": review,
        "round_index": round_index,
        "parent_run_id": parent,
        "plan_id": plan["plan_id"],
        "fingerprint_id": "FP-RGBT-DFINE-V26-LOWLIGHT",
        "experiment_sha": "abcdef1",
    }
    _dump(root / "protocol.json", _protocol())
    _dump(root / "plan.json", plan)
    _dump(root / "contract.json", contract)
    _dump(root / "result.json", result)
    _dump(root / "review.json", review_doc)
    _dump(root / "experiment_run.json", run_doc)
    _dump(root / "metrics.json", {"APS_lowlight": aps, "training": {"seed": seeds[0], "how_id": how_id}})
    _dump(root / "claim_gate.json", {"status": "BLOCKED"})
    _dump(root / "baseline_metrics.json", {"metrics": {"APS_lowlight": 0.0046}})
    if previous_plan is not None:
        _dump(root / "previous_plan.json", previous_plan)
    if events:
        rows = [
            _event(
                "plan_proposal",
                {"source": "llm", "plan_id": plan["plan_id"]},
                plan_id=plan["plan_id"],
                run_id=run_id,
                phase="planning",
                actor_role="planner",
            ),
            _event(
                "gate_decision",
                {
                    "status": "NEED_HUMAN" if human_reject else "APPROVED",
                    "reasons": ["budget"] if human_reject else ["ok"],
                },
                plan_id=plan["plan_id"],
                run_id=run_id,
                phase="planning",
                actor_role="gate",
            ),
            _event(
                "execution",
                {"status": "completed", "metrics_forged": False},
                run_id=run_id,
                phase="experiment",
                actor_role="executor",
            ),
            _event(
                "metrics_parsed",
                {"APS_lowlight": aps, "metrics_forged": False},
                run_id=run_id,
                phase="experiment",
                actor_role="manager",
            ),
            _event(
                "review_decision",
                {"review_decision": review},
                run_id=run_id,
                phase="review",
                actor_role="reviewer",
                decision_summary={
                    "selected_action": review,
                    "decision_basis": ["rubric"],
                    "risk": "KEEP ≠ Claim",
                },
            ),
        ]
        if human_reject:
            rows.append(
                _event(
                    "human_governance",
                    {"action": "NEED_HUMAN", "reason": "budget exhausted"},
                    run_id=run_id,
                    phase="governance",
                    actor_role="human",
                )
            )
        (root / "research_events.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    return root


def test_example_trajectory_step_validates() -> None:
    from scientist_lab.core.schema_registry import load_json

    doc = load_json(SCHEMA_DIR / "examples" / "trajectory_step_plan_proposal.json")
    validate_named("trajectory_step", doc)
    assert doc["r"] is None


def test_round1_how_contrast_maps_four_steps(tmp_path: Path) -> None:
    parent_plan = _plan(how_id="F1", seeds=[42], plan_id="plan_r0", parent=None, round_index=0)
    run_dir = _write_run(
        tmp_path / "v26_r1",
        run_id="run_v26_r1",
        how_id="F3",
        seeds=[42],
        parent="run_v26_r0",
        round_index=1,
        previous_plan=parent_plan,
    )
    events_before = (run_dir / "research_events.jsonl").read_text(encoding="utf-8")
    export_root = tmp_path / "export"
    report = export_run(run_dir, export_root)
    events_after = (run_dir / "research_events.jsonl").read_text(encoding="utf-8")
    assert events_after == events_before
    assert report["n_steps"] == 4
    assert report["contrast_label"] == "how_contrast"
    assert report["high_trust_sft"] is True
    steps = [
        json.loads(line)
        for line in (export_root / "traces" / "run_v26_r1.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["step_kind"] for row in steps] == [
        "plan_proposal",
        "gate_decision",
        "execution",
        "review_decision",
    ]
    for row in steps:
        validate_named("trajectory_step", row)
        assert row["r"] is None
        assert row["m"]["exporter_version"] == EXPORTER_VERSION
        assert row["m"]["reconstructed"] is False
    assert steps[0]["a"]["how_id"] == "F3"
    assert steps[0]["a"]["how_zh"]
    assert "APS_lowlight" not in json.dumps(steps[0]["o"], ensure_ascii=False)
    assert steps[0]["o"]["anchor"]["how_id"] == "F1"
    assert steps[2]["y"]["primary_value"] == 0.02
    assert steps[3]["a"]["action"] == "KEEP"
    sft = json.loads((export_root / "sft" / "run_v26_r1.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert "F1" in sft["input"]
    assert "拼成一张图" in sft["input"]
    assert "APS_lowlight" not in sft["input"]
    assert "APS_lowlight" not in json.dumps(sft["output"], ensure_ascii=False)
    assert sft["output"]["contract"]["how_id"] == "F3"
    assert "0.02" not in sft["input"]


def test_idle_is_dpo_not_sft(tmp_path: Path) -> None:
    parent_plan = _plan(how_id="F1", seeds=[42], plan_id="plan_r0", parent=None, round_index=0)
    run_dir = _write_run(
        tmp_path / "p1",
        run_id="run_p1_idle",
        how_id="F1",
        seeds=[42],
        parent="run_r0",
        round_index=1,
        idle=True,
        previous_plan=parent_plan,
    )
    report = export_run(run_dir, tmp_path / "export")
    assert report["contrast_label"] == "idle"
    assert report["high_trust_sft"] is False
    assert (tmp_path / "export" / "dpo" / "run_p1_idle.jsonl").is_file()
    assert not (tmp_path / "export" / "sft" / "run_p1_idle.jsonl").is_file()


def test_reconstructed_round_is_not_high_trust_sft(tmp_path: Path) -> None:
    parent_plan = _plan(how_id="F1", seeds=[42], plan_id="plan_r0", parent=None, round_index=0)
    run_dir = _write_run(
        tmp_path / "old",
        run_id="run_old",
        how_id="F3",
        seeds=[42],
        parent="run_r0",
        round_index=1,
        events=False,
        previous_plan=parent_plan,
    )
    report = export_run(run_dir, tmp_path / "export")
    assert report["reconstructed"] is True
    assert report["high_trust_sft"] is False
    steps = [
        json.loads(line)
        for line in (tmp_path / "export" / "traces" / "run_old.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert all(row["m"]["reconstructed"] is True for row in steps)
    assert not (tmp_path / "export" / "sft" / "run_old.jsonl").is_file()


def test_human_budget_reject_is_dropped(tmp_path: Path) -> None:
    parent_plan = _plan(how_id="F1", seeds=[42], plan_id="plan_r0", parent=None, round_index=0)
    run_dir = _write_run(
        tmp_path / "reject",
        run_id="run_budget",
        how_id="F3",
        seeds=[42],
        parent="run_r0",
        round_index=1,
        human_reject=True,
        previous_plan=parent_plan,
    )
    report = export_run(run_dir, tmp_path / "export")
    assert report["high_trust_sft"] is False
    story = json.loads((tmp_path / "export" / "stories" / "run_budget.json").read_text(encoding="utf-8"))
    assert story["bucket"] == "drop"
    assert not (tmp_path / "export" / "sft" / "run_budget.jsonl").is_file()
    assert not (tmp_path / "export" / "dpo" / "run_budget.jsonl").is_file()


def test_batch_pairs_idle_with_how_contrast(tmp_path: Path) -> None:
    parent_plan = _plan(how_id="F1", seeds=[42], plan_id="plan_r0", parent=None, round_index=0)
    good = _write_run(
        tmp_path / "good",
        run_id="run_good",
        how_id="F3",
        seeds=[42],
        parent="run_r0",
        round_index=1,
        previous_plan=parent_plan,
    )
    idle = _write_run(
        tmp_path / "idle",
        run_id="run_idle",
        how_id="F1",
        seeds=[42],
        parent="run_r0",
        round_index=1,
        idle=True,
        previous_plan=parent_plan,
    )
    manifest = export_runs([good, idle], tmp_path / "batch")
    assert manifest["counts"]["sft"] == 1
    assert manifest["counts"]["dpo"] == 1
    dpo = json.loads((tmp_path / "batch" / "dpo" / "batch.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert dpo["pair_complete"] is True
    assert dpo["chosen"]["contract"]["how_id"] == "F3"
    sft = [
        json.loads(line)
        for line in (tmp_path / "batch" / "sft" / "batch.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert training_leaks_holdout(sft) == []
    assert all("contrast_m" not in json.dumps(row) for row in sft)


def test_parent_dir_fills_anchor(tmp_path: Path) -> None:
    parent = _write_run(
        tmp_path / "r0",
        run_id="run_v26_r0",
        how_id="F1",
        seeds=[42],
        parent=None,
        round_index=0,
        aps=0.0046,
    )
    child = _write_run(
        tmp_path / "r1",
        run_id="run_v26_r1",
        how_id="F3",
        seeds=[42],
        parent="run_v26_r0",
        round_index=1,
    )
    steps = build_steps(load_run_bundle(child), load_run_bundle(parent))
    assert steps[0]["o"]["anchor"]["how_id"] == "F1"
    assert steps[0]["o"]["anchor"]["seeds"] == [42]


def test_cli_parses_export_trajectory() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["export-trajectory", "--run-dir", "outputs/v26_r1", "--parent-dir", "outputs/v26_r0"]
    )
    assert args.command == "export-trajectory"
    assert args.run_dirs[0].name == "v26_r1"


def test_holdout_case_id_refuses_training_export(tmp_path: Path) -> None:
    parent_plan = _plan(how_id="F1", seeds=[42], plan_id="plan_r0", parent=None, round_index=0)
    run_dir = _write_run(
        tmp_path / "leak",
        run_id="contrast_m01",
        how_id="F3",
        seeds=[42],
        parent="run_r0",
        round_index=1,
        previous_plan=parent_plan,
    )
    try:
        export_run(run_dir, tmp_path / "export")
        raise AssertionError("expected holdout leak to fail closed")
    except ValueError as exc:
        assert "holdout" in str(exc)
