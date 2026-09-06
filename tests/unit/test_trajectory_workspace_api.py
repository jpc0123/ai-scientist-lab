"""Read-only trajectory workspace API: ledger events + six-tuple map."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from scientist_lab.api.app import create_app
from scientist_lab.services.experiment_service import ExperimentService
from scientist_lab.settings import Settings


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


def _write_run(root: Path) -> None:
    how_id = "F3"
    seeds = [42]
    run_id = "run_v26_r1"
    plan = _plan(how_id=how_id, seeds=seeds, plan_id="plan_r1", parent="run_v26_r0", round_index=1)
    previous = _plan(how_id="F1", seeds=[42], plan_id="plan_r0", parent=None, round_index=0)
    _dump(root / "protocol.json", _protocol())
    _dump(root / "plan.json", plan)
    _dump(root / "previous_plan.json", previous)
    _dump(
        root / "contract.json",
        {
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
                    "fusion_method": "gated_multiscale",
                    "neck_type": "standard",
                }
            },
        },
    )
    _dump(
        root / "result.json",
        {"run_id": run_id, "metrics": {"APS_lowlight": 0.02}, "execution": {"status": "success"}},
    )
    _dump(
        root / "review.json",
        {
            "review_decision": "KEEP",
            "hypothesis_status": "INCONCLUSIVE",
            "reasoning_summary": "rubric KEEP is not a claim",
            "research_lessons": [{"lesson_id": "LESSON-r1-001"}],
            "strategies": [{"strategy_id": "STRATEGY-r1-001"}],
            "primary_metric_judgment": {
                "metric": "APS_lowlight",
                "before": 0.0046,
                "after": 0.02,
                "delta": 0.0154,
            },
        },
    )
    _dump(
        root / "experiment_run.json",
        {
            "run_id": run_id,
            "project_id": "project_rgbt_cuda_001",
            "run_state": "MEMORY_WRITTEN",
            "review_decision": "KEEP",
            "round_index": 1,
            "parent_run_id": "run_v26_r0",
            "plan_id": plan["plan_id"],
            "fingerprint_id": "FP-RGBT-DFINE-V26-LOWLIGHT",
            "experiment_sha": "abcdef1",
        },
    )
    _dump(root / "metrics.json", {"APS_lowlight": 0.02, "training": {"seed": 42, "how_id": how_id}})
    _dump(root / "claim_gate.json", {"status": "BLOCKED"})
    _dump(root / "baseline_metrics.json", {"metrics": {"APS_lowlight": 0.0046}})
    rows = [
        _event("plan_proposal", {"source": "llm"}, plan_id=plan["plan_id"], run_id=run_id, actor_role="planner"),
        _event("gate_decision", {"status": "APPROVED"}, plan_id=plan["plan_id"], run_id=run_id, actor_role="gate"),
        _event("execution", {"status": "completed"}, run_id=run_id, phase="experiment", actor_role="executor"),
        _event(
            "review_decision",
            {"review_decision": "KEEP"},
            run_id=run_id,
            phase="review",
            actor_role="reviewer",
            decision_summary={"selected_action": "KEEP", "decision_basis": ["rubric"], "risk": "KEEP ≠ Claim"},
        ),
    ]
    (root / "research_events.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _client(tmp_path: Path) -> TestClient:
    root = tmp_path / "lab"
    run = root / "outputs" / "v26_r1"
    run.mkdir(parents=True)
    _write_run(run)
    campaign = root / ".run" / "autonomous" / "camp_demo"
    campaign.mkdir(parents=True)
    (campaign / "research_events.jsonl").write_text(
        json.dumps(
            _event("plan_proposal", {"source": "campaign"}, run_id="camp_demo", actor_role="planner"),
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    _dump(campaign / "campaign.json", {"title": "演示战役", "experiment_id": "exp_demo"})
    service = ExperimentService(
        settings=Settings(
            project_root=root,
            db_path=tmp_path / "api.db",
            runtime_dir=tmp_path / "runtime",
            outputs_dir=root / "outputs",
            experiment_app_dir=root / "experiment_app",
        ).resolve()
    )
    return TestClient(create_app(service=service))


def test_list_and_show_research_and_mapped_trajectories(tmp_path: Path) -> None:
    client = _client(tmp_path)
    listed = client.get("/api/v1/trajectories")
    assert listed.status_code == 200
    payload = listed.json()
    ids = {row["id"] for row in payload["items"]}
    assert "v26_r1" in ids
    assert "campaign_camp_demo" in ids
    v26 = next(row for row in payload["items"] if row["id"] == "v26_r1")
    assert v26["has_events"] is True
    assert v26["event_count"] == 4
    assert v26["mappable"] is True

    detail = client.get("/api/v1/trajectories/v26_r1")
    assert detail.status_code == 200
    body = detail.json()
    assert len(body["events"]) == 4
    assert body["events"][0]["event_type"] == "plan_proposal"
    assert body["events"][0]["record"]["event_id"] == "evt_plan_proposal"
    assert body["mapping_error"] is None
    kinds = [row["step_kind"] for row in body["mapped_steps"]]
    assert kinds == ["plan_proposal", "gate_decision", "execution", "review_decision"]
    assert body["mapped_steps"][0]["record"]["r"] is None
    assert set(body["mapped_steps"][0]["record"]) >= {"o", "h", "a", "y", "r", "m"}
    assert body["story"]["contrast_label"]

    campaign = client.get("/api/v1/trajectories/campaign_camp_demo")
    assert campaign.status_code == 200
    camp = campaign.json()
    assert len(camp["events"]) == 1
    assert camp["events"][0]["event_type"] == "plan_proposal"


def test_missing_trajectory_is_404(tmp_path: Path) -> None:
    client = _client(tmp_path)
    resp = client.get("/api/v1/trajectories/does_not_exist")
    assert resp.status_code == 404
