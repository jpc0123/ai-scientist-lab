"""Extend round budget for completed max_extra_rounds campaigns."""

from __future__ import annotations

import json
from pathlib import Path

from scientist_lab.services.autonomous_campaign import AutonomousCampaignService


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_extend_round_budget_raises_protocol_max(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    cid = "exp_demo_20260101T000000Z"
    work = root / ".run" / "autonomous" / cid
    _write(
        work / "campaign.json",
        {
            "campaign_id": cid,
            "status": "completed",
            "ok": True,
            "execute": True,
            "confirm_human_gate": True,
            "max_extra_rounds": 11,
            "gpu_rounds": 12,
            "last_action": "NEXT_ROUND",
            "steps": [
                {
                    "action": "NEXT_ROUND",
                    "run_state": "MEMORY_WRITTEN",
                    "idle": True,
                    "reasons": ["max_extra_rounds reached; not starting another round"],
                }
            ],
        },
    )
    _write(
        work / "protocol.json",
        {
            "protocol_id": "research_protocol_rgbt_dfine_v26",
            "protocol_version": 2,
            "stop_rules": {"max_rounds": 12},
        },
    )
    svc = AutonomousCampaignService(project_root=root)
    out = svc.extend_round_budget(
        cid,
        add_rounds=5,
        confirm_protocol_amendment=True,
        resume=False,
    )
    assert out["status"] == "paused"
    assert out["max_extra_rounds"] == 16
    proto = json.loads((work / "protocol.json").read_text(encoding="utf-8"))
    assert proto["stop_rules"]["max_rounds"] == 17
    assert proto["protocol_version"] == 3


def test_resume_allows_failed_path_overflow(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    cid = "exp_path_overflow_20260101T000000Z"
    work = root / ".run" / "autonomous" / cid
    long_parent = "run_plan_round11_from_" + ("round_from_" * 20) + "base"
    _write(
        work / "campaign.json",
        {
            "campaign_id": cid,
            "status": "failed",
            "ok": False,
            "execute": False,
            "confirm_human_gate": True,
            "max_extra_rounds": 21,
            "last_action": "NEXT_ROUND",
            "error": (
                "OSError: [Errno 22] Invalid argument: "
                f"'D:\\\\x\\\\runs\\\\{long_parent}.json'"
            ),
        },
    )
    _write(
        work / "protocol.json",
        {"protocol_id": "p", "protocol_version": 1, "stop_rules": {"max_rounds": 20}},
    )
    # Minimal manager files so worker can start then idle/fail later; resume
    # itself must accept the failed+OSError state.
    _write(
        work / "experiment_run.json",
        {
            "run_id": long_parent,
            "round_index": 11,
            "run_state": "MEMORY_WRITTEN",
            "evidence_status": "VALID",
            "review_decision": "KEEP",
        },
    )
    svc = AutonomousCampaignService(project_root=root)
    out = svc.resume(cid, background=False, live_ready=False, llm_ready=False)
    # execute=False → no live gate; worker may stop quickly, but resume accepted.
    assert out["campaign_id"] == cid
    assert out.get("error") is None or "OSError" not in str(out.get("error"))
