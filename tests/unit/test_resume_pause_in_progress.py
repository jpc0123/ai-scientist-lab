"""Resume must soft-wait / soft-fail while worker is still draining."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from scientist_lab.services.autonomous_campaign import AutonomousCampaignService
from scientist_lab.services.campaign_notebook import build_llm_judgment, build_now_board


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_can_resume_false_while_worker_alive() -> None:
    campaign = {
        "status": "paused",
        "last_action": "NEED_HUMAN",
        "gpu_rounds": 2,
        "worker_alive": True,
        "steps": [],
    }
    judgment = build_llm_judgment(campaign)
    assert judgment["can_resume"] is False
    now = build_now_board(campaign, [], llm_judgment=judgment)
    assert now["can_resume"] is False
    assert "收尾" in now["next_text"] or "worker" in now["next_text"].lower()


def test_resume_returns_pause_in_progress_while_thread_alive(tmp_path: Path) -> None:
    cid = "camp_pause_drain"
    work = tmp_path / ".run" / "autonomous" / cid
    _write(
        work / "campaign.json",
        {
            "campaign_id": cid,
            "status": "paused",
            "last_action": "NEED_HUMAN",
            "execute": False,
            "confirm_human_gate": True,
            "stop_requested": True,
            "gpu_rounds": 2,
            "ok": True,
            "fail_closed": False,
        },
    )
    _write(work / "protocol.json", {"protocol_id": "p", "stop_rules": {"max_rounds": 12}})
    svc = AutonomousCampaignService(project_root=tmp_path)
    hold = threading.Event()

    def _blocker() -> None:
        hold.wait(timeout=30)

    ghost = threading.Thread(target=_blocker, daemon=True, name=f"autonomous-{cid}")
    with svc._lock:
        svc._threads[cid] = ghost
    ghost.start()
    try:
        out = svc.resume(cid, background=False, live_ready=True, llm_ready=True)
        assert out["ok"] is False
        assert out["fail_closed"] is False
        assert out["pause_in_progress"] is True
        assert out["worker_alive"] is True
    finally:
        hold.set()
        ghost.join(timeout=2)
