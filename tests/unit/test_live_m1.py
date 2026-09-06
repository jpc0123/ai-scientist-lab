"""Live M1: LLM-driven experiment brief, not mechanical catalog queue."""

from __future__ import annotations

import json
from pathlib import Path

from scientist_lab.services.evaluation_matrix import build_live_experiment_brief
from scientist_lab.services.sota_pursuit import refresh_live_m1_brief, run_sota_pursuit_tick


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_live_brief_is_advisory_not_script() -> None:
    protocol = {
        "title": "V26",
        "baseline": {"adapter": "dfine"},
        "objective": {"primary": {"metric": "APS_lowlight"}},
        "condition_slice": {"id": "low_light_subset_v1"},
    }
    campaign = {"gpu_rounds": 1, "steps": []}
    rounds = [{"how_id": "F1", "seed": 42, "primary_value": 0.005, "round_index": 1}]
    brief = build_live_experiment_brief(campaign, protocol=protocol, rounds=rounds)
    assert brief.get("advisory_only") is True
    assert brief.get("live_m1") is True
    assert "open_scientific_questions" in brief
    assert len(brief["open_scientific_questions"]) >= 1
    assert "F0" in str(brief.get("gaps"))


def test_live_m1_tick_refreshes_brief_without_catalog_queue(tmp_path: Path) -> None:
    work = tmp_path / "camp"
    work.mkdir()
    _write_json(
        work / "campaign.json",
        {
            "gpu_rounds": 1,
            "experiment_id": "exp_rgbt_dfine_v26_lowlight",
            "steps": [
                {"action": "NEED_PLAN", "report": {"plan": {"how_id": "F1"}}},
                {
                    "action": "NEED_PARSE",
                    "report": {"result": {"metrics": {"APS_lowlight": 0.005}}},
                },
            ],
        },
    )
    _write_json(
        work / "protocol.json",
        {
            "title": "V26",
            "baseline": {"adapter": "dfine"},
            "objective": {"primary": {"metric": "APS_lowlight"}},
            "condition_slice": {"id": "low_light_subset_v1"},
        },
    )
    pursuit = run_sota_pursuit_tick(work, {"live_m1": True})
    assert pursuit["action"] == "brief_refreshed"
    assert pursuit["progressed"] is False
    assert pursuit.get("live_m1_brief", {}).get("advisory_only") is True
    from scientist_lab.core.how_pending import load_store, pending_path

    store = load_store(pending_path(work))
    assert not store.get("candidates")
