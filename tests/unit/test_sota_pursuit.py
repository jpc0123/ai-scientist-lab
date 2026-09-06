"""SOTA pursuit: board, catalog queue, plugin fallback."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scientist_lab.core.how_pending import (
    decide_candidate,
    ingest_catalog_exploration_candidate,
    ingest_plugin_exploration_candidate,
    load_store,
    pending_path,
)
from scientist_lab.services.sota_pursuit import (
    build_sota_board,
    run_sota_pursuit_tick,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_sota_board_remaining_after_f0(tmp_path: Path) -> None:
    campaign = {
        "gpu_rounds": 2,
        "steps": [
            {
                "action": "NEED_PLAN",
                "report": {"plan": {"how_id": "F0"}},
            },
            {
                "action": "NEED_PARSE",
                "report": {"result": {"metrics": {"APS_lowlight": 0.011}}},
            },
        ],
    }
    baseline = {"APS_lowlight": 0.009, "source": "formal_seed"}
    board = build_sota_board(campaign, baseline=baseline)
    assert board["baseline"] == pytest.approx(0.009)
    assert board["best"] == pytest.approx(0.011)
    assert board["next_catalog_how_id"] == "F3"
    assert "F0" in board["used_how_ids"]
    assert "F3" in board["remaining_catalog_how_ids"]


def test_ingest_catalog_exploration_candidate(tmp_path: Path) -> None:
    dest = pending_path(tmp_path / "work")
    store = ingest_catalog_exploration_candidate(dest, "F3", round_id="round_3", reason="try F3")
    rows = store["candidates"]
    assert len(rows) == 1
    row = rows[0]
    assert row["how_id"] == "F3"
    assert row["source"] == "evidence_explore"
    assert row["literature_query_id"] == "evidence_explore"
    assert row["map_to_existing"] == "F3"


def test_run_sota_pursuit_tick_registers_catalog(tmp_path: Path) -> None:
    work = tmp_path / "campaign"
    work.mkdir()
    _write_json(
        work / "campaign.json",
        {
            "gpu_rounds": 2,
            "steps": [
                {"action": "NEED_PLAN", "report": {"plan": {"how_id": "F0"}}},
                {
                    "action": "NEED_PARSE",
                    "report": {"result": {"metrics": {"APS_lowlight": 0.011}}},
                },
            ],
        },
    )
    _write_json(work / "baseline_metrics.json", {"APS_lowlight": 0.009})
    _write_json(
        work / "protocol.json",
        {
            "title": "Low-light RGB-T v26",
            "baseline": {"adapter": "dfine"},
            "objective": {"primary": {"metric": "APS_lowlight"}},
            "condition_slice": {"id": "low_light_subset_v1"},
        },
    )

    pursuit = run_sota_pursuit_tick(work, {"confirm_human_gate": True, "live_m1": False})
    assert pursuit["progressed"] is True
    assert pursuit["action"] == "register_catalog"
    assert "F3" in str(pursuit["reason"])

    store = load_store(pending_path(work))
    registered = [r for r in store["candidates"] if r.get("how_id") == "F3"]
    assert registered
    assert registered[0]["status"] == "registered"


def test_run_sota_pursuit_plugin_when_catalog_exhausted(tmp_path: Path) -> None:
    work = tmp_path / "campaign"
    work.mkdir()
    used = ("F3", "N1", "A4", "F1", "F0", "N0")
    steps = []
    for hid in used:
        steps.append({"action": "NEED_PLAN", "report": {"plan": {"how_id": hid}}})
    steps.append(
        {
            "action": "NEED_PARSE",
            "report": {"result": {"metrics": {"APS_lowlight": 0.015}}},
        }
    )
    _write_json(work / "campaign.json", {"gpu_rounds": 6, "steps": steps})
    _write_json(work / "baseline_metrics.json", {"APS_lowlight": 0.009})
    _write_json(
        work / "protocol.json",
        {
            "title": "Low-light RGB-T v26",
            "baseline": {"adapter": "dfine"},
            "objective": {"primary": {"metric": "APS_lowlight"}},
            "condition_slice": {"id": "low_light_subset_v1"},
        },
    )

    pursuit = run_sota_pursuit_tick(
        work,
        {"llm_how_lifecycle": True, "llm_live": True, "live_m1": False},
    )
    assert pursuit["progressed"] is True
    assert pursuit["action"] == "propose_plugin"
    store = load_store(pending_path(work))
    plugin = [r for r in store["candidates"] if str(r.get("how_id", "")).startswith("P")]
    assert plugin
    assert plugin[0]["source"] == "plugin_explore"


def test_ingest_plugin_exploration_candidate_rejects_catalog_id(tmp_path: Path) -> None:
    dest = pending_path(tmp_path / "work")
    with pytest.raises(Exception):
        ingest_plugin_exploration_candidate(dest, "F0", "should fail")
