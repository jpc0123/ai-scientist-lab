"""Draft Arm: literature HOW drafts; invent fallback awaits human."""

from __future__ import annotations

from pathlib import Path

import pytest

from scientist_lab.core.how_draft_arm import (
    ingest_invent_fallback_candidate,
    release_invent_for_lifecycle,
    run_how_draft_arm,
)
from scientist_lab.core.how_lifecycle import run_how_lifecycle_tick
from scientist_lab.core.how_pending import (
    STATUS_PROPOSED,
    HowPendingError,
    load_store,
    pending_path,
    persist_scout,
)
from scientist_lab.llm.gateway import ScriptedProvider

from tests.unit.test_how_candidates_freeze import _packet
from tests.unit.test_how_plugin_bridge import ROOT


def test_draft_arm_from_literature(tmp_path: Path) -> None:
    packet = _packet(tmp_path)
    path = pending_path(tmp_path / "campaign")
    persist_scout(path, packet)
    result = run_how_draft_arm(
        path,
        confirm_human_gate=True,
        provider=None,
        round_id="r_draft",
    )
    assert result["progressed"] is True
    assert result["action"] == "draft_from_literature"
    assert result["can_enter_claim_gate"] is False
    store = load_store(path)
    rows = [r for r in store["candidates"] if r["status"] == STATUS_PROPOSED]
    assert len(rows) >= 1
    assert rows[0]["source"] == "draft_arm"
    assert rows[0]["draft_origin"] == "literature"
    assert rows[0]["requires_human_review"] is False
    assert rows[0]["paper_refs"]


def test_draft_arm_invent_awaits_human_without_papers(tmp_path: Path) -> None:
    path = pending_path(tmp_path / "campaign")
    result = run_how_draft_arm(
        path,
        confirm_human_gate=True,
        provider=None,
        round_id="r_invent",
    )
    assert result["action"] == "invent_awaiting_human"
    assert result["requires_human_review"] is True
    store = load_store(path)
    row = store["candidates"][0]
    assert row["source"] == "invent_fallback"
    assert row["requires_human_review"] is True
    assert row["paper_refs"] == []

    # Lifecycle must not auto-author invent drafts.
    tick = run_how_lifecycle_tick(
        path,
        project_root=ROOT,
        confirm_human_gate=True,
        live=False,
        sandbox_root=tmp_path / "sandboxes",
    )
    assert tick["action"] == "idle"
    assert load_store(path)["candidates"][0]["status"] == STATUS_PROPOSED

    # Human release unlocks lifecycle eligibility.
    release_invent_for_lifecycle(
        path,
        row["candidate_id"],
        confirm_human_gate=True,
        note="approve invent for author",
    )
    assert load_store(path)["candidates"][0]["requires_human_review"] is False


def test_draft_arm_model_cannot_draft_falls_to_invent(tmp_path: Path) -> None:
    packet = _packet(tmp_path)
    path = pending_path(tmp_path / "campaign")
    persist_scout(path, packet)
    calls = {"n": 0}

    def _script(_request):
        calls["n"] += 1
        if calls["n"] == 1:
            return (
                '{"action":"cannot_draft","reason":"papers not about fusion",'
                '"how_candidates":[]}'
            )
        return (
            '{"how_id":"P1","family":"fusion","mechanism":"gated residual RGB-T blend",'
            '"implementation_intent":"small gate on concat features","reason":"fallback"}'
        )

    result = run_how_draft_arm(
        path,
        confirm_human_gate=True,
        provider=ScriptedProvider(_script),
        round_id="r_cannot",
    )
    assert result["action"] == "invent_awaiting_human"
    store = load_store(path)
    assert store["candidates"][0]["source"] == "invent_fallback"


def test_invent_ingest_rejects_catalog_id(tmp_path: Path) -> None:
    path = pending_path(tmp_path / "campaign")
    with pytest.raises(HowPendingError, match="non-catalog"):
        ingest_invent_fallback_candidate(
            path,
            {
                "how_id": "F1",
                "family": "fusion",
                "mechanism": "should fail",
                "implementation_intent": "no",
            },
            round_id="r_bad",
        )


def test_human_register_on_invent_clears_gate_for_author(tmp_path: Path) -> None:
    from scientist_lab.core.how_draft_arm import ingest_invent_fallback_candidate
    from scientist_lab.core.how_pending import decide_candidate, load_store, pending_path

    path = pending_path(tmp_path / "campaign")
    store = ingest_invent_fallback_candidate(
        path,
        {
            "how_id": "P1",
            "family": "fusion",
            "mechanism": "gated residual RGB-T blend for lowlight",
            "implementation_intent": "small gate on concat features",
        },
        round_id="r_human",
    )
    cid = store["candidates"][0]["candidate_id"]
    assert store["candidates"][0]["requires_human_review"] is True
    decide_candidate(
        path,
        cid,
        decision="register",
        confirm_human_gate=True,
        note="approve invent",
    )
    row = load_store(path)["candidates"][0]
    assert row["requires_human_review"] is False
    assert row["status"] == "proposed"
    assert row["human_decision"] == "release_for_author"


def test_draft_arm_requires_human_gate(tmp_path: Path) -> None:
    path = pending_path(tmp_path / "campaign")
    with pytest.raises(HowPendingError, match="Human Gate"):
        run_how_draft_arm(path, confirm_human_gate=False)


def test_next_plugin_how_id_skips_disk_plugins(tmp_path: Path) -> None:
    from scientist_lab.core.how_draft_arm import (
        _next_plugin_how_id,
        occupied_plugin_how_ids,
    )

    plugins = (
        tmp_path
        / "experiment_apps"
        / "rgbt_detection_real"
        / "models"
        / "how_plugins"
    )
    for hid in ("P1", "P2", "P3"):
        dest = plugins / hid
        dest.mkdir(parents=True)
        (dest / "plugin.py").write_text("# stub\n", encoding="utf-8")
    store = {"candidates": [], "registered_overlay": {}}
    used = occupied_plugin_how_ids(store, project_root=tmp_path)
    assert {"P1", "P2", "P3"} <= used
    assert _next_plugin_how_id(store, project_root=tmp_path) == "P4"


def test_invent_remaps_colliding_disk_how_id(tmp_path: Path) -> None:
    plugins = (
        tmp_path
        / "experiment_apps"
        / "rgbt_detection_real"
        / "models"
        / "how_plugins"
    )
    for hid in ("P1", "P2"):
        dest = plugins / hid
        dest.mkdir(parents=True)
        (dest / "plugin.py").write_text("# stub\n", encoding="utf-8")
    path = pending_path(tmp_path / "campaign")

    def _script(_request):
        return (
            '{"how_id":"P2","family":"neck","mechanism":"reliability gate in fusion",'
            '"implementation_intent":"true HxW reliability mask from luminance and '
            'thermal contrast","reason":"collide on purpose"}'
        )

    result = run_how_draft_arm(
        path,
        confirm_human_gate=True,
        provider=ScriptedProvider(_script),
        round_id="r_collide",
        project_root=tmp_path,
    )
    assert result["action"] == "invent_awaiting_human"
    assert result["how_id"] == "P3"
    row = load_store(path)["candidates"][0]
    assert row["how_id"] == "P3"
    assert row["family"] == "fusion"
    assert row["needs_adapter_work"] is False
