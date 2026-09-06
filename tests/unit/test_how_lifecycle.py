"""LLM HOW lifecycle: add / author / accept overlay. No GPU."""

from __future__ import annotations

from pathlib import Path

import pytest

from scientist_lab.adapters.base import MaterializeRejected
from scientist_lab.adapters.dfine.how import resolve_adapter_how
from scientist_lab.adapters.dfine.how_catalog import planner_visible_how_ids, resolve_how_id
from scientist_lab.core.how_lifecycle import run_how_lifecycle_tick
from scientist_lab.core.how_pending import (
    STATUS_REGISTERED,
    ingest_llm_candidates,
    literature_paper_ids,
    literature_query_id,
    load_store,
    overlay_from_store,
    pending_path,
)
from scientist_lab.core.schema_registry import SCHEMA_DIR, load_json
from scientist_lab.llm.gateway import ScriptedProvider

from tests.unit.test_how_candidates_freeze import _f2_draft, _packet
from tests.unit.test_how_plugin_bridge import ROOT

EXAMPLES = SCHEMA_DIR / "examples"


def test_lifecycle_authors_and_llm_accepts_overlay(tmp_path: Path) -> None:
    packet = _packet(tmp_path)
    qid = literature_query_id(packet)
    papers = sorted(literature_paper_ids(packet))
    path = pending_path(tmp_path / "campaign")
    store = ingest_llm_candidates(
        path,
        [
            _f2_draft(
                qid,
                papers[:1],
                implementation_intent="weighted average of RGB and thermal features",
            )
        ],
        literature=packet,
        round_id="r_lifecycle",
    )
    cid = store["candidates"][0]["candidate_id"]
    with pytest.raises(MaterializeRejected):
        resolve_how_id("F2")
    tick = run_how_lifecycle_tick(
        path,
        project_root=ROOT,
        confirm_human_gate=True,
        live=False,
        sandbox_root=tmp_path / "sandboxes",
    )
    assert tick["progressed"] is True
    assert tick["registered"] is True
    assert tick["gpu"] is False
    assert tick["can_enter_claim_gate"] is False
    saved = load_store(path)
    assert saved["candidates"][0]["status"] == STATUS_REGISTERED
    assert saved["candidates"][0]["decided_by"] == "llm"
    overlay = overlay_from_store(saved)
    spec = resolve_how_id("F2", overlay=overlay)
    assert spec["fusion_method"] == "plugin:F2"
    assert "F2" in planner_visible_how_ids(overlay)
    plan = dict(load_json(EXAMPLES / "experiment_plan_rgbt_dfine_v26_r0.json"))
    plan["how_id"] = "F2"
    plan["how_overlay"] = overlay
    protocol = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v26.json")
    how = resolve_adapter_how(plan, protocol)
    assert how["fusion_method"] == "plugin:F2"
    assert cid


def test_lifecycle_override_authors_when_unsmoked_reject(tmp_path: Path) -> None:
    """Models often reject unsmoked drafts; Stage A must still author them."""
    packet = _packet(tmp_path)
    qid = literature_query_id(packet)
    papers = sorted(literature_paper_ids(packet))
    path = pending_path(tmp_path / "campaign")
    ingest_llm_candidates(
        path,
        [_f2_draft(qid, papers[:1], implementation_intent="weighted RGB-T mix")],
        literature=packet,
    )
    provider = ScriptedProvider(
        '{"action":"reject","reason":"Smoke test failed (smoke_ok=false)."}'
    )
    tick = run_how_lifecycle_tick(
        path,
        project_root=ROOT,
        confirm_human_gate=True,
        live=True,
        provider=provider,
        sandbox_root=tmp_path / "sandboxes",
    )
    assert tick["action"] in {"register", "author_failed", "reject_accept"}
    assert "override" in str(tick.get("reason") or tick.get("draft_arm") or "").lower() or tick.get(
        "registered"
    ) or tick["action"] != "reject_add"
    assert tick["action"] != "reject_add"
    row = load_store(path)["candidates"][0]
    assert row["status"] != "rejected"


def test_lifecycle_override_registers_when_accept_rejects_smoke_ok(tmp_path: Path) -> None:
    """Models reject overlay citing can_enter_claim_gate=false; must still register."""
    packet = _packet(tmp_path)
    qid = literature_query_id(packet)
    papers = sorted(literature_paper_ids(packet))
    path = pending_path(tmp_path / "campaign")
    store = ingest_llm_candidates(
        path,
        [_f2_draft(qid, papers[:1], implementation_intent="weighted RGB-T mix")],
        literature=packet,
    )
    store["candidates"][0]["smoke_ok"] = True
    store["candidates"][0]["plugin_relpath"] = (
        "experiment_apps/rgbt_detection_real/models/how_plugins/F2/plugin.py"
    )
    store["candidates"][0]["can_enter_claim_gate"] = False
    from scientist_lab.core.how_pending import save_store

    save_store(path, store)
    calls = {"n": 0}

    def _script(_request: object) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            return '{"action":"author","reason":"already smoked"}'
        return (
            '{"decision":"reject","reason":"can_enter_claim_gate is false; '
            'smoked HOW plugin cannot enter registered_overlay without claim gate eligibility."}'
        )

    provider = ScriptedProvider(_script)
    tick = run_how_lifecycle_tick(
        path,
        project_root=ROOT,
        confirm_human_gate=True,
        live=True,
        provider=provider,
        sandbox_root=tmp_path / "sandboxes",
    )
    assert tick["registered"] is True
    assert tick["action"] == "register"
    assert "override" in str(tick.get("reason") or "").lower()
    assert load_store(path)["candidates"][0]["status"] == STATUS_REGISTERED
    assert "F2" in overlay_from_store(load_store(path))


def test_lifecycle_llm_can_reject_add_after_smoke(tmp_path: Path) -> None:
    packet = _packet(tmp_path)
    qid = literature_query_id(packet)
    papers = sorted(literature_paper_ids(packet))
    path = pending_path(tmp_path / "campaign")
    store = ingest_llm_candidates(
        path,
        [_f2_draft(qid, papers[:1], implementation_intent="weighted RGB-T mix")],
        literature=packet,
    )
    store["candidates"][0]["smoke_ok"] = True
    store["candidates"][0]["plugin_relpath"] = (
        "experiment_apps/rgbt_detection_real/models/how_plugins/F2/plugin.py"
    )
    from scientist_lab.core.how_pending import save_store

    save_store(path, store)
    provider = ScriptedProvider('{"action":"reject","reason":"prefer catalog F3"}')
    tick = run_how_lifecycle_tick(
        path,
        project_root=ROOT,
        confirm_human_gate=True,
        live=True,
        provider=provider,
        sandbox_root=tmp_path / "sandboxes",
    )
    assert tick["action"] == "reject_add"
    assert tick["registered"] is False
    assert load_store(path)["candidates"][0]["status"] == "rejected"
    with pytest.raises(MaterializeRejected):
        resolve_how_id("F2")


def test_lifecycle_requires_human_gate(tmp_path: Path) -> None:
    from scientist_lab.core.how_pending import HowPendingError

    with pytest.raises(HowPendingError, match="Human Gate"):
        run_how_lifecycle_tick(
            tmp_path / "how_pending.json",
            project_root=ROOT,
            confirm_human_gate=False,
        )


def test_v26_protocol_max_rounds_is_12() -> None:
    doc = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v26.json")
    assert doc["protocol_version"] == 2
    assert doc["stop_rules"]["max_rounds"] == 12
