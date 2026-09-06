"""Detector-slot HOW plugins (neck / backbone_wrap). No GPU."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from scientist_lab.adapters.base import MaterializeRejected
from scientist_lab.adapters.dfine.adapter import DFINEAdapter
from scientist_lab.adapters.dfine.how import resolve_adapter_how
from scientist_lab.adapters.dfine.how_catalog import (
    plugin_overlay_spec,
    protocol_invent_kinds,
    resolve_how_id,
)
from scientist_lab.core.how_pending import (
    STATUS_REGISTERED,
    decide_candidate,
    ingest_llm_candidates,
    literature_paper_ids,
    literature_query_id,
    load_store,
    pending_path,
)
from scientist_lab.core.how_plugin_author import (
    HowPluginAuthorError,
    author_how_patch,
    example_plugin_source,
    plugin_unified_diff_from_source,
)
from scientist_lab.core.schema_registry import SCHEMA_DIR, load_json

from tests.unit.test_how_candidates_freeze import _packet

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "experiment_apps" / "rgbt_detection_real"
sys.path.insert(0, str(APP))

from models.how_plugin_loader import (  # noqa: E402
    detect_plugin_kind,
    smoke_plugin_file,
)
from models.neck_factory import NeckConfig, NeckFactory, parse_neck_config  # noqa: E402


def _neck_draft(qid: str, papers: list[str], **overrides) -> dict:
    row = {
        "how_id": "N7",
        "family": "neck",
        "plugin_kind": "neck",
        "mechanism": "Replace HybridEncoder with a lighter lateral-add pyramid at the encoder slot.",
        "literature_query_id": qid,
        "paper_refs": papers,
        "invented_operators": [],
        "neck_type": "plugin:N7",
        "fusion_method": "none",
        "implementation_intent": "top-down lateral add keeping P3 P4 P5 hidden_dim",
        "needs_adapter_work": True,
    }
    row.update(overrides)
    return row


def test_example_neck_and_backbone_wrap_smoke() -> None:
    neck = APP / "models" / "how_plugins" / "_example_neck" / "plugin.py"
    wrap = APP / "models" / "how_plugins" / "_example_backbone_wrap" / "plugin.py"
    assert detect_plugin_kind(neck) == "neck"
    assert detect_plugin_kind(wrap) == "backbone_wrap"
    neck_report = smoke_plugin_file(neck)
    wrap_report = smoke_plugin_file(wrap)
    assert neck_report["ok"] is True
    assert neck_report["plugin_kind"] == "neck"
    assert neck_report["gpu"] is False
    assert wrap_report["ok"] is True
    assert wrap_report["plugin_kind"] == "backbone_wrap"
    assert wrap_report["can_enter_claim_gate"] is False


def test_neck_factory_plugin_path_matches_hybrid_io() -> None:
    example = APP / "models" / "how_plugins" / "_example_neck" / "plugin.py"
    cfg = parse_neck_config({"neck": {"type": "plugin:N7", "hidden_dim": 16}})
    assert cfg.type == "plugin:n7"
    neck = NeckFactory.create(
        NeckConfig(
            type="plugin:n7",
            in_channels=(32, 64, 128),
            out_channels=16,
            hidden_dim=16,
            feat_strides=(8, 16, 32),
        ),
        plugin_path=example,
    )
    feats = [
        torch.randn(1, 32, 16, 16),
        torch.randn(1, 64, 8, 8),
        torch.randn(1, 128, 4, 4),
    ]
    outs = neck(feats)
    assert len(outs) == 3
    assert outs[0].shape == (1, 16, 16, 16)
    assert outs[2].shape == (1, 16, 4, 4)
    with pytest.raises(ValueError, match="Unknown neck"):
        parse_neck_config({"neck": {"type": "panet"}})


def test_overlay_spec_kinds_and_protocol_default() -> None:
    neck = plugin_overlay_spec("N7", smoke_ok=True, plugin_kind="neck")
    wrap = plugin_overlay_spec("B7", smoke_ok=True, plugin_kind="backbone_wrap")
    fusion = plugin_overlay_spec("F2", smoke_ok=True)
    assert neck["neck_type"] == "plugin:N7"
    assert neck["fusion_method"] == "none"
    assert neck["requires_new_baseline"] is False
    assert wrap["backbone_wrap_method"] == "plugin:B7"
    assert wrap["requires_new_baseline"] is True
    assert fusion["fusion_method"] == "plugin:F2"
    assert protocol_invent_kinds(None) == frozenset({"fusion"})
    protocol = load_json(SCHEMA_DIR / "examples" / "research_protocol_rgbt_dfine_v26.json")
    assert "neck" in protocol_invent_kinds(protocol)
    assert "fusion" in protocol_invent_kinds(protocol)


def test_author_register_neck_plugin_overlay(tmp_path: Path) -> None:
    packet = _packet(tmp_path)
    qid = literature_query_id(packet)
    papers = sorted(literature_paper_ids(packet))
    path = pending_path(tmp_path / "campaign")
    store = ingest_llm_candidates(
        path,
        [_neck_draft(qid, papers[:1])],
        literature=packet,
        round_id="r_neck",
    )
    cid = store["candidates"][0]["candidate_id"]
    source = example_plugin_source(ROOT, plugin_kind="neck")
    promoted = ROOT / "experiment_apps" / "rgbt_detection_real" / "models" / "how_plugins" / "N7" / "plugin.py"
    try:
        authored = author_how_patch(
            path,
            cid,
            project_root=ROOT,
            unified_diff=plugin_unified_diff_from_source("N7", source),
            sandbox_root=tmp_path / "sandboxes",
        )
        assert authored["ok"] is True
        assert authored["gpu"] is False
        saved = load_store(path)
        assert saved["candidates"][0]["plugin_kind"] == "neck"
        assert saved["candidates"][0]["neck_type"] == "plugin:N7"
        decided = decide_candidate(
            path,
            cid,
            decision="register",
            confirm_human_gate=True,
            note="neck plugin smoked; overlay only",
        )
        assert decided["candidates"][0]["status"] == STATUS_REGISTERED
        spec = resolve_how_id("N7", overlay=decided.get("registered_overlay"))
        assert spec["plugin_kind"] == "neck"
        assert spec["neck_type"] == "plugin:N7"
        plan = dict(load_json(SCHEMA_DIR / "examples" / "experiment_plan_rgbt_dfine_v26_r0.json"))
        plan["how_id"] = "N7"
        plan["how_overlay"] = decided.get("registered_overlay")
        plan["modification_scope"] = ["neck"]
        plan["proposed_changes"] = [
            {
                "target": "neck",
                "summary": "N7 plugin neck at HybridEncoder slot",
                "detail": {"how_id": "N7", "neck_type": "plugin:N7"},
            }
        ]
        protocol = load_json(SCHEMA_DIR / "examples" / "research_protocol_rgbt_dfine_v26.json")
        how = resolve_adapter_how(plan, protocol)
        assert how["neck_type"] == "plugin:N7"
        assert how["fusion_method"] == "none"
        contract = DFINEAdapter().materialize_contract(plan, protocol)
        assert contract["materialization"]["legacy_parameters"]["neck"]["type"] == "plugin:N7"
    finally:
        if promoted.is_file():
            promoted.unlink()
        if promoted.parent.is_dir() and not any(promoted.parent.iterdir()):
            promoted.parent.rmdir()


def test_backbone_wrap_requires_new_architecture_id(tmp_path: Path) -> None:
    overlay = {"B7": plugin_overlay_spec("B7", smoke_ok=True, plugin_kind="backbone_wrap")}
    plan = dict(load_json(SCHEMA_DIR / "examples" / "experiment_plan_rgbt_dfine_v26_r0.json"))
    plan["how_id"] = "B7"
    plan["how_overlay"] = overlay
    plan["modification_scope"] = ["backbone"]
    plan["proposed_changes"] = [
        {
            "target": "backbone",
            "summary": "B7 backbone wrap plugin",
            "detail": {"how_id": "B7"},
        }
    ]
    protocol = dict(load_json(SCHEMA_DIR / "examples" / "research_protocol_rgbt_dfine_v26.json"))
    with pytest.raises(MaterializeRejected, match="architecture_id"):
        DFINEAdapter().materialize_contract(plan, protocol)
    protocol["baseline"] = dict(protocol["baseline"])
    protocol["baseline"]["architecture_id"] = "B7"
    contract = DFINEAdapter().materialize_contract(plan, protocol)
    how = contract["materialization"]["how"]
    assert how["requires_new_baseline"] is True
    assert how["backbone_wrap_method"] == "plugin:B7"


def test_invent_policy_blocks_neck_when_fusion_only(tmp_path: Path) -> None:
    packet = _packet(tmp_path)
    qid = literature_query_id(packet)
    papers = sorted(literature_paper_ids(packet))
    path = pending_path(tmp_path / "campaign")
    store = ingest_llm_candidates(
        path,
        [_neck_draft(qid, papers[:1])],
        literature=packet,
    )
    cid = store["candidates"][0]["candidate_id"]
    source = example_plugin_source(ROOT, plugin_kind="neck")
    with pytest.raises(HowPluginAuthorError, match="invent_policy"):
        author_how_patch(
            path,
            cid,
            project_root=ROOT,
            unified_diff=plugin_unified_diff_from_source("N7", source),
            sandbox_root=tmp_path / "sandboxes",
            protocol={"invent_policy": {"kinds": ["fusion"], "one_slot_per_plugin": True}},
        )


def test_backbone_wrap_example_forward() -> None:
    from models.how_plugin_loader import load_plugin_backbone_wrap

    class _Stub(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.out_channels = [8, 16, 32]
            self.stems = nn.ModuleList(nn.Conv2d(3, c, 1) for c in self.out_channels)

        def forward(self, x):
            cur = x[:, :3]
            outs = []
            for stem in self.stems:
                cur = torch.nn.functional.avg_pool2d(cur, 2)
                outs.append(stem(cur))
            return outs

    path = APP / "models" / "how_plugins" / "_example_backbone_wrap" / "plugin.py"
    wrap = load_plugin_backbone_wrap(path, _Stub())
    out = wrap(torch.zeros(1, 3, 32, 32))
    assert len(out) == 3
    assert out[0].shape[1] == 8
