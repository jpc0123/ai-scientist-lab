"""Evaluation matrix: ablation, multi-seed, cross-model requirements."""

from __future__ import annotations

import pytest

from scientist_lab.services.evaluation_matrix import (
    build_evaluation_matrix,
    detect_campaign_mode,
    sota_pursuit_allowed,
)


def test_detect_seed_stability_mode() -> None:
    protocol = {"objective": {"primary": {"metric": "APS"}}, "title": "How stable is APS across seeds?"}
    campaign = {"experiment_title": "How stable is D-FINE APS across seeds?"}
    rounds = [
        {"how_id": "F0", "seed": 42, "primary_value": 0.021},
        {"how_id": "F0", "seed": 43, "primary_value": 0.022},
    ]
    assert detect_campaign_mode(protocol, campaign, rounds=rounds) == "seed_stability"
    allowed, reason = sota_pursuit_allowed(protocol, campaign, rounds=rounds)
    assert allowed is False
    assert "稳定性" in reason


def test_v26_improvement_requires_ablations() -> None:
    protocol = {
        "title": "Low-light RGB-T",
        "baseline": {"adapter": "dfine"},
        "objective": {"primary": {"metric": "APS_lowlight"}},
        "condition_slice": {"id": "low_light_subset_v1"},
    }
    campaign = {"adapter": "dfine", "experiment_id": "exp_rgbt_dfine_v26_lowlight"}
    rounds = [{"how_id": "F1", "seed": 42, "primary_value": 0.005}]
    matrix = build_evaluation_matrix(campaign, protocol=protocol, rounds=rounds)
    assert matrix["mode"] == "v26_improvement"
    assert matrix["demo_risk"] is True
    assert any("fusion 消融" in g for g in matrix["gaps"])
    assert any("RT-DETR" in g for g in matrix["gaps"])


def test_sota_blocked_on_seed_stability_via_tick(tmp_path) -> None:
    import json
    from pathlib import Path

    from scientist_lab.services.sota_pursuit import run_sota_pursuit_tick

    work = tmp_path / "camp"
    work.mkdir()
    steps = [
        {"action": "NEED_PLAN", "report": {"plan": {"how_id": "F0"}}},
        {"action": "NEED_PARSE", "report": {"result": {"metrics": {"APS": 0.021}}}},
        {"action": "NEED_PLAN", "report": {"plan": {"how_id": "F0"}}},
        {"action": "NEED_PARSE", "report": {"result": {"metrics": {"APS": 0.022}}}},
    ]
    (work / "campaign.json").write_text(
        json.dumps(
            {
                "gpu_rounds": 2,
                "experiment_title": "How stable is APS across seeds?",
                "steps": steps,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (work / "protocol.json").write_text(
        json.dumps(
            {
                "title": "How stable is APS across seeds?",
                "objective": {"primary": {"metric": "APS"}},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    pursuit = run_sota_pursuit_tick(work, {"confirm_human_gate": True})
    assert pursuit["action"] == "wrong_campaign_mode"
    assert pursuit["progressed"] is False
