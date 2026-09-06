"""Campaign lab log + RGB-T pair preview helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from scientist_lab.datasets.rgbt_pair_preview import compose_pair_png, list_pair_stems, preview_manifest
from scientist_lab.services.campaign_notebook import (
    build_lab_log,
    build_campaign_notebook,
    build_campaign_story,
    build_experiment_truth,
    build_llm_judgment,
)


def test_lab_log_explains_unstable_f1_and_rejected_hows() -> None:
    campaign = {
        "planner_backend": "llm",
        "steps": [
            {
                "action": "NEED_PLAN",
                "report": {
                    "plan": {
                        "how_id": "F1",
                        "bootstrap": True,
                        "hypothesis": "F1 early_concat is the frozen R0 comparator.",
                    }
                },
            },
            {"action": "NEED_EXECUTION", "report": {"handle": {"run_id": "run_r0"}}},
            {
                "action": "NEED_PARSE",
                "report": {"result": {"metrics": {"APS_lowlight": 0.0405}}},
            },
            {
                "action": "NEED_REVIEW",
                "report": {
                    "review": {
                        "review_decision": "REPLICATE",
                        "objective_check": {"APS_lowlight": {"current": 0.0405, "delta": None}},
                    },
                    "claim_gate": {"status": "C0", "scientific_outcome": "INCONCLUSIVE"},
                },
            },
            {
                "action": "NEXT_ROUND",
                "idle": False,
                "report": {
                    "plan": {
                        "how_id": "F1",
                        "bootstrap": False,
                        "hypothesis": "Replicate F1 on another seed before trying F3.",
                        "candidate_experiments": [
                            {
                                "how_id": "F3",
                                "reason_not_selected": "baseline not stable across seeds",
                            }
                        ],
                    }
                },
            },
            {"action": "NEED_EXECUTION", "report": {"handle": {"run_id": "run_r1"}}},
            {
                "action": "NEED_PARSE",
                "report": {"result": {"metrics": {"APS_lowlight": 0.0136}}},
            },
            {
                "action": "NEED_REVIEW",
                "report": {
                    "review": {
                        "review_decision": "REPLICATE",
                        "objective_check": {"APS_lowlight": {"current": 0.0136, "delta": None}},
                    }
                },
            },
            {
                "action": "NEXT_ROUND",
                "idle": True,
                "reasons": ["max extra rounds reached; not starting another round"],
            },
        ],
    }
    log = build_lab_log(campaign)
    agents = [row["agent"] for row in log]
    assert "planner" in agents
    assert "reviewer" in agents
    assert "executor" in agents
    bodies = "\n".join(row["body"] for row in log)
    assert "0.0405" in bodies or "0.0405"[:6] in bodies
    assert "没有稳住" in bodies
    assert "F3" in bodies
    assert "停轮" in "\n".join(row["title"] for row in log)


def test_round_cards_use_protocol_primary_and_seed() -> None:
    campaign = {
        "status": "paused",
        "gpu_rounds": 2,
        "last_action": "NEED_MEMORY",
        "orphan_reclaimed": True,
        "steps": [
            {
                "action": "NEED_PLAN",
                "report": {
                    "plan": {
                        "how_id": "F0",
                        "bootstrap": True,
                        "hypothesis": "F0 seed-stability R0",
                        "evaluation": {"seeds": [42]},
                    }
                },
            },
            {"action": "NEED_EXECUTION", "report": {"handle": {"run_id": "run_r0", "status": "completed"}}},
            {"action": "NEED_PARSE", "report": {"result": {"metrics": {"APS": 0.02097, "APS_lowlight": 0.001}}}},
            {
                "action": "NEED_REVIEW",
                "report": {
                    "review": {
                        "review_decision": "REPLICATE",
                        "objective_check": {"APS": {"current": 0.02097, "delta": None}},
                    }
                },
            },
            {
                "action": "NEED_PLAN",
                "report": {
                    "plan": {
                        "how_id": "F0",
                        "evaluation": {"seeds": [43]},
                        "hypothesis": "replicate F0 seed 43",
                    }
                },
            },
            {"action": "NEED_EXECUTION", "report": {"handle": {"run_id": "run_r1", "status": "completed"}}},
            {"action": "NEED_PARSE", "report": {"result": {"metrics": {"APS": 0.02185}}}},
            {
                "action": "NEED_REVIEW",
                "report": {
                    "review": {
                        "review_decision": "KEEP",
                        "objective_check": {"APS": {"current": 0.02185, "delta": 0.00088}},
                    }
                },
            },
        ],
    }
    protocol = {"title": "How stable is D-FINE APS across seeds?", "objective": {"primary": {"metric": "APS"}}, "baseline": {"model": "DFINE-S", "dataset": "rgbt_tiny_v1"}}
    story = build_campaign_story(campaign, protocol=protocol)
    assert story["primary_metric"] == "APS"
    assert story["question"] == "How stable is D-FINE APS across seeds?"
    assert "2 轮" in story["did"]
    assert "F0" in story["did"]
    assert "只用 RGB" in story["did"]
    assert len(story["rounds"]) == 2
    assert story["rounds"][0]["seed"] == 42
    assert story["rounds"][1]["seed"] == 43
    assert story["rounds"][0]["value"] == pytest.approx(0.02097)
    assert "只用 RGB" in story["rounds"][0]["how_label"]
    assert "换 seed 再复现" in story["rounds"][0]["decision_label"]
    assert "先留下" in story["rounds"][1]["decision_label"]
    assert "APS=" in story["headline"]
    assert "已跑 2 轮" in story["headline"]
    now = story["now"]
    assert now["round_index"] == 2
    assert now["round_kind"] == "completed"
    assert "第 2 轮" in now["round_label"]
    assert now["next_kind"] == "paused"
    assert "不会自动" in now["next_text"]
    assert now.get("can_resume") is True
    assert now["current_how_id"] == "F0"
    assert now["how_new"] == []
    assert "没有新 HOW" in now["how_new_reason"]
    assert story["rounds"][-1]["is_current"] is True
    assert story["rounds"][-1]["badge"] == "最近"
    bodies = "\n".join(row["body"] for row in story["lab_log"])
    assert "APS=" in bodies
    assert "APS_lowlight=0.001" not in bodies


def test_now_board_explains_missing_how_from_scout_429() -> None:
    campaign = {
        "status": "paused",
        "gpu_rounds": 2,
        "orphan_reclaimed": True,
        "error": "worker lost after process restart; GPU lock released. Not auto-resumed.",
        "max_extra_rounds": 11,
        "steps": [
            {"action": "NEED_PLAN", "report": {"plan": {"how_id": "F0", "evaluation": {"seeds": [42]}}}},
            {"action": "NEED_EXECUTION", "report": {"handle": {"status": "completed"}}},
            {"action": "NEED_PARSE", "report": {"result": {"metrics": {"APS": 0.02}}}},
            {
                "action": "NEED_REVIEW",
                "report": {"review": {"review_decision": "KEEP", "objective_check": {"APS": {"current": 0.02}}}},
            },
        ],
    }
    pending = {
        "candidates": [],
        "scout": {"fail_closed": True, "error": "rate limited (HTTP 429): Too Many Requests"},
    }
    now = build_campaign_story(campaign, protocol={"objective": {"primary": {"metric": "APS"}}}, how_pending=pending)["now"]
    assert now["how_new"] == []
    assert "429" in now["how_new_reason"]
    assert now["how_in_use"][0]["how_id"] == "F0"


def test_can_extend_when_max_extra_rounds_stops_before_protocol_max() -> None:
    """UI must show「提高额度并续跑」even if gpu < protocol.max_rounds."""
    from scientist_lab.services.campaign_notebook import _round_budget_exhausted, build_campaign_story

    campaign = {
        "status": "completed",
        "gpu_rounds": 18,
        "max_extra_rounds": 17,
        "last_action": "NEXT_ROUND",
        "worker_alive": False,
        "steps": [
            {
                "action": "NEXT_ROUND",
                "idle": True,
                "reasons": ["max_extra_rounds reached; not starting another round"],
                "report": {"extra_rounds": 17},
            }
        ],
    }
    protocol = {"stop_rules": {"max_rounds": 21}, "objective": {"primary": {"metric": "APS_lowlight"}}}
    assert _round_budget_exhausted(campaign, protocol) is True
    now = build_campaign_story(campaign, protocol=protocol)["now"]
    assert now["can_extend"] is True


def test_now_board_surfaces_new_how_candidate() -> None:
    campaign = {"status": "running", "gpu_rounds": 1, "last_action": "NEED_MEMORY", "steps": []}
    pending = {
        "candidates": [
            {
                "candidate_id": "how_f3_draft",
                "how_id": "F3",
                "status": "proposed",
                "mechanism": "gated multiscale fusion",
            }
        ]
    }
    now = build_campaign_story(campaign, how_pending=pending)["now"]
    assert now["need_human"] is True
    assert now["how_new"][0]["how_id"] == "F3"
    assert "F3" in now["next_text"]


def test_now_board_prefers_live_plan_over_stale_scoreboard() -> None:
    """Waiting GPU after NEXT_ROUND: show plan HOW, not last reviewed F1/F3."""
    campaign = {
        "status": "waiting_gpu",
        "gpu_rounds": 2,
        "last_action": "NEED_GATE",
        "steps": [
            {"action": "NEED_PLAN", "report": {"plan": {"how_id": "F1", "evaluation": {"seeds": [42]}}}},
            {"action": "NEED_EXECUTION", "report": {"handle": {"status": "completed"}}},
            {"action": "NEED_PARSE", "report": {"result": {"metrics": {"APS_lowlight": 0.01}}}},
            {
                "action": "NEED_REVIEW",
                "report": {
                    "review": {
                        "review_decision": "KEEP",
                        "objective_check": {"APS_lowlight": {"current": 0.01}},
                    }
                },
            },
            {"action": "NEXT_ROUND", "report": {"plan": {"how_id": "P3A", "evaluation": {"seeds": [47]}}}},
            {"action": "NEED_GATE", "report": {}},
        ],
    }
    plan = {"how_id": "P3A", "evaluation": {"seeds": [47]}}
    now = build_campaign_story(campaign, seed_plan=plan)["now"]
    assert now["current_how_id"] == "P3A"
    assert "P3A" in now["next_text"]
    assert now["next_kind"] == "training"


def test_now_board_prefers_live_plan_when_paused_mid_execution() -> None:
    campaign = {
        "status": "paused",
        "gpu_rounds": 12,
        "last_action": "NEED_EXECUTION",
        "stop_requested": True,
        "steps": [
            {"action": "NEED_PLAN", "report": {"plan": {"how_id": "F3", "evaluation": {"seeds": [46]}}}},
            {"action": "NEED_EXECUTION", "report": {"handle": {"status": "completed"}}},
            {
                "action": "NEED_REVIEW",
                "report": {
                    "review": {
                        "review_decision": "KEEP",
                        "objective_check": {"APS_lowlight": {"current": 0.02}},
                    }
                },
            },
        ],
    }
    plan = {"how_id": "P3A", "evaluation": {"seeds": [47]}}
    now = build_campaign_story(campaign, seed_plan=plan)["now"]
    assert now["current_how_id"] == "P3A"
    assert "P3A" in (now.get("current_how_label") or "")


def test_experiment_truth_is_honest_about_catalog_only_runs() -> None:
    campaign = {
        "experiment_id": "exp_rgbt_dfine_v26_lowlight",
        "experiment_title": "V26 lowlight",
        "planner_backend": "rules",
        "reviewer_backend": "rules",
        "catalog_id": "how_catalog_v26_p2",
        "adapter": "dfine",
        "dataset_id": "rgbt_tiny_v1",
        "slice_id": "low_light_subset_v1",
        "gpu_rounds": 2,
        "steps": [
            {
                "action": "NEED_PLAN",
                "report": {"plan": {"how_id": "F1", "bootstrap": True, "evaluation": {"seeds": [42]}}},
            },
            {"action": "NEED_EXECUTION", "report": {"handle": {"status": "completed"}}},
            {
                "action": "NEXT_ROUND",
                "report": {
                    "plan": {
                        "how_id": "F3",
                        "decision_summary": {"selected_action": "contrast_F3"},
                        "evaluation": {"seeds": [42]},
                    }
                },
            },
            {"action": "NEED_EXECUTION", "report": {"handle": {"status": "completed"}}},
        ],
    }
    seed = {"how_id": "F1", "bootstrap": True, "evaluation": {"seeds": [42]}}
    story = build_campaign_story(
        campaign,
        protocol={
            "title": "How stable is APS?",
            "objective": {"primary": {"metric": "APS"}},
            "baseline": {"dataset": "rgbt_tiny_v1", "model": "DFINE-S"},
        },
        seed_plan=seed,
    )
    truth = story["experiment_truth"]
    assert truth["capability"]["all_rounds_catalog_presets"] is True
    assert truth["capability"]["planner_backend"] == "rules"
    assert "contrast" in "\n".join(truth["truth_summary"]).lower() or "规则" in "\n".join(
        truth["truth_summary"]
    )
    assert story["rounds"][0]["how_origin"] == "protocol_seed"
    assert story["rounds"][1]["how_origin"] == "rules_contrast"
    assert len(truth["catalog_scope"]) == 6


def test_llm_judgment_surfaces_planner_reviewer_and_resume_hint() -> None:
    campaign = {
        "status": "paused",
        "gpu_rounds": 2,
        "last_action": "NEED_MEMORY",
        "planner_backend": "llm",
        "reviewer_backend": "llm",
        "steps": [
            {
                "action": "NEXT_ROUND",
                "report": {
                    "plan": {
                        "how_id": "F0",
                        "hypothesis": "Replicate F0 on seed 44.",
                        "candidate_experiments": [
                            {"how_id": "F3", "reason_not_selected": "baseline not stable"},
                        ],
                        "decision_summary": {"selected_action": "replicate_seed"},
                    }
                },
            },
            {
                "action": "NEED_REVIEW",
                "report": {
                    "review": {"review_decision": "KEEP", "reasoning_summary": "Δ small"},
                    "semantic_proposal": {
                        "observation": "APS stable across two seeds.",
                        "interpretation": "Still observational.",
                        "next_research_priority": "Add a third seed before fusion.",
                        "alternative_explanations": ["short training noise"],
                    },
                },
            },
            {"action": "NEED_MEMORY", "report": {"lessons_written": ["seed_sensitivity"]}},
        ],
    }
    pending = {"scout": {"fail_closed": True, "error": "rate limited (HTTP 429)"}}
    judgment = build_llm_judgment(campaign, how_pending=pending)
    assert judgment["can_resume"] is True
    assert judgment["planner"]["selected_how_id"] == "F0"
    assert len(judgment["planner"]["candidates"]) == 1
    assert "third seed" in judgment["reviewer"]["semantic_proposal"]["next_research_priority"]
    assert judgment["scout"]["status"] == "fail_closed"
    assert "429" in str(judgment["scout"]["error"])
    assert "NEXT_ROUND" in judgment["resume_hint"]["next_manager_action"]
    story = build_campaign_story(campaign, how_pending=pending)
    assert story["llm_judgment"]["planner"]["hypothesis"].startswith("Replicate")
    assert story["now"]["can_resume"] is True


def test_compose_pair_png_and_manifest(tmp_path: Path) -> None:
    rgb_dir = tmp_path / "images" / "train" / "rgb"
    thermal_dir = tmp_path / "images" / "train" / "thermal"
    rgb_dir.mkdir(parents=True)
    thermal_dir.mkdir(parents=True)
    Image.new("RGB", (32, 24), (200, 40, 40)).save(rgb_dir / "a.jpg")
    Image.new("L", (32, 24), 80).save(thermal_dir / "a.jpg")
    names = list_pair_stems(tmp_path, split="train", limit=3)
    assert names == ["a.jpg"]
    png = compose_pair_png(rgb_dir / "a.jpg", thermal_dir / "a.jpg")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"

    registry = tmp_path / "data" / "registry"
    registry.mkdir(parents=True)
    (registry / "rgbt_tiny_v1.json").write_text(
        json.dumps({"dataset_id": "rgbt_tiny_v1", "processed_root": str(tmp_path)}),
        encoding="utf-8",
    )
    manifest = preview_manifest(tmp_path, "rgbt_tiny_v1", split="train", limit=3)
    assert manifest["count"] == 1
    assert "thermal" in manifest["note"].lower() or "热红外" in manifest["note"]


def test_notebook_reads_pair_audit(tmp_path: Path, monkeypatch) -> None:
    work = tmp_path / "p0_demo"
    (work / "run").mkdir(parents=True)
    (work / "protocol.json").write_text(
        json.dumps({"baseline": {"dataset": "dataset:rgbt_tiny_v1"}}),
        encoding="utf-8",
    )
    (work / "plan.json").write_text(json.dumps({"how_id": "F1"}), encoding="utf-8")
    (work / "run" / "rgbt_pair_audit.json").write_text(
        json.dumps(
            {
                "splits": {
                    "train": {
                        "n_rgb": 10,
                        "n_thermal": 10,
                        "n_paired": 10,
                        "sample": {"rgb_mode": "RGB", "thermal_mode": "L"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (work / "run" / "dfine_subset.json").write_text(
        json.dumps({"staging_mode": "early_concat_blend", "fusion_method": "early_concat"}),
        encoding="utf-8",
    )
    note = build_campaign_notebook(work, {"steps": []}, project_root=tmp_path)
    assert note["dataset"]["n_rgb_train"] == 10
    assert note["dataset"]["thermal_mode"] == "L"
    assert "灰度" in note["dataset"]["note"]
