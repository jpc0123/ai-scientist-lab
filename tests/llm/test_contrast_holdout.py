"""Holdout contrast exam: labeling, grading, no-train isolation."""

from __future__ import annotations

from pathlib import Path

from scientist_lab.llm.contrast_holdout import (
    bucket_from_label,
    grade_action,
    grade_actions,
    grade_bucket_exam,
    holdout_case_ids,
    label_contrast,
    load_model_exam,
    training_leaks_holdout,
)


def _frozen() -> dict:
    return {
        "dataset_id": "rgbt_tiny_v1",
        "slice_id": "low_light_subset_v1",
        "budget_class": "formal",
    }


def test_bucket_exam_self_grades():
    report = grade_bucket_exam()
    assert report["ok"] is True
    assert report["passed"] == 6


def test_gold_actions_pass_model_exam():
    frozen = _frozen()
    gold = {
        "contrast_m01": {"how_id": "F3", "seeds": [42], **frozen, "stop": False},
        "contrast_m02": {"how_id": "F0", "seeds": [42], **frozen, "stop": False},
        "contrast_m03": {"how_id": "F3", "seeds": [43], **frozen, "stop": False},
        "contrast_m04": {"how_id": "F3", "seeds": [42], **frozen, "stop": False},
        "contrast_m05": {"how_id": "F3", "seeds": [42], **frozen, "stop": False},
        "contrast_m06": {"how_id": "F3", "seeds": [43], **frozen, "stop": False},
        "contrast_m07": {"how_id": "F1", "seeds": [42], **frozen, "stop": True},
        "contrast_m08": {"how_id": "F3", "seeds": [42], **frozen, "stop": False},
        "contrast_m09": {"how_id": "N0", "seeds": [42], **frozen, "stop": False},
        "contrast_m10": {"how_id": "F3", "seeds": [42], **frozen, "stop": False},
        "contrast_m11": {"how_id": "F3", "seeds": [42], **frozen, "stop": False},
        "contrast_m12": {"how_id": "F1", "seeds": [42], **frozen, "stop": True},
    }
    rows = [{"case_id": cid, "action": act} for cid, act in gold.items()]
    report = grade_actions(rows)
    assert report["passed"] == 12
    assert report["ok"] is True


def test_idle_and_confound_fail():
    cases = {row["case_id"]: row for row in load_model_exam()}
    frozen = _frozen()
    idle = grade_action(
        cases["contrast_m01"],
        {"how_id": "F1", "seeds": [42], **frozen, "stop": False},
    )
    confound = grade_action(
        cases["contrast_m04"],
        {"how_id": "F3", "seeds": [43], **frozen, "stop": False},
    )
    dataset = grade_action(
        cases["contrast_m05"],
        {
            "how_id": "F3",
            "seeds": [42],
            "dataset_id": "other_set",
            "slice_id": "low_light_subset_v1",
            "budget_class": "formal",
            "stop": False,
        },
    )
    invented = grade_action(
        cases["contrast_m08"],
        {"how_id": "late_fusion", "seeds": [42], **frozen, "stop": False},
    )
    hidden = grade_action(
        cases["contrast_m09"],
        {"how_id": "N1", "seeds": [42], **frozen, "stop": False},
    )
    assert idle["pass"] is False
    assert confound["pass"] is False
    assert dataset["pass"] is False
    assert invented["pass"] is False
    assert hidden["pass"] is False


def test_p1_idle_is_not_sft_positive():
    label = label_contrast(
        {
            "anchor": {"how_id": "F1", "seeds": [42], **_frozen()},
            "plan": {"how_id": "F1", "seeds": [42], **_frozen(), "stop": False},
            "executed": {"how_id": "F1", "seeds": [42]},
        }
    )
    assert label == "idle"
    assert bucket_from_label(label) == "dpo_rejected"


def test_holdout_ids_must_not_enter_training_blob():
    ids = holdout_case_ids()
    assert "contrast_m01" in ids
    assert "contrast_b02" in ids
    leaks = training_leaks_holdout([{"id": "sft_ok", "input": "ordinary round"}])
    assert leaks == []
    leaked = training_leaks_holdout(
        [{"id": "bad", "observation": "copy of contrast_m01"}]
    )
    assert "contrast_m01" in leaked


def test_jsonl_lives_under_evals(tmp_path: Path):
    root = Path(__file__).resolve().parents[2] / "evals" / "llm" / "contrast"
    assert (root / "contrast_holdout_v1.jsonl").is_file()
    assert (root / "data_bucket_holdout_v1.jsonl").is_file()
    del tmp_path
