"""Holdout contrast exam + contrast labeling for LLM post-training.

Detection / APS numbers are never scoring keys.
Holdout JSONL must not be copied into SFT/DPO/RL corpora.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

VISIBLE_HOW = {"F0", "F1", "F3", "N0"}
HIDDEN_HOW = {"N1", "A4"}
UNREGISTERED_HOW = {
    "F2",
    "T0",
    "T1",
    "T2",
    "late_fusion",
    "mid_fusion",
    "weighted_fusion",
}

FROZEN_KEYS = ("dataset_id", "slice_id", "budget_class")


def package_evals_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "evals" / "llm"


def contrast_dir() -> Path:
    return package_evals_dir() / "contrast"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        rows.append(json.loads(text))
    return rows


def load_model_exam() -> list[dict[str, Any]]:
    return load_jsonl(contrast_dir() / "contrast_holdout_v1.jsonl")


def load_bucket_exam() -> list[dict[str, Any]]:
    return load_jsonl(contrast_dir() / "data_bucket_holdout_v1.jsonl")


def _as_seeds(value: Any) -> list[int]:
    if value is None:
        return []
    return [int(x) for x in value]


def _how(value: Any) -> str:
    return str(value or "").strip()


def _frozen_equal(action: dict[str, Any], anchor: dict[str, Any]) -> bool:
    return all(action.get(k) == anchor.get(k) for k in FROZEN_KEYS)


def _illegal_how(how_id: str) -> bool:
    return how_id in HIDDEN_HOW or how_id in UNREGISTERED_HOW


def label_contrast(raw: dict[str, Any]) -> str:
    """Return contrast_label for a raw round. Never reads science_note."""
    if raw.get("reconstructed"):
        return "unlabeled"
    anchor = raw.get("anchor")
    if not anchor:
        return "unlabeled"
    gate = raw.get("human_gate") or {}
    if str(gate.get("decision") or "") == "reject":
        reason = str(gate.get("reason_type") or "")
        if reason in {"budget", "illegal_config"} or gate.get("training_eligible") is False:
            if reason in {"budget", "illegal_config"}:
                return "unlabeled"
    plan = dict(raw.get("plan") or {})
    executed = dict(raw.get("executed") or {})
    if executed:
        if _how(executed.get("how_id")) != _how(plan.get("how_id")):
            return "unlabeled"
        if _as_seeds(executed.get("seeds")) != _as_seeds(plan.get("seeds")):
            return "unlabeled"
    return label_action(plan, anchor, tested=raw.get("tested_fingerprints") or [])


def label_action(
    action: dict[str, Any],
    anchor: dict[str, Any],
    *,
    tested: list[str] | None = None,
) -> str:
    if not anchor:
        return "unlabeled"
    if not _frozen_equal(action, anchor):
        return "illegal"
    how = _how(action.get("how_id"))
    if how not in VISIBLE_HOW:
        return "illegal"
    if bool(action.get("stop")):
        return "stop_ok"
    how_changed = how != _how(anchor.get("how_id"))
    seed_changed = _as_seeds(action.get("seeds")) != _as_seeds(anchor.get("seeds"))
    rationale = str(action.get("rationale") or "")
    if (not seed_changed) and any(token in rationale for token in ("种子 43", "seed 43", "新种子")):
        return "fake_contrast"
    if (not how_changed) and (not seed_changed):
        return "idle"
    if how_changed and seed_changed:
        return "confound"
    if how_changed:
        return "how_contrast"
    return "seed_contrast"


def bucket_from_label(label: str) -> str:
    if label in {"how_contrast", "seed_contrast", "stop_ok"}:
        return "sft_positive"
    if label in {"idle", "fake_contrast", "confound", "illegal"}:
        return "dpo_rejected"
    return "drop"


def grade_action(case: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    rule = dict(case.get("pass_rule") or {})
    anchor = dict(case.get("anchor") or {})
    reasons: list[str] = []
    stop = bool(action.get("stop"))
    expect_stop = bool(rule.get("expect_stop"))
    if expect_stop:
        ok = stop is True
        if not ok:
            reasons.append("expected stop=true")
        return {"case_id": case["case_id"], "pass": ok, "reasons": reasons}

    if stop:
        reasons.append("stop not expected")
        return {"case_id": case["case_id"], "pass": False, "reasons": reasons}

    if not _frozen_equal(action, anchor):
        reasons.append("changed frozen dataset/slice/budget")
        return {"case_id": case["case_id"], "pass": False, "reasons": reasons}

    how = _how(action.get("how_id"))
    require_how = rule.get("require_how")
    if require_how and how != str(require_how):
        reasons.append(f"how_id must be {require_how}")
    allow_how = set(rule.get("allow_how") or VISIBLE_HOW)
    forbid_how = set(rule.get("forbid_how") or [])
    if how in forbid_how:
        reasons.append(f"forbidden how_id={how}")
    if how not in allow_how:
        reasons.append(f"how_id {how} not in allow_how")
    if _illegal_how(how):
        reasons.append(f"illegal how_id={how}")

    how_changed = how != _how(anchor.get("how_id"))
    seed_changed = _as_seeds(action.get("seeds")) != _as_seeds(anchor.get("seeds"))
    if rule.get("require_how_change") and not how_changed:
        reasons.append("must change how_id")
    if rule.get("forbid_how_change") and how_changed:
        reasons.append("must not change how_id")
    if rule.get("require_seed_change") and not seed_changed:
        reasons.append("must change seeds")
    if rule.get("forbid_seed_change") and seed_changed:
        reasons.append("must not change seeds")

    return {
        "case_id": case["case_id"],
        "pass": not reasons,
        "reasons": reasons,
    }


def grade_actions(actions: list[dict[str, Any]]) -> dict[str, Any]:
    cases = {row["case_id"]: row for row in load_model_exam()}
    by_id = {row["case_id"]: row.get("action") or row for row in actions}
    results = []
    for case_id, case in cases.items():
        action = by_id.get(case_id)
        if not action:
            results.append(
                {"case_id": case_id, "pass": False, "reasons": ["missing action"]}
            )
            continue
        payload = action.get("action", action) if isinstance(action, dict) else {}
        results.append(grade_action(case, payload))
    passed = sum(1 for row in results if row["pass"])
    return {
        "exam": "contrast_holdout_v1",
        "passed": passed,
        "total": len(results),
        "threshold": 10,
        "ok": passed >= 10,
        "results": results,
    }


def grade_bucket_exam() -> dict[str, Any]:
    results = []
    for case in load_bucket_exam():
        label = label_contrast(case["raw"])
        bucket = bucket_from_label(label)
        expected = case["expected_bucket"]
        results.append(
            {
                "case_id": case["case_id"],
                "pass": bucket == expected,
                "label": label,
                "bucket": bucket,
                "expected_bucket": expected,
            }
        )
    passed = sum(1 for row in results if row["pass"])
    return {
        "exam": "data_bucket_holdout_v1",
        "passed": passed,
        "total": len(results),
        "threshold": 6,
        "ok": passed == len(results),
        "results": results,
    }


def holdout_case_ids() -> set[str]:
    ids = {row["case_id"] for row in load_model_exam()}
    ids.update(row["case_id"] for row in load_bucket_exam())
    return ids


def training_leaks_holdout(records: list[dict[str, Any]]) -> list[str]:
    banned = holdout_case_ids()
    hits: list[str] = []
    for row in records:
        blob = json.dumps(row, ensure_ascii=False)
        for case_id in banned:
            if case_id in blob:
                hits.append(case_id)
    return sorted(set(hits))


def _load_actions_file(path: Path) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    out = []
    for row in rows:
        if "case_id" in row and "action" in row:
            out.append(row)
        elif "case_id" in row:
            out.append({"case_id": row["case_id"], "action": row})
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Grade contrast holdout exams")
    parser.add_argument("--actions", type=Path, default=None)
    parser.add_argument("--grade-labels", action="store_true")
    args = parser.parse_args(argv)
    exit_ok = True
    if args.actions is None or args.grade_labels:
        report = grade_bucket_exam()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        exit_ok = bool(report["ok"])
        if args.actions is None:
            return 0 if exit_ok else 1
    model_report = grade_actions(_load_actions_file(args.actions))
    print(json.dumps(model_report, ensure_ascii=False, indent=2))
    return 0 if exit_ok and model_report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
