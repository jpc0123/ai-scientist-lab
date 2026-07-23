"""Normalize comparison outcomes for the workbench UI (v2.0.3)."""

from __future__ import annotations

from typing import Any


CONCLUSION_LABELS = {
    "better": "better（候选更好）",
    "worse": "worse（基线更好）",
    "practically_equivalent": "practically_equivalent（实质等价）",
    "inconclusive": "inconclusive（证据不足/结论不明）",
}


def conclusion_from_relation(relation: str | None) -> str:
    value = (relation or "").strip().lower()
    if value in {"candidate_better", "better", "improved", "supported"}:
        return "better"
    if value in {"baseline_better", "worse", "rejected", "degraded"}:
        return "worse"
    if value in {"practically_equivalent", "equivalent", "tie", "similar"}:
        return "practically_equivalent"
    return "inconclusive"


def enrich_comparison(payload: dict[str, Any], *, mode: str) -> dict[str, Any]:
    """Attach a stable textual ``conclusion`` for dashboard/compare UI."""
    data = dict(payload or {})
    relation = (
        data.get("performance_relation")
        or data.get("hypothesis_status")
        or data.get("overall_decision")
        or data.get("tradeoff_status")
    )
    # Pairwise attempt comparison uses hypothesis_status enum-like values.
    hs = str(data.get("hypothesis_status") or "").lower()
    if hs in {"supported"}:
        conclusion = "better"
    elif hs in {"rejected"}:
        conclusion = "worse"
    elif data.get("practically_equivalent") is True:
        conclusion = "practically_equivalent"
    else:
        conclusion = conclusion_from_relation(str(relation) if relation else "")

    data["conclusion"] = conclusion
    data["conclusion_label"] = CONCLUSION_LABELS[conclusion]
    data["conclusion_note"] = (
        "比较结论使用文字标签，不只依赖颜色。"
        " better/worse/practically_equivalent/inconclusive 四选一。"
    )
    data["compare_mode"] = mode
    return data
