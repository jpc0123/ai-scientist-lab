"""Round-to-round controls: previous-shot Rubric baseline and evaluation seed.

Does not KEEP/DISCARD. Does not invent HOW. Adapter still reads
``plan.evaluation.seeds[0]`` as ``contract.seed``.
"""

from __future__ import annotations

from typing import Any, Mapping


def numeric_metrics(metrics: Mapping[str, Any] | None) -> dict[str, float]:
    """Keep number|null metric maps; drop strings/bools."""
    out: dict[str, float] = {}
    for key, value in dict(metrics or {}).items():
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)):
            out[str(key)] = float(value)
    return out


def plan_seed(plan: Mapping[str, Any] | None) -> int:
    seeds = list((dict(plan or {}).get("evaluation") or {}).get("seeds") or [42])
    if not seeds:
        return 42
    try:
        return int(seeds[0])
    except (TypeError, ValueError):
        return 42


def plan_how_token(plan: Mapping[str, Any] | None) -> str:
    payload = dict(plan or {})
    for key in ("how_id", "catalog_how_id"):
        value = payload.get(key)
        if value:
            return str(value).strip().upper()
    for row in payload.get("proposed_changes") or []:
        if not isinstance(row, Mapping):
            continue
        detail = dict(row.get("detail") or {})
        for key in ("how_id", "catalog_how_id"):
            if detail.get(key):
                return str(detail[key]).strip().upper()
    return ""


def same_how(previous_plan: Mapping[str, Any], next_plan: Mapping[str, Any]) -> bool:
    prev_how = plan_how_token(previous_plan)
    next_how = plan_how_token(next_plan)
    if prev_how and next_how:
        return prev_how == next_how
    return list(previous_plan.get("modification_scope") or []) == list(
        next_plan.get("modification_scope") or []
    )


def coerce_seed(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


FUSION_HOW_IDS = ("F0", "F1", "F3")
_CONTRAST_HOW = {"F1": "F3", "F3": "F0", "F0": "F3"}


def contrast_fusion_how_id(current: str | None) -> str:
    """Next registered fusion HOW. Does not invent operators. N1/A4 stay hidden."""
    token = str(current or "F1").strip().upper() or "F1"
    return _CONTRAST_HOW.get(token, "F3")


def should_contrast_fusion_how(
    *,
    previous_plan: Mapping[str, Any],
    last_review_decision: str | None,
    last_primary_delta: float | None = None,
) -> bool:
    """True when the next shot must change HOW, not idle the same fusion switch.

    Fusion catalog only (F0/F1/F3). Neck-only v1 plans are unchanged.
    """
    how = plan_how_token(previous_plan)
    if how not in FUSION_HOW_IDS:
        return False
    review = str(last_review_decision or "").upper()
    if review == "DISCARD":
        return True
    if last_primary_delta is not None:
        try:
            if float(last_primary_delta) < 0:
                return True
        except (TypeError, ValueError):
            pass
    if bool(previous_plan.get("bootstrap")) and review in {"", "REPLICATE", "KEEP"}:
        return True
    return False


def resolve_next_seed(
    *,
    previous_plan: Mapping[str, Any],
    next_plan: Mapping[str, Any],
    last_review_decision: str | None,
    requested_seed: Any = None,
) -> int:
    """Explicit *new* seed wins. REPLICATE + same HOW + seed 42 again is not explicit."""
    previous = plan_seed(previous_plan)
    explicit = coerce_seed(requested_seed)
    review = str(last_review_decision or "").upper()
    if (
        explicit is not None
        and review == "REPLICATE"
        and same_how(previous_plan, next_plan)
        and explicit == previous
    ):
        explicit = None
    if explicit is not None:
        return explicit
    if review == "REPLICATE" and same_how(previous_plan, next_plan):
        return previous + 1
    return previous


def with_evaluation_seed(evaluation: Mapping[str, Any] | None, seed: int) -> dict[str, Any]:
    payload = dict(evaluation or {"method": "fast_eval"})
    payload["seeds"] = [int(seed)]
    return payload


def apply_next_round_seed(
    plan: Mapping[str, Any],
    *,
    previous_plan: Mapping[str, Any],
    last_review_decision: str | None,
    requested_seed: Any = None,
) -> dict[str, Any]:
    """Write ``evaluation.seeds`` so Adapter ``contract.seed`` actually changes."""
    next_plan = dict(plan)
    seed = resolve_next_seed(
        previous_plan=previous_plan,
        next_plan=next_plan,
        last_review_decision=last_review_decision,
        requested_seed=requested_seed,
    )
    next_plan["evaluation"] = with_evaluation_seed(next_plan.get("evaluation"), seed)
    return next_plan
