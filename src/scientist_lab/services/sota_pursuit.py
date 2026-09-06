"""SOTA pursuit: when metrics stall, queue the next catalog HOW or LLM plugin draft.

Does not forge metrics. Does not bypass Gate. Catalog HOWs are trusted presets;
plugin HOWs go through how_lifecycle (author → smoke → register).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from scientist_lab.adapters.dfine.how_catalog import ALLOWED_HOW, CATALOG_ID
from scientist_lab.core.how_pending import (
    HowPendingError,
    decide_candidate,
    ingest_catalog_exploration_candidate,
    ingest_plugin_exploration_candidate,
    load_store,
    pending_path,
)
from scientist_lab.core.schema_registry import load_json

# Fusion/neck knobs most likely to move APS_lowlight before inventing operators.
CATALOG_SOTA_ORDER = ("F3", "N1", "A4", "F1", "F0", "N0")
# Evidence-gap fill order: RGB-only control → low-light neck (A4) → FDPN/N0.
EVIDENCE_GAP_CATALOG_ORDER = ("F0", "A4", "N1", "N0", "F3", "F1")
# Formal C1 reference (early_concat); informational stretch, not this campaign's Claim.
FORMAL_C1_APS_LOWLIGHT = 0.0326


def _as_record(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _primary_metric(protocol: Mapping[str, Any] | None) -> str:
    primary = _as_record(_as_record(protocol).get("objective")).get("primary")
    token = str(_as_record(primary).get("metric") or "APS_lowlight").strip()
    return token or "APS_lowlight"


def _metric_from_map(metrics: Mapping[str, Any], metric: str) -> float | None:
    for key in (metric, "APS_lowlight", "APS"):
        raw = metrics.get(key)
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return float(raw)
    return None


def used_how_ids(campaign: Mapping[str, Any]) -> set[str]:
    seen: set[str] = set()
    for raw in _as_list(campaign.get("steps")):
        step = _as_record(raw)
        action = str(step.get("action") or "")
        report = _as_record(step.get("report"))
        if action in {"NEED_PLAN", "NEXT_ROUND"}:
            plan = _as_record(report.get("plan"))
            hid = str(plan.get("how_id") or "").strip().upper()
            if hid:
                seen.add(hid)
        if action == "NEED_EXECUTION":
            handle = _as_record(report.get("handle"))
            contract = _as_record(handle.get("contract"))
            mat = _as_record(contract.get("materialization"))
            how = _as_record(mat.get("how"))
            hid = str(how.get("how_id") or "").strip().upper()
            if hid:
                seen.add(hid)
    return seen


def best_primary_metric(campaign: Mapping[str, Any], *, metric: str) -> float | None:
    best: float | None = None
    for raw in _as_list(campaign.get("steps")):
        step = _as_record(raw)
        if str(step.get("action") or "") != "NEED_PARSE":
            continue
        mets = _as_record(_as_record(_as_record(step.get("report")).get("result")).get("metrics"))
        val = _metric_from_map(mets, metric)
        if val is None:
            continue
        best = val if best is None else max(best, val)
    return best


def load_baseline(work: Path) -> dict[str, Any]:
    path = work / "baseline_metrics.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def build_sota_board(
    campaign: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any] | None = None,
    baseline: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metric = _primary_metric(protocol)
    base_row = dict(baseline or {})
    baseline_val = _metric_from_map(base_row, metric)
    best = best_primary_metric(campaign, metric=metric)
    used = sorted(used_how_ids(campaign))
    order = (
        EVIDENCE_GAP_CATALOG_ORDER
        if bool(campaign.get("evidence_gap_priority"))
        else CATALOG_SOTA_ORDER
    )
    remaining = [hid for hid in order if hid in ALLOWED_HOW and hid not in used]
    gap = None
    if baseline_val is not None and best is not None:
        gap = best - baseline_val
    stretch_gap = None
    if best is not None:
        stretch_gap = best - FORMAL_C1_APS_LOWLIGHT
    at_or_above_baseline = (
        baseline_val is not None and best is not None and best > baseline_val
    )
    return {
        "primary_metric": metric,
        "baseline": baseline_val,
        "baseline_source": base_row.get("source"),
        "best": best,
        "gap_vs_baseline": gap,
        "formal_c1_reference": FORMAL_C1_APS_LOWLIGHT,
        "gap_vs_formal_c1": stretch_gap,
        "at_or_above_baseline": at_or_above_baseline,
        "used_how_ids": used,
        "remaining_catalog_how_ids": remaining,
        "next_catalog_how_id": remaining[0] if remaining else None,
        "catalog_id": CATALOG_ID,
        "gpu_rounds": int(campaign.get("gpu_rounds") or 0),
    }


def _next_plugin_how_id(store: Mapping[str, Any], used: set[str]) -> str:
    taken = {
        str(row.get("how_id") or "").strip().upper()
        for row in _as_list(store.get("candidates"))
        if isinstance(row, Mapping)
    }
    taken |= used
    for n in range(1, 20):
        token = f"P{n}"
        if token not in taken and token not in ALLOWED_HOW:
            return token
    return "P99"


def refresh_live_m1_brief(
    work: Path | str,
    spec: Mapping[str, Any],
    *,
    draft_runner: Any | None = None,
    lifecycle_runner: Any | None = None,
) -> dict[str, Any]:
    """Refresh Planner brief from campaign evidence. Does not queue catalog HOW."""
    root = Path(work)
    campaign_path = root / "campaign.json"
    if not campaign_path.is_file():
        raise FileNotFoundError(f"missing campaign.json in {root}")
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    protocol = load_json(root / "protocol.json") if (root / "protocol.json").is_file() else {}
    from scientist_lab.services.campaign_notebook import build_round_cards
    from scientist_lab.services.evaluation_matrix import build_live_experiment_brief, sota_pursuit_allowed
    from scientist_lab.services.registered_experiments import RegisteredExperimentService

    metric = _primary_metric(protocol)
    rounds = build_round_cards(campaign, metric=metric)
    allowed, block_reason = sota_pursuit_allowed(protocol, campaign, rounds=rounds)
    if not allowed:
        board = build_sota_board(campaign, protocol=protocol, baseline=load_baseline(root))
        return {
            "progressed": False,
            "ok": True,
            "action": "wrong_campaign_mode",
            "reason": block_reason,
            "board": board,
        }

    reg_items: list[Any] = []
    try:
        project_root = root.parents[2]
        reg_items = RegisteredExperimentService(project_root).list().get("items") or []
    except (OSError, ValueError, IndexError):
        reg_items = []
    brief = build_live_experiment_brief(
        campaign,
        protocol=protocol,
        rounds=rounds,
        registered_experiments=reg_items,
        how_pending=load_store(pending_path(root)),
    )
    baseline = load_baseline(root)
    board = build_sota_board(campaign, protocol=protocol, baseline=baseline)

    draft_result = None
    lifecycle_result = None
    progressed = False
    action = "brief_refreshed"
    reason = "live M1 brief refreshed for next Planner round; no catalog auto-queue"
    if bool(spec.get("llm_how_lifecycle")) and bool(spec.get("confirm_human_gate")):
        # Prefer lifecycle_runner (includes draft arm). Avoid double draft ticks.
        runner = lifecycle_runner or draft_runner
        if runner is not None:
            lifecycle_result = runner(root, spec)
            if lifecycle_result.get("action") == "invent_awaiting_human":
                return {
                    "progressed": True,
                    "ok": True,
                    "action": "invent_awaiting_human",
                    "reason": lifecycle_result.get("reason"),
                    "live_m1_brief": brief,
                    "board": board,
                    "draft_arm": lifecycle_result.get("draft_arm") or lifecycle_result,
                    "lifecycle": lifecycle_result if lifecycle_runner else None,
                    "requires_human_review": True,
                }
            if lifecycle_result.get("progressed"):
                progressed = True
                action = str(lifecycle_result.get("action") or action)
                reason = str(lifecycle_result.get("reason") or reason)
                draft_result = lifecycle_result.get("draft_arm")

    return {
        "progressed": progressed,
        "ok": True,
        "action": action,
        "reason": reason,
        "live_m1_brief": brief,
        "board": board,
        "draft_arm": draft_result,
        "lifecycle": lifecycle_result if lifecycle_runner else None,
    }


def run_sota_pursuit_tick(
    work: Path | str,
    spec: Mapping[str, Any],
    *,
    draft_runner: Any | None = None,
    lifecycle_runner: Any | None = None,
) -> dict[str, Any]:
    """Live M1: refresh experiment brief for Planner. Legacy catalog queue only if live_m1=false."""
    if spec.get("live_m1", True):
        return refresh_live_m1_brief(
            work,
            spec,
            draft_runner=draft_runner,
            lifecycle_runner=lifecycle_runner,
        )

    # Legacy: mechanical catalog queue when live_m1=false.
    root = Path(work)
    campaign_path = root / "campaign.json"
    if not campaign_path.is_file():
        raise FileNotFoundError(f"missing campaign.json in {root}")
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    protocol = load_json(root / "protocol.json") if (root / "protocol.json").is_file() else {}
    from scientist_lab.services.campaign_notebook import build_round_cards
    from scientist_lab.services.evaluation_matrix import sota_pursuit_allowed

    metric = _primary_metric(protocol)
    rounds = build_round_cards(campaign, metric=metric)
    allowed, block_reason = sota_pursuit_allowed(protocol, campaign, rounds=rounds)
    if not allowed:
        board = build_sota_board(campaign, protocol=protocol, baseline=load_baseline(root))
        return {
            "progressed": False,
            "ok": True,
            "action": "wrong_campaign_mode",
            "reason": block_reason,
            "board": board,
        }

    baseline = load_baseline(root)
    board = build_sota_board(campaign, protocol=protocol, baseline=baseline)
    dest = pending_path(root)
    store = load_store(dest)

    if board.get("at_or_above_baseline") and not board.get("remaining_catalog_how_ids"):
        can_plugin = bool(spec.get("llm_how_lifecycle")) and bool(spec.get("llm_live"))
        if not can_plugin:
            return {
                "progressed": False,
                "ok": True,
                "action": "baseline_met",
                "board": board,
                "reason": "best already above frozen baseline; no mandatory catalog queue",
            }

    progressed = False
    action = "idle"
    reason = ""
    how_id = board.get("next_catalog_how_id")
    round_id = f"round_{int(campaign.get('gpu_rounds') or 0) + 1}"

    if how_id:
        try:
            ingest_catalog_exploration_candidate(
                dest,
                str(how_id),
                round_id=round_id,
                reason=(
                    f"SOTA 推进：目录 {how_id} 尚未在本实验跑过。"
                    f" 当前 best {metric}={board.get('best')}，baseline={board.get('baseline')}。"
                ),
            )
            decide_candidate(
                dest,
                f"howc_explore_{str(how_id).upper()}_{round_id}",
                decision="register",
                actor="sota_pursuit",
                note="catalog preset; auto-register for next Planner round",
                confirm_human_gate=bool(spec.get("confirm_human_gate")),
            )
            progressed = True
            action = "register_catalog"
            reason = f"queued catalog HOW {how_id} for next round"
        except (HowPendingError, ValueError) as exc:
            action = "catalog_failed"
            reason = str(exc)
    elif bool(spec.get("llm_how_lifecycle")) and bool(spec.get("llm_live")):
        plugin_how = _next_plugin_how_id(store, used_how_ids(campaign))
        semantic = _latest_semantic_priority(campaign)
        mechanism = semantic or (
            "LLM-authored fusion plugin: gated RGB-T blend beyond catalog F3 preset."
        )
        try:
            ingest_plugin_exploration_candidate(
                dest,
                plugin_how,
                mechanism,
                round_id=round_id,
                semantic_ref=semantic,
            )
            progressed = True
            action = "propose_plugin"
            reason = f"catalog exhausted; proposed plugin HOW {plugin_how} for lifecycle author"
        except (HowPendingError, ValueError) as exc:
            action = "plugin_propose_failed"
            reason = str(exc)

    lifecycle_result = None
    if lifecycle_runner is not None and bool(spec.get("llm_how_lifecycle")):
        lifecycle_result = lifecycle_runner(root, spec)
        if lifecycle_result.get("progressed"):
            progressed = True
            action = str(lifecycle_result.get("action") or action)

    board = build_sota_board(campaign, protocol=protocol, baseline=baseline)
    return {
        "progressed": progressed,
        "ok": True,
        "action": action,
        "reason": reason,
        "board": board,
        "lifecycle": lifecycle_result,
    }


def _latest_semantic_priority(campaign: Mapping[str, Any]) -> str | None:
    for raw in reversed(_as_list(campaign.get("steps"))):
        step = _as_record(raw)
        if str(step.get("action") or "") != "NEED_REVIEW":
            continue
        proposal = _as_record(_as_record(step.get("report")).get("semantic_proposal"))
        text = str(proposal.get("next_research_priority") or "").strip()
        if text:
            return text
    return None
