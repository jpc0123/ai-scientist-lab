"""What SOTA actually requires: ablations, multi-seed, cross-model — not catalog demo cycling.

Does not forge metrics. Does not auto-start GPU. Status is derived from campaign steps only.
"""

from __future__ import annotations

from typing import Any, Mapping

from scientist_lab.adapters.dfine.how_catalog import ALLOWED_HOW
from scientist_lab.services.registered_experiments import (
    BUILTIN_EXPERIMENT_ID,
    BUILTIN_RTDETR_TRANSFER_ID,
    adapter_token,
    primary_metric_token,
    slice_id_token,
)

# Single-variable ablation rows for V26 low-light D-FINE (ClaimGate C2 needs ablation.present).
FUSION_ABLATION_ROWS = (
    {"how_id": "F0", "axis": "fusion", "role": "rgb_only_control", "label_zh": "RGB-only 对照"},
    {"how_id": "F1", "axis": "fusion", "role": "early_concat_baseline", "label_zh": "Early concat (R0)"},
    {"how_id": "F3", "axis": "fusion", "role": "gated_multiscale", "label_zh": "Gated multiscale 候选"},
)
NECK_ABLATION_ROWS = (
    {"how_id": "N0", "axis": "neck", "role": "standard_neck", "label_zh": "Standard HybridEncoder"},
    {"how_id": "N1", "axis": "neck", "role": "fdpn_neck", "label_zh": "FDPN neck 消融"},
)

CROSS_MODEL_ROWS = (
    {
        "experiment_id": BUILTIN_EXPERIMENT_ID,
        "adapter": "dfine",
        "label_zh": "D-FINE 主战役 (G2)",
    },
    {
        "experiment_id": BUILTIN_RTDETR_TRANSFER_ID,
        "adapter": "rtdetr",
        "label_zh": "RT-DETR 迁移对照 (G3/P4)",
    },
)

MIN_CONFIRMATION_SEEDS = 2


def _as_record(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _round_how_seed(rounds: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in rounds:
        row = _as_record(raw)
        hid = str(row.get("how_id") or "").strip().upper()
        if not hid or hid == "—":
            continue
        rows.append(
            {
                "how_id": hid,
                "seed": row.get("seed"),
                # Round cards store the primary under "value"; older payloads used
                # "primary_value". Accept either so multi-seed / best tracking works.
                "metric_value": row.get("primary_value", row.get("value")),
                "round_index": row.get("round_index"),
            }
        )
    return rows


def detect_campaign_mode(
    protocol: Mapping[str, Any] | None,
    campaign: Mapping[str, Any] | None = None,
    *,
    rounds: list[Mapping[str, Any]] | None = None,
) -> str:
    """Classify what kind of science this campaign is doing."""
    proto = _as_record(protocol)
    camp = _as_record(campaign)
    title = str(camp.get("experiment_title") or proto.get("title") or "").lower()
    if any(token in title for token in ("stable", "stability", "across seed", "across different random seed")):
        return "seed_stability"

    run_rows = _round_how_seed(list(rounds or []))
    if run_rows:
        hows = {r["how_id"] for r in run_rows}
        seeds = {r["seed"] for r in run_rows if r.get("seed") is not None}
        if len(hows) == 1 and len(seeds) >= 2:
            return "seed_stability"

    adapter = adapter_token(proto)
    if adapter in {"rtdetr", "rt_detr"}:
        return "cross_model_transfer"

    metric = primary_metric_token(proto)
    slice_id = slice_id_token(proto)
    if slice_id == "low_light_subset_v1" and metric == "APS_lowlight":
        return "v26_improvement"

    return "exploratory"


def sota_pursuit_allowed(
    protocol: Mapping[str, Any] | None,
    campaign: Mapping[str, Any] | None = None,
    *,
    rounds: list[Mapping[str, Any]] | None = None,
) -> tuple[bool, str]:
    mode = detect_campaign_mode(protocol, campaign, rounds=rounds)
    if mode == "seed_stability":
        return (
            False,
            "当前是 seed 稳定性实验（固定 HOW、变 seed），不能按 SOTA 目录轮换 HOW。"
            "请用 V26 low-light 登记实验做消融/对比。",
        )
    if mode == "exploratory":
        return (
            False,
            "当前协议不是 V26 APS_lowlight + low_light_subset_v1 改进战役；"
            "catalog 轮换不能代替对比实验或消融表。",
        )
    if mode == "cross_model_transfer":
        return True, "RT-DETR 迁移战役：仅 F1 baseline → F3 transfer，不是 catalog 全扫。"
    return True, "V26 改进战役：按消融轴推进 fusion/neck，并需多 seed 确认。"


def _ablation_status(used_hows: set[str], row: Mapping[str, Any]) -> str:
    hid = str(row.get("how_id") or "").upper()
    if hid in used_hows:
        return "done"
    if hid in ALLOWED_HOW:
        return "missing"
    return "unsupported"


def _multi_seed_status(run_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_how: dict[str, set[Any]] = {}
    for row in run_rows:
        by_how.setdefault(row["how_id"], set()).add(row.get("seed"))
    best_how = None
    best_val = None
    for row in run_rows:
        val = row.get("metric_value")
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            if best_val is None or float(val) > best_val:
                best_val = float(val)
                best_how = row["how_id"]
    seed_count = len(by_how.get(best_how or "", set()) - {None})
    return {
        "best_how_id": best_how,
        "best_metric": best_val,
        "seed_count_on_best": seed_count,
        "required_seeds": MIN_CONFIRMATION_SEEDS,
        "status": "done" if seed_count >= MIN_CONFIRMATION_SEEDS else "missing",
    }


def _cross_model_status(registered: list[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    reg_by_id = {
        str(row.get("experiment_id") or ""): _as_record(row)
        for row in _as_list(registered)
    }
    out: list[dict[str, Any]] = []
    for spec in CROSS_MODEL_ROWS:
        eid = str(spec["experiment_id"])
        row = reg_by_id.get(eid) or {}
        out.append(
            {
                **dict(spec),
                "registered": eid in reg_by_id,
                "status": "registered" if eid in reg_by_id else "missing",
                "title": row.get("title"),
            }
        )
    return out


def build_evaluation_matrix(
    campaign: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any] | None = None,
    rounds: list[Mapping[str, Any]] | None = None,
    registered_experiments: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Required vs completed for ablation, multi-seed, and cross-model comparison."""
    proto = _as_record(protocol)
    run_rows = _round_how_seed(list(rounds or []))
    used_hows = {r["how_id"] for r in run_rows}
    mode = detect_campaign_mode(proto, campaign, rounds=rounds)
    allowed, allow_reason = sota_pursuit_allowed(proto, campaign, rounds=rounds)

    fusion_rows = [
        {**dict(spec), "status": _ablation_status(used_hows, spec)}
        for spec in FUSION_ABLATION_ROWS
    ]
    neck_rows = [
        {**dict(spec), "status": _ablation_status(used_hows, spec)}
        for spec in NECK_ABLATION_ROWS
    ]
    multi_seed = _multi_seed_status(run_rows)
    cross_model = _cross_model_status(registered_experiments)

    fusion_done = sum(1 for r in fusion_rows if r["status"] == "done")
    neck_done = sum(1 for r in neck_rows if r["status"] == "done")
    # A4 = early_concat + FDPN; counts as starting the neck axis when N0/N1 absent.
    if "A4" in used_hows:
        neck_done = max(neck_done, 1)
    cross_done = sum(1 for r in cross_model if r["status"] == "registered")

    gaps: list[str] = []
    if mode == "seed_stability":
        gaps.append("本实验测 seed 方差，不是 SOTA 消融/跨模型对比。")
        if multi_seed["status"] != "done":
            gaps.append(
                f"稳定性需要 ≥{MIN_CONFIRMATION_SEEDS} 个 seed（当前 best HOW 上 {multi_seed['seed_count_on_best']} 个）。"
            )
    elif mode == "v26_improvement":
        if fusion_done < len(fusion_rows):
            missing = [r["how_id"] for r in fusion_rows if r["status"] == "missing"]
            gaps.append(f"fusion 消融未完成：缺 {', '.join(missing)}。")
        if neck_done < 1:
            gaps.append("neck 消融（N0 vs N1 / A4）尚未开始。")
        if multi_seed["status"] != "done":
            gaps.append(
                f"best 配置需 ≥{MIN_CONFIRMATION_SEEDS} seed 复现（当前 {multi_seed['seed_count_on_best']}）。"
            )
        if cross_done < len(cross_model):
            gaps.append("跨检测器对照：需单独开 RT-DETR 登记实验（G3），不能在同一战役里混 YOLO。")
    elif mode == "cross_model_transfer":
        if "F1" not in used_hows:
            gaps.append("P4 需先跑 RT-DETR F1 baseline。")
        if "F3" not in used_hows:
            gaps.append("P4 需跑 RT-DETR F3 transfer 对照 D-FINE 方向。")

    catalog_only = used_hows and used_hows <= set(ALLOWED_HOW.keys())
    demo_risk = mode == "v26_improvement" and catalog_only and (
        fusion_done < 2 or multi_seed["status"] != "done"
    )

    return {
        "mode": mode,
        "sota_pursuit_allowed": allowed,
        "sota_pursuit_reason": allow_reason,
        "primary_metric": primary_metric_token(proto) or "APS_lowlight",
        "slice_id": slice_id_token(proto),
        "adapter": adapter_token(proto),
        "fusion_ablations": fusion_rows,
        "neck_ablations": neck_rows,
        "multi_seed": multi_seed,
        "cross_model": cross_model,
        "summary": {
            "fusion_done": fusion_done,
            "fusion_total": len(fusion_rows),
            "neck_done": neck_done,
            "neck_total": len(neck_rows),
            "cross_model_registered": cross_done,
            "cross_model_total": len(cross_model),
        },
        "gaps": gaps,
        "demo_risk": demo_risk,
        "demo_risk_note": (
            "仅轮换 catalog HOW、无消融表/多 seed/跨模型对照，不能支撑 SOTA 或 Claim。"
            if demo_risk
            else None
        ),
        "is_claim": False,
    }


def _open_scientific_questions(
    matrix: Mapping[str, Any],
    run_rows: list[dict[str, Any]],
) -> list[str]:
    """Advisory questions for Planner — not a fixed run order."""
    mode = str(matrix.get("mode") or "")
    questions: list[str] = []
    fusion = {str(r.get("how_id")): r for r in _as_list(matrix.get("fusion_ablations"))}
    ms = _as_record(matrix.get("multi_seed"))
    if mode == "v26_improvement":
        if fusion.get("F1", {}).get("status") != "done":
            questions.append("What is the frozen R0 baseline (typically F1 early_concat) on the primary metric?")
        if fusion.get("F0", {}).get("status") != "done":
            questions.append(
                "Does RGB-only (F0) underperform fusion on the frozen slice, supporting RGB-T necessity?"
            )
        if fusion.get("F3", {}).get("status") != "done":
            questions.append(
                "Does gated multiscale fusion (F3) improve APS_lowlight versus the current best fusion?"
            )
        if ms.get("status") != "done" and ms.get("best_how_id"):
            questions.append(
                f"Is HOW {ms.get('best_how_id')} stable on a second seed (replicate, not a new module)?"
            )
        if not run_rows:
            questions.append("What is the first evidence-driven experiment under the frozen protocol?")
    elif mode == "seed_stability":
        questions.append("How much does the primary metric vary across seeds for the fixed HOW?")
    elif mode == "cross_model_transfer":
        if "F1" not in {r.get("how_id") for r in run_rows}:
            questions.append("What is the RT-DETR F1 baseline on the same slice and metric?")
        if "F3" not in {r.get("how_id") for r in run_rows}:
            questions.append("Does F3 strategy transfer on RT-DETR versus its F1 baseline?")
    if not questions:
        for gap in _as_list(matrix.get("gaps"))[:3]:
            questions.append(str(gap))
    return questions[:6]


def _unused_smoked_plugins(
    campaign: Mapping[str, Any],
    *,
    rounds: list[Mapping[str, Any]] | None,
    how_pending: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Overlay plugins that passed smoke but have never been selected for GPU."""
    store = _as_record(how_pending)
    overlay = _as_record(store.get("registered_overlay"))
    used = {
        str(row.get("how_id") or "").strip().upper()
        for row in _round_how_seed(list(rounds or []))
    } - {""}
    # Also count how_ids from campaign steps if rounds empty.
    if not used:
        for raw in _as_list(campaign.get("steps")):
            step = _as_record(raw)
            if str(step.get("action") or "") not in {"NEED_PLAN", "NEXT_ROUND"}:
                continue
            plan = _as_record(_as_record(step.get("report")).get("plan"))
            hid = str(plan.get("how_id") or "").strip().upper()
            if hid:
                used.add(hid)
    out: list[dict[str, Any]] = []
    for hid, spec in overlay.items():
        token = str(hid or "").strip().upper()
        blob = _as_record(spec)
        if not token or token in used:
            continue
        if not bool(blob.get("smoke_ok")):
            continue
        fusion = str(blob.get("fusion_method") or "")
        if not fusion.startswith("plugin:") and not blob.get("plugin_relpath"):
            continue
        out.append(
            {
                "how_id": token,
                "fusion_method": fusion or f"plugin:{token.lower()}",
                "smoke_ok": True,
                "status": "ready_unused",
                "note": "Smoked overlay plugin; materializable. Prefer over catalog replication.",
            }
        )
    return out


def build_live_experiment_brief(
    campaign: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any] | None = None,
    rounds: list[Mapping[str, Any]] | None = None,
    registered_experiments: list[Mapping[str, Any]] | None = None,
    how_pending: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Planner-facing brief: coverage gaps are advisory; LLM designs the next verification."""
    matrix = build_evaluation_matrix(
        campaign,
        protocol=protocol,
        rounds=rounds,
        registered_experiments=registered_experiments,
    )
    run_rows = _round_how_seed(list(rounds or []))
    best_val = None
    best_how = None
    for row in run_rows:
        val = row.get("metric_value")
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            if best_val is None or float(val) > best_val:
                best_val = float(val)
                best_how = row.get("how_id")
    unused_plugins = _unused_smoked_plugins(
        campaign, rounds=list(rounds or []), how_pending=how_pending
    )
    gap_priority = bool(_as_record(campaign).get("evidence_gap_priority"))
    missing_fusion = [
        str(r.get("how_id"))
        for r in _as_list(matrix.get("fusion_ablations"))
        if r.get("status") == "missing"
    ]
    summary = _as_record(matrix.get("summary"))
    need_neck = int(summary.get("neck_done") or 0) < 1
    ms = _as_record(matrix.get("multi_seed"))
    need_multiseed = ms.get("status") != "done"
    mandate = (
        "Design the next experiment from evidence — do NOT rotate catalog HOW blindly. "
        "Coverage gaps list what a professional write-up eventually needs; you choose order, "
        "hypothesis, and verification. Same HOW + same seed is forbidden unless REPLICATE "
        "with seed+1."
    )
    if gap_priority and (missing_fusion or need_neck or need_multiseed):
        bits: list[str] = []
        if "F0" in missing_fusion:
            bits.append("1) Select F0 (RGB-only control) to close the fusion ablation gap.")
        if need_neck:
            prefer = "A4" if "A4" in ALLOWED_HOW else "N1"
            bits.append(
                f"2) Select {prefer} (or N1) for the neck axis — low-light RGB-T prefers "
                "early_concat+FDPN over RGB-only FDPN when available."
            )
        if need_multiseed:
            best_token = ms.get("best_how_id") or best_how or "current best HOW"
            bits.append(
                f"3) REPLICATE {best_token} with a new seed (seed+1) until ≥"
                f"{ms.get('required_seeds') or 2} seeds exist on that HOW."
            )
        mandate = (
            "PRIORITY: fill evidence gaps (Human Gate: evidence_gap_priority). "
            + " ".join(bits)
            + " Do NOT pick unused overlay plugins until F0, a neck HOW, and best "
            "multi-seed are covered. Same HOW + same seed is forbidden unless "
            "REPLICATE with seed+1."
        )
    elif unused_plugins:
        ids = ", ".join(row["how_id"] for row in unused_plugins)
        mandate = (
            f"PRIORITY: unused smoked overlay plugin(s) ready: {ids}. "
            "For Stage A / method-space expansion, select one of these for the next GPU "
            "round instead of catalog F0/F1/F3/N* replication, unless a hard constraint "
            "blocks them. Catalog gaps remain advisory only. "
            "Same HOW + same seed is forbidden unless REPLICATE with seed+1."
        )
    return {
        **matrix,
        "live_m1": True,
        "advisory_only": not gap_priority,
        "evidence_gap_priority": gap_priority,
        "unused_smoked_plugins": [] if gap_priority else unused_plugins,
        "planner_mandate": mandate,
        "open_scientific_questions": _open_scientific_questions(matrix, run_rows),
        "evidence_snapshot": {
            "gpu_rounds": int(campaign.get("gpu_rounds") or 0),
            "last_action": campaign.get("last_action"),
            "last_review_decision": _latest_review_decision(campaign),
            "best_how_id": best_how or _as_record(matrix.get("multi_seed")).get("best_how_id"),
            "best_metric": best_val or _as_record(matrix.get("multi_seed")).get("best_metric"),
            "used_how_ids": sorted({r["how_id"] for r in run_rows}),
            "unused_smoked_plugin_ids": (
                [] if gap_priority else [row["how_id"] for row in unused_plugins]
            ),
        },
        "is_claim": False,
    }


def _latest_review_decision(campaign: Mapping[str, Any]) -> str | None:
    for raw in reversed(_as_list(campaign.get("steps"))):
        step = _as_record(raw)
        if str(step.get("action") or "") != "NEED_REVIEW":
            continue
        review = _as_record(_as_record(step.get("report")).get("review"))
        token = str(review.get("decision") or "").strip()
        if token:
            return token
    return None
