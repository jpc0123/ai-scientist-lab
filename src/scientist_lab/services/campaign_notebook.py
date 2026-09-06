"""Human-readable lab log for Planner / Executor / Reviewer on a campaign."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from scientist_lab.adapters.dfine.how_catalog import ALLOWED_HOW, CATALOG_ID
from scientist_lab.datasets.rgbt_pair_preview import (
    DEFAULT_DATASET_ID,
    preview_manifest,
    resolve_processed_root,
)


def _as_record(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _fmt(value: Any, digits: int = 4) -> str:
    num = _num(value)
    if num is None:
        return "—"
    return f"{num:.{digits}f}"


def _how(plan: Mapping[str, Any]) -> str:
    return str(plan.get("how_id") or plan.get("how") or "—")


def _primary_metric(protocol: Mapping[str, Any] | None) -> str:
    primary = _as_record(_as_record(_as_record(protocol).get("objective")).get("primary"))
    token = str(primary.get("metric") or "").strip()
    return token or "APS_lowlight"


def _metric_from_map(mets: Mapping[str, Any], metric: str) -> Any:
    for key in (metric, "APS_lowlight", "APS"):
        if key in mets and mets.get(key) is not None:
            return mets.get(key)
    return None


def _seed_of(plan: Mapping[str, Any]) -> Any:
    seeds = _as_list(_as_record(plan.get("evaluation")).get("seeds"))
    if seeds:
        return seeds[0]
    changes = _as_list(plan.get("proposed_changes"))
    detail = _as_record(_as_record(changes[0] if changes else {}).get("detail"))
    if detail.get("seed") is not None:
        return detail.get("seed")
    selected = _as_record(
        _as_record(_as_record(plan.get("llm_trace")).get("parsed")).get("selected")
    )
    return selected.get("seed")


def _objective_block(review: Mapping[str, Any], metric: str) -> dict[str, Any]:
    obj = _as_record(review.get("objective_check"))
    block = _as_record(obj.get(metric))
    if block:
        return block
    for key in ("APS_lowlight", "APS"):
        block = _as_record(obj.get(key))
        if block:
            return block
    for value in obj.values():
        if isinstance(value, Mapping):
            return dict(value)
    return {}


_STATUS_ZH = {
    "queued": "排队中",
    "running": "运行中",
    "waiting_gpu": "训练中",
    "pause_requested": "即将暂停",
    "paused": "已暂停",
    "completed": "已结束",
    "blocked": "已拦住",
    "failed": "失败",
}

_HOW_ZH = {
    "F0": "只用 RGB，不融合热红外",
    "F1": "RGB + 热红外早期拼接",
    "F3": "门控多尺度融合（更重）",
    "N0": "标准 neck",
    "N1": "FDPN neck",
    "A4": "早期拼接 + FDPN neck",
}

_DECISION_ZH = {
    "KEEP": "先留下这条设定",
    "REPLICATE": "换 seed 再复现一轮",
    "DISCARD": "丢掉这轮，不继续这条",
    "PROBE": "证据不够，继续探",
}


def _how_label(how_id: Any) -> str:
    token = str(how_id or "").strip()
    if not token or token == "—":
        return "—"
    gloss = _HOW_ZH.get(token.upper())
    return f"{token} · {gloss}" if gloss else token


def _decision_label(decision: Any) -> str:
    token = str(decision or "").strip()
    if not token:
        return "—"
    gloss = _DECISION_ZH.get(token.upper())
    return f"{token} · {gloss}" if gloss else token


def _question(campaign: Mapping[str, Any], protocol: Mapping[str, Any] | None) -> str:
    proto = _as_record(protocol)
    title = str(proto.get("title") or campaign.get("experiment_title") or "").strip()
    notes = str(_as_record(proto.get("goal")).get("notes") or "").strip()
    return title or notes or "—"


def _did_sentence(
    campaign: Mapping[str, Any],
    protocol: Mapping[str, Any] | None,
    rounds: list[Mapping[str, Any]],
    metric: str,
) -> str:
    n = int(campaign.get("gpu_rounds") or 0)
    baseline = _as_record(_as_record(protocol).get("baseline"))
    model = str(baseline.get("model") or campaign.get("adapter") or "D-FINE")
    dataset = str(
        baseline.get("dataset") or campaign.get("dataset_id") or "rgbt_tiny_v1"
    ).replace("dataset:", "")
    hows: list[str] = []
    seeds: list[str] = []
    for row in rounds:
        hid = str(row.get("how_id") or "").strip()
        if hid and hid != "—" and _how_label(hid) not in hows:
            hows.append(_how_label(hid))
        seed = row.get("seed")
        if seed is not None and str(seed) not in {"", "—"} and str(seed) not in seeds:
            seeds.append(str(seed))
    if n == 0:
        return (
            f"还没有开训。协议冻结的模型是 {model}，数据是 {dataset}，主指标 {metric}。"
        )
    how_txt = "、".join(hows) if hows else "—"
    seed_txt = "、".join(seeds) if seeds else "—"
    return (
        f"用 {model} 在 {dataset} 上真实 GPU 训练了 {n} 轮。"
        f"方法是 {how_txt}。"
        f"随机种子依次为 {seed_txt}。"
        f"主指标 {metric}。没有换数据集、没有换评测器。"
    )


def build_campaign_notebook(
    work_dir: Path | str,
    campaign: Mapping[str, Any],
    *,
    project_root: Path | str | None = None,
) -> dict[str, Any]:
    work = Path(work_dir)
    root = Path(project_root) if project_root is not None else work.parents[2]
    protocol = _as_record(_load_json(work / "protocol.json"))
    plan = _as_record(_load_json(work / "plan.json"))
    audit = _as_record(_load_json(work / "run" / "rgbt_pair_audit.json"))
    subset = _as_record(_load_json(work / "run" / "dfine_subset.json"))
    metrics = _as_record(_load_json(work / "run" / "metrics.json"))
    dataset_id = str(
        _as_record(protocol.get("baseline")).get("dataset") or DEFAULT_DATASET_ID
    ).replace("dataset:", "")

    train = _as_record(_as_record(audit.get("splits")).get("train"))
    sample = _as_record(train.get("sample"))
    training = _as_record(metrics.get("training"))
    dataset_card = {
        "dataset_id": dataset_id,
        "title": "RGBT-Tiny（RGB + 热红外配对）",
        "n_rgb_train": train.get("n_rgb"),
        "n_thermal_train": train.get("n_thermal"),
        "n_paired_train": train.get("n_paired"),
        "thermal_mode": sample.get("thermal_mode") or "L",
        "rgb_mode": sample.get("rgb_mode") or "RGB",
        "staging_mode": subset.get("staging_mode"),
        "fusion_method": subset.get("fusion_method") or training.get("fusion_method"),
        "fusion_applied_module": training.get("fusion_applied"),
        "note": (
            "热红外是单通道灰度，不是第二路彩色相机。"
            "F1 early_concat 把 RGB 与 thermal 各 50% 混合成一张 3 通道图再训练，"
            "所以缓存图看起来仍是彩色。请看下方左右对照。"
        ),
    }
    try:
        resolve_processed_root(root, dataset_id)
        previews = preview_manifest(root, dataset_id, split="train", limit=3)
    except FileNotFoundError:
        previews = {"dataset_id": dataset_id, "items": [], "count": 0}

    pending = _as_record(_load_json(work / "how_pending.json"))
    story = build_campaign_story(
        campaign, protocol=protocol, seed_plan=plan, how_pending=pending
    )
    return {
        "dataset": dataset_card,
        "previews": previews,
        **story,
    }


def _max_rounds(protocol: Mapping[str, Any] | None, campaign: Mapping[str, Any]) -> int | None:
    stop = _as_record(_as_record(protocol).get("stop_rules"))
    raw = stop.get("max_rounds")
    if raw is not None:
        try:
            return int(raw)
        except (TypeError, ValueError):
            pass
    extra = campaign.get("max_extra_rounds")
    if extra is not None:
        try:
            return int(extra) + 1
        except (TypeError, ValueError):
            pass
    return None


def _round_budget_exhausted(
    campaign: Mapping[str, Any],
    protocol: Mapping[str, Any] | None = None,
) -> bool:
    """True when Manager stopped because round budget is used up (not a hang)."""
    status = str(campaign.get("status") or "")
    if status not in {"completed", "paused", "failed", "blocked"}:
        return False
    # Planner/Reviewer NEED_HUMAN is not a budget stop — even if an older
    # NEXT_ROUND step still says max_extra_rounds (stale after extend).
    if str(campaign.get("last_action") or "") == "NEED_HUMAN":
        return False
    max_rounds = _max_rounds(protocol, campaign)
    try:
        gpu = int(campaign.get("gpu_rounds") or 0)
    except (TypeError, ValueError):
        gpu = 0
    for step in reversed(_as_list(campaign.get("steps"))):
        row = _as_record(step)
        action = str(row.get("action") or "")
        if action == "NEED_HUMAN" and row.get("idle"):
            return False
        if action != "NEXT_ROUND":
            continue
        blob = " ".join(str(x) for x in _as_list(row.get("reasons")))
        # Campaign max_extra_rounds can stop before protocol.max_rounds catches up
        # (e.g. gpu=18 < max_rounds=21). Still treat as budget stop so UI shows
        #「提高额度并续跑」.
        if "max_extra_rounds" in blob:
            return True
        if "stop_rules" in blob:
            if max_rounds is not None and gpu >= max_rounds:
                return True
            # Extended protocol max but GPU still under it → not exhausted yet.
            if max_rounds is not None and gpu < max_rounds:
                return False
            return True
        break
    return max_rounds is not None and gpu >= max_rounds


def _scout_block_reason(how_pending: Mapping[str, Any] | None) -> str | None:
    scout = _as_record(_as_record(how_pending).get("scout"))
    if not scout:
        return None
    err = str(scout.get("error") or scout.get("note") or "")
    if not (scout.get("fail_closed") or err):
        return None
    lower = err.lower()
    if "429" in err or "rate limited" in lower:
        return "文献检索被限流（429），fail-closed。没有新 HOW。不允许凭记忆发明方法。"
    short = err.split("{", 1)[0].strip() or err[:160]
    return f"文献检索失败（{short}），没有新 HOW。"


def _how_new_cards(how_pending: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for raw in _as_list(_as_record(how_pending).get("candidates")):
        row = _as_record(raw)
        hid = str(row.get("how_id") or "").strip()
        if not hid:
            continue
        cards.append(
            {
                "candidate_id": row.get("candidate_id"),
                "how_id": hid,
                "how_label": _how_label(hid),
                "status": row.get("status"),
                "family": row.get("family"),
                "mechanism": row.get("mechanism") or row.get("implementation_intent") or "",
                "smoke_ok": bool(row.get("smoke_ok")),
            }
        )
    return cards


def _how_origin(
    plan: Mapping[str, Any],
    campaign: Mapping[str, Any],
    how_id: Any,
) -> tuple[str, str]:
    """Honest label for where a round's HOW came from. Not a Claim."""
    hid = str(how_id or "").strip().upper()
    if hid and hid not in ALLOWED_HOW:
        return "plugin_overlay", "插件 overlay（非目录 preset）"
    blob = _as_record(plan)
    if bool(blob.get("bootstrap")):
        return "protocol_seed", "协议冻结 seed，不是 LLM 发现"
    action = str(_as_record(blob.get("decision_summary")).get("selected_action") or "")
    if action.startswith("contrast_"):
        return "rules_contrast", "规则对照轮换（F1↔F3↔F0）"
    trace = _as_record(blob.get("llm_trace"))
    backend = str(trace.get("backend") or campaign.get("planner_backend") or "rules").lower()
    if backend == "llm":
        return "llm_catalog_pick", "LLM 从固定目录里选"
    return "rules_catalog", "规则 Planner 从固定目录里选"


def _plugin_truth(how_pending: Mapping[str, Any] | None) -> dict[str, Any]:
    candidates = [_as_record(row) for row in _as_list(_as_record(how_pending).get("candidates"))]
    registered = [row for row in candidates if str(row.get("status") or "") == "registered"]
    proposed = [row for row in candidates if str(row.get("status") or "") in {"proposed", "approved_pending_adapter"}]
    authored = [
        row
        for row in candidates
        if row.get("plugin_path") or row.get("authored_at") or row.get("smoke_ok")
    ]
    return {
        "draft_count": len(proposed),
        "registered_count": len(registered),
        "authored_count": len(authored),
        "draft_how_ids": [str(row.get("how_id") or "") for row in proposed if row.get("how_id")],
        "registered_how_ids": [str(row.get("how_id") or "") for row in registered if row.get("how_id")],
    }


def build_experiment_truth(
    campaign: Mapping[str, Any],
    rounds: list[Mapping[str, Any]],
    *,
    protocol: Mapping[str, Any] | None = None,
    seed_plan: Mapping[str, Any] | None = None,
    how_pending: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """What this experiment actually is — and is not. Not a Claim."""
    proto = _as_record(protocol)
    baseline = _as_record(proto.get("baseline"))
    primary = _as_record(_as_record(proto.get("objective")).get("primary"))
    metric = str(primary.get("metric") or "APS_lowlight")
    # Protocol seed HOW = first reviewed/started round, not the live plan.json
    # (live plan is "current"; labeling it seed F1/P3A confuses the loop UI).
    first_round_how = ""
    for row in rounds:
        hid = str(row.get("how_id") or "").strip()
        if hid and hid != "—":
            first_round_how = hid
            break
    seed_how = first_round_how or _how(seed_plan or {}) or str(campaign.get("seed_plan_id") or "")
    used_hows = {str(row.get("how_id") or "").strip().upper() for row in rounds} - {"", "—"}
    plugin = _plugin_truth(how_pending)

    catalog_scope: list[dict[str, Any]] = []
    for hid in sorted(ALLOWED_HOW):
        spec = ALLOWED_HOW[hid]
        catalog_scope.append(
            {
                "how_id": hid,
                "how_label": _how_label(hid),
                "family": spec.get("family"),
                "input_mode": spec.get("input_mode"),
                "fusion_method": spec.get("fusion_method"),
                "used_in_run": hid in used_hows,
            }
        )

    planner = str(campaign.get("planner_backend") or "rules").lower()
    lifecycle = bool(campaign.get("llm_how_lifecycle"))
    origins = {str(row.get("how_origin") or "") for row in rounds} - {""}
    all_presets = not origins or origins <= {
        "protocol_seed",
        "rules_contrast",
        "rules_catalog",
        "llm_catalog_pick",
        "catalog_preset",
    }
    fusion_used = sorted(h for h in used_hows if h in {"F0", "F1", "F3"})

    truth_lines: list[str] = [
        f"登记实验 {campaign.get('experiment_id') or '—'}。"
        f"GPU 轮次只能在 {CATALOG_ID} 的 {len(ALLOWED_HOW)} 个 preset HOW 里切换，"
        "或接入已通过 smoke 的插件；不能改 D-FINE 主干。",
    ]
    if planner == "rules":
        truth_lines.append(
            "Planner 后端是 rules：F1/F3/F0 会按内置 contrast 规则轮换，"
            "不是 LLM 自由探索新架构。"
        )
    else:
        invent_on = bool(
            campaign.get("llm_may_invent_how")
            or (how_pending or {}).get("llm_may_invent_how")
        )
        if invent_on:
            truth_lines.append(
                "Planner 后端是 LLM；Stage B 已翻 llm_may_invent_how："
                "可经 invent-fallback 提出新插件 HOW 草稿（人审后 Diff 写代码），"
                "Plan 仍不得夹 Python；KEEP ≠ Claim。"
            )
        else:
            truth_lines.append(
                "Planner 后端是 LLM，但仍只能从上述目录（+插件）里选 HOW，"
                "llm_may_invent_how=false，不能凭空发明算子。"
            )
    if plugin["authored_count"] == 0 and plugin["registered_count"] == 0:
        truth_lines.append("本实验还没有 LLM 写出并成功接入的 plugin.py。")
    elif plugin["registered_count"] == 0:
        truth_lines.append(
            f"有 {plugin['draft_count']} 个 HOW 草稿，"
            f"已写代码 {plugin['authored_count']} 个，但尚未注册进目录、未上 GPU。"
        )
    else:
        truth_lines.append(
            f"已有 {plugin['registered_count']} 个插件 HOW 进目录："
            + "、".join(plugin["registered_how_ids"])
            + "。"
        )
    if len(rounds) >= 2 and fusion_used and len(fusion_used) <= 3:
        truth_lines.append(
            f"已跑 {len(rounds)} 轮，主要在 fusion preset（{' / '.join(fusion_used)}）间切换；"
            "这是 Adapter 旋钮对照，不是新检测器。"
        )
    if all_presets and plugin["registered_count"] == 0:
        truth_lines.append("目前每一轮都是目录 preset，没有新代码路径。")

    return {
        "identity": {
            "experiment_id": campaign.get("experiment_id"),
            "experiment_title": campaign.get("experiment_title"),
            "protocol_id": campaign.get("protocol_id") or proto.get("protocol_id"),
            "catalog_id": campaign.get("catalog_id") or CATALOG_ID,
            "adapter": campaign.get("adapter") or baseline.get("adapter"),
            "dataset_id": str(campaign.get("dataset_id") or baseline.get("dataset") or "").replace(
                "dataset:", ""
            ),
            "slice_id": campaign.get("slice_id"),
            "primary_metric": metric,
            "seed_how_id": seed_how or None,
            "protocol_fingerprint": campaign.get("protocol_fingerprint"),
        },
        "catalog_scope": catalog_scope,
        "capability": {
            "planner_backend": planner,
            "reviewer_backend": str(campaign.get("reviewer_backend") or "rules").lower(),
            "llm_how_lifecycle": lifecycle,
            "llm_may_invent_how": bool(
                campaign.get("llm_may_invent_how")
                or (how_pending or {}).get("llm_may_invent_how")
            ),
            "all_rounds_catalog_presets": all_presets and plugin["registered_count"] == 0,
            "plugin": plugin,
        },
        "truth_summary": truth_lines,
        "not_this": [
            "不是换 D-FINE 主干或训练框架",
            "不是自动发论文或 Claim",
            "Scout 文献 ≠ 可执行新 HOW",
            "KEEP ≠ Claim",
        ],
    }


def build_llm_judgment(
    campaign: Mapping[str, Any],
    *,
    how_pending: Mapping[str, Any] | None = None,
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Latest Planner / Reviewer / Scout reasoning for the main loop UI. Not a Claim."""
    steps = _as_list(campaign.get("steps"))
    last_plan_step: Mapping[str, Any] | None = None
    last_plan: dict[str, Any] = {}
    last_review_report: dict[str, Any] = {}

    for raw in reversed(steps):
        step = _as_record(raw)
        action = str(step.get("action") or "")
        report = _as_record(step.get("report"))
        if last_plan_step is None and action in {"NEED_PLAN", "NEXT_ROUND"}:
            plan = _as_record(report.get("plan"))
            if plan:
                last_plan_step = step
                last_plan = dict(plan)
        if not last_review_report and action == "NEED_REVIEW":
            last_review_report = dict(report)
        if last_plan_step is not None and last_review_report:
            break

    review = _as_record(last_review_report.get("review"))
    semantic = _as_record(last_review_report.get("semantic_proposal"))
    decision_summary = _as_record(last_plan.get("decision_summary"))
    if not decision_summary:
        decision_summary = _as_record(_as_record(last_plan.get("llm_trace")).get("decision_summary"))

    selected_how = str(last_plan.get("how_id") or "").strip()
    eval_block = _as_record(last_plan.get("evaluation"))
    seeds = _as_list(eval_block.get("seeds"))
    selected_seed = seeds[0] if seeds else last_plan.get("seed")

    candidates: list[dict[str, Any]] = []
    for item in _as_list(last_plan.get("candidate_experiments")):
        row = _as_record(item)
        hid = str(row.get("how_id") or "").strip()
        if not hid:
            continue
        candidates.append(
            {
                "how_id": hid,
                "how_label": _how_label(hid),
                "reason_not_selected": str(row.get("reason_not_selected") or ""),
                "selected": hid == selected_how,
            }
        )

    scout = _as_record(_as_record(how_pending).get("scout"))
    scout_status = "empty"
    if scout.get("fail_closed") or scout.get("error"):
        scout_status = "fail_closed"
    elif scout.get("query") or scout.get("papers"):
        scout_status = "ok"

    last_action = str(campaign.get("last_action") or "")
    status = str(campaign.get("status") or "")
    n = int(campaign.get("gpu_rounds") or 0)
    priority = str(semantic.get("next_research_priority") or "").strip()
    resume_hint: dict[str, Any] | None = None
    if status == "paused" or campaign.get("orphan_reclaimed") or (
        status == "completed" and last_action == "NEED_HUMAN"
    ):
        if last_action == "NEED_HUMAN":
            err = str(campaign.get("error") or "").strip()
            resume_hint = {
                "next_manager_action": "NEXT_ROUND",
                "next_research_priority": priority or None,
                "summary": (
                    f"Planner 曾 fail-closed（{err or 'NEED_HUMAN'}）。"
                    f"续跑后 Manager 重新进入 NEXT_ROUND，规划第 {n + 1} 轮。"
                    + (f" Reviewer 建议：{priority}" if priority else "")
                ),
            }
        elif last_action == "NEED_MEMORY":
            resume_hint = {
                "next_manager_action": "NEXT_ROUND",
                "next_research_priority": priority or None,
                "summary": (
                    f"续跑后 Manager 进入 NEXT_ROUND，Planner 规划第 {n + 1} 轮。"
                    + (f" Reviewer 建议：{priority}" if priority else "")
                ),
            }
        elif last_action in {"NEED_PLAN", "NEXT_ROUND"} and selected_how:
            hypo = str(last_plan.get("hypothesis") or "").strip()
            resume_hint = {
                "next_manager_action": "NEED_EXECUTION",
                "summary": (
                    f"续跑后下一轮可能是 {_how_label(selected_how)} seed {selected_seed or '—'}。"
                    + (f" 假设：{hypo}" if hypo else "")
                ),
            }
        elif last_action:
            resume_hint = {
                "next_manager_action": last_action,
                "summary": f"续跑后从 {last_action} 继续 Manager 状态机。",
            }

    alts: list[str] = []
    for item in _as_list(semantic.get("alternative_explanations")):
        if isinstance(item, str) and item.strip():
            alts.append(item.strip())
        elif isinstance(item, Mapping):
            alts.append(str(item.get("text") or item.get("explanation") or item))

    terminal_actions = {
        "STOP",
        "IDLE",
        "NOVELTY_EXHAUSTED",
        "PROTOCOL_AMENDMENT_REQUIRED",
    }
    can_resume = (
        (status == "paused" and last_action not in terminal_actions)
        or (
            last_action == "NEED_HUMAN"
            and status in {"completed", "failed", "blocked", "running"}
        )
        or bool(campaign.get("pending_harvest"))
        or (
            status in {"waiting_gpu", "running"}
            and last_action
            in {"NEED_EXECUTION", "NEED_GATE"}
            and bool(campaign.get("orphan_reclaimed"))
        )
        or (
            status == "failed"
            and str(campaign.get("error") or "").startswith("ReviewRefused:")
        )
        or (
            status == "failed"
            and last_action in {"NEXT_ROUND", "NEED_HUMAN"}
            and (
                "OSError" in str(campaign.get("error") or "")
                or "Invalid argument" in str(campaign.get("error") or "")
                or "Filename too long" in str(campaign.get("error") or "")
            )
        )
    )
    # Worker still draining (GPU finish / finally{}) — do not invite a resume click.
    if bool(campaign.get("worker_alive")) or status == "pause_requested":
        can_resume = False
    can_extend = _round_budget_exhausted(campaign, protocol)

    semantic_block = None
    if semantic:
        semantic_block = {
            "observation": semantic.get("observation"),
            "interpretation": semantic.get("interpretation"),
            "next_research_priority": semantic.get("next_research_priority"),
            "alternative_explanations": alts,
            "hypothesis_status": semantic.get("hypothesis_status"),
        }

    return {
        "planner_backend": campaign.get("planner_backend"),
        "reviewer_backend": campaign.get("reviewer_backend"),
        "planner": {
            "hypothesis": str(last_plan.get("hypothesis") or ""),
            "selected_how_id": selected_how or None,
            "selected_how_label": _how_label(selected_how) if selected_how else None,
            "selected_seed": selected_seed,
            "decision_summary": decision_summary or None,
            "verification_plan": _as_record(last_plan.get("verification_plan")) or None,
            "candidates": candidates,
            "source_action": str(_as_record(last_plan_step).get("action") or "") or None,
            "bootstrap": bool(last_plan.get("bootstrap")),
        },
        "reviewer": {
            "review_decision": str(review.get("review_decision") or "") or None,
            "reasoning": str(
                review.get("reasoning_summary") or review.get("reason") or ""
            ).strip()
            or None,
            "semantic_proposal": semantic_block,
        },
        "scout": {
            "status": scout_status,
            "query": scout.get("query"),
            "source": scout.get("source"),
            "error": scout.get("error"),
            "papers_count": len(_as_list(scout.get("papers"))),
            "fail_closed": bool(scout.get("fail_closed")),
        },
        "resume_hint": resume_hint,
        "can_resume": can_resume,
        "can_extend": can_extend,
    }


def build_now_board(
    campaign: Mapping[str, Any],
    rounds: list[Mapping[str, Any]],
    *,
    protocol: Mapping[str, Any] | None = None,
    how_pending: Mapping[str, Any] | None = None,
    llm_judgment: Mapping[str, Any] | None = None,
    active_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Where the campaign is, what a human should do, where HOW lives. Not a Claim."""
    status = str(campaign.get("status") or "")
    n = int(campaign.get("gpu_rounds") or 0)
    max_rounds = _max_rounds(protocol, campaign)
    last = _as_record(rounds[-1] if rounds else {})
    last_action = str(campaign.get("last_action") or "")
    plan = _as_record(active_plan)
    plan_how = _how(plan)
    # Scoreboard lags: NEXT_ROUND/NEED_GATE may already point at the next HOW
    # while rounds[-1] is still the previous reviewed preset (e.g. F1/F3).
    # Include paused: API restart / stop often leaves NEED_EXECUTION + plan.json
    # ahead of the scoreboard while status flips to paused.
    in_flight = last_action in {
        "NEED_EXECUTION",
        "NEED_GATE",
        "NEED_MATERIALIZE",
        "NEED_PARSE",
        "NEED_EVIDENCE_CHECK",
        "NEED_REVIEW",
        "NEXT_ROUND",
        "NEED_PLAN",
    } and status not in {"completed", "failed", "blocked"}
    display = last
    if plan_how and in_flight and (
        str(last.get("how_id") or "") != plan_how
        or str(last.get("status") or "") != "training"
    ):
        display = {
            "how_id": plan_how,
            "how_label": _how_label(plan_how),
            "seed": _seed_of(plan),
            "status": "training",
            "decision": None,
            "decision_label": None,
        }
    how_new = _how_new_cards(how_pending)
    training = (
        str(display.get("status") or "") == "training"
        or status == "waiting_gpu"
        or last_action in {"NEED_EXECUTION", "NEED_GATE", "NEED_MATERIALIZE"}
    )

    if n == 0 and not training:
        round_label = "还没开第 1 轮"
        round_kind = "idle"
    elif training:
        # Gate/materialize for the next HOW happens before gpu_rounds increments.
        shown = n + 1 if (
            plan_how
            and last_action in {"NEED_GATE", "NEED_MATERIALIZE", "NEXT_ROUND", "NEED_PLAN"}
            and str(last.get("how_id") or "") != plan_how
        ) else max(n, 1)
        round_label = f"第 {shown} 轮进行中"
        round_kind = "running"
    else:
        round_label = f"第 {n} 轮已跑完"
        round_kind = "completed"

    need_human = False
    next_kind = "idle"
    if status == "waiting_gpu" or training:
        next_kind = "training"
        shown = n + 1 if (
            plan_how
            and last_action in {"NEED_GATE", "NEED_MATERIALIZE", "NEXT_ROUND", "NEED_PLAN"}
            and str(last.get("how_id") or "") != plan_how
        ) else max(n, 1)
        next_text = (
            f"第 {shown} 轮正在 GPU 训练 {display.get('how_label') or _how_label(display.get('how_id'))}"
            f"（seed {display.get('seed') or '—'}）。人这边不用选 HOW。"
        )
    elif status == "completed" and _round_budget_exhausted(campaign, protocol):
        next_kind = "done"
        next_text = (
            f"已跑满额度（约 {n}/{max_rounds or n} 轮）。"
            "点「提高额度并续跑」才会继续；不会自动开下一轮。"
        )
    elif status == "pause_requested" or (
        bool(campaign.get("worker_alive"))
        and status in {"paused", "running", "waiting_gpu", "pause_requested"}
    ):
        next_kind = "paused"
        nxt = n + 1 if n else 1
        next_text = (
            f"正在收尾暂停（worker 仍在退出或 GPU 轮次收割中），第 {nxt} 轮还不能续跑。"
            "等状态变为「已暂停」且 GPU 空闲后再点「继续实验」。"
        )
        if campaign.get("error"):
            next_text += f" 备注：{campaign.get('error')}"
    elif status == "paused" or (
        campaign.get("orphan_reclaimed") and status not in {"completed", "failed", "blocked"}
    ):
        next_kind = "paused"
        nxt = n + 1 if n else 1
        hint = _as_record(_as_record(llm_judgment).get("resume_hint"))
        next_text = (
            f"实验已暂停，GPU 空着，第 {nxt} 轮不会自动开始。"
            "点「继续实验」才会续跑。"
        )
        if hint.get("summary"):
            next_text += f" {hint['summary']}"
        elif campaign.get("error"):
            next_text += f" 原因：{campaign.get('error')}"
    elif status == "completed" and last_action == "NEED_HUMAN":
        next_kind = "paused"
        hint = _as_record(_as_record(llm_judgment).get("resume_hint"))
        next_text = (
            f"Planner 第 {n + 1} 轮 fail-closed，GPU 空着。"
            "点「继续实验」可重新规划下一轮（不会自动续跑）。"
        )
        if hint.get("summary"):
            next_text += f" {hint['summary']}"
        elif campaign.get("error"):
            next_text += f" 原因：{campaign.get('error')}"
    elif status == "completed":
        next_kind = "done"
        next_text = "这场实验已经结束。没有下一轮。"
    elif status in {"failed", "blocked"}:
        next_kind = "blocked"
        next_text = str(campaign.get("error") or "实验被拦住，不会继续训练。")
    elif last_action == "NEED_HUMAN" or how_new:
        pending_ids = [
            str(row.get("how_id"))
            for row in how_new
            if row.get("status") in {"proposed", "approved_pending_adapter", None, ""}
        ]
        if pending_ids or last_action == "NEED_HUMAN":
            need_human = True
            next_kind = "need_human"
            if pending_ids:
                next_text = (
                    "有新 HOW 草稿等人审："
                    + "、".join(pending_ids)
                    + "。批准进目录之后，下一轮才可能选中它。文献不能进 ClaimGate。"
                )
            else:
                next_text = "卡住等人处理。看下方原因，不是训练挂死。"
        elif last_action == "NEED_MEMORY":
            next_kind = "plan_next"
            next_text = (
                f"第 {n} 轮已写完记忆。若 worker 还在跑，下一步是规划第 {n + 1} 轮的 HOW / seed。"
            )
        elif last_action in {"NEED_PLAN", "NEXT_ROUND"}:
            next_kind = "planning"
            next_text = "Planner 正在选下一轮的 HOW 和 seed。还没有开训。"
        else:
            next_text = _STATUS_ZH.get(status, status or "未知") + "。"
    elif last_action == "NEED_MEMORY":
        next_kind = "plan_next"
        next_text = f"第 {n} 轮已写完记忆。若 worker 还在跑，下一步是规划第 {n + 1} 轮的 HOW / seed。"
    elif last_action in {"NEED_PLAN", "NEXT_ROUND"}:
        next_kind = "planning"
        next_text = "Planner 正在选下一轮的 HOW 和 seed。还没有开训。"
    elif last_action == "NEED_EXECUTION":
        next_kind = "training"
        next_text = f"第 {n or 1} 轮即将或正在 GPU 训练。"
    elif last_action == "NEED_PARSE":
        next_kind = "reading"
        next_text = "训练结束，正在读主指标。还没有审阅。"
    elif last_action == "NEED_REVIEW":
        next_kind = "reviewing"
        next_text = "Reviewer 正在判 KEEP / REPLICATE / DISCARD。KEEP ≠ Claim。"
    else:
        next_text = _STATUS_ZH.get(status, status or "未知") + "。"

    used: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rounds:
        hid = str(row.get("how_id") or "").strip()
        if not hid or hid in seen:
            continue
        seen.add(hid)
        origin = str(row.get("how_origin") or "catalog_preset")
        used.append(
            {
                "how_id": hid,
                "how_label": row.get("how_label") or _how_label(hid),
                "source": origin,
                "source_label": row.get("how_origin_label") or origin,
                "rounds": [
                    r.get("round_index")
                    for r in rounds
                    if str(r.get("how_id") or "") == hid
                ],
            }
        )

    how_new_reason = None
    if not how_new:
        how_new_reason = _scout_block_reason(how_pending) or (
            "没有新 HOW 草稿。当前只用目录里已登记的方法。"
        )

    return {
        "round_index": n,
        "max_rounds": max_rounds,
        "round_label": round_label,
        "round_kind": round_kind,
        "current_how_id": display.get("how_id"),
        "current_how_label": display.get("how_label") or _how_label(display.get("how_id")),
        "current_seed": display.get("seed"),
        "current_decision": display.get("decision_label") or display.get("decision"),
        "next_kind": next_kind,
        "next_text": next_text,
        "need_human": need_human,
        "how_in_use": used,
        "how_new": how_new,
        "how_new_reason": how_new_reason,
        "can_resume": bool(_as_record(llm_judgment).get("can_resume")),
        "can_extend": bool(_as_record(llm_judgment).get("can_extend")),
        "resume_summary": _as_record(_as_record(llm_judgment).get("resume_hint")).get("summary"),
    }


def build_campaign_story(
    campaign: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any] | None = None,
    seed_plan: Mapping[str, Any] | None = None,
    how_pending: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Round scoreboard + one-sentence headline. Not a Claim."""
    metric = _primary_metric(protocol)
    rounds = build_round_cards(campaign, metric=metric, seed_plan=seed_plan)
    llm_judgment = build_llm_judgment(
        campaign, how_pending=how_pending, protocol=protocol
    )
    experiment_truth = build_experiment_truth(
        campaign,
        rounds,
        protocol=protocol,
        seed_plan=seed_plan,
        how_pending=how_pending,
    )
    from scientist_lab.services.sota_pursuit import build_sota_board

    baseline_path = None
    if isinstance(campaign.get("work_dir"), str):
        baseline_path = Path(str(campaign["work_dir"])) / "baseline_metrics.json"
    baseline = {}
    if baseline_path and baseline_path.is_file():
        baseline = _load_json(baseline_path) or {}
    sota_board = build_sota_board(campaign, protocol=protocol, baseline=baseline)
    from scientist_lab.services.evaluation_matrix import build_evaluation_matrix
    from scientist_lab.services.registered_experiments import RegisteredExperimentService

    reg_items = []
    if isinstance(campaign.get("work_dir"), str):
        try:
            root = Path(str(campaign["work_dir"])).parents[2]
            reg_items = RegisteredExperimentService(root).list().get("items") or []
        except (OSError, ValueError, IndexError):
            reg_items = []
    evaluation_matrix = build_evaluation_matrix(
        campaign,
        protocol=protocol,
        rounds=rounds,
        registered_experiments=reg_items,
    )
    from scientist_lab.services.evaluation_matrix import build_live_experiment_brief

    # Always rebuild from live rounds + campaign flags (e.g. evidence_gap_priority).
    # Stale campaign["live_m1_brief"] must not pin an old plugin PRIORITY mandate.
    live_m1_brief = build_live_experiment_brief(
        campaign,
        protocol=protocol,
        rounds=rounds,
        registered_experiments=reg_items,
        how_pending=how_pending,
    )
    return {
        "primary_metric": metric,
        "question": _question(campaign, protocol),
        "did": _did_sentence(campaign, protocol, rounds, metric),
        "rounds": rounds,
        "headline": build_headline(campaign, rounds, metric),
        "lab_log": build_lab_log(campaign, seed_plan=seed_plan, metric=metric),
        "llm_judgment": llm_judgment,
        "experiment_truth": experiment_truth,
        "sota_board": sota_board,
        "evaluation_matrix": evaluation_matrix,
        "live_m1_brief": dict(live_m1_brief) if isinstance(live_m1_brief, Mapping) else {},
        "now": build_now_board(
            campaign,
            rounds,
            protocol=protocol,
            how_pending=how_pending,
            llm_judgment=llm_judgment,
            active_plan=seed_plan,
        ),
    }


def build_round_cards(
    campaign: Mapping[str, Any],
    *,
    metric: str,
    seed_plan: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rounds: list[dict[str, Any]] = []
    pending: dict[str, Any] = {
        "how_id": _how(seed_plan or {}),
        "seed": _seed_of(seed_plan or {}),
        "hypothesis": str((seed_plan or {}).get("hypothesis") or ""),
        "bootstrap": bool((seed_plan or {}).get("bootstrap")),
        "plan": dict(seed_plan or {}),
    }
    gpu_round = 0

    def _flush_if_started() -> None:
        rid = pending.get("round_index")
        if rid and not any(row.get("round_index") == rid for row in rounds):
            rounds.append(dict(pending))

    for raw in _as_list(campaign.get("steps")):
        step = _as_record(raw)
        action = str(step.get("action") or "")
        report = _as_record(step.get("report"))
        if action in {"NEED_PLAN", "NEXT_ROUND"}:
            plan = _as_record(report.get("plan"))
            if plan and not (action == "NEXT_ROUND" and step.get("idle")):
                pending["how_id"] = _how(plan) or pending.get("how_id")
                pending["seed"] = _seed_of(plan)
                pending["hypothesis"] = str(plan.get("hypothesis") or pending.get("hypothesis") or "")
                pending["bootstrap"] = bool(plan.get("bootstrap"))
                pending["plan"] = dict(plan)
        elif action == "NEED_EXECUTION":
            _flush_if_started()
            gpu_round += 1
            handle = _as_record(report.get("handle"))
            how_id = pending.get("how_id") or "—"
            plan_blob = _as_record(pending.get("plan"))
            origin, origin_label = _how_origin(plan_blob, campaign, how_id)
            pending = {
                **pending,
                "round_index": gpu_round,
                "run_id": handle.get("run_id"),
                "status": "trained" if handle.get("status") == "completed" else "training",
                "metric": metric,
                "value": None,
                "delta": None,
                "decision": None,
                "how_label": _how_label(how_id),
                "how_origin": origin,
                "how_origin_label": origin_label,
                "decision_label": "—",
                "summary": f"{_how_label(how_id)} · seed {pending.get('seed') or '—'} · {origin_label}",
            }
        elif action == "NEED_PARSE":
            mets = _as_record(_as_record(report.get("result")).get("metrics"))
            value = _metric_from_map(mets, metric)
            pending["value"] = value
            pending["metric"] = metric
        elif action == "NEED_REVIEW":
            review = _as_record(report.get("review"))
            obj = _objective_block(review, metric)
            pending["decision"] = str(review.get("review_decision") or "")
            pending["delta"] = obj.get("delta")
            if pending.get("value") is None:
                pending["value"] = obj.get("current")
            claim = _as_record(report.get("claim_gate"))
            pending["claim_status"] = claim.get("status")
            pending["scientific_outcome"] = claim.get("scientific_outcome")
            how_id = pending.get("how_id") or "—"
            seed = pending.get("seed") or "—"
            value = pending.get("value")
            decision = pending.get("decision") or "—"
            pending["how_label"] = _how_label(how_id)
            pending["decision_label"] = _decision_label(decision)
            pending["summary"] = (
                f"第 {pending.get('round_index') or gpu_round} 轮 {_how_label(how_id)} seed {seed}："
                f"{metric}={_fmt(value)}，审阅 {_decision_label(decision)}。"
            )
            pending["status"] = "reviewed"
            _flush_if_started()
            pending = {
                "how_id": how_id,
                "seed": seed,
                "hypothesis": pending.get("hypothesis") or "",
                "bootstrap": False,
            }
    if pending.get("round_index") and not any(
        row.get("round_index") == pending.get("round_index") for row in rounds
    ):
        rounds.append(dict(pending))
    if rounds:
        last = rounds[-1]
        last["is_current"] = True
        last["badge"] = "进行中" if str(last.get("status") or "") == "training" else "最近"
    return rounds


def build_headline(
    campaign: Mapping[str, Any],
    rounds: list[Mapping[str, Any]],
    metric: str,
) -> str:
    status = _STATUS_ZH.get(str(campaign.get("status") or ""), str(campaign.get("status") or "未知"))
    n = int(campaign.get("gpu_rounds") or 0)
    if not rounds and n == 0:
        return f"{status}。还没有跑完一轮 GPU。"
    hows: list[str] = []
    for row in rounds:
        hid = str(row.get("how_id") or "").strip()
        if hid and hid not in hows and hid != "—":
            hows.append(hid)
    last = _as_record(rounds[-1] if rounds else {})
    bits = [status, f"已跑 {n} 轮 GPU"]
    if hows:
        bits.append("HOW " + " / ".join(hows))
    if last.get("value") is not None:
        bits.append(f"最近 {metric}={_fmt(last.get('value'))}")
    if last.get("decision"):
        bits.append(f"最近一轮 {_decision_label(last.get('decision'))}")
    if campaign.get("orphan_reclaimed"):
        bits.append("进程重启后暂停，GPU 已空，没有续跑")
    elif str(campaign.get("last_action") or "") == "NEED_HUMAN":
        reasons = "；".join(str(x) for x in _as_list(campaign.get("error") and [campaign.get("error")]))
        last_step = None
        for raw in reversed(_as_list(campaign.get("steps"))):
            step = _as_record(raw)
            if str(step.get("action") or "") == "NEED_HUMAN":
                last_step = step
                break
        why = "；".join(str(x) for x in _as_list(_as_record(last_step).get("reasons")))
        bits.append(why or reasons or "Planner 被合同拦住")
    elif str(campaign.get("status") or "") == "paused" and campaign.get("error"):
        bits.append(str(campaign.get("error")))
    return "。".join(bits) + "。KEEP ≠ Claim。"


def build_lab_log(
    campaign: Mapping[str, Any],
    *,
    seed_plan: Mapping[str, Any] | None = None,
    metric: str | None = None,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    last_how = _how(seed_plan or {})
    last_value: float | None = None
    gpu_round = 0
    metric_name = metric or "APS_lowlight"
    for idx, raw in enumerate(_as_list(campaign.get("steps"))):
        step = _as_record(raw)
        action = str(step.get("action") or "")
        report = _as_record(step.get("report"))
        if action in {"NEED_PLAN", "NEXT_ROUND"}:
            plan = _as_record(report.get("plan"))
            if plan:
                last_how = _how(plan) or last_how
                rejected = _as_list(plan.get("candidate_experiments"))
                rejected_txt = "；".join(
                    f"{item.get('how_id')}：{item.get('reason_not_selected')}"
                    for item in rejected
                    if isinstance(item, Mapping)
                )
                hypothesis = str(plan.get("hypothesis") or "")
                body = hypothesis
                if rejected_txt:
                    body = f"{hypothesis} 未采用：{rejected_txt}"
                if action == "NEXT_ROUND" and step.get("idle"):
                    reasons = "；".join(str(x) for x in _as_list(step.get("reasons")))
                    body = reasons or "达到额外轮次上限，实验结束。"
                    entries.append(
                        _entry(
                            agent="manager",
                            title="Manager 停轮",
                            body=body,
                            how_id=last_how,
                            round_index=gpu_round,
                            step_index=idx,
                        )
                    )
                    continue
                title = (
                    "Planner 种子计划（协议冻结，不是 LLM 发现）"
                    if plan.get("bootstrap")
                    else "Planner 根据上一轮结果决定下一轮"
                )
                entries.append(
                    _entry(
                        agent="planner",
                        title=title,
                        body=body or f"下一轮 HOW={last_how}",
                        how_id=last_how,
                        round_index=gpu_round,
                        step_index=idx,
                        extra={
                            "rejected": [
                                {
                                    "how_id": item.get("how_id"),
                                    "reason": item.get("reason_not_selected"),
                                }
                                for item in rejected
                                if isinstance(item, Mapping)
                            ],
                            "source": _as_record(plan.get("llm_trace")).get("backend")
                            or campaign.get("planner_backend"),
                        },
                    )
                )
            elif action == "NEXT_ROUND" and step.get("idle"):
                reasons = "；".join(str(x) for x in _as_list(step.get("reasons")))
                entries.append(
                    _entry(
                        agent="manager",
                        title="Manager 停轮",
                        body=reasons or "达到额外轮次上限，实验结束。",
                        how_id=last_how,
                        round_index=gpu_round,
                        step_index=idx,
                    )
                )
        elif action == "NEED_EXECUTION":
            gpu_round += 1
            handle = _as_record(report.get("handle"))
            entries.append(
                _entry(
                    agent="executor",
                    title=f"Executor 开训 第 {gpu_round} 轮",
                    body=(
                        f"HOW={last_how}。真实 GPU，不伪造 metrics。"
                        f" run_id={handle.get('run_id') or '—'}"
                    ),
                    how_id=last_how,
                    round_index=gpu_round,
                    step_index=idx,
                )
            )
        elif action == "NEED_PARSE":
            result = _as_record(report.get("result"))
            mets = _as_record(result.get("metrics"))
            aps = _num(_metric_from_map(mets, metric_name))
            delta = None if last_value is None or aps is None else aps - last_value
            stable = None
            if delta is None:
                verdict = "这是本实验第一轮读数，还没有对照 Δ，不能判断稳住与否。"
            elif abs(delta) < 0.002:
                verdict = f"相对上一轮 Δ={_fmt(delta)}，波动很小。"
                stable = True
            elif delta < 0:
                verdict = (
                    f"相对上一轮 {metric_name} 从 {_fmt(last_value)} 降到 {_fmt(aps)}"
                    f"（Δ={_fmt(delta)}）。没有稳住。"
                )
                stable = False
            else:
                verdict = (
                    f"相对上一轮 {metric_name} 从 {_fmt(last_value)} 升到 {_fmt(aps)}"
                    f"（Δ={_fmt(delta)}）。仍是单 seed / 短训，不能当声称。"
                )
                stable = False
            last_value = aps if aps is not None else last_value
            entries.append(
                _entry(
                    agent="executor",
                    title=f"Executor 读数 第 {gpu_round} 轮",
                    body=f"HOW={last_how}  {metric_name}={_fmt(aps)}。{verdict}",
                    how_id=last_how,
                    round_index=gpu_round,
                    step_index=idx,
                    extra={
                        metric_name: aps,
                        "delta_vs_previous": delta,
                        "stable": stable,
                    },
                )
            )
        elif action == "NEED_REVIEW":
            review = _as_record(report.get("review"))
            decision = str(review.get("review_decision") or "—")
            obj = _objective_block(review, metric_name)
            delta = obj.get("delta")
            current = obj.get("current")
            reason = str(
                review.get("reasoning_summary")
                or review.get("reason")
                or ""
            )
            if decision == "REPLICATE" and (delta is None or delta == "None"):
                body = (
                    f"Reviewer 判 {decision}：对照 Δ 为空，基线还没接上，"
                    f"当前 {metric_name}={_fmt(current)}。"
                    "这不表示方法成功，只表示证据不够、需要再跑。"
                )
            else:
                body = (
                    f"Reviewer 判 {decision}。"
                    f" 当前 {metric_name}={_fmt(current)}，Δ={_fmt(delta)}。"
                    f" {reason}"
                ).strip()
            claim = _as_record(report.get("claim_gate"))
            entries.append(
                _entry(
                    agent="reviewer",
                    title="Reviewer 反馈",
                    body=body,
                    how_id=last_how,
                    round_index=gpu_round,
                    step_index=idx,
                    extra={
                        "review_decision": decision,
                        metric_name: current,
                        "delta": delta,
                        "claim_status": claim.get("status"),
                        "scientific_outcome": claim.get("scientific_outcome"),
                    },
                )
            )
        elif action == "NEED_MEMORY":
            written = _as_list(report.get("lessons_written"))
            entries.append(
                _entry(
                    agent="memory",
                    title="Memory 写入",
                    body="写入 lessons：" + ("、".join(str(x) for x in written) or "（无）")
                    + "。KEEP 仍不是 Claim。",
                    how_id=last_how,
                    round_index=gpu_round,
                    step_index=idx,
                )
            )
    return entries


def _entry(
    *,
    agent: str,
    title: str,
    body: str,
    how_id: str,
    round_index: int,
    step_index: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "agent": agent,
        "title": title,
        "body": body,
        "how_id": how_id,
        "round_index": round_index,
        "step_index": step_index,
    }
    if extra:
        row.update(extra)
    return row


def _load_json(path: Path) -> Any:
    if not path.is_file():
        return {}
    import json

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
