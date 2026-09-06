"""Workbench command-center chat. LLM talks; slash commands dispatch.

Does not write Memory, KEEP/DISCARD, or ClaimGate. GPU stays off unless
the user later confirms --execute on the loop page.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from scientist_lab.llm.errors import LLMError, MissingAPIKeyError, RealProviderNotEnabledError
from scientist_lab.llm.gateway import complete_chat
from scientist_lab.llm.models import LLMRequest
from scientist_lab.llm.runtime_secrets import apply_runtime_llm_env
from scientist_lab.services.local_run_inspector import inspect_local_run, list_local_runs

AGENT_NAME = "实验室助手"
AGENTS = ("planner", "executor", "reviewer")
AGENT_META: dict[str, dict[str, str]] = {
    "planner": {
        "id": "planner",
        "title": "规划 Agent",
        "role": "WHAT / WHY",
        "hint": "决定下一轮实验为什么这么做，不能绕过 Gate",
    },
    "executor": {
        "id": "executor",
        "title": "执行 Agent",
        "role": "HOW / RUN",
        "hint": "解释 Adapter / 训练进度，不能在聊天里点火 GPU",
    },
    "reviewer": {
        "id": "reviewer",
        "title": "审阅 Agent",
        "role": "KEEP / Claim",
        "hint": "解释 Rubric 与 ClaimGate，KEEP 不是声称",
    },
}

SYSTEM_PROMPT = """你是 Scientist Lab 的实验室助手，正在和人类研究员多轮对话。
产品是可控自主实验系统，不是更好的检测器，也不是会自动发论文的完整 LLM AI Scientist。
对话规则：
- 像同事一样短答：先回当前问题，再给一个下一步。不要一次倒出全部协议。
- KEEP ≠ Claim。LLM Planner 只决定 WHAT/WHY；Adapter 才是 HOW；Rubric 决定 KEEP/DISCARD；ClaimGate 决定能不能这么说。
- Formal C1 才能展示对照数字：baseline APS=0.0163 vs early_concat APS=0.0326（SUPPORTED）。probe APS=0 只是工程闭环，不是科学声称。
- 不要编造 GPU 指标，不要声称 SOTA 或 FDPN 有效，不要输出 JSON。
- 不能在聊天里批准 GPU、改 Memory、覆盖 Rubric。想跑真实验，请对方去「实验闭环」页二次确认。
下面是当前本机快照（JSON）：
"""

AGENT_PROMPTS = {
    "planner": SYSTEM_PROMPT
    + "当前角色：规划 Agent。只谈 WHAT/WHY、下一轮候选与 Gate。不要假装已经跑完 GPU。\n",
    "executor": SYSTEM_PROMPT
    + "当前角色：执行 Agent。只谈 HOW、run 状态、日志与训练进度。聊天不能 --execute。\n",
    "reviewer": SYSTEM_PROMPT
    + "当前角色：审阅 Agent。只谈 Rubric / ClaimGate / 能说什么。不要覆盖 KEEP/DISCARD。\n",
}

_GREETINGS = {
    "你好",
    "您好",
    "你好啊",
    "在吗",
    "hello",
    "hi",
    "hey",
    "早上好",
    "晚上好",
    "嗨",
}


def _suggestions(snapshot: Mapping[str, Any], kind: str) -> list[dict[str, str]]:
    c1 = dict(snapshot.get("c1") or {})
    llm = dict(snapshot.get("llm") or {})
    items: list[dict[str, str]] = [
        {"label": "实验室现在怎样？", "text": "用当前快照告诉我实验室现在处于什么状态。"},
        {"label": "能声称什么？", "text": "用当前实验室快照，告诉我现在能声称什么、不能声称什么。"},
        {"label": "看实验闭环", "text": "/loop"},
    ]
    if c1.get("allowed"):
        items.insert(1, {"label": "解释 Formal C1", "text": "/c1"})
    if kind in {"greet", "talk"} and not llm.get("ready_for_real_calls"):
        items.append({"label": "去配置模型", "text": "模型还没就绪的话，告诉我要去哪一页。"})
    if kind == "replay":
        items = [
            {"label": "看这个包", "text": "/loop"},
            {"label": "能声称什么？", "text": "用当前实验室快照，告诉我现在能声称什么、不能声称什么。"},
        ]
    return items[:4]


def _assistant_reply(
    *,
    text: str,
    snapshot: Mapping[str, Any],
    kind: str = "talk",
    live: bool = False,
    links: list[dict[str, str]] | None = None,
    navigate: str | None = None,
    extra: Mapping[str, Any] | None = None,
    agent: str = "planner",
) -> dict[str, Any]:
    meta = AGENT_META.get(agent) or AGENT_META["planner"]
    reply: dict[str, Any] = {
        "role": "assistant",
        "agent": meta["title"],
        "agent_id": meta["id"],
        "kind": kind,
        "live": live,
        "text": text,
        "links": list(links or []),
        "suggestions": _suggestions(snapshot, kind),
    }
    if navigate:
        reply["navigate"] = navigate
    if extra:
        reply.update(dict(extra))
    return reply


def build_opening_turn(snapshot: Mapping[str, Any], agent: str = "planner") -> dict[str, Any]:
    """First agent utterance when the human opens a thread."""
    name = agent if agent in AGENT_META else "planner"
    meta = AGENT_META[name]
    c1 = dict(snapshot.get("c1") or {})
    lines = [
        f"你好，我是实验室助手的{meta['title']}（{meta['role']}）。{meta['hint']}。",
        "这段对话会单独保存在左侧记录里，可随时翻看。聊天不会改研究 Memory，也不会直接烧 GPU。",
    ]
    if c1.get("allowed"):
        lines.append(
            f"当前 Formal C1：baseline APS {c1.get('baseline_aps')} vs early_concat {c1.get('candidate_aps')}，"
            f"ClaimGate {c1.get('claim_status')}。KEEP 不是声称。"
        )
    return _assistant_reply(text="\n".join(lines), snapshot=snapshot, kind="greet", agent=name)


def _normalize_utterance(text: str) -> str:
    return text.strip().lower().rstrip("！!。.?？~～")


def _looks_like_greeting(text: str) -> bool:
    return _normalize_utterance(text) in _GREETINGS


def _looks_like_status(text: str) -> bool:
    raw = text.strip()
    lowered = raw.lower()
    return lowered in {"/status", "状态", "系统状态", "全局状态"} or any(
        key in raw for key in ("什么状态", "现在怎样", "实验室现在", "全局快照")
    )


def _looks_like_claim(text: str) -> bool:
    raw = text.strip()
    lowered = raw.lower()
    return lowered in {"/c1", "c1"} or any(
        key in raw for key in ("能声称", "不能说", "能说什么", "声称什么", "claimgate", "宣称")
    )


def _looks_like_next(text: str) -> bool:
    raw = text.strip()
    return any(key in raw for key in ("下一步", "接下来", "该做什么", "然后呢"))


def build_console_snapshot(root: Path | str, *, llm_status: Mapping[str, Any] | None = None) -> dict[str, Any]:
    project = Path(root)
    catalog = list_local_runs(project)
    items = list(catalog.get("items") or [])
    c1: dict[str, Any] | None = None
    latest: dict[str, Any] | None = None
    try:
        pack = inspect_local_run(project, "formal_c1_aps_early_concat")
        c1 = dict(pack.get("c1") or {})
    except (FileNotFoundError, PermissionError, ValueError):
        c1 = {"allowed": False, "warning": "Formal C1 pack 不在本机磁盘。"}
    for row in items:
        if row.get("id") == "v25d_llm_real_loop" and row.get("available"):
            latest = {
                "id": row.get("id"),
                "title": row.get("title"),
                "kind": row.get("kind"),
                "note": "probe 工程闭环，不是 C1。",
            }
            break
    llm = dict(llm_status or {})
    payload = {
        "product": "Scientist Lab · 可控自主实验系统",
        "llm": {
            "ready_for_real_calls": bool(llm.get("ready_for_real_calls")),
            "api_key_present": bool(llm.get("api_key_present")),
            "allow_network": bool(llm.get("allow_network")),
            "provider": llm.get("provider"),
            "model": llm.get("model"),
            "missing_for_real": llm.get("missing_for_real") or [],
        },
        "c1": {
            "allowed": bool((c1 or {}).get("allowed")),
            "baseline_aps": (c1 or {}).get("baseline_aps_display")
            or (c1 or {}).get("baseline_aps"),
            "candidate_aps": (c1 or {}).get("candidate_aps_display")
            or (c1 or {}).get("candidate_aps"),
            "claim_status": (c1 or {}).get("claim_status"),
            "keep_is_not_claim": bool((c1 or {}).get("keep_is_not_claim", True)),
            "note": (c1 or {}).get("note") or (c1 or {}).get("warning"),
        },
        "doctor": {
            "note": "打开控制台不探测 GPU。真 --execute 只在实验闭环页二次确认。",
        },
        "latest_loop": latest,
        "packs": [
            {
                "id": row.get("id"),
                "title": row.get("title"),
                "kind": row.get("kind"),
                "available": bool(row.get("available")),
            }
            for row in items
            if row.get("id") in {
                "formal_c1_aps_early_concat",
                "v25d_llm_real_loop",
                "fixture_m4_rounds3_discard",
            }
        ],
    }
    payload["opening"] = build_opening_turn(payload)
    return payload


def _briefing(snapshot: Mapping[str, Any]) -> str:
    llm = dict(snapshot.get("llm") or {})
    c1 = dict(snapshot.get("c1") or {})
    latest = dict(snapshot.get("latest_loop") or {})
    packs = list(snapshot.get("packs") or [])
    ready = "已就绪（可对话）" if llm.get("ready_for_real_calls") else "未允许联网或 Key 不完整"
    lines = [
        f"我是 Scientist Lab 控制台。当前大模型：{ready}。",
        f"模型 {llm.get('model') or '—'} · 提供方 {llm.get('provider') or '—'}",
    ]
    if c1.get("allowed"):
        lines.append(
            f"Formal C1（唯一可展示的对照）：baseline APS {c1.get('baseline_aps')} vs early_concat {c1.get('candidate_aps')}，ClaimGate {c1.get('claim_status')}。KEEP 不是声称。"
        )
    else:
        lines.append(str(c1.get("note") or "本机还没有 Formal C1 产物。"))
    if latest:
        lines.append(f"最近工程闭环：{latest.get('title')}（probe，不是科学声称）。")
    available = [str(p.get("title")) for p in packs if p.get("available")]
    if available:
        lines.append("可打开的证据包：" + "、".join(available))
    lines.append("你可以直接说话。想看手册再输入 /help。")
    return "\n".join(lines)


def _help_text() -> str:
    return (
        f"我是{AGENT_NAME}，可以连续对话，不必先记指令。\n"
        "你可以直接问：实验室现在怎样、能声称什么、下一步做什么。\n"
        "快捷指令仍然有效：\n"
        "• /status — 本机全局快照\n"
        "• /c1 — Formal 对照能说什么、不能说什么\n"
        "• /loop — 打开实验闭环页\n"
        "• /replay — 对 DISCARD fixture 做离线 LLM Planner 考试（不烧 GPU）\n"
        "• /review — 对同一包做 Reviewer 解释（不覆盖 Rubric）\n"
        "勾选「联网对话」后，更开放的问题会带上实验室快照交给大模型。"
    )


def handle_console_command(
    text: str,
    *,
    snapshot: Mapping[str, Any],
    root: Path | str,
    dispatch: Any | None = None,
    agent: str = "planner",
) -> dict[str, Any] | None:
    name = agent if agent in AGENT_META else "planner"

    def speak(**kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("snapshot", snapshot)
        kwargs.setdefault("agent", name)
        return _assistant_reply(**kwargs)

    raw = text.strip()
    lower = raw.lower()
    if not raw:
        return speak(
            text="你可以直接说话，或点下面的建议。想看手册就说「帮助」。",
            kind="greet",
        )
    if _looks_like_greeting(raw):
        return build_opening_turn(snapshot, agent=name)
    if lower in {"/help", "帮助", "help"}:
        return speak(
            text=_help_text(),
            kind="help",
            links=[{"label": "模型配置", "to": "/llm-config"}],
        )
    if _looks_like_status(raw):
        return speak(
            text=_briefing(snapshot),
            kind="status",
            links=[{"label": "实验闭环", "to": "/loop"}],
        )
    if _looks_like_claim(raw):
        c1 = dict(snapshot.get("c1") or {})
        if c1.get("allowed"):
            text_out = (
                f"Formal C1 允许说：在当前 staging 协议与相同 fingerprint 下，"
                f"early_concat APS {c1.get('candidate_aps')} 高于 baseline {c1.get('baseline_aps')}。"
                f"ClaimGate = {c1.get('claim_status')}。不得升级成 FDPN / SOTA / 一般因果。"
            )
        else:
            text_out = str(c1.get("note") or "没有 Formal C1 证据。")
        return speak(
            text=text_out,
            kind="claim",
            links=[{"label": "打开 C1 对照", "to": "/loop/formal_c1_aps_early_concat"}],
        )
    if _looks_like_next(raw):
        return speak(
            text=(
                "下一步仍走闭环，不绕过 Gate：先看 Formal C1 对照，再决定要不要离线 replay。"
                "真 GPU 只在实验闭环页二次确认。"
            ),
            kind="navigate",
            links=[{"label": "打开闭环", "to": "/loop"}],
        )
    if lower in {"/loop", "闭环"}:
        return speak(
            text=(
                "下一步仍走闭环，不绕过 Gate：先看 Formal C1 对照，再决定要不要离线 replay。"
                "真 GPU 只在实验闭环页二次确认。"
            ),
            kind="navigate",
            links=[{"label": "打开闭环", "to": "/loop"}],
            navigate="/loop",
        )
    if lower.startswith("/replay") or lower.startswith("/review"):
        run_id = "fixture_m4_rounds3_discard"
        action = "llm_plan_replay" if lower.startswith("/replay") else "llm_review_replay"
        if dispatch is None:
            return speak(
                text=f"将调度 {action}（dry-run，不 GPU）。",
                kind="replay",
                links=[{"label": "闭环页确认", "to": f"/loop/{run_id}"}],
            )
        result = dispatch(run_id=run_id, action=action)
        return speak(
            text=(
                f"已离线执行 {action}（不烧 GPU）。Gate 仍不可绕过。\n"
                f"{json.dumps(result, ensure_ascii=False, indent=2)[:1800]}"
            ),
            kind="replay",
            links=[{"label": "查看该包", "to": f"/loop/{run_id}"}],
            extra={"payload": result},
        )
    if raw.startswith("/"):
        return speak(
            text=f"未知指令「{raw}」。你也可以直接用普通话问我，或输入 /help。",
            kind="help",
            links=[{"label": "模型配置", "to": "/llm-config"}],
        )
    return None


def run_console_chat(
    *,
    message: str,
    history: list[Mapping[str, Any]] | None,
    snapshot: Mapping[str, Any],
    runtime_dir: Path | str,
    live: bool,
    agent: str = "planner",
    memory_notes: list[str] | None = None,
) -> dict[str, Any]:
    name = agent if agent in AGENT_META else "planner"
    apply_runtime_llm_env(Path(runtime_dir))
    llm = dict(snapshot.get("llm") or {})
    ready = bool(llm.get("ready_for_real_calls"))
    if not live or not ready:
        why = "未勾选联网对话" if not live else "大模型尚未就绪（需要 Key、模型名、Base URL，并允许联网）"
        return _assistant_reply(
            text=f"{why}。我先用本机快照回答，不会假装已经连上公网模型。\n\n{_briefing(snapshot)}",
            snapshot=snapshot,
            kind="status",
            links=[{"label": "去模型配置", "to": "/llm-config"}],
            agent=name,
        )
    notes = [row.strip() for row in (memory_notes or []) if str(row).strip()]
    memory_block = ""
    if notes:
        memory_block = "\n本会话对话记忆（人类钉选，不是研究 Memory）：\n- " + "\n- ".join(notes[:12])
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (AGENT_PROMPTS.get(name) or SYSTEM_PROMPT)
            + json.dumps(snapshot, ensure_ascii=False, indent=2)[:4000]
            + memory_block,
        }
    ]
    for row in (history or [])[-12:]:
        role = str(row.get("role") or "")
        content = str(row.get("content") or row.get("text") or "").strip()
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": message})
    request = LLMRequest(
        purpose="other",
        messages=messages,
        temperature=0.3,
        max_tokens=700,
        metadata={"console_chat": True},
    )
    try:
        response = complete_chat(request, live=True)
    except (MissingAPIKeyError, RealProviderNotEnabledError, LLMError, json.JSONDecodeError) as exc:
        return _assistant_reply(
            text=f"大模型调用被拒绝（fail closed）：{exc}\n\n{_briefing(snapshot)}",
            snapshot=snapshot,
            kind="status",
            links=[{"label": "去模型配置", "to": "/llm-config"}],
            agent=name,
        )
    return _assistant_reply(
        text=str(response.content or "").strip() or "（模型没有返回文本）",
        snapshot=snapshot,
        kind="talk",
        live=True,
        links=[{"label": "实验闭环", "to": "/loop"}],
        extra={
            "model": response.model,
            "provider": response.provider,
            "request_id": response.request_id,
        },
        agent=name,
    )


def handle_console_turn(
    *,
    message: str,
    history: list[Mapping[str, Any]] | None,
    snapshot: Mapping[str, Any],
    runtime_dir: Path | str,
    live: bool,
    root: Path | str,
    dispatch: Any | None = None,
    agent: str = "planner",
    memory_notes: list[str] | None = None,
) -> dict[str, Any]:
    commanded = handle_console_command(
        message,
        snapshot=snapshot,
        root=root,
        dispatch=dispatch,
        agent=agent,
    )
    if commanded is not None:
        return commanded
    return run_console_chat(
        message=message,
        history=history,
        snapshot=snapshot,
        runtime_dir=runtime_dir,
        live=live,
        agent=agent,
        memory_notes=memory_notes,
    )
