"""Campaign-bound scout-intent dialogue. Not a fifth Agent. Not Harness."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from scientist_lab.core.how_pending import (
    INTENT_ACTIVE,
    INTENT_PROPOSED,
    HowPendingError,
    accept_proposed_intent,
    append_scout_dialogue,
    clear_scout_intent,
    load_store,
    reject_proposed_intent,
    set_scout_intent,
)

_FORBIDDEN_REGISTER = (
    "批准进目录",
    "register how",
    "写入 catalog",
    "写入catalog",
    "register_how",
)
_FORBIDDEN_GPU = (
    "开始训练",
    "点 gpu",
    "点gpu",
    "start gpu",
    "--execute",
    "点火",
    "开始自主实验",
)
_FORBIDDEN_CLAIM = (
    "这是 claim",
    "这是claim",
    "可以声称",
    "claimgate supported",
    "写成 claim",
    "写成claim",
)
_PROPOSE_HINTS = (
    "你来定",
    "你来决定",
    "按证据",
    "llm 定",
    "llm定",
    "模型定",
    "帮我找方向",
    "提议查询",
    "propose",
    "you decide",
)
_ACCEPT_HINTS = ("接受这个查询", "接受提议", "就用这个查询", "用这条 query", "accept query")
_REJECT_HINTS = ("拒绝这个查询", "不要这个查询", "换一个查询", "reject query")
_CLEAR_HINTS = ("清空检索", "取消方向", "clear intent", "清掉查询", "清空方向")
_LIBRARY_SEARCH_HINTS = (
    "论文库",
    "文献库",
    "查库",
    "library search",
    "search library",
    "campaign library",
)
_LLM_SEARCH_HINTS = (
    "你来检索",
    "自己检索",
    "llm检索",
    "llm 检索",
    "现在就搜",
    "现在就检索",
    "马上检索",
    "自己搜",
    "llm search",
    "search now",
    "go search",
    "run search",
)
_SEARCH_HINTS = (
    "查",
    "检索",
    "搜",
    "search",
    "query",
    "文献",
    "论文",
    "低光",
    "fusion",
    "加权",
    "neck",
    "模态",
    "weight",
    "rgbt",
    "rgb-t",
)

SCOUT_QUERY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["query", "why"],
    "additionalProperties": True,
    "properties": {
        "query": {"type": "string", "minLength": 8, "maxLength": 400},
        "why": {"type": "string", "maxLength": 400},
    },
}


def _contains(text: str, needles: tuple[str, ...]) -> bool:
    hay = str(text or "").strip().lower()
    return any(item.lower() in hay for item in needles)


def classify_scout_message(text: str, *, proposed: bool = False) -> str:
    raw = str(text or "").strip()
    if not raw:
        return "empty"
    if _contains(raw, _FORBIDDEN_REGISTER):
        return "forbid_register"
    if _contains(raw, _FORBIDDEN_GPU):
        return "forbid_gpu"
    if _contains(raw, _FORBIDDEN_CLAIM):
        return "forbid_claim"
    if proposed and _contains(raw, _ACCEPT_HINTS):
        return "accept"
    if proposed and raw in {"接受", "同意", "用这个", "就这个", "accept"}:
        return "accept"
    if proposed and _contains(raw, _REJECT_HINTS):
        return "reject"
    if _contains(raw, _CLEAR_HINTS):
        return "clear"
    if _contains(raw, _LIBRARY_SEARCH_HINTS):
        return "library_search"
    if _contains(raw, _LLM_SEARCH_HINTS):
        return "llm_search"
    if _contains(raw, _PROPOSE_HINTS):
        return "propose"
    if _contains(raw, _SEARCH_HINTS):
        return "human_set"
    return "help"


def expand_human_query(text: str) -> str:
    raw = " ".join(str(text or "").split())
    extra: list[str] = []
    if any(key in raw for key in ("低光", "模态", "加权", "融合")):
        extra.append("RGB-T fusion low-light")
    if any(key in raw for key in ("加权", "weight", "gated")):
        extra.append("modality weighting gated")
    if any(key in raw.lower() for key in ("不要 neck", "别查 neck", "not neck", "不要neck")):
        extra.append("not neck")
    if extra:
        return f"{raw} {' '.join(extra)}"
    return raw


def heuristic_scout_query(
    evidence: Mapping[str, Any] | None = None,
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    from scientist_lab.literature.query_synth import compose_scout_queries

    composed = compose_scout_queries({}, protocol, evidence, live=False)
    return {"query": str(composed["query"]), "why": str(composed.get("why") or "")}


def _llm_propose_query(
    evidence: Mapping[str, Any] | None,
    protocol: Mapping[str, Any] | None,
    *,
    live: bool,
    provider: Any | None,
) -> dict[str, str] | None:
    if not live:
        return None
    from scientist_lab.literature.query_synth import compose_scout_queries

    composed = compose_scout_queries(
        {},
        protocol,
        evidence,
        live=True,
        provider=provider,
    )
    if str(composed.get("source") or "") != "llm" or composed.get("fallback"):
        return None
    return {"query": str(composed["query"]), "why": str(composed.get("why") or "")}


def _status_help(store: Mapping[str, Any]) -> str:
    intent = dict(store.get("scout_intent") or {})
    status = str(intent.get("status") or "empty")
    query = str(intent.get("query") or "")
    source = str(intent.get("source") or "")
    if status == INTENT_PROPOSED and query:
        return (
            f"当前是 LLM 查询草稿（未生效）：{query}。说「接受这个查询」或改写方向。"
            "聊天不能批准 HOW、不能点 GPU、也不能写 Claim。"
        )
    if status == INTENT_ACTIVE and query:
        return (
            f"下一枪 scout 将用 {source} 指定的查询：{query}。"
            "可改方向、让我按证据提议，或清空后走 fallback。"
        )
    return (
        "还没有活跃检索意图。直接说「往低光模态加权查」，或让我按证据提议。"
        "空着则下一枪用 fallback 模板，并标明 fallback。"
    )


def handle_scout_chat(
    path: Path | str,
    message: str,
    *,
    evidence: Mapping[str, Any] | None = None,
    protocol: Mapping[str, Any] | None = None,
    live: bool = False,
    provider: Any | None = None,
    locale: str | None = None,
    provenance_dir: Path | str | None = None,
    environ: Mapping[str, str] | None = None,
    action: str | None = None,
) -> dict[str, Any]:
    """One campaign-bound turn. Persists intent; never registers HOW or starts GPU."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    text = str(message or "").strip()
    store = load_store(dest)
    proposed = str((store.get("scout_intent") or {}).get("status") or "") == INTENT_PROPOSED
    forced = str(action or "").strip().lower()
    kind = forced if forced in {"llm_search", "library_search"} else classify_scout_message(
        text, proposed=proposed
    )
    append_scout_dialogue(dest, role="human", text=text or kind, intent_action=kind)

    refused = kind.startswith("forbid")
    reply = ""
    action = kind
    try:
        if kind == "empty":
            reply = "请说下一枪要查什么，或让我按证据提议一条 query。"
        elif kind == "forbid_register":
            reply = "检索对话不能批准 HOW。请到待审列表点人审。KEEP ≠ Claim。"
        elif kind == "forbid_gpu":
            reply = "检索对话不能点火 GPU。开始训练仍走战役 Human Gate。"
        elif kind == "forbid_claim":
            reply = "检索对话不能写 Claim。文献不能进 ClaimGate。"
        elif kind == "accept":
            store = accept_proposed_intent(dest)
            intent = dict(store.get("scout_intent") or {})
            reply = f"已接受查询。下一枪 scout 使用：{intent.get('query')}"
        elif kind == "reject":
            store = reject_proposed_intent(dest)
            reply = "已拒绝 LLM 查询草稿。下一枪将用 fallback 模板，并标明 fallback。"
        elif kind == "clear":
            store = clear_scout_intent(dest, why="human cleared scout_intent from dialogue")
            reply = "已清空检索意图。下一枪用 fallback，并标明 fallback。"
        elif kind == "llm_search":
            from scientist_lab.core.how_pending import execute_llm_scout

            result = execute_llm_scout(
                dest,
                protocol=protocol,
                evidence=evidence,
                live=bool(live),
                provider=provider,
                locale=locale,
                provenance_dir=provenance_dir,
                environ=environ,
            )
            return result
        elif kind == "library_search":
            from scientist_lab.core.how_pending import execute_library_scout

            result = execute_library_scout(
                dest,
                protocol=protocol,
                evidence=evidence,
                locale=locale,
                live=bool(live),
                provider=provider,
            )
            return result
        elif kind == "propose":
            current = dict(load_store(dest).get("scout_intent") or {})
            if (
                str(current.get("status") or "") == INTENT_ACTIVE
                and str(current.get("source") or "") == "human"
                and str(current.get("query") or "").strip()
            ):
                action = "human_holds"
                reply = (
                    f"人指定的查询仍有效，LLM 不能盖掉：{current.get('query')}。"
                    "要让我按证据提议，先清空方向。"
                )
            else:
                drafted = _llm_propose_query(
                    evidence, protocol, live=live, provider=provider
                )
                drafted_by = "llm" if drafted else "heuristic"
                if drafted is None:
                    drafted = heuristic_scout_query(evidence, protocol)
                store = set_scout_intent(
                    dest,
                    source="llm",
                    query=drafted["query"],
                    why=drafted["why"],
                    status=INTENT_PROPOSED,
                    drafted_by=drafted_by,
                )
                note = (
                    "这是 LLM 写的 query 草稿。"
                    if drafted_by == "llm"
                    else "LLM 未出可用草稿，按上一枪证据写了一条 query 草稿。"
                )
                reply = (
                    f"{note}尚未生效：{drafted['query']}。"
                    f"{drafted['why']} 说「接受这个查询」后下一枪才用。不能改 catalog。"
                )
        elif kind == "human_set":
            query = expand_human_query(text)
            store = set_scout_intent(
                dest,
                source="human",
                query=query,
                why=text,
                status=INTENT_ACTIVE,
                drafted_by="human",
            )
            reply = f"已记下人指定的检索方向。下一枪 scout 使用：{query}"
        else:
            reply = _status_help(store)
            action = "help"
    except HowPendingError as exc:
        refused = True
        action = "error"
        reply = str(exc)

    store = append_scout_dialogue(
        dest,
        role="assistant",
        text=reply,
        intent_action=action,
        refused=refused,
    )
    return {
        "ok": not refused,
        "refused": refused,
        "action": action,
        "reply": reply,
        "scout_intent": store.get("scout_intent"),
        "scout_dialogue": store.get("scout_dialogue") or [],
        "can_enter_claim_gate": False,
        "cannot": ["register_how", "start_gpu", "write_claim"],
    }
