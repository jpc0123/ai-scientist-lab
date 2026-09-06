"""Campaign Steer: human mid-campaign direction for the next Planner round.

Not Protocol Amendment. Not scout literature intent. Not a Claim. No GPU.
Cannot change frozen dataset / metric / evaluator / claim policy.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

STORE_NAME = "steer_intent.json"
MAX_DIALOGUE = 40
MAX_TEXT = 2000

_AMENDMENT_HINT = re.compile(
    r"("
    r"换\s*数据|改\s*数据|新\s*数据|dataset|换\s*切片|改\s*切片|slice|"
    r"换\s*指标|改\s*指标|主指标|primary[_\s-]?metric|evaluator|评价器|"
    r"改\s*协议|protocol\s*amend|冻结|fingerprint|claim\s*policy|"
    r"换\s*检测器|新\s*backbone|yolo"
    r")",
    re.IGNORECASE,
)


class CampaignSteerError(ValueError):
    """Illegal steer request."""


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def steer_path(campaign_dir: Path | str) -> Path:
    return Path(campaign_dir) / STORE_NAME


def empty_store() -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "steer_intent": None,
        "steer_dialogue": [],
        "can_bypass_gate": False,
        "can_write_claim": False,
        "can_start_gpu": False,
        "not_protocol_amendment": True,
        "updated_at": _now(),
    }


def load_store(path: Path | str) -> dict[str, Any]:
    dest = Path(path)
    if not dest.is_file():
        return empty_store()
    raw = json.loads(dest.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise CampaignSteerError("steer_intent.json must be an object")
    store = empty_store()
    store.update(raw)
    intent = store.get("steer_intent")
    store["steer_intent"] = dict(intent) if isinstance(intent, Mapping) else None
    dialogue = store.get("steer_dialogue")
    store["steer_dialogue"] = list(dialogue) if isinstance(dialogue, list) else []
    store["can_bypass_gate"] = False
    store["can_write_claim"] = False
    store["can_start_gpu"] = False
    store["not_protocol_amendment"] = True
    return store


def save_store(path: Path | str, store: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(store)
    body["updated_at"] = _now()
    body["can_bypass_gate"] = False
    body["can_write_claim"] = False
    body["can_start_gpu"] = False
    body["not_protocol_amendment"] = True
    _dump(Path(path), body)
    return load_store(path)


def needs_protocol_amendment(text: str) -> bool:
    return bool(_AMENDMENT_HINT.search(str(text or "")))


def make_intent(
    *,
    text: str,
    source: str = "human",
    status: str = "active",
    why: str = "",
) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        raise CampaignSteerError("steer text is empty")
    if len(raw) > MAX_TEXT:
        raise CampaignSteerError(f"steer text exceeds {MAX_TEXT} chars")
    amendment = needs_protocol_amendment(raw)
    return {
        "text": raw,
        "source": source,
        "status": status,
        "why": str(why or "").strip() or None,
        "needs_protocol_amendment": amendment,
        "planner_may_use": not amendment,
        "set_at": _now(),
        "cannot": ["bypass_gate", "write_claim", "start_gpu", "edit_frozen_protocol"],
    }


def append_dialogue(
    path: Path | str,
    *,
    role: str,
    text: str,
    action: str | None = None,
) -> dict[str, Any]:
    store = load_store(path)
    turns = list(store.get("steer_dialogue") or [])
    turns.append(
        {
            "id": f"st_{uuid.uuid4().hex[:10]}",
            "role": role,
            "text": str(text or "").strip()[:MAX_TEXT],
            "action": action,
            "at": _now(),
        }
    )
    store["steer_dialogue"] = turns[-MAX_DIALOGUE:]
    return save_store(path, store)


def set_steer(
    path: Path | str,
    *,
    text: str,
    why: str = "",
    source: str = "human",
) -> dict[str, Any]:
    store = load_store(path)
    intent = make_intent(text=text, source=source, status="active", why=why)
    store["steer_intent"] = intent
    save_store(path, store)
    append_dialogue(path, role="human", text=text, action="set")
    if intent["needs_protocol_amendment"]:
        reply = (
            "这段更像要改协议（数据/切片/指标等），不能当本轮 Steer。"
            "请走选题起草或 Protocol Amendment；当前活跃 Steer 已标记为不可给 Planner。"
        )
    else:
        reply = f"已钉住下一轮转向。Planner 下一轮可读：{text}"
    append_dialogue(path, role="assistant", text=reply, action="set")
    return load_store(path)


def clear_steer(path: Path | str, *, why: str = "") -> dict[str, Any]:
    store = load_store(path)
    store["steer_intent"] = {
        "text": None,
        "source": "human",
        "status": "cleared",
        "why": why or "human cleared steer",
        "needs_protocol_amendment": False,
        "planner_may_use": False,
        "set_at": _now(),
        "cannot": ["bypass_gate", "write_claim", "start_gpu", "edit_frozen_protocol"],
    }
    save_store(path, store)
    append_dialogue(path, role="human", text=why or "清空转向", action="clear")
    append_dialogue(
        path,
        role="assistant",
        text="已清空战役 Steer。下一轮 Planner 不再带人转向。",
        action="clear",
    )
    return load_store(path)


def active_steer_for_planner(store: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Return steer payload Planner may honor; skip amendment-like directions."""
    if not store:
        return None
    intent = store.get("steer_intent")
    if not isinstance(intent, Mapping):
        return None
    if str(intent.get("status") or "") != "active":
        return None
    text = str(intent.get("text") or "").strip()
    if not text:
        return None
    if intent.get("needs_protocol_amendment") or not intent.get("planner_may_use", True):
        return {
            "text": text,
            "planner_may_use": False,
            "needs_protocol_amendment": True,
            "note": "Human asked for protocol-level change; Planner must not rewrite frozen fields.",
        }
    return {
        "text": text,
        "planner_may_use": True,
        "needs_protocol_amendment": False,
        "source": intent.get("source") or "human",
        "set_at": intent.get("set_at"),
    }


def public_store(store: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(store)
    row["active_for_planner"] = active_steer_for_planner(row)
    return row
