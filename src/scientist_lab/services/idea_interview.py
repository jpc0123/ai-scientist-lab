"""Idea Interview: multi-turn topic dialogue before experiment propose.

Not the campaign Planner. Not a fifth Agent. No GPU. Not a Claim.
Extracts the user's core intent so propose can draft a protocol that matches it.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from scientist_lab.llm.gateway import complete_chat
from scientist_lab.llm.models import LLMRequest
from scientist_lab.llm.planner_contract import prompt_hash
from scientist_lab.llm.schema_parser import extract_json_object, validate_against_schema

_OTHER_TASK = re.compile(
    r"分类|分割|segmentation|classification|nlp|diffusion|强化学习|reinforcement|"
    r"生成模型|llm.?fine.?tun",
    re.IGNORECASE,
)

IDEA_INTERVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["reply", "brief"],
    "additionalProperties": True,
    "properties": {
        "reply": {"type": "string", "minLength": 1},
        "brief": {
            "type": "object",
            "required": ["core_intent", "ready_to_draft"],
            "additionalProperties": True,
            "properties": {
                "core_intent": {"type": "string"},
                "research_direction": {"type": "string"},
                "hypothesis": {"type": "string"},
                "constraints": {"type": "array", "items": {"type": "string"}},
                "preferred_adapter": {"type": ["string", "null"]},
                "preferred_metric": {"type": ["string", "null"]},
                "preferred_slice": {"type": ["string", "null"]},
                "open_questions": {"type": "array", "items": {"type": "string"}},
                "ready_to_draft": {"type": "boolean"},
                "unsupported_task_note": {"type": ["string", "null"]},
            },
        },
    },
}

INTERVIEW_SYSTEM_PROMPT = """You run an Idea Interview for Scientist Lab topic selection.
You are NOT the campaign Planner Agent and NOT a fifth Agent.
Your only job: clarify the human's real research idea through dialogue, then extract a brief.

Rules:
- Ask at most 1-2 concrete clarifying questions when the idea is vague.
- Extract core_intent, research_direction, hypothesis, constraints.
- Prefer object_detection on adapters dfine/rtdetr with existing HOW ids.
- If the user wants classification/segmentation/other unsupported tasks, say so honestly
  in unsupported_task_note and set ready_to_draft=false.
- When the idea is clear enough to draft a protocol, set ready_to_draft=true and
  summarize the direction in reply (do not invent GPU results or Claims).
- Return JSON only matching the schema. No GPU. KEEP ≠ Claim.
- Reply language should match the human (Chinese if they wrote Chinese).
"""


class IdeaInterviewError(ValueError):
    """Illegal or unusable Idea Interview request."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def empty_brief() -> dict[str, Any]:
    return {
        "core_intent": "",
        "research_direction": "",
        "hypothesis": "",
        "constraints": [],
        "preferred_adapter": None,
        "preferred_metric": None,
        "preferred_slice": None,
        "open_questions": [],
        "ready_to_draft": False,
        "unsupported_task_note": None,
    }


def normalize_brief(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    base = empty_brief()
    if not raw:
        return base
    base["core_intent"] = str(raw.get("core_intent") or "").strip()
    base["research_direction"] = str(raw.get("research_direction") or "").strip()
    base["hypothesis"] = str(raw.get("hypothesis") or "").strip()
    constraints = raw.get("constraints") or []
    base["constraints"] = [
        str(x).strip() for x in constraints if str(x).strip()
    ][:12]
    for key in ("preferred_adapter", "preferred_metric", "preferred_slice"):
        val = raw.get(key)
        base[key] = str(val).strip() if val not in (None, "") else None
    questions = raw.get("open_questions") or []
    base["open_questions"] = [str(x).strip() for x in questions if str(x).strip()][:8]
    base["ready_to_draft"] = bool(raw.get("ready_to_draft"))
    note = raw.get("unsupported_task_note")
    base["unsupported_task_note"] = str(note).strip() if note not in (None, "") else None
    return base


def brief_as_user_intent(brief: Mapping[str, Any] | None) -> str:
    row = normalize_brief(brief)
    parts = [
        row.get("core_intent"),
        row.get("research_direction"),
        row.get("hypothesis"),
    ]
    constraints = row.get("constraints") or []
    if constraints:
        parts.append("constraints: " + "; ".join(str(x) for x in constraints))
    for key in ("preferred_adapter", "preferred_metric", "preferred_slice"):
        if row.get(key):
            parts.append(f"{key}={row[key]}")
    if row.get("unsupported_task_note"):
        parts.append(f"note: {row['unsupported_task_note']}")
    return " | ".join(str(p).strip() for p in parts if p and str(p).strip())


class IdeaInterviewService:
    """Disk-backed Idea Interview sessions under ``.run/idea_interviews``."""

    def __init__(self, project_root: Path | str) -> None:
        self.project_root = Path(project_root)
        self.root = self.project_root / ".run" / "idea_interviews"

    def _path(self, interview_id: str) -> Path:
        return self.root / interview_id / "session.json"

    def create(self) -> dict[str, Any]:
        interview_id = f"idea_{uuid.uuid4().hex[:12]}"
        now = _now()
        session = {
            "interview_id": interview_id,
            "status": "open",
            "turns": [],
            "brief": empty_brief(),
            "created_at": now,
            "updated_at": now,
            "not_planner": True,
            "gpu": False,
            "is_claim": False,
            "note": (
                "Idea Interview only. Not campaign Planner. "
                "No GPU. KEEP ≠ Claim."
            ),
        }
        _dump(self._path(interview_id), session)
        return self.public(session)

    def get(self, interview_id: str) -> dict[str, Any]:
        path = self._path(interview_id)
        if not path.is_file():
            raise IdeaInterviewError(f"unknown idea interview: {interview_id}")
        return self.public(_load(path))

    def clear(self, interview_id: str) -> dict[str, Any]:
        path = self._path(interview_id)
        if not path.is_file():
            raise IdeaInterviewError(f"unknown idea interview: {interview_id}")
        session = _load(path)
        session["turns"] = []
        session["brief"] = empty_brief()
        session["status"] = "cleared"
        session["updated_at"] = _now()
        session.pop("last_draft_experiment_id", None)
        _dump(path, session)
        return self.public(session)

    def mark_drafted(self, interview_id: str, experiment_id: str | None = None) -> dict[str, Any]:
        path = self._path(interview_id)
        if not path.is_file():
            raise IdeaInterviewError(f"unknown idea interview: {interview_id}")
        session = _load(path)
        session["status"] = "drafted"
        session["updated_at"] = _now()
        if experiment_id:
            session["last_draft_experiment_id"] = experiment_id
        _dump(path, session)
        return self.public(session)

    def public(self, session: Mapping[str, Any]) -> dict[str, Any]:
        row = dict(session)
        row["brief"] = normalize_brief(row.get("brief") if isinstance(row.get("brief"), Mapping) else None)
        row["user_intent"] = brief_as_user_intent(row["brief"])
        row["not_planner"] = True
        row["gpu"] = False
        row["is_claim"] = False
        return row

    def chat(
        self,
        interview_id: str,
        message: str,
        *,
        live: bool = False,
        provider: Any | None = None,
    ) -> dict[str, Any]:
        text = str(message or "").strip()
        if not text:
            raise IdeaInterviewError("idea interview message is empty")
        path = self._path(interview_id)
        if not path.is_file():
            raise IdeaInterviewError(f"unknown idea interview: {interview_id}")
        session = _load(path)
        turns = list(session.get("turns") or [])
        human = {
            "id": f"t_{uuid.uuid4().hex[:10]}",
            "role": "human",
            "text": text,
            "at": _now(),
        }
        turns.append(human)

        request = build_interview_request(
            turns=turns,
            brief=normalize_brief(session.get("brief") if isinstance(session.get("brief"), Mapping) else None),
        )
        response = complete_chat(request, provider=provider, live=live)
        raw = str(response.content or "")
        try:
            parsed = _parse_interview(raw)
        except IdeaInterviewError:
            parsed = _fallback_interview(text, normalize_brief(session.get("brief")))

        brief = normalize_brief(parsed.get("brief"))
        if not brief.get("core_intent"):
            brief["core_intent"] = text[:400]
        reply = str(parsed.get("reply") or "").strip() or "已记下。可以继续补充，或基于想法起草。"
        assistant = {
            "id": f"t_{uuid.uuid4().hex[:10]}",
            "role": "assistant",
            "text": reply,
            "at": _now(),
        }
        turns.append(assistant)
        session["turns"] = turns[-40:]
        session["brief"] = brief
        session["status"] = "ready_to_draft" if brief.get("ready_to_draft") else "open"
        session["updated_at"] = _now()
        session["last_llm"] = {
            "provider": getattr(response, "provider", None),
            "model": getattr(response, "model", None),
            "prompt_hash": prompt_hash(request.messages),
            "live": bool(live),
        }
        _dump(path, session)
        payload = self.public(session)
        payload["reply"] = reply
        return payload


def build_interview_request(
    *,
    turns: list[Mapping[str, Any]],
    brief: Mapping[str, Any],
) -> LLMRequest:
    history = [
        {"role": str(t.get("role") or "human"), "text": str(t.get("text") or "")}
        for t in turns[-16:]
        if str(t.get("text") or "").strip()
    ]
    user = {
        "current_brief": normalize_brief(brief),
        "dialogue": history,
        "constraints": {
            "not_planner": True,
            "no_gpu": True,
            "not_a_claim": True,
            "task_preference": "object_detection",
            "allowed_adapters": ["dfine", "rtdetr"],
        },
    }
    return LLMRequest(
        purpose="other",
        messages=[
            {"role": "system", "content": INTERVIEW_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(user, ensure_ascii=False, sort_keys=True),
            },
        ],
        response_schema=IDEA_INTERVIEW_SCHEMA,
        temperature=0.3,
        metadata={"planner_contract": "idea_interview"},
    )


def _parse_interview(raw: str) -> dict[str, Any]:
    try:
        data = extract_json_object(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        raise IdeaInterviewError(f"fail_closed: invalid interview JSON: {exc}") from exc
    errors = validate_against_schema(data, IDEA_INTERVIEW_SCHEMA)
    if errors:
        raise IdeaInterviewError("fail_closed: interview schema: " + "; ".join(errors))
    return dict(data)


def _fallback_interview(message: str, previous: Mapping[str, Any]) -> dict[str, Any]:
    brief = normalize_brief(previous)
    brief["core_intent"] = message[:400]
    if _OTHER_TASK.search(message):
        brief["ready_to_draft"] = False
        brief["unsupported_task_note"] = (
            "当前可开战战役仍限 object_detection + 已支持 Adapter；"
            "可先记笔记，登记后多半停在 draft。"
        )
        reply = (
            "已记下。分类/分割等任务现在还不能当可开战战役跑。"
            "若你愿意，可以改成目标检测方向（例如低光融合、第二检测器迁移、换切片/指标），"
            "再点起草。"
        )
    else:
        brief["research_direction"] = brief.get("research_direction") or message[:240]
        brief["ready_to_draft"] = len(message.strip()) >= 12
        brief["open_questions"] = [
            "主指标更在意 APS_lowlight 还是 full-val APS？",
            "对照想绑在 D-FINE 还是第二检测器？",
        ]
        reply = (
            "已记下你的方向。可以再补一句主指标或对照设置；"
            "若已经够清楚，直接点「基于想法让 LLM 起草」。"
        )
    return {"reply": reply, "brief": brief}
