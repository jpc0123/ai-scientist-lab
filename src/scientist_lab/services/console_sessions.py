"""Persisted command-center sessions (Cursor-like history, not Research Memory).

Stored under ``{runtime_dir}/console_sessions/``. Chat never writes
research Memory / KEEP / ClaimGate. GPU stays off.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from scientist_lab.domain.models import new_id
from scientist_lab.services.console_chat import (
    AGENT_META,
    AGENTS,
    build_console_snapshot,
    build_opening_turn,
    handle_console_turn,
)
from scientist_lab.services.local_run_inspector import inspect_local_run

SESSION_DIRNAME = "console_sessions"
WORKSPACE_FILENAME = "console_workspaces.json"
DEFAULT_WORKSPACE_ID = "ws_default"
WORKSPACE_KINDS = ("inbox", "project", "experiment")
WORKSPACE_KIND_LABELS = {
    "inbox": "临时",
    "project": "项目",
    "experiment": "实验",
}


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sessions_dir(runtime_dir: Path | str) -> Path:
    path = Path(runtime_dir) / SESSION_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def _session_path(runtime_dir: Path | str, session_id: str) -> Path:
    safe = "".join(ch for ch in session_id if ch.isalnum() or ch in {"_", "-"})
    if not safe or not safe.startswith("sess_"):
        raise FileNotFoundError(f"unknown session: {session_id}")
    return _sessions_dir(runtime_dir) / f"{safe}.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _empty_threads() -> dict[str, Any]:
    return {agent: {"messages": []} for agent in AGENTS}


def _normalize_agent(agent: str | None) -> str:
    name = str(agent or "planner").strip().lower()
    return name if name in AGENTS else "planner"


def _normalize_kind(kind: str | None) -> str:
    name = str(kind or "inbox").strip().lower()
    return name if name in WORKSPACE_KINDS else "inbox"


def _normalize_workspace_id(workspace_id: str | None) -> str:
    raw = "".join(ch for ch in str(workspace_id or "").strip() if ch.isalnum() or ch in {"_", "-"})
    if not raw or not raw.startswith("ws_"):
        return DEFAULT_WORKSPACE_ID
    return raw


def _workspaces_path(runtime_dir: Path | str) -> Path:
    return Path(runtime_dir) / WORKSPACE_FILENAME


def _default_workspace() -> dict[str, Any]:
    now = _now()
    return {
        "id": DEFAULT_WORKSPACE_ID,
        "title": "默认工作区",
        "kind": "inbox",
        "kind_label": WORKSPACE_KIND_LABELS["inbox"],
        "project_id": None,
        "pack_id": None,
        "memory": [],
        "created_at": now,
        "updated_at": now,
    }


def _decorate_workspace(row: Mapping[str, Any]) -> dict[str, Any]:
    kind = _normalize_kind(str(row.get("kind") or "inbox"))
    payload = dict(row)
    payload["id"] = _normalize_workspace_id(str(row.get("id") or DEFAULT_WORKSPACE_ID))
    payload["kind"] = kind
    payload["kind_label"] = WORKSPACE_KIND_LABELS[kind]
    payload["title"] = str(row.get("title") or "未命名工作区")[:80]
    payload["memory"] = list(row.get("memory") or [])
    return payload


def _load_workspace_store(runtime_dir: Path | str) -> dict[str, Any]:
    path = _workspaces_path(runtime_dir)
    if not path.is_file():
        store = {"active_id": DEFAULT_WORKSPACE_ID, "items": [_default_workspace()]}
        _dump(path, store)
        return store
    try:
        store = _load(path)
    except (OSError, json.JSONDecodeError, TypeError):
        store = {"active_id": DEFAULT_WORKSPACE_ID, "items": [_default_workspace()]}
    items = [_decorate_workspace(row) for row in list(store.get("items") or []) if isinstance(row, Mapping)]
    if not any(row.get("id") == DEFAULT_WORKSPACE_ID for row in items):
        items.insert(0, _default_workspace())
    store["items"] = items
    store["active_id"] = _normalize_workspace_id(str(store.get("active_id") or DEFAULT_WORKSPACE_ID))
    return store


def _save_workspace_store(runtime_dir: Path | str, store: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(store)
    payload["active_id"] = _normalize_workspace_id(str(payload.get("active_id") or DEFAULT_WORKSPACE_ID))
    payload["items"] = [_decorate_workspace(row) for row in list(payload.get("items") or []) if isinstance(row, Mapping)]
    _dump(_workspaces_path(runtime_dir), payload)
    return payload


def get_workspace(runtime_dir: Path | str, workspace_id: str | None = None) -> dict[str, Any]:
    store = _load_workspace_store(runtime_dir)
    wanted = _normalize_workspace_id(workspace_id)
    for row in list(store.get("items") or []):
        if row.get("id") == wanted:
            return dict(row)
    if wanted == DEFAULT_WORKSPACE_ID:
        store["items"] = [_default_workspace(), *list(store.get("items") or [])]
        _save_workspace_store(runtime_dir, store)
        return _default_workspace()
    raise FileNotFoundError(f"workspace not found: {wanted}")


def summarize_workspace(runtime_dir: Path | str, workspace: Mapping[str, Any]) -> dict[str, Any]:
    ws_id = _normalize_workspace_id(str(workspace.get("id") or DEFAULT_WORKSPACE_ID))
    count = 0
    for path in _sessions_dir(runtime_dir).glob("sess_*.json"):
        try:
            session = _load(path)
        except (OSError, json.JSONDecodeError, TypeError):
            continue
        if _normalize_workspace_id(str(session.get("workspace_id") or DEFAULT_WORKSPACE_ID)) == ws_id:
            count += 1
    payload = _decorate_workspace(workspace)
    payload["session_count"] = count
    payload["memory_count"] = len(list(payload.get("memory") or []))
    return payload


def enrich_workspace(
    workspace: Mapping[str, Any],
    *,
    projects: list[Mapping[str, Any]] | None = None,
    packs: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    payload = dict(workspace)
    project_id = str(payload.get("project_id") or "").strip() or None
    pack_id = str(payload.get("pack_id") or "").strip() or None
    project_title = None
    pack_title = None
    for row in list(projects or []):
        if str(row.get("project_id") or "") == str(project_id or ""):
            project_title = str(row.get("title") or project_id)
            break
    for row in list(packs or []):
        if str(row.get("id") or "") == str(pack_id or ""):
            pack_title = str(row.get("title") or pack_id)
            break
    payload["project_title"] = project_title
    payload["pack_title"] = pack_title
    payload["project_to"] = f"/projects/{project_id}" if project_id else None
    payload["pack_to"] = f"/loop/{pack_id}" if pack_id else None
    return payload


def list_workspaces(runtime_dir: Path | str) -> dict[str, Any]:
    store = _load_workspace_store(runtime_dir)
    items = [summarize_workspace(runtime_dir, row) for row in list(store.get("items") or [])]
    return {
        "ok": True,
        "active_id": store.get("active_id") or DEFAULT_WORKSPACE_ID,
        "items": items,
        "total": len(items),
    }


def create_workspace(
    runtime_dir: Path | str,
    *,
    title: str | None = None,
    kind: str | None = None,
    project_id: str | None = None,
    pack_id: str | None = None,
) -> dict[str, Any]:
    store = _load_workspace_store(runtime_dir)
    now = _now()
    workspace = {
        "id": new_id("ws"),
        "title": (title or "").strip()[:80] or "新工作区",
        "kind": _normalize_kind(kind or "project"),
        "project_id": (project_id or "").strip() or None,
        "pack_id": (pack_id or "").strip() or None,
        "memory": [],
        "created_at": now,
        "updated_at": now,
    }
    workspace = _decorate_workspace(workspace)
    items = list(store.get("items") or [])
    items.append(workspace)
    store["items"] = items
    store["active_id"] = workspace["id"]
    _save_workspace_store(runtime_dir, store)
    return summarize_workspace(runtime_dir, workspace)


def patch_workspace(
    runtime_dir: Path | str,
    workspace_id: str,
    *,
    title: str | None = None,
    kind: str | None = None,
    project_id: str | None = None,
    pack_id: str | None = None,
    active: bool | None = None,
) -> dict[str, Any]:
    store = _load_workspace_store(runtime_dir)
    wanted = _normalize_workspace_id(workspace_id)
    found = None
    items: list[dict[str, Any]] = []
    for row in list(store.get("items") or []):
        current = _decorate_workspace(row)
        if current.get("id") == wanted:
            if title is not None and title.strip():
                current["title"] = title.strip()[:80]
            if kind is not None:
                current["kind"] = _normalize_kind(kind)
            if project_id is not None:
                current["project_id"] = project_id.strip() or None
            if pack_id is not None:
                current["pack_id"] = pack_id.strip() or None
            current["updated_at"] = _now()
            found = current
        items.append(current)
    if found is None:
        raise FileNotFoundError(f"workspace not found: {wanted}")
    store["items"] = items
    if active:
        store["active_id"] = wanted
    _save_workspace_store(runtime_dir, store)
    return summarize_workspace(runtime_dir, found)


def delete_workspace(runtime_dir: Path | str, workspace_id: str) -> dict[str, Any]:
    wanted = _normalize_workspace_id(workspace_id)
    if wanted == DEFAULT_WORKSPACE_ID:
        raise ValueError("默认工作区不能删除")
    store = _load_workspace_store(runtime_dir)
    items = [row for row in list(store.get("items") or []) if _normalize_workspace_id(str(row.get("id"))) != wanted]
    if len(items) == len(list(store.get("items") or [])):
        raise FileNotFoundError(f"workspace not found: {wanted}")
    deleted_sessions: list[str] = []
    for path in list(_sessions_dir(runtime_dir).glob("sess_*.json")):
        try:
            session = _load(path)
        except (OSError, json.JSONDecodeError, TypeError):
            continue
        if _normalize_workspace_id(str(session.get("workspace_id") or DEFAULT_WORKSPACE_ID)) == wanted:
            path.unlink()
            deleted_sessions.append(str(session.get("id") or path.stem))
    store["items"] = items
    if _normalize_workspace_id(str(store.get("active_id"))) == wanted:
        store["active_id"] = DEFAULT_WORKSPACE_ID
    _save_workspace_store(runtime_dir, store)
    return {"ok": True, "id": wanted, "deleted": True, "deleted_sessions": deleted_sessions}


def add_workspace_memory(runtime_dir: Path | str, workspace_id: str, text: str) -> dict[str, Any]:
    note = str(text or "").strip()
    if not note:
        raise ValueError("memory text is empty")
    store = _load_workspace_store(runtime_dir)
    wanted = _normalize_workspace_id(workspace_id)
    found = None
    items: list[dict[str, Any]] = []
    for row in list(store.get("items") or []):
        current = _decorate_workspace(row)
        if current.get("id") == wanted:
            memory = list(current.get("memory") or [])
            memory.append({"id": new_id("wmem"), "text": note[:500], "created_at": _now()})
            current["memory"] = memory[-40:]
            current["updated_at"] = _now()
            found = current
        items.append(current)
    if found is None:
        raise FileNotFoundError(f"workspace not found: {wanted}")
    store["items"] = items
    _save_workspace_store(runtime_dir, store)
    return summarize_workspace(runtime_dir, found)


def delete_workspace_memory(runtime_dir: Path | str, workspace_id: str, note_id: str) -> dict[str, Any]:
    store = _load_workspace_store(runtime_dir)
    wanted = _normalize_workspace_id(workspace_id)
    found = None
    items: list[dict[str, Any]] = []
    for row in list(store.get("items") or []):
        current = _decorate_workspace(row)
        if current.get("id") == wanted:
            current["memory"] = [
                note for note in list(current.get("memory") or []) if str(note.get("id")) != note_id
            ]
            current["updated_at"] = _now()
            found = current
        items.append(current)
    if found is None:
        raise FileNotFoundError(f"workspace not found: {wanted}")
    store["items"] = items
    _save_workspace_store(runtime_dir, store)
    return summarize_workspace(runtime_dir, found)


def _message_line(row: Mapping[str, Any]) -> str:
    return str(row.get("text") or "").strip().splitlines()[0][:72]


def _last_user_preview(messages: list[Any]) -> str:
    for row in reversed(messages):
        if isinstance(row, Mapping) and row.get("role") == "user":
            line = _message_line(row)
            if line:
                return line
    return ""


def _thread_summaries(session: Mapping[str, Any]) -> list[dict[str, Any]]:
    threads = dict(session.get("threads") or {})
    items: list[dict[str, Any]] = []
    for agent in AGENTS:
        meta = AGENT_META[agent]
        messages = list((threads.get(agent) or {}).get("messages") or [])
        turns = sum(1 for row in messages if isinstance(row, Mapping) and row.get("role") == "user")
        last_at = ""
        if messages and isinstance(messages[-1], Mapping):
            last_at = str(messages[-1].get("created_at") or "")
        items.append(
            {
                "id": agent,
                "title": meta["title"],
                "role": meta["role"],
                "count": len(messages),
                "turns": turns,
                "preview": _last_user_preview(messages) or "尚未对话",
                "updated_at": last_at,
            }
        )
    return items


def _preview(session: Mapping[str, Any]) -> str:
    threads = dict(session.get("threads") or {})
    ranked: list[tuple[str, str, str]] = []
    for agent in AGENTS:
        messages = list((threads.get(agent) or {}).get("messages") or [])
        preview = _last_user_preview(messages)
        if not preview:
            continue
        last_at = ""
        if messages and isinstance(messages[-1], Mapping):
            last_at = str(messages[-1].get("created_at") or "")
        ranked.append((last_at, agent, preview))
    if not ranked:
        return "尚未对话"
    ranked.sort(reverse=True)
    _, agent, preview = ranked[0]
    return f"{AGENT_META[agent]['title']} · {preview}"


def _counts(session: Mapping[str, Any]) -> dict[str, int]:
    threads = dict(session.get("threads") or {})
    return {
        agent: len(list((threads.get(agent) or {}).get("messages") or []))
        for agent in AGENTS
    }


def summarize_session(session: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": session.get("id"),
        "title": session.get("title") or "未命名对话",
        "created_at": session.get("created_at"),
        "updated_at": session.get("updated_at"),
        "pinned": bool(session.get("pinned")),
        "active_agent": _normalize_agent(str(session.get("active_agent") or "planner")),
        "workspace_id": _normalize_workspace_id(str(session.get("workspace_id") or DEFAULT_WORKSPACE_ID)),
        "preview": _preview(session),
        "memory_count": len(list(session.get("memory") or [])),
        "counts": _counts(session),
        "agents": _thread_summaries(session),
    }


def list_sessions(runtime_dir: Path | str, workspace_id: str | None = None) -> dict[str, Any]:
    wanted = _normalize_workspace_id(workspace_id) if workspace_id else None
    items: list[dict[str, Any]] = []
    for path in _sessions_dir(runtime_dir).glob("sess_*.json"):
        try:
            session = _load(path)
            if not session.get("workspace_id"):
                session["workspace_id"] = DEFAULT_WORKSPACE_ID
                _dump(path, session)
            summary = summarize_session(session)
        except (OSError, json.JSONDecodeError, TypeError):
            continue
        if wanted and summary.get("workspace_id") != wanted:
            continue
        items.append(summary)
    items.sort(key=lambda row: str(row.get("updated_at") or ""), reverse=True)
    pinned = [row for row in items if row.get("pinned")]
    rest = [row for row in items if not row.get("pinned")]
    return {
        "ok": True,
        "items": pinned + rest,
        "total": len(items),
        "workspace_id": wanted,
    }


def get_session(runtime_dir: Path | str, session_id: str) -> dict[str, Any]:
    path = _session_path(runtime_dir, session_id)
    if not path.is_file():
        raise FileNotFoundError(f"session not found: {session_id}")
    session = _load(path)
    if not session.get("workspace_id"):
        session["workspace_id"] = DEFAULT_WORKSPACE_ID
        _dump(path, session)
    session["summary"] = summarize_session(session)
    try:
        session["workspace"] = get_workspace(runtime_dir, str(session.get("workspace_id")))
    except FileNotFoundError:
        session["workspace"] = _default_workspace()
    return session


def create_session(
    runtime_dir: Path | str,
    *,
    snapshot: Mapping[str, Any] | None = None,
    title: str | None = None,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    now = _now()
    session_id = new_id("sess")
    snap = dict(snapshot or {})
    threads: dict[str, Any] = {}
    for agent in AGENTS:
        opening = build_opening_turn(snap, agent=agent)
        threads[agent] = {
            "messages": [
                {
                    "id": new_id("m"),
                    "role": "assistant",
                    "text": opening.get("text"),
                    "live": False,
                    "agent": opening.get("agent"),
                    "agent_id": opening.get("agent_id") or agent,
                    "kind": opening.get("kind"),
                    "links": opening.get("links") or [],
                    "suggestions": opening.get("suggestions") or [],
                    "created_at": now,
                }
            ]
        }
    workspace = get_workspace(runtime_dir, workspace_id)
    session = {
        "id": session_id,
        "title": (title or "").strip() or "新对话",
        "created_at": now,
        "updated_at": now,
        "pinned": False,
        "active_agent": "planner",
        "workspace_id": workspace["id"],
        "title_locked": False,
        "memory": [],
        "threads": threads,
    }
    _dump(_session_path(runtime_dir, session_id), session)
    session["summary"] = summarize_session(session)
    session["workspace"] = workspace
    return session


def save_session(runtime_dir: Path | str, session: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(session)
    payload["updated_at"] = _now()
    payload.pop("summary", None)
    payload.pop("workspace", None)
    _dump(_session_path(runtime_dir, str(payload["id"])), payload)
    payload["summary"] = summarize_session(payload)
    try:
        payload["workspace"] = get_workspace(runtime_dir, str(payload.get("workspace_id")))
    except FileNotFoundError:
        payload["workspace"] = _default_workspace()
    return payload


def patch_session(
    runtime_dir: Path | str,
    session_id: str,
    *,
    title: str | None = None,
    pinned: bool | None = None,
    active_agent: str | None = None,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    session = get_session(runtime_dir, session_id)
    if title is not None:
        text = title.strip()
        if text:
            session["title"] = text[:80]
            session["title_locked"] = True
    if pinned is not None:
        session["pinned"] = bool(pinned)
    if active_agent is not None:
        session["active_agent"] = _normalize_agent(active_agent)
    if workspace_id is not None:
        target = get_workspace(runtime_dir, workspace_id)
        session["workspace_id"] = target["id"]
    return save_session(runtime_dir, session)


def delete_session(runtime_dir: Path | str, session_id: str) -> dict[str, Any]:
    path = _session_path(runtime_dir, session_id)
    if path.is_file():
        path.unlink()
    return {"ok": True, "id": session_id, "deleted": True}


def add_memory_note(runtime_dir: Path | str, session_id: str, text: str) -> dict[str, Any]:
    note = str(text or "").strip()
    if not note:
        raise ValueError("memory text is empty")
    session = get_session(runtime_dir, session_id)
    memory = list(session.get("memory") or [])
    memory.append(
        {
            "id": new_id("mem"),
            "text": note[:500],
            "created_at": _now(),
        }
    )
    session["memory"] = memory[-40:]
    return save_session(runtime_dir, session)


def delete_memory_note(runtime_dir: Path | str, session_id: str, note_id: str) -> dict[str, Any]:
    session = get_session(runtime_dir, session_id)
    session["memory"] = [
        row for row in list(session.get("memory") or []) if str(row.get("id")) != note_id
    ]
    return save_session(runtime_dir, session)


def _auto_title(session: Mapping[str, Any], user_text: str) -> str:
    if session.get("title_locked"):
        return str(session.get("title") or "未命名对话")
    current = str(session.get("title") or "")
    if current and current != "新对话":
        return current
    line = user_text.strip().splitlines()[0][:36]
    return line or current or "新对话"


def append_turn(
    runtime_dir: Path | str,
    session_id: str,
    *,
    agent: str,
    user_text: str,
    reply: Mapping[str, Any],
) -> dict[str, Any]:
    session = get_session(runtime_dir, session_id)
    name = _normalize_agent(agent)
    session["active_agent"] = name
    session["title"] = _auto_title(session, user_text)
    now = _now()
    thread = dict((session.get("threads") or {}).get(name) or {"messages": []})
    messages = list(thread.get("messages") or [])
    messages.append(
        {
            "id": new_id("m"),
            "role": "user",
            "text": user_text,
            "agent_id": name,
            "created_at": now,
        }
    )
    messages.append(
        {
            "id": new_id("m"),
            "role": "assistant",
            "text": reply.get("text"),
            "live": bool(reply.get("live")),
            "agent": reply.get("agent"),
            "agent_id": reply.get("agent_id") or name,
            "kind": reply.get("kind"),
            "links": reply.get("links") or [],
            "suggestions": reply.get("suggestions") or [],
            "created_at": now,
        }
    )
    thread["messages"] = messages[-200:]
    threads = dict(session.get("threads") or _empty_threads())
    threads[name] = thread
    session["threads"] = threads
    return save_session(runtime_dir, session)


def session_chat(
    runtime_dir: Path | str,
    session_id: str,
    *,
    message: str,
    agent: str,
    live: bool,
    snapshot: Mapping[str, Any],
    root: Path | str,
    dispatch: Any | None = None,
) -> dict[str, Any]:
    name = _normalize_agent(agent)
    session = get_session(runtime_dir, session_id)
    history = list((session.get("threads") or {}).get(name, {}).get("messages") or [])
    workspace = dict(session.get("workspace") or get_workspace(runtime_dir, str(session.get("workspace_id"))))
    memory_notes: list[str] = [
        f"当前工作区：{workspace.get('title')}（{workspace.get('kind_label') or WORKSPACE_KIND_LABELS.get(str(workspace.get('kind')), '临时')}）"
    ]
    if workspace.get("project_id"):
        memory_notes.append(f"绑定研究项目 {workspace.get('project_id')}")
    if workspace.get("pack_id"):
        memory_notes.append(f"绑定实验包 {workspace.get('pack_id')}")
    memory_notes.extend(
        f"工作区记忆：{row.get('text')}"
        for row in list(workspace.get("memory") or [])
        if row.get("text")
    )
    memory_notes.extend(
        f"本对话记忆：{row.get('text')}"
        for row in list(session.get("memory") or [])
        if row.get("text")
    )
    reply = handle_console_turn(
        message=message,
        history=history,
        snapshot=snapshot,
        runtime_dir=runtime_dir,
        live=live,
        root=root,
        dispatch=dispatch,
        agent=name,
        memory_notes=memory_notes,
    )
    session = append_turn(
        runtime_dir,
        session_id,
        agent=name,
        user_text=message,
        reply=reply,
    )
    return {"ok": True, "reply": reply, "session": session}


def _stage_status(loop: list[Any], stage_id: str) -> dict[str, Any]:
    for row in loop:
        if isinstance(row, dict) and row.get("id") == stage_id:
            return row
    return {}


def _progress(parts: list[bool]) -> float:
    if not parts:
        return 0.0
    return round(sum(1 for item in parts if item) / len(parts), 2)


def _idle_agent(agent: str, *, status: str, headline: str, detail: str) -> dict[str, Any]:
    meta = AGENT_META[agent]
    return {
        **meta,
        "status": status,
        "progress": 0.0,
        "headline": headline,
        "detail": detail,
    }


def build_agent_board(
    root: Path | str,
    *,
    llm_status: Mapping[str, Any] | None = None,
    pack_id: str | None = None,
    scoped: bool = False,
) -> dict[str, Any]:
    snap = build_console_snapshot(root, llm_status=llm_status)
    wanted = str(pack_id or "").strip() or None
    if scoped and not wanted:
        return {
            "ok": True,
            "pack_id": None,
            "scoped": True,
            "unbound": True,
            "c1": {"allowed": False},
            "agents": {
                agent: _idle_agent(
                    agent,
                    status="UNBOUND",
                    headline="本工作区未绑定实验包",
                    detail="绑定实验包后，这里才显示该实验的 Gate / Run / Rubric",
                )
                for agent in AGENTS
            },
        }

    pack: dict[str, Any] = {}
    run_ids = [wanted] if wanted else ["formal_c1_aps_early_concat", "v25d_llm_real_loop"]
    for run_id in run_ids:
        try:
            pack = inspect_local_run(root, run_id)
            if pack.get("available"):
                break
        except (FileNotFoundError, PermissionError, ValueError):
            if wanted:
                return {
                    "ok": True,
                    "pack_id": wanted,
                    "scoped": scoped,
                    "unbound": False,
                    "c1": {"allowed": False},
                    "agents": {
                        agent: _idle_agent(
                            agent,
                            status="MISSING",
                            headline=f"实验包 {wanted} 不在本机",
                            detail="打开实验闭环页核对该 pack，或改绑另一个证据包",
                        )
                        for agent in AGENTS
                    },
                }
            continue

    loop = list(pack.get("loop") or [])
    plan = dict(pack.get("plan") or {})
    gate = dict(pack.get("gate") or {})
    run = dict(pack.get("run") or {})
    rubric = dict(pack.get("rubric") or {})
    claim = dict(pack.get("claim_gate") or {})
    pack_c1 = dict(pack.get("c1") or {})
    c1 = pack_c1 if scoped else dict(snap.get("c1") or pack_c1)

    protocol = _stage_status(loop, "protocol")
    gate_stage = _stage_status(loop, "gate")
    run_stage = _stage_status(loop, "run")
    evidence = _stage_status(loop, "evidence")
    rubric_stage = _stage_status(loop, "rubric")
    memory_stage = _stage_status(loop, "memory")

    planner_ready = bool(plan.get("plan_id") or protocol.get("ready"))
    executor_done = str(run.get("run_state") or "").upper() in {"SUCCEEDED", "COMPLETED", "VALID"}

    agents = {
        "planner": {
            **AGENT_META["planner"],
            "status": str(gate.get("status") or gate_stage.get("status") or ("READY" if planner_ready else "IDLE")),
            "progress": _progress([bool(protocol.get("ready")), bool(plan.get("plan_id")), bool(gate_stage.get("ready"))]),
            "headline": str(plan.get("hypothesis") or plan.get("plan_id") or "还没有下一轮 Plan"),
            "detail": str(gate_stage.get("summary") or gate.get("status") or "等待 Gate"),
        },
        "executor": {
            **AGENT_META["executor"],
            "status": str(run.get("run_state") or run_stage.get("status") or "IDLE"),
            "progress": _progress([bool(run_stage.get("ready")), bool(evidence.get("ready")), executor_done]),
            "headline": str(run.get("run_id") or run_stage.get("summary") or "没有正在执行的 run"),
            "detail": str(run.get("evidence_status") or evidence.get("summary") or "聊天不能直接烧 GPU"),
        },
        "reviewer": {
            **AGENT_META["reviewer"],
            "status": str(
                claim.get("status")
                or rubric.get("review_decision")
                or rubric_stage.get("status")
                or "IDLE"
            ),
            "progress": _progress(
                [bool(rubric_stage.get("ready")), bool(claim.get("status")), bool(memory_stage.get("ready"))]
            ),
            "headline": (
                f"ClaimGate {c1.get('claim_status')}"
                if c1.get("allowed")
                else str(rubric.get("review_decision") or "尚未审阅")
            ),
            "detail": str(
                rubric.get("reasoning_summary")
                or claim.get("reason")
                or "KEEP 不是声称"
            ),
        },
    }
    return {
        "ok": True,
        "pack_id": pack.get("id") or wanted,
        "pack_title": str((pack.get("catalog") or {}).get("title") or pack.get("id") or wanted or ""),
        "scoped": scoped,
        "unbound": False,
        "c1": c1,
        "agents": agents,
    }
