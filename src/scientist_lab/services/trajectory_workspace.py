"""Read-only trajectory workspace for the Web console.

Shows Canonical Research Events (research_events.jsonl) and the ATDP
six-tuple projection. Does not write the ledger, does not drive
Planner / Gate / GPU, and does not treat detection scores as rewards.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from scientist_lab.llm.trajectory_export import (
    EXPORTER_VERSION,
    SCHEMA_VERSION,
    build_steps,
    build_story,
    load_events,
    load_run_bundle,
)
from scientist_lab.services.local_run_inspector import (
    ALLOWED_REL_PREFIXES,
    _inspect_root_for,
    _posix,
    is_allowed_relpath,
    list_local_runs,
    resolve_allowed_path,
    resolve_project_root,
)

EVENT_LIMIT = 500
MAX_PREVIEW = 180


def _clip(value: Any, limit: int = MAX_PREVIEW) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, default=str)
    else:
        text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            n += 1
    return n


def _events_dir(pack: Path, inspect: Path) -> Path:
    for root in (inspect, pack):
        if (root / "research_events.jsonl").is_file():
            return root
    return inspect


def _bundle_dir(pack: Path, inspect: Path) -> Path:
    events_root = _events_dir(pack, inspect)
    if (events_root / "plan.json").is_file() or (events_root / "protocol.json").is_file():
        return events_root
    if (inspect / "plan.json").is_file() or (inspect / "protocol.json").is_file():
        return inspect
    return events_root


def _candidate_from_pack(
    *,
    item_id: str,
    relpath: str,
    inspect_subdir: str,
    title: str,
    kind: str,
    blurb: str,
    pack: Path,
) -> dict[str, Any] | None:
    inspect = _inspect_root_for(pack, inspect_subdir)
    events_root = _events_dir(pack, inspect)
    events_path = events_root / "research_events.jsonl"
    bundle_root = _bundle_dir(pack, inspect)
    has_events = events_path.is_file()
    has_plan = (bundle_root / "plan.json").is_file()
    has_protocol = (bundle_root / "protocol.json").is_file()
    export_traces = bundle_root / "export" / "traces"
    exported = False
    if export_traces.is_dir():
        exported = any(export_traces.glob("*.jsonl"))
    if not (has_events or has_plan or has_protocol or exported):
        return None
    if inspect == pack:
        inspect_rel = ""
    else:
        try:
            inspect_rel = _posix(str(inspect.relative_to(pack)))
        except ValueError:
            inspect_rel = inspect.name
    return {
        "id": item_id,
        "relpath": _posix(relpath),
        "inspect_subdir": inspect_rel,
        "title": title,
        "kind": kind,
        "blurb": blurb,
        "event_count": _count_jsonl(events_path),
        "has_events": has_events,
        "has_plan": has_plan,
        "exported_traces": exported,
        "mappable": has_events or has_plan or has_protocol,
    }


def _autonomous_candidates(project: Path) -> list[dict[str, Any]]:
    root = project / ".run" / "autonomous"
    if not root.is_dir():
        return []
    items: list[dict[str, Any]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        events_path = child / "research_events.jsonl"
        if not (
            events_path.is_file()
            or (child / "plan.json").is_file()
            or (child / "protocol.json").is_file()
            or (child / "campaign.json").is_file()
        ):
            continue
        campaign = {}
        campaign_path = child / "campaign.json"
        if campaign_path.is_file():
            try:
                campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                campaign = {}
        title = str(
            campaign.get("title")
            or campaign.get("experiment_id")
            or child.name
        )
        items.append(
            {
                "id": f"campaign_{child.name}",
                "relpath": f".run/autonomous/{child.name}",
                "inspect_subdir": "",
                "title": title,
                "kind": "campaign",
                "blurb": "自主战役账本（只读）。",
                "event_count": _count_jsonl(events_path),
                "has_events": events_path.is_file(),
                "has_plan": (child / "plan.json").is_file(),
                "exported_traces": (child / "export" / "traces").is_dir(),
                "mappable": events_path.is_file() or (child / "plan.json").is_file(),
            }
        )
    return items


def list_trajectories(root: Path | str) -> dict[str, Any]:
    project = resolve_project_root(root)
    catalog = list_local_runs(project)
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in catalog.get("items") or []:
        if not entry.get("available"):
            continue
        relpath = str(entry.get("relpath") or "")
        if not is_allowed_relpath(relpath):
            continue
        pack = resolve_allowed_path(project, relpath)
        if not pack.exists():
            continue
        row = _candidate_from_pack(
            item_id=str(entry.get("id") or pack.name),
            relpath=relpath,
            inspect_subdir=str(entry.get("inspect_subdir") or ""),
            title=str(entry.get("title") or entry.get("id") or pack.name),
            kind=str(entry.get("kind") or "run"),
            blurb=str(entry.get("blurb") or ""),
            pack=pack,
        )
        if row is None:
            continue
        items.append(row)
        seen.add(row["id"])
        seen.add(row["relpath"])

    for extra in _autonomous_candidates(project):
        if extra["id"] in seen or extra["relpath"] in seen:
            continue
        items.append(extra)
        seen.add(extra["id"])
        seen.add(extra["relpath"])

    items.sort(key=lambda row: (0 if row.get("has_events") else 1, str(row.get("title") or "")))
    return {
        "items": items,
        "total": len(items),
        "exporter_version": EXPORTER_VERSION,
        "schema_version": SCHEMA_VERSION,
        "note": (
            "科研轨迹权威记录是 research_events.jsonl；"
            "映射后的六元组是只读投影，不回写账本，r 恒为 null。"
        ),
        "allowed_prefixes": list(ALLOWED_REL_PREFIXES),
    }


def _lookup(root: Path, run_id: str) -> dict[str, Any]:
    catalog = list_trajectories(root)
    wanted = str(run_id or "").strip()
    for row in catalog["items"]:
        if row["id"] == wanted or row["relpath"] == wanted:
            return row
    raise FileNotFoundError(f"trajectory not found: {run_id}")


def _event_row(index: int, event: Mapping[str, Any]) -> dict[str, Any]:
    payload = event.get("payload") if isinstance(event.get("payload"), Mapping) else {}
    decision = (
        event.get("decision_summary")
        if isinstance(event.get("decision_summary"), Mapping)
        else {}
    )
    summary = (
        decision.get("selected_action")
        or payload.get("review_decision")
        or payload.get("status")
        or payload.get("action")
        or payload.get("how_id")
        or ""
    )
    return {
        "index": index,
        "event_id": event.get("event_id") or f"event_{index}",
        "event_type": event.get("event_type"),
        "actor_role": event.get("actor_role"),
        "phase": event.get("phase"),
        "ts": event.get("ts"),
        "run_id": event.get("run_id"),
        "plan_id": event.get("plan_id"),
        "reconstructed": bool(event.get("reconstructed")),
        "summary": _clip(summary, 200),
        "record": dict(event),
    }


def _step_row(step: Mapping[str, Any]) -> dict[str, Any]:
    meta = step.get("m") if isinstance(step.get("m"), Mapping) else {}
    return {
        "step_index": step.get("step_index"),
        "step_kind": step.get("step_kind"),
        "r": step.get("r"),
        "reconstructed": bool(meta.get("reconstructed")),
        "event_id": meta.get("event_id"),
        "source_event_types": list(meta.get("source_event_types") or []),
        "o_preview": _clip(step.get("o")),
        "h_preview": _clip(step.get("h")),
        "a_preview": _clip(step.get("a")),
        "y_preview": _clip(step.get("y")),
        "record": dict(step),
    }


def show_trajectory(root: Path | str, run_id: str) -> dict[str, Any]:
    project = resolve_project_root(root)
    item = _lookup(project, run_id)
    pack = resolve_allowed_path(project, str(item["relpath"]))
    inspect = _inspect_root_for(pack, str(item.get("inspect_subdir") or ""))
    events_root = _events_dir(pack, inspect)
    bundle_root = _bundle_dir(pack, inspect)
    events = load_events(events_root)
    truncated = False
    if len(events) > EVENT_LIMIT:
        events = events[:EVENT_LIMIT]
        truncated = True
    event_rows = [_event_row(i + 1, row) for i, row in enumerate(events)]

    mapped_steps: list[dict[str, Any]] = []
    story: dict[str, Any] | None = None
    mapping_error: str | None = None
    reconstructed = not bool(events)
    try:
        bundle = load_run_bundle(bundle_root)
        reconstructed = bool(bundle.get("reconstructed"))
        steps = build_steps(bundle)
        mapped_steps = [_step_row(step) for step in steps]
        story = build_story(bundle, steps)
    except Exception as exc:  # noqa: BLE001 — UI must still show the ledger
        mapping_error = str(exc)

    return {
        **item,
        "run_id": (story or {}).get("run_id") or item["id"],
        "events": event_rows,
        "events_truncated": truncated,
        "mapped_steps": mapped_steps,
        "story": story,
        "mapping_error": mapping_error,
        "reconstructed": reconstructed,
        "exporter_version": EXPORTER_VERSION,
        "schema_version": SCHEMA_VERSION,
        "events_path": str(events_root / "research_events.jsonl"),
        "bundle_dir": str(bundle_root),
        "note": (
            "科研轨迹 = Canonical Events；"
            "映射后 = ATDP ⟨o,h,a,y,r,m⟩ 只读投影。"
            "检测分数不作对照标签，live 导出 r=null。"
        ),
    }
