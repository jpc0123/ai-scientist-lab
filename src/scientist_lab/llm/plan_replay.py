"""Offline LLM Planner REPLAY. Does not execute GPU.

Loads a historical (or stub) run directory: protocol, previous plan, written
Memory. Calls Planner(backend=llm) → Adapter HOW → Gate. Never --execute.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping

from scientist_lab.core.memory_writer import MemoryWriter
from scientist_lab.core.planner import PlanRefused, propose_and_gate_next
from scientist_lab.core.schema_registry import load_json
from scientist_lab.instrumentation.appender import EventAppender
from scientist_lab.llm.errors import MissingAPIKeyError, RealProviderNotEnabledError
from scientist_lab.llm.gateway import resolve_gateway_provider
from scientist_lab.llm.planner_contract import infer_discarded_modules


def load_replay_bundle(run_dir: Path | str) -> dict[str, Any]:
    root = Path(run_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"replay run-dir not found: {root}")
    protocol_path = root / "protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(f"protocol.json missing in {root}")
    protocol = load_json(protocol_path)
    if (root / "previous_plan.json").is_file():
        previous = load_json(root / "previous_plan.json")
    elif (root / "plan.json").is_file():
        previous = load_json(root / "plan.json")
    else:
        raise FileNotFoundError(f"previous_plan.json or plan.json missing in {root}")
    memory = MemoryWriter(root / "memory")
    review: dict[str, Any] = {}
    if (root / "review.json").is_file():
        review = load_json(root / "review.json")
    last_review = str(review.get("review_decision") or "") or None
    parent = str(
        previous.get("parent_run_id")
        or (load_json(root / "experiment_run.json").get("run_id") if (root / "experiment_run.json").is_file() else "")
        or previous.get("plan_id")
        or ""
    )
    # Prefer the contract run id of the previous plan when present.
    if previous.get("plan_id") and not str(previous.get("parent_run_id") or "").startswith("run_"):
        maybe_run = f"run_{previous['plan_id']}"
        lessons = memory.load_lessons()
        if any(maybe_run in list(row.get("created_from") or []) for row in lessons.values()):
            parent = maybe_run
        elif parent.startswith("run_"):
            pass
        elif any(parent in list(row.get("created_from") or []) for row in lessons.values()):
            pass
        else:
            for row in lessons.values():
                created = list(row.get("created_from") or [])
                if created:
                    parent = str(created[0])
                    break
    evidence: dict[str, Any] = {}
    if (root / "result.json").is_file():
        evidence["result"] = load_json(root / "result.json")
    if review:
        evidence["review_decision"] = last_review
    source: dict[str, Any] = {}
    if (root / "SOURCE.json").is_file():
        source = load_json(root / "SOURCE.json")
    discarded = infer_discarded_modules(
        previous_plan=previous,
        last_review_decision=last_review,
        memory_catalog={
            "lessons": memory.load_lessons(),
            "strategies": memory.load_strategies(),
        },
    )
    return {
        "root": root,
        "protocol": protocol,
        "previous_plan": previous,
        "memory": memory,
        "last_review_decision": last_review,
        "parent_run_id": parent,
        "evidence": evidence,
        "source": source,
        "discarded_modules": discarded,
    }


def _isolated_memory(src: MemoryWriter, dest: Path) -> MemoryWriter:
    """Copy written Memory so REPLAY Gate citation cannot mutate the source pack."""
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("research_memory.json", "strategy_memory.json", "research_trace.json"):
        path = src.root / name
        if path.is_file():
            shutil.copyfile(path, dest / name)
    return MemoryWriter(dest)


def _selected_module(plan: Mapping[str, Any] | None) -> str | None:
    if not plan:
        return None
    scope = list(plan.get("modification_scope") or [])
    return str(scope[0]) if scope else None


def run_llm_plan_replay(
    run_dir: Path | str,
    *,
    live: bool = False,
    provider: str = "mock",
    fallback_to_rules: bool = False,
    output: Path | str | None = None,
    events_path: Path | str | None = None,
    ab: bool = True,
) -> dict[str, Any]:
    """Replay LLM Planner against frozen evidence. Never GPU / never execute.

    Default writes A/B (rules vs LLM) into replay_report.json under
    ``<run-dir>/.llm_plan_replay/``. Does not invent live-API success.
    """
    bundle = load_replay_bundle(run_dir)
    root: Path = bundle["root"]
    work = root / ".llm_plan_replay"
    work.mkdir(parents=True, exist_ok=True)
    discarded = list(bundle.get("discarded_modules") or [])
    source = dict(bundle.get("source") or {})
    report: dict[str, Any] = {
        "ok": False,
        "exam": source.get("exam_point") or "llm_plan_replay",
        "run_dir": str(root),
        "origin_run_dir": source.get("origin_run_dir"),
        "live": bool(live),
        "provider": "openai-compatible" if live else provider,
        "execute": False,
        "gpu": False,
        "fail_closed": False,
        "discarded_module": discarded[0] if discarded else None,
        "discarded_modules": discarded,
        "memory_refs": None,
        "gate": {"status": "REJECTED", "reasons": []},
        "rules_plan": None,
        "llm_plan": None,
        "ab": None,
        "evidence_source": source or None,
    }

    rules_memory = _isolated_memory(bundle["memory"], work / "memory_rules")
    llm_memory = _isolated_memory(bundle["memory"], work / "memory_llm")
    lessons_before = set(bundle["memory"].load_lessons())
    review_before = bundle["last_review_decision"]

    rules_result: dict[str, Any] | None = None
    try:
        rules_result = propose_and_gate_next(
            protocol=bundle["protocol"],
            memory=rules_memory,
            previous_plan=bundle["previous_plan"],
            parent_run_id=str(bundle["parent_run_id"]),
            last_review_decision=bundle["last_review_decision"],
            backend="rules",
        )
        report["rules_plan"] = rules_result["plan"]
        report["rules_gate"] = rules_result["gate"]
        report["rules_source"] = rules_result["source"]
    except PlanRefused as exc:
        report["rules_error"] = str(exc)

    events = EventAppender(Path(events_path) if events_path else work / "events.jsonl")
    try:
        llm_provider = resolve_gateway_provider(live=live, provider=None if live else provider)
        result = propose_and_gate_next(
            protocol=bundle["protocol"],
            memory=llm_memory,
            previous_plan=bundle["previous_plan"],
            parent_run_id=str(bundle["parent_run_id"]),
            last_review_decision=bundle["last_review_decision"],
            events=events,
            backend="llm",
            provider=llm_provider,
            fallback_to_rules=fallback_to_rules,
            live=live,
        )
        llm_plan = result["plan"]
        report.update(
            {
                "ok": result["gate"]["status"] == "APPROVED",
                "plan": llm_plan,
                "llm_plan": llm_plan,
                "decision_summary": result["decision_summary"],
                "source": result["source"],
                "contract": {
                    "run_id": result["contract"].get("run_id"),
                    "allowed_changes": result["contract"].get("allowed_changes"),
                    "how": (result["contract"].get("materialization") or {}).get("how"),
                },
                "gate": result["gate"],
                "llm_trace": llm_plan.get("llm_trace"),
                "candidate_experiments": llm_plan.get("candidate_experiments") or [],
                "memory_refs": llm_plan.get("memory_refs"),
            }
        )
        if discarded and _selected_module(llm_plan) in discarded:
            report["ok"] = False
            report["fail_closed"] = True
            report["error"] = (
                f"selected {_selected_module(llm_plan)!r} repeats discarded {discarded}"
            )
    except (PlanRefused, MissingAPIKeyError, RealProviderNotEnabledError, FileNotFoundError) as exc:
        report["fail_closed"] = True
        report["error"] = str(exc)
        report["gate"] = {"status": "REJECTED", "reasons": [str(exc)]}
        report["ok"] = False

    report["memory_unchanged"] = set(bundle["memory"].load_lessons()) == lessons_before
    report["review_decision_unchanged"] = bundle["last_review_decision"] == review_before
    report["budget_class"] = (report.get("llm_plan") or {}).get("budget_class") or (
        bundle["previous_plan"].get("budget_class")
    )

    if ab:
        rules_plan = report.get("rules_plan") or {}
        llm_plan = report.get("llm_plan") or {}
        rules_mod = _selected_module(rules_plan)
        llm_mod = _selected_module(llm_plan)
        report["ab"] = {
            "compared": True,
            "rules_selected_module": rules_mod,
            "llm_selected_module": llm_mod,
            "same_selected_module": bool(rules_mod and llm_mod and rules_mod == llm_mod),
            "rules_has_candidate_experiments": bool(rules_plan.get("candidate_experiments")),
            "llm_candidate_count": len(llm_plan.get("candidate_experiments") or []),
            "rules_decision_summary": (rules_plan.get("decision_summary") or {}),
            "llm_decision_summary": (llm_plan.get("decision_summary") or {}),
            "rules_is_template_switch": "rules-first Planner" in str(
                (rules_plan.get("decision_summary") or {}).get("decision_basis") or []
            ),
            "llm_has_selection_rationale": bool(
                (llm_plan.get("decision_summary") or {}).get("decision_basis")
            ),
            "historical_rules_dead_cycle": source.get("historical_rules_dead_cycle"),
            "note": (
                "Rules Planner picks the first remaining editable token after DISCARD "
                "(historically neck→fusion→neck). LLM must cite memory_refs, not "
                "repeat discarded selected, and attach >=2 candidates."
            ),
        }

    out = Path(output) if output is not None else work / "replay_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    report["report_path"] = str(out)
    return report
