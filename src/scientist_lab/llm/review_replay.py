"""Offline LLM Reviewer REPLAY. Does not execute GPU.

Loads a historical (or stub) run directory: protocol, plan, result, Rubric
review. Calls Reviewer(backend=llm) for a semantic proposal. MemoryWriter
persists only after evidence_refs validate. Never --execute. Never mutates
the source pack Memory.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from scientist_lab.core.decision_rubric import evaluate_rubric
from scientist_lab.core.evidence_validator import EvidenceValidator
from scientist_lab.core.invariants import InvariantError
from scientist_lab.core.memory_writer import MemoryWriter
from scientist_lab.core.reviewer import ReviewRefused, Reviewer
from scientist_lab.core.schema_registry import load_json
from scientist_lab.instrumentation.appender import EventAppender
from scientist_lab.llm.errors import MissingAPIKeyError, RealProviderNotEnabledError
from scientist_lab.llm.gateway import resolve_gateway_provider
from scientist_lab.llm.plan_replay import load_replay_bundle


def _isolated_memory(src: MemoryWriter, dest: Path) -> MemoryWriter:
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("research_memory.json", "strategy_memory.json", "research_trace.json"):
        path = src.root / name
        if path.is_file():
            shutil.copyfile(path, dest / name)
    return MemoryWriter(dest)


def run_llm_review_replay(
    run_dir: Path | str,
    *,
    live: bool = False,
    provider: str = "mock",
    fallback_to_rules: bool = False,
    output: Path | str | None = None,
    events_path: Path | str | None = None,
    persist_proposal: bool = True,
) -> dict[str, Any]:
    """Replay LLM Reviewer against frozen evidence. Never GPU / never execute.

    Writes ``<run-dir>/.llm_review_replay/``. Does not invent live-API success.
    """
    bundle = load_replay_bundle(run_dir)
    root: Path = bundle["root"]
    work = root / ".llm_review_replay"
    work.mkdir(parents=True, exist_ok=True)
    source = dict(bundle.get("source") or {})
    protocol = dict(bundle["protocol"])
    plan = dict(bundle["previous_plan"])
    evidence_blob = dict(bundle.get("evidence") or {})
    result = dict(evidence_blob.get("result") or {})
    if not result and (root / "result.json").is_file():
        result = load_json(root / "result.json")
    baseline: dict[str, Any] = {}
    if (root / "baseline_metrics.json").is_file():
        baseline = load_json(root / "baseline_metrics.json")
    locked_review = str(bundle.get("last_review_decision") or "")
    lessons_before = set(bundle["memory"].load_lessons())
    isolated = _isolated_memory(bundle["memory"], work / "memory")
    report: dict[str, Any] = {
        "ok": False,
        "exam": source.get("exam_point") or "llm_review_replay",
        "run_dir": str(root),
        "origin_run_dir": source.get("origin_run_dir"),
        "live": bool(live),
        "provider": "openai-compatible" if live else provider,
        "execute": False,
        "gpu": False,
        "fail_closed": False,
        "rubric_review_decision": locked_review,
        "llm_review_decision": None,
        "document_hypothesis_status": None,
        "semantic_proposal": None,
        "writer": None,
        "memory_unchanged": True,
        "evidence_source": source or None,
    }

    events = EventAppender(Path(events_path) if events_path else work / "events.jsonl")
    try:
        if not result:
            raise ReviewRefused("review replay requires result.json")
        from scientist_lab.adapters.dfine.adapter import DFINEAdapter

        contract = DFINEAdapter().materialize_contract(plan, protocol)
        evidence = EvidenceValidator().validate(
            result,
            protocol=protocol,
            expected_artifacts=list(contract.get("expected_artifacts") or []),
        )
        rubric = evaluate_rubric(
            protocol,
            current_metrics=result.get("metrics") or {},
            baseline_metrics={
                k: v for k, v in baseline.items() if isinstance(v, (int, float))
            }
            or None,
        )
        llm_provider = resolve_gateway_provider(
            live=live, provider=None if live else provider
        )
        packet = Reviewer(
            backend="llm",
            provider=llm_provider,
            fallback_to_rules=fallback_to_rules,
            live=live,
        ).review(
            result=result,
            evidence=evidence,
            rubric=rubric,
            contract=contract,
            protocol=protocol,
            plan=plan,
            memory={
                "lessons": isolated.load_lessons(),
                "strategies": isolated.load_strategies(),
            },
            events=events,
        )
        proposal = dict(packet.semantic_proposal or {})
        report.update(
            {
                "ok": True,
                "llm_review_decision": packet.review_decision,
                "document_hypothesis_status": packet.document.get("hypothesis_status"),
                "document_review_decision": packet.document.get("review_decision"),
                "semantic_proposal": proposal,
                "source": packet.source,
                "llm_trace": packet.llm_trace,
                "rubric_locked": packet.review_decision == locked_review
                or not locked_review,
            }
        )
        if locked_review and packet.review_decision != locked_review:
            report["ok"] = False
            report["fail_closed"] = True
            report["error"] = (
                f"LLM must not override Rubric review_decision "
                f"{locked_review} → {packet.review_decision}"
            )
        if persist_proposal and proposal and report["ok"]:
            lesson = packet.research_lessons[0] if packet.research_lessons else {}
            try:
                written = isolated.persist_semantic_proposal(
                    proposal,
                    run_id=str(result.get("run_id") or ""),
                    review_decision=packet.review_decision,
                    lesson_type=str(lesson.get("type") or "negative_evidence"),
                    module=str((lesson.get("scope") or {}).get("module") or "unknown"),
                    task=str((lesson.get("scope") or {}).get("task") or "rgbt_detection"),
                    metric=str(
                        ((protocol.get("objective") or {}).get("primary") or {}).get(
                            "metric"
                        )
                        or "APS"
                    ),
                    delta=rubric.primary_delta,
                )
                report["writer"] = {
                    "accepted": True,
                    "lesson_id": written.get("lesson_id"),
                    "evidence": written.get("evidence"),
                    "created_from": written.get("created_from"),
                }
                events.append(
                    {
                        "project_id": protocol["project_id"],
                        "run_id": str(result.get("run_id") or ""),
                        "plan_id": plan.get("plan_id"),
                        "protocol_version": protocol.get("protocol_version"),
                        "fingerprint_id": protocol.get("fingerprint_id"),
                        "event_type": "memory_write",
                        "actor_role": "reviewer",
                        "phase": "review",
                        "evidence_refs": [str(result.get("run_id") or "")],
                        "payload": {
                            "source": "memory_writer",
                            "lesson_ids": [written.get("lesson_id")],
                            "proposal_only_until_refs_validated": True,
                        },
                    }
                )
            except InvariantError as exc:
                report["ok"] = False
                report["fail_closed"] = True
                report["writer"] = {"accepted": False, "refused": str(exc)}
                report["error"] = str(exc)
    except (
        ReviewRefused,
        MissingAPIKeyError,
        RealProviderNotEnabledError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        report["fail_closed"] = True
        report["error"] = str(exc)
        report["ok"] = False

    report["memory_unchanged"] = set(bundle["memory"].load_lessons()) == lessons_before
    report["rubric_review_decision"] = locked_review
    out = Path(output) if output is not None else work / "replay_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    report["report_path"] = str(out)
    return report
