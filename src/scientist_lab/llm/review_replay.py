"""Offline LLM Reviewer REPLAY. Does not execute GPU.

Loads a historical (or stub) run directory: protocol, plan, result, Rubric
review. Calls Reviewer(backend=llm) for a semantic proposal. MemoryWriter
persists only after evidence_refs validate. Never --execute. Never mutates
the source pack Memory.
"""

from __future__ import annotations

import json
import os
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
from scientist_lab.llm.errors import (
    LLMError,
    MissingAPIKeyError,
    RealProviderNotEnabledError,
)
from scientist_lab.llm.gateway import resolve_gateway_provider
from scientist_lab.llm.plan_replay import load_replay_bundle


def _runtime_dir() -> Path:
    override = str(os.environ.get("SCIENTIST_LAB_RUNTIME_DIR") or "").strip()
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[3] / "runtime"


def _write_fail_closed_pack(root: Path, report: dict[str, Any]) -> None:
    """Persist a fail-closed sidecar. Never writes semantic_review.json."""
    payload = {
        "ok": False,
        "live": bool(report.get("live")),
        "fail_closed": True,
        "role": "reviewer",
        "error": report.get("error"),
        "schema_issues": report.get("schema_issues"),
        "tls_verify_source": report.get("tls_verify_source"),
        "tls_verified": report.get("tls_verified"),
        "semantic_review_written": False,
        "review_json_unchanged": True,
        "claim_gate_unchanged": True,
        "note": (
            "DecisionRubric KEEP/DISCARD remains rules-owned. "
            "Live Reviewer did not overwrite it. No mock success."
        ),
    }
    path = root / "reviewer_live_fail_closed.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    report["pack_write"] = {
        "reviewer_live_fail_closed": str(path),
        "semantic_review": None,
        "review_json_unchanged": True,
    }


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
    persist_pack: bool = False,
) -> dict[str, Any]:
    """Replay LLM Reviewer against frozen evidence. Never GPU / never execute.

    Writes ``<run-dir>/.llm_review_replay/``. Does not invent live-API success.
    ``persist_pack`` may additionally write ``semantic_review.json`` and a
    semantic Memory lesson into the source pack. It never overwrites Rubric
    ``review.json`` KEEP/DISCARD or ClaimGate.
    """
    if live:
        from scientist_lab.llm.runtime_secrets import apply_runtime_llm_env

        apply_runtime_llm_env(_runtime_dir())
        current_timeout = float(os.environ.get("LLM_TIMEOUT_SECONDS") or 60.0)
        if current_timeout < 180.0:
            os.environ["LLM_TIMEOUT_SECONDS"] = "180"
    bundle = load_replay_bundle(run_dir)
    root: Path = bundle["root"]
    work = root / ".llm_review_replay"
    work.mkdir(parents=True, exist_ok=True)
    source = dict(bundle.get("source") or {})
    protocol = dict(bundle["protocol"])
    if (root / "plan.json").is_file():
        plan = load_json(root / "plan.json")
    else:
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
        "persist_pack": bool(persist_pack),
        "plan_id": plan.get("plan_id"),
        "rubric_review_decision": locked_review,
        "llm_review_decision": None,
        "document_hypothesis_status": None,
        "semantic_proposal": None,
        "writer": None,
        "memory_unchanged": True,
        "evidence_source": source or None,
        "tls_verify_source": None,
        "tls_verified": True,
    }
    if live:
        from scientist_lab.llm.http_transport import resolve_tls_verify

        _verify, tls_source = resolve_tls_verify()
        report["tls_verify_source"] = tls_source
        report["tls_verified"] = _verify is not False

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
                if persist_pack:
                    pack_written = bundle["memory"].persist_semantic_proposal(
                        proposal,
                        run_id=str(result.get("run_id") or ""),
                        review_decision=packet.review_decision,
                        lesson_type=str(lesson.get("type") or "negative_evidence"),
                        module=str(
                            (lesson.get("scope") or {}).get("module") or "unknown"
                        ),
                        task=str(
                            (lesson.get("scope") or {}).get("task") or "rgbt_detection"
                        ),
                        metric=str(
                            ((protocol.get("objective") or {}).get("primary") or {}).get(
                                "metric"
                            )
                            or "APS"
                        ),
                        delta=rubric.primary_delta,
                    )
                    sidecar = {
                        "schema_version": "1.0.0",
                        "source": "llm",
                        "proposal_only": True,
                        "keep_is_not_claim": True,
                        "g2_not_claimed": True,
                        "locked_review_decision": packet.review_decision,
                        "document_hypothesis_status": packet.document.get(
                            "hypothesis_status"
                        ),
                        "semantic_proposal": proposal,
                        "run_id": str(result.get("run_id") or ""),
                        "plan_id": plan.get("plan_id"),
                        "primary_metric": (
                            ((protocol.get("objective") or {}).get("primary") or {}).get(
                                "metric"
                            )
                        ),
                    }
                    if packet.llm_trace:
                        sidecar["llm_trace"] = {
                            key: packet.llm_trace.get(key)
                            for key in (
                                "backend",
                                "provider",
                                "model",
                                "prompt_hash",
                                "rubric_locked",
                            )
                        }
                    sidecar_path = root / "semantic_review.json"
                    sidecar_path.write_text(
                        json.dumps(sidecar, ensure_ascii=False, indent=2, default=str)
                        + "\n",
                        encoding="utf-8",
                    )
                    fail_path = root / "reviewer_live_fail_closed.json"
                    if fail_path.is_file():
                        fail_path.unlink()
                    report["pack_write"] = {
                        "semantic_review": str(sidecar_path),
                        "memory_accepted": True,
                        "lesson_id": pack_written.get("lesson_id"),
                        "review_json_unchanged": True,
                        "reviewer_live_fail_closed_removed": True,
                    }
            except InvariantError as exc:
                report["ok"] = False
                report["fail_closed"] = True
                report["writer"] = {"accepted": False, "refused": str(exc)}
                report["error"] = str(exc)
    except (
        ReviewRefused,
        MissingAPIKeyError,
        RealProviderNotEnabledError,
        LLMError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        report["fail_closed"] = True
        report["error"] = str(exc)
        report["ok"] = False
        cause = getattr(exc, "__cause__", None)
        issues = list(getattr(cause, "issues", None) or getattr(exc, "issues", None) or [])
        if issues:
            report["schema_issues"] = issues[:8]
        prior = list(getattr(cause, "prior_issues", None) or [])
        if prior:
            report["prior_schema_errors"] = prior[:8]

    if persist_pack and report.get("fail_closed"):
        _write_fail_closed_pack(root, report)

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
