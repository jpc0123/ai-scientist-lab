"""Gated DFINE Adapter run: GateEngine then execute. No Planner."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from scientist_lab.adapters.base import MaterializeRejected
from scientist_lab.adapters.dfine.adapter import DFINEAdapter
from scientist_lab.adapters.dfine.cuda_runner import make_cuda_live_runner
from scientist_lab.adapters.dfine.fingerprint import compute_fingerprint
from scientist_lab.core.decision_rubric import evaluate_rubric
from scientist_lab.core.evidence_validator import EvidenceValidator
from scientist_lab.core.exception_handler import next_exception_action
from scientist_lab.core.gate_engine import GateEngine, GateStatus
from scientist_lab.core.git_manager import GitManager
from scientist_lab.core.memory_writer import MemoryWriter
from scientist_lab.core.result_parser import ResultParser
from scientist_lab.core.reviewer import ReviewRefused, Reviewer
from scientist_lab.core.schema_registry import load_json, validate_named
from scientist_lab.core.scientific_outcome import project_scientific_outcome
from scientist_lab.core.state_machine import ReviewDecisionValue, RunState
from scientist_lab.instrumentation import EventAppender


def _emit(
    events: EventAppender,
    *,
    protocol: Mapping[str, Any],
    plan: Mapping[str, Any],
    event_type: str,
    actor_role: str,
    phase: str,
    payload: Mapping[str, Any],
    run_id: str | None = None,
    decision_summary: Mapping[str, Any] | None = None,
    evidence_refs: list[str] | None = None,
    reconstructed: bool = False,
) -> None:
    event: dict[str, Any] = {
        "project_id": protocol["project_id"],
        "run_id": run_id,
        "plan_id": plan.get("plan_id"),
        "protocol_version": protocol.get("protocol_version"),
        "fingerprint_id": protocol.get("fingerprint_id"),
        "event_type": event_type,
        "actor_role": actor_role,
        "phase": phase,
        "payload": dict(payload),
        "reconstructed": reconstructed,
    }
    if decision_summary is not None:
        event["decision_summary"] = dict(decision_summary)
    if evidence_refs:
        event["evidence_refs"] = list(evidence_refs)
    events.append(event)


def _resolve_execute_dir(output: Path) -> Path:
    """REPLAY: recover metrics from output_dir or output_dir/run."""
    run_dir = output / "run"
    if (run_dir / "metrics.json").is_file():
        return run_dir
    if (output / "metrics.json").is_file():
        return output
    return run_dir


def run_gated_dfine(
    *,
    protocol: Mapping[str, Any],
    plan: Mapping[str, Any],
    output_dir: Path | str,
    events_path: Path | str | None = None,
    execute: bool = False,
    experiments: Any | None = None,
    require_live_ready: bool = False,
    live_runner: Any | None = None,
    baseline_metrics: Mapping[str, Any] | None = None,
    git_manager: GitManager | None = None,
    memory_writer: MemoryWriter | None = None,
    review: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    # Injected review cannot mint KEEP/DISCARD or lessons; Reviewer runs only on VALID.
    _ = review
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    events = EventAppender(Path(events_path) if events_path else output / "research_events.jsonl")
    writer = memory_writer or MemoryWriter(output / "memory")
    adapter = DFINEAdapter(events=events)
    try:
        contract = adapter.materialize_contract(plan, protocol)
    except MaterializeRejected as exc:
        gate = {
            "status": GateStatus.REJECTED,
            "reasons": [str(exc)],
            "fingerprint_comparable": None,
        }
        _emit(
            events,
            protocol=protocol,
            plan=plan,
            event_type="gate_decision",
            actor_role="gate",
            phase="planning",
            payload=gate,
        )
        return {"gate": gate, "executed": False, "dry_run": not execute}
    bound_fp = compute_fingerprint(protocol, None)
    lessons = writer.load_lessons()
    strategies = writer.load_strategies()
    memory_catalog = (
        {"lessons": lessons, "strategies": strategies}
        if (lessons or strategies)
        else None
    )
    verdict = GateEngine().evaluate(
        protocol,
        plan,
        contract,
        bound_fingerprint=bound_fp,
        memory=memory_catalog,
    )
    report: dict[str, Any] = {
        "gate": verdict.to_dict(),
        "contract_run_id": contract.get("run_id"),
        "executed": False,
        "dry_run": not execute,
        "review_decision": ReviewDecisionValue.PENDING.value,
    }
    _emit(
        events,
        protocol=protocol,
        plan=plan,
        event_type="gate_decision",
        actor_role="gate",
        phase="planning",
        payload=verdict.to_dict(),
        run_id=contract.get("run_id"),
    )
    if verdict.status != GateStatus.APPROVED:
        return report
    if memory_catalog is not None:
        writer.record_plan_citation(plan)

    runner = live_runner
    if execute and runner is None:
        if experiments is None:
            from scientist_lab.services.experiment_service import ExperimentService

            experiments = ExperimentService()
        runner = make_cuda_live_runner(
            experiments,
            execute=True,
            require_live_ready=require_live_ready,
        )
    execute_dir = _resolve_execute_dir(output)
    handle = adapter.execute(
        contract,
        protocol,
        output_dir=execute_dir,
        dry_run=not execute,
        live_runner=runner,
        expected_fingerprint=bound_fp,
    )
    recovered = bool((handle.get("run_view") or {}).get("source") == "existing_artifacts")
    result = ResultParser().from_handle(contract, handle)
    evidence = EvidenceValidator().validate(
        result,
        protocol=protocol,
        expected_artifacts=list(contract.get("expected_artifacts") or []),
        fingerprint_comparable=handle.get("fingerprint_comparable"),
        dry_run=bool(handle.get("dry_run")),
        handle_status=str(handle.get("status") or ""),
    )
    report["executed"] = True
    report["recovered"] = recovered
    report["handle"] = {
        "status": handle.get("status"),
        "fingerprint_comparable": handle.get("fingerprint_comparable"),
        "output_dir": handle.get("output_dir"),
        "recovered": recovered,
    }
    report["evidence"] = evidence.to_dict()
    report["orchestration_hint"] = next_exception_action(
        result, evidence_status=evidence.evidence_status
    )

    rubric_payload = None
    rubric = None
    outcome = project_scientific_outcome(
        run_state=RunState.COMPLETED
        if str(result.get("execution", {}).get("status")) == "success"
        and evidence.evidence_status == "VALID"
        else RunState.FAILED
        if evidence.evidence_status == "NOT_APPLICABLE"
        else RunState.COMPLETED,
        evidence_status=evidence.evidence_status,
        primary_delta=None,
        constraints_ok=True,
    )
    if evidence.review_allowed:
        rubric = evaluate_rubric(
            protocol,
            current_metrics=result.get("metrics") or {},
            baseline_metrics=baseline_metrics,
        )
        rubric_payload = {
            "objective_check": rubric.objective_check,
            "constraint_check": rubric.constraint_check,
            "primary_delta": rubric.primary_delta,
            "constraints_ok": rubric.constraints_ok,
            "suggest_validate": rubric.suggest_validate,
            "suggest_discard_threshold": rubric.suggest_discard_threshold,
            "keep_threshold_ok": rubric.keep_threshold_ok,
        }
        report["rubric"] = rubric_payload
        outcome = project_scientific_outcome(
            run_state=RunState.COMPLETED,
            evidence_status=evidence.evidence_status,
            primary_delta=rubric.primary_delta,
            constraints_ok=rubric.constraints_ok,
        )
        result = dict(result)
        result["scientific_outcome"] = outcome
        result["constraints_check"] = {
            "ok": rubric.constraints_ok,
            "violations": [
                key for key, val in rubric.constraint_check.items() if val == "FAIL"
            ],
        }
        validate_named("experiment_result", result)

    report["scientific_outcome"] = outcome
    if git_manager is not None:
        pointers = git_manager.record_experiment(run_id=str(contract.get("run_id")))
        report["git"] = pointers.to_dict()
        result = dict(result)
        result["experiment_sha"] = pointers.experiment_sha
        validate_named("experiment_result", result)
    report["result"] = result
    _emit(
        events,
        protocol=protocol,
        plan=plan,
        event_type="evidence_check",
        actor_role="system",
        phase="experiment",
        payload={
            **evidence.to_dict(),
            "scientific_outcome": outcome,
            "rubric": rubric_payload,
        },
        run_id=contract.get("run_id"),
        reconstructed=recovered,
    )

    # Reviewer only after VALID evidence. Injected `review` cannot mint KEEP/DISCARD.
    memory_review = None
    if evidence.evidence_status == "VALID" and evidence.review_allowed and rubric is not None:
        try:
            packet = Reviewer().review(
                result=result,
                evidence=evidence,
                rubric=rubric,
                contract=contract,
                protocol=protocol,
                plan=plan,
                memory={
                    "lessons": writer.load_lessons(),
                    "strategies": writer.load_strategies(),
                },
            )
            report["review_decision"] = packet.review_decision
            report["review"] = packet.document
            memory_review = packet.to_memory_review()
            _emit(
                events,
                protocol=protocol,
                plan=plan,
                event_type="review_decision",
                actor_role="reviewer",
                phase="review",
                payload={
                    "review_decision": packet.review_decision,
                    "hypothesis_status": packet.document["hypothesis_status"],
                    "lesson_ids": [row["lesson_id"] for row in packet.research_lessons],
                    "strategy_ids": [row["strategy_id"] for row in packet.strategies],
                },
                run_id=contract.get("run_id"),
                decision_summary=packet.decision_summary,
                evidence_refs=[str(contract.get("run_id"))],
                reconstructed=recovered,
            )
            if packet.strategies:
                _emit(
                    events,
                    protocol=protocol,
                    plan=plan,
                    event_type="strategy_revision",
                    actor_role="reviewer",
                    phase="review",
                    payload={
                        "lesson_id": packet.research_lessons[0]["lesson_id"],
                        "strategy_id": packet.strategies[0]["strategy_id"],
                        "action": packet.strategies[0]["action"],
                    },
                    run_id=contract.get("run_id"),
                    evidence_refs=[str(contract.get("run_id"))],
                    reconstructed=recovered,
                )
            if git_manager is not None:
                pointers = git_manager.apply_review(packet.review_decision)
                report["git"] = pointers.to_dict()
        except ReviewRefused:
            report["review_decision"] = ReviewDecisionValue.PENDING.value
            memory_review = None

    memory_report = writer.consume(events, plan=plan, review=memory_review)
    report["memory"] = {
        "lessons_written": memory_report["lessons_written"],
        "strategies_written": memory_report["strategies_written"],
        "refused": memory_report["refused"],
        "trace_edges": memory_report["trace_edges"],
        "invented_from_metrics": False,
    }
    _emit(
        events,
        protocol=protocol,
        plan=plan,
        event_type="memory_write",
        actor_role="system",
        phase="review",
        payload={
            "lesson_ids": memory_report["lessons_written"],
            "strategy_ids": memory_report["strategies_written"],
            "trace_edges": memory_report["trace_edges"],
        },
        run_id=contract.get("run_id"),
        reconstructed=recovered,
    )
    return report


def run_gated_dfine_from_files(
    protocol_path: Path | str,
    plan_path: Path | str,
    *,
    output_dir: Path | str,
    execute: bool = False,
    require_live_ready: bool = False,
) -> dict[str, Any]:
    return run_gated_dfine(
        protocol=load_json(protocol_path),
        plan=load_json(plan_path),
        output_dir=output_dir,
        execute=execute,
        require_live_ready=require_live_ready,
    )
