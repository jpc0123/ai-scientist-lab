"""v2.5-D Human-gated probe REAL loop.

Historical VALID Evidence → LLM Reviewer proposal → MemoryWriter →
LLM Planner → existing Gate → optional --execute GPU → new Evidence →
LLM Reviewer again.

Does not bypass Gate. Does not auto-promote formal. Does not forge metrics.
Gateway is not a fifth Agent. probe 16/8 is engineering-loop evidence, not C1.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping

from scientist_lab.core.decision_rubric import evaluate_rubric
from scientist_lab.core.evidence_validator import EvidenceValidator
from scientist_lab.core.invariants import InvariantError
from scientist_lab.core.manager import Manager
from scientist_lab.core.memory_writer import MemoryWriter
from scientist_lab.core.planner import PlanRefused, Planner
from scientist_lab.core.reviewer import ReviewRefused, Reviewer
from scientist_lab.core.schema_registry import load_json
from scientist_lab.instrumentation.appender import EventAppender
from scientist_lab.llm.errors import MissingAPIKeyError, RealProviderNotEnabledError
from scientist_lab.llm.gateway import resolve_gateway_provider
from scientist_lab.llm.plan_replay import load_replay_bundle

_PKG_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FIXTURE = (
    _PKG_ROOT / "tests" / "fixtures" / "llm_plan_replay" / "m4_rounds3_discard"
)
DEFAULT_OUTPUT = Path(".run") / "v25d_llm_real_loop"
PROBE_TIMEOUT_SECONDS = 1200


def _numeric_metrics(payload: Mapping[str, Any] | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in dict(payload or {}).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[str(key)] = float(value)
    return out


def _copy_memory(src: Path, dest: Path) -> MemoryWriter:
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("research_memory.json", "strategy_memory.json", "research_trace.json"):
        path = src / name
        if path.is_file():
            shutil.copyfile(path, dest / name)
    return MemoryWriter(dest)


def _how_from_contract(contract: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not contract:
        return None
    mat = dict(contract.get("materialization") or {})
    how = dict(mat.get("how") or {})
    if not how:
        return {
            "allowed_changes": list(contract.get("allowed_changes") or []),
            "legacy_parameters": dict(mat.get("legacy_parameters") or {}),
            "execution_mode": mat.get("execution_mode"),
        }
    return how


def run_v25d_real_loop(
    *,
    source_run_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    execute: bool = False,
    require_live_ready: bool = False,
    live_runner: Any | None = None,
    doctor_fn: Any | None = None,
    planner_backend: str = "llm",
    reviewer_backend: str = "llm",
    llm_live: bool = False,
    provider: str = "mock",
    fallback_to_rules: bool = False,
    max_steps: int = 32,
    human_probe_permission: bool = False,
    llm_provider: Any | None = None,
) -> dict[str, Any]:
    """One probe-class closed loop. Never 4h formal. Never forges metrics.json."""
    source = Path(source_run_dir) if source_run_dir is not None else DEFAULT_FIXTURE
    output = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    history = output / "history"
    round_dir = output / "round"
    if history.exists():
        shutil.rmtree(history)
    if round_dir.exists():
        shutil.rmtree(round_dir)
    shutil.copytree(source, history)
    round_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_replay_bundle(history)
    protocol = dict(bundle["protocol"])
    previous = dict(bundle["previous_plan"])
    result = dict((bundle.get("evidence") or {}).get("result") or {})
    if not result and (history / "result.json").is_file():
        result = load_json(history / "result.json")
    baseline = _numeric_metrics(
        load_json(history / "baseline_metrics.json")
        if (history / "baseline_metrics.json").is_file()
        else {}
    )
    locked_review = str(bundle.get("last_review_decision") or "")
    parent_run_id = str(result.get("run_id") or bundle.get("parent_run_id") or "")
    events = EventAppender(output / "loop_events.jsonl")
    report: dict[str, Any] = {
        "ok": False,
        "stage": "v2.5-D",
        "exam": "human_gated_probe_real_loop",
        "source_run_dir": str(source),
        "output_dir": str(output),
        "execute": bool(execute),
        "require_live_ready": bool(require_live_ready),
        "gpu": False,
        "ignited": False,
        "llm_live": bool(llm_live),
        "planner_backend": planner_backend,
        "reviewer_backend": reviewer_backend,
        "human_probe_permission": bool(human_probe_permission),
        "formal_auto_promote": False,
        "probe_timeout_seconds": PROBE_TIMEOUT_SECONDS,
        "evidence_class": "engineering_probe_not_c1",
        "metrics_forged": False,
        "gate": {"status": "REJECTED", "reasons": []},
        "historical_review_decision": locked_review,
        "claim_gate": None,
        "aps": None,
        "how": None,
        "memory_refs": None,
        "fail_closed": False,
    }

    if execute and not human_probe_permission:
        report["fail_closed"] = True
        report["error"] = (
            "probe REAL requires Human Gate permission; formal still cannot auto-promote"
        )
        _write_report(output, report)
        return report

    try:
        resolved = llm_provider or resolve_gateway_provider(
            live=llm_live,
            provider=None if llm_live else provider,
        )
    except (MissingAPIKeyError, RealProviderNotEnabledError) as exc:
        report["fail_closed"] = True
        report["error"] = str(exc)
        _write_report(output, report)
        return report

    memory = _copy_memory(history / "memory", round_dir / "memory")
    (round_dir / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if baseline:
        (round_dir / "baseline_metrics.json").write_text(
            json.dumps(baseline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    try:
        historical = _historical_review(
            protocol=protocol,
            plan=previous,
            result=result,
            baseline=baseline,
            reviewer_backend=reviewer_backend,
            provider=resolved,
            llm_live=llm_live,
            fallback_to_rules=fallback_to_rules,
            events=events,
        )
    except (ReviewRefused, FileNotFoundError, ValueError) as exc:
        report["fail_closed"] = True
        report["error"] = f"historical review refused: {exc}"
        _write_report(output, report)
        return report

    report["historical"] = {
        "review_decision": historical["packet"].review_decision,
        "document_hypothesis_status": historical["packet"].document.get("hypothesis_status"),
        "semantic_hypothesis_status": (historical["packet"].semantic_proposal or {}).get(
            "hypothesis_status"
        ),
        "rubric_locked": historical["packet"].review_decision == locked_review
        or not locked_review,
        "source": historical["packet"].source,
    }
    if historical["packet"].review_decision != locked_review and locked_review:
        report["fail_closed"] = True
        report["error"] = "LLM must not override historical Rubric review_decision"
        _write_report(output, report)
        return report

    semantic_write: dict[str, Any] = {"attempted": False, "accepted": False}
    proposal = historical["packet"].semantic_proposal
    if proposal:
        lesson = historical["packet"].research_lessons[0]
        try:
            written = memory.persist_semantic_proposal(
                proposal,
                run_id=parent_run_id,
                review_decision=historical["packet"].review_decision,
                lesson_type=str(lesson.get("type") or "negative_evidence"),
                module=str((lesson.get("scope") or {}).get("module") or "neck"),
                task=str((lesson.get("scope") or {}).get("task") or "rgbt_detection"),
                metric=str(
                    ((protocol.get("objective") or {}).get("primary") or {}).get("metric")
                    or "APS"
                ),
                delta=historical["rubric"].primary_delta,
            )
            semantic_write = {
                "attempted": True,
                "accepted": True,
                "lesson_id": written.get("lesson_id"),
            }
        except InvariantError as exc:
            semantic_write = {"attempted": True, "accepted": False, "refused": str(exc)}
    report["semantic_write"] = semantic_write

    try:
        planner = Planner(
            backend=planner_backend,
            provider=resolved,
            live=llm_live,
            fallback_to_rules=fallback_to_rules,
        )
        plan_packet = planner.next_plan(
            protocol=protocol,
            memory=memory,
            previous_plan=previous,
            parent_run_id=parent_run_id,
            last_review_decision=historical["packet"].review_decision,
            events=events,
        )
    except PlanRefused as exc:
        report["fail_closed"] = True
        report["error"] = str(exc)
        _write_report(output, report)
        return report

    next_plan = dict(plan_packet.plan)
    if str(next_plan.get("budget_class") or "probe").lower() == "formal":
        report["fail_closed"] = True
        report["error"] = "LLM/Planner must not auto-promote budget_class to formal"
        report["plan"] = next_plan
        _write_report(output, report)
        return report

    (round_dir / "previous_plan.json").write_text(
        json.dumps(previous, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (round_dir / "plan.json").write_text(
        json.dumps(next_plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report["plan"] = {
        "plan_id": next_plan.get("plan_id"),
        "modification_scope": list(next_plan.get("modification_scope") or []),
        "budget_class": next_plan.get("budget_class"),
        "memory_refs": next_plan.get("memory_refs"),
        "source": plan_packet.source,
    }
    report["memory_refs"] = next_plan.get("memory_refs")

    events.append(
        {
            "project_id": protocol["project_id"],
            "event_type": "human_governance",
            "actor_role": "human",
            "phase": "governance",
            "plan_id": next_plan.get("plan_id"),
            "run_id": parent_run_id,
            "payload": {
                "permission": "probe_real" if execute else "dry_run_only",
                "source": "user_next_step",
                "human_probe_permission": bool(human_probe_permission),
                "formal_auto_promote": False,
                "execute": bool(execute),
            },
        }
    )

    mgr = Manager(
        round_dir,
        protocol=protocol,
        execute=bool(execute),
        require_live_ready=bool(require_live_ready),
        live_runner=live_runner,
        doctor_fn=doctor_fn,
        baseline_metrics=baseline or None,
        max_extra_rounds=0,
        planner_backend=planner_backend,
        reviewer_backend=reviewer_backend,
        llm_provider=resolved,
        llm_live=bool(llm_live),
        fallback_to_rules=bool(fallback_to_rules),
    )
    mgr.initialize_run(
        round_index=int(next_plan.get("round_index") or 2),
        plan_id=str(next_plan.get("plan_id") or ""),
        parent_run_id=parent_run_id,
    )
    steps = mgr.run_until(max_steps=max_steps)
    last = steps[-1] if steps else None
    gate_steps = [s for s in steps if s.action == "NEED_GATE"]
    exec_steps = [s for s in steps if s.action == "NEED_EXECUTION"]
    review_steps = [s for s in steps if s.action == "NEED_REVIEW"]
    gate_report = gate_steps[-1].report if gate_steps else {}
    exec_report = exec_steps[-1].report if exec_steps else {}
    review_report = review_steps[-1].report if review_steps else {}
    report["actions"] = [s.action for s in steps]
    report["last_state"] = last.to_dict() if last is not None else None
    report["gate"] = {
        "status": gate_report.get("status") or last.state.run_state.value if last else "REJECTED",
        "reasons": list(gate_report.get("reasons") or (last.reasons if last else [])),
    }
    if gate_report.get("status"):
        report["gate"]["status"] = gate_report.get("status")
        report["gate"]["reasons"] = list(gate_report.get("reasons") or [])
    report["runner_called"] = bool(exec_report.get("runner_called"))
    report["dry_run"] = bool(exec_report.get("dry_run")) if exec_steps else not execute
    report["ignited"] = bool(execute) and bool(exec_report.get("runner_called"))
    report["gpu"] = bool(report["ignited"]) and live_runner is None
    report["metrics_forged"] = bool(exec_report.get("metrics_forged"))
    contract = load_json(round_dir / "contract.json") if (round_dir / "contract.json").is_file() else {}
    report["how"] = _how_from_contract(contract)
    new_result = load_json(round_dir / "result.json") if (round_dir / "result.json").is_file() else {}
    metrics = dict(new_result.get("metrics") or {})
    report["aps"] = metrics.get("APS")
    report["new_result"] = {
        "run_id": new_result.get("run_id"),
        "execution_status": (new_result.get("execution") or {}).get("status"),
        "metrics": {k: metrics.get(k) for k in ("APS", "mAP50_95", "mAP50") if k in metrics},
    }
    report["rubric_review_decision"] = (review_report.get("review") or {}).get("review_decision")
    report["new_review_source"] = review_report.get("review_source")
    report["new_semantic"] = review_report.get("semantic_proposal")
    claim = review_report.get("claim_gate")
    if claim is None and (round_dir / "claim_gate.json").is_file():
        claim = load_json(round_dir / "claim_gate.json")
    report["claim_gate"] = claim
    if isinstance(claim, Mapping):
        report["claim_gate_not_c1_supported"] = not (
            str(claim.get("status") or "") == "SUPPORTED"
            and str(claim.get("claim_strength") or "") == "C1"
        )
    gate_approved = str(report["gate"].get("status") or "") == "APPROVED"
    blocked_live = bool(exec_report.get("blocked"))
    if report["metrics_forged"]:
        report["ok"] = False
        report["error"] = "metrics must not be forged"
    elif report["fail_closed"]:
        report["ok"] = False
    elif execute and require_live_ready and blocked_live:
        report["ok"] = False
        report["error"] = "; ".join(exec_steps[-1].reasons) if exec_steps else "live_ready=false"
    elif not gate_approved and execute:
        report["ok"] = False
        report["error"] = "Gate not APPROVED; refusing ignition"
    elif gate_approved:
        report["ok"] = True
    else:
        report["ok"] = False
        report["error"] = "Gate did not APPROVE"

    _write_report(output, report)
    return report


def _historical_review(
    *,
    protocol: Mapping[str, Any],
    plan: Mapping[str, Any],
    result: Mapping[str, Any],
    baseline: Mapping[str, Any],
    reviewer_backend: str,
    provider: Any,
    llm_live: bool,
    fallback_to_rules: bool,
    events: EventAppender,
) -> dict[str, Any]:
    from scientist_lab.adapters.dfine.adapter import DFINEAdapter

    if not result:
        raise ReviewRefused("historical loop requires result.json")
    contract = DFINEAdapter().materialize_contract(plan, protocol)
    evidence = EvidenceValidator().validate(
        result,
        protocol=protocol,
        expected_artifacts=list(contract.get("expected_artifacts") or []),
    )
    rubric = evaluate_rubric(
        protocol,
        current_metrics=result.get("metrics") or {},
        baseline_metrics=baseline or None,
    )
    packet = Reviewer(
        backend=reviewer_backend,
        provider=provider,
        live=llm_live,
        fallback_to_rules=fallback_to_rules,
    ).review(
        result=result,
        evidence=evidence,
        rubric=rubric,
        contract=contract,
        protocol=protocol,
        plan=plan,
        events=events,
    )
    return {"packet": packet, "rubric": rubric, "contract": contract, "evidence": evidence}


def _write_report(output: Path, report: dict[str, Any]) -> None:
    path = output / "loop_report.json"
    report["report_path"] = str(path)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
