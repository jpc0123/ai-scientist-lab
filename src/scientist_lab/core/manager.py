"""Thin Freeze Manager: state-machine dispatch. Not an LLM loop.

Reads run_state / evidence_status / review_decision and
next_orchestration_action. Does not invent those facts via prompt.
Does not KEEP/DISCARD. Does not own the state machine.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from scientist_lab.core.claim_gate import (
    ClaimGate,
    candidate_claim_from_plan,
    evaluate_claim,
)
from scientist_lab.core.decision_rubric import evaluate_rubric
from scientist_lab.core.evidence_validator import EvidenceValidator
from scientist_lab.core.exception_handler import next_exception_action
from scientist_lab.core.gate_engine import GateEngine, GateStatus
from scientist_lab.core.git_manager import GitManager, GitManagerError, is_git_repo
from scientist_lab.release.git_adapter import GitAdapterError
from scientist_lab.core.invariants import evaluate_stop_rules
from scientist_lab.core.memory_writer import MemoryWriter
from scientist_lab.core.planner import PlanRefused, Planner
from scientist_lab.core.result_parser import ResultParser
from scientist_lab.core.reviewer import ReviewRefused, Reviewer
from scientist_lab.core.schema_registry import load_json, validate_named
from scientist_lab.core.scientific_outcome import project_scientific_outcome
from scientist_lab.core.state_machine import (
    EvidenceStatus,
    ExperimentRunState,
    InvalidTransition,
    OrchestrationAction,
    ReviewDecisionValue,
    RunState,
    RUN_TRANSITIONS,
    next_orchestration_action,
    transition,
)
from scientist_lab.instrumentation.appender import EventAppender

# Sentinel: require_live_ready blocked REAL ignition (not a callable runner).
_LIVE_READY_BLOCKED = object()

_TERMINAL_ACTIONS = {
    OrchestrationAction.STOP.value,
    OrchestrationAction.NEED_HUMAN.value,
    OrchestrationAction.NOVELTY_EXHAUSTED.value,
    OrchestrationAction.IDLE.value,
    OrchestrationAction.PROTOCOL_AMENDMENT_REQUIRED.value,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_optional(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _state_from_run(doc: Mapping[str, Any]) -> ExperimentRunState:
    return ExperimentRunState(
        run_state=RunState(str(doc["run_state"])),
        evidence_status=EvidenceStatus(str(doc["evidence_status"])),
        review_decision=ReviewDecisionValue(str(doc["review_decision"])),
    )


def _apply_state(doc: dict[str, Any], state: ExperimentRunState) -> dict[str, Any]:
    doc["run_state"] = state.run_state.value
    doc["evidence_status"] = state.evidence_status.value
    doc["review_decision"] = state.review_decision.value
    doc["updated_at"] = _now()
    return doc


@dataclass
class ManagerStep:
    action: str
    state: ExperimentRunState
    reasons: tuple[str, ...] = ()
    idle: bool = False
    report: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "run_state": self.state.run_state.value,
            "evidence_status": self.state.evidence_status.value,
            "review_decision": self.state.review_decision.value,
            "reasons": list(self.reasons),
            "idle": self.idle,
            "report": dict(self.report),
        }


class Manager:
    """Dispatcher only. Persist facts on disk; do not prompt-judge the next role.

    Default ``execute=False``: Adapter dry-run / REPLAY recovery only (no GPU).
    ``execute=True`` uses ``live_runner`` if provided, otherwise
    ``make_cuda_live_runner`` (lazy ExperimentService). ``require_live_ready``
    refuses ignition when CUDA doctor reports ``live_ready=false`` — no fake
    metrics.json. Gate not APPROVED never reaches this path.
    """

    def __init__(
        self,
        project_dir: Path | str,
        *,
        protocol: Mapping[str, Any] | None = None,
        execute: bool = False,
        require_live_ready: bool = False,
        live_runner: Any | None = None,
        doctor_fn: Any | None = None,
        doctor_root: Path | None = None,
        experiments: Any | None = None,
        baseline_metrics: Mapping[str, Any] | None = None,
        max_extra_rounds: int = 1,
        adapter: Any | None = None,
        git_manager: GitManager | None = None,
        git_root: Path | str | None = None,
        git_record_paths: Sequence[str] | None = None,
    ) -> None:
        self.root = Path(project_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.execute = bool(execute)
        self.require_live_ready = bool(require_live_ready)
        self.live_runner = live_runner
        self.doctor_fn = doctor_fn
        self.doctor_root = Path(doctor_root) if doctor_root is not None else None
        self.experiments = experiments
        self._injected_git = git_manager
        self.git_root = Path(git_root) if git_root is not None else None
        self.git_record_paths = tuple(git_record_paths or ())
        self._git: GitManager | None = None
        self._git_resolved = False
        self._git_skip_reason: str | None = None
        self._git_recorded_run: str | None = None
        self._doctor_report: dict[str, Any] | None = None
        self.max_extra_rounds = max(0, int(max_extra_rounds))
        self.protocol_path = self.root / "protocol.json"
        self.run_path = self.root / "experiment_run.json"
        self.plan_path = self.root / "plan.json"
        self.contract_path = self.root / "contract.json"
        self.result_path = self.root / "result.json"
        self.handle_path = self.root / "handle.json"
        self.review_path = self.root / "review.json"
        self.claim_gate_path = self.root / "claim_gate.json"
        self.status_path = self.root / "manager_status.json"
        self.events = EventAppender(self.root / "research_events.jsonl")
        self.memory = MemoryWriter(self.root / "memory")
        if protocol is not None:
            _dump(self.protocol_path, protocol)
        proto = protocol or load_json(self.protocol_path)
        self.protocol = dict(proto)
        self.baseline_metrics = dict(baseline_metrics or {})
        baseline_file = _load_optional(self.root / "baseline_metrics.json")
        if baseline_file:
            self.baseline_metrics.update(baseline_file)
        self.adapter = adapter
        self.planner = Planner()
        self.reviewer = Reviewer()
        self.parser = ResultParser()
        self.evidence = EvidenceValidator()
        self.gate = GateEngine()

    def _adapter(self) -> Any:
        if self.adapter is None:
            from scientist_lab.adapters.dfine.adapter import DFINEAdapter

            self.adapter = DFINEAdapter(events=self.events)
        return self.adapter

    def probe_doctor(self, *, force: bool = False) -> dict[str, Any]:
        """Run CUDA doctor and append a ``live_ready`` fact. Never writes metrics."""
        if self._doctor_report is not None and not force:
            return dict(self._doctor_report)
        if self.doctor_fn is not None:
            report = dict(self.doctor_fn())
        else:
            from scientist_lab.tasks.rgbt_detection.cuda_doctor import build_dfine_cuda_doctor

            root = self.doctor_root
            if root is None:
                root = Path(__file__).resolve().parents[3]
            report = build_dfine_cuda_doctor(root, probe_runtime=True)
        self._doctor_report = report
        self._emit(
            event_type="tool_call",
            phase="experiment",
            payload={
                "tool": "cuda_doctor",
                "live_ready": bool(report.get("live_ready")),
                "overall": report.get("overall"),
                "execute": self.execute,
                "require_live_ready": self.require_live_ready,
            },
        )
        return dict(report)

    def _resolve_live_runner(self) -> Any | None:
        """Return the runner for REAL execute, or None to stay dry-run.

        ``execute=False`` never returns a runner. ``require_live_ready`` plus
        ``live_ready=false`` returns a sentinel that callers treat as blocked.
        """
        if not self.execute:
            return None
        if self.require_live_ready or self.live_runner is None:
            doctor = self.probe_doctor()
            if self.require_live_ready and not bool(doctor.get("live_ready")):
                return _LIVE_READY_BLOCKED
        if self.live_runner is not None:
            return self.live_runner
        from scientist_lab.adapters.dfine.cuda_runner import make_cuda_live_runner

        experiments = self.experiments
        if experiments is None:
            from scientist_lab.services.experiment_service import ExperimentService

            experiments = ExperimentService()
            self.experiments = experiments
        return make_cuda_live_runner(
            experiments,
            execute=True,
            require_live_ready=self.require_live_ready,
        )

    def _block_execution(
        self,
        doc: dict[str, Any],
        state: ExperimentRunState,
        reason: str,
        *,
        live_ready: bool | None = False,
    ) -> ManagerStep:
        doc["last_error"] = reason
        if state.run_state in {RunState.APPROVED, RunState.CONTRACTED}:
            state = transition(state, run_state=RunState.BLOCKED)
        self._save_run(doc, state)
        self._emit(
            event_type="execution",
            phase="experiment",
            payload={
                "status": "blocked",
                "reason": reason,
                "runner_called": False,
                "metrics_forged": False,
                "live_ready": live_ready,
            },
            run_id=str(doc.get("run_id")),
            plan_id=doc.get("plan_id"),
        )
        return ManagerStep(
            OrchestrationAction.NEED_EXECUTION.value,
            state,
            reasons=(reason,),
            idle=True,
            report={
                "blocked": True,
                "live_ready": live_ready,
                "runner_called": False,
                "metrics_forged": False,
            },
        )

    def _status(self) -> dict[str, Any]:
        return _load_optional(self.status_path) or {
            "consecutive_discards": 0,
            "execution_failures": 0,
            "rounds_without_improvement": 0,
            "duplicate_plan_rejects": 0,
            "extra_rounds": 0,
            "steps": 0,
        }

    def _save_status(self, status: Mapping[str, Any]) -> None:
        _dump(self.status_path, status)

    def _run_doc(self) -> dict[str, Any]:
        doc = _load_optional(self.run_path)
        if doc is None:
            raise FileNotFoundError(f"experiment_run.json missing in {self.root}")
        return doc

    def _save_run(self, doc: Mapping[str, Any], state: ExperimentRunState) -> dict[str, Any]:
        payload = _apply_state(dict(doc), state)
        validate_named("experiment_run", payload)
        _dump(self.run_path, payload)
        return payload

    def load_state(self) -> ExperimentRunState:
        return _state_from_run(self._run_doc())

    def peek_action(self) -> str:
        return next_orchestration_action(self.load_state()).value

    def _emit(
        self,
        *,
        event_type: str,
        phase: str,
        payload: Mapping[str, Any],
        run_id: str | None = None,
        plan_id: str | None = None,
        decision_summary: Mapping[str, Any] | None = None,
        actor_role: str = "manager",
        evidence_refs: list[str] | None = None,
    ) -> None:
        event: dict[str, Any] = {
            "project_id": self.protocol["project_id"],
            "run_id": run_id,
            "plan_id": plan_id,
            "protocol_version": self.protocol.get("protocol_version"),
            "fingerprint_id": self.protocol.get("fingerprint_id"),
            "event_type": event_type,
            "actor_role": actor_role,
            "phase": phase,
            "payload": dict(payload),
        }
        if evidence_refs:
            event["evidence_refs"] = list(evidence_refs)
        if decision_summary is not None:
            event["decision_summary"] = dict(decision_summary)
        self.events.append(event)

    def _git_repo_root(self) -> Path:
        return Path(self.git_root) if self.git_root is not None else self.root

    def _resolve_git_manager(self) -> GitManager | None:
        """Use injected GitManager, else ``git_root`` / project_dir if it is a repo.

        Missing ``.git`` is a skip, not a crash. Does not walk parent directories.
        """
        if self._git_resolved:
            return self._git
        self._git_resolved = True
        if self._injected_git is not None:
            self._git = self._injected_git
            return self._git
        root = self._git_repo_root()
        if not is_git_repo(root):
            self._git_skip_reason = "not a git repository"
            return None
        try:
            self._git = GitManager(root)
        except Exception as exc:  # noqa: BLE001 — skip, never crash Manager
            self._git_skip_reason = f"{type(exc).__name__}: {exc}"
            self._git = None
        return self._git

    def _dump_git_snapshot(self, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"skipped": self._git is None}
        if self._git_skip_reason:
            payload["reason"] = self._git_skip_reason
        if self._git is not None:
            try:
                payload.update(self._git.snapshot())
            except Exception as exc:  # noqa: BLE001
                payload["reason"] = f"{type(exc).__name__}: {exc}"
        if extra:
            payload.update(dict(extra))
        _dump(self.root / "git_pointers.json", payload)
        return payload

    def _emit_git(self, payload: Mapping[str, Any], *, run_id: str | None, plan_id: str | None) -> None:
        self._emit(
            event_type="tool_call",
            phase="review",
            payload={"tool": "git_manager", **dict(payload)},
            run_id=run_id,
            plan_id=plan_id,
        )

    def _record_experiment_git(self, doc: Mapping[str, Any]) -> dict[str, Any]:
        run_id = str(doc.get("run_id") or "")
        plan_id = doc.get("plan_id") if isinstance(doc.get("plan_id"), str) else None
        git = self._resolve_git_manager()
        if git is None:
            snapshot = self._dump_git_snapshot({"action": "record_experiment"})
            self._emit_git(
                {
                    "action": "record_experiment",
                    "skipped": True,
                    "reason": self._git_skip_reason,
                    "best_sha_updated": False,
                },
                run_id=run_id or None,
                plan_id=plan_id,
            )
            return snapshot
        try:
            root = git.repo_root
            paths = [p for p in self.git_record_paths if (root / p).is_file()]
            pointers = git.record_experiment(
                run_id=run_id or "unknown_run",
                paths=paths or None,
                allow_empty=not paths,
            )
            self._git_recorded_run = run_id or None
            snapshot = self._dump_git_snapshot(
                {"action": "record_experiment", "skipped": False, "best_sha_updated": False}
            )
            self._emit_git(
                {
                    "action": "record_experiment",
                    "skipped": False,
                    "experiment_sha": pointers.experiment_sha,
                    "best_sha": pointers.best_sha,
                    "best_sha_updated": False,
                    "paths": paths,
                },
                run_id=run_id or None,
                plan_id=plan_id,
            )
            return snapshot
        except (GitManagerError, GitAdapterError, OSError) as exc:
            self._git_skip_reason = f"{type(exc).__name__}: {exc}"
            snapshot = self._dump_git_snapshot({"action": "record_experiment", "error": True})
            self._emit_git(
                {
                    "action": "record_experiment",
                    "skipped": True,
                    "reason": self._git_skip_reason,
                    "best_sha_updated": False,
                },
                run_id=run_id or None,
                plan_id=plan_id,
            )
            return snapshot

    def _apply_review_git(self, doc: Mapping[str, Any], state: ExperimentRunState) -> dict[str, Any]:
        run_id = str(doc.get("run_id") or "")
        decision = state.review_decision.value
        plan_id = doc.get("plan_id") if isinstance(doc.get("plan_id"), str) else None
        if state.evidence_status != EvidenceStatus.VALID:
            snapshot = self._dump_git_snapshot(
                {
                    "action": "apply_review",
                    "skipped": True,
                    "reason": "evidence not VALID; refusing KEEP/DISCARD git apply",
                    "review_decision": decision,
                    "best_sha_updated": False,
                }
            )
            self._emit_git(
                {
                    "action": "apply_review",
                    "skipped": True,
                    "reason": "evidence not VALID",
                    "review_decision": decision,
                    "best_sha_updated": False,
                },
                run_id=run_id or None,
                plan_id=plan_id,
            )
            return snapshot
        git = self._resolve_git_manager()
        if git is None:
            snapshot = self._dump_git_snapshot(
                {"action": "apply_review", "review_decision": decision}
            )
            self._emit_git(
                {
                    "action": "apply_review",
                    "skipped": True,
                    "reason": self._git_skip_reason,
                    "review_decision": decision,
                    "best_sha_updated": False,
                },
                run_id=run_id or None,
                plan_id=plan_id,
            )
            return snapshot
        before = git.load()
        try:
            pointers = git.apply_review(decision)
            best_updated = pointers.best_sha != before.best_sha
            if decision not in {
                ReviewDecisionValue.KEEP.value,
                ReviewDecisionValue.DISCARD.value,
            } and pointers.best_sha != before.best_sha:
                raise GitManagerError(
                    f"{decision} moved best_sha from {before.best_sha} to {pointers.best_sha}"
                )
            if decision not in {
                ReviewDecisionValue.KEEP.value,
                ReviewDecisionValue.DISCARD.value,
            }:
                best_updated = False
            snapshot = self._dump_git_snapshot(
                {
                    "action": "apply_review",
                    "skipped": False,
                    "review_decision": decision,
                    "best_sha_updated": best_updated,
                }
            )
            self._emit_git(
                {
                    "action": "apply_review",
                    "skipped": False,
                    "review_decision": decision,
                    "experiment_sha": pointers.experiment_sha,
                    "best_sha": pointers.best_sha,
                    "best_sha_updated": best_updated,
                    "head": git.current_sha(),
                },
                run_id=run_id or None,
                plan_id=plan_id,
            )
            return snapshot
        except (GitManagerError, GitAdapterError, OSError) as exc:
            snapshot = self._dump_git_snapshot(
                {
                    "action": "apply_review",
                    "error": str(exc),
                    "review_decision": decision,
                    "best_sha_updated": False,
                }
            )
            self._emit_git(
                {
                    "action": "apply_review",
                    "skipped": True,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "review_decision": decision,
                    "best_sha_updated": False,
                },
                run_id=run_id or None,
                plan_id=plan_id,
            )
            return snapshot

    def _stop_action(self, doc: Mapping[str, Any], status: Mapping[str, Any]) -> str | None:
        return evaluate_stop_rules(
            (self.protocol.get("stop_rules") or {}),
            round_index=int(doc.get("round_index") or 0),
            consecutive_discards=int(status.get("consecutive_discards") or 0),
            execution_failures=int(status.get("execution_failures") or 0),
            rounds_without_improvement=int(status.get("rounds_without_improvement") or 0),
            duplicate_plan_rejects=int(status.get("duplicate_plan_rejects") or 0),
        )

    def initialize_run(
        self,
        *,
        round_index: int = 0,
        parent_run_id: str | None = None,
        plan_id: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Write CREATED ExperimentRun. Does not invent a Plan."""
        doc = {
            "schema_version": "1.0.0",
            "run_id": run_id or f"run_round{round_index}_pending",
            "project_id": self.protocol["project_id"],
            "protocol_id": self.protocol["protocol_id"],
            "protocol_version": self.protocol["protocol_version"],
            "round_index": round_index,
            "plan_id": plan_id,
            "parent_run_id": parent_run_id,
            "run_state": RunState.CREATED.value,
            "evidence_status": EvidenceStatus.PENDING.value,
            "review_decision": ReviewDecisionValue.PENDING.value,
            "fingerprint_id": self.protocol.get("fingerprint_id"),
            "updated_at": _now(),
        }
        validate_named("experiment_run", doc)
        _dump(self.run_path, doc)
        return doc

    def step(self) -> ManagerStep:
        """Perform exactly one orchestration action. No unbounded loop."""
        doc = self._run_doc()
        status = self._status()
        status["steps"] = int(status.get("steps") or 0) + 1
        state = _state_from_run(doc)
        halt = self._stop_action(doc, status)
        if halt:
            state = self._halt(doc, state, halt)
            self._save_status(status)
            step = ManagerStep(
                action=halt,
                state=state,
                reasons=(f"stop_rules:{halt}",),
                idle=True,
            )
            self._emit(
                event_type="human_governance",
                phase="governance",
                payload={"action": halt, "source": "stop_rules"},
                run_id=str(doc.get("run_id")),
                plan_id=doc.get("plan_id"),
            )
            return step

        action = next_orchestration_action(state)
        handler = {
            OrchestrationAction.NEED_PLAN: self._need_plan,
            OrchestrationAction.NEED_MATERIALIZE: self._need_materialize,
            OrchestrationAction.NEED_GATE: self._need_gate,
            OrchestrationAction.NEED_HUMAN: self._need_human,
            OrchestrationAction.NEED_EXECUTION: self._need_execution,
            OrchestrationAction.NEED_PARSE: self._need_parse,
            OrchestrationAction.NEED_EVIDENCE_CHECK: self._need_evidence,
            OrchestrationAction.NEED_REVIEW: self._need_review,
            OrchestrationAction.NEED_MEMORY: self._need_memory,
            OrchestrationAction.RETRY_EXECUTION: self._retry_execution,
            OrchestrationAction.NEXT_ROUND: self._next_round,
            OrchestrationAction.STOP: self._stop,
            OrchestrationAction.NOVELTY_EXHAUSTED: self._stop,
            OrchestrationAction.IDLE: self._idle,
        }.get(action, self._idle)
        step = handler(doc, state, status)
        self._save_status(status)
        return step

    def run_until(
        self,
        idle_or_stop: bool = True,
        *,
        max_steps: int = 32,
    ) -> list[ManagerStep]:
        """Bounded dispatch. Hard cap required. Never while True."""
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        steps: list[ManagerStep] = []
        for _ in range(max_steps):
            step = self.step()
            steps.append(step)
            if idle_or_stop and (step.idle or step.action in _TERMINAL_ACTIONS):
                return steps
        return steps

    def _halt(
        self, doc: dict[str, Any], state: ExperimentRunState, action: str
    ) -> ExperimentRunState:
        if state.run_state != RunState.STOPPED:
            allowed = RUN_TRANSITIONS.get(state.run_state, set())
            try:
                if RunState.STOPPED in allowed:
                    state = transition(state, run_state=RunState.STOPPED)
                elif RunState.BLOCKED in allowed:
                    state = transition(state, run_state=RunState.BLOCKED)
                    state = transition(state, run_state=RunState.STOPPED)
                else:
                    doc["last_error"] = f"stop_rules {action} while {state.run_state.value}"
            except InvalidTransition as exc:
                doc["last_error"] = str(exc)
        self._save_run(doc, state)
        return state

    def _need_plan(
        self, doc: dict[str, Any], state: ExperimentRunState, status: dict[str, Any]
    ) -> ManagerStep:
        existing = _load_optional(self.plan_path)
        lessons = self.memory.load_lessons()
        if existing is not None and not lessons:
            # Seed / human-provided first Plan. Do not call Planner to invent refs.
            validate_named("experiment_plan", existing)
            doc["plan_id"] = existing.get("plan_id")
            doc["round_index"] = int(existing.get("round_index") or doc.get("round_index") or 0)
            doc["parent_run_id"] = existing.get("parent_run_id")
            state = transition(state, run_state=RunState.PLANNED)
            self._save_run(doc, state)
            self._emit(
                event_type="plan_proposal",
                phase="planning",
                payload={"source": "seed_plan", "plan_id": existing.get("plan_id")},
                run_id=str(doc.get("run_id")),
                plan_id=existing.get("plan_id"),
            )
            return ManagerStep(OrchestrationAction.NEED_PLAN.value, state, report={"plan": existing})

        previous = existing or _load_optional(self.root / "previous_plan.json")
        if previous is None:
            doc["last_error"] = "NEED_PLAN missing seed/previous plan; Manager will not invent refs"
            self._save_run(doc, state)
            self._emit(
                event_type="human_governance",
                phase="governance",
                payload={"action": "NEED_HUMAN", "forged_refs": False, "reason": doc["last_error"]},
                run_id=str(doc.get("run_id")),
                plan_id=doc.get("plan_id"),
            )
            return ManagerStep(
                OrchestrationAction.NEED_HUMAN.value,
                state,
                reasons=(doc["last_error"],),
                idle=True,
                report={"forged_refs": False},
            )
        parent = str(doc.get("parent_run_id") or previous.get("parent_run_id") or "")
        try:
            packet = self.planner.next_plan(
                protocol=self.protocol,
                memory=self.memory,
                previous_plan=previous,
                parent_run_id=parent or str(doc.get("run_id")),
                last_review_decision=str(doc.get("review_decision") or "") or None,
                events=self.events,
            )
        except PlanRefused as exc:
            doc["last_error"] = str(exc)
            self._save_run(doc, state)
            self._emit(
                event_type="human_governance",
                phase="governance",
                payload={"action": "NEED_HUMAN", "reason": str(exc), "forged_refs": False},
                run_id=str(doc.get("run_id")),
                plan_id=doc.get("plan_id"),
            )
            return ManagerStep(
                OrchestrationAction.NEED_HUMAN.value,
                state,
                reasons=(str(exc),),
                idle=True,
                report={"forged_refs": False, "orchestration": exc.orchestration_action},
            )
        _dump(self.plan_path, packet.plan)
        doc["plan_id"] = packet.plan.get("plan_id")
        doc["round_index"] = int(packet.plan.get("round_index") or doc.get("round_index") or 0)
        state = transition(state, run_state=RunState.PLANNED)
        self._save_run(doc, state)
        return ManagerStep(
            OrchestrationAction.NEED_PLAN.value,
            state,
            report={"plan": packet.plan, "source": packet.source},
        )

    def _need_materialize(
        self, doc: dict[str, Any], state: ExperimentRunState, status: dict[str, Any]
    ) -> ManagerStep:
        plan = load_json(self.plan_path)
        from scientist_lab.adapters.base import MaterializeRejected

        try:
            contract = self._adapter().materialize_contract(plan, self.protocol)
        except MaterializeRejected as exc:
            status["duplicate_plan_rejects"] = int(status.get("duplicate_plan_rejects") or 0) + 1
            doc["last_error"] = str(exc)
            state = transition(state, run_state=RunState.BLOCKED)
            self._save_run(doc, state)
            self._emit(
                event_type="materialize",
                phase="planning",
                payload={"status": "rejected", "reason": str(exc)},
                run_id=str(doc.get("run_id")),
                plan_id=plan.get("plan_id"),
            )
            return ManagerStep(
                OrchestrationAction.NEED_MATERIALIZE.value,
                state,
                reasons=(str(exc),),
                idle=True,
            )
        _dump(self.contract_path, contract)
        doc["run_id"] = contract.get("run_id") or doc.get("run_id")
        doc["contract_ref"] = "contract.json"
        doc["plan_id"] = plan.get("plan_id")
        state = transition(state, run_state=RunState.MATERIALIZED)
        self._save_run(doc, state)
        self._emit(
            event_type="materialize",
            phase="planning",
            payload={"run_id": contract.get("run_id"), "how_only": True},
            run_id=str(contract.get("run_id")),
            plan_id=plan.get("plan_id"),
        )
        return ManagerStep(
            OrchestrationAction.NEED_MATERIALIZE.value,
            state,
            report={"contract": contract},
        )

    def _need_gate(
        self, doc: dict[str, Any], state: ExperimentRunState, status: dict[str, Any]
    ) -> ManagerStep:
        plan = load_json(self.plan_path)
        contract = load_json(self.contract_path)
        lessons = self.memory.load_lessons()
        strategies = self.memory.load_strategies()
        memory = {"lessons": lessons, "strategies": strategies} if (lessons or strategies) else None
        from scientist_lab.adapters.dfine.fingerprint import compute_fingerprint

        bound = compute_fingerprint(self.protocol, None)
        verdict = self.gate.evaluate(
            self.protocol, plan, contract, bound_fingerprint=bound, memory=memory
        )
        self._emit(
            event_type="gate_decision",
            phase="planning",
            payload=verdict.to_dict(),
            run_id=str(doc.get("run_id")),
            plan_id=plan.get("plan_id"),
        )
        if verdict.status == GateStatus.APPROVED:
            if memory is not None:
                self.memory.record_plan_citation(plan)
            state = transition(state, run_state=RunState.APPROVED)
            self._save_run(doc, state)
            return ManagerStep(OrchestrationAction.NEED_GATE.value, state, report=verdict.to_dict())
        if verdict.status == GateStatus.HUMAN_REQUIRED:
            state = transition(state, run_state=RunState.GATED)
            self._save_run(doc, state)
            return ManagerStep(
                OrchestrationAction.NEED_GATE.value,
                state,
                reasons=tuple(verdict.reasons),
                report=verdict.to_dict(),
            )
        state = transition(state, run_state=RunState.BLOCKED)
        doc["last_error"] = "; ".join(verdict.reasons)
        self._save_run(doc, state)
        return ManagerStep(
            OrchestrationAction.NEED_GATE.value,
            state,
            reasons=tuple(verdict.reasons),
            idle=True,
            report=verdict.to_dict(),
        )

    def _need_human(
        self, doc: dict[str, Any], state: ExperimentRunState, status: dict[str, Any]
    ) -> ManagerStep:
        self._emit(
            event_type="human_governance",
            phase="governance",
            payload={
                "action": OrchestrationAction.NEED_HUMAN.value,
                "frozen_bypass": False,
                "run_state": state.run_state.value,
            },
            run_id=str(doc.get("run_id")),
            plan_id=doc.get("plan_id"),
        )
        return ManagerStep(
            OrchestrationAction.NEED_HUMAN.value,
            state,
            reasons=("HUMAN_REQUIRED; Manager will not bypass frozen/forbidden",),
            idle=True,
        )

    def _need_execution(
        self, doc: dict[str, Any], state: ExperimentRunState, status: dict[str, Any]
    ) -> ManagerStep:
        contract = load_json(self.contract_path)
        from scientist_lab.adapters.dfine.fingerprint import compute_fingerprint

        bound = compute_fingerprint(self.protocol, None)
        execute_dir = self.root / "run"
        if not self.execute and (self.root / "metrics.json").is_file():
            # REPLAY recovery of artifacts placed at the project root.
            execute_dir = self.root
        try:
            runner = self._resolve_live_runner()
        except ImportError as exc:
            return self._block_execution(
                doc,
                state,
                f"cannot import live execute path ({exc}); refusing forged metrics.json",
                live_ready=False,
            )
        if runner is _LIVE_READY_BLOCKED:
            return self._block_execution(
                doc,
                state,
                "cuda doctor live_ready=false; refusing GPU and forged metrics.json",
                live_ready=False,
            )
        try:
            handle = self._adapter().execute(
                contract,
                self.protocol,
                output_dir=execute_dir,
                dry_run=not self.execute,
                live_runner=runner,
                expected_fingerprint=bound,
            )
        except Exception as exc:  # noqa: BLE001 — live path must not forge metrics
            status["execution_failures"] = int(status.get("execution_failures") or 0) + 1
            doc["last_error"] = str(exc)
            if state.run_state == RunState.APPROVED:
                state = transition(state, run_state=RunState.RUNNING)
            if state.run_state == RunState.RUNNING:
                state = transition(state, run_state=RunState.FAILED)
            self._save_run(doc, state)
            self._emit(
                event_type="execution",
                phase="experiment",
                payload={
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "reason": str(exc),
                    "runner_called": True,
                    "metrics_forged": False,
                },
                run_id=str(doc.get("run_id")),
                plan_id=doc.get("plan_id"),
            )
            return ManagerStep(
                OrchestrationAction.NEED_EXECUTION.value,
                state,
                reasons=(str(exc),),
                idle=True,
                report={
                    "failed": True,
                    "error_type": type(exc).__name__,
                    "runner_called": True,
                    "metrics_forged": False,
                    "dry_run": not self.execute,
                },
            )
        _dump(self.handle_path, handle)
        handle_status = str(handle.get("status") or "")
        failed = handle_status in {"failed", "timeout", "timed_out", "cancelled"}
        if failed:
            if state.run_state == RunState.APPROVED:
                state = transition(state, run_state=RunState.RUNNING)
            state = transition(state, run_state=RunState.FAILED)
            status["execution_failures"] = int(status.get("execution_failures") or 0) + 1
            nested_err = ((handle.get("run_view") or {}).get("run") or {}).get("error")
            if isinstance(nested_err, dict):
                doc["last_error"] = str(
                    nested_err.get("message") or nested_err.get("error_type") or handle_status
                )
            else:
                doc["last_error"] = str(nested_err or handle_status)
            self._save_run(doc, state)
            return ManagerStep(
                OrchestrationAction.NEED_EXECUTION.value,
                state,
                reasons=(f"execution {handle_status}",),
                idle=True,
                report={
                    "handle": handle,
                    "dry_run": not self.execute,
                    "runner_called": runner is not None,
                    "metrics_forged": False,
                    "orchestration_hint": OrchestrationAction.RETRY_EXECUTION.value,
                },
            )
        state = transition(state, run_state=RunState.RUNNING)
        self._save_run(doc, state)
        return ManagerStep(
            OrchestrationAction.NEED_EXECUTION.value,
            state,
            report={
                "handle": handle,
                "dry_run": not self.execute,
                "runner_called": runner is not None,
            },
        )

    def _need_parse(
        self, doc: dict[str, Any], state: ExperimentRunState, status: dict[str, Any]
    ) -> ManagerStep:
        contract = load_json(self.contract_path)
        handle = load_json(self.handle_path)
        result = self.parser.from_handle(contract, handle)
        _dump(self.result_path, result)
        doc["result_ref"] = "result.json"
        failed = str(result.get("execution", {}).get("status") or "") in {
            "failed",
            "timeout",
        }
        if failed:
            if state.run_state == RunState.RUNNING:
                state = transition(state, run_state=RunState.FAILED)
            status["execution_failures"] = int(status.get("execution_failures") or 0) + 1
        else:
            state = transition(state, run_state=RunState.COMPLETED)
        self._save_run(doc, state)
        git_report = self._record_experiment_git(doc)
        return ManagerStep(
            OrchestrationAction.NEED_PARSE.value,
            state,
            report={"result": result, "git": git_report},
        )

    def _need_evidence(
        self, doc: dict[str, Any], state: ExperimentRunState, status: dict[str, Any]
    ) -> ManagerStep:
        result = load_json(self.result_path)
        contract = load_json(self.contract_path)
        handle = _load_optional(self.handle_path) or {}
        verdict = self.evidence.validate(
            result,
            protocol=self.protocol,
            expected_artifacts=list(contract.get("expected_artifacts") or []),
            fingerprint_comparable=handle.get("fingerprint_comparable"),
            dry_run=bool(handle.get("dry_run")),
            handle_status=str(handle.get("status") or ""),
        )
        ev = EvidenceStatus(verdict.evidence_status)
        state = transition(state, evidence_status=ev)
        outcome = project_scientific_outcome(
            run_state=state.run_state,
            evidence_status=ev,
            primary_delta=None,
        )
        doc["scientific_outcome"] = outcome
        self._save_run(doc, state)
        self._emit(
            event_type="evidence_check",
            phase="experiment",
            payload=verdict.to_dict(),
            run_id=str(doc.get("run_id")),
            plan_id=doc.get("plan_id"),
        )
        return ManagerStep(
            OrchestrationAction.NEED_EVIDENCE_CHECK.value,
            state,
            report=verdict.to_dict(),
        )

    def _need_review(
        self, doc: dict[str, Any], state: ExperimentRunState, status: dict[str, Any]
    ) -> ManagerStep:
        if state.evidence_status != EvidenceStatus.VALID:
            raise ReviewRefused(
                f"Manager will not send non-VALID evidence to Reviewer "
                f"({state.evidence_status.value})"
            )
        result = load_json(self.result_path)
        contract = load_json(self.contract_path)
        plan = _load_optional(self.plan_path) or {}
        handle = _load_optional(self.handle_path) or {}
        evidence = self.evidence.validate(
            result,
            protocol=self.protocol,
            expected_artifacts=list(contract.get("expected_artifacts") or []),
            fingerprint_comparable=handle.get("fingerprint_comparable"),
            dry_run=bool(handle.get("dry_run")),
            handle_status=str(handle.get("status") or ""),
        )
        rubric = evaluate_rubric(
            self.protocol,
            current_metrics=result.get("metrics") or {},
            baseline_metrics=self.baseline_metrics or None,
        )
        packet = self.reviewer.review(
            result=result,
            evidence=evidence,
            rubric=rubric,
            contract=contract,
            protocol=self.protocol,
            plan=plan,
            memory={"lessons": self.memory.load_lessons(), "strategies": self.memory.load_strategies()},
        )
        rd = ReviewDecisionValue(packet.review_decision)
        state = transition(state, review_decision=rd)
        _dump(self.review_path, packet.document)
        doc["review_ref"] = "review.json"
        if rd == ReviewDecisionValue.DISCARD:
            status["consecutive_discards"] = int(status.get("consecutive_discards") or 0) + 1
        else:
            status["consecutive_discards"] = 0
        self._save_run(doc, state)
        self._emit(
            event_type="review_decision",
            phase="review",
            payload={"review_decision": packet.review_decision},
            run_id=str(doc.get("run_id")),
            plan_id=doc.get("plan_id"),
            decision_summary=packet.decision_summary,
        )
        claim_gate_doc = self._attach_claim_gate(
            doc,
            result=result,
            plan=plan,
            handle=handle,
            review_decision=packet.review_decision,
        )
        git_report = self._apply_review_git(doc, state)
        return ManagerStep(
            OrchestrationAction.NEED_REVIEW.value,
            state,
            report={
                "review": packet.document,
                "git": git_report,
                "claim_gate": claim_gate_doc,
            },
        )

    def _attach_claim_gate(
        self,
        doc: Mapping[str, Any],
        *,
        result: Mapping[str, Any],
        plan: Mapping[str, Any],
        handle: Mapping[str, Any],
        review_decision: str,
    ) -> dict[str, Any]:
        """Write ClaimGate after Reviewer. Never overwrites review_decision."""
        has_baseline = bool(self.baseline_metrics)
        claim = candidate_claim_from_plan(
            plan, self.protocol, has_baseline=has_baseline
        )
        evidence = {
            "result": result,
            "metrics": dict(result.get("metrics") or {}),
            "run_id": result.get("run_id") or doc.get("run_id"),
            "run_state": doc.get("run_state"),
            "evidence_status": doc.get("evidence_status"),
            "budget_class": plan.get("budget_class") or "probe",
            "run_level": plan.get("budget_class") or "probe",
            "review_decision": review_decision,
            "fingerprint_comparable": handle.get("fingerprint_comparable"),
            "handle_status": handle.get("status"),
            "baseline": {
                "present": has_baseline,
                "metrics": dict(self.baseline_metrics or {}),
                "matched_fingerprint": False,
                "budget_class": "probe",
            }
            if has_baseline
            else {"present": False},
            "result_ref": "result.json",
            "review_ref": "review.json",
            "scientific_outcome": result.get("scientific_outcome") or doc.get("scientific_outcome"),
        }
        verdict = evaluate_claim(
            claim,
            protocol=self.protocol,
            evidence=evidence,
            plan=plan,
            review_decision=review_decision,
            scientific_outcome=evidence.get("scientific_outcome"),
        )
        _dump(self.claim_gate_path, verdict)
        self._emit(
            event_type="claim_gate",
            phase="review",
            actor_role="gate",
            payload={
                "status": verdict.get("status"),
                "claim_strength": verdict.get("claim_strength"),
                "keep_is_not_claim": True,
                "review_decision_unchanged": review_decision,
            },
            run_id=str(doc.get("run_id")),
            plan_id=doc.get("plan_id"),
            evidence_refs=list(verdict.get("evidence_refs") or []),
        )
        return verdict

    def _need_memory(
        self, doc: dict[str, Any], state: ExperimentRunState, status: dict[str, Any]
    ) -> ManagerStep:
        plan = _load_optional(self.plan_path)
        review_doc = _load_optional(self.review_path)
        memory_review = None
        if (
            state.evidence_status == EvidenceStatus.VALID
            and state.review_decision != ReviewDecisionValue.PENDING
            and review_doc
        ):
            memory_review = {
                "research_lessons": list(review_doc.get("research_lessons") or []),
                "strategies": list(review_doc.get("strategies") or []),
            }
        report = self.memory.consume(self.events, plan=plan, review=memory_review)
        state = transition(state, run_state=RunState.MEMORY_WRITTEN)
        self._save_run(doc, state)
        self._emit(
            event_type="memory_write",
            phase="review",
            payload={
                "lessons_written": report.get("lessons_written"),
                "invented_from_metrics": False,
            },
            run_id=str(doc.get("run_id")),
            plan_id=doc.get("plan_id"),
        )
        return ManagerStep(OrchestrationAction.NEED_MEMORY.value, state, report=report)

    def _retry_execution(
        self, doc: dict[str, Any], state: ExperimentRunState, status: dict[str, Any]
    ) -> ManagerStep:
        result = _load_optional(self.result_path) or {}
        hint = next_exception_action(
            result, evidence_status=state.evidence_status.value
        ) or OrchestrationAction.RETRY_EXECUTION.value
        if hint != OrchestrationAction.RETRY_EXECUTION.value:
            self._emit(
                event_type="human_governance",
                phase="governance",
                payload={"action": hint, "retry_not_replicate": True},
                run_id=str(doc.get("run_id")),
                plan_id=doc.get("plan_id"),
            )
            return ManagerStep(
                hint,
                state,
                reasons=(hint,),
                idle=True,
                report={"retry_not_replicate": True},
            )
        # Same Contract: FAILED → APPROVED (legal), next step NEED_EXECUTION.
        state = transition(state, run_state=RunState.APPROVED)
        self._save_run(doc, state)
        return ManagerStep(
            OrchestrationAction.RETRY_EXECUTION.value,
            state,
            report={"same_contract": True, "retry_not_replicate": True},
        )

    def _next_round(
        self, doc: dict[str, Any], state: ExperimentRunState, status: dict[str, Any]
    ) -> ManagerStep:
        extra = int(status.get("extra_rounds") or 0)
        if extra >= self.max_extra_rounds:
            return ManagerStep(
                OrchestrationAction.NEXT_ROUND.value,
                state,
                reasons=("max_extra_rounds reached; not starting another round",),
                idle=True,
                report={"extra_rounds": extra, "candidate": True},
            )
        next_index = int(doc.get("round_index") or 0) + 1
        halt = evaluate_stop_rules(
            (self.protocol.get("stop_rules") or {}),
            round_index=next_index,
            consecutive_discards=int(status.get("consecutive_discards") or 0),
            execution_failures=int(status.get("execution_failures") or 0),
            rounds_without_improvement=int(status.get("rounds_without_improvement") or 0),
            duplicate_plan_rejects=int(status.get("duplicate_plan_rejects") or 0),
        )
        if halt:
            state = self._halt(doc, state, halt)
            return ManagerStep(halt, state, reasons=(f"stop_rules:{halt}",), idle=True)

        previous = load_json(self.plan_path)
        parent = str(doc.get("run_id"))
        _dump(self.root / "previous_plan.json", previous)
        archive = self.root / "runs" / f"{parent}.json"
        _dump(archive, doc)
        self._archive_round_artifacts(parent)
        try:
            packet = self.planner.next_plan(
                protocol=self.protocol,
                memory=self.memory,
                previous_plan=previous,
                parent_run_id=parent,
                last_review_decision=state.review_decision.value,
                events=self.events,
            )
        except PlanRefused as exc:
            return ManagerStep(
                OrchestrationAction.NEED_HUMAN.value,
                state,
                reasons=(str(exc),),
                idle=True,
                report={"forged_refs": False},
            )
        _dump(self.plan_path, packet.plan)
        status["extra_rounds"] = extra + 1
        new_doc = self.initialize_run(
            round_index=int(packet.plan.get("round_index") or next_index),
            parent_run_id=parent,
            plan_id=packet.plan.get("plan_id"),
            run_id=f"run_{packet.plan.get('plan_id')}",
        )
        new_state = _state_from_run(new_doc)
        new_state = transition(new_state, run_state=RunState.PLANNED)
        self._save_run(new_doc, new_state)
        return ManagerStep(
            OrchestrationAction.NEXT_ROUND.value,
            new_state,
            report={"plan": packet.plan, "parent_run_id": parent},
        )

    def _archive_round_artifacts(self, parent_run_id: str) -> None:
        """Keep N-round GPU artifacts so N+1 does not overwrite Trace evidence."""
        dest = self.root / "runs" / parent_run_id
        dest.mkdir(parents=True, exist_ok=True)
        for name in (
            "result.json",
            "handle.json",
            "review.json",
            "contract.json",
            "git_pointers.json",
        ):
            src = self.root / name
            if src.is_file():
                shutil.copy2(src, dest / name)
        run_dir = self.root / "run"
        if run_dir.is_dir():
            target = dest / "run"
            if target.exists():
                shutil.rmtree(target)
            shutil.move(str(run_dir), str(target))

    def _stop(
        self, doc: dict[str, Any], state: ExperimentRunState, status: dict[str, Any]
    ) -> ManagerStep:
        action = next_orchestration_action(state).value
        state = self._halt(doc, state, action)
        return ManagerStep(action, state, idle=True)

    def _idle(
        self, doc: dict[str, Any], state: ExperimentRunState, status: dict[str, Any]
    ) -> ManagerStep:
        return ManagerStep(OrchestrationAction.IDLE.value, state, idle=True)
