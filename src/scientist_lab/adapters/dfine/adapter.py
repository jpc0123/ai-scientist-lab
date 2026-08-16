"""DFINEAdapter: model-specific HOW translation. No new scientific hypotheses."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from scientist_lab.adapters.base import MaterializeRejected
from scientist_lab.adapters.dfine.artifact_collector import collect_artifacts
from scientist_lab.adapters.dfine.fingerprint import compute_fingerprint, fingerprints_equivalent
from scientist_lab.adapters.dfine.how import (
    FORMAL_TIMEOUT_SECONDS,
    how_identity,
    parse_budget_seconds,
    resolve_adapter_how,
)
from scientist_lab.adapters.dfine.metrics_parser import parse_metrics
from scientist_lab.core.result_parser import build_experiment_result
from scientist_lab.core.schema_registry import validate_named
from scientist_lab.instrumentation.appender import EventAppender

LiveRunner = Callable[[Mapping[str, Any], Path], Mapping[str, Any]]


class DFINEAdapter:
    adapter_key = "dfine"

    def __init__(self, events: EventAppender | None = None) -> None:
        self.events = events

    def materialize_contract(
        self, plan: Mapping[str, Any], protocol: Mapping[str, Any]
    ) -> dict[str, Any]:
        if protocol.get("baseline", {}).get("adapter", "dfine") not in {None, "dfine"}:
            raise MaterializeRejected("protocol baseline.adapter is not dfine")
        changes = list(plan.get("proposed_changes") or [])
        scope = list(plan.get("modification_scope") or [])
        if not scope:
            raise MaterializeRejected("Plan missing modification_scope")
        if not changes or not all(c.get("target") and c.get("summary") for c in changes):
            raise MaterializeRejected(
                "Plan semantics insufficient: need proposed_changes with target+summary; "
                "Adapter will not invent a module (e.g. FDPN) from a vague goal"
            )
        editable = set(protocol.get("editable_scope") or [])
        if any(token not in editable for token in scope):
            raise MaterializeRejected("modification_scope not subset of editable_scope")

        how = resolve_adapter_how(plan)
        fingerprint_id = protocol.get("fingerprint_id")
        run_id = f"run_{plan.get('plan_id', 'unknown')}"
        budget_class = str(plan.get("budget_class") or "probe")
        formal = budget_class == "formal"
        if formal:
            timeout = parse_budget_seconds(
                (protocol.get("experiment_budget") or {}).get("formal"),
                default=FORMAL_TIMEOUT_SECONDS,
            )
            max_runtime = str(
                (protocol.get("experiment_budget") or {}).get("formal") or "4h"
            )
        else:
            timeout = 1200
            max_runtime = "20min"
        contract = {
            "schema_version": "1.0.0",
            "run_id": run_id,
            "parent_run": plan.get("parent_run_id"),
            "project_id": plan["project_id"],
            "protocol_id": plan["protocol_id"],
            "protocol_version": plan["protocol_version"],
            "plan_id": plan["plan_id"],
            "hypothesis": plan["hypothesis"],
            "allowed_changes": scope,
            "frozen_variables": list(protocol.get("frozen_scope") or []),
            "dataset": {
                "reference": protocol.get("baseline", {}).get("dataset", "dataset:unknown"),
            },
            "seed": (plan.get("evaluation") or {}).get("seeds", [42])[0],
            "budget_class": budget_class,
            "budget": {
                "timeout_seconds": timeout,
                "gpu_count": 1,
                "max_runtime": max_runtime,
            },
            "metrics_spec": {
                "primary": protocol["objective"]["primary"]["metric"],
                "secondary": [s["metric"] for s in protocol["objective"].get("secondary") or []],
            },
            "commands": {
                "train": ["python", "run_detection_experiment.py", "--mode", "train"],
                "evaluate": ["python", "run_detection_experiment.py", "--mode", "eval"],
            },
            "expected_artifacts": ["metrics.json", "checkpoint_selection.json"],
            "git_base_sha": "abcdef1",
            "adapter": self.adapter_key,
            "approval_status": "candidate",
            "frozen_fingerprint_id": fingerprint_id,
            "materialization": {
                "proposed_changes": changes,
                "how_only": True,
                "how": how,
                "how_signature": how["signature"],
                "legacy_parameters": dict(how["legacy_parameters"]),
                "execution_mode": how.get("execution_mode", "fast_eval"),
                "evaluation_scope": how.get("evaluation_scope", "fast_eval_subset"),
                "claim_level": how.get("claim_level", "exploratory_comparison"),
            },
        }
        validate_named("experiment_contract", contract)
        return contract

    def validate_contract(self, candidate: Mapping[str, Any]) -> None:
        validate_named("experiment_contract", dict(candidate))
        if candidate.get("adapter") != self.adapter_key:
            raise MaterializeRejected("adapter key mismatch")

    def compute_fingerprint(
        self, protocol: Mapping[str, Any], contract: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        return compute_fingerprint(protocol, contract)

    def parse_metrics(self, output_dir: Path | str) -> dict[str, Any]:
        return parse_metrics(output_dir)

    def collect_artifacts(
        self, output_dir: Path | str, contract: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        expected = list((contract or {}).get("expected_artifacts") or [])
        return collect_artifacts(output_dir, expected or None)

    def execute(
        self,
        contract: Mapping[str, Any],
        protocol: Mapping[str, Any],
        *,
        output_dir: Path | str,
        dry_run: bool = True,
        live_runner: LiveRunner | None = None,
        expected_fingerprint: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run or dry-run. Default dry_run=True: no GPU.

        live_runner is optional HOW hook to existing CUDA orchestrator.
        Fingerprint mismatch is reported; Adapter does not KEEP/DISCARD.
        """
        self.validate_contract(contract)
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        fingerprint = self.compute_fingerprint(protocol, contract)
        comparable = True
        if expected_fingerprint is not None:
            comparable = fingerprints_equivalent(expected_fingerprint, fingerprint)

        how = dict((contract.get("materialization") or {}).get("how") or {})
        how_sig = str(
            (contract.get("materialization") or {}).get("how_signature")
            or how.get("signature")
            or ""
        )
        self._emit(
            contract,
            event_type="fingerprint_check",
            phase="experiment",
            actor_role="adapter",
            payload={
                "comparable": comparable,
                "fingerprint_id": fingerprint["fingerprint_id"],
                "how_signature": how_sig,
                "how": how_identity(how) if how else {},
                "frozen_hashes_exclude_how": True,
            },
        )
        self._emit(
            contract,
            event_type="tool_call",
            phase="experiment",
            actor_role="adapter",
            payload={
                "tool": "run_experiment",
                "dry_run": dry_run,
                "commands": contract.get("commands"),
                "how_signature": how_sig,
                "legacy_parameters": dict(
                    (contract.get("materialization") or {}).get("legacy_parameters") or {}
                ),
            },
        )

        if not dry_run:
            if live_runner is None:
                raise MaterializeRejected(
                    "live execute requires live_runner; default path is dry_run"
                )
            run_view = dict(live_runner(contract, output))
            status = str(run_view.get("status") or "failed")
            if status == "timed_out":
                status = "timeout"
        elif (output / "metrics.json").is_file():
            # Offline recovery of already-produced artifacts (no new GPU work).
            status = "completed"
            run_view = {"status": status, "source": "existing_artifacts"}
        else:
            status = "dry_run"
            run_view = {"status": status, "would_run": contract.get("commands")}

        parsed = self.parse_metrics(output) if status in {"completed", "success", "dry_run"} else {
            "metrics": {},
            "raw": {},
            "sources": [],
        }
        artifacts = self.collect_artifacts(output, contract)
        self._emit(
            contract,
            event_type="execution",
            phase="experiment",
            actor_role="adapter",
            payload={"status": status, "comparable": comparable, "output_dir": str(output)},
            artifact_refs=list(artifacts.get("paths") or []),
        )
        if parsed.get("sources"):
            self._emit(
                contract,
                event_type="metrics_parsed",
                phase="experiment",
                actor_role="adapter",
                payload={"metrics": parsed.get("metrics"), "sources": parsed.get("sources")},
            )

        handle = {
            "run_id": contract["run_id"],
            "status": status,
            "dry_run": bool(dry_run),
            "output_dir": str(output),
            "fingerprint": fingerprint,
            "fingerprint_comparable": comparable,
            "metrics": parsed.get("metrics") or {},
            "raw_metric_refs": parsed.get("sources") or [],
            "artifacts": artifacts,
            "run_view": run_view,
        }
        return handle

    def evaluate(
        self,
        contract: Mapping[str, Any],
        handle: Mapping[str, Any],
    ) -> dict[str, Any]:
        return build_experiment_result(
            contract,
            metrics=handle.get("metrics") or {},
            artifacts=handle.get("artifacts") or {"paths": [], "missing_expected": []},
            execution_status=str(handle.get("status") or "failed"),
            raw_metric_refs=list(handle.get("raw_metric_refs") or []),
        )

    def _emit(
        self,
        contract: Mapping[str, Any],
        *,
        event_type: str,
        phase: str,
        actor_role: str,
        payload: dict[str, Any],
        artifact_refs: list[str] | None = None,
    ) -> None:
        if self.events is None:
            return
        self.events.append(
            {
                "project_id": contract["project_id"],
                "run_id": contract.get("run_id"),
                "plan_id": contract.get("plan_id"),
                "protocol_version": contract.get("protocol_version"),
                "fingerprint_id": contract.get("frozen_fingerprint_id"),
                "event_type": event_type,
                "actor_role": actor_role,
                "phase": phase,
                "artifact_refs": artifact_refs or [],
                "payload": payload,
            }
        )
