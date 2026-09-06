"""ResultParser: Adapter.parse_metrics → canonical ExperimentResult.

Does not KEEP/DISCARD. Does not invent APS from mAP.
"""

from __future__ import annotations

from typing import Any, Mapping

from scientist_lab.core.schema_registry import validate_named

_STATUS_MAP = {
    "dry_run": "success",
    "completed": "success",
    "success": "success",
    "failed": "failed",
    "timeout": "timeout",
    "timed_out": "timeout",
    "cancelled": "cancelled",
}


def normalize_execution_status(status: str | None) -> str:
    raw = str(status or "failed")
    if raw == "timed_out":
        raw = "timeout"
    return _STATUS_MAP.get(raw, "failed")


def build_experiment_result(
    contract: Mapping[str, Any],
    *,
    metrics: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    execution_status: str,
    experiment_sha: str | None = None,
    runtime_seconds: float | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
    scientific_outcome: str | None = None,
    raw_metric_refs: list[str] | None = None,
) -> dict[str, Any]:
    run_id = contract.get("run_id") or (contract.get("run") or {}).get("run_id")
    if not run_id:
        raise KeyError("experiment_result requires run_id on Freeze contract or handle")
    result = {
        "schema_version": "1.0.0",
        "run_id": run_id,
        "experiment_sha": experiment_sha,
        "metrics": dict(metrics),
        "artifacts": {
            "paths": list(artifacts.get("paths") or []),
            "missing_expected": list(artifacts.get("missing_expected") or []),
        },
        "execution": {
            "status": normalize_execution_status(execution_status),
            "exit_code": 0 if normalize_execution_status(execution_status) == "success" else 1,
            "runtime_seconds": runtime_seconds,
            "error_type": error_type,
            "error_message": error_message,
        },
        "raw_metric_refs": list(raw_metric_refs or artifacts.get("paths") or []),
    }
    if scientific_outcome:
        result["scientific_outcome"] = scientific_outcome
    if experiment_sha is None:
        del result["experiment_sha"]
    validate_named("experiment_result", result)
    return result


class ResultParser:
    def from_handle(
        self,
        contract: Mapping[str, Any],
        handle: Mapping[str, Any],
    ) -> dict[str, Any]:
        merged = dict(contract)
        if not merged.get("run_id"):
            merged["run_id"] = handle.get("run_id")
        run_view = handle.get("run_view") if isinstance(handle.get("run_view"), dict) else {}
        nested = run_view.get("run") if isinstance(run_view.get("run"), dict) else {}
        err = nested.get("error") if isinstance(nested.get("error"), dict) else {}
        return build_experiment_result(
            merged,
            metrics=handle.get("metrics") or {},
            artifacts=handle.get("artifacts") or {"paths": [], "missing_expected": []},
            execution_status=str(handle.get("status") or nested.get("status") or "failed"),
            raw_metric_refs=list(handle.get("raw_metric_refs") or []),
            error_type=err.get("error_type") or run_view.get("error_type"),
            error_message=err.get("message") or run_view.get("error_message"),
        )
