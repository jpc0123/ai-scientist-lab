"""CUDA Vendor DFINE Fast Eval orchestrator (v2.3.2).

Wraps ExperimentService.run_contract with doctor gates, exploratory claim
semantics, and structured execution metadata. Never declares formal success.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scientist_lab.domain.contracts import ExperimentContract
from scientist_lab.tasks.rgbt_detection.adapter import RGBTDetectionAdapter
from scientist_lab.tasks.rgbt_detection.cuda_doctor import (
    CUDA_ENV_KEY,
    EXAMPLE_CONTRACT,
    build_dfine_cuda_doctor,
)
from scientist_lab.tasks.rgbt_detection.decision_rules import (
    is_exploratory_fast_eval_context,
)
from scientist_lab.tasks.rgbt_detection.vendor_audit import PINNED_COMMIT


class DfineCudaOrchestratorError(ValueError):
    """Raised when Fast Eval orchestration cannot proceed safely."""


DEFAULT_CONTRACT_REL = EXAMPLE_CONTRACT
ORCHESTRATOR_ID = "dfine_cuda_fast_eval_v232"


@dataclass
class DfineCudaFastEvalPlan:
    contract: ExperimentContract
    contract_path: Path | None
    doctor_report: dict[str, Any]
    claim_gate: dict[str, Any]
    runner_profile: str
    environment_key: str = CUDA_ENV_KEY
    requested_backend: str = "dfine"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_path": str(self.contract_path) if self.contract_path else None,
            "node_id": self.contract.node_id,
            "project_id": self.contract.project_id,
            "runner_profile": self.runner_profile,
            "environment_key": self.environment_key,
            "requested_backend": self.requested_backend,
            "execution_mode": self.contract.execution_mode,
            "claim_level": (self.contract.task_config or {}).get("claim_level"),
            "doctor_overall": self.doctor_report.get("overall"),
            "live_ready": self.doctor_report.get("live_ready"),
            "claim_gate": self.claim_gate,
            "metadata": self.metadata,
        }


class DfineCudaFastEvalOrchestrator:
    def __init__(self, experiments: Any) -> None:
        self.experiments = experiments
        self.project_root = Path(experiments.settings.project_root).resolve()
        self.outputs_root = Path(experiments.settings.outputs_dir).resolve()

    def doctor(
        self,
        *,
        probe_runtime: bool = True,
        require_live_ready: bool = False,
    ) -> dict[str, Any]:
        report = build_dfine_cuda_doctor(
            self.project_root,
            image_registry=dict(self.experiments.settings.image_registry or {}),
            probe_runtime=probe_runtime,
        )
        if require_live_ready and not report.get("live_ready"):
            raise DfineCudaOrchestratorError(
                "CUDA DFINE live_ready=false; "
                "fix docker image / nvidia-smi / vendor pin, "
                "or omit --require-live-ready / use --dry-run"
            )
        return report

    def resolve_contract_path(
        self, path: Path | str | None = None
    ) -> Path:
        if path is None:
            return self.project_root / DEFAULT_CONTRACT_REL
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.project_root / candidate
        return candidate.resolve()

    def load_contract(
        self,
        path: Path | str | None = None,
        *,
        overrides: dict[str, Any] | None = None,
    ) -> ExperimentContract:
        from scientist_lab.services.experiment_service import load_contract

        resolved = self.resolve_contract_path(path)
        if not resolved.is_file():
            raise DfineCudaOrchestratorError(f"contract not found: {resolved}")
        contract = load_contract(resolved)
        if overrides:
            data = contract.model_dump(mode="json")
            for key, value in overrides.items():
                if key == "parameters" and isinstance(value, dict):
                    params = dict(data.get("parameters") or {})
                    params.update(value)
                    data["parameters"] = params
                elif key == "task_config" and isinstance(value, dict):
                    tc = dict(data.get("task_config") or {})
                    tc.update(value)
                    data["task_config"] = tc
                else:
                    data[key] = value
            contract = ExperimentContract.model_validate(data)
        return contract

    def _claim_gate(self, contract: ExperimentContract) -> dict[str, Any]:
        claim = str((contract.task_config or {}).get("claim_level") or "")
        scope = str((contract.task_config or {}).get("evaluation_scope") or "")
        exploratory = is_exploratory_fast_eval_context(
            task_type=contract.task_type,
            execution_mode=contract.execution_mode,
            claim_level=claim,
            evaluation_scope=scope,
        )
        return {
            "exploratory_only": True,
            "formal_success": False,
            "formal_claim_allowed": False,
            "is_exploratory_fast_eval_context": exploratory,
            "claim_level": claim or None,
            "evaluation_scope": scope or None,
            "note": (
                "Fast Eval Vendor run is exploratory; "
                "stand-in or incomplete evidence cannot unlock claim_formal_dfine"
            ),
        }

    def plan(
        self,
        *,
        contract_path: Path | str | None = None,
        runner_profile: str | None = None,
        probe_runtime: bool = True,
        require_live_ready: bool = False,
        overrides: dict[str, Any] | None = None,
    ) -> DfineCudaFastEvalPlan:
        doctor_report = self.doctor(
            probe_runtime=probe_runtime,
            require_live_ready=require_live_ready,
        )
        resolved = self.resolve_contract_path(contract_path)
        contract = self.load_contract(resolved, overrides=overrides)

        if contract.task_type != "rgbt_detection":
            raise DfineCudaOrchestratorError(
                f"expected task_type=rgbt_detection, got {contract.task_type!r}"
            )
        if contract.execution_mode != "fast_eval":
            raise DfineCudaOrchestratorError(
                f"expected execution_mode=fast_eval, got {contract.execution_mode!r}"
            )
        if contract.environment_key != CUDA_ENV_KEY:
            raise DfineCudaOrchestratorError(
                f"expected environment_key={CUDA_ENV_KEY}, "
                f"got {contract.environment_key!r}"
            )

        profile = runner_profile or contract.runner_profile
        if runner_profile and runner_profile != contract.runner_profile:
            data = contract.model_dump(mode="json")
            data["runner_profile"] = runner_profile
            contract = ExperimentContract.model_validate(data)
            profile = runner_profile

        RGBTDetectionAdapter().validate_contract(contract)
        params = dict(contract.parameters or {})
        backend = str(
            params.get("dfine_backend") or params.get("baseline_backend") or "auto"
        ).strip().lower()
        claim_gate = self._claim_gate(contract)
        if not claim_gate["is_exploratory_fast_eval_context"]:
            raise DfineCudaOrchestratorError(
                "contract is not an exploratory Fast Eval context; "
                "set claim_level=exploratory_comparison or "
                "evaluation_scope=fast_eval_subset"
            )

        return DfineCudaFastEvalPlan(
            contract=contract,
            contract_path=resolved,
            doctor_report=doctor_report,
            claim_gate=claim_gate,
            runner_profile=profile,
            environment_key=contract.environment_key,
            requested_backend=backend,
            metadata={
                "orchestrator": ORCHESTRATOR_ID,
                "vendor_commit_expected": PINNED_COMMIT,
                "baseline": params.get("baseline") or "dfine_s",
            },
        )

    def record_execution_metadata(
        self,
        execution_id: str,
        *,
        plan: DfineCudaFastEvalPlan,
        run_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist exploratory metadata for audit (does not unlock formal claims)."""
        output_dir = None
        if run_result:
            output_dir = run_result.get("output_directory")
        model_summary: dict[str, Any] = {}
        resource_usage: dict[str, Any] = {}
        if output_dir:
            root = Path(str(output_dir))
            ms = root / "model_summary.json"
            ru = root / "resource_usage.json"
            if ms.is_file():
                model_summary = json.loads(ms.read_text(encoding="utf-8"))
            if ru.is_file():
                resource_usage = json.loads(ru.read_text(encoding="utf-8"))

        impl = str(
            model_summary.get("baseline_implementation")
            or model_summary.get("implementation")
            or ""
        )
        standin = "standin" in impl.lower() or "stand_in" in impl.lower()
        if not impl and plan.requested_backend in {"standin", "torch_mini", "mini"}:
            standin = True
        vendor = plan.requested_backend in {"dfine", "dfine_s", "vendor"} and not standin

        meta = {
            "orchestrator": ORCHESTRATOR_ID,
            "execution_id": execution_id,
            "dfine_backend_requested": plan.requested_backend,
            "baseline_implementation": impl or None,
            "standin_or_vendor": "standin" if standin else ("vendor" if vendor else "unknown"),
            "device": resource_usage.get("device") or model_summary.get("device"),
            "cuda_available": resource_usage.get("cuda_available"),
            "environment_key": plan.environment_key,
            "runner_profile": plan.runner_profile,
            "claim_level": plan.claim_gate.get("claim_level"),
            "exploratory_only": True,
            "formal_success": False,
            "vendor_commit_expected": PINNED_COMMIT,
            "status": (run_result or {}).get("status"),
            "output_directory": output_dir,
        }

        out_dir = (
            self.outputs_root
            / plan.contract.project_id
            / "_dfine_cuda_runs"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{execution_id}.json"
        path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        meta["metadata_path"] = str(path)
        return meta

    def submit(
        self,
        plan: DfineCudaFastEvalPlan,
        *,
        wait: bool = True,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        payload = {
            "orchestrator": ORCHESTRATOR_ID,
            "exploratory_only": True,
            "formal_success": False,
            "dry_run": bool(dry_run),
            "plan": plan.to_dict(),
            "doctor": plan.doctor_report,
            "claim_gate": plan.claim_gate,
        }
        if dry_run:
            payload["would_run"] = {
                "contract_path": str(plan.contract_path) if plan.contract_path else None,
                "node_id": plan.contract.node_id,
                "runner_profile": plan.runner_profile,
                "environment_key": plan.environment_key,
                "dfine_backend": plan.requested_backend,
                "wait": wait,
            }
            payload["status"] = "dry_run"
            return payload

        result = self.experiments.run_contract(plan.contract, wait=wait)
        run_view = {
            "execution_id": result.execution_id,
            "status": str(result.status),
            "return_code": result.return_code,
            "metrics": result.metrics,
            "artifact_count": len(result.artifacts or []),
            "output_directory": result.output_directory,
            "error": None
            if result.error is None
            else (
                result.error.model_dump()
                if hasattr(result.error, "model_dump")
                else str(result.error)
            ),
        }
        meta = self.record_execution_metadata(
            result.execution_id,
            plan=plan,
            run_result=run_view,
        )
        payload["run"] = run_view
        payload["execution_metadata"] = meta
        payload["status"] = run_view["status"]

        # v2.3.3: annotate Evidence + metrics feedback (exploratory only)
        if str(result.status) == "completed":
            try:
                from scientist_lab.tasks.rgbt_detection.dfine_evidence import (
                    record_dfine_fast_eval_evidence,
                )

                evidence_pkg = record_dfine_fast_eval_evidence(
                    self.experiments,
                    execution_id=result.execution_id,
                    execution_metadata={
                        **meta,
                        "project_id": plan.contract.project_id,
                    },
                    metrics=dict(result.metrics or {}),
                    contract=plan.contract.model_dump(mode="json"),
                    refresh_claim_matrix=True,
                )
                payload["evidence_feedback"] = evidence_pkg
            except Exception as exc:  # noqa: BLE001
                payload["evidence_feedback_error"] = str(exc)
        return payload


def run_dfine_cuda_fast_eval(
    experiments: Any,
    *,
    contract_path: Path | str | None = None,
    runner_profile: str | None = None,
    wait: bool = True,
    dry_run: bool = False,
    require_live_ready: bool = False,
    probe_runtime: bool = True,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    orch = DfineCudaFastEvalOrchestrator(experiments)
    # Dry-run never requires live GPU readiness.
    plan = orch.plan(
        contract_path=contract_path,
        runner_profile=runner_profile,
        probe_runtime=False if dry_run else probe_runtime,
        require_live_ready=False if dry_run else require_live_ready,
        overrides=overrides,
    )
    return orch.submit(plan, wait=wait, dry_run=dry_run)
