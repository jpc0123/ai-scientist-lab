"""CUDA Formal Triad orchestrator (v2.3.4).

Plans/submits rgb+thermal+fusion Vendor DFINE Fast Eval contracts under a
shared CUDA protocol. Default dry-run. Never claims formal superiority.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scientist_lab.domain.contracts import ExperimentContract
from scientist_lab.protocols.formal_triad import (
    DEFAULT_CUDA_FORMAL_PROTOCOL_ID,
    DEFAULT_CUDA_FORMAL_TRIAD_NODES,
    infer_triad_role,
    validate_formal_triad,
)
from scientist_lab.tasks.rgbt_detection.dfine_cuda_orchestrator import (
    DfineCudaFastEvalOrchestrator,
    DfineCudaOrchestratorError,
    ORCHESTRATOR_ID as SINGLE_ORCHESTRATOR_ID,
)
from scientist_lab.tasks.rgbt_detection.fast_eval_triad import CAVEATS, TRIAD_ROLES


ORCHESTRATOR_ID = "dfine_cuda_formal_triad_v234"

DEFAULT_CUDA_PROTOCOL_REL = "examples/rgbt_protocol_cuda.json"
DEFAULT_CUDA_CONTRACT_RELS = {
    "rgb": "examples/rgbt_formal_cuda_rgb_contract.json",
    "thermal": "examples/rgbt_formal_cuda_thermal_contract.json",
    "fusion": "examples/rgbt_formal_cuda_fusion_contract.json",
}


class DfineCudaFormalTriadOrchestrator:
    def __init__(self, experiments: Any) -> None:
        self.experiments = experiments
        self.project_root = Path(experiments.settings.project_root).resolve()
        self.single = DfineCudaFastEvalOrchestrator(experiments)

    def _resolve(self, rel_or_path: Path | str) -> Path:
        path = Path(rel_or_path)
        if not path.is_absolute():
            path = self.project_root / path
        return path.resolve()

    def ensure_cuda_protocol(
        self,
        *,
        protocol_path: Path | str | None = None,
        protocol_id: str = DEFAULT_CUDA_FORMAL_PROTOCOL_ID,
    ) -> dict[str, Any]:
        path = self._resolve(protocol_path or DEFAULT_CUDA_PROTOCOL_REL)
        if not path.is_file():
            raise DfineCudaOrchestratorError(f"CUDA protocol missing: {path}")
        existing = self.experiments.protocols.get(protocol_id)
        if existing is None:
            created = self.experiments.protocols.create_from_path(path)
            return {
                "protocol_id": created.protocol_id,
                "created": True,
                "path": str(path),
            }
        return {
            "protocol_id": existing.protocol_id,
            "created": False,
            "path": str(path),
        }

    def load_triad_contracts(
        self,
        *,
        contract_paths: dict[str, Path | str] | None = None,
    ) -> dict[str, ExperimentContract]:
        from scientist_lab.services.experiment_service import load_contract

        rels = contract_paths or DEFAULT_CUDA_CONTRACT_RELS
        loaded: dict[str, ExperimentContract] = {}
        for role in TRIAD_ROLES:
            if role not in rels:
                raise DfineCudaOrchestratorError(f"missing contract for role={role}")
            path = self._resolve(rels[role])
            if not path.is_file():
                raise DfineCudaOrchestratorError(f"contract not found: {path}")
            contract = load_contract(path)
            inferred = infer_triad_role(contract)
            if inferred != role:
                raise DfineCudaOrchestratorError(
                    f"contract {path.name} inferred role={inferred!r}, expected {role!r}"
                )
            # Prefer stable CUDA formal node ids when missing mapping
            expected_id = DEFAULT_CUDA_FORMAL_TRIAD_NODES[role]
            if contract.node_id != expected_id and inferred == role:
                # allow if role matches via input_mode
                pass
            loaded[role] = contract
        return loaded

    def validate(
        self,
        contracts: dict[str, ExperimentContract],
        *,
        protocol_id: str = DEFAULT_CUDA_FORMAL_PROTOCOL_ID,
    ) -> dict[str, Any]:
        protocol = self.experiments.protocols.require(protocol_id)
        report = validate_formal_triad(protocol, contracts)
        return {
            "ok": bool(report.valid),
            "valid": bool(report.valid),
            "protocol_id": protocol.protocol_id,
            "blocking_issues": list(report.blocking_issues or []),
            "warnings": list(report.warnings or []),
            "roles": {
                role: {
                    "node_id": contracts[role].node_id,
                    "input_mode": (contracts[role].parameters or {}).get("input_mode"),
                    "fusion_method": (contracts[role].parameters or {}).get(
                        "fusion_method"
                    ),
                }
                for role in TRIAD_ROLES
            },
        }

    def plan(
        self,
        *,
        contract_paths: dict[str, Path | str] | None = None,
        protocol_path: Path | str | None = None,
        protocol_id: str = DEFAULT_CUDA_FORMAL_PROTOCOL_ID,
        probe_runtime: bool = True,
        require_live_ready: bool = False,
    ) -> dict[str, Any]:
        doctor = self.single.doctor(
            probe_runtime=probe_runtime,
            require_live_ready=require_live_ready,
        )
        proto_meta = self.ensure_cuda_protocol(
            protocol_path=protocol_path, protocol_id=protocol_id
        )
        contracts = self.load_triad_contracts(contract_paths=contract_paths)
        validation = self.validate(contracts, protocol_id=protocol_id)
        if not validation.get("ok"):
            raise DfineCudaOrchestratorError(
                "CUDA formal triad validation failed: "
                + "; ".join(validation.get("blocking_issues") or [])
            )

        role_plans = {}
        for role, contract in contracts.items():
            # Plan via single-node orchestrator semantics without reloading path
            from scientist_lab.tasks.rgbt_detection.dfine_cuda_orchestrator import (
                DfineCudaFastEvalPlan,
            )
            from scientist_lab.tasks.rgbt_detection.adapter import RGBTDetectionAdapter
            from scientist_lab.tasks.rgbt_detection.cuda_doctor import CUDA_ENV_KEY

            if contract.environment_key != CUDA_ENV_KEY:
                raise DfineCudaOrchestratorError(
                    f"{role}: environment_key must be {CUDA_ENV_KEY}"
                )
            RGBTDetectionAdapter().validate_contract(contract)
            claim_gate = self.single._claim_gate(contract)
            if not claim_gate["is_exploratory_fast_eval_context"]:
                raise DfineCudaOrchestratorError(
                    f"{role}: not exploratory Fast Eval context"
                )
            params = dict(contract.parameters or {})
            backend = str(params.get("dfine_backend") or "dfine").strip().lower()
            role_plans[role] = DfineCudaFastEvalPlan(
                contract=contract,
                contract_path=self._resolve(
                    (contract_paths or DEFAULT_CUDA_CONTRACT_RELS)[role]
                ),
                doctor_report=doctor,
                claim_gate=claim_gate,
                runner_profile=contract.runner_profile,
                environment_key=contract.environment_key,
                requested_backend=backend,
                metadata={
                    "orchestrator": ORCHESTRATOR_ID,
                    "role": role,
                    "protocol_id": protocol_id,
                    "single_orchestrator": SINGLE_ORCHESTRATOR_ID,
                },
            ).to_dict()

        return {
            "orchestrator": ORCHESTRATOR_ID,
            "exploratory_only": True,
            "formal_success": False,
            "formal_superiority_claimed": False,
            "protocol": proto_meta,
            "doctor": doctor,
            "validation": validation,
            "role_plans": role_plans,
            "caveats": list(CAVEATS)
            + [
                "CUDA formal triad remains Fast Eval / exploratory; "
                "do not claim formal DFINE superiority from these runs."
            ],
        }

    def submit(
        self,
        *,
        contract_paths: dict[str, Path | str] | None = None,
        protocol_path: Path | str | None = None,
        protocol_id: str = DEFAULT_CUDA_FORMAL_PROTOCOL_ID,
        wait: bool = True,
        dry_run: bool = True,
        probe_runtime: bool = True,
        require_live_ready: bool = False,
    ) -> dict[str, Any]:
        plan = self.plan(
            contract_paths=contract_paths,
            protocol_path=protocol_path,
            protocol_id=protocol_id,
            probe_runtime=False if dry_run else probe_runtime,
            require_live_ready=False if dry_run else require_live_ready,
        )
        payload: dict[str, Any] = {
            **plan,
            "dry_run": bool(dry_run),
            "status": "dry_run" if dry_run else "running",
            "role_results": {},
        }
        if dry_run:
            payload["would_run"] = {
                role: {
                    "node_id": item.get("node_id"),
                    "environment_key": item.get("environment_key"),
                    "dfine_backend": item.get("requested_backend"),
                    "runner_profile": item.get("runner_profile"),
                }
                for role, item in (plan.get("role_plans") or {}).items()
            }
            return payload

        contracts = self.load_triad_contracts(contract_paths=contract_paths)
        results = {}
        all_ok = True
        for role, contract in contracts.items():
            role_plan = self.single.plan(
                contract_path=(contract_paths or DEFAULT_CUDA_CONTRACT_RELS)[role],
                runner_profile=contract.runner_profile,
                probe_runtime=probe_runtime,
                require_live_ready=require_live_ready,
            )
            # Ensure the loaded plan matches triad contract node
            if role_plan.contract.node_id != contract.node_id:
                role_plan.contract = contract
            result = self.single.submit(role_plan, wait=wait, dry_run=False)
            results[role] = result
            status = str((result.get("run") or {}).get("status") or result.get("status"))
            if status != "completed":
                all_ok = False
        payload["role_results"] = results
        payload["status"] = "completed" if all_ok else "failed"
        payload["formal_success"] = False
        payload["formal_superiority_claimed"] = False
        return payload


def run_dfine_cuda_formal_triad(
    experiments: Any,
    *,
    dry_run: bool = True,
    wait: bool = True,
    require_live_ready: bool = False,
    probe_runtime: bool = True,
    protocol_id: str = DEFAULT_CUDA_FORMAL_PROTOCOL_ID,
) -> dict[str, Any]:
    orch = DfineCudaFormalTriadOrchestrator(experiments)
    return orch.submit(
        dry_run=dry_run,
        wait=wait,
        require_live_ready=require_live_ready,
        probe_runtime=probe_runtime,
        protocol_id=protocol_id,
    )
