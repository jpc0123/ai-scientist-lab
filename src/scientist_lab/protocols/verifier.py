from __future__ import annotations

from typing import Any

from scientist_lab.domain.contracts import ExperimentContract
from scientist_lab.protocols.models import ExperimentProtocol, ProtocolVerificationReport


class ProtocolViolationError(ValueError):
    """Raised when a contract violates its declared ExperimentProtocol."""

    def __init__(self, report: ProtocolVerificationReport) -> None:
        self.report = report
        issues = "; ".join(report.blocking_issues) or "protocol violation"
        super().__init__(issues)


class ProtocolVerifier:
    """Verify contracts against a project-level ExperimentProtocol."""

    def validate_protocol(self, protocol: ExperimentProtocol) -> ProtocolVerificationReport:
        warnings: list[str] = []
        blocking: list[str] = []

        if not protocol.seeds:
            blocking.append("seeds must be non-empty")
        if protocol.execution_mode not in {"fast_eval", "full_train"}:
            blocking.append(f"unsupported execution_mode: {protocol.execution_mode}")
        if not protocol.primary_metric.strip():
            blocking.append("primary_metric required")
        if not protocol.dataset_reference.strip():
            blocking.append("dataset_reference required")
        if not protocol.environment_key.strip():
            blocking.append("environment_key required")
        if not protocol.code_reference.strip():
            blocking.append("code_reference required")
        if not protocol.code_version.strip():
            blocking.append("code_version required")

        overlap = set(protocol.allowed_variables) & set(protocol.fixed_parameters)
        if overlap:
            warnings.append(
                "allowed_variables overlap fixed_parameters keys: "
                + ", ".join(sorted(overlap))
            )

        if protocol.claim_level == "benchmark_evidence" and protocol.execution_mode == "fast_eval":
            warnings.append(
                "claim_level=benchmark_evidence with execution_mode=fast_eval "
                "is usually too strong for subset budgets"
            )

        return ProtocolVerificationReport(
            valid=len(blocking) == 0,
            warnings=warnings,
            blocking_issues=blocking,
            protocol_id=protocol.protocol_id,
        )

    def verify_contract(
        self,
        protocol: ExperimentProtocol,
        contract: ExperimentContract,
    ) -> ProtocolVerificationReport:
        warnings: list[str] = []
        blocking: list[str] = []

        if contract.protocol_id and contract.protocol_id != protocol.protocol_id:
            blocking.append(
                f"protocol_id mismatch: contract={contract.protocol_id}, "
                f"expected={protocol.protocol_id}"
            )

        if contract.project_id != protocol.project_id:
            blocking.append(
                f"project_id differs: contract={contract.project_id}, "
                f"protocol={protocol.project_id}"
            )

        if (contract.task_type or "") and contract.task_type != protocol.task_type:
            blocking.append(
                f"task_type differs: contract={contract.task_type}, "
                f"protocol={protocol.task_type}"
            )

        if contract.dataset_reference != protocol.dataset_reference:
            blocking.append(
                f"dataset_reference differs: contract={contract.dataset_reference}, "
                f"protocol={protocol.dataset_reference}"
            )

        if contract.environment_key != protocol.environment_key:
            blocking.append(
                f"environment_key differs: contract={contract.environment_key}, "
                f"protocol={protocol.environment_key}"
            )

        if contract.code_reference != protocol.code_reference:
            blocking.append(
                f"code_reference differs: contract={contract.code_reference}, "
                f"protocol={protocol.code_reference}"
            )

        if contract.code_version:
            if contract.code_version != protocol.code_version:
                blocking.append(
                    f"code_version differs: contract={contract.code_version}, "
                    f"protocol={protocol.code_version}"
                )
        else:
            warnings.append(
                f"contract code_version missing; protocol expects {protocol.code_version}"
            )

        if contract.execution_mode != protocol.execution_mode:
            blocking.append(
                f"execution_mode differs: contract={contract.execution_mode}, "
                f"protocol={protocol.execution_mode}"
            )

        if contract.seed not in protocol.seeds:
            blocking.append(
                f"seed {contract.seed} not in protocol.seeds {protocol.seeds}"
            )

        params = dict(contract.parameters or {})
        for key, expected in (protocol.fixed_parameters or {}).items():
            if key not in params:
                blocking.append(f"fixed parameter missing on contract: {key}")
                continue
            if not _values_equal(params[key], expected):
                blocking.append(
                    f"fixed parameter {key} differs: contract={params[key]!r}, "
                    f"protocol={expected!r}"
                )

        allowed = set(protocol.allowed_variables or [])
        fixed_keys = set(protocol.fixed_parameters or {})
        for key, value in params.items():
            if key in allowed:
                continue
            if key in fixed_keys:
                continue
            # Extra keys beyond fixed + allowed are blocked for formal protocol runs.
            blocking.append(
                f"parameter {key}={value!r} is not in allowed_variables "
                f"and not declared in fixed_parameters"
            )

        task_config = dict(contract.task_config or {})
        claim = str(task_config.get("claim_level") or "").strip()
        if claim and claim != protocol.claim_level:
            blocking.append(
                f"claim_level differs: contract={claim}, protocol={protocol.claim_level}"
            )
        elif not claim:
            warnings.append(
                f"task_config.claim_level missing; protocol expects {protocol.claim_level}"
            )

        primary = str(task_config.get("primary_metric") or "").strip()
        if primary and primary != protocol.primary_metric:
            blocking.append(
                f"primary_metric differs: contract={primary}, "
                f"protocol={protocol.primary_metric}"
            )

        split = str(task_config.get("split_reference") or "").strip()
        if split and split != protocol.split_reference:
            blocking.append(
                f"split_reference differs: contract={split}, "
                f"protocol={protocol.split_reference}"
            )
        elif not split:
            warnings.append(
                f"task_config.split_reference missing; "
                f"protocol expects {protocol.split_reference}"
            )

        dataset_version = str(task_config.get("dataset_version") or "").strip()
        if dataset_version and dataset_version != protocol.dataset_version:
            blocking.append(
                f"dataset_version differs: contract={dataset_version}, "
                f"protocol={protocol.dataset_version}"
            )

        return ProtocolVerificationReport(
            valid=len(blocking) == 0,
            warnings=warnings,
            blocking_issues=blocking,
            protocol_id=protocol.protocol_id,
            contract_node_id=contract.node_id,
        )

    def verify_contracts_share_protocol(
        self,
        protocol: ExperimentProtocol,
        contracts: list[ExperimentContract],
    ) -> ProtocolVerificationReport:
        warnings: list[str] = []
        blocking: list[str] = []
        for contract in contracts:
            report = self.verify_contract(protocol, contract)
            warnings.extend(report.warnings)
            blocking.extend(
                [f"{contract.node_id}: {issue}" for issue in report.blocking_issues]
            )
        return ProtocolVerificationReport(
            valid=len(blocking) == 0,
            warnings=warnings,
            blocking_issues=blocking,
            protocol_id=protocol.protocol_id,
        )


def _values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left) == float(right)
    return left == right
