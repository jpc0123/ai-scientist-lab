"""Formal RGB / Thermal / Fusion triad under a shared ExperimentProtocol."""

from __future__ import annotations

from typing import Any

from scientist_lab.domain.contracts import ExperimentContract
from scientist_lab.protocols.models import ExperimentProtocol, ProtocolVerificationReport
from scientist_lab.protocols.verifier import ProtocolVerifier, _values_equal
from scientist_lab.tasks.rgbt_detection.fast_eval_triad import (
    EXPECTED_MODES,
    TRIAD_ROLES,
    validate_triad_contracts,
)


DEFAULT_FORMAL_TRIAD_NODES = {
    "rgb": "rgbt_formal_node_001",
    "thermal": "rgbt_formal_node_002",
    "fusion": "rgbt_formal_node_003",
}

DEFAULT_FORMAL_PROTOCOL_ID = "protocol_rgbt_001"

DEFAULT_MATCHED_SEEDS = [42, 43, 44]


def infer_triad_role(contract: dict[str, Any] | ExperimentContract) -> str | None:
    if isinstance(contract, ExperimentContract):
        payload = contract.model_dump(mode="json")
    else:
        payload = contract
    node_id = str(payload.get("node_id") or "")
    for role, expected in DEFAULT_FORMAL_TRIAD_NODES.items():
        if node_id == expected:
            return role
    params = dict(payload.get("parameters") or {})
    mode = str(params.get("input_mode") or "")
    fusion = str(params.get("fusion_method") or "")
    if mode == "rgb" and fusion in {"", "none"}:
        return "rgb"
    if mode == "thermal" and fusion in {"", "none"}:
        return "thermal"
    if mode in {"rgbt", "rgb_thermal"} and fusion == "early_concat":
        return "fusion"
    return None


def validate_formal_triad(
    protocol: ExperimentProtocol,
    contracts: dict[str, ExperimentContract | dict[str, Any]],
    *,
    matched_seeds: list[int] | None = None,
) -> ProtocolVerificationReport:
    """Validate three formal nodes share one protocol and only allowed variables differ."""
    warnings: list[str] = []
    blocking: list[str] = []
    verifier = ProtocolVerifier()

    missing = [role for role in TRIAD_ROLES if role not in contracts]
    if missing:
        return ProtocolVerificationReport(
            valid=False,
            warnings=warnings,
            blocking_issues=[f"missing roles: {', '.join(missing)}"],
            protocol_id=protocol.protocol_id,
        )

    normalized: dict[str, ExperimentContract] = {}
    for role, raw in contracts.items():
        if isinstance(raw, ExperimentContract):
            normalized[role] = raw
        else:
            normalized[role] = ExperimentContract.model_validate(raw)

    # Legacy fairness checks (budget / modes / exploratory claim).
    fairness = validate_triad_contracts(
        {role: c.model_dump(mode="json") for role, c in normalized.items()}
    )
    for issue in fairness.get("issues") or []:
        blocking.append(str(issue))
    for warning in fairness.get("warnings") or []:
        warnings.append(str(warning))

    seeds = list(matched_seeds or protocol.seeds or DEFAULT_MATCHED_SEEDS)
    if seeds != list(protocol.seeds):
        warnings.append(
            f"matched_seeds {seeds} differ from protocol.seeds {protocol.seeds}"
        )
    if len(seeds) < 3:
        warnings.append(
            f"matched seed count is {len(seeds)}; formal comparison prefers >= 3"
        )

    allowed = set(protocol.allowed_variables or [])
    reference = normalized["rgb"]

    for role, contract in normalized.items():
        if not contract.protocol_id:
            blocking.append(f"{role}: protocol_id required for formal triad")
        elif contract.protocol_id != protocol.protocol_id:
            blocking.append(
                f"{role}: protocol_id={contract.protocol_id!r} "
                f"differs from {protocol.protocol_id!r}"
            )

        report = verifier.verify_contract(protocol, contract)
        for issue in report.blocking_issues:
            blocking.append(f"{role}: {issue}")
        for warning in report.warnings:
            warnings.append(f"{role}: {warning}")

        expected = EXPECTED_MODES[role]
        params = dict(contract.parameters or {})
        for key, value in expected.items():
            if str(params.get(key) or "") != value:
                blocking.append(
                    f"{role}: parameters.{key} expected {value!r}, got {params.get(key)!r}"
                )

        impl = str((contract.task_config or {}).get("implementation") or "").strip()
        if impl in {"", "stand_in", "stand-in"}:
            warnings.append(
                f"{role}: implementation={impl or 'unspecified'!r}; "
                "do not treat results as formal DFINE evidence"
            )

    # Cross-node: only allowed_variables may differ in parameters.
    ref_params = dict(reference.parameters or {})
    for role in ("thermal", "fusion"):
        other = normalized[role]
        other_params = dict(other.parameters or {})
        keys = set(ref_params) | set(other_params)
        for key in sorted(keys):
            left = ref_params.get(key)
            right = other_params.get(key)
            if _values_equal(left, right):
                continue
            if key not in allowed:
                blocking.append(
                    f"{role}: parameter {key} differs from rgb but is not "
                    f"in allowed_variables ({left!r} vs {right!r})"
                )

        for field in (
            "dataset_reference",
            "environment_key",
            "code_reference",
            "code_version",
            "execution_mode",
            "project_id",
            "protocol_id",
        ):
            if getattr(reference, field) != getattr(other, field):
                blocking.append(
                    f"{role}: {field} differs from rgb "
                    f"({getattr(reference, field)!r} vs {getattr(other, field)!r})"
                )

    return ProtocolVerificationReport(
        valid=len(blocking) == 0,
        warnings=warnings,
        blocking_issues=blocking,
        protocol_id=protocol.protocol_id,
        contract_node_id=reference.node_id,
    )
