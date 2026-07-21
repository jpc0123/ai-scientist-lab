from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.orm import sessionmaker

from scientist_lab.domain.contracts import ExperimentContract
from scientist_lab.protocols.models import ExperimentProtocol, ProtocolVerificationReport
from scientist_lab.protocols.repository import ProtocolRepository
from scientist_lab.protocols.verifier import ProtocolVerifier, ProtocolViolationError


class ProtocolService:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._repo = ProtocolRepository(session_factory)
        self.verifier = ProtocolVerifier()

    def create_from_path(self, path: Path) -> ExperimentProtocol:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return self.create_from_dict(data)

    def create_from_dict(self, data: dict[str, Any]) -> ExperimentProtocol:
        payload = dict(data)
        if "created_at" not in payload or payload.get("created_at") in (None, ""):
            payload["created_at"] = datetime.now(timezone.utc).replace(microsecond=0)
        protocol = ExperimentProtocol.model_validate(payload)
        integrity = self.verifier.validate_protocol(protocol)
        if not integrity.valid:
            raise ValueError(
                "invalid protocol: " + "; ".join(integrity.blocking_issues)
            )
        return self._repo.upsert(protocol)

    def get(self, protocol_id: str) -> ExperimentProtocol | None:
        return self._repo.get(protocol_id)

    def require(self, protocol_id: str) -> ExperimentProtocol:
        protocol = self.get(protocol_id)
        if protocol is None:
            raise KeyError(f"protocol not found: {protocol_id}")
        return protocol

    def list_protocols(
        self,
        *,
        project_id: str | None = None,
    ) -> list[ExperimentProtocol]:
        return self._repo.list_protocols(project_id=project_id)

    def validate(
        self,
        protocol_id: str,
        *,
        contract: ExperimentContract | None = None,
    ) -> ProtocolVerificationReport:
        protocol = self.require(protocol_id)
        integrity = self.verifier.validate_protocol(protocol)
        if contract is None:
            return integrity
        contract_report = self.verifier.verify_contract(protocol, contract)
        return ProtocolVerificationReport(
            valid=integrity.valid and contract_report.valid,
            warnings=list(integrity.warnings) + list(contract_report.warnings),
            blocking_issues=list(integrity.blocking_issues)
            + list(contract_report.blocking_issues),
            protocol_id=protocol.protocol_id,
            contract_node_id=contract.node_id,
        )

    def enforce_contract(self, contract: ExperimentContract) -> ProtocolVerificationReport | None:
        """If contract declares protocol_id, verify and raise on blocking issues."""
        protocol_id = (contract.protocol_id or "").strip()
        if not protocol_id:
            return None
        protocol = self.require(protocol_id)
        report = self.verifier.verify_contract(protocol, contract)
        if not report.valid:
            raise ProtocolViolationError(report)
        return report

    def validate_formal_triad(
        self,
        contracts: dict[str, ExperimentContract | dict],
        *,
        protocol_id: str | None = None,
        matched_seeds: list[int] | None = None,
    ) -> ProtocolVerificationReport:
        from scientist_lab.protocols.formal_triad import validate_formal_triad

        resolved_protocol_id = protocol_id
        if not resolved_protocol_id:
            for raw in contracts.values():
                if isinstance(raw, ExperimentContract):
                    candidate = raw.protocol_id
                else:
                    candidate = (raw or {}).get("protocol_id")
                if candidate:
                    resolved_protocol_id = str(candidate)
                    break
        if not resolved_protocol_id:
            raise ValueError("protocol_id required to validate formal triad")
        protocol = self.require(resolved_protocol_id)
        return validate_formal_triad(
            protocol, contracts, matched_seeds=matched_seeds
        )
