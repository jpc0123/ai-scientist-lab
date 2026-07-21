from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from sqlalchemy.orm import sessionmaker

from scientist_lab.domain.models import utc_now_iso
from scientist_lab.evidence.builder import (
    build_paired_comparison_evidence,
    build_resource_comparison_evidence,
)
from scientist_lab.evidence.claim_matrix import (
    build_claim_support_matrix,
    matrix_export_dict,
)
from scientist_lab.evidence.models import ClaimSupportMatrix, EvidenceRecord
from scientist_lab.evidence.repository import EvidenceRepository
from scientist_lab.storage.artifact_store import write_json


class EvidenceService:
    def __init__(
        self,
        session_factory: sessionmaker,
        *,
        outputs_root: Path,
    ) -> None:
        self._repo = EvidenceRepository(session_factory)
        self.outputs_root = Path(outputs_root)

    def get(self, evidence_id: str) -> EvidenceRecord | None:
        return self._repo.get(evidence_id)

    def require(self, evidence_id: str) -> EvidenceRecord:
        record = self.get(evidence_id)
        if record is None:
            raise KeyError(f"evidence not found: {evidence_id}")
        return record

    def list_evidence(
        self,
        *,
        project_id: str | None = None,
        protocol_id: str | None = None,
        evidence_type: str | None = None,
    ) -> list[EvidenceRecord]:
        return self._repo.list_evidence(
            project_id=project_id,
            protocol_id=protocol_id,
            evidence_type=evidence_type,
        )

    def persist(self, record: EvidenceRecord) -> EvidenceRecord:
        stored = self._repo.upsert(record)
        evidence_dir = self.outputs_root / stored.project_id / "evidence"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        path = evidence_dir / f"{stored.evidence_id}.json"
        payload = stored.model_dump(mode="json")
        payload["persisted_at"] = utc_now_iso()
        write_json(path, payload)
        return stored.model_copy(update={"comparison_path": stored.comparison_path})

    def evidence_path(self, evidence_id: str) -> Path:
        record = self.require(evidence_id)
        return self.outputs_root / record.project_id / "evidence" / f"{evidence_id}.json"

    def claim_matrix_path(self, project_id: str) -> Path:
        return self.outputs_root / project_id / "evidence" / "claim_support_matrix.json"

    def build_claim_matrix(
        self,
        project_id: str,
        *,
        protocol_id: str | None = None,
    ) -> dict[str, Any]:
        project = (project_id or "").strip()
        if not project:
            raise ValueError("project_id required")
        records = self.list_evidence(project_id=project)
        matrix = build_claim_support_matrix(
            project_id=project,
            records=records,
            protocol_id=protocol_id,
        )
        path = self.claim_matrix_path(project)
        path.parent.mkdir(parents=True, exist_ok=True)
        matrix = matrix.model_copy(update={"matrix_path": str(path)})
        stored = self._repo.upsert_claim_matrix(matrix)
        export = matrix_export_dict(stored)
        export["persisted_at"] = utc_now_iso()
        write_json(path, export)
        return export

    def get_claim_matrix(self, project_id: str) -> ClaimSupportMatrix | None:
        return self._repo.get_claim_matrix(project_id)

    def show_claim_matrix(self, project_id: str) -> dict[str, Any]:
        project = (project_id or "").strip()
        if not project:
            raise ValueError("project_id required")
        matrix = self.get_claim_matrix(project)
        path = self.claim_matrix_path(project)
        if matrix is None and path.is_file():
            import json

            data = json.loads(path.read_text(encoding="utf-8"))
            data["matrix_path"] = str(path)
            data["recovered_from"] = "filesystem"
            return data
        if matrix is None:
            raise KeyError(f"claim support matrix not found for project: {project}")
        export = matrix_export_dict(matrix)
        export["matrix_path"] = str(path) if path.exists() else matrix.matrix_path
        export["recovered_from"] = "sqlite"
        return export

    def build_from_node_group_comparison(
        self,
        comparison: dict[str, Any],
        *,
        project_id: str | None = None,
        sample_contract: dict[str, Any] | None = None,
        artifact_resolver: Callable[[str], list[str]] | None = None,
        include_resource_evidence: bool = True,
        has_ablation: bool = False,
        formal_implementation: bool = False,
        comparison_path: str | None = None,
    ) -> dict[str, Any]:
        resolved_project = project_id
        if not resolved_project:
            for key in ("candidate_aggregate", "baseline_aggregate"):
                value = (comparison.get(key) or {}).get("project_id")
                if value:
                    resolved_project = str(value)
                    break
        if not resolved_project and sample_contract:
            resolved_project = str(sample_contract.get("project_id") or "")
        if not resolved_project:
            raise ValueError("project_id required to build evidence")

        execution_ids: list[str] = []
        for item in comparison.get("paired_deltas") or []:
            for key in ("baseline_execution_id", "candidate_execution_id"):
                value = item.get(key)
                if value and value not in execution_ids:
                    execution_ids.append(str(value))

        artifact_ids: list[str] = []
        if artifact_resolver is not None:
            for execution_id in execution_ids:
                for artifact_id in artifact_resolver(execution_id):
                    if artifact_id not in artifact_ids:
                        artifact_ids.append(artifact_id)

        protocol_id = None
        if sample_contract:
            protocol_id = sample_contract.get("protocol_id")
        if not protocol_id:
            protocol_id = comparison.get("protocol_id")

        paired = build_paired_comparison_evidence(
            comparison,
            project_id=resolved_project,
            source_artifact_ids=artifact_ids,
            protocol_id=str(protocol_id) if protocol_id else None,
            sample_contract=sample_contract,
            has_ablation=has_ablation,
            formal_implementation=formal_implementation,
            comparison_path=comparison_path,
        )
        paired = self.persist(paired)
        records = [paired]

        if include_resource_evidence:
            resource = build_resource_comparison_evidence(
                comparison,
                project_id=resolved_project,
                paired_evidence_id=paired.evidence_id,
                source_artifact_ids=artifact_ids,
                protocol_id=str(protocol_id) if protocol_id else None,
                sample_contract=sample_contract,
            )
            resource = self.persist(resource)
            records.append(resource)

        return {
            "project_id": resolved_project,
            "evidence_ids": [item.evidence_id for item in records],
            "records": [item.model_dump(mode="json") for item in records],
            "evidence_dir": str(self.outputs_root / resolved_project / "evidence"),
        }
