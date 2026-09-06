from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.orm import sessionmaker

from scientist_lab.ablations.models import AblationPlan, AblationVerificationReport
from scientist_lab.ablations.planner import plan_variant_contracts
from scientist_lab.ablations.repository import AblationRepository
from scientist_lab.ablations.verifier import AblationVerifier
from scientist_lab.domain.contracts import ExperimentContract
from scientist_lab.protocols.models import ExperimentProtocol
from scientist_lab.storage.artifact_store import write_json


class AblationService:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._repo = AblationRepository(session_factory)
        self.verifier = AblationVerifier()

    def create_from_path(self, path: Path) -> AblationPlan:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return self.create_from_dict(data)

    def create_from_dict(
        self,
        data: dict[str, Any],
        *,
        protocol: ExperimentProtocol | None = None,
        deduplicate: bool = True,
    ) -> AblationPlan:
        payload = dict(data)
        if "created_at" not in payload or payload.get("created_at") in (None, ""):
            payload["created_at"] = datetime.now(timezone.utc).replace(microsecond=0)
        plan = AblationPlan.model_validate(payload)
        report = self.verifier.validate_plan(plan, protocol=protocol)
        if not report.valid:
            raise ValueError(
                "invalid ablation plan: " + "; ".join(report.blocking_issues)
            )
        if deduplicate:
            plan = self.verifier.deduplicate_plan(plan)
        return self._repo.upsert(plan)

    def get(self, ablation_id: str) -> AblationPlan | None:
        return self._repo.get(ablation_id)

    def require(self, ablation_id: str) -> AblationPlan:
        plan = self.get(ablation_id)
        if plan is None:
            raise KeyError(f"ablation plan not found: {ablation_id}")
        return plan

    def list_plans(
        self,
        *,
        project_id: str | None = None,
        protocol_id: str | None = None,
    ) -> list[AblationPlan]:
        return self._repo.list_plans(project_id=project_id, protocol_id=protocol_id)

    def validate(
        self,
        ablation_id: str,
        *,
        protocol: ExperimentProtocol | None = None,
    ) -> AblationVerificationReport:
        plan = self.require(ablation_id)
        return self.verifier.validate_plan(plan, protocol=protocol)

    def materialize(
        self,
        ablation_id: str,
        reference: ExperimentContract,
        *,
        output_dir: Path | None = None,
        deduplicate: bool = True,
    ) -> dict[str, Any]:
        plan = self.require(ablation_id)
        items = plan_variant_contracts(
            plan, reference, deduplicate=deduplicate
        )
        written: list[str] = []
        if output_dir is not None:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            for item in items:
                path = out / f"{item['variant_id']}_{item['node_id']}_contract.json"
                write_json(path, item["contract"])
                written.append(str(path))
                item["contract_path"] = str(path)
        return {
            "ablation_id": plan.ablation_id,
            "reference_node_id": plan.reference_node_id,
            "protocol_id": plan.protocol_id,
            "variant_count": len(items),
            "variants": items,
            "written_paths": written,
        }
