from __future__ import annotations

import json
from typing import Any

from scientist_lab.ablations.models import AblationPlan, AblationVerificationReport
from scientist_lab.protocols.models import ExperimentProtocol


class AblationVerifier:
    def validate_plan(
        self,
        plan: AblationPlan,
        *,
        protocol: ExperimentProtocol | None = None,
    ) -> AblationVerificationReport:
        warnings: list[str] = []
        blocking: list[str] = []
        deduped: list[str] = []

        controlled = set(plan.controlled_variables)
        if not controlled:
            blocking.append("controlled_variables must be non-empty")

        seen_ids: set[str] = set()
        signature_to_ids: dict[str, list[str]] = {}

        for variant in plan.variants:
            if variant.variant_id in seen_ids:
                blocking.append(f"duplicate variant_id: {variant.variant_id}")
            seen_ids.add(variant.variant_id)

            changes = dict(variant.parameter_changes or {})
            if not changes:
                warnings.append(f"{variant.variant_id}: empty parameter_changes")

            extra = sorted(set(changes) - controlled)
            if extra:
                blocking.append(
                    f"{variant.variant_id}: parameter_changes keys not in "
                    f"controlled_variables: {', '.join(extra)}"
                )

            signature = json.dumps(changes, sort_keys=True, default=str)
            signature_to_ids.setdefault(signature, []).append(variant.variant_id)

        for ids in signature_to_ids.values():
            if len(ids) > 1:
                # Keep first; mark the rest as duplicates to drop.
                deduped.extend(ids[1:])
                warnings.append(
                    "duplicate parameter_changes across variants "
                    f"{ids}; will keep {ids[0]} and drop {', '.join(ids[1:])}"
                )

        if protocol is not None:
            if plan.protocol_id != protocol.protocol_id:
                blocking.append(
                    f"protocol_id mismatch: plan={plan.protocol_id}, "
                    f"protocol={protocol.protocol_id}"
                )
            if plan.project_id != protocol.project_id:
                blocking.append(
                    f"project_id mismatch: plan={plan.project_id}, "
                    f"protocol={protocol.project_id}"
                )
            allowed = set(protocol.allowed_variables or [])
            if allowed and not controlled.issubset(allowed):
                blocking.append(
                    "controlled_variables must be subset of protocol.allowed_variables: "
                    + ", ".join(sorted(controlled - allowed))
                )

        return AblationVerificationReport(
            valid=len(blocking) == 0,
            warnings=warnings,
            blocking_issues=blocking,
            ablation_id=plan.ablation_id,
            deduplicated_variant_ids=deduped,
        )

    def deduplicate_plan(self, plan: AblationPlan) -> AblationPlan:
        """Drop later variants that share identical parameter_changes."""
        report = self.validate_plan(plan)
        drop = set(report.deduplicated_variant_ids)
        if not drop:
            return plan
        kept = [v for v in plan.variants if v.variant_id not in drop]
        return plan.model_copy(update={"variants": kept})
