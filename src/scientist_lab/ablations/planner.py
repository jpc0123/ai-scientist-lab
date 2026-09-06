"""Generate protocol-compliant contracts from an AblationPlan + reference contract."""

from __future__ import annotations

from typing import Any

from scientist_lab.ablations.models import AblationPlan, AblationVariant
from scientist_lab.ablations.verifier import AblationVerifier
from scientist_lab.domain.contracts import ExperimentContract


# First-version modality ablation → formal triad node ids when changes match.
DEFAULT_NODE_ID_BY_SIGNATURE = {
    '{"fusion_method": "none", "input_mode": "rgb"}': "rgbt_formal_node_001",
    '{"fusion_method": "none", "input_mode": "thermal"}': "rgbt_formal_node_002",
    '{"fusion_method": "early_concat", "input_mode": "rgbt"}': "rgbt_formal_node_003",
}


def _signature(changes: dict[str, Any]) -> str:
    import json

    return json.dumps(changes, sort_keys=True, default=str)


def resolve_variant_node_id(
    plan: AblationPlan,
    variant: AblationVariant,
) -> str:
    if variant.node_id:
        return variant.node_id
    mapped = DEFAULT_NODE_ID_BY_SIGNATURE.get(_signature(dict(variant.parameter_changes)))
    if mapped:
        return mapped
    return f"{plan.reference_node_id}_ablation_{variant.variant_id}"


def materialize_variant_contract(
    *,
    plan: AblationPlan,
    variant: AblationVariant,
    reference: ExperimentContract,
) -> ExperimentContract:
    params = dict(reference.parameters or {})
    params.update(dict(variant.parameter_changes or {}))
    node_id = resolve_variant_node_id(plan, variant)
    title = variant.title or f"{plan.ablation_id}:{variant.variant_id}"
    hypothesis = (
        f"Ablation {variant.variant_id}: {variant.expected_effect}".strip()
        if variant.expected_effect
        else f"Ablation variant {variant.variant_id} under {plan.ablation_id}"
    )
    return reference.model_copy(
        update={
            "project_id": plan.project_id,
            "node_id": node_id,
            "parent_node_id": plan.reference_node_id,
            "protocol_id": plan.protocol_id,
            "title": title,
            "hypothesis": hypothesis,
            "parameters": params,
        }
    )


def plan_variant_contracts(
    plan: AblationPlan,
    reference: ExperimentContract,
    *,
    deduplicate: bool = True,
) -> list[dict[str, Any]]:
    """Return materialized contracts (and metadata) for each kept variant."""
    verifier = AblationVerifier()
    working = verifier.deduplicate_plan(plan) if deduplicate else plan
    items: list[dict[str, Any]] = []
    for variant in working.variants:
        contract = materialize_variant_contract(
            plan=working, variant=variant, reference=reference
        )
        items.append(
            {
                "variant_id": variant.variant_id,
                "title": variant.title,
                "expected_effect": variant.expected_effect,
                "node_id": contract.node_id,
                "parameter_changes": dict(variant.parameter_changes or {}),
                "contract": contract.model_dump(mode="json"),
            }
        )
    return items


def default_modality_ablation_plan(
    *,
    project_id: str = "project_rgbt_003",
    protocol_id: str = "protocol_rgbt_001",
    reference_node_id: str = "rgbt_formal_node_003",
) -> AblationPlan:
    """A0 RGB / A1 Thermal / A2 Early Fusion (v0.9.4 first version)."""
    from datetime import datetime, timezone

    from scientist_lab.ablations.models import AblationVariant

    return AblationPlan(
        ablation_id="ablation_rgbt_modality_001",
        project_id=project_id,
        reference_node_id=reference_node_id,
        protocol_id=protocol_id,
        title="RGB-T modality ablation (A0/A1/A2)",
        controlled_variables=["input_mode", "fusion_method"],
        variants=[
            AblationVariant(
                variant_id="A0",
                title="RGB-only",
                parameter_changes={"input_mode": "rgb", "fusion_method": "none"},
                expected_effect="Single-modality RGB baseline.",
                node_id="rgbt_formal_node_001",
            ),
            AblationVariant(
                variant_id="A1",
                title="Thermal-only",
                parameter_changes={
                    "input_mode": "thermal",
                    "fusion_method": "none",
                },
                expected_effect="Single-modality thermal baseline.",
                node_id="rgbt_formal_node_002",
            ),
            AblationVariant(
                variant_id="A2",
                title="Early Fusion",
                parameter_changes={
                    "input_mode": "rgbt",
                    "fusion_method": "early_concat",
                },
                expected_effect="Early-concat RGB-T fusion candidate.",
                node_id="rgbt_formal_node_003",
            ),
        ],
        created_at=datetime.now(timezone.utc).replace(microsecond=0),
    )
