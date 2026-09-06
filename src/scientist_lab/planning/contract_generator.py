"""Generate ExperimentContract drafts from approved candidates only."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from scientist_lab.agents.models import BLOCKED_PARAMETER_KEYS, CandidateExperiment
from scientist_lab.domain.contracts import ExperimentContract
from scientist_lab.storage.artifact_store import write_json


def _next_node_id(parent_node_id: str, existing_ids: set[str]) -> str:
    match = re.fullmatch(r"([A-Za-z0-9_]+_node_)(\d+)", parent_node_id)
    if match:
        width = max(3, len(match.group(2)))
        number = int(match.group(2)) + 1
        while True:
            candidate = f"{match.group(1)}{number:0{width}d}"
            if candidate not in existing_ids:
                return candidate
            number += 1
    suffix = 1
    while True:
        candidate = f"{parent_node_id}_plan_{suffix}"
        if candidate not in existing_ids:
            return candidate
        suffix += 1


def generate_contract_from_candidate(
    *,
    parent_contract: dict[str, Any] | ExperimentContract,
    candidate: CandidateExperiment,
    plan_id: str,
    existing_node_ids: set[str],
    output_dir: Path | None = None,
) -> dict[str, Any]:
    if isinstance(parent_contract, ExperimentContract):
        base = parent_contract.model_dump(mode="json")
    else:
        base = dict(parent_contract)

    changes = dict(candidate.parameter_changes or {})
    blocked = sorted(set(changes) & BLOCKED_PARAMETER_KEYS)
    if blocked:
        raise ValueError(f"refusing blocked parameter keys: {blocked}")

    parameters = dict(base.get("parameters") or {})
    parameters.update(changes)

    node_id = _next_node_id(candidate.parent_node_id, existing_node_ids)
    task_config = dict(base.get("task_config") or {})
    task_config["plan_id"] = plan_id
    task_config["candidate_id"] = candidate.candidate_id
    task_config.setdefault("claim_level", task_config.get("claim_level") or "exploratory_comparison")

    contract = dict(base)
    contract.update(
        {
            "node_id": node_id,
            "parent_node_id": candidate.parent_node_id,
            "title": candidate.title,
            "hypothesis": candidate.hypothesis,
            "parameters": parameters,
            "task_config": task_config,
        }
    )
    # Never allow candidate to rewrite safety fields via parent overwrite mistakes.
    for key in (
        "environment_key",
        "dataset_reference",
        "code_reference",
        "entrypoint",
        "protocol_id",
    ):
        if key in base:
            contract[key] = base[key]

    validated = ExperimentContract.model_validate(contract)
    payload = validated.model_dump(mode="json")

    written = None
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{candidate.candidate_id}_{node_id}_contract.json"
        write_json(path, payload)
        written = str(path)

    return {
        "contract": payload,
        "contract_path": written,
        "plan_id": plan_id,
        "candidate_id": candidate.candidate_id,
        "node_id": node_id,
    }
