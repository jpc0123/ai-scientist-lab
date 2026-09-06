"""Tree expansion helpers: plan-next under Best-First parent (v1.1.4)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from scientist_lab.search.models import ExperimentTree, TreeNode


def _parameter_fingerprint(changes: dict[str, Any]) -> str:
    payload = json.dumps(changes or {}, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def remaining_candidate_slots(
    tree: ExperimentTree,
    parent: TreeNode,
    *,
    child_count: int,
    node_count: int,
) -> int:
    """How many new candidates may be proposed from this parent this round."""
    by_children = max(0, tree.max_children_per_node - int(child_count))
    by_nodes = max(0, tree.max_nodes - int(node_count))
    return max(0, min(3, by_children, by_nodes))


def tree_parameter_fingerprints(
    experiment_nodes: list[Any],
    *,
    allowed_keys: list[str] | None = None,
) -> set[str]:
    """Fingerprints of tunable parameters already present on tree ExperimentNodes."""
    fingerprints: set[str] = set()
    keys = list(allowed_keys or [])
    for node in experiment_nodes:
        contract = getattr(node, "contract_json", None)
        if contract is None and isinstance(node, dict):
            contract = node.get("contract_json")
        params: dict[str, Any] = {}
        if isinstance(contract, dict):
            params = dict(contract.get("parameters") or {})
        if not params:
            continue
        if keys:
            subset = {k: params[k] for k in keys if k in params}
        else:
            subset = params
        if subset:
            fingerprints.add(_parameter_fingerprint(subset))
    return fingerprints


def annotate_candidates_against_tree(
    candidates: list[dict[str, Any]],
    *,
    parent_experiment_node_id: str,
    existing_fingerprints: set[str],
    max_candidates: int,
) -> dict[str, Any]:
    """Keep parent-matched, non-duplicate candidates; cap to max_candidates."""
    kept: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for raw in candidates:
        item = dict(raw)
        parent_id = str(item.get("parent_node_id") or "")
        changes = dict(item.get("parameter_changes") or {})
        status = str(item.get("status") or "")

        if parent_id and parent_id != parent_experiment_node_id:
            item["tree_filter"] = "parent_mismatch"
            rejected.append(item)
            continue
        if status in {"rejected"}:
            item["tree_filter"] = "verifier_rejected"
            rejected.append(item)
            continue

        fp = _parameter_fingerprint(changes) if changes else ""
        if fp and fp in existing_fingerprints:
            item["tree_filter"] = "duplicate_configuration"
            rejected.append(item)
            continue

        if len(kept) < max_candidates:
            item["tree_filter"] = "accepted"
            kept.append(item)
        else:
            item["tree_filter"] = "deferred_over_limit"
            deferred.append(item)

    return {
        "accepted": kept,
        "deferred": deferred,
        "rejected": rejected,
        "accepted_count": len(kept),
        "max_candidates": max_candidates,
    }
