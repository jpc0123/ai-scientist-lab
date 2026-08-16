"""Comparability Contract fingerprint — Adapter maps D-FINE onto this.

Not a unified harness. Gate compares hashes; Adapter only computes.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from scientist_lab.core.schema_registry import validate_named

_HASH_FIELDS = (
    "dataset_split_hash",
    "evaluator_hash",
    "metric_definition_hash",
    "baseline_config_hash",
    "data_manifest_hash",
)


def _stable_hash(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def compute_fingerprint(
    protocol: Mapping[str, Any],
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    dataset = dict((contract or {}).get("dataset") or {})
    if not dataset:
        baseline = protocol.get("baseline") or {}
        if baseline.get("dataset"):
            dataset = {"reference": baseline["dataset"]}
    frozen = sorted(protocol.get("frozen_scope") or [])
    objective = protocol.get("objective") or {}
    metrics_spec = (contract or {}).get("metrics_spec") or {
        "primary": (objective.get("primary") or {}).get("metric"),
        "secondary": [s.get("metric") for s in (objective.get("secondary") or [])],
    }
    fingerprint = {
        "schema_version": "1.0.0",
        "fingerprint_id": str(
            protocol.get("fingerprint_id")
            or (contract or {}).get("frozen_fingerprint_id")
            or "FP-UNBOUND"
        ),
        "protocol_id": protocol["protocol_id"],
        "protocol_version": int(protocol["protocol_version"]),
        "dataset_split_hash": _stable_hash(
            {
                "reference": dataset.get("reference"),
                "version": dataset.get("version"),
                "split_reference": dataset.get("split_reference"),
            }
        ),
        "evaluator_hash": _stable_hash({"frozen_scope": frozen, "evaluator": "protocol_frozen"}),
        "metric_definition_hash": _stable_hash({"objective": objective, "metrics_spec": metrics_spec}),
        "baseline_config_hash": _stable_hash(protocol.get("baseline") or {}),
        "annotation_version": str(dataset.get("version") or dataset.get("reference") or "unspecified"),
        "data_manifest_hash": _stable_hash(dataset),
        "notes": _how_notes(contract),
    }
    validate_named("frozen_fingerprint", fingerprint)
    return fingerprint


def _how_notes(contract: Mapping[str, Any] | None) -> str:
    """HOW identity lives in notes (not Frozen hash fields) so Gate stays comparable."""
    mat = dict((contract or {}).get("materialization") or {})
    how = dict(mat.get("how") or {})
    signature = str(mat.get("how_signature") or how.get("signature") or "")
    if not signature:
        return (
            "Computed from Protocol + Contract identity; Adapter does not import a global harness."
        )
    return (
        f"HOW signature={signature}; "
        f"module={how.get('primary_module')}; "
        f"input_mode={how.get('input_mode')}; "
        f"fusion_method={how.get('fusion_method')}; "
        f"neck_type={how.get('neck_type')}. "
        "Frozen hashes exclude HOW (comparability contract)."
    )


def fingerprints_equivalent(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Compare semantic hashes only (id/notes may differ)."""
    return all(left.get(k) == right.get(k) for k in _HASH_FIELDS)
