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
    slice_spec = dict(protocol.get("condition_slice") or {})
    if slice_spec.get("id") and not dataset.get("split_reference"):
        rule = str(slice_spec.get("rule_hash") or "")
        dataset = {
            **dataset,
            "split_reference": f"{slice_spec['id']}@{rule}" if rule else str(slice_spec["id"]),
            "version": dataset.get("version") or slice_spec.get("version"),
        }
    frozen = sorted(protocol.get("frozen_scope") or [])
    objective = protocol.get("objective") or {}
    metrics_spec = (contract or {}).get("metrics_spec") or {
        "primary": (objective.get("primary") or {}).get("metric"),
        "secondary": [s.get("metric") for s in (objective.get("secondary") or [])],
    }
    split_identity = {
        "reference": dataset.get("reference"),
        "split_reference": dataset.get("split_reference"),
    }
    if slice_spec:
        split_identity["condition_slice"] = {
            "id": slice_spec.get("id"),
            "rule_hash": slice_spec.get("rule_hash"),
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
        "dataset_split_hash": _stable_hash(split_identity),
        "evaluator_hash": _stable_hash({"frozen_scope": frozen, "evaluator": "protocol_frozen"}),
        "metric_definition_hash": _stable_hash({"objective": objective, "metrics_spec": metrics_spec}),
        "baseline_config_hash": _stable_hash(protocol.get("baseline") or {}),
        "annotation_version": str(dataset.get("version") or dataset.get("reference") or "unspecified"),
        "data_manifest_hash": _stable_hash(
            {
                "reference": dataset.get("reference"),
                "split_reference": dataset.get("split_reference"),
                "slice_id": slice_spec.get("id"),
                "rule_hash": slice_spec.get("rule_hash"),
            }
            if slice_spec
            else dataset
        ),
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
        f"neck_type={how.get('neck_type')}; "
        f"plugin_kind={how.get('plugin_kind')}; "
        f"backbone_wrap={how.get('backbone_wrap_method')}. "
        "Frozen hashes exclude fusion/neck HOW (comparability contract). "
        "backbone_wrap plugins require a rebound architecture_id / new R0."
    )


def fingerprints_equivalent(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Compare semantic hashes only (id/notes may differ)."""
    return all(left.get(k) == right.get(k) for k in _HASH_FIELDS)
