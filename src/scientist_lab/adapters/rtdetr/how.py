"""RT-DETR Adapter HOW: same F0/F1/F3 catalog, native decoder family.

Does not stuff D-FINE FDR / FDPN into the second detector.
Fusion HOW (early_concat staging, gated_multiscale dual-stream wrap) is the
transferable strategy; the detector is RT-DETRTransformer, not DFINETransformer.
"""

from __future__ import annotations

from typing import Any, Mapping

from scientist_lab.adapters.base import MaterializeRejected
from scientist_lab.adapters.dfine.how import list_adapter_capabilities as list_dfine_capabilities
from scientist_lab.adapters.dfine.how import resolve_adapter_how as resolve_dfine_how

ADAPTER_KEY = "rtdetr"
BASELINE_KEY = "rtdetr_s"
BACKEND_KEY = "rtdetr"


def list_adapter_capabilities() -> list[dict[str, Any]]:
    rows = []
    for row in list_dfine_capabilities():
        item = dict(row)
        item["adapter"] = ADAPTER_KEY
        rows.append(item)
    return rows


def resolve_adapter_how(
    plan: Mapping[str, Any],
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    body = resolve_dfine_how(plan, protocol)
    how_id = str(body.get("how_id") or "").strip().upper()
    if how_id == "N1" or str(body.get("neck_type") or "") == "fdpn":
        raise MaterializeRejected(
            "RT-DETR Adapter does not implement N1/FDPN; "
            "do not stuff the D-FINE neck into the second detector"
        )
    legacy = dict(body.get("legacy_parameters") or {})
    legacy["baseline"] = BASELINE_KEY
    legacy["dfine_backend"] = BACKEND_KEY
    legacy["detector"] = ADAPTER_KEY
    # D-FINE solver + RT-DETR CDN losses NaN under AMP fp16 on this 160 budget.
    # Adapter glue, not a new HOW. D-FINE Adapter keeps mixed_precision.
    legacy["mixed_precision"] = False
    body["legacy_parameters"] = legacy
    body["detector"] = ADAPTER_KEY
    body["decoder_family"] = "RTDETRTransformer"
    return body
