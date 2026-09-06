"""v2.6 trusted HOW catalog. Unregistered ids cannot be materialized."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from scientist_lab.adapters.base import MaterializeRejected

PLUGIN_RELPATH_PREFIX = "experiment_apps/rgbt_detection_real/models/how_plugins/"
PLUGIN_KINDS = ("fusion", "neck", "backbone_wrap")
DEFAULT_INVENT_KINDS = ("fusion",)
PLUGIN_AUTHORABLE_FAMILIES = frozenset(PLUGIN_KINDS)

CATALOG_ID = "how_catalog_v26_p2"

# Already implemented on the D-FINE train path. F2/T1/T2 stay off-catalog.
ALLOWED_HOW: dict[str, dict[str, Any]] = {
    "F0": {
        "id": "F0",
        "family": "fusion",
        "primary_module": "fusion",
        "input_mode": "rgb",
        "fusion_method": "none",
        "neck_type": "standard",
        "existing_capability": "rgb_only",
    },
    "F1": {
        "id": "F1",
        "family": "fusion",
        "primary_module": "fusion",
        "input_mode": "rgbt",
        "fusion_method": "early_concat",
        "neck_type": "standard",
        "existing_capability": "early_concat_blend",
    },
    "F3": {
        "id": "F3",
        "family": "fusion",
        "primary_module": "fusion",
        "input_mode": "rgbt",
        "fusion_method": "gated_multiscale",
        "neck_type": "standard",
        "existing_capability": "gated_multiscale",
        "cost_note": "heavier dual-stream path",
    },
    "N0": {
        "id": "N0",
        "family": "neck",
        "primary_module": "neck",
        "input_mode": "rgb",
        "fusion_method": "none",
        "neck_type": "standard",
        "existing_capability": "rgb_hybrid_encoder",
    },
    "N1": {
        "id": "N1",
        "family": "neck",
        "primary_module": "neck",
        "input_mode": "rgb",
        "fusion_method": "none",
        "neck_type": "fdpn",
        "existing_capability": "fdpn_neck",
        "cost_note": "existing fdpn neck switch; not a newly invented operator",
    },
    "A4": {
        "id": "A4",
        "family": "fusion",
        "primary_module": "fusion",
        "input_mode": "rgbt",
        "fusion_method": "early_concat",
        "neck_type": "fdpn",
        "existing_capability": "early_concat_fdpn",
        "cost_note": (
            "Composes existing F1 fusion + N1 neck knobs. "
            "Not a freeze answer key and not a new train operator."
        ),
    },
}

NOT_REGISTERED = frozenset({"F2", "T0", "T1", "T2", "weighted_fusion", "mid_fusion", "late_fusion"})

# Live Planner sees every materializable catalog id. Empty on purpose:
# N1/A4 used to be hidden; that lock caused idle F0/F1/F3 loops.
HIDDEN_FROM_PLANNER: dict[str, str] = {}


def normalize_plugin_kind(raw: Any) -> str:
    token = str(raw or "fusion").strip().lower()
    return token if token in PLUGIN_KINDS else "fusion"


def candidate_plugin_kind(row: Mapping[str, Any] | None) -> str:
    blob = dict(row or {})
    explicit = str(blob.get("plugin_kind") or "").strip().lower()
    if explicit in PLUGIN_KINDS:
        return explicit
    family = str(blob.get("family") or "").strip().lower()
    if family in PLUGIN_KINDS:
        return family
    if str(blob.get("neck_type") or "").lower().startswith("plugin:"):
        return "neck"
    if str(blob.get("backbone_wrap_method") or "").lower().startswith("plugin:"):
        return "backbone_wrap"
    return "fusion"


def protocol_invent_kinds(protocol: Mapping[str, Any] | None) -> frozenset[str]:
    """Campaign invent slots. Missing invent_policy defaults to fusion-only."""
    policy = dict((protocol or {}).get("invent_policy") or {})
    kinds = policy.get("kinds")
    if not kinds:
        return frozenset(DEFAULT_INVENT_KINDS)
    out = {normalize_plugin_kind(item) for item in kinds}
    return out or frozenset(DEFAULT_INVENT_KINDS)


def family_needs_adapter_work(family: str) -> bool:
    return str(family or "").strip().lower() not in PLUGIN_AUTHORABLE_FAMILIES


def plugin_overlay_spec(
    how_id: str,
    *,
    smoke_ok: bool = False,
    plugin_kind: str = "fusion",
) -> dict[str, Any]:
    token = str(how_id or "").strip().upper()
    if token.startswith("PLUGIN:"):
        token = token.split(":", 1)[1].strip().upper()
    rel = f"{PLUGIN_RELPATH_PREFIX}{token}/plugin.py"
    kind = normalize_plugin_kind(plugin_kind)
    shared = {
        "id": token,
        "plugin_how_id": token,
        "plugin_kind": kind,
        "plugin_relpath": rel,
        "smoke_ok": bool(smoke_ok),
    }
    if kind == "neck":
        return {
            **shared,
            "family": "neck",
            "primary_module": "neck",
            "input_mode": "rgb",
            "fusion_method": "none",
            "neck_type": f"plugin:{token}",
            "existing_capability": "how_plugin_neck",
            "requires_new_baseline": False,
            "cost_note": (
                "HOW plugin neck/encoder in the D-FINE HybridEncoder slot. "
                "Comparable to N0/N1 on the same frozen protocol. "
                "Not a Claim. Not vendor D-FINE."
            ),
        }
    if kind == "backbone_wrap":
        return {
            **shared,
            "family": "backbone_wrap",
            "primary_module": "backbone",
            "input_mode": "rgb",
            "fusion_method": "none",
            "neck_type": "standard",
            "backbone_wrap_method": f"plugin:{token}",
            "existing_capability": "how_plugin_backbone_wrap",
            "requires_new_baseline": True,
            "cost_note": (
                "HOW plugin backbone wrap. Requires protocol.baseline.architecture_id "
                "and a new R0; do not KEEP against the old fusion/neck baseline. "
                "Not a Claim. Not vendor D-FINE."
            ),
        }
    return {
        **shared,
        "family": "fusion",
        "primary_module": "fusion",
        "input_mode": "rgbt",
        "fusion_method": f"plugin:{token}",
        "neck_type": "standard",
        "existing_capability": "how_plugin",
        "requires_new_baseline": False,
        "cost_note": "HOW plugin FeatureFusion. Not a Claim. Not vendor D-FINE.",
    }


def overlay_is_plugin(spec: Mapping[str, Any] | None) -> bool:
    blob = dict(spec or {})
    if blob.get("plugin_how_id") or str(blob.get("existing_capability") or "").startswith(
        "how_plugin"
    ):
        return True
    if str(blob.get("fusion_method") or "").lower().startswith("plugin:"):
        return True
    if str(blob.get("neck_type") or "").lower().startswith("plugin:"):
        return True
    return str(blob.get("backbone_wrap_method") or "").lower().startswith("plugin:")


def catalog_payload() -> dict[str, Any]:
    return {
        "catalog_id": CATALOG_ID,
        "adapter": "dfine",
        "allowed": [dict(row) for row in ALLOWED_HOW.values()],
        "not_registered": sorted(NOT_REGISTERED),
        "hidden_from_planner": dict(HIDDEN_FROM_PLANNER),
        "llm_may_invent_how": False,
    }


def default_catalog_path() -> Path:
    return (
        Path(__file__).resolve().parents[4]
        / "docs"
        / "research"
        / "v26"
        / "HOW_CATALOG_V26.json"
    )


def load_catalog(path: Path | str | None = None) -> dict[str, Any]:
    dest = Path(path) if path is not None else default_catalog_path()
    if dest.is_file():
        return json.loads(dest.read_text(encoding="utf-8"))
    return catalog_payload()


def materializable_how_ids(
    overlay: Mapping[str, Mapping[str, Any]] | None = None,
) -> frozenset[str]:
    """HOW ids Adapter can materialize today. Frozen catalog plus smoked overlay."""
    ids = set(ALLOWED_HOW)
    for key, spec in dict(overlay or {}).items():
        token = str(key or "").strip().upper()
        if not token:
            continue
        blob = dict(spec or {})
        if overlay_is_plugin(blob) and blob.get("smoke_ok"):
            ids.add(token)
        elif token not in NOT_REGISTERED:
            ids.add(token)
    return frozenset(ids)


def planner_visible_how_ids(
    overlay: Mapping[str, Mapping[str, Any]] | None = None,
) -> frozenset[str]:
    """Live Planner may select catalog ids and campaign overlay plugins."""
    return materializable_how_ids(overlay)


PLANNER_VISIBLE_HOW = materializable_how_ids()
ADAPTER_FUSION_METHODS = frozenset(
    str(row["fusion_method"]) for row in ALLOWED_HOW.values()
)
ADAPTER_NECK_TYPES = frozenset(str(row["neck_type"]) for row in ALLOWED_HOW.values())


def adapter_can_map_candidate(row: Mapping[str, Any]) -> bool:
    """True when Adapter has knobs, or a smoked HOW plugin is ready to overlay."""
    how_id = str(row.get("how_id") or "").strip().upper()
    if not how_id or how_id in HIDDEN_FROM_PLANNER:
        return False
    if overlay_is_plugin(row) and bool(row.get("smoke_ok")):
        return True
    if how_id in NOT_REGISTERED:
        return False
    mapped = str(row.get("map_to_existing") or "").strip().upper()
    if mapped in ALLOWED_HOW:
        return how_id == mapped
    if how_id in ALLOWED_HOW:
        return True
    fusion = str(row.get("fusion_method") or "").strip()
    neck = str(row.get("neck_type") or "standard").strip()
    return fusion in ADAPTER_FUSION_METHODS and neck in ADAPTER_NECK_TYPES


def resolve_how_id(
    how_id: str,
    *,
    overlay: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    token = str(how_id or "").strip().upper()
    extra = dict(overlay or {})
    if token in extra:
        spec = dict(extra[token])
        spec.setdefault("id", token)
        if overlay_is_plugin(spec) and spec.get("smoke_ok"):
            return spec
        if token not in NOT_REGISTERED and token not in HIDDEN_FROM_PLANNER:
            return spec
    if token in NOT_REGISTERED or token not in ALLOWED_HOW:
        raise MaterializeRejected(
            f"HOW {token or how_id!r} is not in the v2.6 trusted catalog "
            f"(allowed={sorted(ALLOWED_HOW)}; not_registered={sorted(NOT_REGISTERED)})"
        )
    return dict(ALLOWED_HOW[token])


def plan_how_id(plan: Mapping[str, Any]) -> str | None:
    for key in ("how_id", "catalog_how_id"):
        value = plan.get(key)
        if value:
            return str(value)
    for row in plan.get("proposed_changes") or []:
        if not isinstance(row, Mapping):
            continue
        detail = dict(row.get("detail") or {})
        for key in ("how_id", "catalog_how_id"):
            if detail.get(key):
                return str(detail[key])
    return None
