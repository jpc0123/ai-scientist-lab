"""Map Planner modification_scope onto existing DFINE execution knobs.

Adapter HOW only. Does not invent hypotheses, FDPN, or new fusion operators.
Existing capabilities live in experiment_apps/rgbt_detection_real
(fusion_names.IMPLEMENTED_FUSION_METHODS, neck_factory.NeckConfig).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

# Existing fusion_method values the CUDA train path already accepts.
_EXISTING_FUSION_METHODS = frozenset(
    {"none", "early_concat", "concat", "early", "gated_multiscale", "full_fusion"}
)
_EXISTING_NECK_TYPES = frozenset({"standard", "fdpn"})

# Probe-safe fusion HOW: in-repo early_concat (pixel blend at staging).
# gated_multiscale also exists but is a heavier dual-stream path; Adapter does
# not invent it and does not select it merely to look like a new network.
_NECK_HOW = {
    "primary_module": "neck",
    "input_mode": "rgb",
    "fusion_method": "none",
    "neck_type": "standard",
    "invented_operators": [],
    "existing_capability": "rgb_hybrid_encoder",
    "gap": (
        "neck.type=fdpn exists in-repo; Adapter does not apply it unless the "
        "Plan names that operator. standard HybridEncoder is the existing neck path."
    ),
}

_FUSION_HOW = {
    "primary_module": "fusion",
    "input_mode": "rgbt",
    "fusion_method": "early_concat",
    "neck_type": "standard",
    "invented_operators": [],
    "existing_capability": "early_concat_blend",
    "gap": (
        "gated_multiscale/full_fusion also exists in-repo; probe HOW uses the "
        "existing early_concat switch, not a newly invented fusion network."
    ),
}

_FALLBACK_HOW = {
    "input_mode": "rgb",
    "fusion_method": "none",
    "neck_type": "standard",
    "invented_operators": [],
    "existing_capability": "rgb_hybrid_encoder",
}

# Modules with a distinct existing HOW mapping. LLM Planner may request only these.
HOW_MODULES = frozenset({"neck", "fusion"})


def list_adapter_capabilities() -> list[dict[str, Any]]:
    """Existing Adapter HOW surface. Not a new operator catalog; no FDPN invention."""
    return [
        {
            "module": "neck",
            "input_mode": _NECK_HOW["input_mode"],
            "fusion_method": _NECK_HOW["fusion_method"],
            "neck_type": _NECK_HOW["neck_type"],
            "existing_capability": _NECK_HOW["existing_capability"],
            "invented_operators": [],
        },
        {
            "module": "fusion",
            "input_mode": _FUSION_HOW["input_mode"],
            "fusion_method": _FUSION_HOW["fusion_method"],
            "neck_type": _FUSION_HOW["neck_type"],
            "existing_capability": _FUSION_HOW["existing_capability"],
            "invented_operators": [],
        },
    ]


# Minimum formal that ClaimGate can accept: full split, non-probe subset,
# enough steps that APS is not a 16/8 untrained observation. Not 640/20ep.
_FORMAL_TRAIN_KNOBS = {
    "epochs": 2,
    "pretrained": True,
    "mixed_precision": True,
    "batch_size": 2,
    "image_width": 160,
    "image_height": 160,
    "learning_rate": 0.0002,
    # 160x160 has 25 coarse tokens; D-FINE default 300 queries cannot fit
    # without scaling. 640 unscaled is many hours on this laptop. Both C1
    # arms use the same scaled-160 HOW so the comparison stays matched.
    "scale_queries_to_tokens": True,
}

FORMAL_TIMEOUT_SECONDS = 14400


def parse_budget_seconds(raw: Any, *, default: int = 1200) -> int:
    """Parse protocol experiment_budget strings like 4h / 60min / 1200s."""
    text = str(raw or "").strip().lower()
    if not text or text == "approval_required":
        return int(default)
    try:
        if text.endswith("min"):
            return max(1, int(float(text[:-3]) * 60))
        if text.endswith("h"):
            return max(1, int(float(text[:-1]) * 3600))
        if text.endswith("s"):
            return max(1, int(float(text[:-1])))
        return max(1, int(float(text)))
    except (TypeError, ValueError):
        return int(default)


def _stable_hash(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def how_identity(how: Mapping[str, Any]) -> dict[str, str]:
    """Fields that distinguish neck HOW vs fusion HOW (not Frozen Fingerprint hashes)."""
    return {
        "primary_module": str(how.get("primary_module") or ""),
        "input_mode": str(how.get("input_mode") or "rgb"),
        "fusion_method": str(how.get("fusion_method") or "none"),
        "neck_type": str(how.get("neck_type") or "standard"),
    }


def resolve_adapter_how(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Translate Plan.modification_scope[0] into existing train/config knobs."""
    scope = [str(tok) for tok in (plan.get("modification_scope") or [])]
    primary = scope[0] if scope else ""
    if primary == "fusion":
        body = dict(_FUSION_HOW)
    elif primary == "neck":
        body = dict(_NECK_HOW)
    else:
        body = dict(_FALLBACK_HOW)
        body["primary_module"] = primary or "unspecified"
        body["gap"] = (
            f"module {primary!r} has no distinct existing fusion/neck knob "
            "mapping; Adapter does not invent an operator"
        )

    identity = how_identity(body)
    if identity["fusion_method"] not in _EXISTING_FUSION_METHODS:
        raise ValueError(f"Adapter HOW selected unknown fusion_method={identity['fusion_method']!r}")
    if identity["neck_type"] not in _EXISTING_NECK_TYPES:
        raise ValueError(f"Adapter HOW selected unknown neck_type={identity['neck_type']!r}")

    signature = _stable_hash(identity)
    body["signature"] = signature
    body["legacy_parameters"] = {
        "input_mode": identity["input_mode"],
        "fusion_method": identity["fusion_method"],
        "neck": {"type": identity["neck_type"]},
    }
    budget = str(plan.get("budget_class") or "probe").strip().lower()
    body["budget_class"] = budget
    if budget == "formal":
        body["legacy_parameters"].update(_FORMAL_TRAIN_KNOBS)
        body["execution_mode"] = "full_train"
        body["evaluation_scope"] = "rgbt_tiny_v1_full"
        body["claim_level"] = "exploratory_comparison"
        extra = (
            " Formal HOW uses full_train (no 16/8 subset) at 160x160 with "
            "scale_queries_to_tokens so 300 queries fit 25 tokens; "
            "early_concat is existing staging blend, not FDPN / gated_multiscale. "
            "Not 640/20ep paper protocol."
        )
        body["gap"] = str(body.get("gap") or "") + extra
    else:
        body["execution_mode"] = "fast_eval"
        body["evaluation_scope"] = "fast_eval_subset"
        body["claim_level"] = "exploratory_comparison"
    return body
