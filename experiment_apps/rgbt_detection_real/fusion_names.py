"""Torch-free fusion method name helpers (safe for staging / contracts)."""

from __future__ import annotations

IMPLEMENTED_FUSION_METHODS = frozenset(
    {
        "none",
        "early_concat",
        "concat",
        "early",
        "gated_multiscale",
        "full_fusion",
    }
)

BLOCKED_FUSION_METHODS = frozenset(
    {
        "fdpn",
        "full",
        "full_method",
        "complete",
        "mid_fusion",
        "late_fusion",
        "dual_stream",
    }
)


def normalize_fusion_method(fusion_method: str) -> str:
    name = str(fusion_method or "none").strip().lower()
    if name in {"concat", "early"}:
        return "early_concat"
    if name == "full_fusion":
        return "gated_multiscale"
    return name


def is_gated_multiscale(fusion_method: str) -> bool:
    return normalize_fusion_method(fusion_method) == "gated_multiscale"


def is_early_concat(fusion_method: str) -> bool:
    return normalize_fusion_method(fusion_method) == "early_concat"
