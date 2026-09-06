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
    raw = str(fusion_method or "none").strip()
    # Preserve HOW id case for plugin:<HOW> filesystem paths; loader uppercases.
    if raw.lower().startswith("plugin:"):
        how = raw.split(":", 1)[1].strip()
        return f"plugin:{how}" if how else "plugin:"
    name = raw.lower()
    if name in {"concat", "early"}:
        return "early_concat"
    if name == "full_fusion":
        return "gated_multiscale"
    return name


def is_plugin_fusion(fusion_method: str | None) -> bool:
    """True for explicit plugin:<HOW> tokens (torch-free; staging-safe)."""
    return str(fusion_method or "").strip().lower().startswith("plugin:")


def is_gated_multiscale(fusion_method: str) -> bool:
    return normalize_fusion_method(fusion_method) == "gated_multiscale"


def is_early_concat(fusion_method: str) -> bool:
    return normalize_fusion_method(fusion_method) == "early_concat"


def needs_paired_thermal(fusion_method: str) -> bool:
    """Gated catalog fusion and HOW plugins both need dual-stream thermal folders."""
    return is_gated_multiscale(fusion_method) or is_plugin_fusion(fusion_method)
