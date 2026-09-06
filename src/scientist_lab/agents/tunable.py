"""Tunable parameter registry for protocol-constrained planning."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TunableParameter(BaseModel):
    name: str
    value_type: str  # enum | float | int | str
    allowed_values: list[Any] | None = None
    minimum: float | None = None
    maximum: float | None = None
    scientific_role: str = ""


DEFAULT_RGBT_TUNABLES: list[TunableParameter] = [
    TunableParameter(
        name="input_mode",
        value_type="enum",
        allowed_values=["rgb", "thermal", "rgbt"],
        scientific_role="Selects modality input for the detector.",
    ),
    TunableParameter(
        name="fusion_method",
        value_type="enum",
        allowed_values=["none", "early_concat"],
        scientific_role="Controls how RGB and thermal are combined.",
    ),
    TunableParameter(
        name="fusion_stage",
        value_type="enum",
        allowed_values=["early", "middle", "late"],
        scientific_role="Controls where RGB and thermal features are combined.",
    ),
]


def tunables_for_protocol(protocol: dict[str, Any] | None) -> list[TunableParameter]:
    allowed = set((protocol or {}).get("allowed_variables") or [])
    if not allowed:
        return list(DEFAULT_RGBT_TUNABLES)
    return [item for item in DEFAULT_RGBT_TUNABLES if item.name in allowed]


def validate_tunable_value(
    name: str,
    value: Any,
    *,
    tunables: list[TunableParameter] | None = None,
) -> str | None:
    """Return blocking issue string or None if ok."""
    registry = {item.name: item for item in (tunables or DEFAULT_RGBT_TUNABLES)}
    spec = registry.get(name)
    if spec is None:
        return None  # allow-list handled elsewhere
    if spec.allowed_values is not None and value not in spec.allowed_values:
        return (
            f"parameter {name}={value!r} not in allowed_values "
            f"{spec.allowed_values}"
        )
    if spec.value_type in {"float", "int"} and isinstance(value, (int, float)):
        if spec.minimum is not None and float(value) < spec.minimum:
            return f"parameter {name}={value} below minimum {spec.minimum}"
        if spec.maximum is not None and float(value) > spec.maximum:
            return f"parameter {name}={value} above maximum {spec.maximum}"
    return None
