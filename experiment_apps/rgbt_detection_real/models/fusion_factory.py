"""Config-driven fusion construction (no experiment_id branching)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from fusion_names import (
    BLOCKED_FUSION_METHODS,
    IMPLEMENTED_FUSION_METHODS,
    is_early_concat,
    is_gated_multiscale,
    normalize_fusion_method,
)

from .feature_fusion import FeatureFusion, GatedMultiscaleFusion

_FULL_FUSION_ALIASES = frozenset({"gated_multiscale", "full_fusion"})


@dataclass(frozen=True)
class FusionConfig:
    """Serializable fusion knobs for A3 dual-stream gated multiscale."""

    type: str = "gated_multiscale"
    levels: tuple[str, ...] = ("p3", "p4", "p5")
    share_backbone: bool = False
    residual: bool = True
    channels: tuple[int, ...] = field(default_factory=lambda: (256, 512, 1024))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> FusionConfig:
        data = dict(raw or {})
        levels = data.get("levels") or ("p3", "p4", "p5")
        channels = data.get("channels") or data.get("in_channels") or (256, 512, 1024)
        ftype = normalize_fusion_method(str(data.get("type") or "gated_multiscale"))
        if ftype not in _FULL_FUSION_ALIASES:
            raise ValueError(
                f"FusionConfig.type must be gated_multiscale/full_fusion, got {ftype!r}"
            )
        return cls(
            type="gated_multiscale",
            levels=tuple(str(x) for x in levels),
            share_backbone=bool(data.get("share_backbone", False)),
            residual=bool(data.get("residual", True)),
            channels=tuple(int(c) for c in channels),
        )


def parse_fusion_config(
    parameters: dict[str, Any] | None,
    *,
    fusion_method: str | None = None,
    default_channels: Sequence[int] = (256, 512, 1024),
) -> FusionConfig | None:
    """Build FusionConfig when fusion_method selects full gated multiscale."""
    params = dict(parameters or {})
    method = normalize_fusion_method(
        fusion_method if fusion_method is not None else str(params.get("fusion_method", "none"))
    )
    if method != "gated_multiscale":
        return None
    nested = params.get("fusion")
    if nested is None:
        nested = {}
    if not isinstance(nested, dict):
        raise ValueError("parameters.fusion must be a mapping when present")
    payload = {
        "type": nested.get("type") or method,
        "levels": nested.get("levels") or ("p3", "p4", "p5"),
        "share_backbone": nested.get("share_backbone", False),
        "residual": nested.get("residual", True),
        "channels": nested.get("channels")
        or nested.get("in_channels")
        or list(default_channels),
    }
    return FusionConfig.from_mapping(payload)


def build_feature_fusion(config: FusionConfig) -> FeatureFusion:
    if config.type == "gated_multiscale":
        return GatedMultiscaleFusion(
            config.channels,
            residual=config.residual,
            levels=config.levels,
        )
    raise ValueError(f"Unknown fusion type: {config.type!r}")
