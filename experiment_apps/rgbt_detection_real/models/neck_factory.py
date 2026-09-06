"""Config-driven neck/encoder factory (standard HybridEncoder vs FDPN)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import torch.nn as nn

from .fdpn import FDPN
from .how_plugin_loader import is_plugin_neck_type, load_how_plugin_neck, normalize_plugin_how_id


@dataclass(frozen=True)
class NeckConfig:
    type: str = "standard"
    in_channels: tuple[int, ...] = (256, 512, 1024)
    out_channels: int = 256
    levels: tuple[str, ...] = ("p3", "p4", "p5")
    feat_strides: tuple[int, ...] = (8, 16, 32)
    hidden_dim: int | None = None
    residual_scale: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> NeckConfig:
        data = dict(raw or {})
        ntype = str(data.get("type") or "standard").strip().lower()
        if ntype in {"hybrid", "hybrid_encoder", "default"}:
            ntype = "standard"
        if ntype not in {"standard", "fdpn"} and not is_plugin_neck_type(ntype):
            raise ValueError(
                f"Unknown neck type: {ntype!r}; supported: standard, fdpn, plugin:<HOW>"
            )
        in_ch = data.get("in_channels") or (256, 512, 1024)
        levels = data.get("levels") or ("p3", "p4", "p5")
        strides = data.get("feat_strides") or data.get("strides") or (8, 16, 32)
        out_ch = int(data.get("out_channels") or data.get("hidden_dim") or 256)
        hidden = data.get("hidden_dim")
        residual_scale = data.get("residual_scale")
        return cls(
            type=ntype,
            in_channels=tuple(int(c) for c in in_ch),
            out_channels=out_ch,
            levels=tuple(str(x) for x in levels),
            feat_strides=tuple(int(s) for s in strides),
            hidden_dim=int(hidden) if hidden is not None else out_ch,
            residual_scale=float(residual_scale) if residual_scale is not None else None,
        )


def parse_neck_config(parameters: dict[str, Any] | None) -> NeckConfig:
    params = dict(parameters or {})
    nested = params.get("neck")
    if nested is None:
        return NeckConfig(type="standard")
    if not isinstance(nested, dict):
        raise ValueError("parameters.neck must be a mapping when present")
    return NeckConfig.from_mapping(nested)


class NeckFactory:
    """Build DFINE-compatible encoder/neck modules from NeckConfig."""

    @staticmethod
    def create(
        config: NeckConfig,
        *,
        eval_spatial_size: tuple[int, int] | list[int] | None = None,
        existing_encoder: nn.Module | None = None,
        plugin_path: Any | None = None,
    ) -> nn.Module:
        if config.type == "standard":
            if existing_encoder is None:
                raise ValueError(
                    "neck.type=standard requires existing HybridEncoder "
                    "(pass through DFINE YAML encoder)"
                )
            return existing_encoder
        if config.type == "fdpn":
            hidden = int(config.hidden_dim or config.out_channels)
            return FDPN(
                in_channels=config.in_channels,
                hidden_dim=hidden,
                feat_strides=config.feat_strides,
                levels=config.levels,
                eval_spatial_size=eval_spatial_size,
                residual_scale=config.residual_scale,
            )
        if is_plugin_neck_type(config.type):
            hidden = int(config.hidden_dim or config.out_channels)
            if plugin_path is not None:
                from .how_plugin_loader import load_plugin_neck

                neck = load_plugin_neck(
                    plugin_path,
                    in_channels=config.in_channels,
                    hidden_dim=hidden,
                    feat_strides=config.feat_strides,
                    eval_spatial_size=eval_spatial_size,
                    residual_scale=config.residual_scale,
                )
            else:
                how_id = normalize_plugin_how_id(config.type)
                neck = load_how_plugin_neck(
                    how_id,
                    in_channels=config.in_channels,
                    hidden_dim=hidden,
                    feat_strides=config.feat_strides,
                    eval_spatial_size=eval_spatial_size,
                    residual_scale=config.residual_scale,
                )
            if not hasattr(neck, "in_channels"):
                neck.in_channels = list(config.in_channels)
            if not hasattr(neck, "hidden_dim"):
                neck.hidden_dim = hidden
            if not hasattr(neck, "feat_strides"):
                neck.feat_strides = list(config.feat_strides)
            if not hasattr(neck, "out_channels"):
                neck.out_channels = [hidden] * len(config.in_channels)
            if not hasattr(neck, "out_strides"):
                neck.out_strides = list(config.feat_strides)
            return neck
        raise ValueError(f"Unknown neck type: {config.type!r}")
