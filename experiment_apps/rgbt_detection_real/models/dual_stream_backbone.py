"""Dual-stream backbone wrapper: 6ch input → RGB/Thermal → FeatureFusion → P3/P4/P5."""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn

from .feature_fusion import FeatureFusion
from .fusion_factory import FusionConfig, build_feature_fusion


class DualStreamGatedBackbone(nn.Module):
    """Replace DFINE.backbone for gated multiscale RGB-T fusion.

    Expects input tensor ``[B, 6, H, W]`` = RGB (0:3) + Thermal (3:6).
    """

    def __init__(
        self,
        rgb_backbone: nn.Module,
        thermal_backbone: nn.Module,
        fusion: FeatureFusion,
        *,
        share_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.rgb_backbone = rgb_backbone
        self.thermal_backbone = thermal_backbone
        self.fusion = fusion
        self.share_backbone = bool(share_backbone)
        self._out_channels = _infer_out_channels(rgb_backbone)

    @property
    def out_channels(self) -> list[int]:
        return list(self._out_channels)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(
                "DualStreamGatedBackbone expects 4D input [B, C, H, W], "
                f"got shape={tuple(x.shape)}"
            )
        if x.shape[1] == 3:
            # Vendor FLOPs/profiler (calflops) probes with 3-channel tensors.
            # Pair RGB with itself so shape math stays valid; real training uses 6ch.
            x = torch.cat([x, x], dim=1)
        if x.shape[1] != 6:
            raise ValueError(
                "DualStreamGatedBackbone expects [B, 6, H, W] input "
                "(or 3ch profiler probe), "
                f"got shape={tuple(x.shape)}"
            )
        rgb = x[:, :3]
        thermal = x[:, 3:6]
        rgb_features = self.rgb_backbone(rgb)
        thermal_features = self.thermal_backbone(thermal)
        if not isinstance(rgb_features, (list, tuple)):
            raise TypeError("rgb_backbone must return a list/tuple of feature maps")
        if not isinstance(thermal_features, (list, tuple)):
            raise TypeError("thermal_backbone must return a list/tuple of feature maps")
        rgb_list = list(rgb_features)
        thermal_list = list(thermal_features)
        fused = self.fusion(rgb_list, thermal_list)
        if getattr(self, "diag_enabled", False):
            self._last_rgb_features = [t.detach() for t in rgb_list]
            self._last_thermal_features = [t.detach() for t in thermal_list]
            self._last_fused_features = [t.detach() for t in fused]
        return fused


def build_dual_stream_backbone(
    rgb_backbone: nn.Module,
    fusion_config: FusionConfig,
) -> DualStreamGatedBackbone:
    """Clone or share thermal stream and attach config-built fusion."""
    channels = _infer_out_channels(rgb_backbone)
    cfg = FusionConfig(
        type=fusion_config.type,
        levels=fusion_config.levels,
        share_backbone=fusion_config.share_backbone,
        residual=fusion_config.residual,
        channels=tuple(channels) if channels else fusion_config.channels,
    )
    fusion = build_feature_fusion(cfg)
    return build_dual_stream_backbone_from_fusion(
        rgb_backbone,
        fusion,
        share_backbone=cfg.share_backbone,
    )


def build_dual_stream_backbone_from_fusion(
    rgb_backbone: nn.Module,
    fusion: FeatureFusion,
    *,
    share_backbone: bool = False,
) -> DualStreamGatedBackbone:
    """Attach an already-built FeatureFusion (catalog or HOW plugin)."""
    if share_backbone:
        thermal = rgb_backbone
    else:
        thermal = copy.deepcopy(rgb_backbone)
    return DualStreamGatedBackbone(
        rgb_backbone,
        thermal,
        fusion,
        share_backbone=share_backbone,
    )


def _infer_out_channels(backbone: nn.Module) -> list[int]:
    if hasattr(backbone, "out_channels"):
        value = getattr(backbone, "out_channels")
        if callable(value):
            value = value()
        return [int(c) for c in value]
    if hasattr(backbone, "_out_channels"):
        raw = list(getattr(backbone, "_out_channels"))
        return_idx = list(getattr(backbone, "return_idx", range(len(raw))))
        return [int(raw[i]) for i in return_idx]
    return []


def dual_stream_state_summary(module: DualStreamGatedBackbone) -> dict[str, Any]:
    return {
        "share_backbone": module.share_backbone,
        "out_channels": module.out_channels,
        "fusion_type": type(module.fusion).__name__,
        "rgb_params": int(sum(p.numel() for p in module.rgb_backbone.parameters())),
        "thermal_params": int(
            sum(p.numel() for p in module.thermal_backbone.parameters())
        ),
        "fusion_params": int(sum(p.numel() for p in module.fusion.parameters())),
    }
