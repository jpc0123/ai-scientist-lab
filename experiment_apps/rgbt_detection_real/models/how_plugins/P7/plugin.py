"""HOW plugin P7. Implements FeatureFusion with thermal-gated spatial attention."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from models.feature_fusion import FeatureFusion


class ThermalGatedSpatialFusion(FeatureFusion):
    """Thermal spatial attention gates RGB, then concatenated with thermal."""

    def __init__(self, channels: Sequence[int], *, residual: bool = True) -> None:
        super().__init__()
        chans = [int(c) for c in channels]
        if not chans:
            raise ValueError("channels must be non-empty")
        self.channels = chans
        self.residual = bool(residual)
        # 1x1 conv to produce spatial attention mask from thermal features
        self.spatial_conv = nn.ModuleList(
            [nn.Conv2d(c, 1, kernel_size=1, stride=1, padding=0) for c in chans]
        )
        # 1x1 conv to reduce concatenated channels back to original
        self.reduce_conv = nn.ModuleList(
            [nn.Conv2d(c * 2, c, kernel_size=1, stride=1, padding=0) for c in chans]
        )

    def forward(
        self,
        rgb_features: list[torch.Tensor],
        thermal_features: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        if len(rgb_features) != len(thermal_features):
            raise ValueError("rgb/thermal feature list lengths must match")
        if len(rgb_features) != len(self.channels):
            raise ValueError(
                f"expected {len(self.channels)} levels, got {len(rgb_features)}"
            )
        fused: list[torch.Tensor] = []
        for rgb, thermal, spat_conv, red_conv, ch in zip(
            rgb_features,
            thermal_features,
            self.spatial_conv,
            self.reduce_conv,
            self.channels,
        ):
            if rgb.shape[1] != ch or thermal.shape[1] != ch:
                raise ValueError(
                    f"expected {ch} channels, got {rgb.shape[1]}/{thermal.shape[1]}"
                )
            # Generate spatial attention mask (N, 1, H, W) from thermal
            spatial_mask = torch.sigmoid(spat_conv(thermal))
            # Gate RGB features with thermal spatial attention
            gated_rgb = rgb * spatial_mask
            # Concatenate gated RGB with thermal along channel dimension
            concat = torch.cat([gated_rgb, thermal], dim=1)
            # Reduce back to original channel count
            out = red_conv(concat)
            # Apply residual connection if enabled
            if self.residual:
                out = out + rgb
            fused.append(out)
        return fused


def build_fusion(
    channels: Sequence[int] = (256, 512, 1024),
    residual: bool = True,
    **_: object,
) -> FeatureFusion:
    return ThermalGatedSpatialFusion(channels, residual=residual)
