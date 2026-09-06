"""HOW plugin P6. Thermal-guided spatial attention fusion."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from models.feature_fusion import FeatureFusion


class ThermalSpatialAttentionFusion(FeatureFusion):
    """Fuses RGB and thermal features using a true spatial attention mask derived
    exclusively from the thermal (infrared) feature map.

    For each pyramid level, a 1x1 convolution followed by a sigmoid produces
    a spatial saliency mask of shape (N, 1, H, W). This mask highlights regions
    with strong thermal signatures and is broadcast-multiplied with the RGB
    features before being combined with the thermal features.
    """

    def __init__(self, channels: Sequence[int], *, residual: bool = True) -> None:
        super().__init__()
        chans = [int(c) for c in channels]
        if not chans:
            raise ValueError("channels must be non-empty")
        self.channels = chans
        self.residual = bool(residual)

        # Per-level 1x1 conv to produce spatial attention mask from thermal features
        self.spatial_conv = nn.ModuleList(
            [nn.Conv2d(in_channels=c, out_channels=1, kernel_size=1) for c in chans]
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
        for rgb, thermal, conv, ch in zip(
            rgb_features, thermal_features, self.spatial_conv, self.channels
        ):
            if rgb.shape[1] != ch or thermal.shape[1] != ch:
                raise ValueError(
                    f"expected {ch} channels, got {rgb.shape[1]}/{thermal.shape[1]}"
                )

            # Generate true spatial attention mask (N, 1, H, W) from thermal features
            spatial_mask = torch.sigmoid(conv(thermal))  # (N, 1, H, W)

            # Apply spatial mask to RGB features via broadcast multiplication
            rgb_gated = rgb * spatial_mask  # (N, C, H, W)

            # Combine gated RGB with thermal
            mixed = rgb_gated + thermal

            if self.residual:
                fused.append(mixed + rgb)
            else:
                fused.append(mixed)

        return fused


def build_fusion(
    channels: Sequence[int] = (256, 512, 1024),
    residual: bool = True,
    **_: object,
) -> FeatureFusion:
    return ThermalSpatialAttentionFusion(channels, residual=residual)
