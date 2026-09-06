"""HOW plugin P9. Spatial-attention + illumination-confidence FeatureFusion."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.feature_fusion import FeatureFusion


class SpatialIlluminationFusion(FeatureFusion):
    """Thermal spatial attention + RGB illumination confidence fusion."""

    def __init__(self, channels: Sequence[int], *, residual: bool = True) -> None:
        super().__init__()
        chans = [int(c) for c in channels]
        if not chans:
            raise ValueError("channels must be non-empty")
        self.channels = chans
        self.residual = bool(residual)

        # Thermal spatial attention: 1x1 conv -> sigmoid per level
        self.spatial_conv = nn.ModuleList(
            [nn.Conv2d(c, 1, kernel_size=1) for c in chans]
        )

        # RGB illumination confidence: GAP -> small MLP -> scalar per sample
        self.illum_mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(c, max(c // 4, 4)),
                    nn.ReLU(inplace=True),
                    nn.Linear(max(c // 4, 4), 1),
                    nn.Sigmoid(),
                )
                for c in chans
            ]
        )

        # Learnable threshold for illumination confidence
        self.illum_thresh = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.5)) for _ in chans]
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
        for i, (rgb, thermal) in enumerate(zip(rgb_features, thermal_features)):
            # Spatial attention mask from thermal: (N,1,H,W)
            spatial_mask = torch.sigmoid(self.spatial_conv[i](thermal))

            # Illumination confidence from RGB: GAP -> MLP -> (N,1)
            rgb_gap = F.adaptive_avg_pool2d(rgb, 1).flatten(1)  # (N, C)
            illum_score = self.illum_mlp[i](rgb_gap)  # (N, 1)

            # Threshold gate: use thermal more when illumination is low
            gate = (illum_score < self.illum_thresh[i]).float()  # (N, 1)
            gate = gate.unsqueeze(-1).unsqueeze(-1)  # (N, 1, 1, 1)

            # Blend: when illum low, weight thermal more via spatial mask
            thermal_att = thermal * spatial_mask
            blend = rgb + gate * (thermal_att - rgb)

            if self.residual:
                out = rgb + thermal + blend
            else:
                out = blend

            fused.append(out)

        return fused


def build_fusion(channels, residual=True):
    return SpatialIlluminationFusion(channels, residual=residual)
