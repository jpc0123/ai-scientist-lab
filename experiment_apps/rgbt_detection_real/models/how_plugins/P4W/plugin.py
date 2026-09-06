"""HOW plugin P4W. Thermal spatial gate mix; NCHW per P3/P4/P5."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.feature_fusion import FeatureFusion


class ThermalSpatialGateFusion(FeatureFusion):
    """Per-level thermal spatial gate mix with optional residual."""

    def __init__(self, channels: Sequence[int], *, residual: bool = True) -> None:
        super().__init__()
        chans = [int(c) for c in channels]
        if not chans:
            raise ValueError("channels must be non-empty")
        self.channels = chans
        self.residual = bool(residual)
        self.gate_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(c, c // 4, kernel_size=3, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(c // 4, 1, kernel_size=1, bias=True),
                )
                for c in chans
            ]
        )

    def forward(
        self,
        rgb_features: list[torch.Tensor],
        thermal_features: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        if len(rgb_features) != len(thermal_features):
            raise ValueError("rgb/thermal feature lists must match in length")
        if len(rgb_features) != len(self.channels):
            raise ValueError("feature list length must match channels length")
        out: list[torch.Tensor] = []
        for i, (rgb, therm) in enumerate(zip(rgb_features, thermal_features)):
            gate = torch.sigmoid(self.gate_convs[i](therm))
            fused = rgb * gate + therm * (1.0 - gate)
            if self.residual:
                fused = fused + rgb
            out.append(fused)
        return out


def build_fusion(channels: Sequence[int], *, residual: bool = True) -> FeatureFusion:
    """Factory for P4W plugin."""
    return ThermalSpatialGateFusion(channels, residual=residual)
