"""P3A HOW plugin. Not a Claim. Thermal-guided channel attention fusion."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from models.feature_fusion import FeatureFusion


class ThermalGuidedChannelFusion(FeatureFusion):
    """Modality-dominant blend: thermal gates RGB channels per level."""

    def __init__(self, channels: Sequence[int], *, residual: bool = True) -> None:
        super().__init__()
        chans = [int(c) for c in channels]
        if not chans:
            raise ValueError("channels must be non-empty")
        self.channels = chans
        self.residual = bool(residual)
        self.gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(ch, max(ch // 4, 8), kernel_size=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(max(ch // 4, 8), ch, kernel_size=1, bias=True),
                    nn.Sigmoid(),
                )
                for ch in chans
            ]
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
        for rgb, thermal, gate_mod, ch in zip(
            rgb_features, thermal_features, self.gates, self.channels
        ):
            if rgb.shape[1] != ch or thermal.shape[1] != ch:
                raise ValueError(
                    f"expected {ch} channels, got {rgb.shape[1]}/{thermal.shape[1]}"
                )
            gate = gate_mod(thermal)
            mixed = rgb * gate + thermal * (1.0 - gate)
            fused.append(mixed + rgb if self.residual else mixed)
        return fused


def build_fusion(
    channels: Sequence[int] = (256, 512, 1024),
    residual: bool = True,
    **_: object,
) -> FeatureFusion:
    return ThermalGuidedChannelFusion(channels, residual=residual)
