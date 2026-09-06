"""Example HOW plugin. Not registered. Not a Claim. Implements FeatureFusion."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from models.feature_fusion import FeatureFusion

PLUGIN_KIND = "fusion"


class WeightedAverageFusion(FeatureFusion):
    """Per-level learned RGB/thermal mix. Existing FeatureFusion contract."""

    def __init__(self, channels: Sequence[int], *, residual: bool = True) -> None:
        super().__init__()
        chans = [int(c) for c in channels]
        if not chans:
            raise ValueError("channels must be non-empty")
        self.channels = chans
        self.residual = bool(residual)
        self.mix = nn.ParameterList(
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
        for rgb, thermal, weight, ch in zip(
            rgb_features, thermal_features, self.mix, self.channels
        ):
            if rgb.shape[1] != ch or thermal.shape[1] != ch:
                raise ValueError(
                    f"expected {ch} channels, got {rgb.shape[1]}/{thermal.shape[1]}"
                )
            gate = torch.sigmoid(weight)
            mixed = gate * rgb + (1.0 - gate) * thermal
            fused.append(mixed + rgb if self.residual else mixed)
        return fused


def build_fusion(
    channels: Sequence[int] = (256, 512, 1024),
    residual: bool = True,
    **_: object,
) -> FeatureFusion:
    return WeightedAverageFusion(channels, residual=residual)
