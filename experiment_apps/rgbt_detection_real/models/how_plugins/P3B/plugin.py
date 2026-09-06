"""P3B HOW plugin. Not a Claim. Cross-modal complementary residual fusion."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from models.feature_fusion import FeatureFusion


class ComplementaryCrossModalFusion(FeatureFusion):
    """Explicit complementary RGB-thermal mix with learned per-level scale."""

    def __init__(self, channels: Sequence[int], *, residual: bool = True) -> None:
        super().__init__()
        chans = [int(c) for c in channels]
        if not chans:
            raise ValueError("channels must be non-empty")
        self.channels = chans
        self.residual = bool(residual)
        self.rgb_proj = nn.ModuleList(
            [nn.Conv2d(ch, ch, kernel_size=1, bias=False) for ch in chans]
        )
        self.th_proj = nn.ModuleList(
            [nn.Conv2d(ch, ch, kernel_size=1, bias=False) for ch in chans]
        )
        self.mix = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.0)) for _ in chans]
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
        for rgb, thermal, rp, tp, scale, ch in zip(
            rgb_features,
            thermal_features,
            self.rgb_proj,
            self.th_proj,
            self.mix,
            self.channels,
        ):
            if rgb.shape[1] != ch or thermal.shape[1] != ch:
                raise ValueError(
                    f"expected {ch} channels, got {rgb.shape[1]}/{thermal.shape[1]}"
                )
            alpha = torch.sigmoid(scale)
            mixed = rp(rgb) * alpha + tp(thermal) * (1.0 - alpha)
            fused.append(mixed + rgb if self.residual else mixed)
        return fused


def build_fusion(
    channels: Sequence[int] = (256, 512, 1024),
    residual: bool = True,
    **_: object,
) -> FeatureFusion:
    return ComplementaryCrossModalFusion(channels, residual=residual)
