"""HOW P11 fusion plugin: spatially gated RGB-T blending."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from models.feature_fusion import FeatureFusion

PLUGIN_KIND = "fusion"


class SpatialGateFusion(FeatureFusion):
    """Fuse aligned RGB and thermal features with true HxW spatial masks.

    For each pyramid level, a small convolution over concatenated RGB/thermal
    features predicts a spatial blending mask of shape (N, 1, H, W). The mask
    selects or blends RGB versus thermal information per pixel, preserving the
    downstream NCHW feature-map contract.
    """

    def __init__(self, channels: Sequence[int], *, residual: bool = True) -> None:
        super().__init__()
        chans = [int(c) for c in channels]
        if not chans:
            raise ValueError("channels must be non-empty")
        self.channels = chans
        self.residual = bool(residual)

        self.mask_convs = nn.ModuleList()
        self.refine_convs = nn.ModuleList()
        for ch in chans:
            self.mask_convs.append(
                nn.Sequential(
                    nn.Conv2d(ch * 2, max(ch // 4, 8), kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(max(ch // 4, 8)),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(max(ch // 4, 8), 1, kernel_size=1, bias=True),
                )
            )
            self.refine_convs.append(
                nn.Sequential(
                    nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(ch),
                    nn.ReLU(inplace=True),
                )
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
        for rgb, thermal, ch, mask_conv, refine_conv in zip(
            rgb_features,
            thermal_features,
            self.channels,
            self.mask_convs,
            self.refine_convs,
        ):
            if rgb.dim() != 4 or thermal.dim() != 4:
                raise ValueError("features must be NCHW tensors")
            if rgb.shape[1] != ch or thermal.shape[1] != ch:
                raise ValueError(
                    f"expected {ch} channels, got {rgb.shape[1]}/{thermal.shape[1]}"
                )
            if rgb.shape[-2:] != thermal.shape[-2:]:
                raise ValueError("rgb and thermal features must share HxW resolution")

            pair = torch.cat([rgb, thermal], dim=1)
            mask = torch.sigmoid(mask_conv(pair))  # (N, 1, H, W) spatial reliability mask
            mixed = mask * rgb + (1.0 - mask) * thermal
            out = refine_conv(mixed)
            if self.residual:
                out = out + rgb
            fused.append(out)
        return fused


def build_fusion(
    channels: Sequence[int] = (256, 512, 1024),
    residual: bool = True,
    **_: object,
) -> FeatureFusion:
    return SpatialGateFusion(channels, residual=residual)
