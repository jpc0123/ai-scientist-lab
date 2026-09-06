"""HOW plugin P13. Implements FeatureFusion with spatial attention."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.feature_fusion import FeatureFusion

PLUGIN_KIND = "fusion"


class SpatialSimilarityFusion(FeatureFusion):
    """Fuses RGB and thermal features using per-pixel spatial attention masks
    derived from local similarity between the two modalities."""

    def __init__(self, channels: Sequence[int], *, residual: bool = True) -> None:
        super().__init__()
        chans = [int(c) for c in channels]
        if not chans:
            raise ValueError("channels must be non-empty")
        self.channels = chans
        self.residual = bool(residual)

        # Per-level modules for spatial attention and fusion
        self.attention_convs = nn.ModuleList()
        self.fusion_convs = nn.ModuleList()

        for ch in chans:
            # Attention: takes concatenated RGB+Thermal (2*ch) and outputs a 1-channel spatial mask
            self.attention_convs.append(
                nn.Sequential(
                    nn.Conv2d(ch * 2, ch // 4, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(ch // 4),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(ch // 4, 1, kernel_size=1, padding=0, bias=True),
                )
            )
            # Fusion: combines the weighted sum back to `ch` dimensions
            self.fusion_convs.append(
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
        for i, (rgb, thermal) in enumerate(zip(rgb_features, thermal_features)):
            ch = self.channels[i]
            if rgb.shape[1] != ch or thermal.shape[1] != ch:
                raise ValueError(
                    f"expected {ch} channels, got {rgb.shape[1]}/{thermal.shape[1]}"
                )

            # Compute spatial attention mask (N, 1, H, W)
            concat = torch.cat([rgb, thermal], dim=1)
            mask_logits = self.attention_convs[i](concat)
            mask = torch.sigmoid(mask_logits)  # Values in [0, 1]

            # Apply per-pixel weighting
            weighted_rgb = rgb * mask
            weighted_thermal = thermal * (1.0 - mask)
            
            # Combine and refine
            mixed = weighted_rgb + weighted_thermal
            out = self.fusion_convs[i](mixed)

            if self.residual:
                out = out + rgb

            fused.append(out)

        return fused


def build_fusion(
    channels: Sequence[int] = (256, 512, 1024),
    residual: bool = True,
    **_: object,
) -> FeatureFusion:
    return SpatialSimilarityFusion(channels, residual=residual)
