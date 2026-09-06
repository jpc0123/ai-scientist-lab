"""HOW plugin P4. Implements FeatureFusion with spatial confidence gating."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from models.feature_fusion import FeatureFusion


class SpatialConfidenceFusion(FeatureFusion):
    """Spatial attention fusion using thermal and RGB confidence masks."""

    def __init__(self, channels: Sequence[int], *, residual: bool = True) -> None:
        super().__init__()
        chans = [int(c) for c in channels]
        if not chans:
            raise ValueError("channels must be non-empty")
        self.channels = chans
        self.residual = bool(residual)
        
        # Thermal confidence mask generator: 1x1 conv -> sigmoid -> (N,1,H,W)
        self.thermal_conv = nn.ModuleList([
            nn.Conv2d(ch, 1, kernel_size=1, stride=1, padding=0)
            for ch in chans
        ])
        
        # RGB spatial attention mask generator: 1x1 conv -> sigmoid -> (N,1,H,W)
        self.rgb_conv = nn.ModuleList([
            nn.Conv2d(ch, 1, kernel_size=1, stride=1, padding=0)
            for ch in chans
        ])

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
            if rgb.shape[1] != self.channels[i] or thermal.shape[1] != self.channels[i]:
                raise ValueError(
                    f"expected {self.channels[i]} channels, got {rgb.shape[1]}/{thermal.shape[1]}"
                )
            
            # Generate thermal confidence mask: (N,1,H,W)
            thermal_mask = torch.sigmoid(self.thermal_conv[i](thermal))
            
            # Generate RGB spatial attention mask: (N,1,H,W)
            rgb_mask = torch.sigmoid(self.rgb_conv[i](rgb))
            
            # Gate RGB features by thermal confidence
            rgb_gated = rgb * thermal_mask
            
            # Gate thermal features by RGB confidence
            thermal_gated = thermal * rgb_mask
            
            # Combine gated features
            mixed = rgb_gated + thermal_gated
            
            # Apply residual connection if enabled
            if self.residual:
                mixed = mixed + rgb
            
            fused.append(mixed)
        
        return fused


def build_fusion(
    channels: Sequence[int] = (256, 512, 1024),
    residual: bool = True,
    **_: object,
) -> FeatureFusion:
    """Build spatial confidence fusion plugin."""
    return SpatialConfidenceFusion(channels, residual=residual)
