"""HOW plugin P12. Implements FeatureFusion with spatial excitation."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from models.feature_fusion import FeatureFusion

PLUGIN_KIND = "fusion"


class SpatialExcitationFusion(FeatureFusion):
    """Thermally-guided RGB and RGB-guided thermal spatial attention fusion."""

    def __init__(self, channels: Sequence[int], *, residual: bool = True) -> None:
        super().__init__()
        chans = [int(c) for c in channels]
        if not chans:
            raise ValueError("channels must be non-empty")
        self.channels = chans
        self.residual = bool(residual)
        
        # Spatial attention generators: 1x1 conv -> sigmoid for (N,1,H,W) masks
        self.thermal_to_rgb_mask = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, 1, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid()
            ) for ch in chans
        ])
        
        self.rgb_to_thermal_mask = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, 1, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid()
            ) for ch in chans
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
            ch = self.channels[i]
            if rgb.shape[1] != ch or thermal.shape[1] != ch:
                raise ValueError(
                    f"expected {ch} channels, got {rgb.shape[1]}/{thermal.shape[1]}"
                )
            
            # Generate spatial mask from thermal: (N,1,H,W)
            thermal_mask = self.thermal_to_rgb_mask[i](thermal)
            # Apply to RGB: element-wise multiply with broadcasting
            guided_rgb = rgb * thermal_mask
            
            # Generate spatial mask from RGB: (N,1,H,W)
            rgb_mask = self.rgb_to_thermal_mask[i](rgb)
            # Apply to thermal: element-wise multiply with broadcasting
            guided_thermal = thermal * rgb_mask
            
            # Combine guided features
            combined = guided_rgb + guided_thermal
            
            # Apply residual connection if enabled
            if self.residual:
                # Use average of original features as residual base
                residual_base = 0.5 * (rgb + thermal)
                output = combined + residual_base
            else:
                output = combined
            
            fused.append(output)
        
        return fused


def build_fusion(
    channels: Sequence[int] = (256, 512, 1024),
    residual: bool = True,
    **_: object,
) -> FeatureFusion:
    return SpatialExcitationFusion(channels, residual=residual)
