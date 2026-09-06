"""HOW plugin P5. Modality-reliability spatial gate inside FeatureFusion (invent-P2 fix)."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.feature_fusion import FeatureFusion


class _ReliabilitySpatialGate(nn.Module):
    """True (N,1,H,W) reliability mask from RGB luminance + thermal local contrast."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, rgb: torch.Tensor, thermal: torch.Tensor) -> torch.Tensor:
        lum = rgb.mean(dim=1, keepdim=True)
        t_mean = thermal.mean(dim=1, keepdim=True)
        t_blur = F.avg_pool2d(t_mean, kernel_size=3, stride=1, padding=1)
        contrast = (t_mean - t_blur).abs()
        return self.net(torch.cat([lum, contrast], dim=1))


class P5ReliabilityFusion(FeatureFusion):
    """Reliability-gated RGB↔thermal blend; FeatureFusion-only (no neck edits)."""

    def __init__(self, channels: Sequence[int], *, residual: bool = True) -> None:
        super().__init__()
        chans = [int(c) for c in channels]
        if not chans:
            raise ValueError("channels must be non-empty")
        self.channels = chans
        self.residual = bool(residual)
        self.gates = nn.ModuleList([_ReliabilitySpatialGate() for _ in chans])
        self.fuse = nn.ModuleList(
            [nn.Conv2d(c * 2, c, kernel_size=1, bias=True) for c in chans]
        )

    def forward(
        self, rgb_feats: Sequence[torch.Tensor], thermal_feats: Sequence[torch.Tensor]
    ) -> list[torch.Tensor]:
        if len(rgb_feats) != len(thermal_feats):
            raise ValueError("rgb and thermal feature lists must have same length")
        if len(rgb_feats) != len(self.channels):
            raise ValueError("number of feature levels must match channels")
        out: list[torch.Tensor] = []
        for gate, conv, r, t in zip(self.gates, self.fuse, rgb_feats, thermal_feats):
            # g high → prefer thermal (low-light / high thermal contrast regions).
            g = gate(r, t)
            mixed = g * t + (1.0 - g) * r
            fused = conv(torch.cat([mixed, t], dim=1))
            if self.residual:
                fused = fused + r
            out.append(fused)
        return out


def build_fusion(
    channels: Sequence[int] = (256, 512, 1024),
    residual: bool = True,
    **_: object,
) -> FeatureFusion:
    return P5ReliabilityFusion(channels, residual=residual)
